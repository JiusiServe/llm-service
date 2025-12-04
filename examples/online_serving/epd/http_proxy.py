#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
"""
disagg_encoder_proxy.py
Proxy that routes OpenAI-compatible “/v1/chat/completions” requests to two
clusters:
  • encode  (multimodal feature extraction)
  • decode  (language-model inference)
For MM input we:
    1. Extract *every* image/audio item.
    2. Fire N concurrent requests to the encoder cluster
       (one request per item, with **all text removed**).
    3. Wait for all of them to succeed.
    4. Forward the *original* request to a decode server.
"""

from __future__ import annotations

import argparse
import uuid
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from lm_service.apis.vllm.proxy import Proxy
from typing import Optional
from vllm import SamplingParams
import inspect

###############################################################################
# FastAPI app & global state
###############################################################################

app = FastAPI()
proxy_client: Optional[Proxy] = None
logger = logging.getLogger("http_proxy")

###############################################################################
# Middleware for request/response logging
###############################################################################


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests and responses"""
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    # Log incoming request
    logger.info(
        ">>> [%s] %s %s from %s",
        req_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    try:
        # Process request
        response = await call_next(request)

        # Log response
        logger.info(
            "<<< [%s] %s %s completed with status %d",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
        )

        return response
    except Exception as e:
        # Log errors
        logger.exception(
            "!!! [%s] %s %s failed with error: %s",
            req_id,
            request.method,
            request.url.path,
            str(e),
        )
        raise


###############################################################################
# FastAPI lifecycle
###############################################################################


@app.on_event("startup")
async def on_startup() -> None:
    global proxy_client
    if proxy_client is None:
        proxy_client = Proxy(
            encode_addr_list=app.state.e_addr_list,
            p_addr_list=app.state.p_addr_list,
            d_addr_list=app.state.d_addr_list,
            pd_addr_list=app.state.pd_addr_list,
            model=app.state.model,
        )
        logger.info("Proxy client initialized")
    else:
        logger.info("Proxy client already initialized")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    # Shutdown the incremental Proxy client if initialized
    global proxy_client
    try:
        if proxy_client is not None:
            proxy_client.shutdown()
            proxy_client = None
            logger.info("Proxy client shut down")
    except Exception:
        logger.exception("Error shutting down Proxy client")


###############################################################################
# Public routes
###############################################################################


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global proxy_client
    if proxy_client is None:
        raise HTTPException(
            status_code=503, detail="Proxy client not initialized"
        )

    try:
        req_data = await request.json()
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        is_streaming = req_data.get("stream", False)
        prompt = req_data.get("prompt", "")
        valid_keys = inspect.signature(SamplingParams).parameters.keys()
        params = {k: v for k, v in req_data.items() if k in valid_keys}
        sampling_params = SamplingParams(**params)
        if is_streaming:
            return StreamingResponse(
                proxy_client.generate(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=req_id,
                )
            )
        result = await proxy_client.generate(
            prompt=prompt, sampling_params=sampling_params, request_id=req_id
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in chat_completions endpoint: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Request processing error: {str(e)}"
        ) from e


@app.get("/health")
async def health_check():
    global proxy_client
    if proxy_client is None:
        return JSONResponse(
            {"proxy": "unhealthy", "reason": "Proxy client not initialized"},
            status_code=503,
        )

    instances_status: dict[str, dict[str, bool]] = (
        proxy_client.get_all_instance_status() or {}
    )

    instances_json: dict[str, dict[str, bool]] = {
        cluster: {addr: bool(ok) for addr, ok in (status_dict or {}).items()}
        for cluster, status_dict in instances_status.items()
    }

    # overall_healthy is False if any instance across any cluster reports False
    overall_healthy = not any(
        not ok
        for status_dict in instances_json.values()
        for ok in status_dict.values()
    )

    status_code = 200 if overall_healthy else 503

    result = {
        "proxy": "healthy" if overall_healthy else "unhealthy",
        "overall_healthy": overall_healthy,
    }

    # Merge cluster -> status into result
    result.update(instances_json)

    return JSONResponse(result, status_code=status_code)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    global proxy_client
    if proxy_client is None:
        return JSONResponse(
            {"error": "Proxy client not initialized"}, status_code=500
        )
    return await proxy_client.stop_profile()


@app.post("/start_profile")
async def start_profile(request: Request):
    global proxy_client
    if proxy_client is None:
        return JSONResponse(
            {"error": "Proxy client not initialized"}, status_code=500
        )
    return await proxy_client.start_profile()


@app.post("/get_metrics")
async def get_metrics(request: Request):
    global proxy_client
    if proxy_client is None:
        return JSONResponse(
            {"error": "Proxy client not initialized"}, status_code=500
        )
    return await proxy_client.get_metrics()


###############################################################################
# Main entrypoint
###############################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--encode-servers-urls",
        required=True,
        help='Comma-separated encode URLs ("http://e1:8001,http://e2:8001")',
    )
    parser.add_argument(
        "--prefill-servers-urls",
        required=False,
        help=(
            'Comma-separated prefill URLs ("http://p1:8003,http://p2:8004") ',
        ),
    )
    parser.add_argument(
        "--decode-servers-urls",
        required=False,
        help='Comma-separated decode URLs ("http://d1:8005,http://d2:8006")',
    )
    parser.add_argument(
        "--pd-servers-urls",
        required=False,
        help='Comma-separated decode URLs ("http://d1:8007,http://d2:8008")',
    )
    parser.add_argument(
        "--model",
        required=True,
        help='Model name ("gpt-4", "gpt-4-vision", etc.)',
    )

    def parse_urls(urls: str | None) -> list[str]:
        if not urls:
            return []
        return [u.strip() for u in urls.split(",") if u.strip()]

    args = parser.parse_args()

    app.state.e_addr_list = parse_urls(args.encode_servers_urls)
    app.state.p_addr_list = parse_urls(args.prefill_servers_urls)
    app.state.d_addr_list = parse_urls(args.decode_servers_urls)
    app.state.pd_addr_list = parse_urls(args.pd_servers_urls)

    logger.info("Proxy listening on %s:%s", args.host, args.port)
    logger.info("Encode servers: %s", app.state.e_addr_list)
    logger.info("Prefill instances %s", app.state.p_addr_list)
    logger.info("Decode servers: %s", app.state.d_addr_list)
    logger.info("PD servers: %s", app.state.pd_addr_list)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="uvloop",
        access_log=True,
    )
