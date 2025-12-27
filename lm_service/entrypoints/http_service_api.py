#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

Usage
For E + PD setup:
$ python disagg_encoder_proxy.py \
      --encode-servers-urls "http://e1:8001,http://e2:8002" \
      --prefill-servers-urls "disable" \
      --decode-servers-urls "http://pd1:8003,http://pd2:8004"

For E + P + D setup:
$ python disagg_encoder_proxy.py \
      --encode-servers-urls "http://e1:8001,http://e2:8001" \
      --prefill-servers-urls "http://p1:8003,http://p2:8004" \ 
      --decode-servers-urls "http://d1:8005,http://d2:8006"
"""

from __future__ import annotations

import argparse
import logging
import uuid
from lm_service.apis.vllm.proxy import HTTPProxy
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

###############################################################################
# FastAPI app & global state
###############################################################################

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("proxy")

app = FastAPI()

###############################################################################
# FastAPI lifecycle
###############################################################################

@app.on_event("startup")
async def on_startup() -> None:
    if app.state.p_urls:
        app.state.proxy_client = HTTPProxy(
            encode_addr_list=app.state.e_urls,
            p_addr_list=app.state.p_urls,
            d_addr_list=app.state.d_urls,
            model_name=args.model,
        )
    else:
        app.state.proxy_client = HTTPProxy(
            encode_addr_list=app.state.e_urls,
            pd_addr_list=app.state.d_urls,
            model_name=args.model,
        )

@app.on_event("shutdown")
async def on_shutdown() -> None:
    await app.state.proxy_client.shutdown()

###############################################################################
# Interfaces exposed via FastAPI
###############################################################################

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req_data = await request.json()
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    is_streaming = req_data.get("stream", False)

    if is_streaming:
        return StreamingResponse(
            app.state.proxy_client.generate(req_data, req_id),
            media_type="text/event-stream",
        )
    result = []
    async for chunk in app.state.proxy_client.generate(req_data, req_id):
        result.append(chunk)
    return JSONResponse(content=result)


@app.get("/health")
async def health_check():
    resp = await app.state.proxy_client.get_overall_health_states()
    overall_health = any(not status for status in resp.values())
    status_code = 200 if overall_health else 503
    resp["proxy"] = "healthy"
    return JSONResponse(
        resp,
        status_code=status_code,
    )


@app.post("/start_profile")
async def start_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    return await app.state.proxy_client.profile("start", body)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    return await app.state.proxy_client.profile("stop", body)

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
        required=True,
        help=(
            'Comma-separated prefill URLs ("http://p1:8003,http://p2:8004") ',
            'to enable E->P->D, set "disable" or "none" to enable E->PD',
        ),
    )
    parser.add_argument(
        "--decode-servers-urls",
        required=True,
        help='Comma-separated decode URLs ("http://d1:8005,http://d2:8006")',
    )
    parser.add_argument(
        "--model",
        required=True,
        help='Model name to be used for routing (e.g., "gpt-4")',
    )
    args = parser.parse_args()
    app.state.e_urls = [
        u.strip() for u in args.encode_servers_urls.split(",") if u.strip()
    ]
    app.state.d_urls = [
        u.strip() for u in args.decode_servers_urls.split(",") if u.strip()
    ]
    # handle prefill instances
    if args.prefill_servers_urls.lower() in ("disable", "none", ""):
        app.state.p_urls = []
        logger.info(
            "Disaggregated prefill phase explicitly disabled by user. Running E + PD..."
        )
    else:
        app.state.p_urls = [
            u.strip() for u in args.prefill_servers_urls.split(",") if u.strip()
        ]
        logger.info(
            "Disaggregated prefill phase is enabled. Running E + P + D..."
        )

    logger.info("Proxy listening on %s:%s", args.host, args.port)
    logger.info("Encode servers: %s", app.state.e_urls)
    logger.info("Prefill instances %s", app.state.p_urls)
    logger.info("Decode servers: %s", app.state.d_urls)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="uvloop",
        access_log=False,
    )
