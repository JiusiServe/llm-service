# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import signal
import socket
import zmq
import uvicorn
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any
import uvloop
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from lm_service.entrypoints.cli_args import make_arg_parser
from lm_service.apis.vllm.proxy import Proxy
from lm_service.logger_utils import init_logger
from lm_service.service_discovery import HealthCheckServiceDiscovery
from vllm.entrypoints.openai.api_server import (
    validate_json_request,
    ChatCompletionRequest,
    CompletionRequest,
    setup_server,
    init_app_state,
    chat,
    completion,
    base,
    engine_client,
)
from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    ChatCompletionResponse,
    CompletionResponse,
)
import vllm.envs as envs
from vllm.entrypoints.utils import with_cancellation
from vllm.engine.protocol import EngineClient
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser, decorate_logs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import find_process_using_port


logger = init_logger(__name__)

router = APIRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    try:
        yield
    finally:
        del app.state


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def create_chat_completion(
    request: ChatCompletionRequest, raw_request: Request
):
    handler = chat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Chat Completions API"
        )
    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e
    # non-streaming response
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())
    # streaming response
    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def create_completion(request: CompletionRequest, raw_request: Request):
    handler = completion(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API"
        )
    try:
        generator = await handler.create_completion(request, raw_request)
    except OverflowError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e
    # non-streaming response
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())
    # streaming response
    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.get("/check_health")
@with_cancellation
async def check_health(raw_request: Request):
    proxy_client: EngineClient = engine_client(raw_request)
    results: dict[str, str] = {}
    for server_type in proxy_client.active_types:
        sockets: dict[str, zmq.asyncio.Socket] = (
            proxy_client.server_to_socket_map.get(server_type)
        )
        for addr in sockets:
            service_discovery: HealthCheckServiceDiscovery = (
                proxy_client.instance_clusters[server_type].service_discovery
            )
            check_health = service_discovery.get_health_check_func()
            health_check_interval = (
                service_discovery.get_health_check_interval()
            )
            try:
                result: bool = await asyncio.wait_for(
                    check_health(server_type, addr),
                    timeout=health_check_interval,
                )
                results[str(server_type.name) + "-" + addr] = (
                    "Healthy" if result else "Unhealthy"
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=HTTPStatus.GATEWAY_TIMEOUT.value,
                    detail=f"Health check timed out for server_type={server_type.name}, addr={addr}",
                )
            except Exception as e:
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    detail=str(e),
                ) from e
    return JSONResponse(content={"results": results})


@router.get("/metrics")
@with_cancellation
async def metrics(raw_request: Request):
    proxy_client: EngineClient = engine_client(raw_request)
    try:
        # total_metrics: [server_type [addr, metrics_msg or error_msg]]
        total_metrics = await proxy_client.get_metrics()
        return PlainTextResponse(
            status_code=HTTPStatus.OK.value,
            content=metrics_to_readable_format(total_metrics),
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e


@router.post("/abort")
@with_cancellation
async def abort(raw_request: Request):
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED.value,
        detail="The /abort endpoint is not implemented.",
    )


if envs.VLLM_TORCH_PROFILER_DIR:

    @router.post("/start_profile")
    async def start_profile(raw_request: Request):
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
            detail="The /start_profile endpoint is not implemented.",
        )

    @router.post("/stop_profile")
    async def stop_profile(raw_request: Request):
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
            detail="The /stop_profile endpoint is not implemented.",
        )


async def run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""
    decorate_logs("APIServer")
    listen_address, sock = setup_server(args)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    **uvicorn_kwargs: Any,
):
    """Serve the HTTP app using Uvicorn."""
    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    loop = asyncio.get_event_loop()

    # start the server
    server_task = loop.create_task(
        server.serve(sockets=[sock] if sock else None)
    )

    # handle SIGTERM for graceful shutdown
    def signal_handler() -> None:
        server_task.cancel()

    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    async def dummy_shutdown() -> None:
        pass

    try:
        await server_task
        return dummy_shutdown
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.warning(
                "port %s is used by process %s launched with command:\n%s",
                port,
                process,
                " ".join(process.cmdline()),
            )
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()


async def run_server_worker(
    listen_address: str,
    sock: socket.socket,
    args: Namespace,
    **uvicorn_kwargs,
) -> None:
    """Run a single worker of the API server."""
    async with build_async_proxy_client(args) as proxy_client:
        app = build_app(args)
        vllm_config = proxy_client.vllm_config
        await init_app_state(proxy_client, vllm_config, app.state, args)
        logger.info(
            "Starting vLLM API server %d on %s",
            proxy_client.vllm_config.parallel_config._api_process_rank,
            listen_address,
        )
        shutdown_task = await serve_http(
            app, sock, host=args.host, port=args.port, **uvicorn_kwargs
        )
    try:
        await shutdown_task()
    finally:
        sock.close()


@asynccontextmanager
async def build_async_proxy_client(
    args: Namespace,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
) -> AsyncIterator[EngineClient]:
    """Build the async engine client."""
    proxy_args = AsyncEngineArgs.from_cli_args(args)
    vllm_config = proxy_args.create_engine_config(usage_context=usage_context)
    p = Proxy(
        vllm_config=vllm_config,
        proxy_addr=args.proxy_addr,
        encode_addr_list=args.encode_addr_list,
        pd_addr_list=args.pd_addr_list,
        p_addr_list=args.p_addr_list,
        d_addr_list=args.d_addr_list,
        transfer_protocol=args.transfer_protocol,
        metastore_client_config=args.metastore_client_config,
        log_stats=not args.disable_log_stats,
        model_name=args.model,
        enable_health_monitor=False,
    )
    yield p
    p.shutdown()


def build_app(args: Namespace) -> FastAPI:
    """Build the FastAPI app."""
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app


def metrics_to_readable_format(metrics: dict) -> str:
    """Convert metrics dict to a readable string format."""
    lines = []
    for server_type, addr_metrics in metrics.items():
        lines.append(f"Server Type: {server_type}")
        for addr, metric_msg in addr_metrics.items():
            lines.append(f"  Address: {addr}")
            sub_msgs = metric_msg.strip().split(", ")
            for sub_msg in sub_msgs:
                if sub_msg:
                    lines.append(f"    {sub_msg}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    uvloop.run(run_server(args))
