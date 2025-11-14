# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import asyncio
import os
import re
import time
from typing import Any, Optional, Union

import msgspec
import numpy as np
from numpy.typing import NDArray
import zmq
import zmq.asyncio

from llm_service.stats_loggers import DisaggWorkerStatsLogger
from llm_service.protocol.protocol import (
    FailureResponse,
    GenerationRequest,
    GenerationResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    MetricsRequest,
    MetricsResponse,
    RequestType,
    ResponseType,
)
from vllm.engine.protocol import EngineClient
import vllm.envs as envs
import llm_service.envs as llm_service_envs

from llm_service.logger_utils import init_logger

logger = init_logger(__name__)


class DisaggWorker:
    def __init__(
        self,
        engine: EngineClient,
        address: str,
        proxy_addr: str | list[str],
        transfer_protocol: Optional[str] = None,
    ):
        self.engine = engine
        self.transfer_protocol = (
            llm_service_envs.TRANSFER_PROTOCOL or transfer_protocol or "ipc"
        )
        self.worker_addr = f"{self.transfer_protocol}://{address}"
        self.proxy_addr_list = [
            f"{self.transfer_protocol}://{addr}" for addr in proxy_addr
        ]
        self.ctx = zmq.asyncio.Context()
        ipv6_pattern = r"^\[(.*?)\]:(\d+)$"
        if re.match(ipv6_pattern, address) and self.transfer_protocol == "tcp":
            self.ctx.setsockopt(zmq.constants.IPV6, 1)
        self.from_proxy = self.ctx.socket(zmq.constants.PULL)
        self.from_proxy.bind(self.worker_addr)
        self.to_proxy: dict[str, zmq.asyncio.Socket] = {}
        for addr in self.proxy_addr_list:
            socket = self.ctx.socket(zmq.constants.PUSH)
            socket.connect(addr)
            self.to_proxy[addr] = socket
        logger.info(
            f"Worker address: {self.worker_addr}, proxy_addr: {self.proxy_addr_list}"
        )
        self.decoder_generate = msgspec.msgpack.Decoder(GenerationRequest)
        self.decoder_heartbeat = msgspec.msgpack.Decoder(HeartbeatRequest)
        self.decoder_abort = msgspec.msgpack.Decoder(GenerationRequest)
        self.decoder_metrics = msgspec.msgpack.Decoder(MetricsRequest)
        self.encoder = msgspec.msgpack.Encoder()

        self.running_requests: set[asyncio.Task] = set()

    def shutdown(self):
        self.ctx.destroy()

        for running_request in self.running_requests:
            running_request.cancel()

        socket_path = self.worker_addr.replace(
            f"{self.transfer_protocol}://", ""
        )
        if self.transfer_protocol == "ipc" and os.path.exists(socket_path):
            os.remove(socket_path)

    async def run_busy_loop(self):
        logger.info("DisaggWorker is ready To handle requests.")

        poller = zmq.asyncio.Poller()
        poller.register(self.from_proxy, zmq.POLLIN)
        if llm_service_envs.TIMECOUNT_ENABLED:
            # log engine stats (logger stats and EPD stats (if enabled))
            async def _force_log():
                while True:
                    await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                    await self.engine.do_log_stats()

            task = asyncio.create_task(_force_log())
            self.running_requests.add(task)
            task.add_done_callback(self.running_requests.discard)
        while True:
            req_type, req_data = await self.from_proxy.recv_multipart()
            await self._handle_request(req_type, req_data)

    async def _handle_request(self, req_type: bytes, req_data: bytes):
        if req_type == RequestType.ENCODE:
            gen_req = self.decoder_generate.decode(req_data)
            gen_req.sampling_params.max_tokens = 1
            await self._encode_handler(gen_req)
        elif req_type == RequestType.PREFILL:
            gen_req = self.decoder_generate.decode(req_data)
            gen_req.sampling_params.max_tokens = 1
            await self._prefill_handler(gen_req)
        elif req_type == RequestType.GENERATION:
            gen_req = self.decoder_generate.decode(req_data)
            await self._generation_handler(gen_req)
        elif req_type == RequestType.ABORT:
            gen_req = self.decoder_abort.decode(req_data)
            await self._abort_handler(gen_req)
        elif req_type == RequestType.HEARTBEAT:
            hb_req = self.decoder_heartbeat.decode(req_data)
            await self._heartbeat_handler(hb_req)
        elif req_type == RequestType.METRICS:
            metrics_req = self.decoder_metrics.decode(req_data)
            await self._metrics_handler(metrics_req)
        else:
            raise Exception(f"Unknown Request Type: {req_type.decode()}.")

    async def _prefill_handler(self, req: GenerationRequest):
        task = asyncio.create_task(
            self._generate(req, lambda b: (ResponseType.PREFILL, b))
        )
        self.running_requests.add(task)
        task.add_done_callback(self.running_requests.discard)

    async def _handle_response(self, req, msg):
        if req.proxy_addr not in self.to_proxy:
            logger.error(
                f"request {req.request_id} could not find proxy address {req.proxy_addr}."
            )
            return

        await self.to_proxy[req.proxy_addr].send_multipart(msg, copy=False)

    async def _encode_handler(self, req: GenerationRequest):
        task = asyncio.create_task(
            self._generate(req, lambda b: (ResponseType.ENCODE, b))
        )
        self.running_requests.add(task)
        task.add_done_callback(self.running_requests.discard)

    async def _generation_handler(self, req: GenerationRequest):
        task = asyncio.create_task(
            self._generate(req, lambda b: (ResponseType.GENERATION, b))
        )
        self.running_requests.add(task)
        task.add_done_callback(self.running_requests.discard)

    async def _abort_handler(self, req: GenerationRequest):
        self.engine.abort(request_id=req.request_id)

    async def _heartbeat_handler(self, req: HeartbeatRequest):
        msg = (
            ResponseType.HEARTBEAT,
            self.encoder.encode(
                HeartbeatResponse(request_id=req.request_id, status="OK")
            ),
        )
        await self._handle_response(req, msg)

    async def _metrics_handler(self, req: MetricsRequest):
        stats_logger: Optional[dict[int, dict[str, Union[int, float]]]] = (
            DisaggWorkerStatsLogger.get_stats_snapshot_avg()
        )
        msg = (
            ResponseType.METRICS,
            self.encoder.encode(
                MetricsResponse(request_id=req.request_id, metrics=stats_logger)
            ),
        )
        await self._handle_response(req, msg)

    async def _generate(
        self,
        req: GenerationRequest,
        make_msg_func,
    ):
        request_id = req.request_id
        # time of the first token worker receive request from proxy
        if llm_service_envs.TIMECOUNT_ENABLED:
            recv_timestamp = time.perf_counter()
        first_token_flag = True
        try:
            prompt_payload: dict[str, Any] = {"prompt": req.prompt}
            if req.multi_modal_data is not None:
                prompt_payload["multi_modal_data"] = _decode_mm_data(
                    req.multi_modal_data
                )

            generator = self.engine.generate(
                prompt=prompt_payload,
                sampling_params=req.sampling_params,
                request_id=request_id,
            )

            async for request_output in generator:
                response = GenerationResponse.from_request_output(
                    request_output
                )
                if llm_service_envs.TIMECOUNT_ENABLED and first_token_flag:
                    response.proxy_to_worker_time_end = recv_timestamp  # type: ignore
                    first_token_flag = False
                response_bytes = self.encoder.encode(response)
                msg = make_msg_func(response_bytes)
                await self._handle_response(req, msg)
        except Exception as e:
            logger.exception("Generation failed for request %s", request_id)
            failure_resp = FailureResponse(
                request_id=request_id, error_message=str(e) or type(e).__name__
            )
            response_bytes = self.encoder.encode(failure_resp)
            msg = (ResponseType.FAILURE, response_bytes)
            await self._handle_response(req, msg)


def _decode_mm_data(mm_data: dict[str, Any]) -> dict[str, Any]:
    images = mm_data.get("image", [])
    if not isinstance(images, list):
        images = [images]
    decoded_list: list[NDArray[Any]] = []
    for img in images:
        if img["type"] == "ndarray":
            decoded_img = np.frombuffer(
                bytes(img["data"]), dtype=img["dtype"]
            ).reshape(img["shape"])
            decoded_list.append(decoded_img)
    result_images: list[NDArray[Any]] | NDArray[Any]
    if len(decoded_list) == 1:
        result_images = decoded_list[0]
    else:
        result_images = decoded_list
    return {"image": result_images}
