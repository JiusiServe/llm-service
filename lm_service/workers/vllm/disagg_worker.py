# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import asyncio
import os
import time
import uuid
from typing import Any, Optional, Union

import msgspec
import numpy as np
from numpy.typing import NDArray
import zmq
import zmq.asyncio
from vllm.engine.protocol import EngineClient
from vllm.utils import get_ip, get_open_port
import vllm.envs as envs
from vllm.config import ECTransferConfig, KVTransferConfig

from lm_service.stats_loggers import DisaggWorkerStatsLogger
from lm_service.protocol.protocol import (
    ExitRequest,
    FailureResponse,
    GenerationRequest,
    GenerationResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    MetricsRequest,
    MetricsResponse,
    RequestType,
    ResponseType,
    ServerType,
    WorkerRegisterRequest,
)
import lm_service.envs as lm_service_envs
from lm_service.metastore_client.factory import (
    MetastoreClientFactory,
)
from lm_service.metastore_client.metastore_client_config import (
    MetastoreClientConfig,
    json_to_metastore_config,
)
from lm_service.metastore_client.metastore_client import (
    MetastoreClientBase,
)
from lm_service.utils import is_addr_ipv6

from lm_service.logger_utils import init_logger

logger = init_logger(__name__)


class DisaggWorker:
    def __init__(
        self,
        engine: EngineClient,
        address: Optional[str] = None,
        proxy_addr: Optional[str | list[str]] = None,
        transfer_protocol: Optional[str] = None,
        metastore_client_config: Optional[dict] = None,
        ec_transfer_config: Optional[ECTransferConfig] = None,
        kv_transfer_config: Optional[KVTransferConfig] = None,
    ):
        self.engine = engine
        self.transfer_protocol = (
            lm_service_envs.TRANSFER_PROTOCOL or transfer_protocol or "ipc"
        )
        self.ec_transfer_config = ec_transfer_config
        self.kv_transfer_config = kv_transfer_config
        self.server_type = self.get_server_type()
        self.to_proxy: dict[str, zmq.asyncio.Socket] = {}
        self.metastore_client: Optional[MetastoreClientBase] = None
        if (
            metastore_client_config is not None
            or lm_service_envs.LM_SERVICE_METASTORE_CLIENT is not None
        ):
            config: MetastoreClientConfig = json_to_metastore_config(
                metastore_client_config
            )
            if address is None:
                worker_ip = lm_service_envs.LM_SERVICE_HOST_IP or get_ip()
                worker_port = (
                    int(lm_service_envs.LM_SERVICE_RPC_PORT)
                    if lm_service_envs.LM_SERVICE_RPC_PORT
                    else get_open_port()
                )
                address = f"{worker_ip}:{worker_port}"
            self.worker_addr = f"{self.transfer_protocol}://{address}"
            self.ctx = zmq.asyncio.Context()
            if is_addr_ipv6(address) and self.transfer_protocol == "tcp":
                self.ctx.setsockopt(zmq.constants.IPV6, 1)
            self.from_proxy = self.ctx.socket(zmq.constants.PULL)
            self.from_proxy.bind(self.worker_addr)
            self.metastore_client = (
                MetastoreClientFactory.create_metastore_client(
                    config=config,
                    server_type=self.server_type.value,
                    node_info=self.worker_addr,
                    to_proxy=self.to_proxy,
                )
            )
        elif proxy_addr is None or address is None:
            raise ValueError(
                "proxy_addr and address must be provided if metastore_client_config is None"
            )
        else:
            self.worker_addr = f"{self.transfer_protocol}://{address}"
            self.proxy_addr_list = [
                f"{self.transfer_protocol}://{addr}" for addr in proxy_addr
            ]
            self.ctx = zmq.asyncio.Context()
            if is_addr_ipv6(address) and self.transfer_protocol == "tcp":
                self.ctx.setsockopt(zmq.constants.IPV6, 1)
            self.from_proxy = self.ctx.socket(zmq.constants.PULL)
            self.from_proxy.bind(self.worker_addr)
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
        self.stopping = False  # whether the worker is stopping
        self.running_requests: set[asyncio.Task] = set()
        self.poller = zmq.asyncio.Poller()
        self._force_log_task: Optional[asyncio.Task] = None
        self._exit_done_event = asyncio.Event()
        self._exit_started = False

    def shutdown(self):
        for socket in self.to_proxy.values():
            socket.close(
                linger=lm_service_envs.LM_SERVICE_WORKER_GRACEFUL_EXIT_TIMEOUT_SEC
                * 1000
            )
        self.ctx.destroy()

        for running_request in self.running_requests:
            running_request.cancel()

        socket_path = self.worker_addr.replace(
            f"{self.transfer_protocol}://", ""
        )
        if self.transfer_protocol == "ipc" and os.path.exists(socket_path):
            os.remove(socket_path)
        if self.metastore_client is not None:
            self.metastore_client.close()

    def get_server_type(self) -> ServerType:
        # E - encode cache producer
        # P - encode cache consumer & kv cache producer
        # D - kv cache consumer
        # PD - ec consumer

        if self.kv_transfer_config:
            if self.kv_transfer_config.kv_role == "kv_producer":
                return ServerType.P_INSTANCE
            elif self.kv_transfer_config.kv_role == "kv_consumer":
                return ServerType.D_INSTANCE

        if self.ec_transfer_config:
            if self.ec_transfer_config.ec_role == "ec_producer":
                return ServerType.E_INSTANCE
            elif self.ec_transfer_config.ec_role == "ec_consumer":
                return ServerType.PD_INSTANCE

        return ServerType.PROXY

    async def _send_worker_register_request(
        self, socket_list: list[zmq.asyncio.Socket]
    ):
        """Handle worker register request."""
        for socket in socket_list:
            request_id = str(uuid.uuid4())
            msg = (
                RequestType.REGISTER,
                self.encoder.encode(
                    WorkerRegisterRequest(
                        request_id=request_id,
                        server_type=self.server_type,
                        address=self.worker_addr,
                    )
                ),
            )
            await socket.send_multipart(msg)
            logger.info(
                f"Worker {self.worker_addr} sent worker register request to proxy"
            )

    async def run_busy_loop(self):
        logger.info("DisaggWorker is ready To handle requests.")

        self.poller.register(self.from_proxy, zmq.POLLIN)
        if self.metastore_client is None:
            discovery_task = asyncio.create_task(
                self._send_worker_register_request(self.to_proxy.values())
            )
            self.running_requests.add(discovery_task)
            discovery_task.add_done_callback(self.running_requests.discard)

        # log engine stats (logger stats and EPD stats (if enabled))
        async def _force_log():
            try:
                while True:
                    await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                    await self.engine.do_log_stats()
            except asyncio.CancelledError:
                pass

        self._force_log_task = asyncio.create_task(
            _force_log(), name="force_log"
        )
        self.running_requests.add(self._force_log_task)
        self._force_log_task.add_done_callback(self.running_requests.discard)
        while not self.stopping:
            # poll for requests from proxy
            # if worker is stopping, exit the loop
            try:
                events = dict(await self.poller.poll(1000))
            except asyncio.CancelledError:
                # When the worker is stopping, the poller may be cancelled.
                # So we don't raise error.
                # Just catch the exception and exit the loop
                if self.stopping:
                    logger.info("Poll cancelled due to worker shutdown.")
                    break
                raise
            if not events:
                continue
            if self.from_proxy in events:
                try:
                    req_type, req_data = await self.from_proxy.recv_multipart()
                except zmq.ZMQError:
                    # When the worker is stopping, the socket may be closed.
                    # So we don't raise error.
                    if self.stopping:
                        logger.info(
                            "ZMQError received, shutting down DisaggWorker."
                        )
                        break
                    raise
                await self._handle_request(req_type, req_data)
        await self._exit_done_event.wait()
        logger.info("Worker loop stopped.")

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
            socket = self.ctx.socket(zmq.constants.PUSH)
            socket.connect(req.proxy_addr)
            self.to_proxy[req.proxy_addr] = socket

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

    # handle exit request from proxy and do graceful shutdown on SIGTERM
    async def _exit_handler(self) -> None:
        if self._exit_started:
            return
        # set stopping flag to exit busy loop
        self._exit_started = True
        if not self.stopping:
            self.stopping = True
        # cancel force log task
        try:
            log_task = getattr(self, "_force_log_task", None)
            if log_task:
                log_task.cancel()
                try:
                    await asyncio.wait_for(log_task, timeout=1)
                except Exception:
                    pass
                self.running_requests.discard(log_task)
                self._force_log_task = None
                logger.info("Force log task cancelled during shutdown.")
            # wait for all running requests to finish
            pending = {t for t in self.running_requests if not t.done()}
            if pending:
                try:
                    _, not_done = await asyncio.wait(
                        pending,
                        timeout=lm_service_envs.LM_SERVICE_WORKER_GRACEFUL_EXIT_TIMEOUT_SEC,
                    )
                except Exception:
                    logger.warning(
                        "Some tasks did not finish cleanup in %s.",
                        lm_service_envs.LM_SERVICE_WORKER_GRACEFUL_EXIT_TIMEOUT_SEC,
                    )
                    not_done_left = pending
                else:
                    not_done_left = not_done
                # cancel all running requests
                for t in not_done_left:
                    t.cancel()
            try:
                # Unregister from poller before closing the socket
                try:
                    self.poller.unregister(self.from_proxy)
                except Exception:
                    logger.warning(
                        "Could not unregister from_proxy from poller during shutdown."
                    )

                self.from_proxy.close(linger=0)
            except Exception as e:
                logger.error(
                    "Error closing from_proxy socket during shutdown: %s",
                    e,
                    exc_info=True,
                )
            # delete metadata from metastore
            node_key = f"{lm_service_envs.LM_SERVICE_REDIS_KEY_PREFIX}_{self.server_type.value}"
            if (
                hasattr(self, "metastore_client")
                and self.metastore_client is not None
                and hasattr(self.metastore_client, "delete_metadata")
            ):
                self.metastore_client.delete_metadata(
                    node_key, self.worker_addr
                )
                logger.info(
                    f"Deleted metadata for {self.worker_addr} from metastore."
                )
        finally:
            if not self._exit_done_event.is_set():
                self._exit_done_event.set()
            logger.info("DisaggWorker shutdown complete.")

    # graceful shutdown on SIGTERM
    async def _shutdown_handler(self, reason: str) -> None:
        request_id = str(uuid.uuid4())
        # send exit request to the proxy
        msg = (
            RequestType.EXIT,
            self.encoder.encode(
                ExitRequest(
                    request_id=request_id,
                    addr=self.worker_addr,
                    server_type=self.get_server_type(),
                    in_flight=len(self.running_requests),
                    reason=reason,
                )
            ),
        )
        for socket in self.to_proxy.values():
            await socket.send_multipart(msg, copy=False)
        await self._exit_handler()

    async def _generate(
        self,
        req: GenerationRequest,
        make_msg_func,
    ):
        request_id = req.request_id
        # time of the first token worker receive request from proxy
        if lm_service_envs.TIMECOUNT_ENABLED:
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
                if lm_service_envs.TIMECOUNT_ENABLED and first_token_flag:
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
