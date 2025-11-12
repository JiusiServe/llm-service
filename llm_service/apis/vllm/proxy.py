# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import asyncio
from collections import defaultdict
import os
import time
import uuid
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Optional, Union

import msgspec
import numpy as np
import zmq
import zmq.asyncio

from vllm.config import ModelConfig, VllmConfig
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
    ServerType,
    SERVER_ROUTE_MAP,
)
from llm_service.request_stats import RequestStatsMonitor
from llm_service.routing_logic import RandomRouter, RoutingInterface
from llm_service.service_discovery import HealthCheckServiceDiscovery
from llm_service.stats_loggers import MetricsReporter

from vllm.engine.protocol import EngineClient
from vllm.inputs.data import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import Device
import llm_service.envs as llm_service_envs
from llm_service.logger_utils import init_logger

logger = init_logger(__name__)


class InstanceGroup:
    """Encapsulates per-server-type runtime components."""

    def __init__(
        self, sockets, service_discovery, stats_monitor, router, metrics_logger
    ):
        self.sockets = sockets
        self.service_discovery = service_discovery
        self.stats_monitor = stats_monitor
        self.router = router
        self.metrics_logger = metrics_logger
        self.time_count: dict[int, int] = defaultdict(int)
        self.time_total: dict[int, float] = defaultdict(float)


class Proxy(EngineClient):
    """
    Proxy
    """

    def __init__(
        self,
        proxy_addr: str,
        encode_addr_list: list[str],
        model_name: str,
        encode_router: type[RoutingInterface] = RandomRouter,
        pd_addr_list: Optional[list[str]] = None,
        pd_router: type[RoutingInterface] = RandomRouter,
        prefill_addr_list: Optional[list[str]] = None,
        prefill_router: type[RoutingInterface] = RandomRouter,
        decode_addr_list: Optional[list[str]] = None,
        decode_router: type[RoutingInterface] = RandomRouter,
        enable_health_monitor=True,
        health_check_interval=10,
        health_threshold=3,
    ):
        init_params = locals()

        self.queues: dict[str, asyncio.Queue] = {}
        self.encoder = msgspec.msgpack.Encoder()
        self.instances: dict[ServerType, InstanceGroup] = {}

        # Now only support P-D merged and P-D disaggregated circumstances.
        # Validate the input addresses.
        self.is_pd_merged = False
        if prefill_addr_list is not None and decode_addr_list is not None:
            if pd_addr_list is not None:
                raise ValueError(
                    "Invalid input: pd_addr_list should be None when both"
                    "prefill_addr_list and decode_addr_list are provided."
                )
        elif (
            prefill_addr_list is None
            and decode_addr_list is None
            and pd_addr_list is not None
        ):
            self.is_pd_merged = True
        else:
            raise ValueError(
                "Invalid input: either both prefill_addr_list"
                "and decode_addr_list should be provided,"
            )

        self.ctx = zmq.asyncio.Context()
        self.proxy_addr = f"ipc://{proxy_addr}"
        for server_type, key_map in SERVER_ROUTE_MAP:
            addr_list = init_params.get(key_map["addr_key"])
            router = init_params.get(key_map["router_key"])
            if addr_list is not None:
                self._init_instance_group(
                    server_type,
                    addr_list,
                    router,
                    enable_health_monitor,
                    health_check_interval,
                    health_threshold,
                )

        self.output_handler: Optional[asyncio.Task] = None

        # Dummy: needed for EngineClient Protocol.
        self.model_config = ModelConfig(
            model=model_name,
            tokenizer=model_name,
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="auto",
            task="generate",
            seed=42,
        )

    def _init_instance_group(
        self,
        server_type: ServerType,
        addr_list: list[str],
        router_cls: type[RoutingInterface],
        enable_health_monitor,
        health_check_interval,
        health_threshold,
    ):
        ipc_addrs = [f"ipc://{addr}" for addr in addr_list]
        sockets = []
        for addr in ipc_addrs:
            s = self.ctx.socket(zmq.constants.PUSH)
            s.connect(addr)
            sockets.append(s)

        service_discovery = HealthCheckServiceDiscovery(
            server_type=server_type,
            instances=list(range(len(ipc_addrs))),
            enable_health_monitor=self.enable_health_monitor,
            health_check_interval=self.health_check_interval,
            health_threshold=self.health_threshold,
            health_check_func=self.check_health,
        )

        stats_monitor = RequestStatsMonitor(list(range(len(ipc_addrs))))
        router = router_cls()
        metrics_logger = MetricsReporter(
            server_type=server_type,
            instances=list(range(len(ipc_addrs))),
            addr=ipc_addrs,
            get_metrics_func=self.get_metrics,
        )

        self.instances[server_type] = InstanceGroup(
            sockets, service_discovery, stats_monitor, router, metrics_logger
        )

    async def _send_request(
        self,
        server_type: ServerType,
        request_type: RequestType,
        request: Any,
        q: asyncio.Queue,
        expect_stream: bool = False,
    ):
        """统一处理 encode / PD / P / D 的发送逻辑，支持流式和非流式"""
        group = self.instances[server_type]
        if not group.sockets:
            raise RuntimeError(f"No {server_type.name} workers configured.")

        # 序列化请求
        try:
            payload = self.encoder.encode(request)
        except Exception as e:
            raise RuntimeError(
                f"Failed to serialize {request_type.name}"
            ) from e

        msg = (request_type, payload)
        health_endpoints = group.service_discovery.get_health_endpoints()
        request_stats = group.stats_monitor.get_request_stats()
        idx = group.router.route_request(health_endpoints, request_stats)
        group.stats_monitor.on_new_request(idx, request_id=request.request_id)

        socket = group.sockets[idx]
        start_time = (
            time.perf_counter() if llm_service_envs.TIMECOUNT_ENABLED else None
        )

        await socket.send_multipart(msg, copy=False)

        async def handle_response(resp):
            if isinstance(resp, Exception):
                raise resp
            # 使用 InstanceGroup 属性记录时间
            if (
                llm_service_envs.TIMECOUNT_ENABLED
                and isinstance(resp, GenerationResponse)
                and getattr(resp, "proxy_to_worker_time_end", None)
            ):
                key = (server_type, idx)
                group.proxy_to_time_count[key] = (
                    group.proxy_to_time_count.get(key, 0) + 1
                )
                group.proxy_to_time_total[key] = group.proxy_to_time_total.get(
                    key, 0.0
                ) + (resp.proxy_to_worker_time_end - start_time)
            return resp

        try:
            if expect_stream:
                # 流式返回
                finished = False
                while not finished:
                    resp = await q.get()
                    resp = await handle_response(resp)
                    finished = getattr(resp, "finish_reason", None) is not None
                    yield resp
            else:
                # 非流式返回（Prefill）
                resp = await q.get()
                resp = await handle_response(resp)
                return resp
        finally:
            group.stats_monitor.on_request_completed(
                idx, request_id=request.request_id
            )

            if isinstance(resp, Exception):
                raise resp
                # 使用 InstanceGroup 属性记录时间
            if (
                llm_service_envs.TIMECOUNT_ENABLED
                and isinstance(resp, GenerationResponse)
                and getattr(resp, "proxy_to_worker_time_end", None)
            ):
                key = (server_type, idx)
                group.proxy_to_time_count[key] = (
                    group.proxy_to_time_count.get(key, 0) + 1
                )
                group.proxy_to_time_total[key] = group.proxy_to_time_total.get(
                    key, 0.0
                ) + (resp.proxy_to_worker_time_end - start_time)
                return resp

            try:
                if expect_stream:
                    # 流式返回
                    finished = False
                    while not finished:
                        resp = await q.get()
                        resp = await handle_response(resp)
                        finished = (
                            getattr(resp, "finish_reason", None) is not None
                        )
                        yield resp
                else:
                    # 非流式返回（Prefill）
                    resp = await q.get()
                    resp = await handle_response(resp)
                    return resp
            finally:
                group.stats_monitor.on_request_completed(
                    idx, request_id=request.request_id
                )

    def shutdown(self):
        self.ctx.destroy()
        if (task := self.output_handler) is not None:
            task.cancel()
        socket_path = self.proxy_addr.replace("ipc://", "")
        if os.path.exists(socket_path):
            os.remove(socket_path)

    async def log_metrics(self):
        for group in self.instances.values():
            await group.metrics_logger.get_metrics()

    def _to_request_output(self, resp: GenerationResponse) -> RequestOutput:
        """Convert a PD/Generate response to vLLM RequestOutput.

        This creates a single CompletionOutput. If the response includes
        text/token_ids attributes, they are used; otherwise defaults are used.
        """
        text = getattr(resp, "text", "")
        token_ids = getattr(resp, "token_ids", [])

        completion = CompletionOutput(
            index=0,
            text=text,
            token_ids=token_ids,
            cumulative_logprob=None,
            logprobs=None,
            finish_reason=resp.finish_reason,
            stop_reason=resp.stop_reason,
        )

        return RequestOutput(
            request_id=resp.request_id,
            prompt=None,
            prompt_token_ids=resp.prompt_token_ids,
            prompt_logprobs=None,
            outputs=[completion],
            finished=resp.finish_reason is not None,
        )

    async def _run_worker(
        self, server_type: ServerType, request: Any, q: asyncio.Queue
    ):
        """统一执行不同 server_type 的 worker"""
        cfg = SERVER_ROUTE_MAP[server_type]
        expect_stream = cfg["expect_stream"]
        request_type = cfg["request_type"]

        if expect_stream:
            async for resp in self._send_request(
                server_type, request_type, request, q, expect_stream=True
            ):
                yield resp
        else:
            # 非流式，Prefill阶段
            await self._send_request(
                server_type, request_type, request, q, expect_stream=False
            )

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ):
        # lazy init output handler
        if self.output_handler is None:
            self.output_handler = asyncio.create_task(
                self._run_output_handler()
            )

        request_id = request_id or uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue()
        if request_id in self.queues:
            raise ValueError(f"Request id {request_id} already running.")
        self.queues[request_id] = q

        prompt_text = prompt["prompt"] if isinstance(prompt, dict) else prompt
        multi_modal_data = (
            prompt.get("multi_modal_data") if isinstance(prompt, dict) else None
        )
        request = GenerationRequest(
            request_id=request_id,
            prompt=prompt_text,
            sampling_params=sampling_params,
            multi_modal_data=multi_modal_data,
        )

        try:
            req_dict = msgspec.to_builtins(request)
            request = msgspec.convert(req_dict, GenerationRequest, strict=True)
            if multi_modal_data:
                request.multi_modal_data = _encode_mm_data(multi_modal_data)
                await self._run_worker(ServerType.E_INSTANCE, request, q)
            if self.is_pd_merged:
                async for pd_response in self._run_worker(
                    ServerType.PD_INSTANCE, request, q
                ):
                    yield self._to_request_output(pd_response)
            else:
                await self._run_worker(ServerType.P_INSTANCE, request, q)
                async for d_response in self._run_worker(
                    ServerType.D_INSTANCE, request, q
                ):
                    yield self._to_request_output(d_response)

        except msgspec.ValidationError as e:
            raise RuntimeError(f"Invalid Parameters: {e}.") from e
        finally:
            self.queues.pop(request_id, None)

    async def abort_requests_from_unhealth_endpoints(
        self, server_type, unhealth_endpoints, request_stats_monitor
    ) -> None:
        request_stats = request_stats_monitor.get_request_stats()

        async def fail_request(req_id, iid):
            if req_id in self.queues:
                await self.queues[req_id].put(
                    RuntimeError(
                        f"{server_type} instance {iid} is unhealthy, "
                        f"so abort its request {req_id}."
                    )
                )

        tasks = [
            asyncio.create_task(fail_request(req_id, iid))
            for iid in unhealth_endpoints
            for req_id in request_stats.get(iid).in_flight_requests
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_output_handler(self) -> None:
        """Background task to pull responses and dispatch to request queues.

        Binds a PULL socket on proxy_addr and receives multipart messages of
        the form (response_type, payload). Decodes payload into a
        GenerationResponse and enqueues it into the corresponding request queue
        keyed by request_id.
        """
        socket: Optional[zmq.asyncio.Socket] = None
        decoder = msgspec.msgpack.Decoder(GenerationResponse)
        failure_decoder = msgspec.msgpack.Decoder(FailureResponse)
        heartbeat_decoder = msgspec.msgpack.Decoder(HeartbeatResponse)
        metrics_decoder = msgspec.msgpack.Decoder(MetricsResponse)
        try:
            socket = self.ctx.socket(zmq.constants.PULL)
            socket.bind(self.proxy_addr)
            timeout = self.health_check_interval * self.health_threshold / 2

            while True:
                encode_unhealths = (
                    self.encode_service_discovery.get_unhealth_endpoints()
                )
                pd_unhealths = (
                    self.pd_service_discovery.get_unhealth_endpoints()
                )
                tasks = []
                if encode_unhealths:
                    tasks.append(
                        self.abort_requests_from_unhealth_endpoints(
                            server_type=ServerType.E_INSTANCE,
                            unhealth_endpoints=encode_unhealths,
                            request_stats_monitor=self.encode_request_stats_monitor,
                        )
                    )
                if pd_unhealths:
                    tasks.append(
                        self.abort_requests_from_unhealth_endpoints(
                            server_type=ServerType.PD_INSTANCE,
                            unhealth_endpoints=pd_unhealths,
                            request_stats_monitor=self.pd_request_stats_monitor,
                        )
                    )
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Check if the engine is alive:
                if not await socket.poll(timeout=timeout):
                    continue
                resp_type, payload = await socket.recv_multipart()

                # Decode response according to its type.
                resp: Union[
                    GenerationResponse,
                    HeartbeatResponse,
                    FailureResponse,
                    MetricsResponse,
                ]
                if resp_type in (ResponseType.GENERATION, ResponseType.ENCODE):
                    resp = decoder.decode(payload)
                elif resp_type == ResponseType.HEARTBEAT:
                    resp = heartbeat_decoder.decode(payload)
                elif resp_type == ResponseType.FAILURE:
                    resp = failure_decoder.decode(payload)
                elif resp_type == ResponseType.METRICS:
                    resp = metrics_decoder.decode(payload)
                else:
                    raise RuntimeError(
                        f"Unknown response type from worker: {resp_type.decode()}"
                    )

                if resp.request_id not in self.queues:
                    if resp_type not in (
                        ResponseType.HEARTBEAT,
                        ResponseType.METRICS,
                    ):
                        logger.warning(
                            "Request %s may have been aborted, ignore response.",
                            resp.request_id,
                        )
                elif isinstance(resp, FailureResponse):
                    self.queues[resp.request_id].put_nowait(
                        RuntimeError(f"Request error: {resp.error_message}")
                    )
                else:
                    self.queues[resp.request_id].put_nowait(resp)

        except Exception as e:
            # TODO: maybe there is a more fine-grained way to handle errors.
            # For now, if there is any error, we terminate all requests.
            for q in self.queues.values():
                q.put_nowait(e)
        finally:
            if socket is not None:
                socket.close(linger=0)

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        raise NotImplementedError

    async def abort(self, request_id: str) -> None:
        raise NotImplementedError

    async def get_vllm_config(self) -> VllmConfig:
        """Get the vllm configuration of the vLLM engine."""
        raise NotImplementedError

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def get_input_preprocessor(self) -> InputPreprocessor:
        raise NotImplementedError

    async def get_tokenizer(self) -> AnyTokenizer:
        raise NotImplementedError

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(self) -> None:
        pass

    async def check_health(self, server_type: ServerType, id: int):
        request_id = str(uuid.uuid4())
        q = asyncio.Queue()
        self.queues[request_id] = q
        try:
            req = HeartbeatRequest(request_id=request_id)
            resp = await self._send_request(
                server_type, RequestType.HEARTBEAT, req, q, expect_stream=False
            )
            return isinstance(resp, HeartbeatResponse) and resp.status == "OK"
        finally:
            self.queues.pop(request_id, None)

    async def get_metrics(self, server_type: ServerType, id: int):
        request_id = str(uuid.uuid4())
        q = asyncio.Queue()
        self.queues[request_id] = q
        try:
            req = MetricsRequest(request_id=request_id)
            async for resp in self._send_request(
                server_type, RequestType.METRICS, req, q, expect_stream=False
            ):
                if (
                    isinstance(resp, MetricsResponse)
                    and resp.metrics is not None
                ):
                    key = (server_type, id)
                    cnt = self.proxy_to_time_count[key]
                    total = self.proxy_to_time_total[key]
                    avg = total / cnt if cnt > 0 else 0.0
                    for engine_id in resp.metrics:
                        resp.metrics[engine_id].update(
                            {
                                f"proxy_to_{server_type.name.lower()}_time_avg": avg
                            }
                        )
                    return resp.metrics
                elif isinstance(resp, Exception):
                    raise resp
                return None
        finally:
            self.queues.pop(request_id, None)

    async def start_profile(self) -> None:
        raise NotImplementedError

    async def stop_profile(self) -> None:
        raise NotImplementedError

    async def reset_prefix_cache(self, device: Optional[Device] = None) -> None:
        raise NotImplementedError

    async def sleep(self, level: int = 1) -> None:
        raise NotImplementedError

    async def wake_up(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    async def is_sleeping(self) -> bool:
        return False

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    @property
    def errored(self) -> bool:
        return False

    def dead_error(self) -> Exception:
        return Exception("PDController has failed.")

    def is_running(self) -> bool:
        return True

    def is_stopped(self) -> bool:
        return False

    async def reset_mm_cache(self) -> None:
        raise NotImplementedError


def _has_mm_data(prompt: PromptType) -> bool:
    if isinstance(prompt, dict):
        return "multi_modal_data" in prompt
    return False


def _encode_mm_data(mm_data: dict[str, Any]) -> dict[str, Any]:
    images = mm_data.get("image", [])
    if not isinstance(images, list):
        images = [images]
    encoded_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            encoded_img = {
                "type": "ndarray",
                "data": img.tobytes(),
                "shape": img.shape,
                "dtype": str(img.dtype),
            }
            encoded_images.append(encoded_img)
    return {"image": encoded_images}
