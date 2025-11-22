# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import asyncio
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
from vllm.engine.protocol import EngineClient
from vllm.inputs.data import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import Device, get_ip, get_open_port

from lm_service.protocol.protocol import (
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
)
from lm_service.request_stats import RequestStatsMonitor
from lm_service.routing_logic import (
    RoutingInterface,
    RandomRouter,
    RoundRobinRouter,
    LeastInFlightRouter,
)
from lm_service.service_discovery import HealthCheckServiceDiscovery
from lm_service.stats_loggers import MetricsReporter
import lm_service.envs as lm_service_envs
from lm_service.metastore_client.factory import (
    MetastoreClientFactory,
)
from lm_service.metastore_client.metastore_client_config import (
    MetastoreClientConfig,
    json_to_metastore_config,
)
from lm_service.utils import is_addr_ipv6

from lm_service.logger_utils import init_logger

logger = init_logger(__name__)

ROUTER_MAP = {
    "RandomRouter": RandomRouter,
    "RoundRobinRouter": RoundRobinRouter,
    "LeastInFlightRouter": LeastInFlightRouter,
}

SERVER_PARAMS_MAP = {
    ServerType.E_INSTANCE: {
        "addr_list_name": "encode_addr_list",
        "run_request_type": RequestType.ENCODE,
    },
    ServerType.P_INSTANCE: {
        "addr_list_name": "p_encode_addr_list",
        "run_request_type": RequestType.PREFILL,
    },
    ServerType.D_INSTANCE: {
        "addr_list_name": "d_encode_addr_list",
        "run_request_type": RequestType.GENERATION,
    },
    ServerType.PD_INSTANCE: {
        "addr_list_name": "pd_addr_list",
        "run_request_type": RequestType.GENERATION,
    },
}


class InstanceCluster:
    """
    Encapsulates per-server-type runtime components.
    """

    def __init__(
        self,
        server_type,
        sockets,
        service_discovery,
        stats_monitor,
        router,
        metrics_logger,
    ):
        self.server_type = server_type
        self.sockets = sockets
        self.service_discovery = service_discovery
        self.stats_monitor = stats_monitor
        self.router = router
        self.metrics_logger = metrics_logger
        self.encoder = msgspec.msgpack.Encoder()

    def _prepare_run(self, request):
        if not self.sockets:
            raise RuntimeError(f"No available {self.server_type.name} workers.")

        # encode payload
        try:
            payload = self.encoder.encode(request)
        except Exception as e:
            raise RuntimeError("Failed to serialize GenerationRequest") from e

        msg = (SERVER_PARAMS_MAP[self.server_type]["run_request_type"], payload)
        health_endpoints = self._get_health_endpoints()
        request_stats = self.stats_monitor.get_request_stats()
        # routing
        addr = self.router.route_request(health_endpoints, request_stats)
        return addr, msg

    async def run_with_stream(self, request, q):
        addr, msg = self._prepare_run(request)
        self.stats_monitor.on_new_request(addr, request_id=request.request_id)
        try:
            socket = self.sockets[addr]
            if lm_service_envs.TIMECOUNT_ENABLED:
                start_time = time.perf_counter()
            await socket.send_multipart(msg, copy=False)
            finished = False
            while not finished:
                response = await self._await_with_timeout(request.request_id, q)
                if isinstance(response, Exception):
                    raise response
                self._record_proxy_to_instance_time(addr, response, start_time)
                finished = response.finish_reason is not None
                yield response

        finally:
            self.stats_monitor.on_request_completed(
                addr, request_id=request.request_id
            )

    async def run_without_stream(self, request, q):
        addr, msg = self._prepare_run(request)
        self.stats_monitor.on_new_request(addr, request_id=request.request_id)
        try:
            socket = self.sockets[addr]
            if lm_service_envs.TIMECOUNT_ENABLED:
                start_time = time.perf_counter()
            await socket.send_multipart(msg, copy=False)
            response = await self._await_with_timeout(request.request_id, q)
            self._record_proxy_to_instance_time(addr, response, start_time)
            if isinstance(response, Exception):
                raise response
        finally:
            self.stats_monitor.on_request_completed(
                addr, request_id=request.request_id
            )

    def _record_proxy_to_instance_time(self, addr, response, start_time):
        if (
            lm_service_envs.TIMECOUNT_ENABLED
            and isinstance(response, GenerationResponse)
            and response.proxy_to_worker_time_end
            and start_time is not None
        ):
            self.metrics_logger.add_proxy_to_instance_time(
                addr,
                response.proxy_to_worker_time_end - start_time,
            )

    async def _await_with_timeout(
        self,
        request_id: str,
        q: asyncio.Queue[Union[Exception, GenerationResponse]],
    ) -> Union[Exception, GenerationResponse]:
        """wait for response from queue with timeout handling."""
        try:
            resp = await asyncio.wait_for(
                q.get(),
                timeout=lm_service_envs.LM_SERVICE_REQUEST_TIMEOUT_SECONDS,
            )
            return resp
        except asyncio.TimeoutError:
            return RuntimeError(
                f"Request {request_id} timed out "
                f"after {lm_service_envs.LM_SERVICE_REQUEST_TIMEOUT_SECONDS}s "
                f"without worker response."
            )

    def get_metrics(self):
        return self.metrics_logger.get_metrics()

    def _get_health_endpoints(self):
        return self.service_discovery.get_health_endpoints()

    def _route_request(self, health_endpoints, request_stats):
        return self.router.route_request(health_endpoints, request_stats)

    def lazy_init_health_monitor(self):
        if self.should_launch_health_monitor():
            self.launch_health_monitor()

    def should_launch_health_monitor(self):
        return self.service_discovery.should_launch_health_monitor()

    def launch_health_monitor(self):
        self.service_discovery.launch_health_monitor()

    def get_unhealthy_endpoints(self):
        return self.service_discovery.get_unhealth_endpoints()

    def get_avg_proxy_ttft(self):
        return self.metrics_logger.get_avg_proxy_ttft()

    def cal_proxy_ttft(
        self,
        ttft_recorded_flag: bool,
        proxy_ttft_start: float,
        response: GenerationResponse,
    ) -> bool:
        return self.metrics_logger.cal_proxy_ttft(
            ttft_recorded_flag,
            proxy_ttft_start,
            response,
        )

    def get_avg_proxy_to_instance_time(self, addr: str) -> float:
        return self.metrics_logger.get_avg_proxy_to_instance_time(addr)


class Proxy(EngineClient):
    """
    Proxy
    """

    def __init__(
        self,
        proxy_addr: Optional[str] = None,
        encode_addr_list: Optional[list[str]] = None,
        pd_addr_list: Optional[list[str]] = None,
        p_addr_list: Optional[list[str]] = None,
        d_addr_list: Optional[list[str]] = None,
        model_name: str = "",
        router: type[RoutingInterface] = RandomRouter,
        enable_health_monitor: bool = True,
        health_check_interval: float = 10.0,
        health_threshold: int = 3,
        transfer_protocol: Optional[str] = None,
        metastore_client_config: Optional[dict] = None,
    ):
        init_params = locals()
        self.instance_clusters: dict[ServerType, InstanceCluster] = {}
        self.queues: dict[str, asyncio.Queue] = {}
        # This "Encoder" is used for handling message types, not for "Encode - Prefill - Decode"
        self.encoder = msgspec.msgpack.Encoder()
        self.transfer_protocol = (
            lm_service_envs.TRANSFER_PROTOCOL or transfer_protocol or "ipc"
        )
        self.ctx = zmq.asyncio.Context()
        self.enable_health_monitor = enable_health_monitor
        self.health_check_interval = health_check_interval
        self.health_threshold = health_threshold
        self.output_handler: Optional[asyncio.Task] = None
        self.router = router
        self.is_pd_merged = True
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
        if (
            metastore_client_config is not None
            or lm_service_envs.LM_SERVICE_METASTORE_CLIENT is not None
        ):
            config: MetastoreClientConfig = json_to_metastore_config(
                metastore_client_config
            )
            local_ip = get_ip()
            proxy_port = (
                int(lm_service_envs.LM_SERVICE_RPC_PORT)
                if lm_service_envs.LM_SERVICE_RPC_PORT
                else get_open_port()
            )
            proxy_addr = f"{local_ip}:{proxy_port}"
            self.proxy_addr = f"{self.transfer_protocol}://{proxy_addr}"
            if is_addr_ipv6(proxy_addr) and self.transfer_protocol == "tcp":
                self.ctx.setsockopt(zmq.constants.IPV6, 1)
            self.metastore_client = (
                MetastoreClientFactory.create_metastore_client(
                    config=config,
                    node_info=self.proxy_addr,
                    engine_type=ServerType.PROXY.value,
                    to_encode_sockets=self.to_encode_sockets,
                    to_pd_sockets=self.to_pd_sockets,
                    to_p_sockets=self.to_p_sockets,
                    to_d_sockets=self.to_d_sockets,
                )
            )
            self.is_pd_merged = self.metastore_client.is_pd_merge
        else:
            self._validate_input_addr_and_judge_pd_merged(
                proxy_addr=proxy_addr,
                encode_addr_list=encode_addr_list,
                pd_addr_list=pd_addr_list,
                p_addr_list=p_addr_list,
                d_addr_list=d_addr_list,
            )

        # TODO : amy-why should confirm what is the proxy_addr ipv6 judging rule
        self.proxy_addr = f"{self.transfer_protocol}://{proxy_addr}"
        if is_addr_ipv6(self.proxy_addr) and self.transfer_protocol == "tcp":
            self.ctx.setsockopt(zmq.constants.IPV6, 1)

        for server_type in SERVER_PARAMS_MAP:
            addr_param_name = SERVER_PARAMS_MAP[server_type]["addr_list_name"]
            addr_list = init_params[str(addr_param_name)]
            if addr_list is None:
                continue

            addr_list = [
                f"{self.transfer_protocol}://{addr}" for addr in addr_list
            ]
            sockets = self.connect_to_socket(addr_list)
            self._initialize_instance_clusters(server_type, sockets)

    def _validate_input_addr_and_judge_pd_merged(
        self,
        proxy_addr,
        encode_addr_list,
        pd_addr_list,
        p_addr_list,
        d_addr_list,
    ):
        if not proxy_addr:
            raise ValueError("proxy_addr must be provided")

        if not encode_addr_list:
            raise ValueError("encode_addr_list must be provided")

        if pd_addr_list is not None and not p_addr_list and not d_addr_list:
            self.is_pd_merged = True
            return

        if (
            pd_addr_list is None
            and p_addr_list is not None
            and d_addr_list is not None
        ):
            self.is_pd_merged = False
            return

        raise ValueError(
            "Either pd_addr_list or both p_addr_list and d_addr_list must be provided"
        )

    def _initialize_instance_clusters(
        self,
        engine_type: ServerType,
        socket_dict: dict[str, zmq.asyncio.Socket],
    ):
        service_discovery = HealthCheckServiceDiscovery(
            server_type=engine_type,
            instances=socket_dict,
            enable_health_monitor=self.enable_health_monitor,
            health_check_interval=self.health_check_interval,
            health_threshold=self.health_threshold,
            health_check_func=self.check_health,
        )
        metrics_logger = MetricsReporter(
            server_type=engine_type,
            instances=socket_dict,
            get_metrics_func=self.get_metrics,
        )
        request_stats_monitor = RequestStatsMonitor(socket_dict)
        route_policy = f"LM_SERVICE_{engine_type.name}_ROUTER"
        instance_router = (
            ROUTER_MAP.get(getattr(lm_service_envs, route_policy), None)
            or self.router
        )()
        self.instance_clusters[engine_type] = InstanceCluster(
            server_type=engine_type,
            sockets=socket_dict,
            service_discovery=service_discovery,
            stats_monitor=request_stats_monitor,
            router=instance_router,
            metrics_logger=metrics_logger,
        )

    def shutdown(self):
        self.ctx.destroy()
        if (task := self.output_handler) is not None:
            task.cancel()

        socket_path = self.proxy_addr.replace(
            f"{self.transfer_protocol}://", ""
        )
        if self.transfer_protocol == "ipc" and os.path.exists(socket_path):
            os.remove(socket_path)

    async def log_metrics(self) -> None:
        if self.is_pd_merged:
            await self.instance_clusters[ServerType.PD_INSTANCE].get_metrics()
        else:
            await self.instance_clusters[ServerType.P_INSTANCE].get_metrics()
            await self.instance_clusters[ServerType.D_INSTANCE].get_metrics()

        await self.instance_clusters[ServerType.E_INSTANCE].get_metrics()

    def connect_to_socket(
        self, addr_list: list[str]
    ) -> dict[str, zmq.asyncio.Socket]:
        """
        Connect to a list of ZMQ PUSH sockets.

        Args:
            addr_list: A list of ZMQ socket addresses to connect to.

        Returns:
            A dict of connected ZMQ PUSH sockets, with addr as key.
        """
        to_sockets = {}
        for addr in addr_list:
            socket = self.ctx.socket(zmq.constants.PUSH)
            socket.connect(addr)
            to_sockets[addr] = socket
        return to_sockets

    # TODO : we split the _run method into 2 methods for better readability
    # Since we need two ways, one is stream output and the other is not
    async def _run_without_stream(
        self,
        server_type: ServerType,
        request: GenerationRequest,
        q: asyncio.Queue[Union[Exception, GenerationResponse]],
    ):
        cluster = self.instance_clusters[server_type]
        await cluster.run_without_stream(request, q)

    async def _run_with_stream(
        self,
        server_type: ServerType,
        request: GenerationRequest,
        q: asyncio.Queue[Union[Exception, GenerationResponse]],
    ):
        cluster = self.instance_clusters[server_type]
        yield cluster.run_with_stream(request, q)

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

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ):
        # lazy initialization
        if self.output_handler is None:
            self.output_handler = asyncio.create_task(
                self._run_output_handler()
            )

        # lazy init all health monitors
        for cluster in self.instance_clusters.values():
            cluster.lazy_init_health_monitor()

        if not request_id:
            request_id = uuid.uuid4().hex

        q: asyncio.Queue = asyncio.Queue()
        if request_id in self.queues:
            raise ValueError(f"Request id {request_id} already running.")
        else:
            self.queues[request_id] = q

        # Support both raw string prompts and dict prompts with multimodal data
        prompt_text = prompt["prompt"] if isinstance(prompt, dict) else prompt

        request = GenerationRequest(
            request_id=request_id,
            prompt=prompt_text,
            sampling_params=sampling_params,
            proxy_addr=self.proxy_addr,
        )

        try:
            proxy_ttft_start: float = time.perf_counter()
            ttft_recorded_flag: bool = False
            # need to validate to avoid decode failed later
            req_dict = msgspec.to_builtins(request)
            request = msgspec.convert(req_dict, GenerationRequest, strict=True)

            if _has_mm_data(prompt):
                request.multi_modal_data = _encode_mm_data(
                    prompt["multi_modal_data"]
                )
                await self._run_without_stream(
                    ServerType.E_INSTANCE, request, q
                )

            if self.is_pd_merged:
                pd_cluster = self.instance_clusters[ServerType.PD_INSTANCE]
                async for pd_response in self._run_with_stream(
                    ServerType.PD_INSTANCE, request, q
                ):
                    yield self._to_request_output(pd_response)
                    ttft_recorded_flag = pd_cluster.cal_proxy_ttft(
                        ttft_recorded_flag,
                        proxy_ttft_start,
                        pd_response,
                    )
            else:
                await self._run_without_stream(
                    ServerType.P_INSTANCE, request, q
                )
                d_cluster = self.instance_clusters[ServerType.D_INSTANCE]
                async for d_response in self._run_with_stream(
                    ServerType.D_INSTANCE, request, q
                ):
                    yield self._to_request_output(d_response)
                    ttft_recorded_flag = d_cluster.cal_proxy_ttft(
                        ttft_recorded_flag,
                        proxy_ttft_start,
                        d_response,
                    )

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

    def get_unhealthy_task(self, engine_type):
        """Return a Task to abort requests from unhealthy endpoints.

        If there are no unhealthy endpoints, return None.
        """
        cluster = self.instance_clusters[engine_type]
        unhealthy_endpoints = cluster.get_unhealthy_endpoints()

        if not unhealthy_endpoints:
            return None  # nothing to do

        return self.abort_requests_from_unhealth_endpoints(
            server_type=engine_type,
            unhealth_endpoints=unhealthy_endpoints,
            request_stats_monitor=cluster.stats_monitor,
        )

    async def _run_output_handler(self) -> None:
        """Background task to pull responses and dispatch to request queues.

        Binds a PULL socket on proxy_addr and receives multipart messages of
        the form (response_type, payload). Decodes payload into a
        GenerationResponse and enqueues it into the corresponding request queue
        keyed by request_id.
        """
        socket: Optional[zmq.asyncio.Socket] = None

        # prepare decoders once
        gen_decoder = msgspec.msgpack.Decoder(GenerationResponse)
        heartbeat_decoder = msgspec.msgpack.Decoder(HeartbeatResponse)
        failure_decoder = msgspec.msgpack.Decoder(FailureResponse)
        metrics_decoder = msgspec.msgpack.Decoder(MetricsResponse)

        resp_type_decoder_map = {
            ResponseType.GENERATION: gen_decoder,
            ResponseType.ENCODE: gen_decoder,
            ResponseType.PREFILL: gen_decoder,
            ResponseType.HEARTBEAT: heartbeat_decoder,
            ResponseType.FAILURE: failure_decoder,
            ResponseType.METRICS: metrics_decoder,
        }

        timeout = self.health_check_interval * self.health_threshold / 2

        try:
            socket = self.ctx.socket(zmq.PULL)
            socket.bind(self.proxy_addr)

            while True:
                # ---- health check tasks ----
                tasks = [
                    t
                    for server_type in self.instance_clusters
                    if (t := self.get_unhealthy_task(server_type)) is not None
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # ---- no response from workers ----
                if not await socket.poll(timeout):
                    continue

                # ---- receive and decode ----
                resp_type, payload = await socket.recv_multipart()
                decoder = resp_type_decoder_map.get(resp_type)
                if decoder is None:
                    raise RuntimeError(
                        f"Unknown response type from worker: {resp_type.decode()}"
                    )

                resp = decoder.decode(payload)
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

    async def check_health(self, server_type: ServerType, addr: str):
        # lazy initialization
        if self.output_handler is None:
            self.output_handler = asyncio.create_task(
                self._run_output_handler()
            )
        request_id = str(uuid.uuid4())
        request = HeartbeatRequest(
            request_id=request_id, proxy_addr=self.proxy_addr
        )
        q: asyncio.Queue = asyncio.Queue()
        self.queues[request_id] = q
        try:
            payload = self.encoder.encode(request)
            msg = (RequestType.HEARTBEAT, payload)
            socket = self.instance_clusters[server_type].sockets[addr]
            await socket.send_multipart(msg, copy=False)
            response = await q.get()
            if (
                isinstance(response, HeartbeatResponse)
                and response.status == "OK"
            ):
                return True
            elif isinstance(response, Exception):
                raise response
            else:
                return False

        except Exception as e:
            raise RuntimeError(
                f"Health check failed for {server_type} {addr}, exception: {e}"
            ) from e
        finally:
            self.queues.pop(request_id, None)

    async def get_metrics(self, server_type: ServerType, addr: str):
        request_id = str(uuid.uuid4())
        request = MetricsRequest(
            request_id=request_id, proxy_addr=self.proxy_addr
        )
        q: asyncio.Queue = asyncio.Queue()
        self.queues[request_id] = q
        try:
            payload = self.encoder.encode(request)
            msg = (RequestType.METRICS, payload)
            cluster = self.instance_clusters[server_type]
            socket = cluster.sockets[addr]
            await socket.send_multipart(msg, copy=False)
            response = await q.get()
            # calculate proxy to pd/encode time
            if (
                isinstance(response, MetricsResponse)
                and response.metrics is not None
            ):
                # calculate proxy to pd/encode time average
                # add to metrics
                proxy_ttft_avg: float = 0.0
                proxy2instance_avg: float = (
                    cluster.get_avg_proxy_to_instance_time(addr)
                )
                if server_type in [
                    ServerType.PD_INSTANCE,
                    ServerType.D_INSTANCE,
                ]:
                    proxy_ttft_avg = cluster.get_avg_proxy_ttft()
                for engine_id in response.metrics:
                    response.metrics[engine_id].update(
                        {
                            "proxy_to_instance_time_avg": proxy2instance_avg,  # type: ignore
                            "proxy_ttft_avg": proxy_ttft_avg,  # type: ignore
                        }
                    )

                return response.metrics
            elif isinstance(response, Exception):
                raise response
            else:
                return None

        except Exception as e:
            raise RuntimeError(
                "Get metrics failed for %s %s, exception: %s"
                % (server_type, addr, e)
            ) from e
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
