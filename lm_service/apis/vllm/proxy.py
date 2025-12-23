# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import asyncio
from http.client import HTTPException
import os
import time
from PIL import Image
import uuid
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Optional, Union

import msgspec
import numpy as np
import zmq
import zmq.asyncio
import aiohttp
from vllm.config import ModelConfig, VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.inputs.data import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import (
    AnyTokenizer,
    init_tokenizer_from_configs,
)
from vllm.tasks import SupportedTask
from vllm.utils import Device, get_ip, get_open_port
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
from lm_service.request_stats import RequestStatsMonitor
from lm_service.routing_logic import (
    RoutingInterface,
    RandomRouter,
    RoundRobinRouter,
    LeastInFlightRouter,
)
from lm_service.service_discovery import (
    HealthCheckServiceDiscovery,
    HTTPHealthCheckServiceDiscovery,
)
from lm_service.stats_loggers import MetricsReporter, HTTPMetricsReporter
import lm_service.envs as lm_service_envs
from lm_service.metastore_client.factory import (
    MetastoreClientFactory,
)
from lm_service.metastore_client.metastore_client import (
    MetastoreClientBase,
)
from lm_service.metastore_client.metastore_client_config import (
    MetastoreClientConfig,
    json_to_metastore_config,
)
from lm_service.utils import is_addr_ipv6

from lm_service.logger_utils import init_logger

from lm_service.instance_cluster import (
    InstanceCluster,
    HTTPInstanceCluster,
    SERVER_PARAMS_MAP,
)

logger = init_logger(__name__)

ROUTER_MAP = {
    "RandomRouter": RandomRouter,
    "RoundRobinRouter": RoundRobinRouter,
    "LeastInFlightRouter": LeastInFlightRouter,
}


class BaseProxy(EngineClient):
    def __init__(
        self,
        vllm_config: Optional[VllmConfig],
        model_name: str,
        router: type[RoutingInterface],
        enable_health_monitor: bool,
        health_check_interval: float,
        health_threshold: int,
        log_stats: bool = True,
    ):
        self.vllm_config = vllm_config
        # Validate input parameters for some components
        self._check_type("enable_health_monitor", enable_health_monitor, bool)
        self._check_positive("health_check_interval", health_check_interval)
        self._check_positive("health_threshold", health_threshold)
        self._check_subclass("router", router, RoutingInterface)

        self.log_stats = log_stats
        self.enable_health_monitor = enable_health_monitor
        self.health_check_interval = health_check_interval
        self.health_threshold = health_threshold
        self.router = router
        self.is_pd_merged = True
        self.tokenizer = (
            init_tokenizer_from_configs(model_config=vllm_config.model_config)
            if vllm_config
            else None
        )

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

    def _check_type(self, name, value, expected_type):
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{name} must be {expected_type.__name__}, ",
                f"got {type(value).__name__}",
            )

    def _check_positive(self, name, value):
        try:
            if value <= 0:
                raise ValueError
        except Exception:
            raise ValueError(f"{name} must be a positive number")

    def _check_subclass(self, name, value, base_class):
        if not isinstance(value, type) or not issubclass(value, base_class):
            raise TypeError(
                f"{name} must be a subclass of {base_class.__name__}"
            )

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
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer not initialized. Ensure vllm_config "
                "is provided when creating the Proxy instance."
            )

        return self.tokenizer

    def get_check_health_results(self) -> dict[str, dict[str, bool]]:
        # Return health check results for each server type
        results: dict[str, dict[str, bool]] = {}
        for server_type, cluster in self.instance_clusters.items():
            service_discovery = cluster.service_discovery
            states = service_discovery.get_instances_states()
            results[server_type.name] = states
        return results

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(self) -> None:
        pass

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

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate",)


class Proxy(BaseProxy):
    """
    Proxy
    """

    def __init__(
        self,
        vllm_config: Optional[VllmConfig] = None,
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
        log_stats: bool = True,
    ):
        super().__init__(
            vllm_config,
            model_name,
            router,
            enable_health_monitor,
            health_check_interval,
            health_threshold,
            log_stats,
        )
        self.instance_clusters: dict[ServerType, InstanceCluster] = {}
        self.queues: dict[str, asyncio.Queue] = {}
        # This "Encoder" is used for handling message types, not for "Encode - Prefill - Decode"
        self.encoder = msgspec.msgpack.Encoder()
        self.transfer_protocol = (
            lm_service_envs.TRANSFER_PROTOCOL or transfer_protocol or "ipc"
        )
        self.ctx = zmq.asyncio.Context()
        self.output_handler: Optional[asyncio.Task] = None
        self.metastore_client: Optional[MetastoreClientBase] = None
        # Logically, there is no essential difference between PD instances and D instances in handling tokens.
        # Therefore, in internal processing, we treat them as equivalent to simplify the logic.

        if (
            metastore_client_config is not None
            or lm_service_envs.LM_SERVICE_METASTORE_CLIENT is not None
        ):
            self._init_cluster_with_metastore(
                metastore_client_config, proxy_addr
            )
        else:
            self._init_cluster_with_addr_list(
                proxy_addr,
                encode_addr_list,
                p_addr_list,
                d_addr_list or pd_addr_list,
            )

    def _init_cluster_with_addr_list(
        self,
        proxy_addr,
        encode_addr_list,
        p_addr_list,
        d_addr_list,
    ):
        if not proxy_addr:
            raise ValueError("proxy_addr must be provided")

        if not encode_addr_list:
            raise ValueError("encode_addr_list must be provided")

        if not d_addr_list:
            raise ValueError("d_addr_list or pd_addr_list must be provided")

        self.is_pd_merged = not bool(p_addr_list)
        self.proxy_addr = f"{self.transfer_protocol}://{proxy_addr}"
        if is_addr_ipv6(proxy_addr) and self.transfer_protocol == "tcp":
            self.ctx.setsockopt(zmq.constants.IPV6, 1)
        init_params = locals()
        active_server_types = [ServerType.E_INSTANCE] + (
            [ServerType.PD_INSTANCE]
            if self.is_pd_merged
            else [ServerType.P_INSTANCE, ServerType.D_INSTANCE]
        )
        for server_type in active_server_types:
            addr_param_name = str(
                SERVER_PARAMS_MAP[server_type]["addr_list_name"]
            )
            addr_list = [
                f"{self.transfer_protocol}://{addr}"
                for addr in (init_params.get(addr_param_name) or [])
            ]
            sockets = self.connect_to_socket(addr_list)
            self._initialize_instance_clusters(server_type, sockets)

    def _init_cluster_with_metastore(self, metastore_client_config, proxy_addr):
        config: MetastoreClientConfig = json_to_metastore_config(
            metastore_client_config
        )
        if proxy_addr is None:
            local_ip = lm_service_envs.LM_SERVICE_HOST_IP or get_ip()
            proxy_port = (
                int(lm_service_envs.LM_SERVICE_RPC_PORT)
                if lm_service_envs.LM_SERVICE_RPC_PORT
                else get_open_port()
            )
            proxy_addr = f"{local_ip}:{proxy_port}"

        self.proxy_addr = f"{self.transfer_protocol}://{proxy_addr}"
        if is_addr_ipv6(proxy_addr) and self.transfer_protocol == "tcp":
            self.ctx.setsockopt(zmq.constants.IPV6, 1)

        to_e_sockets: dict[str, zmq.asyncio.Socket] = {}
        to_p_sockets: dict[str, zmq.asyncio.Socket] = {}
        to_d_sockets: dict[str, zmq.asyncio.Socket] = {}

        self.metastore_client = MetastoreClientFactory.create_metastore_client(
            config=config,
            node_info=self.proxy_addr,
            server_type=ServerType.PROXY.value,
            to_e_sockets=to_e_sockets,
            to_p_sockets=to_p_sockets,
            to_d_sockets=to_d_sockets,
        )
        self.is_pd_merged = self.metastore_client.is_pd_merged
        init_params = locals()
        active_server_types = [ServerType.E_INSTANCE] + (
            [ServerType.PD_INSTANCE]
            if self.is_pd_merged
            else [ServerType.P_INSTANCE, ServerType.D_INSTANCE]
        )
        for server_type in active_server_types:
            sockets = init_params[
                SERVER_PARAMS_MAP[server_type]["socket_list_name"]
            ]
            self._initialize_instance_clusters(server_type, sockets)

    def _initialize_instance_clusters(
        self,
        engine_type: ServerType,
        socket_dict: dict[str, zmq.asyncio.Socket],
    ):
        lock = asyncio.Lock()
        service_discovery = HealthCheckServiceDiscovery(
            server_type=engine_type,
            instances=socket_dict,
            enable_health_monitor=self.enable_health_monitor,
            health_check_interval=self.health_check_interval,
            health_threshold=self.health_threshold,
            health_check_func=self.check_health,
            lock=lock,
        )
        metrics_logger = MetricsReporter(
            server_type=engine_type,
            instances=socket_dict,
            get_metrics_func=self.fetch_metrics_from_instance,
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
            socket_lock=lock,
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

        if self.metastore_client is not None:
            self.metastore_client.close()

    # TODO: Optimize log metrics logic; make it a built-in capability
    # and print at regular intervals.
    async def log_metrics(self) -> None:
        # lazy initialization
        if self.output_handler is None:
            self.output_handler = asyncio.create_task(
                self._run_output_handler()
            )
        for server_type in self.instance_clusters:
            cluster = self.instance_clusters[server_type]
            await cluster.log_metrics()

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
            logger.info(f"Connected to worker {addr} success")
        return to_sockets

    async def _process_request(
        self,
        server_type: ServerType,
        request: GenerationRequest,
        q: asyncio.Queue[Union[Exception, GenerationResponse]],
    ):
        cluster = self.instance_clusters[server_type]
        response = await cluster.process_request(request, q)
        return response

    async def _process_request_streaming_response(
        self,
        server_type: ServerType,
        request: GenerationRequest,
        q: asyncio.Queue[Union[Exception, GenerationResponse]],
    ):
        cluster = self.instance_clusters[server_type]
        async for resp in cluster.process_request_streaming_response(
            request, q
        ):
            yield resp

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
            capture_metrics_result=resp.capture_metrics_result,
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

        if not request_id:
            request_id = uuid.uuid4().hex

        q: asyncio.Queue = asyncio.Queue()
        if request_id in self.queues:
            raise ValueError(f"Request id {request_id} already running.")
        else:
            self.queues[request_id] = q

        enable_metrics: Optional[dict[str, bool]] = None
        # Support both raw string prompts and dict prompts with multimodal data.
        if isinstance(prompt, dict):
            enable_metrics = prompt.get("enable_metrics", None)
            if "prompt" in prompt:
                prompt_text = prompt["prompt"]
                prompt_token_ids = None
            elif "prompt_token_ids" in prompt:
                prompt_text = None
                prompt_token_ids = prompt["prompt_token_ids"]
            else:
                raise ValueError(
                    "Invalid prompt dictionary: "
                    "must contain 'prompt' or 'prompt_token_ids'."
                )
        else:
            # raw string prompt
            prompt_text = prompt
            prompt_token_ids = None

        request = GenerationRequest(
            request_id=request_id,
            prompt=prompt_text,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            proxy_addr=self.proxy_addr,
            enable_metrics=enable_metrics,
        )

        try:
            proxy_ttft_start: float = time.perf_counter()
            ttft_recorded_flag: bool = False
            # need to validate to avoid decode failed later
            req_dict = msgspec.to_builtins(request)
            request = msgspec.convert(req_dict, GenerationRequest, strict=True)

            # Step 1 : Encode the multimodal data if any
            encode_time: float = 0.0
            if _has_mm_data(prompt):
                request.multi_modal_data = _encode_mm_data(
                    prompt["multi_modal_data"]
                )
                await self._process_request(ServerType.E_INSTANCE, request, q)
                encode_time = cal_exec_time(start=proxy_ttft_start)

            # Step 2 : Maybe Prefill
            if not self.is_pd_merged:
                response = await self._process_request(
                    ServerType.P_INSTANCE, request, q
                )
                kv_transfer_params = response.kv_transfer_params
                request.sampling_params.extra_args["kv_transfer_params"] = (
                    kv_transfer_params
                )

            # Step 3 : Decode
            decode_server_type = (
                ServerType.PD_INSTANCE
                if self.is_pd_merged
                else ServerType.D_INSTANCE
            )
            decode_cluster = self.instance_clusters[decode_server_type]
            async for d_response in self._process_request_streaming_response(
                decode_server_type, request, q
            ):
                if metrics_enabled(request, "encode"):
                    if not (metrics := d_response.capture_metrics_result):
                        d_response.capture_metrics_result = metrics = {}
                    metrics["encode_time_ms"] = encode_time * 1000
                yield self._to_request_output(d_response)
                ttft_recorded_flag = decode_cluster.cal_proxy_ttft(
                    ttft_recorded_flag,
                    proxy_ttft_start,
                    d_response,
                )
        except msgspec.ValidationError as e:
            raise RuntimeError(f"Invalid Parameters: {e}.") from e
        except RuntimeError as e:
            logger.error(f"Runtime error during generate: {e}")
        except Exception as e:
            # Log any unexpected exception but do not re-raise to ensure
            # request cleanup in finally block.
            logger.error("Unexpected error during generate: %s", e)
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

    def add_unhealthy_task(self, engine_type, tasks):
        """Return a Task to abort requests from unhealthy endpoints.

        If there are no unhealthy endpoints, return None.
        """
        cluster = self.instance_clusters[engine_type]
        unhealthy_endpoints = cluster.get_unhealthy_endpoints()

        if not unhealthy_endpoints:
            return

        tasks.append(
            self.abort_requests_from_unhealth_endpoints(
                server_type=engine_type,
                unhealth_endpoints=unhealthy_endpoints,
                request_stats_monitor=cluster.stats_monitor,
            )
        )

    async def _worker_register_handler(
        self, worker_register_req: WorkerRegisterRequest
    ):
        """Handle worker register request."""
        address = worker_register_req.address
        server_type = worker_register_req.server_type

        if server_type in self.instance_clusters:
            cluster = self.instance_clusters[server_type]
            socket_dict = cluster.sockets
            if address not in socket_dict:
                try:
                    socket = self.ctx.socket(zmq.constants.PUSH)
                    socket.connect(address)
                except zmq.ZMQError as e:
                    logger.error(
                        f"Failed to connect to worker {address} with error: {e}"
                    )
                    return
                cluster_lock = cluster.socket_lock
                async with cluster_lock:
                    socket_dict[address] = socket
                    logger.info(f"Connected to worker {address} success")
        else:
            logger.error(
                f"_worker_register_handler fail, unknown server type {server_type}"
            )
            return

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
        exit_decoder = msgspec.msgpack.Decoder(ExitRequest)
        worker_register_decoder = msgspec.msgpack.Decoder(WorkerRegisterRequest)

        try:
            socket = self.ctx.socket(zmq.constants.PULL)
            socket.bind(self.proxy_addr)
            timeout = self.health_check_interval * self.health_threshold / 2
            # lazy init all health monitors
            for cluster in self.instance_clusters.values():
                cluster.lazy_init_health_monitor()

            if self.metastore_client is not None:
                self.metastore_client.launch_proxy_task()

            while True:
                # To kill the failed requests quickly, we check unhealthy endpoints
                tasks: list[asyncio.Task] = []
                for engine_type in self.instance_clusters:
                    self.add_unhealthy_task(engine_type, tasks)

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Check if the engine is alive:
                if not await socket.poll(timeout=timeout):
                    continue
                resp_type, payload = await socket.recv_multipart()

                # Decode response according to its type.
                # TODO : judge whether we need to add PREFILL response type
                resp: Union[
                    GenerationResponse,
                    HeartbeatResponse,
                    FailureResponse,
                    MetricsResponse,
                    ExitRequest,
                    WorkerRegisterRequest,
                ]
                # TODO: maybe we can have a mapping from resp_type to prefill
                if resp_type in (
                    ResponseType.GENERATION,
                    ResponseType.ENCODE,
                    ResponseType.PREFILL,
                ):
                    resp = decoder.decode(payload)
                elif resp_type == ResponseType.HEARTBEAT:
                    resp = heartbeat_decoder.decode(payload)
                elif resp_type == ResponseType.FAILURE:
                    resp = failure_decoder.decode(payload)
                elif resp_type == ResponseType.METRICS:
                    resp = metrics_decoder.decode(payload)
                elif resp_type == RequestType.EXIT:
                    resp = exit_decoder.decode(payload)
                    self.create_handle_exit_task(resp)
                elif resp_type == RequestType.REGISTER:
                    resp = worker_register_decoder.decode(payload)
                    asyncio.create_task(self._worker_register_handler(resp))
                else:
                    raise RuntimeError(
                        f"Unknown response type from worker: {resp_type.decode()}"
                    )

                if resp.request_id not in self.queues:
                    if resp_type not in (
                        ResponseType.HEARTBEAT,
                        ResponseType.METRICS,
                        RequestType.EXIT,
                        RequestType.REGISTER,
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
            socket = await self._get_socket_and_server_types_from_addr(
                addr, server_type
            )
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

    async def fetch_metrics_from_instance(
        self, server_type: ServerType, addr: str
    ) -> dict[str, dict[str, str]]:
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
            socket = await self._get_socket_and_server_types_from_addr(
                addr, server_type
            )
            await socket.send_multipart(msg, copy=False)
            response = await q.get()
            # calculate proxy to pd/encode time
            if isinstance(response, Exception):
                raise response
            else:
                # calculate proxy to pd/encode time average
                # add to metrics
                proxy_ttft_avg: float = 0.0
                proxy2instance_avg: float = (
                    cluster.get_avg_proxy_to_instance_time(addr)
                )
                if server_type in (
                    ServerType.D_INSTANCE,
                    ServerType.PD_INSTANCE,
                ):
                    proxy_ttft_avg = cluster.get_avg_proxy_ttft()
                for engine_id in response.metrics:
                    response.metrics[engine_id].update(
                        {
                            "proxy_to_instance_time_avg": proxy2instance_avg,  # type: ignore
                            "proxy_ttft_avg": proxy_ttft_avg,  # type: ignore
                        }
                    )

                return response.metrics

        except Exception as e:
            raise RuntimeError(
                "Get metrics failed for %s %s, exception: %s"
                % (server_type, addr, e)
            ) from e
        finally:
            self.queues.pop(request_id, None)

    async def handle_exit_from_worker(self, req: ExitRequest) -> None:
        # lazy initialization
        if self.output_handler is None:
            self.output_handler = asyncio.create_task(
                self._run_output_handler()
            )
        server_type = req.server_type

        # stop routing new requests to it
        if req.addr and server_type:
            await self._remove_instance_from_registry(req.addr, server_type)
        else:
            logger.warning(
                "Exit instance handling failed, addr or server_type is None.",
            )
            return
        logger.info(
            "Instance %s addr %s is exiting (reason=%s, in_flight=%d).",
            server_type,
            req.addr,
            req.reason,
            req.in_flight,
        )

    def create_handle_exit_task(self, resp: ExitRequest) -> None:
        task = asyncio.create_task(self.handle_exit_from_worker(resp))
        task.add_done_callback(
            lambda t: logger.error(
                "Exception in handle_exit_from_worker: %s",
                t.exception(),
            )
            if t.exception() is not None and not t.cancelled()
            else None
        )

    async def _get_socket_and_server_types_from_addr(
        self,
        addr: str,
        server_type: ServerType,
    ) -> zmq.asyncio.Socket:
        cluster = self.instance_clusters[server_type]
        cluster_lock = cluster.socket_lock
        async with cluster_lock:
            socket = cluster.sockets.get(addr)
        if socket:
            return socket
        raise ValueError(
            f"Address {addr} not found in any {server_type.name} sockets."
        )

    async def _remove_instance_from_registry(
        self, addr: str, server_type: ServerType
    ) -> None:
        cluster = self.instance_clusters[server_type]
        await cluster.service_discovery.remove_instance(addr)


class HTTPProxy(BaseProxy):
    """
    HTTP Proxy
    """

    def __init__(
        self,
        vllm_config: Optional[VllmConfig] = None,
        encode_addr_list: Optional[list[str]] = None,
        pd_addr_list: Optional[list[str]] = None,
        p_addr_list: Optional[list[str]] = None,
        d_addr_list: Optional[list[str]] = None,
        model_name: str = "",
        router: type[RoutingInterface] = RandomRouter,
        enable_health_monitor: bool = True,
        health_check_interval: float = 10.0,
        health_threshold: int = 3,
        log_stats: bool = True,
    ):
        super().__init__(
            vllm_config,
            model_name,
            router,
            enable_health_monitor,
            health_check_interval,
            health_threshold,
            log_stats,
        )
        self.instance_clusters: dict[ServerType, HTTPInstanceCluster] = {}
        self._init_cluster_with_addr_list(
            encode_addr_list,
            p_addr_list,
            d_addr_list or pd_addr_list,
        )

    def _init_cluster_with_addr_list(
        self,
        encode_addr_list,
        p_addr_list,
        d_addr_list,
    ):
        if not encode_addr_list:
            raise ValueError("encode_addr_list must be provided")

        if not d_addr_list:
            raise ValueError("d_addr_list or pd_addr_list must be provided")

        self.is_pd_merged = not bool(p_addr_list)
        init_params = locals()
        active_server_types = [ServerType.E_INSTANCE] + (
            [ServerType.PD_INSTANCE]
            if self.is_pd_merged
            else [ServerType.P_INSTANCE, ServerType.D_INSTANCE]
        )
        for server_type in active_server_types:
            addr_param_name = str(
                SERVER_PARAMS_MAP[server_type]["addr_list_name"]
            )
            urls = init_params.get(addr_param_name) or []
            self._initialize_instance_clusters(server_type, urls)

    def _initialize_instance_clusters(
        self,
        engine_type: ServerType,
        urls: list[str],
    ):
        service_discovery = HTTPHealthCheckServiceDiscovery(
            server_type=engine_type,
            urls=urls,
            enable_health_monitor=self.enable_health_monitor,
            health_check_interval=self.health_check_interval,
            health_threshold=self.health_threshold,
            health_check_func=self.check_health,
        )
        metrics_logger = HTTPMetricsReporter(
            server_type=engine_type,
            urls=urls,
            get_metrics_func=self.fetch_metrics_from_instance,
        )
        request_stats_monitor = RequestStatsMonitor(urls)
        route_policy = f"LM_SERVICE_{engine_type.name}_ROUTER"
        instance_router = (
            ROUTER_MAP.get(getattr(lm_service_envs, route_policy), None)
            or self.router
        )()
        timeout = aiohttp.ClientTimeout(total=100_000)
        keepalive_timeout = int(
            os.getenv("LM_SERVICE_CLIENT_HTTP_TIMEOUT_KEEP_ALIVE", 0)
        )
        connector = aiohttp.TCPConnector(
            limit=0, force_close=False, keepalive_timeout=keepalive_timeout
        )
        self.instance_clusters[engine_type] = HTTPInstanceCluster(
            server_type=engine_type,
            urls=urls,
            service_discovery=service_discovery,
            stats_monitor=request_stats_monitor,
            router=instance_router,
            metrics_logger=metrics_logger,
            session_timeout=timeout,
            session_connector=connector,
        )

    async def generate(self, request_data, request_id):
        # for each of the multi-modal data, extract them and send to encoder
        if not request_id:
            request_id = uuid.uuid4().hex
        proxy_ttft_start: float = time.perf_counter()
        ttft_recorded_flag: bool = False
        mm_items = extract_mm_items(request_data)
        # Step 1 : Encode the multimodal data
        if mm_items:
            await self.encode_mm_data_tasks(
                mm_items, request_id, request_data.get("model")
            )

        # Step 2 : Maybe Prefill
        if not self.is_pd_merged:
            request_data = await self.prefill_request(request_data, request_id)

        # Step 3 : Decode
        headers = {"x-request-id": request_id}
        decode_server_type = (
            ServerType.PD_INSTANCE
            if self.is_pd_merged
            else ServerType.D_INSTANCE
        )
        decode_cluster = self.instance_clusters[decode_server_type]

        async with decode_cluster.process_request_streaming_response(
            json=request_data, headers=headers
        ) as resp:
            resp.raise_for_status()
            yield resp
            ttft_recorded_flag = decode_cluster.cal_proxy_ttft(
                ttft_recorded_flag,
                proxy_ttft_start,
                resp,
            )

    async def encode_mm_data_tasks(self, mm_items, request_id, model):
        tasks = []
        encode_cluster = self.instance_clusters[ServerType.E_INSTANCE]
        for idx, item in enumerate(mm_items):
            child_req_id = f"{request_id}:{idx}:{uuid.uuid4().hex[:6]}"
            headers = {"x-request-id": child_req_id}
            encoder_req = {
                "model": model,
                "messages": [
                    {"role": "user", "content": [item]},
                ],
                "max_tokens": 1,
                "stream": False,
            }
            encode_cluster.add_batch_request(encoder_req, headers, tasks)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Fail fast if any sub-request failed
        for r in results:
            if isinstance(r, Exception):
                logger.error("Encoder request raised: %s", r)
                raise HTTPException(status_code=502, detail=str(r))
            if r.status != 200:
                try:
                    detail = await r.text()
                except Exception:
                    detail = "<unable to read body>"
                logger.error(
                    "Encoder request returned %s: %s", r.status, detail
                )
                raise HTTPException(
                    status_code=r.status,
                    detail=f"Encoder request failed: {detail}",
                )

    async def prefill_request(self, request_data, request_id):
        logger.debug(
            "Processing through prefill for req_id: %s/ url: %s",
            request_id,
            request_data.get("url"),
        )
        prefill_request = request_data.copy()
        prefill_request["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        prefill_request["stream"] = False
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1
        if "stream_options" in prefill_request:
            del prefill_request["stream_options"]

        headers = {"x-request-id": request_id}
        prefill_cluster = self.instance_clusters[ServerType.P_INSTANCE]
        try:
            resp = await prefill_cluster.run_single_request(
                json=prefill_request, headers=headers
            )
            if resp.status != 200:
                error_text = await resp.text()
                raise HTTPException(
                    status_code=resp.status,
                    detail={
                        "error": "Prefill request failed",
                        "message": error_text,
                    },
                )
            logger.debug(
                "Prefill processing completed successfully for req_id: %s",
                request_id,
            )
            return resp
        except Exception as e:
            logger.error("Prefill processing failed: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail={"error": "Prefill processing error", "message": str(e)},
            ) from e

    async def fetch_metrics_from_instance(
        self, server_type: ServerType, url: str
    ) -> dict[str, dict[str, str]]:
        """Fetch metrics from an HTTP instance.

        Uses the single endpoint `{url}/metrics`. The returned JSON is
        expected to be a mapping of engine ids to metric dicts. We augment
        each engine's metrics with proxy-to-instance timing metrics before
        returning.
        """
        cluster = self.instance_clusters[server_type]
        session = cluster.session
        endpoint = f"{url}/metrics"
        try:
            async with session.get(endpoint) as resp:
                resp.raise_for_status()
                resp_json = await resp.json()

            proxy2instance_avg = cluster.get_avg_proxy_to_instance_time(url)
            proxy_ttft_avg: float = 0.0
            if server_type in (ServerType.D_INSTANCE, ServerType.PD_INSTANCE):
                proxy_ttft_avg = cluster.get_avg_proxy_ttft()

            for engine_id in resp_json:
                resp_json[engine_id].update(
                    {
                        "proxy_to_instance_time_avg": proxy2instance_avg,  # type: ignore
                        "proxy_ttft_avg": proxy_ttft_avg,  # type: ignore
                    }
                )

            return resp_json

        except Exception as e:
            raise RuntimeError(
                "Get metrics failed for %s %s, exception: %s"
                % (server_type, url, e)
            ) from e

    async def get_overall_health_states(self):
        overall_health_states = {}
        for engine_type, cluster in self.instance_clusters.items():
            resp = await cluster.service_discovery.get_overall_health_states()
            overall_health_states.update(resp)
        return overall_health_states

    async def shutdown(self):
        for cluster in self.instance_clusters.values():
            await cluster.session.close()

    async def profile(self, cmd: str, payload: dict):
        encode_task = self.instance_clusters[ServerType.E_INSTANCE].profile_cmd(
            cmd, payload
        )
        prefill_task = (
            asyncio.sleep(0)
            if self.is_pd_merged
            else self.instance_clusters[ServerType.P_INSTANCE].profile_cmd(
                cmd, payload
            )
        )
        decode_task = self.instance_clusters[
            ServerType.PD_INSTANCE
            if self.is_pd_merged
            else ServerType.D_INSTANCE
        ].profile_cmd(cmd, payload)
        encode_res, prefill_res, decode_res = await asyncio.gather(
            encode_task, prefill_task, decode_task
        )
        if encode_res is prefill_res is decode_res is None:
            raise HTTPException(
                status_code=503,
                detail="Profiling endpoints are disabled on all clusters",
            )
        return {
            "encode": encode_res,  # may be None
            "prefill": prefill_res,  # may be None
            "decode": decode_res,  # may be None
        }

    async def check_health(self, server_type, addr):
        cluster = self.instance_clusters[server_type]
        session = cluster.session
        endpoint = f"{addr}/health"
        try:
            async with session.get(endpoint) as resp:
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.error(
                f"Health check failed for {server_type} {addr}, exception: {e}"
            )
            return False

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
        elif isinstance(img, Image.Image):
            # Convert PIL Image to bytes
            encoded_img = {
                "type": "pil",
                "data": img.tobytes(),
                "size": img.size,
                "mode": img.mode,
            }
        else:
            raise ValueError(
                f"Unsupported image type: {type(img)}. "
                "Supported types are numpy.ndarray and PIL.Image.Image."
            )
        encoded_images.append(encoded_img)
    return {"image": encoded_images}


def extract_mm_items(request_data: dict) -> list[dict]:
    """
    Return *all* image/audio items that appear anywhere in `messages`.

    Each returned dict looks like:
        { "type": "image_url", "image_url": {...} }
    """
    items: list[dict] = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") in {"image_url", "audio_url", "input_audio"}:
                items.append(item)
    return items


def metrics_enabled(req: GenerationRequest, key: str) -> bool:
    # Check if metrics collection is enabled for a specific key in the request.

    req_enable_metrics = getattr(req, "enable_metrics", None)
    return isinstance(req_enable_metrics, dict) and bool(
        req_enable_metrics.get(key, False)
    )


def cal_exec_time(start: float) -> float:
    """Calculate elapsed time in seconds since a given start timestamp.

    Args:
        start: The start time, typically obtained from time.perf_counter().

    Returns:
        The elapsed time in seconds as a floating-point number.
    """
    return time.perf_counter() - start


def extract_mm_items(request_data: dict) -> list[dict]:
    """
    Return *all* image/audio items that appear anywhere in `messages`.

    Each returned dict looks like:
        { "type": "image_url", "image_url": {...} }
    """
    items: list[dict] = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") in {"image_url", "audio_url", "input_audio"}:
                items.append(item)
    return items
