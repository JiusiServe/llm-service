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
from lm_service.service_discovery import HealthCheckServiceDiscovery
from lm_service.stats_loggers import MetricsReporter
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

from lm_service.instance_cluster import InstanceCluster, SERVER_PARAMS_MAP

logger = init_logger(__name__)

ROUTER_MAP = {
    "RandomRouter": RandomRouter,
    "RoundRobinRouter": RoundRobinRouter,
    "LeastInFlightRouter": LeastInFlightRouter,
}


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
        # Validate input parameters for some components
        self._check_type("enable_health_monitor", enable_health_monitor, bool)
        self._check_positive("health_check_interval", health_check_interval)
        self._check_positive("health_threshold", health_threshold)
        self._check_subclass("router", router, RoutingInterface)

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
        self.metastore_client: Optional[MetastoreClientBase] = None
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

        for server_type in SERVER_PARAMS_MAP:
            if self.is_pd_merged and server_type == ServerType.D_INSTANCE:
                continue
            addr_param_name = str(
                SERVER_PARAMS_MAP[server_type]["addr_list_name"]
            )
            addr_list = [
                f"{self.transfer_protocol}://{addr}"
                for addr in (init_params.get(addr_param_name) or [])
            ]
            if addr_list:
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
        for server_type in SERVER_PARAMS_MAP:
            if self.is_pd_merged and server_type == ServerType.P_INSTANCE:
                continue
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

    async def log_metrics(self) -> None:
        for server_type in self.instance_clusters:
            cluster = self.instance_clusters[server_type]
            await cluster.get_metrics()

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
        await cluster.process_request(request, q)

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

            # Step 1 : Encode the multimodal data if any
            if _has_mm_data(prompt):
                request.multi_modal_data = _encode_mm_data(
                    prompt["multi_modal_data"]
                )
                await self._process_request(ServerType.E_INSTANCE, request, q)

            # Step 2 : Maybe Prefill
            if not self.is_pd_merged:
                await self._process_request(ServerType.P_INSTANCE, request, q)

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
            socket = await self._get_socket_and_server_types_from_addr(
                addr, server_type
            )
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
