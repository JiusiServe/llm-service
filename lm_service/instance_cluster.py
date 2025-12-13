# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
from typing import Union
import asyncio
import time
import zmq
import zmq.asyncio
from lm_service.request_stats import RequestStatsMonitor
from lm_service.routing_logic import RoutingInterface
from lm_service.service_discovery import HealthCheckServiceDiscovery
from lm_service.stats_loggers import MetricsReporter
import msgspec
import lm_service.envs as lm_service_envs
from lm_service.protocol.protocol import (
    GenerationResponse,
    RequestType,
    ServerType,
)

SERVER_PARAMS_MAP = {
    ServerType.E_INSTANCE: {
        "addr_list_name": "encode_addr_list",
        "run_request_type": RequestType.ENCODE,
        "socket_list_name": "to_e_sockets",
    },
    ServerType.P_INSTANCE: {
        "addr_list_name": "p_addr_list",
        "run_request_type": RequestType.PREFILL,
        "socket_list_name": "to_p_sockets",
    },
    ServerType.D_INSTANCE: {
        "addr_list_name": "d_addr_list",
        "run_request_type": RequestType.GENERATION,
        "socket_list_name": "to_d_sockets",
    },
    ServerType.PD_INSTANCE: {
        "addr_list_name": "d_addr_list",
        "run_request_type": RequestType.GENERATION,
        "socket_list_name": "to_d_sockets",
    },
}


class InstanceCluster:
    """
    Encapsulates per-server-type runtime components.
    """

    def __init__(
        self,
        server_type: ServerType,
        sockets: dict[str, zmq.asyncio.Socket],
        service_discovery: HealthCheckServiceDiscovery,
        stats_monitor: RequestStatsMonitor,
        router: RoutingInterface,
        metrics_logger: MetricsReporter,
        socket_lock: asyncio.Lock,
    ):
        self.server_type = server_type
        self.sockets = sockets
        self.service_discovery = service_discovery
        self.stats_monitor = stats_monitor
        self.router = router
        self.metrics_logger = metrics_logger
        self.encoder = msgspec.msgpack.Encoder()
        self.socket_lock = socket_lock

    def _prepare_msg(self, request):
        if not self.sockets:
            raise RuntimeError(f"No available {self.server_type.name} workers.")

        # encode payload
        try:
            payload = self.encoder.encode(request)
        except Exception as e:
            raise RuntimeError("Failed to serialize GenerationRequest") from e

        msg = (SERVER_PARAMS_MAP[self.server_type]["run_request_type"], payload)

        return msg

    async def process_request_streaming_response(self, request, q):
        msg = self._prepare_msg(request)
        async with self.socket_lock:
            health_endpoints = self._get_health_endpoints()
            request_stats = self.stats_monitor.get_request_stats()
            addr = self._route_request(health_endpoints, request_stats)
            self.stats_monitor.on_new_request(
                addr, request_id=request.request_id
            )
            socket = self.sockets[addr]

        try:
            start_time = (
                time.perf_counter()
                if lm_service_envs.TIMECOUNT_ENABLED
                else None
            )
            await socket.send_multipart(msg, copy=False)
            finished = False
            while not finished:
                response = await self._await_with_timeout(request.request_id, q)
                if isinstance(response, Exception):
                    raise response
                self._record_proxy_to_instance_time(addr, response, start_time)
                finished = response.finish_reason is not None
                # mark instance latest successful response time
                self.service_discovery.update_latest_success(addr)
                yield response

        finally:
            self.stats_monitor.on_request_completed(
                addr, request_id=request.request_id
            )

    async def process_request(self, request, q):
        msg = self._prepare_msg(request)
        async with self.socket_lock:
            health_endpoints = self._get_health_endpoints()
            request_stats = self.stats_monitor.get_request_stats()
            addr = self._route_request(health_endpoints, request_stats)
            self.stats_monitor.on_new_request(
                addr, request_id=request.request_id
            )
            socket = self.sockets[addr]

        try:
            start_time = (
                time.perf_counter()
                if lm_service_envs.TIMECOUNT_ENABLED
                else None
            )
            await socket.send_multipart(msg, copy=False)
            response = await self._await_with_timeout(request.request_id, q)
            self._record_proxy_to_instance_time(addr, response, start_time)
            if isinstance(response, Exception):
                raise response
            else:
                # mark instance latest successful response time
                self.service_discovery.update_latest_success(addr)
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

    async def get_metrics(self):
        await self.metrics_logger.get_metrics()

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
