# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import asyncio
import time
from collections import defaultdict
from abc import ABC, abstractmethod

import zmq
import zmq.asyncio

import zmq
import zmq.asyncio

from lm_service.protocol.protocol import ServerType
from lm_service.logger_utils import init_logger

logger = init_logger(__name__)


class ServiceDiscovery(ABC):
    @abstractmethod
    def get_health_endpoints(self) -> list[str]:
        """
        Retrieve a list of available instances for the given service name.

        Args:
            service_name (str): The name of the service to discover.
        Returns:
            list[int]: A list of available instance IDs.
        """
        pass

    @abstractmethod
    def get_unhealth_endpoints(self) -> list[str]:
        """
        Retrieve a list of available instances for the given service name.

        Args:
            service_name (str): The name of the service to discover.

        Returns:
            list[int]: A list of available instance IDs.
        """
        pass


class HealthCheckServiceDiscovery(ServiceDiscovery):
    def __init__(
        self,
        server_type: ServerType,
        instances: dict[str, zmq.asyncio.Socket],
        enable_health_monitor: bool,
        health_check_interval: float,
        health_threshold: int,
        health_check_func,
    ):
        self.server_type = server_type
        self._instances = instances
        self._instances_states: dict[str, bool] = defaultdict(lambda: True)
        self._cached_health_instances = [addr for addr in instances.keys()]
        self._cached_unhealth_instances: list[str] = []
        self.enable_health_monitor = enable_health_monitor
        self._health_check_interval = health_check_interval
        self._health_threshold = health_threshold
        self._success_count: dict[str, int] = defaultdict(lambda: 0)
        self._fail_count: dict[str, int] = defaultdict(lambda: 0)
        self._health_check_func = health_check_func
        self._health_monitor_handler = None

    def should_launch_health_monitor(self) -> bool:
        return (
            self.enable_health_monitor and self._health_monitor_handler is None
        )

    def launch_health_monitor(self):
        self._health_monitor_handler = asyncio.create_task(
            self.run_health_check_loop()
        )
        logger.info("Health monitor for %s launched.", self.server_type)

    def get_health_endpoints(self) -> list[str]:
        self._update_health_status()
        return self._cached_health_instances

    def get_unhealth_endpoints(self) -> list[str]:
        self._update_health_status()
        return self._cached_unhealth_instances

    async def run_health_check_loop(self):
        while True:
            start_time = time.monotonic()
            tasks = [
                asyncio.create_task(
                    asyncio.wait_for(
                        self._health_check_func(self.server_type, addr),
                        timeout=self._health_check_interval,
                    )
                )
                for addr in self._instances.keys()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for addr, result in zip(self._instances.keys(), results):
                if isinstance(result, bool) and result:
                    self._update_health_counts(addr, True)
                else:
                    self._update_health_counts(addr, False)
                    logger.warning(
                        "Health check for %s %s failed, reason is (%s).",
                        self.server_type,
                        addr,
                        "timeout"
                        if isinstance(result, asyncio.TimeoutError)
                        else result,
                    )

            self._update_health_status()

            elapsed = time.monotonic() - start_time
            sleep_time = max(0, self._health_check_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def _update_health_counts(self, addr: str, is_succ: bool):
        if is_succ:
            self._success_count[addr] = min(
                self._health_threshold, self._success_count.get(addr, 0) + 1
            )
            self._fail_count[addr] = 0
        else:
            self._fail_count[addr] = min(
                self._health_threshold, self._fail_count.get(addr, 0) + 1
            )
            self._success_count[addr] = 0

    def _update_health_status(self):
        for addr in self._instances.keys():
            if (
                self._instances_states[addr]
                and self._fail_count.get(addr, 0) >= self._health_threshold
            ):
                self._instances_states[addr] = False
                logger.info(
                    "Instance %s %s marked as unhealthy.",
                    self.server_type,
                    addr,
                )
            elif (
                not self._instances_states[addr]
                and self._success_count.get(addr, 0) >= self._health_threshold
            ):
                self._instances_states[addr] = True
                logger.info(
                    "Instance %s %s marked as healthy.", self.server_type, addr
                )

        self._cached_health_instances = [
            addr for addr, healthy in self._instances_states.items() if healthy
        ]
        self._cached_unhealth_instances = [
            addr
            for addr, healthy in self._instances_states.items()
            if not healthy
        ]
