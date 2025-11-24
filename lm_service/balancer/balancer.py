# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
from dataclasses import dataclass
from typing import List, Dict, Optional

from .task_handle import TaskHandle
from .common import Stage
from .connector.kv_connector import KvConnector
from .dynamic_pd import DynamicPd
from .endpoint import Endpoint, EndpointListener
from .endpoint_tracker import EndpointTrackerListener, EndpointTracker
from .router.router import Router
from .task import Task
from .task_route import TaskRoute


class BalancerConfig:

    @dataclass
    class ServiceLevelObj:
        p_quantile: float = 0.99
        ttft: float = 1
        tpot: float = 0.25

    @dataclass
    class DynamicPd:
        update_on_requests: int = 100
        min_update_time: float = 10

    def __init__(self):
        self.service_level_obj: BalancerConfig.ServiceLevelObj = None
        self.dynamic_pd: DynamicPd = BalancerConfig.DynamicPd()


class Balancer(EndpointTrackerListener, EndpointListener):

    def __init__(self,
                 config: BalancerConfig,
                 tracker: EndpointTracker,
                 routers: Dict[Stage, Router],
                 kv_connector: Optional[KvConnector] = None):
        self.config = config
        self._tracker = tracker
        self._tracker.add_listener(self)
        self._kv_connector = kv_connector
        self._routers = routers
        for stage, router in self._routers.items():
            if stage not in router.for_stages:
                raise ValueError(f"{router.__class__.__name__} is not for stage:{stage}")
            router.on_registered(self)
        self._dynamic_pd = DynamicPd(self)

        for endpoint in self._tracker.get_up_endpoints():
            endpoint.set_listener(self)

    @property
    def kv_connector(self) -> Optional[KvConnector]:
        return self._kv_connector

    @property
    def dynamic_pd(self) -> DynamicPd:
        return self._dynamic_pd

    def get_candidates(self, task) -> List[Endpoint]:
        return self._tracker.get_up_endpoints(stages=(task.stage,))

    def get_up_endpoints(self) -> List[Endpoint]:
        return self._tracker.get_up_endpoints()

    def route(self, task: Task, candidates: List[Endpoint] = None) -> TaskRoute:
        router = self._routers.get(task.stage)
        if router is None:
            raise ValueError(f"Stage {task.stage} task is not supported by routers")
        if not candidates:
            candidates = self.get_candidates(task)
        if not candidates:
            raise RuntimeError("No candidate")
        return router.route(task, candidates)

    def on_endpoints_changed(self, new_ups: List[Endpoint], new_downs: List[Endpoint]):
        for endpoint in new_downs:
            endpoint.set_listener(None)
        for endpoint in new_ups:
            endpoint.set_listener(self)

    def on_task_ended(self, handle: TaskHandle):
        self._dynamic_pd.on_task_ended(handle)
