# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import time
from dataclasses import dataclass
from typing import List, Optional

from .common import Stage
from .task_handle import TaskHandle, TaskHandleFactory
from .task_route import TaskRoute


@dataclass
class EndpointConfig:
    endpoint_id: str = ""
    stage: Stage = Stage.PREFILL
    cache_instance_id: str = ""
    is_dynamic_pd: bool = False


class EndpointListener:
    def on_task_submit(self, route: TaskRoute, handle: TaskHandle):
        pass

    def on_task_ended(self, handle: TaskHandle):
        pass

    def on_stage_changed(self, endpoint: "Endpoint", old_stage: Stage):
        pass


class Endpoint:
    def __init__(self, config: EndpointConfig):
        self.config: EndpointConfig = config
        self._open_tasks: List[TaskHandle] = []
        self._listener: Optional[EndpointListener] = None
        self._stage = self.config.stage

        if self.config.is_dynamic_pd and self._stage not in (Stage.PREFILL, Stage.DECODE):
            raise ValueError(f"Stage is not PREFILL nor DECODE in dynamic P/D endpoint")

    @property
    def id(self) -> str:
        return self.config.endpoint_id

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def is_dynamic_pd(self) -> bool:
        return self.config.is_dynamic_pd

    def set_listener(self, listener: Optional[EndpointListener]):
        self._listener = listener

    def queue_length(self) -> int:
        return len(self._open_tasks)

    def queue_workload(self) -> float:
        workload = 0
        for handle in self._open_tasks:
            workload += handle.todo_workload()
        return workload

    def set_stage(self, stage: Stage):
        if not self.config.is_dynamic_pd:
            raise RuntimeError(f"Tried to switch stage of non dynamic P/D endpoint")
        if stage not in (Stage.PREFILL, Stage.DECODE):
            raise ValueError(f"Stage is not PREFILL nor DECODE")
        if self._stage != stage:
            old_stage = self._stage
            self._stage = stage
            if self._listener:
                self._listener.on_stage_changed(self, old_stage)

    def on_task_submit(self, route: TaskRoute) -> TaskHandle:
        if self.stage != route.stage:
            raise ValueError("Stage not matched")
        handle = TaskHandleFactory.create(route, time.time())
        self._open_tasks.append(handle)
        if self._listener:
            self._listener.on_task_submit(route, handle)
        return handle

    def on_task_ended(self, handle: TaskHandle):
        self._open_tasks.remove(handle)
        if self._listener:
            self._listener.on_task_ended(handle)
