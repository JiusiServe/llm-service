# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from typing import Tuple, List

from ..common import Stage
from ..endpoint import Endpoint
from ..router.router import Router
from ..task import Task
from ..task_route import TaskRoute


class RoundRobinRouter(Router):

    def __init__(self):
        super().__init__()
        self._endpoint_index = -1

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        return Stage.ENCODE, Stage.PREFILL, Stage.DECODE, Stage.PREFILL_THEN_DECODE

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        self._endpoint_index = (self._endpoint_index + 1) % len(endpoints)
        return self._create_nonworkload_route(task, endpoints[self._endpoint_index])
