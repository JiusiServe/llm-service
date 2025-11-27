# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import random
from typing import Tuple, List

from ..common import Stage
from ..endpoint import Endpoint
from ..router.router import Router
from ..task import Task
from ..task_route import TaskRoute


class RandomRouter(Router):

    def __init__(self):
        super().__init__()

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        return Stage.ENCODE, Stage.PREFILL, Stage.DECODE, Stage.PREFILL_THEN_DECODE

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        endpoint_index = random.randint(0, len(endpoints) - 1)
        return self._create_nonworkload_route(task, endpoints[endpoint_index])

    def batch_route(self, tasks: List[Task], endpoints: List[Endpoint]) -> List[TaskRoute]:
        return [self.route(task, endpoints) for task in tasks]
