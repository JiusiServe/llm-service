# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from typing import Tuple, List

from ..common import Stage
from ..endpoint import Endpoint
from ..router.router import Router
from ..task import Task
from ..task_route import TaskRoute


class QueueLenRouter(Router):

    def __init__(self):
        super().__init__()

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        return Stage.ENCODE, Stage.PREFILL, Stage.DECODE, Stage.PREFILL_THEN_DECODE

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        endpoint_index = self._route_by_queue_len(endpoints)
        return self._create_nonworkload_route(task, endpoints[endpoint_index])

    def batch_route(self, tasks: List[Task], endpoints: List[Endpoint]) -> List[TaskRoute]:
        queue_lengths = [ep.queue_length() for ep in endpoints]
        routes = []
        for task in tasks:
            min_queue_len = -1
            min_queue_ep_i = -1
            for ep_i, in range(len(endpoints)):
                if min_queue_ep_i == -1 or queue_lengths[ep_i] < min_queue_len:
                    min_queue_len = queue_lengths[ep_i]
                    min_queue_ep_i = ep_i
            if min_queue_ep_i == -1:
                raise RuntimeError("Found no Endpoint")
            routes.append(self._create_nonworkload_route(task, endpoints[min_queue_ep_i]))
            queue_lengths[min_queue_ep_i] += 1
        return routes
