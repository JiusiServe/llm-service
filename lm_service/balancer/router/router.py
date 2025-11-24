# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from typing import List, Tuple

from ..common import Stage
from ..endpoint import Endpoint
from ..task import Task, EncodeTask, PrefillTask, DecodeTask, PrefillThenDecodeTask
from ..task_route import TaskRoute, EncodeRoute, PrefillRoute, DecodeRoute, PrefillThenDecodeRoute


class Router:

    def __init__(self):
        self._balancer: "Balancer" = None

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        raise NotImplementedError

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        raise NotImplementedError

    def on_registered(self, balancer: "Balancer"):
        self._balancer = balancer

    @staticmethod
    def _create_nonworkload_route(task: Task, endpoint: Endpoint) -> TaskRoute:
        if isinstance(task, EncodeTask):
            return EncodeRoute(request_id=task.request_id,
                               endpoint=endpoint,
                               workload=-1)
        if isinstance(task, PrefillTask):
            return PrefillRoute(request_id=task.request_id,
                                endpoint=endpoint,
                                workload=-1,
                                num_prompt_tokens=len(task.prompt_tokens),
                                num_cached_tokens=-1)
        if isinstance(task, DecodeTask):
            return DecodeRoute(request_id=task.request_id,
                               endpoint=endpoint,
                               workload=-1,
                               predicted_decode_len=task.predicted_decode_len,
                               len_extend_rate=0)
        if isinstance(task, PrefillThenDecodeTask):
            return PrefillThenDecodeRoute(request_id=task.request_id,
                                          endpoint=endpoint,
                                          workload=-1,
                                          num_prompt_tokens=len(task.prompt_tokens),
                                          num_cached_tokens=-1,
                                          prefill_workload=-1,
                                          predicted_decode_len=task.predicted_decode_len,
                                          len_extend_rate=0)
        raise ValueError(f"Unsupported stage:{task.stage}")

    @staticmethod
    def _route_by_queue_len(endpoints: List[Endpoint]) -> int:
        min_queue_len = -1
        min_queue_ep_i = -1
        for i, endpoint in enumerate(endpoints):
            if min_queue_ep_i == -1 or endpoint.queue_length() < min_queue_len:
                min_queue_len = endpoint.queue_length()
                min_queue_ep_i = i
        if min_queue_ep_i == -1:
            raise RuntimeError("Found no Endpoint")
        return min_queue_ep_i
