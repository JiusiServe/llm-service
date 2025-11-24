# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from typing import Tuple, List, Set

from ..common import Stage
from ..endpoint import Endpoint
from ..router.router import Router
from ..task import Task
from ..task_route import TaskRoute


class KvawareRouter(Router):

    def __init__(self):
        super().__init__()

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        return Stage.PREFILL, Stage.PREFILL_THEN_DECODE

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        if self._balancer.kv_connector is None:
            raise RuntimeError("KvConnector is None")
        try:
            hit_lens = self._query_cache_hit(task.prompt_tokens,
                                             set([ep.config.cache_instance_id for ep in endpoints]))
            endpoint = self._find_best_endpoint(endpoints, hit_lens)
            return self._create_nonworkload_route(task, endpoint)
        except ValueError:
            pass
        idx = self._route_by_queue_len(endpoints)
        return self._create_nonworkload_route(task, endpoints[idx])

    def _query_cache_hit(self, prompt_tokens: List[int], cache_instance_ids: Set[str]):
        hit_lens = self._balancer.kv_connector.query_hit_len(prompt_tokens, cache_instance_ids)
        if not hit_lens:
            raise ValueError
        return hit_lens

    @staticmethod
    def _find_best_endpoint(endpoints, hit_lens) -> Endpoint:
        max_hit_len = -1
        max_hit_instance = None
        for instance_id, hit_len in hit_lens.items():
            if max_hit_instance is None or hit_len > max_hit_len:
                max_hit_instance = instance_id
                max_hit_len = hit_len
        if max_hit_instance is None:
            raise ValueError
        for endpoint in endpoints:
            if endpoint.config.cache_instance_id == max_hit_instance:
                return endpoint
        raise ValueError
