# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import math
from dataclasses import dataclass
from typing import Tuple, List, Set

from ..common import Stage
from ..endpoint import Endpoint
from ..router.router import Router
from ..task import Task, PrefillThenDecodeTask, PrefillTask
from ..task_route import TaskRoute, PrefillThenDecodeRoute, PrefillRoute
from ..workload import prefill_atten_workload, decode_atten_workload


class PrefillRouter(Router):

    @dataclass
    class _Workloads:
        total_workload: float = -1
        task_workload: float = -1
        prefill_workload: float = -1
        num_cached_tokens: int = 0

    def __init__(self, len_extend_rate: float = 0.2):
        super().__init__()
        self._len_extend_rate = len_extend_rate

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        return Stage.PREFILL, Stage.PREFILL_THEN_DECODE

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        try:
            hit_lens = self._query_cache_hit(task.prompt_tokens,
                                             set([ep.config.cache_instance_id for ep in endpoints]))
            endpoint, workloads = \
                self._find_best_endpoint(task, endpoints, hit_lens)
            return self._create_route(task, endpoint, workloads)
        except (ValueError, RuntimeError):
            pass
        idx = self._route_by_queue_len(endpoints)
        workloads = self._estimate_workloads(task, endpoints[idx], None, None)
        return self._create_route(task, endpoint, workloads)

    def _query_cache_hit(self, prompt_tokens: List[int], cache_instance_ids: Set[str]):
        if self._balancer.kv_connector is None:
            return {}
        return self._balancer.kv_connector.query_hit_len(prompt_tokens, cache_instance_ids)

    def _find_best_endpoint(self, task, endpoints, hit_lens):
        if hit_lens:
            max_hit_len = max(hit_lens.values())
        else:
            max_hit_len = 0

        endpoint_d = {}
        for endpoint in endpoints:
            endpoint_d[endpoint.config.cache_instance_id] = endpoint

        min_total_workload = math.inf
        picked_endpoint = None
        picked_workloads = None

        for cache_instance_id, hit_len in hit_lens.items():
            endpoint = endpoint_d[cache_instance_id]
            workloads = \
                self._estimate_workloads(task, endpoint, hit_len, max_hit_len)
            if picked_endpoint is None or workloads.total_workload <= min_total_workload:
                picked_endpoint = endpoint
                picked_workloads = workloads
                min_total_workload = workloads.total_workload

        for endpoint in endpoints:
            # for the endpoint IDs not in the hit_lens dict
            if endpoint.config.cache_instance_id in hit_lens:
                continue
            workloads = \
                self._estimate_workloads(task, endpoint, 0, max_hit_len)
            if picked_endpoint is None or workloads.total_workload <= min_total_workload:
                picked_endpoint = endpoint
                picked_workloads = workloads
                min_total_workload = workloads.total_workload

        if picked_endpoint is None:
            raise ValueError(f"no best endpoint was found")
        return picked_endpoint, picked_workloads

    def _estimate_workloads(self, task, endpoint, hit_len, max_hit_len):
        if self._balancer.kv_connector is None or not self._balancer.kv_connector.is_p2p_enabled:
            num_cached_tokens = hit_len
        else:
            num_cached_tokens = max(max_hit_len, 0)
        prompt_len = len(task.prompt_tokens)
        prefill_workload = prefill_atten_workload(prompt_len, num_cached_tokens)
        if isinstance(task, PrefillThenDecodeTask):
            decode_workload = decode_atten_workload(prompt_len, task.predicted_decode_len, 0)
            task_workload = prefill_workload + decode_workload
        else:
            task_workload = prefill_workload
        total_workload = endpoint.queue_workload() + task_workload
        return self._Workloads(total_workload=total_workload,
                               task_workload=task_workload,
                               prefill_workload=prefill_workload,
                               num_cached_tokens=num_cached_tokens)

    def _create_route(self, task, endpoint, workloads):
        if task.stage != endpoint.stage:
            raise RuntimeError(f"Task and endpoint stage not match")
        if isinstance(task, PrefillThenDecodeTask):
            return PrefillThenDecodeRoute(request_id=task.request_id,
                                          endpoint=endpoint,
                                          workload=workloads.task_workload,
                                          num_prompt_tokens=len(task.prompt_tokens),
                                          num_cached_tokens=workloads.num_cached_tokens,
                                          prefill_workload=workloads.prefill_workload,
                                          predicted_decode_len=task.predicted_decode_len,
                                          len_extend_rate=self._len_extend_rate)
        elif isinstance(task, PrefillTask):
            return PrefillRoute(request_id=task.request_id,
                                endpoint=endpoint,
                                workload=workloads.task_workload,
                                num_prompt_tokens=len(task.prompt_tokens),
                                num_cached_tokens=workloads.num_cached_tokens)
        raise RuntimeError(f"Unsupported task type: {task.__class__.__name__}")
