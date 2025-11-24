from typing import Tuple, List

from ..common import Stage
from ..task import Task
from ..task_route import TaskRoute, DecodeRoute
from ..endpoint import Endpoint
from ..router import Router
from ..workload import decode_atten_workload


class DecodeRouter(Router):

    def __init__(self, len_extend_rate: float = 0.2):
        super().__init__()
        self._len_extend_rate = len_extend_rate

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        return Stage.DECODE,

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        num_prompt_tokens = task.num_prompt_tokens
        predicted_decode_len = task.predicted_decode_len
        if num_prompt_tokens <= 0:
            raise ValueError("Invalid num_prompt_tokens")
        if predicted_decode_len <= 0:
            raise ValueError("Invalid predicted_decode_len")
        workload = decode_atten_workload(num_prompt_tokens,
                                         predicted_decode_len ,
                                         num_prompt_tokens)
        try:
            endpoint = self._find_best_endpoint(endpoints)
            return DecodeRoute(request_id=task.request_id,
                               endpoint=endpoint,
                               workload=workload,
                               num_prompt_tokens=num_prompt_tokens,
                               predicted_decode_len=predicted_decode_len,
                               len_extend_rate=self._len_extend_rate)
        except ValueError:
            pass
        idx = self._route_by_queue_len(endpoints)
        return DecodeRoute(request_id=task.request_id,
                           endpoint=endpoints[idx],
                           workload=workload,
                           num_prompt_tokens=num_prompt_tokens,
                           predicted_decode_len=predicted_decode_len,
                           len_extend_rate=self._len_extend_rate)

    @staticmethod
    def _find_best_endpoint(endpoints):
        min_workload = -1
        best = None
        for endpoint in endpoints:
            # new task's workload are the same among endpoints, so don't care
            workload = endpoint.queue_workload()
            if best is None or workload < min_workload:
                best = endpoint
                min_workload = workload
        if best is None:
            raise RuntimeError("No best endpoint found")
        return best
