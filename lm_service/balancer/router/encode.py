from typing import Tuple, List
import numpy as np

from ..common import Stage
from ..task import Task
from ..task_route import TaskRoute, EncodeRoute
from ..endpoint import Endpoint
from ..router import Router


class EncodeRouter(Router):

    def __init__(self):
        super().__init__()

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        return Stage.ENCODE,

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        workload = task.estimate_workload()
        try:
            endpoint = self._find_best_endpoint(endpoints)
            return EncodeRoute(request_id=task.request_id,
                               endpoint=endpoint,
                               workload=workload)
        except ValueError:
            pass
        idx = self._route_by_queue_len(endpoints)
        return EncodeRoute(request_id=task.request_id,
                           endpoint=endpoints[idx],
                           workload=workload)

    def batch_route(self, tasks: List[Task], endpoints: List[Endpoint]) -> List[TaskRoute]:
        task_workloads = np.empty((len(endpoints), len(tasks)), dtype=np.float64)
        for task_i, task in enumerate(tasks):
            task_workloads[:, task_i] = task.estimate_workload()

        assign = self._optimize_batch_route(task_workloads, endpoints)

        routes = []
        for task_i, task in enumerate(tasks):
            endpoint_i = assign[task_i]
            routes.append(
                EncodeRoute(request_id=task.request_id,
                            endpoint=endpoints[endpoint_i],
                            workload=task_workloads[endpoint_i, task_i]))
        return routes

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
