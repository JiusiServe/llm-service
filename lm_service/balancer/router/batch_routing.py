import numpy as np


class BatchRouteOptimizer:

    def optimize(self, task_workloads, queue_workloads):
        raise NotImplementedError


class BatchRouteLocalSearch(BatchRouteOptimizer):

    def __init__(self, max_itr=20, eps=1e-9, rng=None):
        self.max_itr = max_itr
        self.eps = eps
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def optimize(self, task_workloads, queue_workloads):
        assign, worker_workloads = self._ini_assign(task_workloads, queue_workloads)
        num_workers = task_workloads.shape[0]
        num_tasks = task_workloads.shape[1]

        objective = np.sum(worker_workloads ** 2)
        improved = True
        itr = 0

        while itr < self.max_itr and improved:
            itr += 1
            improved = False
            task_order = self.rng.permutation(num_tasks)
            worker_order = self.rng.permutation(num_workers)
            for task in task_order:
                cur_worker = assign[task]
                cur_cost = task_workloads[cur_worker, task]
                for worker in worker_order:
                    if worker == cur_worker:
                        continue
                    new_cost = task_workloads[worker, task]
                    delta = (
                        (worker_workloads[worker] + new_cost) ** 2 - worker_workloads[worker] ** 2 +
                        (worker_workloads[cur_worker] - cur_cost) ** 2 - worker_workloads[cur_worker] ** 2
                    )
                    if delta < -self.eps:
                        worker_workloads[cur_worker] -= cur_cost
                        worker_workloads[worker] += new_cost
                        assign[task] = worker
                        objective += delta
                        improved = True
                        break
        return assign, worker_workloads

    @staticmethod
    def _ini_assign(task_workloads, queue_workloads):
        num_workers = task_workloads.shape[0]
        num_tasks = task_workloads.shape[1]
        task_order = [(task, np.min(task_workloads[:, task])) for task in range(num_tasks)]
        task_order = sorted(task_order, key=lambda x: x[1], reverse=True)
        worker_workloads = queue_workloads.copy()
        assign = np.empty(num_tasks, dtype=np.int32)
        for task, _ in task_order:
            best_worker = -1
            min_new_workload = -1
            for worker in range(num_workers):
                new_workload = worker_workloads[worker] + task_workloads[worker, task]
                if best_worker == -1 or new_workload < min_new_workload:
                    best_worker = worker
                    min_new_workload = new_workload
            if best_worker == -1:
                raise RuntimeError("no best worker")
            assign[task] = best_worker
            worker_workloads[best_worker] += task_workloads[best_worker, task]
        return assign, worker_workloads
