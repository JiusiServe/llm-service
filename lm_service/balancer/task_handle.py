# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import time

from .common import Stage
from .task_route import TaskRoute
from .workload import estimate_decode_len, decode_atten_workload


class TaskHandle:

    def __init__(self, route: "TaskRoute", submit_time: float):
        self.route: "TaskRoute" = route
        self.submit_time: float = submit_time
        self.end_time: float = -1
        self.responded_len: int = 0
        self.error: Exception = None

    @property
    def request_id(self) -> str:
        return self.route.request_id

    @property
    def stage(self) -> Stage:
        return self.route.stage

    @property
    def endpoint(self) -> "Endpoint":
        return self.route.endpoint

    @property
    def is_ended(self):
        return self.end_time != -1

    def todo_workload(self) -> float:
        if self.is_ended:
            return 0
        return self.route.workload

    def on_respond(self, chunk_len: int):
        self.responded_len += chunk_len

    def on_finished(self):
        if self.end_time != -1:
            raise RuntimeError("Task already finished")
        self.end_time = time.time()
        self.endpoint.on_task_ended(self)

    def on_error(self, error: Exception):
        self.error = error
        if self.end_time == -1:
            self.end_time = time.time()
            self.endpoint.on_task_ended(self)


class EncodeHandle(TaskHandle):

    def __init__(self, route: "EncodeRoute", submit_time: float):
        super().__init__(route, submit_time)


class PrefillHandle(TaskHandle):

    def __init__(self, route: "PrefillRoute", submit_time: float):
        super().__init__(route, submit_time)
        self.first_token_time: float = -1
        self.ttft: float = -1

    def on_respond(self, chunk_len: int):
        super().on_respond(chunk_len)
        self._update_ttft()

    def on_finished(self):
        super().on_finished()
        self._update_ttft()

    def _update_ttft(self):
        if self.first_token_time == -1:
            self.first_token_time = time.time()
            self.ttft = self.first_token_time - self.submit_time


class DecodeHandle(TaskHandle):

    def __init__(self, route: "DecodeRoute", submit_time: float):
        super().__init__(route, submit_time)
        self.tpot: float = -1

    def todo_workload(self) -> float:
        if self.is_ended:
            return 0
        if self.route.predicted_decode_len > 0:
            decode_len = \
                estimate_decode_len(self.route.predicted_decode_len,
                                    self.responded_len,
                                    self.route.len_extend_rate)
            workload = \
                decode_atten_workload(self.route.num_prompt_tokens,
                                      decode_len,
                                      self.responded_len)
            return max(workload, 0)
        return -1

    def on_finished(self):
        super().on_finished()
        elapsed = self.end_time - self.submit_time
        if elapsed > 0:
            self.tpot = self.responded_len / elapsed


class PrefillThenDecodeHandle(TaskHandle):

    def __init__(self, route: "PrefillThenDecodeRoute", submit_time: float):
        super().__init__(route, submit_time)
        self.first_token_time: float = -1
        self.ttft: float = -1
        self.tpot: float = -1

    def todo_workload(self) -> float:
        if self.is_ended:
            return 0
        workload = 0
        if self.first_token_time == -1:
            workload += self.route.prefill_workload
        decode_len = \
            estimate_decode_len(self.route.predicted_decode_len,
                                self.responded_len, self.route.len_extend_rate)
        workload += \
            decode_atten_workload(self.route.num_prompt_tokens,
                                  decode_len,
                                  self.responded_len)
        return max(workload, 0)

    def on_respond(self, chunk_len: int):
        super().on_respond(chunk_len)
        if self.first_token_time == -1:
            self.first_token_time = time.time()
            self.ttft = self.first_token_time - self.submit_time

    def on_finished(self):
        super().on_finished()
        elapsed = self.end_time - self.submit_time
        if elapsed > 0:
            self.tpot = self.responded_len / elapsed


class TaskHandleFactory:
    _handle_classes = {
        Stage.ENCODE: EncodeHandle,
        Stage.PREFILL: PrefillHandle,
        Stage.DECODE: DecodeHandle,
        Stage.PREFILL_THEN_DECODE: PrefillThenDecodeHandle,
    }

    @classmethod
    def create(cls, route: "Route", submit_time: float) -> TaskHandle:
        handle_cls = cls._handle_classes.get(route.stage)
        if handle_cls is None:
            raise ValueError(f"Unsupported stage: {route.stage}")
        return handle_cls(route, submit_time)
