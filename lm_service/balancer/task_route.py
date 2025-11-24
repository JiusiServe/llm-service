# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from dataclasses import dataclass

from .common import Stage


@dataclass
class TaskRoute:
    request_id: str
    endpoint: "Endpoint"
    workload: float

    @property
    def stage(self) -> Stage:
        raise NotImplementedError

    def on_submit(self) -> "TaskHandle":
        return self.endpoint.on_task_submit(self)


@dataclass
class EncodeRoute(TaskRoute):
    @property
    def stage(self) -> Stage:
        return Stage.ENCODE


@dataclass
class PrefillRoute(TaskRoute):
    num_prompt_tokens: int
    num_cached_tokens: int

    @property
    def stage(self) -> Stage:
        return Stage.PREFILL


@dataclass
class DecodeRoute(TaskRoute):
    num_prompt_tokens: int
    predicted_decode_len: int
    len_extend_rate: float

    @property
    def stage(self) -> Stage:
        return Stage.DECODE


@dataclass
class PrefillThenDecodeRoute(TaskRoute):
    num_prompt_tokens: int
    num_cached_tokens: int
    prefill_workload: float
    predicted_decode_len: int
    len_extend_rate: float

    @property
    def stage(self) -> Stage:
        return Stage.PREFILL_THEN_DECODE
