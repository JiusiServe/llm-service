# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from dataclasses import dataclass
from typing import List

from .common import Stage


@dataclass
class Task:
    request_id: str

    @property
    def stage(self) -> Stage:
        raise NotImplementedError


@dataclass
class EncodeTask(Task):
    @property
    def stage(self) -> Stage:
        raise Stage.ENCODE

    def estimate_workload(self) -> float:
        raise NotImplementedError


@dataclass
class ViTEncodeTask(EncodeTask):
    num_patches: int

    def estimate_workload(self) -> float:
        plus_one = self.num_patches + 1
        return float(plus_one * plus_one)


@dataclass
class PrefillTask(Task):
    prompt_tokens: List[int]

    @property
    def stage(self) -> Stage:
        return Stage.PREFILL


@dataclass
class DecodeTask(Task):
    num_prompt_tokens: int
    predicted_decode_len: int

    @property
    def stage(self) -> Stage:
        return Stage.DECODE


@dataclass
class PrefillThenDecodeTask(Task):
    prompt_tokens: List[int]
    predicted_decode_len: int

    @property
    def stage(self) -> Stage:
        return Stage.PREFILL_THEN_DECODE
