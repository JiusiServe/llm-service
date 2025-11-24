# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project


def prefill_atten_workload(prompt_len: int, cached_len: int) -> float:
    top = cached_len + 1
    bottom = prompt_len
    height = prompt_len - cached_len
    return (top + bottom) * height / 2


def decode_atten_workload(prefill_len: int, decode_len: int, done_len: int) -> float:
    return (decode_len - done_len) * prefill_len\
           + ((decode_len + done_len + 1) * (decode_len - done_len)) / 2


def estimate_decode_len(predicted_len, responded_len, extend_rate) -> float:
    return max(predicted_len, responded_len * (1 + extend_rate))
