# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

from enum import Enum, auto
from typing import Any, Optional, Union

import msgspec
from vllm import SamplingParams
from vllm.outputs import RequestOutput

# NOTE FOR DEVELOPERS:
# DO NOT USE PICKLE FOR THESE CLASSES. IN A MULTI NODE
# SETUP WE WILL USE TCP. WE CANNOT USE PICKLE OTHERWISE
# WE RISK REMOTE CODE EXECUTION FROM UNSTRUSTED USERS.


class ServerType(Enum):
    E_INSTANCE = auto()
    PD_INSTANCE = auto()
    P_INSTANCE = auto()
    D_INSTANCE = auto()
    PROXY = auto()


class RequestType:
    GENERATION = b"\x00"
    ABORT = b"\x01"
    ENCODE = b"\x02"
    HEARTBEAT = b"\x03"
    METRICS = b"\x04"
    PREFILL = b"\x05"


class PDAbortRequest(msgspec.Struct):
    request_id: str


class ResponseType:
    GENERATION = b"\x00"
    FAILURE = b"\x01"
    ENCODE = b"\x02"
    HEARTBEAT = b"\x03"
    METRICS = b"\x04"
    PREFILL = b"\x05"


class GenerationResponse(msgspec.Struct):
    request_id: str
    text: str
    token_ids: list[int]
    prompt_token_ids: list[int]
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    # TODO: support full protocol.
    logprobs = None
    proxy_to_worker_time_end: Optional[float] = None

    @classmethod
    def from_request_output(
        self, request_output: RequestOutput
    ) -> "GenerationResponse":
        assert len(request_output.outputs) == 1, "Only support N=1 right now."
        out = request_output.outputs[0]
        return GenerationResponse(
            request_id=request_output.request_id,
            text=out.text,
            token_ids=out.token_ids,
            prompt_token_ids=request_output.prompt_token_ids,
            finish_reason=out.finish_reason,
            stop_reason=str(out.stop_reason),
        )


class GenerationRequest(msgspec.Struct):
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    proxy_addr: str
    multi_modal_data: Optional[dict[str, Any]] = None


class HeartbeatRequest(msgspec.Struct):
    request_id: str
    proxy_addr: str


class HeartbeatResponse(msgspec.Struct):
    request_id: str
    status: str = "OK"


class FailureResponse(msgspec.Struct):
    request_id: str
    error_message: str


class MetricsRequest(msgspec.Struct):
    request_id: str
    proxy_addr: str


class MetricsResponse(msgspec.Struct):
    request_id: str
    metrics: Optional[dict[str, dict[str, Union[str, int, float]]]]
