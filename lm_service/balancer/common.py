# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from enum import Enum


class Stage(Enum):
    ENCODE = "ENCODE"
    PREFILL = "PREFILL"
    DECODE = "DECODE"
    PREFILL_THEN_DECODE = "PREFILL_THEN_DECODE"
