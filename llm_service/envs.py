# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, Callable

from vllm.envs import env_with_choices

_TRUE_VALUES = {"1", "true", "t", "y", "yes", "on"}

# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    "TIMECOUNT_ENABLED": lambda: os.getenv("TIMECOUNT_ENABLED", "0").lower()
    in _TRUE_VALUES,
    "TRANSFER_PROTOCOL": env_with_choices(
        "TRANSFER_PROTOCOL", None, ["tcp", "ipc"]
    ),
    "LM_SERVICE_P_INSTANCE_ROUTER": env_with_choices(
        "LM_SERVICE_P_INSTANCE_ROUTER",
        "RandomRouter",
        ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"],
    ),
    "LM_SERVICE_D_INSTANCE_ROUTER": env_with_choices(
        "LM_SERVICE_D_INSTANCE_ROUTER",
        "RandomRouter",
        ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"],
    ),
    "LM_SERVICE_PD_INSTANCE_ROUTER": env_with_choices(
        "LM_SERVICE_PD_INSTANCE_ROUTER",
        "RandomRouter",
        ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"],
    ),
    "LM_SERVICE_E_INSTANCE_ROUTER": env_with_choices(
        "LM_SERVICE_E_INSTANCE_ROUTER",
        "RandomRouter",
        ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"],
    ),
    "REDIS_IP": lambda: os.getenv("REDIS_IP", "localhost"),
    "REDIS_PORT": lambda: int(os.getenv("REDIS_PORT", "6379")),
    "REDIS_PASSWORD": lambda: os.getenv("REDIS_PASSWORD", ""),
    "REDIS_KEY_PREFIX": lambda: os.getenv("REDIS_KEY_PREFIX", "llm_service"),
    "REDIS_INTERVAL": lambda: int(os.getenv("REDIS_INTERVAL", "10")),
    "NODE_PORT": lambda: os.getenv("NODE_PORT", None),
    "SERVER_TYPE": lambda: int(os.getenv("SERVER_TYPE", "0")),
    "AUTO_DISCOVERY_SERVICE": lambda: os.getenv(
        "AUTO_DISCOVERY_SERVICE", "0"
    ).lower()
    in _TRUE_VALUES,
    "DEPLOY_FORM": env_with_choices(
        "DEPLOY_FORM", "E-PD", ["MERGE", "E-PD", "EP-D", "E-P-D"]
    ),
}

# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
