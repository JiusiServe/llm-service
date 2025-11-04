# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

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
    "LM_SERVICE_REDIS_IP": lambda: os.getenv(
        "LM_SERVICE_REDIS_IP", "localhost"
    ),
    "LM_SERVICE_REDIS_PORT": lambda: int(
        os.getenv("LM_SERVICE_REDIS_PORT", "6379")
    ),
    "LM_SERVICE_REDIS_DB": lambda: int(os.getenv("LM_SERVICE_REDIS_DB", "0")),
    "LM_SERVICE_REDIS_PASSWORD": lambda: os.getenv(
        "LM_SERVICE_REDIS_PASSWORD", ""
    ),
    "LM_SERVICE_REDIS_KEY_PREFIX": lambda: os.getenv(
        "LM_SERVICE_REDIS_KEY_PREFIX", "lm_service"
    ),
    "LM_SERVICE_REDIS_INTERVAL": lambda: int(
        os.getenv("LM_SERVICE_REDIS_INTERVAL", "10")
    ),
    "LM_SERVICE_REDIS_KEY_TTL": lambda: int(
        os.getenv("LM_SERVICE_REDIS_KEY_TTL", "600")
    ),
    "LM_SERVICE_RPC_PORT": lambda: os.getenv("LM_SERVICE_RPC_PORT", None),
    "LM_SERVICE_METASTORE_CLIENT": lambda: os.getenv(
        "LM_SERVICE_METASTORE_CLIENT", None
    ),
    "LM_SERVICE_METASTORE_CLIENT_CONFIG": lambda: os.getenv(
        "LM_SERVICE_METASTORE_CLIENT_CONFIG", None
    ),
    "LM_SERVICE_STARTUP_WAIT_TIME": lambda: int(
        os.getenv("LM_SERVICE_STARTUP_WAIT_TIME", "120")
    ),
    "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": lambda: int(
        os.getenv("LM_SERVICE_REQUEST_TIMEOUT_SECONDS", 120)
    ),
}

# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
