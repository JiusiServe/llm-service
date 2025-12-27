# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import os
from typing import Any, Callable, Optional, Type, TypeVar

from vllm.envs import env_with_choices

from lm_service.logger_utils import init_logger

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "t", "y", "yes", "on"}
T = TypeVar("T", int, float)

# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    # Enable timecount profiling
    "TIMECOUNT_ENABLED": lambda: os.getenv("TIMECOUNT_ENABLED", "0").lower()
    in _TRUE_VALUES,
    "TRANSFER_PROTOCOL": env_with_choices(
        "TRANSFER_PROTOCOL", None, ["tcp", "ipc"]
    ),
    "LM_SERVICE_E_INSTANCE_ROUTER": env_with_choices(
        "LM_SERVICE_E_INSTANCE_ROUTER",
        "RandomRouter",
        ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"],
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
    "LM_SERVICE_REDIS_ADDRESS": lambda: os.getenv(
        "LM_SERVICE_REDIS_ADDRESS", None
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
    "LM_SERVICE_SOCKET_TIMEOUT": lambda: int(
        os.getenv("LM_SERVICE_SOCKET_TIMEOUT", "60")
    ),
    "LM_SERVICE_HOST_IP": lambda: os.getenv("LM_SERVICE_HOST_IP", None),
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
    # Timeout in seconds for requests.
    "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": lambda: check_parameter_range(
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS", 120, min_value=0
    ),
    # Timeout in seconds for graceful worker shutdown and request cleanup.
    # Used to control how long a worker waits to finish processing before exiting.
    "LM_SERVICE_WORKER_GRACEFUL_EXIT_TIMEOUT_SEC": lambda: int(
        os.getenv("LM_SERVICE_WORKER_GRACEFUL_EXIT_TIMEOUT_SEC", 600)
    ),
}

# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def check_parameter_range(
    param_name: str,
    default_value: T,
    min_value: Optional[T] = None,
    max_value: Optional[T] = None,
    param_type: Type[T] = int,
) -> T:
    """
    check the environment variable value is within the specified range.

    Args:
        param_name:     environment variable name
        default_value:  default value
        min_value:      minimum value
        max_value:      maximum value
        param_type:     type, int/float, default int

    Returns:
        value after type and range validation
    """
    raw = os.getenv(param_name, None)
    try:
        value = param_type(raw) if raw is not None else default_value
    except (ValueError, TypeError):
        logger.warning(
            f"{param_name}: Unable to convert '{raw}' to {param_type.__name__}, "
            f"using default: {default_value}."
        )
        value = default_value

    if min_value is not None and value <= min_value:
        raise ValueError(f"{param_name}: {value} <= min_value {min_value}")

    if max_value is not None and value > max_value:
        raise ValueError(f"{param_name}: {value} > max_value {max_value}")

    return value
