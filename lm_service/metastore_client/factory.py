# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import importlib
from typing import Callable
from lm_service.metastore_client.metastore_client_config import (
    MetastoreClientConfig,
)
from lm_service.metastore_client.metastore_client import MetastoreClientBase
import lm_service.envs as lm_service_envs

from lm_service.logger_utils import init_logger

logger = init_logger(__name__)


class MetastoreClientFactory:
    _registry: dict[str, Callable[[], type[MetastoreClientBase]]] = {}

    @classmethod
    def register_metastore_client(
        cls, name: str, module_path: str, class_name: str
    ) -> None:
        """Register a metastore client with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(
                f"Metastore client '{name}' is already registered."
            )

        def loader() -> type[MetastoreClientBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_metastore_client_class(
        cls, metastore_client_config: "MetastoreClientConfig"
    ) -> type[MetastoreClientBase]:
        """Get the metastore client class by name."""
        metastore_client_name = (
            metastore_client_config.metastore_client
            or lm_service_envs.LM_SERVICE_METASTORE_CLIENT
        )
        if metastore_client_name is None:
            raise ValueError("Metastore client must not be None")
        elif metastore_client_name in cls._registry:
            metastore_client_cls = cls._registry[metastore_client_name]()
        else:
            metastore_client_module_path = (
                metastore_client_config.metastore_client_module_path
            )
            if metastore_client_module_path is None:
                raise ValueError(
                    f"Unsupported metastore client type: {metastore_client_name}"
                )
            metastore_client_module = importlib.import_module(
                metastore_client_module_path
            )
            metastore_client_cls = getattr(
                metastore_client_module, metastore_client_name
            )
        return metastore_client_cls

    @classmethod
    def create_metastore_client(
        cls, config: "MetastoreClientConfig", **kwargs
    ) -> MetastoreClientBase:
        metastore_client_config = config
        if metastore_client_config is None:
            raise ValueError(
                "metastore_client_config must be set to create a metastore client"
            )
        metastore_client_cls = cls.get_metastore_client_class(
            metastore_client_config
        )
        logger.info(
            "Creating metastore client with name: %s and engine_id: %s",
            metastore_client_cls.__name__,
            metastore_client_config.metastore_client,
        )

        return metastore_client_cls(config, **kwargs)


MetastoreClientFactory.register_metastore_client(
    "RedisMetastoreClient",
    "lm_service.metastore_client.redis_client",
    "RedisMetastoreClient",
)
