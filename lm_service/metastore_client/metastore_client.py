# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

from abc import ABC
from typing import Optional, Any

import zmq
import zmq.asyncio

from lm_service.metastore_client.metastore_client_config import (
    MetastoreClientConfig,
)


class MetastoreClientBase(ABC):
    def __init__(
        self,
        metastore_client_config: Optional[MetastoreClientConfig] = None,
        node_info: str = "",
        engine_type: Optional[int] = None,
        to_proxy: dict[str, zmq.asyncio.Socket] = {},
        to_encode_sockets: dict[str, zmq.asyncio.Socket] = {},
        to_pd_sockets: dict[str, zmq.asyncio.Socket] = {},
        to_p_sockets: dict[str, zmq.asyncio.Socket] = {},
        to_d_sockets: dict[str, zmq.asyncio.Socket] = {},
        *args,
        **kwargs,
    ):
        pass

    def save_metadata(self, key: str, field: str, value: Any) -> Optional[bool]:
        """
        Save metadata to metastore

        Args:
            key: Key name
            field: Field name
            value: Value to save

        Returns:
            Optional[bool]: True if success, False if error occurs
        """
        pass

    @property
    def is_pd_merge(self) -> bool:
        """
        Check if metastore is merged with pd

        Returns:
            bool: True if merged, False if not
        """
        return True

    def get_metadata(self, key: str) -> Optional[Any]:
        """
        Get metadata from metastore

        Args:
            key: Key name

        Returns:
            Optional[str]: Retrieved value, None if not exists or error occurs
        """
        pass

    async def save_metadata_async(
        self, key: str, field: str, value: Any
    ) -> Optional[bool]:
        """
        Save metadata to metastore asynchronously

        Args:
            key: Key name
            field: Field name
            value: Value to save

        Returns:
            Optional[bool]: True if success, False if error occurs
        """
        pass

    async def get_metadata_async(self, key: str) -> Optional[Any]:
        """
        Get metadata from metastore asynchronously

        Args:
            key: Key name

        Returns:
            Optional[Any]: Retrieved value, None if not exists or error occurs
        """
        pass

    def delete_metadata(self, key: str) -> Optional[bool]:
        """
        Delete metadata from metastore

        Args:
            key: Key name

        Returns:
            Optional[bool]: True if success, False if error occurs
        """
        pass
