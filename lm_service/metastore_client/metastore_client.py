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
        server_type: Optional[int] = None,
        to_proxy: Optional[dict[str, zmq.asyncio.Socket]] = None,
        to_e_sockets: Optional[dict[str, zmq.asyncio.Socket]] = None,
        to_p_sockets: Optional[dict[str, zmq.asyncio.Socket]] = None,
        to_d_sockets: Optional[dict[str, zmq.asyncio.Socket]] = None,
        *args,
        **kwargs,
    ):
        self.metastore_client_config = metastore_client_config
        self.node_info = node_info
        self.server_type = server_type
        self.to_e_sockets: dict[str, zmq.asyncio.Socket] = (
            to_e_sockets if to_e_sockets is not None else {}
        )
        self.to_p_sockets: dict[str, zmq.asyncio.Socket] = (
            to_p_sockets if to_p_sockets is not None else {}
        )
        self.to_d_sockets: dict[str, zmq.asyncio.Socket] = (
            to_d_sockets if to_d_sockets is not None else {}
        )
        self.to_proxy: dict[str, zmq.asyncio.Socket] = (
            to_proxy if to_proxy is not None else {}
        )

    def launch_proxy_task(self):
        """
        Launch proxy task to report node info and update socket.
        """
        pass

    def close(self):
        """
        Close metastore client
        """
        pass

    def async_close(self):
        """
        Close metastore client asynchronously
        """
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

    def delete_metadata(self, key: str, field: str) -> Optional[bool]:
        """
        Delete metadata from metastore

        Args:
            key: Key name
            field: Field name

        Returns:
            Optional[bool]: True if success, False if error occurs
        """
        pass

    async def delete_metadata_async(
        self, key: str, field: str
    ) -> Optional[bool]:
        """
        Delete metadata from metastore asynchronously

        Args:
            key: Key name
            field: Field name

        Returns:
            Optional[bool]: True if success, False if error occurs
        """
        pass
