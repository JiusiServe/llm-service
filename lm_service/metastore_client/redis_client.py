# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import time
from typing import Optional, Any
import asyncio

import redis
import redis.asyncio as redis_async
import zmq
import zmq.asyncio

from lm_service.protocol.protocol import ServerType
from lm_service.logger_utils import init_logger
import lm_service.envs as lm_service_envs
from lm_service.metastore_client.metastore_client import MetastoreClientBase
from lm_service.metastore_client.metastore_client_config import (
    MetastoreClientConfig,
)


logger = init_logger(__name__)


class RedisMetastoreClient(MetastoreClientBase):
    """
    Redis client class providing both synchronous and asynchronous
    Redis operation interfaces
    """

    def __init__(
        self,
        metastore_client_config: Optional[MetastoreClientConfig] = None,
        node_info: str = "",
        engine_type: Optional[int] = None,
        to_proxy: Optional[dict[str, zmq.asyncio.Socket]] = None,
        to_encode_sockets: Optional[dict[str, zmq.asyncio.Socket]] = None,
        to_pd_sockets: Optional[dict[str, zmq.asyncio.Socket]] = None,
        to_p_sockets: Optional[dict[str, zmq.asyncio.Socket]] = None,
        to_d_sockets: Optional[dict[str, zmq.asyncio.Socket]] = None,
    ):
        """
        Initialize Redis client

        Args:
            redis_host: Redis server host address
            redis_port: Redis server port
            db: Redis database index
            node_info: Node information string identifier
            engine_type: Type of the engine instance
            to_proxy: Dictionary of ZMQ sockets for proxy communication
            to_encode_sockets: Dictionary of ZMQ sockets for encoding service communication
            to_pd_sockets: Dictionary of ZMQ sockets for parameter distribution service
            to_p_sockets: Dictionary of ZMQ sockets for parameter service
            to_d_sockets: Dictionary of ZMQ sockets for data service
        """

        super().__init__(
            metastore_client_config,
            node_info,
            engine_type,
            to_proxy,
            to_encode_sockets,
            to_pd_sockets,
            to_p_sockets,
            to_d_sockets,
        )

        def _get_config_value(env_var, config_attr):
            return env_var or getattr(
                metastore_client_config, config_attr, None
            )

        self.address = _get_config_value(
            lm_service_envs.LM_SERVICE_REDIS_ADDRESS, "metastore_address"
        )
        self.socket_timeout = _get_config_value(
            lm_service_envs.LM_SERVICE_SOCKET_TIMEOUT,
            "metastore_socket_timeout",
        )
        self.redis_client = None  # Synchronous client
        self.async_redis_client = None  # Asynchronous client
        self.ctx = zmq.asyncio.Context()

        self.node_key = (
            f"{lm_service_envs.LM_SERVICE_REDIS_KEY_PREFIX}_{self.engine_type}"
        )
        self.server_key = (
            f"{lm_service_envs.LM_SERVICE_REDIS_KEY_PREFIX}_SERVERTYPE"
        )
        self._initialize_clients()
        if self.redis_client is None or self.async_redis_client is None:
            raise RuntimeError("Redis client initialization failed")
        self.async_task: list[asyncio.Task] = []
        # Start a task to save metadata asynchronously without blocking initialization
        self.async_task.append(
            asyncio.create_task(
                self.save_metadata_async(self.node_key, self.node_info, "0")
            )
        )
        logger.info(
            f"Node {self.node_info} registered to Redis key {self.node_key}"
        )

        interval = lm_service_envs.LM_SERVICE_REDIS_INTERVAL
        if engine_type == ServerType.PROXY.value:
            self.async_task.append(
                asyncio.create_task(
                    self.async_update_socket(
                        ServerType.E_INSTANCE.value, interval
                    )
                )
            )
            self.is_pd_merged = self._get_deploy_form()
            logger.info(f"Deploy form is E-PD: {self.is_pd_merged}")
            if self.is_pd_merged:
                self.async_task.append(
                    asyncio.create_task(
                        self.async_update_socket(
                            ServerType.PD_INSTANCE.value, interval
                        )
                    )
                )
            else:
                self.async_task.append(
                    asyncio.create_task(
                        self.async_update_socket(
                            ServerType.P_INSTANCE.value, interval
                        )
                    )
                )
                self.async_task.append(
                    asyncio.create_task(
                        self.async_update_socket(
                            ServerType.D_INSTANCE.value, interval
                        )
                    )
                )
        else:
            if self.engine_type != ServerType.E_INSTANCE.value:
                self.async_task.append(
                    asyncio.create_task(
                        self.save_metadata_async(
                            self.server_key, self.node_key, "0"
                        )
                    )
                )
            self.async_task.append(
                asyncio.create_task(
                    self.async_update_socket(ServerType.PROXY.value, interval)
                )
            )

    def _initialize_clients(self):
        """
        Initialize Redis client connections
        """
        try:
            self.redis_client = redis.Redis.from_url(
                self.address,
                socket_timeout=self.socket_timeout,
                decode_responses=True,
            )
            self.async_redis_client = redis_async.Redis.from_url(
                self.address,
                socket_timeout=self.socket_timeout,
                decode_responses=True,
            )
            self.redis_client.ping()  # Test connection
            # We'll test the async connection later in an async context
            logger.info("Redis clients initialized successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Redis client: {str(e)}")

    @property
    def is_pd_merge(self):
        return self.is_pd_merged

    def _get_deploy_form(self) -> bool:
        """
        Check if the deploy form is E-PD

        Returns:
            bool: True if E-PD, False if E-P-D
        """
        try:
            if not self.redis_client:
                logger.error("Synchronous Redis client not initialized")
                return False
            retry_times = lm_service_envs.LM_SERVICE_STARTUP_WAIT_TIME
            pd_node_key = f"{lm_service_envs.LM_SERVICE_REDIS_KEY_PREFIX}_{ServerType.PD_INSTANCE.value}"
            p_node_key = f"{lm_service_envs.LM_SERVICE_REDIS_KEY_PREFIX}_{ServerType.P_INSTANCE.value}"
            d_node_key = f"{lm_service_envs.LM_SERVICE_REDIS_KEY_PREFIX}_{ServerType.D_INSTANCE.value}"
            for _ in range(retry_times):
                deploy_form = self.get_metadata(self.server_key)
                if pd_node_key in deploy_form.keys():
                    return True
                elif (
                    p_node_key in deploy_form.keys()
                    and d_node_key in deploy_form.keys()
                ):
                    return False
                time.sleep(1)
            logger.warning(
                f"Waiting for {pd_node_key}, {p_node_key}, {d_node_key} timeout, "
                f"Use the default PD merge deployment mode"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to get deploy form from Redis: {str(e)}")
            return False

    def save_metadata(self, key: str, field: str, value: Any) -> bool:
        """
        Synchronously set Redis key-value pair

        Args:
            key: Key name
            field: Field name
            value: Value (will be converted to string)

        Returns:
            bool: Whether the setting was successful
        """
        try:
            if not self.redis_client:
                logger.error("Synchronous Redis client not initialized")
                return False
            # Ensure value is a string
            value_str = str(value)
            self.redis_client.hset(key, field, value_str)
            self.redis_client.expire(
                key, lm_service_envs.LM_SERVICE_REDIS_KEY_TTL
            )
            logger.debug(f"Redis key set successfully: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to set Redis key {key}: {str(e)}")
            return False

    async def save_metadata_async(
        self, key: str, field: str, value: Any
    ) -> Optional[bool]:
        """
        Asynchronously set Redis key-value pair

        Args:
            key: Key name
            field: Field name
            value: Value (will be converted to string)

        Returns:
            bool: Whether the setting was successful
        """
        try:
            if not self.async_redis_client:
                logger.error("Asynchronous Redis client not initialized")
                return False
            # Ensure value is a string
            value_str = str(value)
            await self.async_redis_client.hset(key, field, value_str)
            await self.async_redis_client.expire(
                key, lm_service_envs.LM_SERVICE_REDIS_KEY_TTL
            )
            logger.debug(f"Redis key set successfully (async): {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to set Redis key {key} (async): {str(e)}")
            return False

    def get_metadata(self, key: str) -> Optional[Any]:
        """
        Synchronously get Redis key value

        Args:
            key: Key name

        Returns:
            Optional[Any]: Retrieved value, None if not exists or error occurs
        """
        try:
            if not self.redis_client:
                logger.error("Synchronous Redis client not initialized")
                return None

            value = self.redis_client.hgetall(key)
            logger.debug(f"Redis key retrieved: {key}, value: {value}")
            return value
        except Exception as e:
            logger.error(f"Failed to get Redis key {key}: {str(e)}")
            return None

    async def get_metadata_async(self, key: str) -> Optional[Any]:
        """
        Asynchronously get Redis key value

        Args:
            key: Key name

        Returns:
            Optional[Any]: Retrieved value, None if not exists or error occurs
        """
        try:
            if not self.async_redis_client:
                logger.error("Asynchronous Redis client not initialized")
                return None

            value = await self.async_redis_client.hgetall(key)
            logger.debug(f"Redis key retrieved (async): {key}, value: {value}")
            return value
        except Exception as e:
            logger.error(f"Failed to get Redis key {key} (async): {str(e)}")
            return None

    def delete_metadata(self, key: str, field: str) -> Optional[bool]:
        """
        Synchronously delete Redis key field

        Args:
            key: Key name
            field: Field name

        Returns:
            Optional[bool]: Whether the deletion was successful
        """
        try:
            if not self.redis_client:
                logger.error("Synchronous Redis client not initialized")
                return False

            self.redis_client.hdel(key, field)
            logger.debug(f"Redis key field deleted: {key}, field: {field}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Redis key field {key}: {str(e)}")
            return False

    async def delete_metadata_async(
        self, key: str, field: str
    ) -> Optional[bool]:
        """
        Asynchronously delete Redis key field

        Args:
            key: Key name
            field: Field name

        Returns:
            Optional[bool]: Whether the deletion was successful
        """
        try:
            if not self.async_redis_client:
                logger.error("Asynchronous Redis client not initialized")
                return False

            await self.async_redis_client.hdel(key, field)
            logger.debug(
                f"Redis key field deleted (async): {key}, field: {field}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to delete Redis key field {key} (async): {str(e)}"
            )
            return False

    def close(self):
        """
        Close Redis connections
        """
        try:
            # Cancel all async tasks
            if hasattr(self, "async_task") and self.async_task:
                for task in self.async_task:
                    if not task.done():
                        task.cancel()
                logger.info(f"Cancelled {len(self.async_task)} async tasks")
                self.async_task.clear()

            if self.redis_client:
                self.delete_metadata(self.node_key, self.node_info)
                self.redis_client.close()
                logger.info("Synchronous Redis connection closed")

            # For async client, we should close it in an async context
            # This is just a placeholder; the actual closing should be done in an async method
            if self.async_redis_client:
                logger.info(
                    "Asynchronous Redis client needs to be closed in an async context"
                )

        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")

    async def async_close(self):
        """
        Asynchronously close Redis connections
        """
        try:
            # Cancel all async tasks
            if hasattr(self, "async_task") and self.async_task:
                for task in self.async_task:
                    if not task.done():
                        task.cancel()
                # Wait for all tasks to be cancelled
                if self.async_task:
                    await asyncio.gather(
                        *self.async_task, return_exceptions=True
                    )
                logger.info(f"Cancelled {len(self.async_task)} async tasks")
                self.async_task.clear()

            if self.async_redis_client:
                # Delete node info asynchronously
                await self.delete_metadata_async(self.node_key, self.node_info)
                await self.async_redis_client.close()
                logger.info("Asynchronous Redis connection closed")

            # Close synchronous client
            if self.redis_client:
                self.redis_client.close()
                logger.info("Synchronous Redis connection closed")

        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")

    def connect_to_server(
        self, current_servers, established_socket_dict
    ) -> None:
        # Check for servers that need to be connected
        for server_address in current_servers:
            # Check if this server is already connected
            if server_address not in established_socket_dict:
                logger.info(f"Establishing ZMQ connection to {server_address}")
                # Create a new async ZMQ socket (matching socket_dict type)
                new_socket = self.ctx.socket(zmq.constants.PUSH)
                new_socket.setsockopt(zmq.LINGER, 0)  # Non-blocking close
                try:
                    new_socket.connect(server_address)
                    logger.info(f"Successfully connected to {server_address}")
                    established_socket_dict[server_address] = new_socket
                except Exception as conn_error:
                    logger.error(
                        f"Failed to connect to {server_address}: {str(conn_error)}"
                    )
                    # Close the socket if connection failed
                    new_socket.close()
                    return

    def update_proxy_sockets(self):
        node_key = f"{lm_service_envs.LM_SERVICE_REDIS_KEY_PREFIX}_{ServerType.PROXY.value}"
        servers_dict = self.get_metadata(node_key)
        if not servers_dict:
            return

        current_servers = servers_dict.keys()
        self.connect_to_server(current_servers, self.to_proxy)

    async def async_update_proxy_sockets(self):
        """
        Asynchronously update proxy sockets
        """
        node_key = f"{lm_service_envs.LM_SERVICE_REDIS_KEY_PREFIX}_{ServerType.PROXY.value}"
        servers_dict = await self.get_metadata_async(node_key)
        if not servers_dict:
            return

        current_servers = servers_dict.keys()
        self.connect_to_server(current_servers, self.to_proxy)

    async def async_update_socket(self, engine_type: int, interval: int):
        """
        Asynchronously update socket connections for a given engine type

        Args:
            engine_type: The type of server (proxy, encode, or pd)
            interval: Check interval in seconds
        """
        try:
            while True:
                if self.async_redis_client is None:
                    logger.error("Redis client not initialized")
                    return
                node_key = f"{lm_service_envs.LM_SERVICE_REDIS_KEY_PREFIX}_{engine_type}"
                servers_dict = self.get_metadata(node_key)
                if servers_dict:
                    logger.debug(
                        f"Retrieved servers from Redis: {servers_dict}"
                    )

                    # Parse the servers_list string into a list of server info
                    current_servers = servers_dict.keys()
                    if engine_type == ServerType.PROXY.value:
                        self.connect_to_server(current_servers, self.to_proxy)
                    elif engine_type == ServerType.E_INSTANCE.value:
                        self.connect_to_server(
                            current_servers, self.to_encode_sockets
                        )
                    elif engine_type == ServerType.PD_INSTANCE.value:
                        self.connect_to_server(
                            current_servers, self.to_pd_sockets
                        )
                    elif engine_type == ServerType.P_INSTANCE.value:
                        self.connect_to_server(
                            current_servers, self.to_p_sockets
                        )
                    elif engine_type == ServerType.D_INSTANCE.value:
                        self.connect_to_server(
                            current_servers, self.to_d_sockets
                        )

                # Wait for next check interval
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Socket update task cancelled")
            # Close all sockets on cancellation
            for socket in (
                list(self.to_encode_sockets.values())
                + list(self.to_pd_sockets.values())
                + list(self.to_p_sockets.values())
                + list(self.to_d_sockets.values())
                + list(self.to_proxy.values())
            ):
                socket.close()
