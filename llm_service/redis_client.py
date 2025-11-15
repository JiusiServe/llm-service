# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
import redis
from typing import Optional, Any
import asyncio
import zmq

from llm_service.protocol.protocol import ServerType
from llm_service.logger_utils import init_logger
import llm_service.envs as llm_service_envs

logger = init_logger(__name__)


class RedisClient:
    """
    Redis client class providing both synchronous and asynchronous
    Redis operation interfaces
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        db: int = 0,
        node_info: str = "127.0.0.1_49999",
        engine_type: Optional[int] = None,
        to_proxy: dict[str, zmq.asyncio.Socket] = {},
        to_encode_sockets: dict[str, zmq.asyncio.Socket] = {},
        to_pd_sockets: dict[str, zmq.asyncio.Socket] = {},
        to_p_sockets: dict[str, zmq.asyncio.Socket] = {},
        to_d_sockets: dict[str, zmq.asyncio.Socket] = {},
        password: Optional[str] = None,
    ):
        """
        Initialize Redis client

        Args:
            redis_host: Redis server host address
            redis_port: Redis server port
            db: Redis database index
            node_info: Node information string identifier
            engine_type: Type of the engine instance
            to_proxy: List of sockets connected to proxy servers
            to_encode_sockets: List of sockets connected to encoding servers
            to_pd_sockets: List of sockets connected to PD servers
            password: Redis authentication password
        """
        self.host = llm_service_envs.REDIS_IP or redis_host
        self.port = llm_service_envs.REDIS_PORT or redis_port
        self.db = db
        self.password = llm_service_envs.REDIS_PASSWORD or password
        self.redis_client = None  # Synchronous client
        self.reporting_task = None  # Asynchronous task for node reporting
        self.update_socket_task = None  # Asynchronous task for socket updating
        self.established_sockets: dict[str, zmq.asyncio.Socket] = {}
        self.ctx = zmq.asyncio.Context()
        self.to_encode_sockets = to_encode_sockets
        self.to_pd_sockets = to_pd_sockets
        self.to_proxy = to_proxy

        self.node_info = node_info
        self.node_key = f"{llm_service_envs.REDIS_KEY_PREFIX}_{engine_type}"
        self._initialize_clients()
        if self.redis_client is None:
            raise RuntimeError("Redis client initialization failed")
        self.set_key(self.node_key, self.node_info, "0")
        logger.info(
            f"Node {self.node_info} registered to Redis key {self.node_key}"
        )

        interval = llm_service_envs.REDIS_INTERVAL
        if engine_type == ServerType.PROXY.value:
            self.update_socket_task = asyncio.create_task(
                self._update_socket(ServerType.E_INSTANCE.value, interval)
            )
            self.update_socket_task = asyncio.create_task(
                self._update_socket(ServerType.PD_INSTANCE.value, interval)
            )
        else:
            self.update_socket_task = asyncio.create_task(
                self._update_socket(ServerType.PROXY.value, interval)
            )

    def _initialize_clients(self):
        """
        Initialize Redis client connections
        """
        try:
            # Initialize synchronous Redis client
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
            self.redis_client.ping()  # Test connection

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Redis client: {str(e)}")

    def set_key(
        self, key: str, field: str, value: Any, expire: Optional[int] = None
    ) -> bool:
        """
        Synchronously set Redis key-value pair

        Args:
            key: Key name
            value: Value (will be converted to string)
            expire: Expiration time in seconds, optional

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
            logger.debug(f"Redis key set successfully: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to set Redis key {key}: {str(e)}")
            return False

    def get_key(self, key: str) -> Optional[str]:
        """
        Synchronously get Redis key value

        Args:
            key: Key name

        Returns:
            Optional[str]: Retrieved value, None if not exists or error occurs
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

    def close(self):
        """
        Close Redis connections
        """
        try:
            # Stop all async tasks
            if self.reporting_task and not self.reporting_task.done():
                self.reporting_task.cancel()
                logger.info("Node reporting task cancelled")

            if self.update_socket_task and not self.update_socket_task.done():
                self.update_socket_task.cancel()
                logger.info("Socket update task cancelled")

            if self.redis_client:
                self.redis_client.close()
                logger.info("Synchronous Redis connection closed")

        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")

    async def stop_node_reporting(self):
        """
        Stop the node reporting asynchronous task
        """
        if self.reporting_task and not self.reporting_task.done():
            self.reporting_task.cancel()
            try:
                await asyncio.wait_for(self.reporting_task, timeout=5)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for node reporting task to stop"
                )
            self.reporting_task = None
            logger.info("Node reporting task stopped")

        if self.update_socket_task and not self.update_socket_task.done():
            self.update_socket_task.cancel()
            try:
                await asyncio.wait_for(self.update_socket_task, timeout=5)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for socket update task to stop")
            self.update_socket_task = None
            logger.info("Socket update task stopped")

    async def _report_node_info(self, interval):
        """
        Asynchronous task to continuously report node information to Redis

        Args:
            node_port: The port of the current node
            interval: Reporting interval in seconds
        """
        try:
            while True:
                # Use async_set to store node info with expiration
                self.set_key(self.node_key, self.node_info, "0")
                logger.debug(f"Reported node info to Redis: {self.node_info}")
                # Wait for next reporting interval
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Node reporting task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in node reporting task: {str(e)}")

    def connect_to_server(self, current_servers, socket_dict):
        # Check for servers that need to be connected
        for server_address in current_servers:
            # Check if this server is already connected
            if server_address not in socket_dict:
                logger.info(f"Establishing ZMQ connection to {server_address}")
                # Create a new ZMQ socket
                socket = self.ctx.socket(zmq.constants.PUSH)
                socket.setsockopt(zmq.LINGER, 0)  # Non-blocking close
                try:
                    socket.connect(server_address)
                    logger.info(f"Successfully connected to {server_address}")
                    socket_dict[server_address] = socket
                except Exception as conn_error:
                    logger.error(
                        f"Failed to connect to {server_address}: {str(conn_error)}"
                    )
                    # Close the socket if connection failed
                    socket.close()

    def update_proxy_sockets(self):
        node_key = (
            f"{llm_service_envs.REDIS_KEY_PREFIX}_{ServerType.PROXY.value}"
        )
        servers_dict = self.get_key(node_key)
        if not servers_dict:
            return

        current_servers = servers_dict.keys()
        self.connect_to_server(current_servers, self.to_proxy)

    async def _update_socket(self, engine_type: int, interval: int):
        """
        Asynchronous task to continuously get keys from Redis and update sockets
        using ZeroMQ for connections

        Args:
            engine_type: The type of server (proxy, encode, or pd)
            interval: Check interval in seconds
        """
        try:
            while True:
                if self.redis_client is None:
                    logger.error("Redis client not initialized")
                    return
                node_key = f"{llm_service_envs.REDIS_KEY_PREFIX}_{engine_type}"
                servers_dict = self.get_key(node_key)
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

                # Wait for next check interval
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Socket update task cancelled")
            # Close all sockets on cancellation
            for socket in (
                list(self.to_encode_sockets.values())
                + list(self.to_pd_sockets.values())
                + list(self.to_proxy.values())
            ):
                socket.close()
