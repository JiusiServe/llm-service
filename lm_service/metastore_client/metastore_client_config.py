# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

"""
Metastore Client Configuration

This module defines the configuration class for the metastore client,
providing settings for connecting to and authenticating with the metastore service.
"""

from typing import Optional, Dict, Any, get_origin, get_args, Union
from dataclasses import dataclass, fields

from vllm.config.utils import config

# Import logger at the end to avoid circular imports
from lm_service.logger_utils import init_logger

logger = init_logger(__name__)


@config
@dataclass
class MetastoreClientConfig:
    """
    Configuration class for the metastore client.

    This class defines all necessary parameters to establish a connection
    to the metastore service, including connection details, authentication,
    and SSL settings.
    """

    metastore_client: Optional[str] = None
    """
    The name of the metastore client to use.
    """

    metastore_client_module_path: Optional[str] = None
    """
    The module path of the metastore client to use.
    """

    metastore_service_type: Optional[str] = None
    """
    The type of the metastore service to use.
    """
    # Host address of the metastore service
    metastore_host: Optional[str] = None

    # Port number of the metastore service
    metastore_port: Optional[int] = None

    # Password for authenticating with the metastore service
    metastore_password: str = ""

    # Prefix to use for all keys stored in the metastore
    metastore_key_prefix: str = "lm_service"

    # Database index to use in the metastore
    metastore_db: int = 0

    # Whether to use SSL for the connection
    metastore_ssl: bool = False

    # Path to the SSL certificate file (if SSL is enabled)
    metastore_ssl_certfile: Optional[str] = None

    # Path to the SSL key file (if SSL is enabled)
    metastore_ssl_keyfile: Optional[str] = None

    # Path to the SSL CA certificate file (if SSL is enabled)
    metastore_ssl_cafile: Optional[str] = None


def json_to_metastore_config(
    json_data: Optional[Dict[str, Any]],
) -> MetastoreClientConfig:
    """
    Convert JSON data to MetastoreClientConfig object.

    Args:
        json_data: JSON string or dictionary containing metastore configuration

    Returns:
        MetastoreClientConfig: Configured metastore client config object

    Example:
        ```python
        # From JSON string
        config = json_to_metastore_config('{"metastore_host": "redis.example.com", "metastore_port": 6379}')

        # From dictionary
        config = json_to_metastore_config({"metastore_host": "redis.example.com", "metastore_port": 6379})
        ```
    """
    if json_data is None:
        return MetastoreClientConfig()
    config_dict = json_data.copy()
    # Get all fields from the dataclass
    config_fields = {f.name: f.type for f in fields(MetastoreClientConfig)}

    # Validate and convert fields to correct types
    validated_config: Dict[Any, Any] = {}
    for field_name, field_value in config_dict.items():
        if field_name in config_fields:
            # Handle type conversion based on field type
            field_type = config_fields[field_name]
            # Get the actual type for conversion
            actual_type = None
            # Check for Optional or Union types using get_origin and get_args
            if isinstance(field_type, type) or hasattr(
                field_type, "__origin__"
            ):
                origin = get_origin(field_type)
                if origin is not None:
                    args = get_args(field_type)
                    # Handle Optional[Type] which is Union[Type, None]
                    if (
                        origin is Union
                        and len(args) == 2
                        and args[1] is type(None)
                    ):  # noqa
                        actual_type = args[0]
                    # Handle Union types if needed in the future
                else:
                    # Not a parameterized generic type
                    actual_type = field_type
            else:
                # If field_type is not a type object (e.g. a string), use it as is
                actual_type = field_type
            # Convert based on whether it's optional and the actual type
            if field_value is None:
                validated_config[field_name] = None
            else:
                # Determine the type to convert to
                target_type = (
                    actual_type if actual_type is not None else field_type
                )
                try:
                    # Convert value to the target type
                    if target_type is bool and isinstance(field_value, str):
                        # Handle string boolean values
                        validated_config[field_name] = field_value.lower() in (
                            "true",
                            "1",
                            "yes",
                            "y",
                        )
                    elif target_type in (int, float, str, bool) and callable(
                        target_type
                    ):
                        # Convert to basic types only if target_type is callable
                        validated_config[field_name] = target_type(field_value)
                    else:
                        # Use the value as is for complex types
                        validated_config[field_name] = field_value
                except (ValueError, TypeError):
                    logger.warning(
                        f"Failed to convert {field_name} to {target_type}, using original value"
                    )
                    validated_config[field_name] = field_value

    # Create and return the config object
    return MetastoreClientConfig(**validated_config)
