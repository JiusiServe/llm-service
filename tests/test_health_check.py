# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from unittest.mock import AsyncMock, patch

import pytest

try:
    from llm_service.apis.vllm.proxy import Proxy
    from llm_service.protocol.protocol import ServerType
except ImportError:
    pytest.skip(
        "vllm dependencies not available for integration test",
        allow_module_level=True,
    )


class TestIndividualHealthChecks:
    """Test suite for individual health check functionality."""

    @pytest.fixture
    def mock_proxy(self):
        """Create a mock Proxy instance with minimal setup."""
        with patch("llm_service.apis.vllm.proxy.zmq.asyncio.Context"):
            proxy = Proxy(
                proxy_addr="/tmp/test_proxy.sock",
                encode_addr_list=[
                    "/tmp/test_encode_0.sock",
                    "/tmp/test_encode_1.sock",
                ],
                pd_addr_list=[
                    "/tmp/test_pd_0.sock",
                    "/tmp/test_pd_1.sock",
                ],
                model_name="test-model",
                enable_health_monitor=False,
            )
            # Mock the check_health method
            proxy.check_health = AsyncMock()
            yield proxy
            proxy.shutdown()

    @pytest.mark.asyncio
    async def test_check_encode_instance_health_valid(self, mock_proxy):
        """Test checking health of a valid encode instance."""
        mock_proxy.check_health.return_value = True

        result = await mock_proxy.check_encode_instance_health(0)

        assert result is True
        mock_proxy.check_health.assert_called_once_with(ServerType.E_INSTANCE, 0)

    @pytest.mark.asyncio
    async def test_check_encode_instance_health_invalid_id(self, mock_proxy):
        """Test checking health with invalid encode instance ID."""
        with pytest.raises(ValueError, match="Invalid encode instance ID: 5"):
            await mock_proxy.check_encode_instance_health(5)

        with pytest.raises(ValueError, match="Invalid encode instance ID: -1"):
            await mock_proxy.check_encode_instance_health(-1)

    @pytest.mark.asyncio
    async def test_check_pd_instance_health_valid(self, mock_proxy):
        """Test checking health of a valid PD instance."""
        mock_proxy.check_health.return_value = True

        result = await mock_proxy.check_pd_instance_health(1)

        assert result is True
        mock_proxy.check_health.assert_called_once_with(ServerType.PD_INSTANCE, 1)

    @pytest.mark.asyncio
    async def test_check_pd_instance_health_invalid_id(self, mock_proxy):
        """Test checking health with invalid PD instance ID."""
        with pytest.raises(ValueError, match="Invalid PD instance ID: 10"):
            await mock_proxy.check_pd_instance_health(10)

        with pytest.raises(ValueError, match="Invalid PD instance ID: -5"):
            await mock_proxy.check_pd_instance_health(-5)

    @pytest.mark.asyncio
    async def test_check_all_instances_health_all_healthy(self, mock_proxy):
        """Test checking health of all instances when all are healthy."""
        mock_proxy.check_health.return_value = True

        result = await mock_proxy.check_all_instances_health()

        assert result == {
            "encode": {0: True, 1: True},
            "pd": {0: True, 1: True},
        }
        # Should have been called for all instances
        assert mock_proxy.check_health.call_count == 4

    @pytest.mark.asyncio
    async def test_check_all_instances_health_some_unhealthy(self, mock_proxy):
        """Test checking health when some instances are unhealthy."""

        # Mock different health states for different instances
        def check_health_side_effect(server_type, instance_id):
            if server_type == ServerType.E_INSTANCE and instance_id == 0:
                return False
            if server_type == ServerType.PD_INSTANCE and instance_id == 1:
                return False
            return True

        mock_proxy.check_health.side_effect = check_health_side_effect

        result = await mock_proxy.check_all_instances_health()

        assert result == {
            "encode": {0: False, 1: True},
            "pd": {0: True, 1: False},
        }

    @pytest.mark.asyncio
    async def test_check_all_instances_health_with_exceptions(self, mock_proxy):
        """Test checking health when some checks raise exceptions."""

        # Mock exceptions for some instances
        def check_health_side_effect(server_type, instance_id):
            if server_type == ServerType.E_INSTANCE and instance_id == 0:
                raise RuntimeError("Connection failed")
            if server_type == ServerType.PD_INSTANCE and instance_id == 1:
                raise RuntimeError("Timeout")
            return True

        mock_proxy.check_health.side_effect = check_health_side_effect

        result = await mock_proxy.check_all_instances_health()

        # Exceptions should be treated as unhealthy
        assert result == {
            "encode": {0: False, 1: True},
            "pd": {0: True, 1: False},
        }

    @pytest.mark.asyncio
    async def test_check_encode_instance_health_unhealthy(self, mock_proxy):
        """Test checking health of an unhealthy encode instance."""
        mock_proxy.check_health.return_value = False

        result = await mock_proxy.check_encode_instance_health(0)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_pd_instance_health_unhealthy(self, mock_proxy):
        """Test checking health of an unhealthy PD instance."""
        mock_proxy.check_health.return_value = False

        result = await mock_proxy.check_pd_instance_health(0)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_encode_instance_health_raises_exception(self, mock_proxy):
        """Test that exceptions from check_health are propagated."""
        mock_proxy.check_health.side_effect = RuntimeError("Health check failed")

        with pytest.raises(RuntimeError, match="Health check failed"):
            await mock_proxy.check_encode_instance_health(0)

    @pytest.mark.asyncio
    async def test_check_pd_instance_health_raises_exception(self, mock_proxy):
        """Test that exceptions from check_health are propagated."""
        mock_proxy.check_health.side_effect = RuntimeError("Health check failed")

        with pytest.raises(RuntimeError, match="Health check failed"):
            await mock_proxy.check_pd_instance_health(0)

    @pytest.mark.asyncio
    async def test_check_all_instances_health_empty_lists(self):
        """Test checking health when no instances are configured."""
        with patch("llm_service.apis.vllm.proxy.zmq.asyncio.Context"):
            proxy = Proxy(
                proxy_addr="/tmp/test_proxy.sock",
                encode_addr_list=[],
                pd_addr_list=[],
                model_name="test-model",
                enable_health_monitor=False,
            )
            proxy.check_health = AsyncMock()

            try:
                result = await proxy.check_all_instances_health()

                assert result == {"encode": {}, "pd": {}}
                # Should not call check_health when no instances exist
                proxy.check_health.assert_not_called()
            finally:
                proxy.shutdown()
