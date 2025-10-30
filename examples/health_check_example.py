# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
"""
Example demonstrating individual health check calls.

This example shows how to use the new health check methods to monitor
individual instances or all instances at once.
"""

import asyncio

from llm_service.apis.vllm.proxy import Proxy


async def main():
    # Create a proxy instance
    proxy = Proxy(
        proxy_addr="/tmp/proxy.sock",
        encode_addr_list=[
            "/tmp/encode_0.sock",
            "/tmp/encode_1.sock",
        ],
        pd_addr_list=[
            "/tmp/pd_0.sock",
            "/tmp/pd_1.sock",
        ],
        model_name="my-model",
        enable_health_monitor=True,  # Background monitor still runs
    )

    try:
        # Example 1: Check health of a specific encode instance
        print("Checking health of encode instance 0...")
        try:
            is_healthy = await proxy.check_encode_instance_health(0)
            print(f"Encode instance 0 is {'healthy' if is_healthy else 'unhealthy'}")
        except RuntimeError as e:
            print(f"Health check failed: {e}")

        # Example 2: Check health of a specific PD instance
        print("\nChecking health of PD instance 1...")
        try:
            is_healthy = await proxy.check_pd_instance_health(1)
            print(f"PD instance 1 is {'healthy' if is_healthy else 'unhealthy'}")
        except RuntimeError as e:
            print(f"Health check failed: {e}")

        # Example 3: Check health of all instances at once
        print("\nChecking health of all instances...")
        health_status = await proxy.check_all_instances_health()
        print("Health status of all instances:")
        print(f"  Encode instances: {health_status['encode']}")
        print(f"  PD instances: {health_status['pd']}")

        # Example 4: Error handling for invalid instance IDs
        print("\nTrying to check invalid instance ID...")
        try:
            await proxy.check_encode_instance_health(999)
        except ValueError as e:
            print(f"Caught expected error: {e}")

    finally:
        proxy.shutdown()


if __name__ == "__main__":
    # Note: This example requires actual encode and PD workers to be running
    # For demonstration purposes only
    print("Note: This example requires actual workers to be running.")
    print("It demonstrates the API usage, but will fail without workers.\n")
    asyncio.run(main())
