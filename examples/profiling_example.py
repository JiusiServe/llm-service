# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

"""
Example demonstrating the use of start_profile and stop_profile APIs.

This example shows how to use the profiling functionality with the Proxy
class to profile inference requests across disaggregated workers.

Usage:
    python examples/profiling_example.py \\
        --proxy-addr /tmp/proxy.sock \\
        --encode-addr-list /tmp/encode1.sock /tmp/encode2.sock \\
        --pd-addr-list /tmp/pd1.sock /tmp/pd2.sock \\
        --model-name meta-llama/Meta-Llama-3-8B

Before running this example, ensure:
1. The disaggregated workers are running with the specified addresses
2. The VLLM_TORCH_PROFILER_DIR environment variable is set to the
   directory where you want to save profiling traces
3. The workers have been started with appropriate model configuration

The profiling trace files will be written to the directory specified by
VLLM_TORCH_PROFILER_DIR and can be visualized using tools like Perfetto:
https://ui.perfetto.dev/
"""

import argparse
import asyncio
import os

from llm_service.apis.vllm.proxy import Proxy
from vllm import SamplingParams


async def main():
    parser = argparse.ArgumentParser(
        description="Profile inference with disaggregated vLLM workers"
    )
    parser.add_argument("--proxy-addr", required=True, help="Proxy address")
    parser.add_argument(
        "--encode-addr-list",
        required=True,
        nargs="+",
        help="List of encode addresses",
    )
    parser.add_argument(
        "--pd-addr-list",
        required=True,
        nargs="+",
        help="List of pd addresses",
    )
    parser.add_argument("--model-name", required=True, help="Model name")
    args = parser.parse_args()

    # Ensure profiling directory is set
    if "VLLM_TORCH_PROFILER_DIR" not in os.environ:
        profile_dir = "./vllm_profile"
        os.environ["VLLM_TORCH_PROFILER_DIR"] = profile_dir
        print(f"Setting VLLM_TORCH_PROFILER_DIR to {profile_dir}")

    # Initialize the proxy
    proxy = Proxy(
        proxy_addr=args.proxy_addr,
        encode_addr_list=args.encode_addr_list,
        pd_addr_list=args.pd_addr_list,
        model_name=args.model_name,
    )

    try:
        # Define test prompts
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "The future of AI is",
        ]

        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        # Start profiling
        print("Starting profiling...")
        await proxy.start_profile()
        print("Profiling started successfully!")

        # Generate outputs while profiling
        print(f"\nGenerating {len(prompts)} prompts...")
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i + 1}/{len(prompts)}: {prompt}")
            async for output in proxy.generate(
                prompt=prompt, sampling_params=sampling_params
            ):
                if output.finished:
                    print(f"  Generated: {output.outputs[0].text[:50]}...")

        # Stop profiling
        print("\nStopping profiling...")
        await proxy.stop_profile()
        print("Profiling stopped successfully!")

        profile_dir = os.environ["VLLM_TORCH_PROFILER_DIR"]
        print(f"\nProfiling traces have been written to: {profile_dir}")
        print("You can visualize them at https://ui.perfetto.dev/")

    finally:
        proxy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
