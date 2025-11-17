# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
import argparse
import asyncio
import uuid

import numpy as np
from PIL import Image

from vllm import SamplingParams
from lm_service.apis.vllm.proxy import Proxy
import vllm.envs as envs
import lm_service.envs as lm_service_envs

PROXY_NUM = 1
PROXY_PORT_BASE = 38000
TRANSFER_PROTOCOL = "ipc"

parser = argparse.ArgumentParser()
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
parser.add_argument(
    "--transfer-protocol",
    type=str,
    default="ipc",
    choices=["ipc", "tcp"],
    help="ZMQ transfer protocol, whether ZMQ uses IPC or TCP connection",
)
parser.add_argument("--model-name", required=True, help="Model name")
parser.add_argument("--image-path", required=True, help="Path to the image")
args = parser.parse_args()


# prepare image
image = Image.open(args.image_path)
image_array = np.array(image)


async def run_single_request(i, prompt, image_array, sampling_params, p):
    outputs = p.generate(
        prompt={
            "prompt": prompt,
            "multi_modal_data": {"image": image_array},
        },
        sampling_params=sampling_params,
        request_id=str(uuid.uuid4()),
    )
    async for o in outputs:
        generated_text = o.outputs[0].text
        print(f"Request({i}) generated_text: {generated_text}", flush=True)


async def run_single_proxy(proxy_addr):
    # new proxy
    p = Proxy(
        proxy_addr=proxy_addr,
        encode_addr_list=args.encode_addr_list,
        pd_addr_list=args.pd_addr_list,
        model_name=args.model_name,
        enable_health_monitor=False,
        transfer_protocol=args.transfer_protocol,
    )
    try:
        # The current prompt format follows Qwen2.5-VL-3B-Instruct.
        # You may need to adjust it if using a different model.
        prompt = (
            "<|im_start|> system\n"
            "You are a helpful assistant.<|im_end|> \n"
            "<|im_start|> user\n"
            "<|vision_start|><|image_pad|><|vision_end|> \n"
            "What is the text in the illustration?<|im_end|> \n"
            "<|im_start|> assistant\n"
        )
        sampling_params = SamplingParams(max_tokens=128)

        tasks = [
            asyncio.create_task(
                run_single_request(i, prompt, image_array, sampling_params, p)
            )
            for i in range(10)
        ]
        await asyncio.gather(*tasks)
        if lm_service_envs.TIMECOUNT_ENABLED:
            # wait for logging
            await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
            await p.log_metrics()
    finally:
        p.shutdown()


async def main():
    if TRANSFER_PROTOCOL == "tcp":
        proxy_addr_list = [
            f"{args.proxy_addr}:{PROXY_PORT_BASE + i}" for i in range(PROXY_NUM)
        ]
    else:
        proxy_addr_list = [f"{args.proxy_addr}_{i}" for i in range(PROXY_NUM)]
    tasks = [run_single_proxy(proxy_addr) for proxy_addr in proxy_addr_list]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
