# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
import argparse
import asyncio
import uuid
import json

import numpy as np
from PIL import Image

from vllm import SamplingParams
import vllm.envs as envs
from lm_service.apis.vllm.proxy import Proxy
import lm_service.envs as lm_service_envs

parser = argparse.ArgumentParser()
parser.add_argument("--proxy-addr", required=False, help="Proxy address")
parser.add_argument(
    "--encode-addr-list",
    required=False,
    nargs="+",
    help="List of encode addresses",
)
parser.add_argument(
    "--pd-addr-list",
    required=False,
    nargs="+",
    help="List of pd addresses",
)
parser.add_argument(
    "--p-addr-list",
    required=False,
    nargs="+",
    help="List of prefill addresses",
)
parser.add_argument(
    "--d-addr-list",
    required=False,
    nargs="+",
    help="List of decode addresses",
)
parser.add_argument(
    "--transfer-protocol",
    type=str,
    default="ipc",
    choices=["ipc", "tcp"],
    help="ZMQ transfer protocol, whether ZMQ uses IPC or TCP connection",
)
parser.add_argument(
    "--metastore-client-config",
    type=json.loads,
    default=None,
    help="Enable metastore client config.",
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


async def main():
    # new proxy
    p = Proxy(
        proxy_addr=args.proxy_addr,
        encode_addr_list=args.encode_addr_list,
        pd_addr_list=args.pd_addr_list,
        p_addr_list=args.p_addr_list,
        d_addr_list=args.d_addr_list,
        model_name=args.model_name,
        enable_health_monitor=False,
        transfer_protocol=args.transfer_protocol,
        metastore_client_config=args.metastore_client_config,
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
            asyncio.create_task(p.log_metrics())
        # test for exit_instance
        pd_num = len(args.pd_addr_list)
        exit_tasks = []
        for i in range(pd_num):
            exit_task = asyncio.create_task(
                asyncio.wait_for(
                    p.exit_instance(addr=args.pd_addr_list[i]),
                    timeout=lm_service_envs.LM_SERVICE_WORKER_EXIT_TIMEOUT,
                )
            )
            exit_tasks.append(exit_task)
        if exit_tasks:
            await asyncio.gather(*exit_tasks)
    finally:
        p.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
