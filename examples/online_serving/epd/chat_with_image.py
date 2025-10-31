# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
import argparse
import asyncio
import uuid

import numpy as np
from PIL import Image

from vllm import SamplingParams
from llm_service.apis.vllm.proxy import Proxy
import os

TIMECOUNT_ENABLED = os.getenv("TIMECOUNT_ENABLED",
                              "0") in ("1", "true", "True")

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
        model_name=args.model_name,
        enable_health_monitor=False,
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
        if TIMECOUNT_ENABLED:
            await p.pd_metrics_logger.get_metrics()
            await p.encoder_metrics_logger.get_metrics()
    finally:
        p.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
