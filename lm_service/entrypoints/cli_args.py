# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
"""
This file contains the command line arguments for the vLLM's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import json

from lm_service.logger_utils import init_logger

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import FrontendArgs
from vllm.utils import FlexibleArgumentParser


logger = init_logger(__name__)


def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    """Create the CLI argument parser used by the OpenAI API server."""
    parser.add_argument(
        "--proxy-addr",
        required=False,
        help="Address for the proxy.",
    )
    parser.add_argument(
        "--encode-addr-list",
        required=False,
        nargs="+",
        help="List of addresses for the encoder.",
    )
    parser.add_argument(
        "--pd-addr-list",
        required=False,
        nargs="+",
        help="List of addresses for the prefill-decode.",
    )
    parser.add_argument(
        "--p-addr-list",
        required=False,
        nargs="+",
        help="List of addresses for the prefill.",
    )
    parser.add_argument(
        "--d-addr-list",
        required=False,
        nargs="+",
        help="List of addresses for the decode.",
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

    parser.add_argument(
        "model_tag",
        type=str,
        nargs="?",
        help="The model tag to serve (optional if specified in config)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode. See multi-node data parallel "
        "documentation for more details.",
    )
    parser.add_argument(
        "--api-server-count",
        "-asc",
        type=int,
        default=1,
        help="How many API server processes to run.",
    )
    parser.add_argument(
        "--config",
        help="Read CLI options from a config file. "
        "Must be a YAML with the following options: "
        "https://docs.vllm.ai/en/latest/configuration/serve_args.html",
    )
    parser = FrontendArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)

    return parser
