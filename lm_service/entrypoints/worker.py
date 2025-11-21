# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import signal
import json

import asyncio
import uvloop
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION
from lm_service.stats_loggers import DisaggWorkerStatsLogger
from lm_service.workers.vllm.disagg_worker import DisaggWorker
import lm_service.envs as lm_service_envs
from lm_service.logger_utils import init_logger

logger = init_logger(__name__)


async def run(args, engine: EngineClient):
    logger.info("Initializing disaggregated worker")

    # Signal handler used for graceful termination.
    # SystemExit exception is only raised once to allow this and worker
    # processes to terminate without error
    loop = asyncio.get_event_loop()
    exit_exiting = asyncio.Event()

    async def do_graceful_exit():
        if exit_exiting.is_set():
            return
        exit_exiting.set()
        logger.info("Shutdown requested by signal.")
        await worker._shutdown_handler("SIGTERM received")

    loop.add_signal_handler(
        signal.SIGTERM, lambda: asyncio.create_task(do_graceful_exit())
    )

    worker = DisaggWorker(
        engine=engine,
        address=args.worker_addr,
        proxy_addr=args.proxy_addr,
        transfer_protocol=args.transfer_protocol,
        metastore_client_config=args.metastore_client_config,
        ec_transfer_config=args.ec_transfer_config,
        kv_transfer_config=args.kv_transfer_config,
    )

    try:
        await worker.run_busy_loop()
    finally:
        worker.shutdown()


async def main(args) -> None:
    logger.info("Disaggregated Worker Server, vLLM ver. %s", VLLM_VERSION)
    logger.info("Args: %s", args)

    stat_loggers = None
    if lm_service_envs.TIMECOUNT_ENABLED:
        stat_loggers = [DisaggWorkerStatsLogger]
        logger.info("Time counting is enabled.")
    if getattr(args.ec_transfer_config, "ec_role", None) == "ec_producer":
        if getattr(args, "enable_prefix_caching", None) in (None, True):
            logger.warning(
                "Encoder doesn't support prefix caching, "
                "disable it in the config."
            )
            args.enable_prefix_caching = False

    engine_args = AsyncEngineArgs.from_cli_args(args)

    engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=stat_loggers)
    try:
        await run(args, engine)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--proxy-addr",
        required=False,
        default=None,
        nargs="+",
        help="List of proxy addresses",
    )
    parser.add_argument(
        "--worker-addr",
        required=False,
        default=None,
        type=str,
        help="The address of the worker.",
    )
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        help="Disable MQLLMEngine for AsyncLLMEngine.",
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
    AsyncEngineArgs.add_cli_args(parser)
    uvloop.run(main(parser.parse_args()))
