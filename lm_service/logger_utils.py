# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
import logging
from vllm.logger import init_logger as vllm_init_logger


def init_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Fork the log format from vllm
    """
    vllm_logger = logging.getLogger("vllm")

    logger = vllm_init_logger(name)
    logger.setLevel(level)
    logger.propagate = False

    # The current module name is llm_serivice,
    # and it cannot automatically inherit from vllm.
    logger.handlers.clear()
    for h in vllm_logger.handlers:
        logger.addHandler(h)

    return logger
