# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import asyncio
from collections import defaultdict
import threading
import time
from typing import ClassVar, Optional, Union

import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats
from lm_service.protocol.protocol import ServerType
from lm_service.logger_utils import init_logger

logger = init_logger(__name__)


class DisaggWorkerStatsLogger(StatLoggerBase):
    _LOCK: ClassVar[threading.Lock] = threading.Lock()
    # { engine_idx: { key: {"latest": float, "overall": float}, ... } }
    SNAPSHOTS_AVG: ClassVar[
        dict[int, dict[str, dict[str, Union[int, float]]]]
    ] = {}

    def __init__(self, vllm_config: VllmConfig, engine_idx: int = 0):
        self.EPD_STATS_KEYS = [
            "e2e_time_requests",
            "queue_time_requests",
            "prefill_time_requests",
            "mean_time_per_output_token_requests",
            "time_to_first_token",
        ]
        self.finished_request_attr = [
            "e2e_latency",
            "queued_time",
            "prefill_time",
            "mean_time_per_output_token",
        ]
        self.engine_index = engine_idx
        self.vllm_config = vllm_config
        self.last_scheduler_stats = SchedulerStats()
        self.last_prompt_throughput: float = 0.0
        self.last_generation_throughput: float = 0.0
        self.last_log_time = time.monotonic()
        self.num_prompt_tokens: int = 0
        self.num_generation_tokens: int = 0
        # [count_num, total_seconds]
        # init stats dict
        self.stats_dict = {
            key: {"latest": [0, 0.0], "overall": [0, 0.0]}
            for key in self.EPD_STATS_KEYS
        }
        self.stats_dict_avg: dict[str, dict[str, Union[int, float]]] = (
            defaultdict(dict)
        )
        """
        e.g.,
        self.stats_dict_avg = {
            "e2e_time_requests": {"latest": ..., "overall": ...},
            "queue_time_requests": {"latest": ..., "overall": ...},
            "prefill_time_requests": {"latest": ..., "overall": ...},
            "mean_time_per_output_token_requests": 
            {"latest": ..., "overall": ...},
            "time_to_first_token": {"latest": ..., "overall": ...}
            }
        """

    def _reset(self, now):
        self.last_log_time = now

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens = 0
        self.num_generation_tokens = 0

        for key in self.stats_dict:
            self.stats_dict[key]["latest"] = [0, 0.0]

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        # Save tracked stats for token counters.
        self.num_prompt_tokens += iteration_stats.num_prompt_tokens
        self.num_generation_tokens += iteration_stats.num_generation_tokens

    def _get_throughput(self, tracked_stats: int, now: float) -> float:
        # Compute summary metrics for tracked stats
        delta_time = now - self.last_log_time
        if delta_time <= 0.0:
            return 0.0
        return float(tracked_stats / delta_time)

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        engine_idx: int = 0,
    ):
        """Log Stats to standard output."""
        if iteration_stats:
            self._track_iteration_stats(iteration_stats)
            self._observe(iteration_stats)

    def log(self):
        now = time.monotonic()
        prompt_throughput = self._get_throughput(self.num_prompt_tokens, now)
        generation_throughput = self._get_throughput(
            self.num_generation_tokens, now
        )

        log_fn = logger.info
        if not any(
            (
                prompt_throughput,
                generation_throughput,
                self.last_prompt_throughput,
                self.last_generation_throughput,
            )
        ):
            # Avoid log noise on an idle production system
            log_fn = logger.debug
        self.last_generation_throughput = generation_throughput
        self.last_prompt_throughput = prompt_throughput
        # compute average stats
        self.stats_dict_avg = {
            key: {
                phase: (
                    self.stats_dict[key][phase][1]
                    / self.stats_dict[key][phase][0]
                    if self.stats_dict[key][phase][0] > 0
                    else 0.0
                )
                for phase in ["latest", "overall"]
            }
            for key in self.EPD_STATS_KEYS
        }

        # write into class variable SNAPSHOTS_AVG
        with self.__class__._LOCK:
            # deepcopy from stats_dict_avg
            snapshot = {
                key: {
                    "latest": float(self.stats_dict_avg[key]["latest"]),
                    "overall": float(self.stats_dict_avg[key]["overall"]),
                }
                for key in self.EPD_STATS_KEYS
            }
            self.__class__.SNAPSHOTS_AVG[self.engine_index] = snapshot

        log_msg = "Engine %d: " + ", ".join(
            [
                f"Avg {key.replace('_', ' ')}: %.3f ms"
                for key in self.EPD_STATS_KEYS
            ]
        )

        log_args = [self.engine_index] + [
            self.stats_dict_avg[key]["latest"] for key in self.EPD_STATS_KEYS
        ]
        log_fn(log_msg, *log_args)
        # clear latest stats
        self._reset(now)

    # Get a snapshot of the current stats averages
    @classmethod
    def get_stats_snapshot_avg(
        cls,
    ) -> dict[int, dict[str, Union[int, float]]]:
        """
        return:
        {
          0: {"engine_idx": 0, "e2e_time_requests": ..., ...},
          1: {...},
        }
        """
        with cls._LOCK:
            snapshot = {}
            for idx, stats in cls.SNAPSHOTS_AVG.items():
                snapshot[idx] = {
                    "engine_idx": idx,
                    **{key: stats[key]["overall"] for key in stats},
                }
            return snapshot

    # Observe per-request stats
    # [latest_count_num, latest_seconds, overall_count_num, overall_seconds]
    def _observe(self, iteration_stats: IterationStats):
        # update stats_dict
        # last item is time_to_first_token
        # it should be handled separately from time_to_first_tokens_iter
        for finished_request in iteration_stats.finished_requests:
            for key, attr in zip(
                self.EPD_STATS_KEYS[:-1], self.finished_request_attr
            ):
                value = getattr(finished_request, attr, 0.0)
                value *= 1000.0  # convert to milliseconds
                self.stats_dict[key]["latest"][0] += 1
                self.stats_dict[key]["latest"][1] += value
                self.stats_dict[key]["overall"][0] += 1
                self.stats_dict[key]["overall"][1] += value
        for ttft in iteration_stats.time_to_first_tokens_iter:
            ttft *= 1000.0  # convert to milliseconds
            self.stats_dict["time_to_first_token"]["latest"][0] += 1
            self.stats_dict["time_to_first_token"]["latest"][1] += ttft
            self.stats_dict["time_to_first_token"]["overall"][0] += 1
            self.stats_dict["time_to_first_token"]["overall"][1] += ttft

    def log_engine_initialized(self):
        if self.vllm_config.cache_config.num_gpu_blocks:
            logger.info(
                "Engine %03d: vllm cache_config_info with initialization "
                "after num_gpu_blocks is: %d",
                self.engine_index,
                self.vllm_config.cache_config.num_gpu_blocks,
            )


class MetricsReporter:
    def __init__(
        self,
        server_type: ServerType,
        instances: dict[str, zmq.asyncio.Socket],
        get_metrics_func,
    ):
        self.server_type = server_type
        self._instances = instances
        self._get_metrics_func = get_metrics_func
        self.proxy_to_instance_time_count: defaultdict[str, int] = defaultdict(
            int
        )
        self.proxy_to_instance_time_total: defaultdict[str, float] = (
            defaultdict(float)
        )
        self.proxy_ttft_total: float = 0.0
        self.proxy_ttft_count: int = 0

    def get_avg_proxy_to_instance_time(self, work_addr: str) -> float:
        return (
            self.proxy_to_instance_time_total[work_addr]
            * 1000.0
            / self.proxy_to_instance_time_count[work_addr]
            if self.proxy_to_instance_time_count[work_addr] > 0
            else 0.0
        )

    def get_avg_proxy_ttft(self) -> float:
        return (
            self.proxy_ttft_total * 1000.0 / self.proxy_ttft_count
            if self.proxy_ttft_count > 0
            else 0.0
        )

    def add_proxy_to_instance_time(self, work_addr: str, time: float):
        self.proxy_to_instance_time_count[work_addr] += 1
        self.proxy_to_instance_time_total[work_addr] += time

    def cal_proxy_ttft(
        self, ttft_recorded_flag: bool, start: float, resp
    ) -> bool:
        if ttft_recorded_flag:
            return True
        token_ids: Optional[list[int]] = getattr(resp, "token_ids", None)
        has_first_token: bool = token_ids is not None and len(token_ids) > 0
        if not has_first_token:
            return False
        self.proxy_ttft_count += 1
        self.proxy_ttft_total += time.perf_counter() - start
        return True

    async def log_metrics(self) -> None:
        # metrics: [addr, metrics_msg or error_msg]
        metrics = await self.build_metrics_msg()
        logger.info("Metrics for %s instances:" % self.server_type)
        for msg in metrics.values():
            if "failed" in msg:
                logger.error(msg)
            logger.info(msg)

    async def build_metrics_msg(self) -> dict[str, str]:
        # work_addr -> msg
        metrics: dict[str, str] = {}
        tasks = [
            asyncio.create_task(
                asyncio.wait_for(
                    self._get_metrics_func(self.server_type, work_addr),
                    timeout=1.0,
                )
            )
            for work_addr in self._instances.keys()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        log_msg = (
            "ec_role: %s, "
            "addr: %s, "
            "Avg e2e time requests: %.3f ms, "
            "Avg queue time requests: %.3f ms, "
            "Avg prefill time requests: %.3f ms, "
            "Avg preprocess time requests: %.3f ms, "
            "Avg mean time per output token requests: %.3f ms, "
            "Avg time to first token: %.3f ms, "
            "Avg proxy ttft: %.3f ms, "
            "Avg proxy to instance requests time: %.3f ms "
        )
        msg: str = ""
        for work_addr, result in zip(self._instances.keys(), results):
            if isinstance(result, dict):
                for _, value in result.items():
                    msg = log_msg % (
                        self.server_type.name,
                        work_addr,
                        value.get("e2e_time_requests", 0.0),
                        value.get("queue_time_requests", 0.0),
                        value.get("prefill_time_requests", 0.0),
                        # preprocess time = ttft - queue - prefill
                        value.get("time_to_first_token", 0.0)
                        - value.get("queue_time_requests", 0.0)
                        - value.get("prefill_time_requests", 0.0),
                        value.get("mean_time_per_output_token_requests", 0.0),
                        value.get("time_to_first_token", 0.0)
                        if self.has_d_instance()
                        else 0.0,
                        value.get("proxy_ttft_avg", 0.0)
                        if self.has_d_instance()
                        else 0.0,
                        value.get("proxy_to_instance_time_avg", 0.0),
                    )

                metrics[work_addr] = msg
            else:
                error_msg = (
                    f"Get metrics for {self.server_type} {work_addr} failed, reason is "
                    f"({'timeout' if isinstance(result, asyncio.TimeoutError) else result})."
                )
                metrics[work_addr] = error_msg
        return metrics

    def has_d_instance(self) -> bool:
        return (
            self.server_type == ServerType.D_INSTANCE
            or self.server_type == ServerType.PD_INSTANCE
        )
