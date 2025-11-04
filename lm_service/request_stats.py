# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

from dataclasses import dataclass
from collections import defaultdict

import zmq
import zmq.asyncio


@dataclass
class RequestStats:
    # Inflight request count
    in_flight_requests: set[str]
    # Other stats can be added for more complex scheduling


class RequestStatsMonitor:
    """
    Monitors and records request statistics for all instances.
    """

    def __init__(self, instances: dict[str, zmq.asyncio.Socket]):
        # Key: instance addr
        self.request_stats: dict[str, RequestStats] = defaultdict(
            lambda: RequestStats(in_flight_requests=set())
        )
        self.instances = instances

    def on_new_request(self, instance_addr: str, request_id: str):
        self.request_stats[instance_addr].in_flight_requests.add(request_id)

    def on_request_completed(self, instance_addr: str, request_id: str):
        self.request_stats[instance_addr].in_flight_requests.discard(request_id)

    def get_request_stats(self):
        return self.request_stats
