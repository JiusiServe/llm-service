# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import random

from lm_service.request_stats import RequestStats


class RoutingInterface:
    def route_request(self, endpoints: list[int], request_stats: dict) -> int:
        """
        Route the request to a specific instance based on the request stats.
        It can also be based on engine stats in the future.

        Args:
            endpoints (list[int]): The list of instance IDs.
            request_stats (dict): The incoming request stats.

        Returns:
            int: The ID of the selected instance.
        """

        # Implement your routing logic here
        raise NotImplementedError("Subclasses should implement this method.")


class RandomRouter(RoutingInterface):
    def route_request(self, endpoints: list[int], request_stats: dict) -> int:
        if not endpoints:
            raise RuntimeError("No healthy endpoints available for routing.")
        return random.choice(endpoints)


class RoundRobinRouter(RoutingInterface):
    def __init__(self):
        self.current_index = 0

    def route_request(self, endpoints: list[int], request_stats: dict) -> int:
        if not endpoints:
            raise RuntimeError("No healthy endpoints available for routing.")
        selected_index = self.current_index % len(endpoints)
        self.current_index = selected_index + 1
        return endpoints[selected_index]


class LeastInFlightRouter(RoutingInterface):
    def route_request(self, endpoints: list[int], request_stats: dict) -> int:
        if not endpoints:
            raise RuntimeError("No healthy endpoints available for routing.")

        def get_in_flight_count(endpoint_id: int) -> int:
            stats: RequestStats | None = request_stats.get(endpoint_id)
            return len(stats.in_flight_requests) if stats else 0

        return min(endpoints, key=get_in_flight_count)
