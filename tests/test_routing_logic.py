# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import pytest

from lm_service.routing_logic import LeastInFlightRouter
from lm_service.request_stats import RequestStats


class TestLeastInFlightRouter:
    def test_selects_endpoint_with_fewest_in_flight(self):
        router = LeastInFlightRouter()
        endpoints = [1, 2, 3]
        request_stats = {
            1: RequestStats(in_flight_requests={"a", "b", "c"}),  # 3
            2: RequestStats(in_flight_requests={"x"}),  # 1
            3: RequestStats(in_flight_requests={"m", "n"}),  # 2
        }

        selected = router.route_request(endpoints, request_stats)
        assert selected == 2

    def test_missing_stats_treated_as_zero(self):
        router = LeastInFlightRouter()
        endpoints = [1, 2]
        request_stats = {
            1: RequestStats(in_flight_requests={"a"}),  # 1
            # 2 missing -> treated as 0
        }

        selected = router.route_request(endpoints, request_stats)
        assert selected == 2

    def test_raises_when_no_endpoints(self):
        router = LeastInFlightRouter()
        with pytest.raises(RuntimeError):
            router.route_request([], {})

    def test_tie_breaker_any_minimal_endpoint(self):
        router = LeastInFlightRouter()
        endpoints = [5, 3, 9]
        request_stats = {
            5: RequestStats(in_flight_requests=set()),  # 0
            3: RequestStats(in_flight_requests=set()),  # 0
            9: RequestStats(in_flight_requests={"a"}),  # 1
        }

        selected = router.route_request(endpoints, request_stats)
        # Both 5 and 3 have minimal (0) in-flight; either is acceptable
        assert selected in {5, 3}
