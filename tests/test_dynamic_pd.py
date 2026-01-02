# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import numpy as np

from lm_service.dynamic_pd import (
    DynamicPd,
    SloConfig,
    StatsConfig,
    PdEndpointInfo,
)


class TestDynamicPd:
    def test_too_few_hist(self):
        dynamic_pd = self._create_dynamic_pd()
        self._gen_ttft_tpot(dynamic_pd, 10, (5.0, 10.0), (12.5, 25.0))

        num_prefills = 3
        num_decodes = 3
        endpoints = self._create_endpoints(num_prefills, num_decodes)

        switch_advice = dynamic_pd.advise_switch(endpoints)
        elastic_advice = dynamic_pd.advise_elastic(endpoints)

        assert switch_advice is None
        assert elastic_advice is None

    def test_good_ttft_good_tpot(self):
        dynamic_pd = self._create_dynamic_pd()
        self._gen_ttft_tpot(dynamic_pd, 1000, (0.5, 1.0), (0.125, 0.25))

        num_prefills = 3
        num_decodes = 3
        endpoints = self._create_endpoints(num_prefills, num_decodes)

        switch_advice = dynamic_pd.advise_switch(endpoints)
        elastic_advice = dynamic_pd.advise_elastic(endpoints)

        assert switch_advice is None
        assert elastic_advice is None

    def test_excel_ttft_excel_tpot(self):
        dynamic_pd = self._create_dynamic_pd()
        self._gen_ttft_tpot(dynamic_pd, 1000, (0.05, 0.1), (0.0125, 0.025))

        num_prefills = 3
        num_decodes = 3
        endpoints = self._create_endpoints(num_prefills, num_decodes)

        switch_advice = dynamic_pd.advise_switch(endpoints)
        elastic_advice = dynamic_pd.advise_elastic(endpoints)

        assert switch_advice is None

        assert elastic_advice is not None
        assert elastic_advice.drop_prefills is not None
        assert len(elastic_advice.drop_prefills) >= 1
        assert elastic_advice.drop_decodes is not None
        assert len(elastic_advice.drop_decodes) >= 1
        assert elastic_advice.delta_prefills <= -1
        assert elastic_advice.delta_decodes <= -1
        assert (
            elastic_advice.new_total_prefills
            == num_prefills + elastic_advice.delta_prefills
        )
        assert (
            elastic_advice.new_total_decodes
            == num_decodes + elastic_advice.delta_decodes
        )

    def test_bad_ttft_good_tpot(self):
        dynamic_pd = self._create_dynamic_pd()
        self._gen_ttft_tpot(dynamic_pd, 1000, (5.0, 10.0), (0.125, 0.25))

        num_prefills = 3
        num_decodes = 3
        endpoints = self._create_endpoints(num_prefills, num_decodes)

        switch_advice = dynamic_pd.advise_switch(endpoints)
        elastic_advice = dynamic_pd.advise_elastic(endpoints)

        assert switch_advice is None

        assert elastic_advice is not None
        assert elastic_advice.drop_prefills is None
        assert elastic_advice.drop_decodes is None
        assert elastic_advice.delta_prefills >= 1
        assert elastic_advice.delta_decodes == 0
        assert (
            elastic_advice.new_total_prefills
            == num_prefills + elastic_advice.delta_prefills
        )
        assert (
            elastic_advice.new_total_decodes
            == num_decodes + elastic_advice.delta_decodes
        )

    def test_good_ttft_bad_tpot(self):
        dynamic_pd = self._create_dynamic_pd()
        self._gen_ttft_tpot(dynamic_pd, 1000, (0.5, 1.0), (12.5, 25.0))

        num_prefills = 3
        num_decodes = 3
        endpoints = self._create_endpoints(num_prefills, num_decodes)

        switch_advice = dynamic_pd.advise_switch(endpoints)
        elastic_advice = dynamic_pd.advise_elastic(endpoints)

        assert switch_advice is None

        assert elastic_advice is not None
        assert elastic_advice.drop_prefills is None
        assert elastic_advice.drop_decodes is None
        assert elastic_advice.delta_prefills == 0
        assert elastic_advice.delta_decodes >= 1
        assert (
            elastic_advice.new_total_prefills
            == num_prefills + elastic_advice.delta_prefills
        )
        assert (
            elastic_advice.new_total_decodes
            == num_decodes + elastic_advice.delta_decodes
        )

    def test_excel_ttft_bad_tpot(self):
        dynamic_pd = self._create_dynamic_pd()
        self._gen_ttft_tpot(dynamic_pd, 1000, (0.05, 0.1), (12.5, 25.0))

        num_prefills = 3
        num_decodes = 3
        endpoints = self._create_endpoints(num_prefills, num_decodes)

        switch_advice = dynamic_pd.advise_switch(endpoints)
        elastic_advice = dynamic_pd.advise_elastic(endpoints)

        assert switch_advice is not None
        assert endpoints[switch_advice.switch_endpoints[0]].is_prefill

        assert elastic_advice is not None
        assert elastic_advice.drop_prefills is not None
        assert len(elastic_advice.drop_prefills) >= 1
        assert elastic_advice.drop_decodes is None
        assert elastic_advice.delta_prefills <= -1
        assert elastic_advice.delta_decodes >= 1
        assert (
            elastic_advice.new_total_prefills
            == num_prefills + elastic_advice.delta_prefills
        )
        assert (
            elastic_advice.new_total_decodes
            == num_decodes + elastic_advice.delta_decodes
        )

    def test_bad_ttft_excel_tpot(self):
        dynamic_pd = self._create_dynamic_pd()
        self._gen_ttft_tpot(dynamic_pd, 1000, (5.0, 10.0), (0.0125, 0.025))

        num_prefills = 3
        num_decodes = 3
        endpoints = self._create_endpoints(num_prefills, num_decodes)

        switch_advice = dynamic_pd.advise_switch(endpoints)
        elastic_advice = dynamic_pd.advise_elastic(endpoints)

        assert switch_advice is not None
        assert not endpoints[switch_advice.switch_endpoints[0]].is_prefill

        assert elastic_advice is not None
        assert elastic_advice.drop_prefills is None
        assert elastic_advice.drop_decodes is not None
        assert len(elastic_advice.drop_decodes) >= 1
        assert elastic_advice.delta_prefills >= 1
        assert elastic_advice.delta_decodes <= -1
        assert (
            elastic_advice.new_total_prefills
            == num_prefills + elastic_advice.delta_prefills
        )
        assert (
            elastic_advice.new_total_decodes
            == num_decodes + elastic_advice.delta_decodes
        )

    def test_bad_ttft_bad_tpot_queue_len_guided(self):
        dynamic_pd = self._create_dynamic_pd()
        self._gen_ttft_tpot(dynamic_pd, 1000, (5.0, 10.0), (1.25, 2.5))

        endpoints = [
            PdEndpointInfo(
                is_prefill=True,
                is_switchable=True,
                queue_length=100,
            ),
            PdEndpointInfo(
                is_prefill=True,
                is_switchable=True,
                queue_length=100,
            ),
            PdEndpointInfo(
                is_prefill=False,
                is_switchable=True,
                queue_length=1,
            ),
            PdEndpointInfo(
                is_prefill=False,
                is_switchable=True,
                queue_length=1,
            ),
        ]

        switch_advice = dynamic_pd.advise_switch(endpoints)
        elastic_advice = dynamic_pd.advise_elastic(endpoints)

        assert switch_advice is not None
        assert not endpoints[switch_advice.switch_endpoints[0]].is_prefill

        assert elastic_advice is not None
        assert elastic_advice.drop_prefills is None
        assert elastic_advice.drop_decodes is None
        assert elastic_advice.delta_prefills >= 1
        assert elastic_advice.delta_decodes >= 1
        assert (
            elastic_advice.new_total_prefills
            == 2 + elastic_advice.delta_prefills
        )
        assert (
            elastic_advice.new_total_decodes == 2 + elastic_advice.delta_decodes
        )

    @staticmethod
    def _create_dynamic_pd():
        return DynamicPd(slo_config=SloConfig(), stats_config=StatsConfig())

    @staticmethod
    def _gen_ttft_tpot(dynamic_pd, num_reqs, ttft_rng, tpot_rng):
        ttfts = np.random.uniform(ttft_rng[0], ttft_rng[1], num_reqs)
        tpots = np.random.uniform(tpot_rng[0], tpot_rng[1], num_reqs)
        for ttft, tpot in zip(ttfts, tpots):
            dynamic_pd.on_request_finished(ttft, tpot)

    @staticmethod
    def _create_endpoints(num_prefills, num_decodes):
        num_endpoints = num_prefills + num_decodes
        return [
            PdEndpointInfo(
                is_prefill=(i < num_prefills),
                is_switchable=True,
                queue_length=10,
            )
            for i in range(num_endpoints)
        ]
