# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import time
from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np


_SLO_EXCEL_SLOT: int = 0
_SLO_GOOD_SLOT: int = 1
_SLO_BAD_SLOT: int = 2
_MAX_Q_LEN_THD_PRE_EP_REQ: int = 5


@dataclass
class SloConfig:
    """
    SLO Configurations.
    """

    # max. TTFT that is pass
    pass_ttft: float = 1.0
    # max. TPOT that is pass
    pass_tpot: float = 0.25
    # max. TTFT that is excellent, -1 means pass_ttft/2 by auto
    excel_ttft: float = -1
    # max. TPOT that is excellent, -1 means pass_tpot/2 by auto
    excel_tpot: float = -1
    # [0.0-1.0] cut-point for finding quantile of TTFT and TPOT
    p_quantile: float = 0.99

    @property
    def auto_excel_ttft(self):
        if self.excel_ttft < 0:
            return self.pass_ttft / 2
        return self.excel_ttft

    @property
    def auto_excel_tpot(self):
        if self.excel_tpot < 0:
            return self.pass_tpot / 2
        return self.excel_tpot


@dataclass
class StatsConfig:
    """
    Statistics Configurations.
    """

    # max. no. of TTFT/TPOT history values to be stored
    max_history: int = 1000
    # min. no. of TTFT/TPOT history values required for generating advice
    min_history: int = 100
    # look back window in seconds for generating advice
    time_window: float = 5 * 60


@dataclass
class PdEndpointInfo:
    """
    P/D Disaggregated Endpoint Info.
    """

    # is it a prefill endpoint
    is_prefill: bool
    # is it switchable from P to D or D to P
    is_switchable: bool
    # no. of open tasks in the endpoint's queue
    queue_length: int


@dataclass
class SwitchAdvice:
    """
    Switch Advice for fixed number of P/D endpoints.
    """

    # Indices of endpoints to be switched
    switch_endpoints: List[int]


@dataclass
class ElasticAdvice:
    """
    Elastic Advice for unlimited number of P/D endpoints.
    """

    # Indices of endpoints to be dropped (as a prefill)
    drop_prefills: List[int] | None = None
    # Indices of endpoints to be dropped (as a decode)
    drop_decodes: List[int] | None = None
    # no. of prefill endpoints to be added
    delta_prefills: int = 0
    # no. of decode endpoints to be added
    delta_decodes: int = 0
    # new total no. of prefill endpoints
    new_total_prefills: int = -1
    # new total no. of decode endpoints
    new_total_decodes: int = -1


@dataclass
class _State:
    endpoints: List[PdEndpointInfo]
    switchable_prefills: List[int]
    switchable_decodes: List[int]
    num_prefill_only: int = -1
    num_decode_only: int = -1
    ttft_slot: int = -1
    tpot_slot: int = -1

    @property
    def num_droppable_p(self):
        return max(self.num_prefills - 1, 0)

    @property
    def num_droppable_d(self):
        return max(self.num_decodes - 1, 0)

    @property
    def can_p2d(self):
        return self.num_prefills >= 2 and len(self.switchable_prefills) >= 1

    @property
    def can_d2p(self):
        return self.num_decodes >= 2 and len(self.switchable_decodes) >= 1

    @property
    def num_prefills(self):
        return len(self.switchable_prefills) + self.num_prefill_only

    @property
    def num_decodes(self):
        return len(self.switchable_decodes) + self.num_decode_only


def _slo_mid_point(excel: float, pass_: float):
    mid = (2 * excel * pass_) / (excel + pass_)
    return excel, mid, pass_


def _to_slot(metric: float, excel: float, pass_: float) -> int:
    if metric > pass_:
        return _SLO_BAD_SLOT
    if metric <= excel:
        return _SLO_EXCEL_SLOT
    return _SLO_GOOD_SLOT


class _SwitchAdviser:
    class _Action(Enum):
        NO_ACTION = 0
        P2D = 1
        D2P = 2
        KEEP_P2D_BY_BAD_TPOT = 3
        KEEP_D2P_BY_BAD_TTFT = 4

    def __init__(self, slo_config: SloConfig):
        self._slo_config = slo_config
        self._last_action = self._Action.NO_ACTION

    def advise(self, state: _State) -> SwitchAdvice | None:
        if (
            state.ttft_slot <= _SLO_GOOD_SLOT
            and state.tpot_slot <= _SLO_GOOD_SLOT
        ):
            action = self._decide_excel_or_good_slo(state)
        elif (
            state.ttft_slot == _SLO_EXCEL_SLOT
            and state.tpot_slot == _SLO_BAD_SLOT
        ):
            action = self._decide_excel_ttft_bad_tpot(state)
        elif (
            state.ttft_slot == _SLO_BAD_SLOT
            and state.tpot_slot == _SLO_EXCEL_SLOT
        ):
            action = self._decide_bad_ttft_excel_tpot(state)
        else:
            action = self._decide_queue_len_guided(state)

        advice = self._get_advice(state, action)
        self._last_action = action
        return advice

    def _get_advice(self, state, action):
        if action == self._Action.NO_ACTION:
            return None
        if action in (self._Action.P2D, self._Action.KEEP_P2D_BY_BAD_TPOT):
            if not state.can_p2d:
                raise RuntimeError("Cannot P2D")
            is_to_prefill = False
        elif action in (self._Action.D2P, self._Action.KEEP_D2P_BY_BAD_TTFT):
            if not state.can_d2p:
                raise RuntimeError("Cannot D2P")
            is_to_prefill = True
        else:
            raise ValueError(f"Unsupported action: {action}")
        switch_endpoints = [self._find_best_switchable(state, is_to_prefill)]
        return SwitchAdvice(switch_endpoints=switch_endpoints)

    @staticmethod
    def _find_best_switchable(state, is_to_prefill):
        best = -1
        min_length = -1
        switchables = (
            state.switchable_decodes
            if is_to_prefill
            else state.switchable_prefills
        )
        for endpoint_i in switchables:
            length = state.endpoints[endpoint_i].queue_length
            if best == -1 or length < min_length:
                best = endpoint_i
                min_length = length
        if best == -1:
            raise RuntimeError("Cannot find the best switchable")
        return best

    def _decide_excel_or_good_slo(self, state):
        mid_tpot = _slo_mid_point(
            self._slo_config.auto_excel_tpot, self._slo_config.pass_tpot
        )
        mid_ttft = _slo_mid_point(
            self._slo_config.auto_excel_ttft, self._slo_config.pass_ttft
        )
        if self._last_action == self._Action.KEEP_D2P_BY_BAD_TTFT:
            if (
                state.tpot_quantile < mid_tpot
                and state.ttft_quantile > mid_ttft
            ):
                return self._decide_keep_d2p_by_bad_ttft(state)
        elif self._last_action == self._Action.KEEP_P2D_BY_BAD_TPOT:
            if (
                state.ttft_quantile < mid_ttft
                and state.tpot_quantile > mid_tpot
            ):
                return self._decide_keep_p2d_by_bad_tpot(state)
        return self._Action.NO_ACTION

    def _decide_queue_len_guided(self, state):
        num_prefill_ep = 0
        num_decode_ep = 0
        prefill_queue_len = 0
        decode_queue_len = 0
        for endpoint in state.endpoints:
            if endpoint.is_prefill:
                num_prefill_ep += 1
                prefill_queue_len += endpoint.queue_length
            else:
                num_decode_ep += 1
                decode_queue_len += endpoint.queue_length
        if prefill_queue_len > decode_queue_len:
            max_queue_len = prefill_queue_len
            queue_len_threshold = num_prefill_ep * _MAX_Q_LEN_THD_PRE_EP_REQ
        else:
            max_queue_len = decode_queue_len
            queue_len_threshold = num_decode_ep * _MAX_Q_LEN_THD_PRE_EP_REQ
        if max_queue_len <= queue_len_threshold:
            return self._Action.NO_ACTION
        switch_threshold = max_queue_len / 2

        if prefill_queue_len < switch_threshold:
            if state.can_p2d:
                return self._Action.P2D
        elif decode_queue_len < switch_threshold:
            if state.can_d2p:
                return self._Action.D2P
        return self._Action.NO_ACTION

    def _decide_excel_ttft_bad_tpot(self, state):
        action = self._decide_keep_p2d_by_bad_tpot(state)
        if action == self._Action.NO_ACTION:
            return self._decide_queue_len_guided(state)
        return action

    def _decide_bad_ttft_excel_tpot(self, state):
        action = self._decide_keep_d2p_by_bad_ttft(state)
        if action == self._Action.NO_ACTION:
            return self._decide_queue_len_guided(state)
        return action

    def _decide_keep_d2p_by_bad_ttft(self, state):
        if state.can_d2p:
            if len(state.switchable_decodes) == 1:
                # the last one to be switched in the prolonged action
                return self._Action.D2P
            return self._Action.KEEP_D2P_BY_BAD_TTFT
        return self._Action.NO_ACTION

    def _decide_keep_p2d_by_bad_tpot(self, state):
        if state.can_p2d:
            if len(state.switchable_prefills) == 1:
                # the last one to be switched in the prolonged action
                return self._Action.P2D
            return self._Action.KEEP_P2D_BY_BAD_TPOT
        return self._Action.NO_ACTION


class _ElasticAdviser:
    @staticmethod
    def advise(state: _State) -> ElasticAdvice | None:
        has_advice = False
        advice = ElasticAdvice()
        if state.ttft_slot == _SLO_EXCEL_SLOT:
            if state.num_droppable_p > 0:
                # TODO find out how many instances to be dropped
                advice.drop_prefills = [
                    np.argmin(
                        [
                            e.queue_length
                            for e in state.endpoints
                            if e.is_prefill
                        ]
                    ).tolist()
                ]

                advice.delta_prefills = -len(advice.drop_prefills)
                has_advice = True
        elif state.ttft_slot == _SLO_BAD_SLOT:
            advice.delta_prefills = 1
            has_advice = True

        if state.tpot_slot == _SLO_EXCEL_SLOT:
            if state.num_droppable_d > 0:
                # TODO find out how many instances to be dropped
                advice.drop_decodes = [
                    np.argmin(
                        [
                            e.queue_length
                            for e in state.endpoints
                            if not e.is_prefill
                        ]
                    ).tolist()
                ]
                advice.delta_decodes = -len(advice.drop_decodes)
                has_advice = True
        elif state.tpot_slot == _SLO_BAD_SLOT:
            advice.delta_decodes = 1
            has_advice = True

        advice.new_total_prefills = state.num_prefills + advice.delta_prefills
        advice.new_total_decodes = state.num_decodes + advice.delta_decodes
        return advice if has_advice else None


class _CircularList:
    def __init__(self, max_len, dtype=np.float32):
        self.ary = np.empty(max_len, dtype=dtype)
        self.max_len = max_len
        self.position = -1
        self._length = 0

    def length(self):
        return self._length

    def append(self, element):
        self.position = (self.position + 1) % self.max_len
        if self.position >= self._length:
            self._length = self.position + 1
        self.ary[self.position] = element
        return self.position

    def clear(self):
        self._length = 0
        self.position = -1


class DynamicPd:
    """
    Dynamic P/D endpoints switching or elastic allocations.
    """

    def __init__(self, slo_config: SloConfig, stats_config: StatsConfig):
        self._slo_config = slo_config
        self._stats_config = stats_config
        self._ttft_hist = _CircularList(stats_config.max_history)
        self._tpot_hist = _CircularList(stats_config.max_history)
        self._time_hist = _CircularList(stats_config.max_history)
        self._switch_adviser = _SwitchAdviser(slo_config)
        self._elastic_adviser = _ElasticAdviser()

    def on_request_finished(
        self, ttft: float, tpot: float, finish_time: float = -1
    ):
        """
        Gather a request's TTFT & TPOT.

        Args:
            ttft (float): TTFT.
            tpot (float): TPOT.
            finish_time (float): request finish time.

        Returns:
            Switch Advice or None if there is no advice.
        """
        if ttft > 0 and tpot > 0:
            self._ttft_hist.append(ttft)
            self._tpot_hist.append(tpot)
            self._time_hist.append(
                finish_time if finish_time > 0 else time.time()
            )

    def advise_switch(
        self, endpoints: List[PdEndpointInfo]
    ) -> SwitchAdvice | None:
        """
        Give Switch Advice.

        Args:
            endpoints (list[PdEndpointInfo]): List of P/D disaggregated endpoints.

        Returns:
            Switch Advice or None if there is no advice.
        """
        state = self._gather_state(endpoints)
        if state is None:
            return None
        return self._switch_adviser.advise(state)

    def advise_elastic(
        self, endpoints: List[PdEndpointInfo]
    ) -> ElasticAdvice | None:
        """
        Give Elastic Advice.

        Args:
            endpoints (list[PdEndpointInfo]): List of P/D disaggregated endpoints.

        Returns:
            Elastic Advice or None if there is no advice.
        """
        state = self._gather_state(endpoints)
        if state is None:
            return None
        return self._elastic_adviser.advise(state)

    def _gather_state(self, endpoints: List[PdEndpointInfo]):
        if self._time_hist.length() < self._stats_config.min_history:
            return None
        cutoff = time.time() - self._stats_config.time_window
        window = self._time_hist.ary > cutoff
        use_count = np.sum(window)
        if use_count < self._stats_config.min_history:
            return None
        ttft_wind = self._ttft_hist.ary[window]
        tpot_wind = self._tpot_hist.ary[window]
        ttft_quantile = np.quantile(ttft_wind, self._slo_config.p_quantile)
        tpot_quantile = np.quantile(tpot_wind, self._slo_config.p_quantile)

        ttft_slot = _to_slot(
            ttft_quantile,
            self._slo_config.auto_excel_ttft,
            self._slo_config.pass_ttft,
        )
        tpot_slot = _to_slot(
            tpot_quantile,
            self._slo_config.auto_excel_tpot,
            self._slo_config.pass_tpot,
        )

        switchable_prefills = []
        switchable_decodes = []
        num_prefill_only = 0
        num_decode_only = 0
        for i, endpoint in enumerate(endpoints):
            if endpoint.is_prefill:
                if endpoint.is_switchable:
                    switchable_prefills.append(i)
                else:
                    num_prefill_only += 1
            else:
                if endpoint.is_switchable:
                    switchable_decodes.append(i)
                else:
                    num_decode_only += 1
        return _State(
            endpoints=endpoints,
            switchable_prefills=switchable_prefills,
            switchable_decodes=switchable_decodes,
            num_prefill_only=num_prefill_only,
            num_decode_only=num_decode_only,
            ttft_slot=ttft_slot,
            tpot_slot=tpot_slot,
        )
