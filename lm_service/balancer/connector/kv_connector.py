# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from typing import List, Set, Dict, Optional


class KvConnector:
    @property
    def is_p2p_enabled(self) -> bool:
        raise NotImplementedError

    def query_hit_len(self, tokens: List[int], instance_ids: Optional[Set[str]] = None) -> Dict[str, int]:
        raise NotImplementedError
