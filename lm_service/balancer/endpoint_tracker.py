# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
from threading import Lock
from typing import List, Optional, Tuple, Set, Union

from .common import Stage
from .endpoint import Endpoint


class EndpointTrackerListener:
    def on_endpoints_changed(self, new_ups: List[Endpoint], new_downs: List[Endpoint]):
        pass


class EndpointTracker:
    def __init__(self):
        self._listeners: Set[EndpointTrackerListener] = set()

    def get_lock(self) -> Optional[Lock]:
        raise NotImplementedError

    def get_up_endpoints(self, stages: Optional[Union[Tuple[Stage, ...], List[Stage]]] = None) -> List[Endpoint]:
        raise NotImplementedError

    def add_listener(self, listener: EndpointTrackerListener):
        lock = self.get_lock()
        if lock:
            with lock:
                self._listeners.add(listener)
        else:
            self._listeners.add(listener)

    def remove_listener(self, listener: EndpointTrackerListener):
        lock = self.get_lock()
        if lock:
            with lock:
                self._listeners.discard(listener)
        else:
            self._listeners.discard(listener)

    def on_endpoints_changed(self, new_ups: List[Endpoint], new_downs: List[Endpoint]):
        """To be called by the sub-classes."""
        lock = self.get_lock()
        if lock:
            with lock:
                for listener in self._listeners:
                    listener.on_endpoints_changed(new_ups, new_downs)
        else:
            for listener in self._listeners:
                listener.on_endpoints_changed(new_ups, new_downs)


class StaticEndpointTracker(EndpointTracker):
    def __init__(self, endpoints: List[Endpoint]):
        super().__init__()
        self._endpoints = endpoints

    def get_lock(self) -> Optional[Lock]:
        return None

    def get_up_endpoints(self, stages: Optional[Union[Tuple[Stage, ...], List[Stage]]] = None) -> List[Endpoint]:
        if stages:
            return [ep for ep in self._endpoints if ep.stage in stages]
        return self._endpoints.copy()
