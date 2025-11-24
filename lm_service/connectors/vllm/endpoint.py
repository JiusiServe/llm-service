from dataclasses import dataclass

from ...balancer import EndpointConfig, Endpoint
from openai import AsyncOpenAI, OpenAI


class VllmEndpointConfig(EndpointConfig):
    def __init__(self):
        self.base_url: str = ""
        self.api_key: str = ""
        self.kv_event_endpoint: str = ""


class VllmEndpoint(Endpoint):

    def __init__(self, config: VllmEndpointConfig):
        super().__init__(config)
        self._openai_client = None

    def get_openai_client(self):
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
        return self._openai_client
