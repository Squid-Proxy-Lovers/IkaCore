import logging
from typing import Optional

from .openai import OpenAIModel

_LOG = logging.getLogger(__name__)


class GrokModel(OpenAIModel):
    def __init__(
        self,
        model_id: str = "grok-beta",
        api_key: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        custom_role_conversions: Optional[dict[str, str]] = None,
        **kwargs
    ):
        base_url = "https://api.x.ai/v1"
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            requests_per_minute=requests_per_minute,
            custom_role_conversions=custom_role_conversions,
            **kwargs
        )
        _LOG.info("Initialized Grok model: %s", model_id)

    def _get_model_cost(self, model_id):
        return {
            "grok-beta": (0.40, 0.05, 0.40),
            "grok-4": (0.40, 0.05, 0.40),
            "grok-4-0709": (0.40, 0.05, 0.40),
            "grok-4-1-fast-reasoning": (0.40, 0.05, 0.40),
            "grok-4-1-fast": (0.40, 0.05, 0.40),
            "grok-2-1212": (0.40, 0.05, 0.40),
            "grok-2-vision-1212": (0.40, 0.05, 0.40),
            "grok-2-mini-1212": (0.10, 0.01, 0.10),
        }.get(model_id)

