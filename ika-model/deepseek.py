import logging
from typing import Optional

from .openai import OpenAIModel

_LOG = logging.getLogger(__name__)


class DeepSeekModel(OpenAIModel):
    def __init__(
        self,
        model_id: str = "deepseek-chat",
        api_key: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        custom_role_conversions: Optional[dict[str, str]] = None,
        **kwargs
    ):
        base_url = "https://api.deepseek.com"
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            requests_per_minute=requests_per_minute,
            custom_role_conversions=custom_role_conversions,
            **kwargs
        )
        _LOG.info("Initialized DeepSeek model: %s", model_id)

    def _get_model_cost(self, model_id):
        return {
            "deepseek-chat": (0.14, 0.14, 0.28),
            "deepseek-coder": (0.14, 0.14, 0.28),
            "deepseek-reasoner": (0.55, 0.55, 2.19),
            "deepseek-v3": (0.14, 0.14, 0.28),
            "deepseek-v3-chat": (0.14, 0.14, 0.28),
        }.get(model_id)

