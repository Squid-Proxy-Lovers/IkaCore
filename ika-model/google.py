import logging
from typing import Optional

from .openai import OpenAIModel

_LOG = logging.getLogger(__name__)


class GeminiAPIModel(OpenAIModel):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        requests_per_minute: Optional[float] = None,
        custom_role_conversions: Optional[dict[str, str]] = None,
        **kwargs
    ):
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        super().__init__(
            model_id, api_key=api_key, base_url=base_url,
            requests_per_minute=requests_per_minute,
            custom_role_conversions=custom_role_conversions,
            **kwargs
        )

    def _get_model_cost(self, model_id):
        return {
        }.get(model_id)

