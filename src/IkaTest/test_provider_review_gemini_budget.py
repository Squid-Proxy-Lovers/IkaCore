import pytest

from IkaModel.base import BareBoneModel
from IkaModel.gemini.google import gemini_fill_payload


@pytest.mark.parametrize("name,disabled", [
    ("gemini-2.5-pro", False), ("gemini-2.5-pro-preview-06-05", False),
    ("gemini-2.5-flash", True), ("gemini-2.5-flash-lite", True),
    ("gemini-2.0-flash", False), ("gemini-3.1-pro", False),
])
def test_gemini_disables_thinking_only_on_models_that_support_it(name, disabled):
    model = BareBoneModel(name, "fixture", "https://generativelanguage.googleapis.com/v1beta")
    config = gemini_fill_payload(model, [{"role": "user", "content": "task"}])["generationConfig"]
    assert config.get("thinkingConfig") == ({"thinkingBudget": 0} if disabled else None)
