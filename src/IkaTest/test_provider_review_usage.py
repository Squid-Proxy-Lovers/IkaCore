import pytest

from IkaModel.chat_interface.response_usage import extract_usage
from IkaModel.chat_interface.types import UsageInfo
from IkaModel.summarization import extract_summary_tokens


def test_anthropic_cache_categories_contribute_to_turn_and_summary_usage():
    data = {"usage": {"input_tokens": 50, "output_tokens": 25,
                      "cache_read_input_tokens": 100000, "cache_creation_input_tokens": 1000}}
    usage = extract_usage("anthropic", data)
    assert usage == {"input_tokens": 101050, "output_tokens": 25, "total_tokens": 101075,
                     "input_cached_tokens": 100000, "input_cache_creation_tokens": 1000}
    assert extract_summary_tokens("anthropic", data) == 101075
    accumulated = UsageInfo.from_mapping(usage)
    accumulated.add_mapping(usage)
    assert accumulated.to_dict() == {key: value * 2 for key, value in usage.items()}


@pytest.mark.parametrize("meta,expected", [
    ({"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 80, "totalTokenCount": 200}, (100, 100, 200)),
    ({"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 80}, (100, 100, 200)),
    ({"promptTokenCount": 100, "thoughtsTokenCount": 80, "totalTokenCount": 180}, (100, 80, 180)),
    ({"promptTokenCount": 100, "totalTokenCount": 180}, (100, 80, 180)),
    ({"promptTokenCount": 100, "candidatesTokenCount": 20}, (100, 20, 120)),
])
def test_gemini_output_usage_includes_thought_tokens(meta, expected):
    usage = extract_usage("gemini", {"usageMetadata": meta})
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == expected
