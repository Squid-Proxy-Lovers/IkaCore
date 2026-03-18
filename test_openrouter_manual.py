"""Manual integration test for OpenRouter API.

This script demonstrates how to use OpenRouter with IkaCore.
Replace the api_key with your actual OpenRouter API key to test.
"""
import sys
from pathlib import Path

src = Path(__file__).parent / "src"
sys.path.insert(0, str(src))

from IkaModel.base import BareBoneModel
from IkaModel.request_interface import get_provider


def test_provider_detection():
    """Test that OpenRouter models are correctly detected."""
    print("Testing provider detection...")

    test_cases = [
        ("meta-llama/llama-3.1-70b-instruct", "openrouter"),
        ("qwen/qwen-2.5-72b-instruct", "openrouter"),
        ("mistralai/mistral-7b-instruct", "openrouter"),
        ("gpt-4o", "openai"),
        ("claude-3-sonnet", "anthropic"),
    ]

    for model_id, expected_provider in test_cases:
        detected = get_provider(model_id)
        status = "✓" if detected == expected_provider else "✗"
        print(f"  {status} {model_id}: {detected} (expected: {expected_provider})")


def test_payload_construction():
    """Test that payload is constructed correctly for OpenRouter."""
    print("\nTesting payload construction...")

    from IkaModel.openrouter import openrouter_fill_payload

    model = BareBoneModel(
        model_id="meta-llama/llama-3.1-70b-instruct",
        api_key="test-key",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        max_tokens=1000,
        temperature=0.7,
        suppress_init_output=True
    )

    messages = [{"role": "user", "content": "Hello, world!"}]
    payload = openrouter_fill_payload(model, messages, None)

    print(f"  ✓ Model ID: {payload['model']}")
    print(f"  ✓ Max tokens: {payload.get('max_tokens', payload.get('max_completion_tokens'))}")
    print(f"  ✓ Temperature: {payload.get('temperature', 'not set')}")
    print(f"  ✓ Messages: {len(payload['messages'])} message(s)")


def test_request_building():
    """Test that request is built correctly with headers."""
    print("\nTesting request building...")

    from IkaModel.chat_helpers_openrouter import build_openrouter_request

    model = BareBoneModel(
        model_id="meta-llama/llama-3.1-70b-instruct",
        api_key="sk-or-v1-test-key",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        suppress_init_output=True
    )

    # Add optional headers
    model.http_referer = "https://ikacore.example.com"
    model.x_title = "IkaCore Test"

    messages = [{"role": "user", "content": "Test"}]
    history = {
        "system": {"message": ""},
        "first_input": {"message": ""},
        "summary": {"message": ""},
        "messages": {}
    }

    api_url, headers, payload = build_openrouter_request(model, messages, history)

    print(f"  ✓ API URL: {api_url}")
    print(f"  ✓ Authorization header: {headers['Authorization'][:20]}...")
    print(f"  ✓ Content-Type: {headers['Content-Type']}")
    print(f"  ✓ HTTP-Referer: {headers.get('HTTP-Referer', 'not set')}")
    print(f"  ✓ X-Title: {headers.get('X-Title', 'not set')}")


def test_response_parsing():
    """Test that OpenRouter responses are parsed correctly."""
    print("\nTesting response parsing...")

    from IkaModel.chat_helpers_openrouter import parse_openrouter_response

    # Simulate OpenRouter response
    response_data = {
        "id": "gen-test-123",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "This is a test response from OpenRouter."
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
            "cost": 0.000012
        }
    }

    content, reasoning, tool_calls, tokens = parse_openrouter_response(
        response_data, "meta-llama/llama-3.1-70b-instruct"
    )

    print(f"  ✓ Content: {content}")
    print(f"  ✓ Reasoning: {reasoning}")
    print(f"  ✓ Tool calls: {len(tool_calls)}")
    print(f"  ✓ Total tokens: {tokens}")


def test_live_api_call():
    """
    Test a live API call to OpenRouter.

    NOTE: This requires a valid OpenRouter API key.
    Set OPENROUTER_API_KEY environment variable or update the code below.
    """
    print("\nTesting live API call...")
    print("  ⚠ Skipped - requires valid API key")
    print("  To test live API calls:")
    print("    1. Get an API key from https://openrouter.ai/")
    print("    2. Create an agent with model_id='meta-llama/llama-3.1-70b-instruct'")
    print("    3. Set api_url='https://openrouter.ai/api/v1/chat/completions'")
    print("    4. Call agent.execution(user_input='Hello, world!')")


if __name__ == "__main__":
    print("=" * 60)
    print("OpenRouter Integration Manual Tests")
    print("=" * 60)

    test_provider_detection()
    test_payload_construction()
    test_request_building()
    test_response_parsing()
    test_live_api_call()

    print("\n" + "=" * 60)
    print("All manual tests completed!")
    print("=" * 60)
