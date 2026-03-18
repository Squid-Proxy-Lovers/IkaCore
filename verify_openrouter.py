#!/usr/bin/env python3
"""
Quick verification script for OpenRouter integration.
Run this to verify the integration is working correctly.
"""

import sys
from pathlib import Path

src = Path(__file__).parent / "src"
sys.path.insert(0, str(src))


def verify_imports():
    """Verify all OpenRouter modules can be imported."""
    print("1. Verifying imports...")
    try:
        from IkaModel.openrouter import openrouter_fill_payload
        from IkaModel.chat_helpers_openrouter import (
            build_openrouter_request,
            parse_openrouter_response,
            append_openrouter_tool_messages
        )
        from IkaModel.request_interface import get_provider
        from IkaModel.chat_helpers_common import (
            build_provider_request,
            parse_provider_response,
            append_provider_tool_messages
        )
        print("   ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False


def verify_provider_detection():
    """Verify provider detection works."""
    print("\n2. Verifying provider detection...")
    from IkaModel.request_interface import get_provider

    tests = [
        ("meta-llama/llama-3.1-70b-instruct", "openrouter"),
        ("qwen/qwen-2.5-72b-instruct", "openrouter"),
        ("mistralai/mistral-7b-instruct", "openrouter"),
        ("gpt-4o", "openai"),
    ]

    all_passed = True
    for model_id, expected in tests:
        result = get_provider(model_id)
        if result == expected:
            print(f"   ✅ {model_id} → {result}")
        else:
            print(f"   ❌ {model_id} → {result} (expected {expected})")
            all_passed = False

    return all_passed


def verify_payload_construction():
    """Verify payload construction works."""
    print("\n3. Verifying payload construction...")
    from IkaModel.base import BareBoneModel
    from IkaModel.openrouter import openrouter_fill_payload

    try:
        model = BareBoneModel(
            model_id="meta-llama/llama-3.1-70b-instruct",
            api_key="test-key",
            api_url="https://openrouter.ai/api/v1/chat/completions",
            max_tokens=1000,
            temperature=0.7,
            suppress_init_output=True
        )

        messages = [{"role": "user", "content": "Test"}]
        payload = openrouter_fill_payload(model, messages, None)

        checks = [
            ("model" in payload, "model field present"),
            ("messages" in payload, "messages field present"),
            ("max_tokens" in payload or "max_completion_tokens" in payload, "max_tokens field present"),
            (payload["model"] == "meta-llama/llama-3.1-70b-instruct", "correct model ID"),
        ]

        all_passed = True
        for check, description in checks:
            if check:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description}")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"   ❌ Payload construction failed: {e}")
        return False


def verify_request_building():
    """Verify request building works."""
    print("\n4. Verifying request building...")
    from IkaModel.base import BareBoneModel
    from IkaModel.chat_helpers_openrouter import build_openrouter_request

    try:
        model = BareBoneModel(
            model_id="meta-llama/llama-3.1-70b-instruct",
            api_key="sk-or-test-key",
            api_url="https://openrouter.ai/api/v1/chat/completions",
            suppress_init_output=True
        )

        messages = [{"role": "user", "content": "Test"}]
        history = {
            "system": {"message": ""},
            "first_input": {"message": ""},
            "summary": {"message": ""},
            "messages": {}
        }

        api_url, headers, payload = build_openrouter_request(model, messages, history)

        checks = [
            (api_url == "https://openrouter.ai/api/v1/chat/completions", "correct API URL"),
            ("Authorization" in headers, "Authorization header present"),
            ("Content-Type" in headers, "Content-Type header present"),
            (headers["Authorization"].startswith("Bearer "), "Bearer token format"),
            ("model" in payload, "payload has model"),
        ]

        all_passed = True
        for check, description in checks:
            if check:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description}")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"   ❌ Request building failed: {e}")
        return False


def verify_response_parsing():
    """Verify response parsing works."""
    print("\n5. Verifying response parsing...")
    from IkaModel.chat_helpers_openrouter import parse_openrouter_response

    try:
        response_data = {
            "id": "gen-test",
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.000012
            }
        }

        content, reasoning, tool_calls, tokens = parse_openrouter_response(
            response_data, "meta-llama/llama-3.1-70b-instruct"
        )

        checks = [
            (content == "Test response", "content extracted"),
            (reasoning is None, "reasoning is None"),
            (tool_calls == [], "tool_calls is empty list"),
            (tokens == 15, "tokens counted"),
        ]

        all_passed = True
        for check, description in checks:
            if check:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description}")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"   ❌ Response parsing failed: {e}")
        return False


def verify_provider_routing():
    """Verify provider routing works."""
    print("\n6. Verifying provider routing...")
    from IkaModel.base import BareBoneModel
    from IkaModel.chat_helpers_common import build_provider_request

    try:
        model = BareBoneModel(
            model_id="meta-llama/llama-3.1-70b-instruct",
            api_key="test-key",
            api_url="https://openrouter.ai/api/v1/chat/completions",
            suppress_init_output=True
        )

        messages = [{"role": "user", "content": "Test"}]
        history = {
            "system": {"message": ""},
            "first_input": {"message": ""},
            "summary": {"message": ""},
            "messages": {}
        }

        api_url, headers, payload = build_provider_request(
            "openrouter", model, messages, history
        )

        checks = [
            (api_url is not None, "API URL returned"),
            (headers is not None, "headers returned"),
            (payload is not None, "payload returned"),
            ("Authorization" in headers, "Authorization header present"),
        ]

        all_passed = True
        for check, description in checks:
            if check:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description}")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"   ❌ Provider routing failed: {e}")
        return False


def main():
    print("=" * 70)
    print("OpenRouter Integration Verification")
    print("=" * 70)

    results = []

    results.append(("Imports", verify_imports()))
    results.append(("Provider Detection", verify_provider_detection()))
    results.append(("Payload Construction", verify_payload_construction()))
    results.append(("Request Building", verify_request_building()))
    results.append(("Response Parsing", verify_response_parsing()))
    results.append(("Provider Routing", verify_provider_routing()))

    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 All verification checks passed!")
        print("\nOpenRouter integration is ready to use.")
        print("\nNext steps:")
        print("  1. Get an API key from https://openrouter.ai/keys")
        print("  2. Update openrouter.json with your API key")
        print("  3. Run: python3 examples/openrouter_usage.py")
        return 0
    else:
        print("\n❌ Some verification checks failed.")
        print("Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
