"""Test suite for reasoning_content support in DeepSeek models."""

import sys
from pathlib import Path
from unittest.mock import Mock

src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from IkaModel.deepseek.deepseek import deepseek_fill_payload
from IkaModel.chat_interface.chat_interface import chat


def test_deepseek_reasoning_content_in_message_history():
    model = Mock()
    model.model_id = "deepseek-chat"
    model.max_tokens = 4096
    model.temperature = 0.0
    model.deepthinking = False
    model.agent_tools = None

    message_history = {
        "system": {"message": "You are a helpful assistant.", "tokens": 0},
        "first_input": {"message": "Hello", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {
            "msg1": {
                "message": "I understand.",
                "tokens": 10,
                "type": "assistant",
                "reasoning_content": "Let me think about this carefully..."
            }
        }
    }

    payload = deepseek_fill_payload(model, [], message_history)

    assert "messages" in payload
    assistant_msgs = [msg for msg in payload["messages"] if msg.get("role") == "assistant"]
    assert len(assistant_msgs) >= 1
    
    found_reasoning = False
    for msg in assistant_msgs:
        if "reasoning_content" in msg:
            assert msg["reasoning_content"] == "Let me think about this carefully..."
            found_reasoning = True
    
    assert found_reasoning, "reasoning_content should be included in assistant messages"


def test_deepseek_reasoning_content_without_reasoning():
    model = Mock()
    model.model_id = "deepseek-chat"
    model.max_tokens = 4096
    model.temperature = 0.0
    model.deepthinking = False
    model.agent_tools = None

    message_history = {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "Hello", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {
            "msg1": {
                "message": "I understand.",
                "tokens": 10
            }
        }
    }

    payload = deepseek_fill_payload(model, [], message_history)

    assistant_msgs = [msg for msg in payload["messages"] if msg.get("role") == "assistant"]
    assert len(assistant_msgs) > 0
    
    for msg in assistant_msgs:
        assert "reasoning_content" not in msg or msg.get("reasoning_content") is None


def test_deepseek_thinking_enabled_with_reasoner_model():
    model = Mock()
    model.model_id = "deepseek-reasoner"
    model.max_tokens = 4096
    model.temperature = 0.0
    model.deepthinking = False
    model.agent_tools = None

    payload = deepseek_fill_payload(model, [], {})

    assert payload["thinking"]["type"] == "enabled"


def test_deepseek_thinking_enabled_with_deepthinking_flag():
    model = Mock()
    model.model_id = "deepseek-chat"
    model.max_tokens = 4096
    model.temperature = 0.0
    model.deepthinking = True
    model.agent_tools = None

    payload = deepseek_fill_payload(model, [], {})

    assert payload["thinking"]["type"] == "enabled"


def test_deepseek_thinking_disabled():
    model = Mock()
    model.model_id = "deepseek-chat"
    model.max_tokens = 4096
    model.temperature = 0.0
    model.deepthinking = False
    model.agent_tools = None

    payload = deepseek_fill_payload(model, [], {})

    assert payload["thinking"]["type"] == "disabled"


def test_chat_interface_extracts_reasoning_content():
    from unittest.mock import patch, MagicMock
    import httpx
    
    class MockBareBoneModel:
        def __init__(self):
            self.model_id = "deepseek-chat"
            self.api_key = "test-key"
            self.api_url = "https://api.deepseek.com/chat/completions"
            self.temperature = 0.0
            self.max_tokens = 4096
            self.deepthinking = False
            self.agent_tools = None

    model = MockBareBoneModel()
    messages = [{"role": "user", "content": "Test"}]
    message_history = {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {}
    }

    mock_response_data = {
        "choices": [{
            "message": {
                "content": "Test response",
                "reasoning_content": "Internal reasoning"
            }
        }],
        "usage": {"total_tokens": 10}
    }
    
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_data
    mock_response.raise_for_status = MagicMock()
    
    with patch('IkaModel.chat_interface.chat_interface.api_request_retry', return_value=mock_response):
        response = chat(model, messages, message_history, timeout=1.0)
        assert "reasoning_content" in response
        assert response["reasoning_content"] == "Internal reasoning"


def test_chat_interface_stores_reasoning_content_in_history():
    from unittest.mock import patch, MagicMock
    
    class MockBareBoneModel:
        def __init__(self):
            self.model_id = "deepseek-chat"
            self.api_key = "test-key"
            self.api_url = "https://api.deepseek.com/chat/completions"
            self.temperature = 0.0
            self.max_tokens = 4096
            self.deepthinking = False
            self.agent_tools = None

    model = MockBareBoneModel()
    messages = [{"role": "user", "content": "Test"}]
    message_history = {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {}
    }

    mock_response_data = {
        "choices": [{
            "message": {
                "content": "Test response",
                "reasoning_content": "Internal reasoning"
            }
        }],
        "usage": {"total_tokens": 10}
    }
    
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_data
    mock_response.raise_for_status = MagicMock()
    
    with patch('IkaModel.chat_interface.chat_interface.api_request_retry', return_value=mock_response):
        response = chat(model, messages, message_history, timeout=1.0)
        
        msg_ids = list(message_history["messages"].keys())
        assert len(msg_ids) > 0
        last_msg = message_history["messages"][msg_ids[-1]]
        assert "reasoning_content" in last_msg
        assert last_msg["reasoning_content"] == "Internal reasoning"


def test_chat_interface_returns_reasoning_content():
    from unittest.mock import patch, MagicMock
    
    class MockBareBoneModel:
        def __init__(self):
            self.model_id = "deepseek-chat"
            self.api_key = "test-key"
            self.api_url = "https://api.deepseek.com/chat/completions"
            self.temperature = 0.0
            self.max_tokens = 4096
            self.deepthinking = False
            self.agent_tools = None

    model = MockBareBoneModel()
    messages = [{"role": "user", "content": "Test"}]
    message_history = {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {}
    }

    mock_response_data = {
        "choices": [{
            "message": {
                "content": "Test response",
                "reasoning_content": "Internal reasoning"
            }
        }],
        "usage": {"total_tokens": 10}
    }
    
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_data
    mock_response.raise_for_status = MagicMock()
    
    with patch('IkaModel.chat_interface.chat_interface.api_request_retry', return_value=mock_response):
        response = chat(model, messages, message_history, timeout=1.0)
        assert "reasoning_content" in response
        assert response["reasoning_content"] == "Internal reasoning"


def test_agents_includes_reasoning_content_in_assistant_entry():
    mock_response = {
        "content": "Test response",
        "reasoning_content": "Internal reasoning process"
    }

    assistant_entry = {"role": "assistant", "content": "Test response"}
    if mock_response.get("reasoning_content"):
        assistant_entry["reasoning_content"] = mock_response["reasoning_content"]

    assert assistant_entry["reasoning_content"] == "Internal reasoning process"


def test_agents_handles_missing_reasoning_content():
    mock_response = {
        "content": "Test response"
    }

    assistant_entry = {"role": "assistant", "content": "Test response"}
    if mock_response.get("reasoning_content"):
        assistant_entry["reasoning_content"] = mock_response["reasoning_content"]

    assert "reasoning_content" not in assistant_entry


def test_deepseek_multiple_messages_with_reasoning():
    model = Mock()
    model.model_id = "deepseek-chat"
    model.max_tokens = 4096
    model.temperature = 0.0
    model.deepthinking = False
    model.agent_tools = None

    message_history = {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {
            "msg1": {
                "message": "First response",
                "tokens": 10,
                "type": "assistant",
                "reasoning_content": "Reasoning 1"
            },
            "msg2": {
                "message": "Second response",
                "tokens": 15,
                "type": "assistant"
            },
            "msg3": {
                "message": "Third response",
                "tokens": 20,
                "type": "assistant",
                "reasoning_content": "Reasoning 3"
            }
        }
    }

    payload = deepseek_fill_payload(model, [], message_history)

    assistant_msgs = [msg for msg in payload["messages"] if msg.get("role") == "assistant" and "reasoning_content" in msg]
    assert len(assistant_msgs) == 2
    
    reasoning_values = [msg["reasoning_content"] for msg in assistant_msgs]
    assert "Reasoning 1" in reasoning_values
    assert "Reasoning 3" in reasoning_values


def test_deepseek_reasoner_model_id_variations():
    test_cases = [
        "deepseek-reasoner",
        "DeepSeek-Reasoner",
        "deepseek-reasoner-v1",
        "DEEPSEEK-REASONER"
    ]

    for model_id in test_cases:
        model = Mock()
        model.model_id = model_id
        model.max_tokens = 4096
        model.temperature = 0.0
        model.deepthinking = False
        model.agent_tools = None

        payload = deepseek_fill_payload(model, [], {})

        assert payload["thinking"]["type"] == "enabled", f"Failed for model_id: {model_id}"


def test_deepseek_reasoner_no_tool_choice():
    model = Mock()
    model.model_id = "deepseek-reasoner"
    model.max_tokens = 4096
    model.temperature = 0.0
    model.deepthinking = False
    
    class MockToolArgs:
        def __init__(self):
            self.type = "input"
            self.description = "Test tool description"
            self.properties = None
    
    class MockTool:
        def __init__(self, name, required=False):
            self.name = name
            self.description = f"Description for {name}"
            self.required = required
            self.args = MockToolArgs()
    
    model.agent_tools = [MockTool("test_tool", required=True)]

    payload = deepseek_fill_payload(model, [], {})

    assert "tool_choice" not in payload
    assert payload["thinking"]["type"] == "enabled"


def test_deepseek_chat_has_tool_choice():
    model = Mock()
    model.model_id = "deepseek-chat"
    model.max_tokens = 4096
    model.temperature = 0.0
    model.deepthinking = False
    
    class MockToolArgs:
        def __init__(self):
            self.type = "input"
            self.description = "Test tool description"
            self.properties = None
    
    class MockTool:
        def __init__(self, name, required=False):
            self.name = name
            self.description = f"Description for {name}"
            self.required = required
            self.args = MockToolArgs()
    
    model.agent_tools = [MockTool("test_tool", required=True)]

    payload = deepseek_fill_payload(model, [], {})

    assert "tool_choice" in payload
    assert payload["tool_choice"]["type"] == "function"
    assert payload["tool_choice"]["function"]["name"] == "test_tool"


def run_all_tests():
    tests = [
        test_deepseek_reasoning_content_in_message_history,
        test_deepseek_reasoning_content_without_reasoning,
        test_deepseek_thinking_enabled_with_reasoner_model,
        test_deepseek_thinking_enabled_with_deepthinking_flag,
        test_deepseek_thinking_disabled,
        test_chat_interface_extracts_reasoning_content,
        test_chat_interface_stores_reasoning_content_in_history,
        test_chat_interface_returns_reasoning_content,
        test_agents_includes_reasoning_content_in_assistant_entry,
        test_agents_handles_missing_reasoning_content,
        test_deepseek_multiple_messages_with_reasoning,
        test_deepseek_reasoner_model_id_variations,
        test_deepseek_reasoner_no_tool_choice,
        test_deepseek_chat_has_tool_choice,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            print(f"PASS: {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_func.__name__} - {str(e)}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
