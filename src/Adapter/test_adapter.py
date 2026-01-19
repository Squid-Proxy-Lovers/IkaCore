"""Test suite for IkaCore adapter functionality."""

import json
import sys
from pathlib import Path
from typing import Dict, Any

src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from Adapter.adapter import (
    tool,
    function_to_ika_tool,
    collect_tools_from_module,
    make_delegate_tool,
    make_delegate_tools_from_agents
)
from IkaCore.tools import IkaTools


def test_tool_decorator():
    @tool
    def simple_tool():
        return "success"
    
    assert hasattr(simple_tool, "_ika_tool")
    assert simple_tool._ika_tool is True


def test_function_to_ika_tool_basic():
    def add_numbers(a: int, b: int) -> int:
        return a + b
    
    ika_tool = function_to_ika_tool(add_numbers)
    
    assert isinstance(ika_tool, IkaTools)
    assert ika_tool.name == "add_numbers"
    assert "Add two numbers together" in ika_tool.description
    assert "a" in ika_tool.parameters
    assert "b" in ika_tool.parameters
    
    result = ika_tool.execute({"a": 5, "b": 3})
    assert result == "8"


def test_function_to_ika_tool_with_defaults():
    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"
    
    ika_tool = function_to_ika_tool(greet)
    
    result1 = ika_tool.execute({"name": "Alice"})
    assert "Hello, Alice!" in result1
    
    result2 = ika_tool.execute({"name": "Bob", "greeting": "Hi"})
    assert "Hi, Bob!" in result2


def test_function_to_ika_tool_no_params():
    @tool
    def get_timestamp() -> str:
        return "2024-01-01T00:00:00"
    
    ika_tool = function_to_ika_tool(get_timestamp)
    
    result = ika_tool.execute({})
    assert "2024-01-01" in result


def test_function_to_ika_tool_dict_return():
    def get_info(name: str) -> Dict[str, Any]:
        return {"name": name, "status": "active"}
    
    ika_tool = function_to_ika_tool(get_info)
    
    result = ika_tool.execute({"name": "test"})
    parsed = json.loads(result)
    assert parsed["name"] == "test"
    assert parsed["status"] == "active"


def test_function_to_ika_tool_error_handling():
    def failing_tool(x: int) -> int:
        if x < 0:
            raise ValueError("Negative numbers not allowed")
        return x * 2
    
    ika_tool = function_to_ika_tool(failing_tool)
    
    result = ika_tool.execute({"x": -5})
    parsed = json.loads(result)
    assert "error" in parsed
    assert "Negative" in parsed["error"]


def test_collect_tools_from_module():
    class TestModule:
        @tool
        def tool1(self):
            return "tool1"
        
        @tool
        def tool2(self, x: str):
            return f"tool2: {x}"
        
        def not_a_tool(self):
            return "not a tool"
    
    test_mod = TestModule()
    tools = collect_tools_from_module(test_mod)
    
    assert len(tools) == 2
    assert any(t.name == "tool1" for t in tools)
    assert any(t.name == "tool2" for t in tools)


def test_make_delegate_tool():
    class MockAgent:
        def __init__(self, name: str):
            self.name = name
            self.message_history = {"first_input": {"message": ""}}
            self.prompt = ""
        
        def execution(self):
            return {"final_message": f"Agent {self.name} completed task: {self.prompt}"}
    
    agent = MockAgent("test_agent")
    delegate_tool = make_delegate_tool("test_agent", agent, "Test agent tool")
    
    assert isinstance(delegate_tool, IkaTools)
    assert delegate_tool.name == "test_agent"
    
    result = delegate_tool.execute({"input": "Do something"})
    assert "test_agent" in result
    assert "Do something" in result


def test_make_delegate_tools_from_agents():
    class MockAgent:
        def __init__(self, name: str):
            self.name = name
            self.message_history = {"first_input": {"message": ""}}
            self.prompt = ""
        
        def execution(self):
            return {"final_message": f"Agent {self.name} completed"}
    
    agents = {
        "agent1": MockAgent("agent1"),
        "agent2": MockAgent("agent2"),
        "manager_agent": MockAgent("manager_agent"),
    }
    
    tools = make_delegate_tools_from_agents(agents)
    
    assert len(tools) == 2
    assert any(t.name == "agent1" for t in tools)
    assert any(t.name == "agent2" for t in tools)
    assert not any(t.name == "manager_agent" for t in tools)


def test_make_delegate_tools_with_include_exclude():
    class MockAgent:
        def __init__(self, name: str):
            self.name = name
            self.message_history = {"first_input": {"message": ""}}
            self.prompt = ""
        
        def execution(self):
            return {"final_message": f"Agent {self.name} completed"}
    
    agents = {
        "agent1": MockAgent("agent1"),
        "agent2": MockAgent("agent2"),
        "agent3": MockAgent("agent3"),
    }
    
    tools_include = make_delegate_tools_from_agents(agents, include=["agent1", "agent2"])
    assert len(tools_include) == 2
    
    tools_exclude = make_delegate_tools_from_agents(agents, exclude=["agent2"])
    assert len(tools_exclude) == 2
    assert not any(t.name == "agent2" for t in tools_exclude)


def run_all_tests():
    tests = [
        test_tool_decorator,
        test_function_to_ika_tool_basic,
        test_function_to_ika_tool_with_defaults,
        test_function_to_ika_tool_no_params,
        test_function_to_ika_tool_dict_return,
        test_function_to_ika_tool_error_handling,
        test_collect_tools_from_module,
        test_make_delegate_tool,
        test_make_delegate_tools_from_agents,
        test_make_delegate_tools_with_include_exclude,
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
