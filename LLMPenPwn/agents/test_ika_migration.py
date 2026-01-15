#!/usr/bin/env python3
"""
Test script to verify Ika-Core migration is working.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from base import BaseSystem
from env import HostEnvironment
from tools import get_recon_toolset, get_core_toolset

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from IkaCore.agents import IkaBaseAgent
        from IkaCore.tools import IkaTools
        from ika_tool_adapter import function_to_ika_tool
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_tool_conversion():
    """Test that tools are converted correctly."""
    print("\nTesting tool conversion...")
    try:
        tools = get_recon_toolset()
        print(f"✓ Converted {len(tools)} tools to IkaTools")
        for tool in tools[:3]:
            print(f"  - {tool.name}: {tool.description[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Tool conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_base_system():
    """Test BaseSystem creation."""
    print("\nTesting BaseSystem creation...")
    try:
        env = HostEnvironment()
        tools = get_core_toolset()[:2]
        
        system = BaseSystem.from_model_and_tools(
            model_id="claude-3-5-sonnet-20241022",
            api_key="test-key-not-used",
            tools=tools,
            env=env,
            max_steps=10,
            name="test_agent",
            description="Test agent for migration verification",
            role="tester",
            system_prompt="You are a test agent."
        )
        
        print(f"✓ BaseSystem created successfully")
        print(f"  - Agent name: {system.agent.name}")
        print(f"  - Agent role: {system.agent.role}")
        print(f"  - Number of tools: {len(system.agent.tools)}")
        print(f"  - Max steps: {system.agent.maxsteps}")
        return True
    except Exception as e:
        print(f"✗ BaseSystem creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Ika-Core Migration Verification Test")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Tool Conversion", test_tool_conversion()))
    results.append(("BaseSystem Creation", test_base_system()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✓ All tests passed! Migration appears successful.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
