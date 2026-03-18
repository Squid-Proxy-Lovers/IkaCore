"""
OpenRouter API Integration - Usage Examples

This file demonstrates how to use OpenRouter with IkaCore agents.
OpenRouter provides unified access to multiple LLM providers including:
- Meta's Llama models
- Anthropic's Claude (via OpenRouter)
- Mistral models
- Qwen models
- And many more

For API keys and pricing: https://openrouter.ai/
"""

import sys
from pathlib import Path

# Add src to path
src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src))

from IkaCore.agents import IkaBaseAgent
from IkaModel.base import AgentTool, ToolArgs


# Example 1: Basic OpenRouter Usage
def example_basic_usage():
    """Simple agent using OpenRouter with Llama model."""
    print("\n=== Example 1: Basic OpenRouter Usage ===\n")

    agent = IkaBaseAgent(
        name="OpenRouterAgent",
        description="A test agent using OpenRouter",
        prompt="You are a helpful AI assistant.",
        model_id="meta-llama/llama-3.1-70b-instruct",  # OpenRouter model format: provider/model
        api_key="sk-or-v1-YOUR-KEY-HERE",  # Replace with your OpenRouter API key
        api_url="https://openrouter.ai/api/v1/chat/completions",
        max_tokens=500,
        temperature=0.7
    )

    # The provider will be auto-detected as "openrouter" from the "/" in model_id
    print(f"Agent created with model: {agent.model_id}")
    print(f"Provider auto-detected as: openrouter")


# Example 2: Using Different OpenRouter Models
def example_different_models():
    """Examples of different OpenRouter model IDs."""
    print("\n=== Example 2: Different OpenRouter Models ===\n")

    models = [
        "meta-llama/llama-3.1-70b-instruct",      # Llama 3.1 70B
        "meta-llama/llama-3.1-8b-instruct",       # Llama 3.1 8B (cheaper)
        "qwen/qwen-2.5-72b-instruct",             # Qwen 2.5 72B
        "mistralai/mistral-7b-instruct",          # Mistral 7B
        "google/gemma-2-9b-it",                   # Gemma 2 9B
    ]

    for model_id in models:
        print(f"  - {model_id}")

    print("\nSee https://openrouter.ai/models for full list")


# Example 3: Using OpenRouter with Tools
def example_with_tools():
    """Agent with custom tools using OpenRouter."""
    print("\n=== Example 3: OpenRouter with Tools ===\n")

    # Define a custom tool
    search_tool = AgentTool(
        id="search_web",
        name="search_web",
        description="Search the web for information",
        args=ToolArgs(
            type="input",
            description="Search query to look up"
        ),
        required=False
    )

    agent_end = AgentTool(
        id="agent_end",
        name="agent_end",
        description="End task with final answer",
        args=ToolArgs(
            type="input",
            description="Final response to user"
        ),
        required=True
    )

    agent = IkaBaseAgent(
        name="ToolAgent",
        description="Agent with tools",
        prompt="You are a helpful assistant. Use tools when needed.",
        model_id="meta-llama/llama-3.1-70b-instruct",
        api_key="sk-or-v1-YOUR-KEY-HERE",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        max_tokens=1000,
        temperature=0.0
    )

    agent.agent_tools = [search_tool, agent_end]
    print(f"Agent created with {len(agent.agent_tools)} tools")


# Example 4: OpenRouter with Optional Headers
def example_with_headers():
    """Using OpenRouter attribution headers."""
    print("\n=== Example 4: OpenRouter Attribution Headers ===\n")

    agent = IkaBaseAgent(
        name="AttributedAgent",
        description="Agent with attribution headers",
        prompt="You are a helpful assistant.",
        model_id="meta-llama/llama-3.1-70b-instruct",
        api_key="sk-or-v1-YOUR-KEY-HERE",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        max_tokens=500,
        temperature=0.7
    )

    # Add optional attribution headers for ranking on OpenRouter
    agent.http_referer = "https://your-app.com"  # Your app URL
    agent.x_title = "IkaCore Agent"              # Your app name

    print("Optional headers set for OpenRouter:")
    print(f"  HTTP-Referer: {agent.http_referer}")
    print(f"  X-Title: {agent.x_title}")


# Example 5: OpenRouter with Plugins (Advanced)
def example_with_plugins():
    """Using OpenRouter-specific plugins."""
    print("\n=== Example 5: OpenRouter Plugins (Advanced) ===\n")

    agent = IkaBaseAgent(
        name="PluginAgent",
        description="Agent with OpenRouter plugins",
        prompt="You are a helpful assistant with web access.",
        model_id="meta-llama/llama-3.1-70b-instruct",
        api_key="sk-or-v1-YOUR-KEY-HERE",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        max_tokens=1000,
        temperature=0.7
    )

    # OpenRouter-specific plugins (if supported by the model)
    agent.openrouter_plugins = ["web_search", "code_execution"]

    print("OpenRouter plugins enabled:")
    print(f"  Plugins: {agent.openrouter_plugins}")
    print("\nNote: Plugin support varies by model. Check OpenRouter docs.")


# Example 6: Cost Tracking
def example_cost_tracking():
    """OpenRouter provides cost tracking in responses."""
    print("\n=== Example 6: Cost Tracking ===\n")

    print("OpenRouter includes cost information in API responses:")
    print("  - The 'usage' field contains a 'cost' value in USD")
    print("  - IkaCore logs this automatically at DEBUG level")
    print("  - Example: 'OpenRouter request cost: $0.000012'")
    print("\nTo see cost tracking, enable DEBUG logging:")
    print("  import logging")
    print("  logging.basicConfig(level=logging.DEBUG)")


# Example 7: Using OpenRouter Config File
def example_config_file():
    """Loading OpenRouter configuration from JSON."""
    print("\n=== Example 7: Configuration File ===\n")

    print("You can store OpenRouter config in openrouter.json:")
    print("""
{
  "api_key": "sk-or-v1-your-key",
  "default_model": "meta-llama/llama-3.1-70b-instruct",
  "http_referer": "https://your-app.com",
  "x_title": "IkaCore Agent",
  "models": {
    "meta-llama/llama-3.1-70b-instruct": {
      "max_tokens": 131072,
      "temperature": 0.7
    },
    "qwen/qwen-2.5-72b-instruct": {
      "max_tokens": 131072,
      "temperature": 0.0
    }
  }
}
    """)


if __name__ == "__main__":
    print("=" * 70)
    print("OpenRouter Integration - Usage Examples")
    print("=" * 70)

    example_basic_usage()
    example_different_models()
    example_with_tools()
    example_with_headers()
    example_with_plugins()
    example_cost_tracking()
    example_config_file()

    print("\n" + "=" * 70)
    print("For more information:")
    print("  - OpenRouter docs: https://openrouter.ai/docs")
    print("  - OpenRouter models: https://openrouter.ai/models")
    print("  - OpenRouter pricing: https://openrouter.ai/pricing")
    print("=" * 70)
