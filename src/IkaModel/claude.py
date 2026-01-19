import json
from typing import Any, Dict, List, Optional


def anthropic_fill_payload(model, messages: List[Dict[str, Any]], message_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    message_history = message_history or {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {}
    }
    api_messages = []
    system_prompt = message_history["system"]["message"] or model.system_prompt or ""
    
    if message_history["first_input"]["message"]:
        api_messages.append({
            "role": "user",
            "content": message_history["first_input"]["message"]
        })
    
    if message_history["summary"]["message"]:
        api_messages.append({
            "role": "assistant",
            "content": message_history["summary"]["message"]
        })
    
    for msg_id in message_history["messages"]:
        msg = message_history["messages"][msg_id]
        msg_type = msg.get("type", "assistant")
        if msg_type == "assistant_with_tools" or msg_type == "tool":
            continue
        else:
            api_messages.append({
                "role": "assistant",
                "content": msg["message"]
            })
    
    for msg in messages:
        if isinstance(msg, dict):
            if "role" in msg and "content" in msg:
                if msg["role"] == "user":
                    api_messages.append({
                        "role": "user",
                        "content": msg["content"]
                    })
                elif msg["role"] == "assistant":
                    api_messages.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })
            elif "content" in msg:
                api_messages.append({"role": "user", "content": msg["content"]})
            else:
                api_messages.append({"role": "user", "content": str(msg)})
        else:
            api_messages.append({"role": "user", "content": str(msg)})
    
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    
    # Cap max_tokens for models with lower limits
    model_id_lower = model.model_id.lower()
    if "haiku" in model_id_lower:
        # Claude Haiku has a max of 4096 tokens
        if max_tokens_value > 4096:
            max_tokens_value = 4096
    
    payload = {
        "model": model.model_id,
        "max_tokens": max_tokens_value,
        "temperature": model.temperature,
        "system": system_prompt,
        "messages": api_messages
    }
    
    # Add parallel tool use prompt for Claude 4 models if enabled
    if hasattr(model, 'parallel_tool_calls') and model.parallel_tool_calls:
        if "opus-4" in model_id_lower or "sonnet-4" in model_id_lower or "claude-4" in model_id_lower:
            parallel_prompt = "\n\n<use_parallel_tool_calls>\nFor maximum efficiency, whenever you perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially. Prioritize calling tools in parallel whenever possible. For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into context at the same time. When running multiple read-only commands like `ls` or `list_dir`, always run all of the commands in parallel. Err on the side of maximizing parallel tool calls rather than running too many tools sequentially.\n</use_parallel_tool_calls>"
            current_system = payload.get("system", "")
            if parallel_prompt not in current_system:
                payload["system"] = current_system + parallel_prompt
    
    if system_prompt:
        payload["system"] = system_prompt
    
    if model.agent_tools:
        tools = []
        for tool in model.agent_tools:
            if tool.name == "agent_end" and tool.args.type == "input":
                input_schema = {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": tool.args.description or "Final response content. This is REQUIRED - provide your complete final answer here."
                        }
                    },
                    "required": ["input"]
                }
            elif tool.args.type == "input":
                input_schema = {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": tool.args.description or f"Input for {tool.name}"
                        }
                    },
                    "required": ["input"]
                }
            elif tool.args.properties and len(tool.args.properties) > 0:
                required_list = tool.args.properties.pop("__required__", [])
                input_schema = {
                    "type": "object",
                    "properties": tool.args.properties,
                    "required": required_list
                }
            else:
                arg_name = tool.args.type
                json_type = "string"
                if arg_name in ["stage_index"]:
                    json_type = "integer"
                elif arg_name == "input":
                    json_type = "string"
                
                input_schema = {
                    "type": "object",
                    "properties": {
                        arg_name: {
                            "type": json_type,
                            "description": tool.args.description
                        }
                    },
                    "required": []
                }

            tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": input_schema
            })
        payload["tools"] = tools

        # Set tool_choice with optional parallel tool use control
        # disable_parallel_tool_use must be inside tool_choice, not at top level
        required_tools = [t for t in model.agent_tools if t.required]
        if len(required_tools) == 1:
            tool_choice = {"type": "tool", "name": required_tools[0].name}
        elif len(required_tools) > 1:
            tool_choice = {"type": "required"}
        else:
            tool_choice = {"type": "auto"}

        # Disable parallel tool use if the model doesn't support it
        # Only available for Claude 4 models (opus-4, sonnet-4)
        # By default Claude allows parallel, so we disable it if parallel_tool_calls is False
        model_id_lower = model.model_id.lower()
        if hasattr(model, 'parallel_tool_calls') and not model.parallel_tool_calls:
            if "opus-4" in model_id_lower or "sonnet-4" in model_id_lower or "claude-4" in model_id_lower:
                if isinstance(tool_choice, dict) and tool_choice.get("type") in ["auto", "required"]:
                    tool_choice["disable_parallel_tool_use"] = True

        payload["tool_choice"] = tool_choice
    
    return payload
