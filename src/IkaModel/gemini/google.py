import json
from typing import Any, Dict, List, Optional


def gemini_fill_payload(model, messages: List[Dict[str, Any]], message_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    message_history = message_history or {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {}
    }
    contents = []

    system_instruction = message_history["system"]["message"] or model.system_prompt

    if message_history["first_input"]["message"]:
        contents.append({
            "role": "user",
            "parts": [{"text": message_history["first_input"]["message"]}]
        })

    if message_history["summary"]["message"]:
        contents.append({
            "role": "model",
            "parts": [{"text": message_history["summary"]["message"]}]
        })

    for msg_id in message_history["messages"]:
        msg = message_history["messages"][msg_id]
        msg_type = msg.get("type", "assistant")
        raw = msg.get("message", "")
        if msg_type == "assistant_with_tools" or msg_type == "tool":
            try:
                contents.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                contents.append({"role": "model", "parts": [{"text": raw if isinstance(raw, str) else str(raw)}]})
        else:
            contents.append({
                "role": "model",
                "parts": [{"text": raw if isinstance(raw, str) else str(raw)}]
            })

    first_input_text = (message_history.get("first_input") or {}).get("message") or ""
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            text = str(msg)
            if i == 0 and first_input_text and text.strip() == first_input_text.strip():
                continue
            contents.append({"role": "user", "parts": [{"text": text}]})
            continue
        if "parts" in msg and "role" in msg:
            if i == 0 and msg.get("role") == "user" and first_input_text:
                parts_text = " ".join(
                    p.get("text", "") for p in msg.get("parts", []) if isinstance(p, dict) and "text" in p
                ).strip()
                if parts_text.strip() == first_input_text.strip():
                    continue
            contents.append(msg)
            continue
        text = msg.get("content", str(msg))
        if i == 0 and first_input_text and text.strip() == first_input_text.strip():
            continue
        contents.append({"role": "user", "parts": [{"text": text}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": model.temperature,
            "maxOutputTokens": model.max_tokens if model.max_tokens and model.max_tokens > 0 else 100,
            "thinkingConfig": {
                "thinkingBudget": 0,
            },
        }
    }

    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    if model.agent_tools:
        function_declarations = []
        for tool in model.agent_tools:
            # For agent_end and subagent tools with type="input", always use input parameter
            if tool.name == "agent_end" and tool.args.type == "input":
                parameters = {
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
                parameters = {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": tool.args.description or f"Input for {tool.name}"
                        }
                    },
                    "required": ["input"]
                }
            # Handle tools with explicit properties
            elif tool.args.properties and len(tool.args.properties) > 0:
                required_list = list(tool.args.properties.get("__required__", []))
                properties = {}
                for prop_name, prop_def in tool.args.properties.items():
                    if prop_name == "__required__":
                        continue
                    prop_schema = {
                        "type": prop_def.get("type", "string"),
                        "description": prop_def.get("description", "")
                    }
                    if "items" in prop_def:
                        prop_schema["items"] = prop_def["items"]
                    if "enum" in prop_def:
                        prop_schema["enum"] = prop_def["enum"]
                    if "properties" in prop_def:
                        prop_schema["properties"] = prop_def["properties"]
                    properties[prop_name] = prop_schema
                parameters = {
                    "type": "object",
                    "properties": properties,
                    "required": required_list
                }
            else:
                # Simple single-argument tool
                arg_name = tool.args.type
                json_type = "string"
                if arg_name in ["stage_index"]:
                    json_type = "integer"
                elif arg_name == "input":
                    json_type = "string"
                parameters = {
                    "type": "object",
                    "properties": {
                        arg_name: {
                            "type": json_type,
                            "description": tool.args.description
                        }
                    },
                    "required": []
                }

            function_declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters
            })

        payload["tools"] = [{"functionDeclarations": function_declarations}]
    
    return payload
