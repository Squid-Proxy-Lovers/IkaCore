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
    
    for msg in messages:
        if isinstance(msg, dict):
            if "content" in msg:
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg["content"]}]
                })
            else:
                contents.append({
                    "role": "user",
                    "parts": [{"text": str(msg)}]
                })
        else:
            contents.append({
                "role": "user",
                "parts": [{"text": str(msg)}]
            })
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": model.temperature,
            "maxOutputTokens": model.max_tokens if model.max_tokens and model.max_tokens > 0 else 100
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
            # Handle tools with explicit properties (skip __required__; each value must be a Schema object)
            elif tool.args.properties and len(tool.args.properties) > 0:
                required_list = list(tool.args.properties.get("__required__", []))
                properties = {}
                for prop_name, prop_def in tool.args.properties.items():
                    if prop_name == "__required__":
                        continue
                    if isinstance(prop_def, dict):
                        p = {
                            "type": prop_def.get("type", "string"),
                            "description": prop_def.get("description", "")
                        }
                        if p["type"] == "array":
                            p["items"] = prop_def.get("items") if prop_def.get("items") else {"type": "string"}
                        properties[prop_name] = p
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
