import inspect
import keyword
import logging
from typing import Any

_LOG = logging.getLogger(__name__)

AUTHORIZED_TYPES = {
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "array",
    "object",
    "any",
    "null",
}

def _valid_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name) if isinstance(name, str) else False

def _normalize_types(t: Any) -> list[str]:
    if isinstance(t, str):
        return [t]
    if isinstance(t, list) and all(isinstance(x, str) for x in t):
        return t
    raise TypeError("type must be str or list[str]")

class Tool:
    name: str = ""
    description: str = ""
    inputs: dict[str, dict[str, Any]] = {}
    output_type: str = "string"
    tool_guidelines: str = ""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls._validate()

    def to_json(self):
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "output_type": self.output_type,
        }

    @classmethod
    def _validate(cls) -> None:
        if not _valid_name(cls.name):
            raise TypeError(f"invalid tool name: {cls.name!r}")
        if not isinstance(cls.description, str) or not cls.description.strip():
            raise TypeError("description must be a non-empty str")
        if not isinstance(cls.inputs, dict) or not cls.inputs:
            raise TypeError("inputs must be a non-empty dict")
        if cls.output_type not in AUTHORIZED_TYPES:
            raise TypeError(f"output_type must be one of {AUTHORIZED_TYPES}")

        for key, spec in cls.inputs.items():
            if not isinstance(spec, dict):
                raise TypeError(f"inputs[{key!r}] must be a dict")
            if "type" not in spec or "description" not in spec:
                raise TypeError(f"inputs[{key!r}] must have 'type' and 'description'")
            types = _normalize_types(spec["type"])
            bad = [t for t in types if t not in AUTHORIZED_TYPES]
            if bad:
                raise TypeError(f"inputs[{key!r}] invalid types {bad}; allowed {AUTHORIZED_TYPES}")
            if "nullable" in spec and not isinstance(spec["nullable"], bool):
                raise TypeError(f"inputs[{key!r}]['nullable'] must be bool if present")

        sig = inspect.signature(cls.forward)
        params = [p for p in sig.parameters if p != "self"]
        if set(params) != set(cls.inputs):
            raise TypeError(f"forward params {params} must match inputs keys {list(cls.inputs.keys())}")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def to_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []

        for name, spec in self.inputs.items():
            t = spec.get("type", "string")
            if isinstance(t, list):
                t = t[0] if t else "string"
            if t == "any":
                t = "string"

            prop = {"type": t}

            if t == "array":
                prop["items"] = spec.get("items", {"type": "string"})

            if "description" in spec:
                prop["description"] = spec["description"]
            if spec.get("nullable", False):
                prop["nullable"] = True
            else:
                required.append(name)

            if "enum" in spec:
                prop["enum"] = spec["enum"]

            properties[name] = prop

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                },
            },
        }