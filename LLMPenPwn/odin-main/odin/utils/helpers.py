from typing import Any, Dict

from ..model import (Model, OpenAIModel, OpenAIResponsesModel,
                     OpenRouterAPIModel, XAIAPIModel, PortKeyModel, DeepSeekAPIModel)
from ..runtime.environment import Environment
from ..tools import RemoteDecompileBinaryTool, Tool

MODELS = [OpenAIModel, OpenAIResponsesModel, OpenRouterAPIModel, XAIAPIModel, PortKeyModel, DeepSeekAPIModel]


def get_model_from_config(cfg: Dict[str, Any]) -> Model:
    model_cls = cfg.get("model_class", OpenAIModel.__name__)
    for cls in MODELS:
        if cls.__name__ == model_cls:
            return cls(**cfg.get("arguments", {}))
    raise ValueError(f"Unknown model class: {model_cls}")


def get_standard_tools(
    env: Environment,
    *,
    include_decompiler: bool = False,
    include_sagemath: bool = False,
) -> list[Tool]:
    tools = [env.get_bash_tool(), env.get_python_tool()]
    if include_decompiler:
        tools.append(RemoteDecompileBinaryTool(container=env.container))
    if include_sagemath:
        tools.append(env.get_sagemath_tool())
    return tools

