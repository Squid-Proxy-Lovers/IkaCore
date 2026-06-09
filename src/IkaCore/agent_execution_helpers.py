# pyright: strict

from __future__ import annotations

from typing import Any

from IkaCore.prompts import AGENT_END_INSTRUCTION, HITL_INSTRUCTION, STAGE_MOVEMENT_INSTRUCTION


def build_stage_content_prompt(agent_prompt: str, stages: list[Any], stage: Any, stage_index: int) -> str:
    parts: list[str] = []
    if agent_prompt:
        parts.append(agent_prompt)
    if stages:
        stages_block = "\n\n".join(
            f"Stage {i} ({s.name}):\n{s.prompt or ''}" for i, s in enumerate(stages)
        )
        parts.append("STAGES:\n\n" + stages_block)

    parts.append("CURRENT STAGE IS:\n\n" + (stage.prompt or ""))
    parts.append(STAGE_MOVEMENT_INSTRUCTION)

    if getattr(stage, "hitl", False):
        parts.append(HITL_INSTRUCTION)

    if stage_index == len(stages) - 1:
        parts.append(AGENT_END_INSTRUCTION)

    return "\n\n".join(parts)


def resolve_stage_model_overrides(agent: Any, stage: Any) -> dict[str, Any]:
    model_overrides: dict[str, Any] = {}
    if getattr(stage, "model_id", None) is not None:
        model_overrides["model_id"] = stage.model_id
    if getattr(stage, "api_key", None) is not None:
        model_overrides["api_key"] = stage.api_key
    if getattr(stage, "api_url", None) is not None:
        model_overrides["api_url"] = stage.api_url
    elif model_overrides.get("model_id") is not None:
        model_overrides["api_url"] = agent.geturl(
            model_overrides["model_id"],
            use_responses_api=agent.use_responses_api,
        )
    if getattr(stage, "max_tokens", None) is not None:
        model_overrides["max_tokens"] = stage.max_tokens
    if getattr(stage, "temperature", None) is not None:
        model_overrides["temperature"] = stage.temperature
    return model_overrides


def stage_step_limit(stage: Any, remaining_steps: int) -> int:
    stage_max = getattr(stage, "stage_max_step", 1)
    if stage_max == 0:
        return max(1, remaining_steps)
    return min(stage_max, max(1, remaining_steps))
