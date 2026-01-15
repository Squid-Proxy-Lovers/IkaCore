import datetime
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import List

from ..agents import MultiStepAgent
from ..model import Model, OpenAIResponsesModel
from ..tools import Tool
from ..utils.helpers import get_model_from_config
from .environment import Environment


class Workflow(ABC):
    @classmethod
    def add_arguments(cls, parser) -> None:
        pass

    @classmethod
    def add_common_arguments(cls, group) -> None:
        group.add_argument(
            "--output-trace",
            help="Output file for the agent's trace",
            default=f"agent-trace-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
            required=False,
            type=str,
        )
        group.add_argument(
            "--model",
            help="OpenAI model to use (default: gpt-5)",
            default="gpt-5",
            type=str,
        )
        group.add_argument(
            "--reasoning-effort",
            help="Reasoning effort level (low, medium, high)",
            default="medium",
            type=str,
        )
        group.add_argument(
            "--max-steps",
            help="Maximum number of steps for the agent (default: 150)",
            default=150,
            type=int,
        )

    def __init__(self, code_path: Path, **kwargs) -> None:
        self.code_path = code_path
        self.kwargs = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_model(
        self,
        default_model: str = "gpt-5",
        default_effort: str = "medium",
        add_search_tool: bool = False,
    ) -> Model:
        if self.kwargs.get("model_cfg"):
            return get_model_from_config(self.kwargs.get("model_cfg"))
        else:
            return OpenAIResponsesModel(
                model_id=self.kwargs.get("model", default_model),
                api_key=os.environ.get("OPENAI_API_KEY"),
                requests_per_minute=60,
                reasoning={
                    "summary": "detailed",
                    "effort": self.kwargs.get("reasoning_effort", default_effort),
                },
                add_search_tool=self.kwargs.get("add_search_tool", False) or add_search_tool,
            )

    def create_agent(
        self,
        model: Model,
        tools: List[Tool],
        max_steps: int | None = None,
    ) -> MultiStepAgent:
        return MultiStepAgent(
            model=model,
            max_steps=max_steps or self.kwargs.get("max_steps", 150),
            tools=tools,
            on_step_completed_callback=self.kwargs.get("trace_callback"),
            verbose=True,
        )

    def save_trace(self, agent: MultiStepAgent, output_path: str | None = None) -> None:
        fp = output_path or self.kwargs.get("output_trace")
        if fp:
            with open(fp, "w") as f:
                json.dump(asdict(agent.get_trace()), f, indent=2)

    def format_task(self, task: str, env: Environment) -> str:
        return f"""<task>\n{task}\n</task>\n{env.get_environment_prompt()}""".strip()

    @abstractmethod
    def run(self, env: Environment):
        raise NotImplementedError


