from ..runtime.workflow import Workflow
from ..utils.helpers import get_standard_tools


class PassthroughWorkflow(Workflow):
    @classmethod
    def add_arguments(cls, group):
        super().add_common_arguments(group)
        group.add_argument(
            "--prompt",
            help="Prompt to send to the agent",
            default="",
            required=False,
            type=str,
        )
        group.add_argument(
            "--task-file",
            help="Task file to read instead of prompt",
            default="",
            type=str,
        )
        group.add_argument(
            "--add-search-tool",
            help="Add a search tool to the agent",
            action="store_true",
            default=False,
        )

    def run(self, env) -> str:
        self.logger.info("Starting passthrough workflow")

        model = self.create_model(
            add_search_tool=self.kwargs.get("add_search_tool", False),
        )

        tools = get_standard_tools(
            env,
            include_sagemath=self.kwargs.get("add_sagemath", False),
        )
        agent = self.create_agent(model, tools)

        task = (self.kwargs.get("prompt") or "").strip()
        if self.kwargs.get("task_file") and not task:
            with open(self.kwargs["task_file"], "r") as f:
                task = f.read().strip()

        if not task:
            raise ValueError("No task or prompt provided, use --task-file or --prompt")

        agent_output = agent.run(f"{task}\n{env.get_environment_prompt()}")

        self.save_trace(agent)
        self.logger.info("Agent analysis complete")
        return agent_output.output


