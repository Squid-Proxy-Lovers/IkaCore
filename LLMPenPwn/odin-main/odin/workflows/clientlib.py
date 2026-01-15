import textwrap

from ..runtime.workflow import Workflow
from ..utils.helpers import get_standard_tools


class ClientlibWorkflow(Workflow):
    @classmethod
    def add_arguments(cls, group):
        super().add_common_arguments(group)

    def run(self, env) -> str:
        self.logger.info("Starting client library generation workflow")

        model = self.create_model()

        tools = get_standard_tools(env, include_decompiler=True)
        agent = self.create_agent(model, tools)

        task = textwrap.dedent(f"""
        You are a software & security expert. You have been given access to a codebase located at `/opt/resources/` within a read-only container environment. The codebase is mounted at this path and you can explore it using the available tools.

        Your goal is to identify the main components, service entry points, and data flows within the codebase. You will be writing a python library that interacts with the web/socket service. That library will be simple, but should cover ALL the functionality of the service.

        Process:
        1. Begin by exploring the codebase structure with `ls -la /opt/resources`
        2. Focus on entry points (web routes, sockets, etc.)
        3. Identify all components and how users can interact with them
        4. Write a simple python client library that interacts with the service, covering ALL functionality.
            - We have some preferences for certain python dependencies:
                1. For web interactions, use `requests`
                2. For websockets, use `websockets` or `socketio`
                3. For socket connections, use `pwntools` ONLY. IMPORTANT: Do not use `socket` library.
        5. Test your library code using the python REPL tool before finalizing it to ensure all functions work as expected.
        6. When you are confident in your library, provide the complete code and state any caveats/limitations in your final message.

        IMPORTANT:
        - Ensure your final message contains a large code block with the complete python library code. Only the largest codeblock of your final message will be used. Other codeblocks could be library usage examples.
        - Ensure your final message is markdown formatted giving a simple overview of the library, how to use it, and any caveats/limitations.

        Be thorough and methodical in your approach. Only your final message will be recorded, so ensure it is complete and accurate.
        """).strip()

        output = agent.run(self.format_task(task, env))

        self.save_trace(agent)

        return output.output


