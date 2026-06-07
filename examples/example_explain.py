import os

from IkaCore import IkaBaseAgent, IkaTools

AsynPrompt = """
For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.
"""
# Manager Agent
SYSTEM_PROMPT = """ You are a general purpose agent. You can use the tools provided to you to achieve your goal.""" + AsynPrompt

PROMPT = """
Explain to me how Ika Core works in detail I only care about python code, Find files and pass them into your subagent to explain them.
Make sure to pass the files to the subagent in a list, provide the full path to the file, do not pass the file content, only the path.

Please provide 5-10 files to the subagent, and please try to batch your tool call request so they run in parallel (this is CRITICAL for performance).
This is especially important for the list_files tool and subagent tool calls.
"""

# Subagent
SUBAGENT_SYSTEM_PROMPT = """You are a subagent. You can use the tools provided to you to achieve your goal.""" + AsynPrompt

SUBAGENT_PROMPT = """Explain what this file(s) do in detail, do not waste time trying ot figure out the directory structure, just explain the file(s) you are given to you."""


def main():
    api_key = os.getenv("API_KEY")
    model_id = os.getenv("MODEL_ID", "deepseek-chat")
    if not api_key:
        raise ValueError("API_KEY is not set")
    if not model_id:
        raise ValueError("MODEL_ID is not set")

    def read_file_execute(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    get_pwd = IkaTools(
        name="get_pwd",
        description="Get the current working directory",
        parameters={
            "None": "No parameters are required",
        },
        execute_function=lambda _: os.getcwd(),
    )

    list_files = IkaTools(
        name="list_files",
        description="List files in a directory",
        parameters={
            "directory_path": {
                "type": "string",
                "description": "The path to the directory to list files from",
                "required": True,
            },
        },
        execute_function=lambda x: os.listdir(x["directory_path"]),
    )

    read_file = IkaTools(
        name="read_files",
        description="Read file content",
        parameters={
            "file_path": "The path to the file to read",
        },
        execute_function=lambda x: read_file_execute(x["file_path"]),
    )

    Subagent = IkaBaseAgent(
        name="subagent",
        description="File reader subagent",
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        prompt=SUBAGENT_PROMPT,
        tools=[read_file],
        model_id=model_id,
        api_key=api_key,
    )

    ManagerAgent = IkaBaseAgent(
        name="example_system",
        description="A simple example system",
        system_prompt=SYSTEM_PROMPT,
        prompt=PROMPT,
        tools=[get_pwd, list_files],
        model_id=model_id,
        api_key=api_key,
        subagents=[Subagent],
        maxsteps=30,
        logging_level=2,
    )
    ManagerAgent.execution()


if __name__ == "__main__":
    main()
