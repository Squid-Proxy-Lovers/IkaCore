import os

from IkaCore import IkaBaseAgent, IkaTools

MODEL_ID = "qwen/qwen3-coder-next"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _load_key():
    script_dir = Path(__file__).resolve().parent
    keys_file = script_dir / "keys.cfg"
    if keys_file.exists():
        with open(keys_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("openrouter="):
                    return line.split("=", 1)[1].strip()
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")


API_KEY = _load_key()
if not API_KEY:
    raise ValueError("Set OPENROUTER_API_KEY or API_KEY, or add openrouter=<key> to examples/keys.cfg")
if not API_KEY.startswith("sk-or-"):
    raise ValueError(
        "OpenRouter key must start with sk-or-. Get a valid key at https://openrouter.ai/keys "
        "and set it in examples/keys.cfg (openrouter=<key>) or OPENROUTER_API_KEY"
    )
if not MODEL_ID:
    raise ValueError("MODEL_ID is not set")


SYSTEM_PROMPT = """ You are a general purpose agent. You can use the tools provided to you to achieve your goal."""


PROMPT = """Your task: summarize the files in the current working directory and then call agent_end with that summary.

Steps (do each at most once unless you need to read a file):
1. Call get_pwd once to get the current directory path.
2. Call list_files once with that path (or use "." for current directory). Use the real path from get_pwd—do not invent paths like /home/user/project.
3. Optionally call read_files for any file you want to include in the summary.
4. Call agent_end with your short summary in the "input" argument (e.g. "Summary: This directory contains ...").

Rules:
- Do not call get_pwd or list_files more than once. You already have the information after the first call.
- Use only paths returned by get_pwd or "." for list_files. Never use made-up paths.
- When done, call agent_end exactly once with a non-empty "input" containing your summary.
"""

def main():

    def read_file_execute(file_path: str) -> str:
        return open(file_path, "r").read()



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
                "required": True
            },
        },
        execute_function=lambda x: os.listdir(x["directory_path"]),
        #required=True,
    )

    read_file = IkaTools(
        name="read_files",
        description="Read file content",
        parameters={
            "file_path": "The path to the file to read",
        },
        execute_function=lambda x: read_file_execute(x["file_path"]),
    )

    agent = IkaBaseAgent(
        name="example_system",
        description="A simple example system",
        api_url=OPENROUTER_URL,
        system_prompt=SYSTEM_PROMPT,
        prompt=PROMPT,
        tools=[get_pwd, read_file, list_files],
        model_id=MODEL_ID,
        api_key=API_KEY,
        maxsteps=30,
        logging_level=3,
    )
    agent.execution()

if __name__ == "__main__":
    main()
