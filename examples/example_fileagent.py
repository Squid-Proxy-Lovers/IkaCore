import sys
import os
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # Add src to sys.path

from IkaCore.agents import IkaBaseAgent
from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools
from IkaCore.workflow import IkaWorkflow

API_KEY = os.getenv("API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "deepseek-chat")
if not API_KEY:
    raise ValueError("API_KEY is not set")
if not MODEL_ID:
    raise ValueError("MODEL_ID is not set")


SYSTEM_PROMPT = """ You are a general purpose agent. You can use the tools provided to you to achieve your goal."""


PROMPT = """You are a simple example system. You can read and list files, 
your goal is to return a summary of the files in the current working directory,
you will need to use the list_files and read_file tools to achieve your goal."""

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
        system_prompt=SYSTEM_PROMPT,
        prompt=PROMPT,
        tools=[get_pwd, read_file, list_files],
        model_id=MODEL_ID,
        api_key=API_KEY,
        #logging_level=3,
    )
    agent.execution()

if __name__ == "__main__":
    main()