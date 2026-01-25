import sys
import os
from pathlib import Path
from typing import final
from time import time


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # Add src to sys.path

from IkaCore.agents import IkaBaseAgent
from IkaCore.stages import IkaStage
from IkaCore.tools import IkaTools
from IkaCore.workflow import IkaWorkflow

API_KEY = os.getenv("API_KEY")
MODEL_ID = "deepseek-chat"
if not API_KEY:
    raise ValueError("API_KEY is not set")
if not MODEL_ID:
    raise ValueError("MODEL_ID is not set")

API_KEY2 = os.getenv("API_KEY2")
MODEL_ID2 = "gpt-4o-mini"
if not API_KEY2:
    raise ValueError("API_KEY2 is not set")
if not MODEL_ID2:
    raise ValueError("MODEL_ID2 is not set")
    


SYSTEM_PROMPT = """You are a general purpose agent. You can use the tools provided to you to achieve your goal."""


PROMPT = """
You are a simple example system. You can read and list files, 
your goal is to return a summary of the files in the current working directory,
You must excute all the stages as defined by the system prompt.
"""

STAGE1 = """
In this stage you will need to list the files in the current working directory.
You will need to use the list_files tool to achieve your goal.
you should explore the files in the current working directory and subdirectories.

after you have listed the files, you should move to the next stage.

you should only use get_pwd tool once in this stage. 
"""

STAGE2 = """
In this stage you will need to read the files in the current working directory.
You will need to use the read_file tool to achieve your goal.
You will need to return the content of the files that you have listed in the previous stage.

after you have read the files, you should move to the next stage.

you should only use get_pwd tool once in this stage. 
"""

STAGE3 = """
In this stage you will need to summarize the content of the files in the current working directory.
You will need to return the summary of the files in the current working directory.
 If you have read the files in the previous stage, you should use the content of the files to summarize the files.
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
        limit_calls=1,
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
        execute_function=lambda x: os.listdir(x.get("directory_path") or os.getcwd()),
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

    list_stage = IkaStage(
        name="list_stage",
        prompt=STAGE1,
        tools=[list_files, get_pwd],
        model_id=MODEL_ID2,
        api_key=API_KEY2,
        stage_max_step=30,
    )

    read_stage = IkaStage(
        name="read_stage",
        prompt=STAGE2,
        tools=[read_file,get_pwd],
        model_id=MODEL_ID2,
        api_key=API_KEY2,
        stage_max_step=30,
    )

    summarize_stage = IkaStage(
        name="summarize_stage",
        prompt=STAGE3,
        tools=[],
        allowed_back_to=[1,2],
        model_id=MODEL_ID,
        api_key=API_KEY,
    )
    agent = IkaBaseAgent(
        name="example_system",
        description="A simple example system",
        system_prompt=SYSTEM_PROMPT,
        prompt=PROMPT,
        tools=[],
        Stages=[list_stage, read_stage, summarize_stage],
        model_id=MODEL_ID,
        api_key=API_KEY,
        #logging_level=3,
        maxsteps=100,
    )
    start_time = time()
    finalmsg = agent.execution()
    end_time = time()
    print("-"*100)
    print(f"Time taken: {end_time - start_time} seconds")
    print(finalmsg)

if __name__ == "__main__":
    main()