import os
import subprocess

from IkaCore import IkaBaseAgent, IkaStage, IkaTools

SYSTEM_PROMPT = """You are a general purpose agent. You can use the tools provided to you to achieve your goal."""


PROMPT = """
You are a simple example system. You can read and list files, 
your goal is to return a summary of the files in the current working directory,
You must excute all the stages as defined by the system prompt. please use the ask_user tool to ask the user for more information if needed.
"""

STAGE1 = """
In this stage you will need to list the directory that the user has requested. 
Please ask the user for the directory they want to explore using the ask_user tool.
You will need to use the list_files and get_tree tool to achieve your goal. If you cant find the directory, 
use the tree tool to get the tree of the directory and only ask the user for more information if you are sure that the directory does not exist.
"""

STAGE2 = """
In this stage you will need to read the files that were in the directory that the user has requested.
You will need to use the read_file tool to achieve your goal. make sure to identify all key files in the directory.
"""

STAGE3 = """
In this stage you will need to summarize the content of the files that the user has requested.
You will need to return the summary of the files that were in the directory that the user has requested.
If you have read the files in the previous stage, you should use the content of the files to summarize the files. 
Make sure to include all key files in the summary and a detailed summary of the files.
"""

def get_pwd_execute(x: dict) -> str:
    try:
        return os.getcwd()
    except OSError as e:
        return f"Error getting working directory: {e}"


def get_tree_execute(x: dict) -> str:
    path = x.get("directory_path") or os.getcwd()
    try:
        if not os.path.exists(path):
            return f"Error: path does not exist: {path}"
        if not os.path.isdir(path):
            return f"Error: '{path}' is not a directory."
        ex = subprocess.run(["tree", path], capture_output=True, text=True, timeout=30)
        if ex.returncode != 0:
            return f"Error running tree: {ex.stderr or ex.stdout or 'unknown'}"
        return ex.stdout or "(empty)"
    except subprocess.TimeoutExpired:
        return "Error: tree command timed out."
    except Exception as e:
        return f"Error: {e}"


def list_files_execute(x: dict):
    path = x.get("directory_path") or os.getcwd()
    try:
        if not os.path.exists(path):
            return f"Error: path does not exist: {path}"
        if not os.path.isdir(path):
            return f"Error: '{path}' is not a directory."
        return os.listdir(path)
    except PermissionError:
        return f"Error: permission denied for {path}"
    except Exception as e:
        return f"Error: {e}"


def read_files_execute(x: dict) -> str:
    path = x.get("file_path") or ""
    if "checkpoints.db" in path:
        return "Error: checkpoints.db is a database file, not a file to read."
    if "logs.txt" in path:
        return "Error: logs.txt is a log file, not a file to read."
    if not path:
        return "Error: file_path is required."
    try:
        if not os.path.exists(path):
            return f"Error: path does not exist: {path}"
        if os.path.isdir(path):
            return f"Error: '{path}' is a directory, not a file. Use list_files to list directory contents."
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except PermissionError:
        return f"Error: permission denied reading {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def main():
    api_key = os.getenv("API_KEY")
    model_id = os.getenv("MODEL_ID", "deepseek-chat")
    if not api_key:
        raise ValueError("API_KEY is not set")
    if not model_id:
        raise ValueError("MODEL_ID is not set")

    api_key2 = os.getenv("API_KEY2")
    model_id2 = os.getenv("MODEL_ID2", "gpt-4o-mini")
    if not api_key2:
        raise ValueError("API_KEY2 is not set")
    if not model_id2:
        raise ValueError("MODEL_ID2 is not set")

    get_tree = IkaTools(
        name="get_tree",
        description="Get the tree of the directory that the user has requested",
        parameters={
            "directory_path": {
                "type": "string",
                "description": "The path to the directory to get the tree from",
            },
        },
        execute_function=get_tree_execute,
    )

    get_pwd = IkaTools(
        name="get_pwd",
        description="Get the current working directory",
        parameters={
            "None": "No parameters are required",
        },
        execute_function=get_pwd_execute,
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
        execute_function=list_files_execute,
    )

    read_file = IkaTools(
        name="read_files",
        description="Read file content. Pass a file path, not a directory. Use list_files to discover files.",
        parameters={
            "file_path": "The path to the file to read",
        },
        execute_function=read_files_execute,
    )

    list_stage = IkaStage(
        name="list_stage",
        prompt=STAGE1,
        tools=[list_files, get_pwd, get_tree],
        model_id=model_id2,
        hitl=True,
        api_key=api_key2,
        stage_max_step=30,
    )

    read_stage = IkaStage(
        name="read_stage",
        prompt=STAGE2,
        tools=[read_file, get_tree],
        model_id=model_id2,
        api_key=api_key2,
        stage_max_step=30,
        checkpoint=True,
    )

    summarize_stage = IkaStage(
        name="summarize_stage",
        prompt=STAGE3,
        tools=[],
        allowed_back_to=[1, 2],
        model_id=model_id,
        api_key=api_key,
    )

    agent = IkaBaseAgent(
        name="example_system",
        description="A simple example system",
        system_prompt=SYSTEM_PROMPT,
        prompt=PROMPT,
        tools=[],
        Stages=[list_stage, read_stage, summarize_stage],
        model_id=model_id,
        api_key=api_key,
        checkpoint=True,
        logging_level=2,
        maxsteps=100,
    )
    # start_time = time()
    agent.execution(checkpoint_uid="df16883c-e384-4d15-bf11-faf547a69b81")
    # finalmsg = agent.execution()
    # end_time = time()
    # # print("-"*100)
    # # print(f"Time taken: {end_time - start_time} seconds")
    # # print(finalmsg)


if __name__ == "__main__":
    main()
