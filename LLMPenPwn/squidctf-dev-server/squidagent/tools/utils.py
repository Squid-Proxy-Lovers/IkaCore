def run_command(command: str) -> str:
    import subprocess

    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred while executing the command: {e.stderr}"

# squidagent/tools/utils.py
