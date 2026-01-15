#!/usr/bin/env python3
"""
Summarize a CTF solve .log file using OpenAI API with file upload (multimodal).
"""

import os
from openai import OpenAI

SYSTEM_PROMPT = """You are an expert CTF log summarizer.
Read the attached `.log` file and produce:
1️⃣  A concise overview of what was attempted and the final result.
2️⃣  Key steps (enumeration, exploitation, tools, commands).
3️⃣  Findings (flag, vuln type, environment quirks).
4️⃣  Advice or next steps.

Be concise but technical, clearly structured, and quote short snippets when useful.
"""

def summarize_log(api_key: str, log_path: str, focus="exploit and flag recovery steps", model="gpt-5-mini"):
    """
    Summarize the given .log file using OpenAI API with file upload.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"File not found: {log_path}")

    client = OpenAI(api_key=api_key)

    # Upload the file to OpenAI
    print(f"Uploading {log_path}...")
    with open(log_path, "rb") as f:
        file_upload = client.files.create(
            file=f,
            purpose="assistants"
        )

    print(f"File uploaded with ID: {file_upload.id}")

    # Create an assistant with file search capability
    assistant = client.beta.assistants.create(
        name="CTF Log Summarizer",
        instructions=SYSTEM_PROMPT,
        model=model,
        tools=[{"type": "file_search"}]
    )

    # Create a thread with the file attached
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": f"Summarize the attached CTF solve log. Focus on {focus}.",
                "attachments": [
                    {
                        "file_id": file_upload.id,
                        "tools": [{"type": "file_search"}]
                    }
                ]
            }
        ]
    )

    # Run the assistant
    print("Processing...")
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Get the response
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        response_content = messages.data[0].content[0].text.value

        # Cleanup
        print("Cleaning up...")
        client.files.delete(file_upload.id)
        client.beta.assistants.delete(assistant.id)

        return response_content
    else:
        raise RuntimeError(f"Assistant run failed with status: {run.status}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize a CTF solve .log using OpenAI API with file upload.")
    parser.add_argument("logfile", help="Path to .log file")
    parser.add_argument("--api-key", help="OpenAI API key or set OPENAI_API_KEY env var")
    parser.add_argument("--focus", default="exploit and flag recovery steps")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OpenAI API key (use --api-key or set OPENAI_API_KEY).")

    summary = summarize_log(api_key, args.logfile, focus=args.focus, model=args.model)

    print("\n--- SUMMARY ---\n")
    print(summary)
