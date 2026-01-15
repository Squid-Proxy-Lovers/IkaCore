#!/usr/bin/env python3
"""
summarize_ctf_log.py (Responses API version)
- Converts .log -> .md
- Uses OpenAI Responses API for file upload support
- Supports markdown, PDF, and other file types
- Supports writeup file/url compare
"""

import os
import re
import argparse
from openai import OpenAI
from typing import Optional

SYSTEM_PROMPT = """You are a senior CTF analyst and technical writer.
You will read a CTF solve log (a terminal transcript) provided as an uploaded file
and optionally a reference write-up.

Produce a structured output with sections:
[Overview], [Technical Steps], [Key Findings], [Comparison] (if reference given), [Recommendations].

Be explicit about commands, payloads and exact steps used for flag recovery.
If something is uncertain, mark it "uncertain" and quote the short snippet (<=120 chars).
Return the final summary in Markdown.
"""

ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def convert_log_to_md(log_path: str, output_path: str = None) -> str:
    """Convert a .log file to markdown format."""
    with open(log_path, "r", errors="ignore") as f:
        content = f.read()

    # strip ANSI escapes
    content = ANSI_RE.sub("", content)

    lines = content.splitlines()
    md_lines = []
    in_code_block = False

    for line in lines:
        # heuristic for shell prompt lines, keep them inside bash block
        if re.match(r'^[\w\-.]+@[\w\-.]+:.*[#\$] ', line) or line.strip().startswith("$ "):
            if not in_code_block:
                md_lines.append("```bash")
                in_code_block = True
            md_lines.append(line)
        elif line.strip() == "":
            if in_code_block:
                md_lines.append("```")
                in_code_block = False
            md_lines.append("")
        else:
            if in_code_block:
                md_lines.append("```")
                in_code_block = False
            md_lines.append(line)

    if in_code_block:
        md_lines.append("```")

    md_content = "# CTF Solve Log\n\n" + "\n".join(md_lines)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Converted log written to: {output_path}")
    return md_content

def upload_file_to_openai(client: OpenAI, file_path: str) -> str:
    """
    Upload a file to OpenAI and return the file ID.
    """
    print(f"Uploading file: {file_path}")
    with open(file_path, "rb") as f:
        file_obj = client.files.create(
            file=f,
            purpose="responses"
        )
    print(f"✓ File uploaded with ID: {file_obj.id}")
    return file_obj.id

def summarize_ctf_log(
    client: OpenAI,
    log_file_path: str,
    focus: str = "general",
    writeup_file: Optional[str] = None,
    writeup_url: Optional[str] = None,
    model: str = "gpt-5-mini"
):
    """
    Upload files and call OpenAI Responses API to summarize the CTF log.
    
    Args:
        client: OpenAI client instance
        log_file_path: Path to the markdown log file to upload
        focus: Focus area for the summary
        writeup_file: Optional path to reference writeup file
        writeup_url: Optional URL to reference writeup
        model: Model to use (gpt-5 or gpt-5-mini)
    """
    # Upload the main log file
    log_file_id = upload_file_to_openai(client, log_file_path)
    
    # Upload writeup file if provided
    writeup_file_id = None
    if writeup_file and os.path.exists(writeup_file):
        print(f"Uploading reference writeup: {writeup_file}")
        writeup_file_id = upload_file_to_openai(client, writeup_file)
    
    # Build the user message content
    user_message = f"""I have uploaded a CTF solve log file for analysis.

Focus: {focus}

Please read the attached solve log file and produce a structured summary in Markdown with these sections:
[Overview], [Technical Steps], [Key Findings], [Comparison] (if reference provided), [Recommendations].

Be explicit about commands, payloads and exact steps used for flag recovery.
If something is uncertain, mark it "uncertain" and quote the short snippet (<=120 chars)."""

    if writeup_file_id:
        user_message += "\n\nI've also attached a reference writeup. Please compare the solve log with this writeup in your [Comparison] section."
    elif writeup_url:
        user_message += f"\n\nReference writeup URL: {writeup_url}\nPlease compare the solve log with the content at this URL in your [Comparison] section."
    
    # Prepare attachments
    attachments = [{"file_id": log_file_id}]
    if writeup_file_id:
        attachments.append({"file_id": writeup_file_id})
    
    print(f"\nCalling OpenAI Responses API with model: {model}")
    print(f"Files attached: {len(attachments)} file(s)")
    
    try:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": user_message,
                    "attachments": attachments
                }
            ],
            temperature=0.3,
            max_completion_tokens=4096
        )
        
        # Extract the summary from the response
        summary = response.output[0].content
        
        # Clean up uploaded files
        print("\nCleaning up uploaded files...")
        try:
            client.files.delete(log_file_id)
            print(f"✓ Deleted log file: {log_file_id}")
            if writeup_file_id:
                client.files.delete(writeup_file_id)
                print(f"✓ Deleted writeup file: {writeup_file_id}")
        except Exception as e:
            print(f"Warning: Could not delete files: {e}")
        
        return summary
    except Exception as e:
        print(f"Error calling OpenAI Responses API: {e}")
        # Try to clean up on error
        try:
            client.files.delete(log_file_id)
            if writeup_file_id:
                client.files.delete(writeup_file_id)
        except:
            pass
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Convert CTF .log to .md and summarize using OpenAI Responses API"
    )
    parser.add_argument("log_file", help="Path to the .log file")
    parser.add_argument(
        "-o", "--output",
        help="Output path for .md file (default: same as log_file with .md extension)"
    )
    parser.add_argument(
        "-s", "--summary-output",
        help="Output path for summary (default: log_file with .summary.md extension)"
    )
    parser.add_argument(
        "-f", "--focus",
        default="general",
        help="Focus area for summary (e.g., 'web exploitation', 'crypto', 'general')"
    )
    parser.add_argument(
        "-w", "--writeup-file",
        help="Path to reference writeup file for comparison"
    )
    parser.add_argument(
        "-u", "--writeup-url",
        help="URL to reference writeup for comparison"
    )
    parser.add_argument(
        "-m", "--model",
        default="gpt-5-mini",
        choices=["gpt-5", "gpt-5-mini"],
        help="OpenAI model to use (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't delete uploaded files after processing"
    )

    args = parser.parse_args()

    # Determine output paths
    base_name = os.path.splitext(args.log_file)[0]
    md_output = args.output or f"{base_name}.md"
    summary_output = args.summary_output or f"{base_name}.summary.md"

    # Convert log to markdown
    print(f"Converting {args.log_file} to Markdown...")
    md_content = convert_log_to_md(args.log_file, md_output)

    # Initialize OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided. Set OPENAI_API_KEY or use --api-key")
        return 1

    client = OpenAI(api_key=api_key)

    # Generate summary using Responses API
    print(f"\nGenerating summary with focus: {args.focus}")
    try:
        summary = summarize_ctf_log(
            client=client,
            log_file_path=md_output,  # Upload the converted .md file
            focus=args.focus,
            writeup_file=args.writeup_file,
            writeup_url=args.writeup_url,
            model=args.model
        )

        # Save summary
        with open(summary_output, "w", encoding="utf-8") as f:
            f.write(summary)
        
        print(f"\n{'='*60}")
        print(f"✓ Summary saved to: {summary_output}")
        print(f"{'='*60}\n")
        print(summary[:500] + "..." if len(summary) > 500 else summary)

        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
