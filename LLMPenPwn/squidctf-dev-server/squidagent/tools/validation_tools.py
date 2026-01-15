#!/usr/bin/env python3
"""
Validation tools for CTF challenge solving - ensures agents get feedback
"""
import re
import subprocess
import json
import sys
from pathlib import Path
from smolagents import tool

squid_agent_path = Path(__file__).parent.parent.parent / "Squid_Agent"
sys.path.insert(0, str(squid_agent_path))
from squidagent.tools.ctf_tools import get_ctf_environment 

@tool
def run_exploit_and_capture_flag(script_path: str, timeout: int = 120) -> str:
    """
    Run an exploit script in the container, capture flag output, and return results.

    Args:
        script_path: Path to the Python exploit script to run (inside the container)
        timeout: Maximum seconds to wait for the script (default: 30)

    Returns:
        JSON string with:
          'success': (bool) True if flag found,
          'flag': (str) Extracted flag or None,
          'output': (str) First 2000 chars of script output,
          'error': (str) Error message or None
    """
    env = get_ctf_environment()
    if not env:
        return json.dumps({
            'success': False,
            'flag': None,
            'output': '',
            'error': 'No CTF environment is currently active'
        })

    try:
        # Run the script in the container
        output = env.run_command_in_container(
            f"python3 '{script_path}'", 
            workdir="/home/ctfplayer/", 
            timeout=timeout
        )
        
        # Check if there was an error (the method returns error messages as strings)
        exit_code = 0
        error_msg = None
        if output and "Error:" in output:
            error_msg = output
            exit_code = 1
        elif output and "Command failed" in output:
            error_msg = output
            exit_code = 1

        # Recognize CTF flags using common regexes
        flag_patterns = [
            r'flag\{[^}]+\}',
            r'FLAG\{[^}]+\}',
            r'csawctf\{[^}]+\}',
            r'picoCTF\{[^}]+\}',
            r'HTB\{[^}]+\}',
            r'CHTB\{[^}]+\}',
        ]

        found_flag = None
        for pattern in flag_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                found_flag = match.group(0)
                break

        return json.dumps({
            'success': found_flag is not None,
            'flag': found_flag,
            'output': output[:2000] if output else '',  # Don't return super long output
            'error': error_msg if error_msg else None
        })
    
    except subprocess.TimeoutExpired:
        return json.dumps({
            'success': False,
            'flag': None,
            'output': '',
            'error': f'Script timed out after {timeout} seconds'
        })
    except Exception as e:
        return json.dumps({
            'success': False,
            'flag': None,
            'output': '',
            'error': f'Exception: {str(e)}'
        })

# @tool
# def check_if_flag_valid(potential_flag: str) -> str:
#     """
#     Validate if a string looks like a valid CTF flag.
    
#     Args:
#         potential_flag: The string to validate as a flag
    
#     Returns:
#         JSON with 'is_valid' (bool), 'format' (str), 'reason' (str)
#     """
#     import json
    
#     if not potential_flag:
#         return json.dumps({
#             'is_valid': False,
#             'format': 'empty',
#             'reason': 'Empty string provided'
#         })
    
#     # Common flag formats
#     flag_formats = {
#         r'^flag\{.{10,}\}$': 'standard flag{...}',
#         r'^FLAG\{.{10,}\}$': 'uppercase FLAG{...}',
#         r'^csawctf\{.{10,}\}$': 'CSAW format',
#         r'^picoCTF\{.{10,}\}$': 'picoCTF format',
#         r'^CHTB\{.{10,}\}$': 'HackTheBox format',
#         r'^\w{20,}$': 'raw hex/base64 flag (no braces)',
#     }
    
#     for pattern, fmt in flag_formats.items():
#         if re.match(pattern, potential_flag, re.IGNORECASE):
#             return json.dumps({
#                 'is_valid': True,
#                 'format': fmt,
#                 'reason': f'Matches pattern: {fmt}'
#             })
    
#     return json.dumps({
#         'is_valid': False,
#         'format': 'unknown',
#         'reason': 'Does not match any known flag format'
#     })


@tool
def test_exploit_locally(script_path: str, binary_path: str = None) -> str:
    """
    Test an exploit script locally before trying remote.
    
    Args:
        script_path: Path to exploit script
        binary_path: Optional path to local binary to test against
    
    Returns:
        JSON with test results
    """
    import json
    
    # TODO: Could add logic to:
    # 1. Start local binary in background
    # 2. Run exploit against localhost
    # 3. Check if it works before hitting remote
    
    return json.dumps({
        'status': 'not_implemented',
        'message': 'Local testing stub - modify script to test locally first'
    })

