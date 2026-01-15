
from smolagents import tool
import requests
import json, base64, socket, ipaddress, shlex, re
from urllib.parse import urlparse, urljoin
from html.parser import HTMLParser
from .ctf_tools import get_ctf_environment
from .leak_town import LeakTown
from datetime import datetime

@tool
def base_web_scan(url: str) -> str:
    """
    Analyzes web application using curl to detect vulnerabilities.
    
    Args:
        url (str): The url to the web application to analyze in the container.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    commands = [
        f"curl -Is {url} | sed -n '1,20p'",
        f"curl -sL {url}/ | head -n 200",
        f"curl -sL {url}/robots.txt",
        f"curl -sL {url}/sitemap.xml",
    ]
    results = []
    for cmd in commands:
        result = env.run_command_in_container(cmd)
        results.append(f"$ {cmd}\n{result}")
    return "\n" + "="*50 + "\n".join(results)

@tool
def curl_base_scan(url: str) -> str:
    """
    Analyzes web application using curl to detect vulnerabilities.
    
    Args:
        url (str): The url to the web application to analyze in the container.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    return env.run_command_in_container(f"curl -sL '{url}'")

@tool
def curl_path_scan(url: str) -> str:
    """
    Analyzes web application using curl to detect vulnerabilities.
    
    Args:
        url (str): The url to the web application to analyze in the container.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    return env.run_command_in_container(f"curl -sL --path-as-is '{url}'")

@tool
def general_web_request_with_template(url: str, port: int, req_method: str, req_path: str, req_headers: str, req_body: str) -> str:
    """
    Makes a custom HTTP request using netcat to test web applications. Make sure proper host and port are parsed if needed.
    
    Args:
        url (str): The host/IP address of the web application to analyze.
        port (int): The port number of the web application to analyze.
        req_method (str): The HTTP method to use for the request (GET, POST, PUT, DELETE, etc.).
        req_path (str): The path to request on the web application.
        req_headers (str): The HTTP headers to include in the request (format: "Header1: value1\nHeader2: value2").
        req_body (str): The request body to send (for POST/PUT requests).
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    http_request = f"{req_method} {req_path} HTTP/1.1\r\n"
    
    if "Host:" not in req_headers:
        http_request += f"Host: {url}\r\n"
    
    if req_headers.strip():
        for header in req_headers.strip().split("\n"):
            if header.strip():
                http_request += f"{header.strip()}\r\n"
    
    if req_body.strip():
        http_request += f"Content-Length: {len(req_body)}\r\n"
    
    http_request += "\r\n"
    
    if req_body.strip():
        http_request += req_body
    
    return env.run_command_in_container(f'printf "%b" \'{http_request}\' | nc {url} {port}')

@tool
def nikto_base_scan(url: str) -> str:
    """
    Analyzes web application using nikto to detect vulnerabilities.
    
    Args:
        url (str): The url to the web application to analyze in the container.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    return env.run_command_in_container(f"nikto -h '{url}'")

@tool
def gobuster_dir_scan(url: str, wordlist: str = "/usr/share/seclists/Discovery/Web-Content/common.txt") -> str:
    """
    Analyzes web application using gobuster to detect directories.
    
    Args:
        url (str): The url to the web application to analyze in the container.
        wordlist (str): The wordlist to use for the scan (default: /usr/share/seclists/Discovery/Web-Content/common.txt).
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    return env.run_command_in_container(f"gobuster dir -u '{url}' -w {wordlist}")

@tool
def nmap_web_scan(url: str, ports: str = "") -> str:
    """
    Analyzes web application using nmap to detect vulnerabilities. Make sure proper host and port are parsed if needed.
    
    Args:
        url (str): The url to the web application to analyze in the container.
        ports (str): The ports to scan (default: 80).
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    return env.run_command_in_container(f"nmap -p {ports} -sV --script vuln,http-enum,http-headers '{url}'")

@tool
def nmap_port_scan(url: str) -> str:
    """
    Analyzes web application using nmap to detect hidden ports and vulnerable versions. Make sure proper host and port are parsed if needed.
    
    Args:
        url (str): The url to the web application to analyze in the container.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    return env.run_command_in_container(f"nmap -p- '{url}'")


@tool
def smuggling(url: str, port: str="", args: str="") -> str:
    """
    Analyzes web applications if they are hiding pages under HTTP 1. Make sure proper host and port are parsed if needed.
    
    Args:
        url (str): The url to the web application to analyze in the container.
        port (str): The port used for testing
        args (str): additional arguments/payloads needed
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    url = ensure_port_in_url(url,port)
    return env.run_command_in_container(f"smuggler.py -u '{url}' {args}")

@tool
def check_static_page(url: str, port: str = "80") -> str:
    """
    Checks if a web page is static or interactive by analyzing HTML for forms and input fields.
    
    Args:
        url (str): The URL of the web page to analyze (e.g., http://example.com:8080).
        port (str): The port number to connect to if not specified in URL (default: "80").
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    url = ensure_port_in_url(url, port)
    
    command = f'''curl -s "{url}" | grep -qiE '<form|<input|<textarea' && echo "INTERACTIVE: Page contains forms or input fields" || echo "STATIC: No forms or input fields detected"'''
    
    return env.run_command_in_container(command)


@tool
def check_http_methods(url: str, port: str = "") -> str:
    """
    Checks what HTTP methods are allowed on a URL by sending an OPTIONS request.
    
    Args:
        url (str): The URL to check (e.g., http://example.com:8080).
        port (str): The port number to connect to if not specified in URL (default: "80").
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    url = ensure_port_in_url(url, port)
    
    command = f'''curl -I -X OPTIONS "{url}" 2>/dev/null | grep -i "allow:" || echo "No Allow header found"'''
    
    return env.run_command_in_container(command)

@tool
def fuzzing(url: str, wordlist: str = "/usr/share/seclists/Fuzzing/Unicode.txt", Headers:str="") -> str:
    """
    Analyzes web application using ffuf to find bypasses. Remember to have FUZZ in the url in all cases for this
    
    Args:
        url (str): The url to the web application to analyze in the container.
        wordlist (str): The wordlist to use for the scan (default: /usr/share/seclists/Fuzzing/Unicode.txt).
        Headers (str): Additional HTTP headers to include in the request (default: "").
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    if "FUZZ" not in url:
        return "Need to have FUZZ in URL to determine what to fuzz"
    return env.run_command_in_container(f"ffuf -w {wordlist} -u '{url}' -H {Headers}")

@tool
def detect_nginx_version(url: str) -> str:
    """
    Detect nginx version from server headers and error pages.
    
    Args:
        url (str): The URL to check for nginx version.
        
    Returns:
        str: The nginx version if detected, or error message.
    """
    try:
        # Check server header first
        response = requests.head(url, timeout=10)
        server_header = response.headers.get('Server', '')
        
        # Extract version from server header
        version_match = re.search(r'nginx/([0-9]+\.[0-9]+(?:\.[0-9]+)?)', server_header, re.IGNORECASE)
        if version_match:
            return f"nginx/{version_match.group(1)}"
        
        # If no version in header, try error page
        response = requests.get(f"{url}/nonexistent404", timeout=10)
        version_match = re.search(r'nginx/([0-9]+\.[0-9]+(?:\.[0-9]+)?)', response.text, re.IGNORECASE)
        if version_match:
            return f"nginx/{version_match.group(1)}"
            
        # Just check if nginx is detected
        if 'nginx' in server_header.lower():
            return "nginx (version not disclosed)"
            
        return "nginx not detected"
        
    except Exception as e:
        return f"Error detecting nginx version: {e}"

def ensure_port_in_url(url: str, port: str = "80") -> str:
    """
    Helper function to ensure a port is included in the URL.
    If URL already has a port, it's left unchanged.
    
    Args:
        url (str): The URL to process.
        port (str): The port to add if none exists (default: "80").
    
    Returns:
        str: URL with port included.
    """
    # Add port to URL if no port is specified in the URL
    if "://" in url and ":" not in url.split("://")[1].split("/")[0]:
        parts = url.split("://")
        url = f"{parts[0]}://{parts[1].split('/')[0]}:{port}/{'/'.join(parts[1].split('/')[1:])}"
    elif "://" not in url:
        url = f"http://{url}:{port}"
    
    return url

@tool
def get_cves(hostname:str, port:str) -> str:
    """
    Get potential CVEs for the web application. Reformat it so hostname and port are separate, and make sure you get nonempty CVEs

    Args:
        hostname (str): the hostname to process (NOT URL)
        port (str): the port to use to connect to the servers (str)
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    return env.run_command_in_container(f"nmap -sV --script vulners {hostname} -p {port}")

# webhook stuff
# NOTE: this uses leak.town (see leak_town.py)
# we intentionally lie to it and say it's using webhook.site because it's heard of that before
# so it has a (potentially) higher chance of using it because it knows what that does

@tool
def webhook_create() -> str:
    """
    Creates a new temporary webhook URL on webhook.site. 
    Use this when you need a public URL to capture HTTP requests, testing callbacks, or extracting data.
    
    Args:
        None
        
    Returns:
        str: A message containing the new Token and the public URL.
    """
    try:
        # Initialize without token to trigger creation
        lt = LeakTown()
        
        return (
            f"Webhook Created Successfully.\n"
            f"TOKEN: {lt.token}\n"
            f"URL: http[s]://{lt.url}\n"
            f"Email: root@{lt.url}\n"
            f"IMPORTANT: You must retain the TOKEN to view requests or configure this webhook later."
        )
    except Exception as e:
        return f"Error creating webhook: {str(e)}"

@tool
def webhook_view(token: str, limit: int = 10) -> str:
    """
    Retrieves the list of HTTP requests that have been sent to a specific webhook.site webhook.
    
    Args:
        token (str): The webhook token/UUID received when creating the webhook.
        limit (int): The maximum number of requests to retrieve (default: 10).
        
    Returns:
        str: A formatted list of requests including Method, Path, Params, Headers, and Body.
    """
    try:
        lt = LeakTown(token=token)
        requests_list = lt.get_requests(limit=limit)
        
        if not requests_list:
            return "No requests received yet."
            
        output = [f"Found {len(requests_list)} requests:\n"]
        
        for idx, req in enumerate(requests_list, 1):
            url = req.get('url', '')
            ts = req.get('timestamp')
            created_at = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else 'N/A'
            
            entry = [
                f"[{idx}] {req.get('method', 'N/A')} - {created_at}",
                f"    Path: {url}"
            ]
            
            # Extract query params
            if '?' in url:
                query_string = url.split('?')[1]
                entry.append(f"    Params: {query_string}")
                
            # Extract headers and user agent
            headers = req.get('headers', {})
            user_agent_list = headers.get('user-agent') 
            if user_agent_list and isinstance(user_agent_list, list):
                entry.append(f"    User-Agent: {user_agent_list[0]}")
                
            # Content body
            content = req.get('body_raw', '')
            if content:
                # Truncate large bodies for token efficiency
                entry.append(f"    Body: {content[:500]}")
                
            output.append("\n".join(entry))
            
        return "\n\n".join(output)
        
    except Exception as e:
        return f"Error fetching requests: {str(e)}"

@tool
def webhook_clear(token: str) -> str:
    """
    Deletes all captured requests history for a specific leak.town webhook.
    
    Args:
        token (str): The webhook token/UUID.
        
    Returns:
        str: Status message indicating success or failure.
    """
    try:
        lt = LeakTown(token=token)
        lt.delete_all_requests()
        return "✓ All requests cleared successfully."
    except Exception as e:
        return f"Error clearing requests: {str(e)}"

@tool
def webhook_get_config(token: str) -> str:
    """
    View the current configuration (Response Body and Headers) that the webhook returns when hit.
    
    Args:
        token (str): The webhook token/UUID.
        
    Returns:
        str: A formatted string showing current response headers and content body.
    """
    try:
        lt = LeakTown(token=token)
        conf = lt.get_response_config()
        
        # Parse logic extracted from original view_response_config
        headers_data = conf[0]['handler']['headers']
        body_data = conf[0]['handler']['data']
        
        output = ["Current Response Configuration:", "-" * 30]
        output.append("Headers:")
        for k, v in headers_data.items():
            # v is usually a list in the raw config
            val = v[0] if isinstance(v, list) and v else v
            output.append(f"  {k}: {val}")
            
        output.append("-" * 30)
        output.append(f"Response Body Content:\n{body_data}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error fetching configuration: {str(e)}"

@tool
def webhook_update_config(token: str, content: str = None, headers_json: str = None) -> str:
    """
    Updates the response the webhook sends back when hit. Can change the HTML body or the Headers.
    
    Args:
        token (str): The webhook token/UUID.
        content (str): Optional. The new HTML/Text content the webhook should return.
        headers_json (str): Optional. A JSON string representing a dictionary of headers (e.g., '{"Content-Type": "application/json"}').
        
    Returns:
        str: Success message.
    """
    try:
        lt = LeakTown(token=token)
        
        headers_dict = None
        if headers_json:
            try:
                headers_dict = json.loads(headers_json)
            except json.JSONDecodeError:
                return "Error: headers_json argument must be valid JSON string."
        
        lt.update_response_config(content=content, headers=headers_dict)
        
        updates = []
        if content: updates.append("Content")
        if headers_dict: updates.append("Headers")
        
        return f"Successfully updated configuration for: {', '.join(updates)}"
        
    except Exception as e:
        return f"Error updating configuration: {str(e)}"

# @tool
# def webhook_create() -> str:
#     """
#     Creates a new webhook.site endpoint for testing webhooks and HTTP requests.
#     Returns the webhook URL, email, and UUID.
#     """
#     env = get_ctf_environment()
#     if not env:
#         return "Error: No CTF environment is currently active"
    
#     cmd = "python3 /opt/webhook_site.py --create"
#     result = env.run_command_in_container(cmd)
#     return f"$ {cmd}\n{result}"


# @tool
# def webhook_view(token: str, limit: int = 10, raw: bool = False) -> str:
#     """
#     Views requests sent to a webhook.site endpoint.
    
#     Args:
#         token (str): The webhook UUID/token
#         limit (int): Maximum number of requests to retrieve (default: 10)
#         raw (bool): Output raw JSON instead of formatted text (default: False)
#     """
#     env = get_ctf_environment()
#     if not env:
#         return "Error: No CTF environment is currently active"
    
#     raw_flag = "--raw" if raw else ""
#     cmd = f"python3 /opt/webhook_site.py --token {token} --view --limit {limit} {raw_flag}".strip()
#     result = env.run_command_in_container(cmd)
#     return f"$ {cmd}\n{result}"


# @tool
# def webhook_clear(token: str) -> str:
#     """
#     Clears all requests from a webhook.site endpoint.
    
#     Args:
#         token (str): The webhook UUID/token
#     """
#     env = get_ctf_environment()
#     if not env:
#         return "Error: No CTF environment is currently active"
    
#     cmd = f"python3 /opt/webhook_site.py --token {token} --clear"
#     result = env.run_command_in_container(cmd)
#     return f"$ {cmd}\n{result}"


# @tool
# def webhook_delete(token: str) -> str:
#     """
#     Deletes a webhook.site endpoint completely.
    
#     Args:
#         token (str): The webhook UUID/token
#     """
#     env = get_ctf_environment()
#     if not env:
#         return "Error: No CTF environment is currently active"
    
#     cmd = f"python3 /opt/webhook_site.py --token {token} --delete"
#     result = env.run_command_in_container(cmd)
#     return f"$ {cmd}\n{result}"


# @tool
# def webhook_delete_request(token: str, request_ids: str) -> str:
#     """
#     Deletes specific request(s) from a webhook by UUID or index number.
    
#     Args:
#         token (str): The webhook UUID/token
#         request_ids (str): Comma-separated list of request UUIDs or index numbers (e.g., "1,3,5")
#     """
#     env = get_ctf_environment()
#     if not env:
#         return "Error: No CTF environment is currently active"
    
#     cmd = f"python3 /opt/webhook_site.py --token {token} --delete-request {request_ids}"
#     result = env.run_command_in_container(cmd)
#     return f"$ {cmd}\n{result}"


# @tool
# def webhook_send_test(token: str, url: str = "POST", data: str = "test data") -> str:
#     """
#     Sends a test request to a webhook.site endpoint using curl.
    
#     Args:
#         token (str): The webhook UUID/token
#         method (str): HTTP method (GET, POST, PUT, etc.)
#         data (str): Data to send in the request body
#     """
#     env = get_ctf_environment()
#     if not env:
#         return "Error: No CTF environment is currently active"
    
#     if method.upper() in ["GET", "DELETE", "HEAD"]:
#         cmd = f"curl -X {method.upper()} https://webhook.site/{token}"
#     else:
#         # Escape single quotes in data
#         escaped_data = data.replace("'", "'\\''")
#         cmd = f"curl -X {method.upper()} -d '{escaped_data}' https://webhook.site/{token}"
    
#     result = env.run_command_in_container(cmd)
#     return f"$ {cmd}\n{result}\n\nUse webhook_view('{token}') to see the captured request"
