#!/usr/bin/env python3
import requests
import sys
import re
import os
import json
from urllib.parse import urljoin, urlparse
from pathlib import Path

def normalize_site(site):
    if not site.startswith(("http://", "https://")):
        site = "https://" + site
    return site.rstrip('/')

def get_csrf_nonce(html):
    # Try to extract the JS variable: 'csrfNonce': "..."
    m = re.search(r"'csrfNonce'\s*:\s*\"([0-9a-fA-F]+)\"", html)
    if m:
        return m.group(1)
    # Fallback: try input field
    m = re.search(r'<input[^>]+name=["\']nonce["\'][^>]+value=["\']([^"\']+)["\']', html)
    if m:
        return m.group(1)
    return None

def login(site, username, password, session):
    login_url = urljoin(site + "/", "login")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ctfd-login-script/1.1)",
        "Referer": login_url,
    }

    # Step 1: Get login page to retrieve CSRF nonce
    resp = session.get(login_url, headers=headers, timeout=10)
    if resp.status_code != 200:
        print(f"[!] GET {login_url} returned status {resp.status_code}")
        print(resp.text[:1000])
        return False

    nonce = get_csrf_nonce(resp.text)
    if nonce:
        print(f"[+] Found nonce: {nonce}")
    else:
        print("[!] Could not find csrf nonce, continuing anyway...")

    # Step 2: Build login payload
    payload = {
        "name": username,
        "password": password,
        "_submit": "Submit",
    }
    if nonce:
        payload["nonce"] = nonce

    # Step 3: POST login form
    post_resp = session.post(login_url, data=payload, headers=headers, timeout=10, allow_redirects=True)
    print(f"[+] POST {login_url} -> {post_resp.status_code}")
    
    # Step 4: Check cookies
    print("[*] Cookies after login:")
    if session.cookies:
        for cookie in session.cookies:
            print(f"    {cookie.name} = {cookie.value}")
    else:
        print("    (no cookies set)")

    # Step 5: Try visiting /user to confirm login
    test_urls = ["/user"]
    for path in test_urls:
        test_url = urljoin(site + "/", path)
        check = session.get(test_url, headers=headers, timeout=10)
        print(f"[+] Checking {test_url} -> {check.status_code}")
        if check.status_code == 200 and username.lower() in check.text.lower():
            print(f"[✓] Successfully logged in! Username '{username}' found on {path}.")
            return True

    print("[-] Could not confirm login automatically.")
    return False

def sanitize_filename(name):
    """Sanitize a string to be used as a filename/directory name"""
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^\w\-]', '_', name)
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    return name.strip('_').lower()

def get_event_name(site):
    """Extract event name from site URL"""
    parsed = urlparse(site)
    hostname = parsed.hostname or parsed.path
    # Try to extract meaningful name (e.g., "flare-on12" from "flare-on12.ctfd.io")
    parts = hostname.split('.')
    if len(parts) > 0:
        return sanitize_filename(parts[0])
    return "ctf_event"

def fetch_challenges(site, session):
    """Fetch all challenges from CTFd API"""
    api_url = urljoin(site + "/", "api/v1/challenges")
    print(f"[+] Fetching challenges from {api_url}")
    
    resp = session.get(api_url, timeout=30)
    if resp.status_code != 200:
        print(f"[!] Failed to fetch challenges: {resp.status_code}")
        return None
    
    data = resp.json()
    if not data.get('success'):
        print(f"[!] API returned success=false")
        return None
    
    return data.get('data', [])

def fetch_challenge_details(site, session, challenge_id):
    """Fetch detailed information about a specific challenge"""
    api_url = urljoin(site + "/", f"api/v1/challenges/{challenge_id}")
    print(f"[+] Fetching details for challenge {challenge_id}")
    
    resp = session.get(api_url, timeout=30)
    if resp.status_code != 200:
        print(f"[!] Failed to fetch challenge {challenge_id}: {resp.status_code}")
        return None
    
    data = resp.json()
    if not data.get('success'):
        return None
    
    return data.get('data', {})

def download_file(site, session, file_url, dest_path):
    """Download a file from CTFd"""
    full_url = urljoin(site + "/", file_url)
    print(f"[+] Downloading {full_url}")
    
    try:
        resp = session.get(full_url, timeout=60, stream=True)
        if resp.status_code != 200:
            print(f"[!] Failed to download file: {resp.status_code}")
            return False
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"[✓] Downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"[!] Error downloading file: {e}")
        return False

def parse_and_download_challenges(site, session, event_name, year="2024"):
    """Parse all challenges and create the folder structure"""
    challenges = fetch_challenges(site, session)
    if not challenges:
        print("[!] No challenges found")
        return
    
    print(f"[+] Found {len(challenges)} challenges")
    
    # Create base directory
    base_dir = Path(event_name)
    base_dir.mkdir(exist_ok=True)
    
    master_json = {}
    
    for chal in challenges:
        chal_id = chal.get('id')
        chal_name = chal.get('name', 'unknown')
        chal_category = chal.get('category', 'misc')
        print(f"\n[*] Processing: {chal_name} ({chal_category})")
        
        # Fetch detailed information
        details = fetch_challenge_details(site, session, chal_id)
        if not details:
            print(f"[!] Could not fetch details for {chal_name}")
            continue
        
        # Sanitize names for filesystem
        cat_safe = sanitize_filename(chal_category)
        chal_safe = sanitize_filename(chal_name)
        
        # Create directory structure: event/category/challenge
        chal_dir = base_dir / cat_safe / chal_safe
        chal_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare challenge metadata
        challenge_meta = {
            "name": chal_name,
            "category": chal_category.lower(),
            "description": details.get('description', ''),
            "files": []
        }
        
        # Add optional fields if present
        if 'value' in details:
            challenge_meta['value'] = details['value']
        if 'connection_info' in details and details['connection_info']:
            challenge_meta['connection_info'] = details['connection_info']
        
        # Download files
        if 'files' in details and details['files']:
            for file_info in details['files']:
                file_url = file_info
                if isinstance(file_info, dict):
                    file_url = file_info.get('location', '')
                
                # Extract filename from URL
                filename = os.path.basename(file_url.split('?')[0])
                dest_path = chal_dir / filename
                
                if download_file(site, session, file_url, str(dest_path)):
                    challenge_meta['files'].append(filename)
        
        # Save challenge metadata
        meta_path = chal_dir / 'challenge.json'
        with open(meta_path, 'w') as f:
            json.dump(challenge_meta, f, indent=2)
        print(f"[✓] Saved metadata to {meta_path}")
        
        # Add to master JSON
        key = f"{cat_safe[:3]}-{chal_safe}"
        master_json[key] = {
            "year": year,
            "event": event_name,
            "category": chal_category,
            "challenge": chal_name,
            "path": f"{cat_safe}/{chal_safe}"
        }
    
    # Save master JSON
    master_path = base_dir / 'challenges.json'
    with open(master_path, 'w') as f:
        json.dump(master_json, f, indent=2)
    print(f"\n[✓] Saved master index to {master_path}")
    print(f"[✓] Downloaded {len(master_json)} challenges to {base_dir}/")

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python3 main.py <site> <username> <password> [year]")
        print("Example: python3 main.py flare-on12.ctfd.io user pass 2024")
        sys.exit(1)

    site = normalize_site(sys.argv[1])
    username = sys.argv[2]
    password = sys.argv[3]
    year = sys.argv[4] if len(sys.argv) == 5 else "2024"

    sess = requests.Session()
    try:
        print("[*] Logging in...")
        if not login(site, username, password, sess):
            print("[!] Login may have failed, but continuing anyway...")
        
        print("\n" + "="*50)
        event_name = get_event_name(site)
        print(f"[*] Event name: {event_name}")
        print(f"[*] Starting challenge download...")
        print("="*50 + "\n")
        
        parse_and_download_challenges(site, sess, event_name, year)
        
    except Exception as e:
        print(f"[!] Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
