#!/usr/bin/env python3
import requests
import sys
import re
import os
import json
from urllib.parse import urljoin, urlparse
from pathlib import Path
from bs4 import BeautifulSoup

# --- Dependencies ---
# pip install requests beautifulsoup4
# --------------------

def normalize_site(site):
    """Ensure site URL is in the correct format."""
    if not site.startswith(("http://", "https://")):
        site = "https://" + site
    return site.rstrip('/')

def get_csrf_token(html):
    """Extract the csrfmiddlewaretoken from the HTML."""
    soup = BeautifulSoup(html, "html.parser")
    token_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
    if token_input and 'value' in token_input.attrs:
        return token_input['value']
    
    # Fallback regex
    m = re.search(r'<input[^>]+name=["\']csrfmiddlewaretoken["\'][^>]+value=["\']([^"\']+)["\']', html)
    if m:
        return m.group(1)
    return None

def login(site, username, password, session):
    """Log in to pwnable.xyz."""
    login_url = urljoin(site + "/", "login/") # pwnable.xyz uses /login/
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; pwnable-xyz-parser/1.0)",
        "Referer": login_url,
    }

    # Step 1: Get login page to retrieve CSRF token
    try:
        resp = session.get(login_url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[!] GET {login_url} failed: {e}")
        return False

    token = get_csrf_token(resp.text)
    if token:
        print(f"[+] Found csrfmiddlewaretoken: {token[:10]}...")
    else:
        print("[!] Could not find csrfmiddlewaretoken, continuing anyway...")
        return False # Login will fail without it

    # Step 2: Build login payload
    payload = {
        "username": username, # pwnable.xyz uses 'username'
        "password": password,
        "csrfmiddlewaretoken": token,
    }

    # Step 3: POST login form
    try:
        post_resp = session.post(login_url, data=payload, headers=headers, timeout=10, allow_redirects=True)
        post_resp.raise_for_status()
        print(f"[+] POST {login_url} -> {post_resp.status_code}")
    except requests.RequestException as e:
        print(f"[!] POST {login_url} failed: {e}")
        return False

    # Step 4: Check if login was successful by looking for username on challenge page
    check_url = urljoin(site + "/", "/challenges/")
    try:
        check = session.get(check_url, headers=headers, timeout=10)
        if username.lower() in check.text.lower():
            print(f"[✓] Successfully logged in! Username '{username}' found on {check_url}.")
            return True
        else:
            print(f"[-] Login failed. Could not find '{username}' on challenge page.")
            print("[-] Check your username and password.")
            return False
    except requests.RequestException as e:
        print(f"[!] Failed to verify login at {check_url}: {e}")
        return False


def sanitize_filename(name):
    """Sanitize a string to be used as a filename/directory name"""
    name = str(name)
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^\w\-]', '_', name)
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    return name.strip('_').lower()

def download_file(site, session, file_url, dest_path):
    """Download a file from pwnable.xyz"""
    # file_url from page is relative, urljoin handles it
    full_url = urljoin(site + "/", file_url)
    print(f"[+] Downloading {full_url}")
    
    try:
        resp = session.get(full_url, timeout=60, stream=True)
        resp.raise_for_status()
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"[✓] Downloaded to {dest_path}")
        return True
    except requests.RequestException as e:
        print(f"[!] Failed to download file {full_url}: {e}")
        return False
    except Exception as e:
        print(f"[!] Error saving file: {e}")
        return False

def parse_and_download_challenges(site, session, event_name="pwnable_xyz"):
    """Parse all challenges from the HTML and create the folder structure"""
    challenges_url = urljoin(site + "/", "challenges/")
    print(f"[+] Fetching challenges from {challenges_url}")
    
    try:
        resp = session.get(challenges_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[!] Failed to fetch challenges page: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    modals = soup.find_all("div", class_="modal")
    
    if not modals:
        print("[!] No challenge modals found on the page.")
        return
        
    print(f"[+] Found {len(modals)} challenge modals")
    
    # Create base directory
    base_dir = Path(event_name)
    base_dir.mkdir(exist_ok=True)
    
    master_json = {}
    
    for modal in modals:
        card = modal.find("div", class_="card")
        if not card:
            continue

        try:
            # --- Extract Data from Modal ---
            
            # Header 1: Name and Points
            header1 = card.find("div", class_="card-header")
            chal_name = header1.find("div", recursive=False).get_text(strip=True)
            chal_points = header1.find("div", class_="ml-auto").get_text(strip=True)

            # Description
            chal_desc = card.find("div", class_="card-body").get_text(strip=True)

            # Header 2: Connection, Author, File, Solves
            headers = card.find_all("div", class_="card-header")
            if len(headers) < 2:
                print(f"[!] Skipping {chal_name}: Modal format unexpected.")
                continue
            header2 = headers[1]

            # Connection Info
            conn_div = header2.find("div", recursive=False)
            chal_conn = conn_div.get_text(strip=True) if conn_div and ":" in conn_div.get_text() else None

            # Download Link
            download_link = header2.find("a", href=re.compile(r'/redisfiles/'))
            file_url = download_link['href'] if download_link else None
            
            # Solves
            solves_div = header2.find("div", string=re.compile(r'Solves:'))
            chal_solves = solves_div.get_text(strip=True) if solves_div else "Solves: 0"

            print(f"\n[*] Processing: {chal_name} ({chal_points} pts)")

            # --- Create Structure and Metadata ---
            
            # Set 'pwn' as the category
            cat_safe = "pwn"
            chal_safe = sanitize_filename(chal_name)
            
            # Create directory structure: event/pwn/challenge
            chal_dir = base_dir / cat_safe / chal_safe
            chal_dir.mkdir(parents=True, exist_ok=True)
            
            # --- MODIFICATION ---
            # Combine description and connection info
            full_description = chal_desc
            if chal_conn:
                full_description += f"\n\nConnection Info:\n{chal_conn}"
            # --- END MODIFICATION ---

            # Prepare challenge metadata
            challenge_meta = {
                "name": chal_name,
                "category": "pwn",
                "description": full_description, # <-- Use the combined string
                "value": int(chal_points),
                "solves": int(chal_solves.split(':')[-1].strip()),
                "files": []
            }
            
            # Download file
            if file_url:
                filename = os.path.basename(file_url.split('?')[0])
                dest_path = chal_dir / filename
                
                if download_file(site, session, file_url, str(dest_path)):
                    challenge_meta['files'].append(filename)
            else:
                print("[*] No download file found for this challenge.")
            
            # Save challenge metadata
            meta_path = chal_dir / 'challenge.json'
            with open(meta_path, 'w') as f:
                json.dump(challenge_meta, f, indent=2)
            print(f"[✓] Saved metadata to {meta_path}")
            
            # Add to master JSON
            key = f"{cat_safe}-{chal_safe}"
            master_json[key] = {
                "event": event_name,
                "category": "pwn",
                "challenge": chal_name,
                "path": f"{cat_safe}/{chal_safe}"
            }
        
        except Exception as e:
            chal_name_safe = modal.get('id', 'unknown_modal')
            print(f"[!] Failed to parse modal {chal_name_safe}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save master JSON
    master_path = base_dir / 'challenges.json'
    with open(master_path, 'w') as f:
        json.dump(master_json, f, indent=2)
    print(f"\n[✓] Saved master index to {master_path}")
    print(f"[✓] Downloaded {len(master_json)} challenges to {base_dir}/")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 pwnable_xyz_parser_pwn_desc.py <username> <password>")
        print("Example: python3 pwnable_xyz_parser_pwn_desc.py user pass123")
        sys.exit(1)

    site = normalize_site("https://pwnable.xyz")
    username = sys.argv[1]
    password = sys.argv[2]

    sess = requests.Session()
    try:
        print(f"[*] Logging into {site}...")
        if not login(site, username, password, sess):
            print("[!] Login failed. Exiting.")
            sys.exit(1)
        
        print("\n" + "="*50)
        print(f"[*] Starting challenge download...")
        print("="*5)
        
        parse_and_download_challenges(site, sess)
        
    except Exception as e:
        print(f"[!] Unhandled Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
