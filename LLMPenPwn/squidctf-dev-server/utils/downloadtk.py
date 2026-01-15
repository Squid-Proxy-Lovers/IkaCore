#!/usr/bin/env python3
import requests
import sys
import re
import os
import json
from urllib.parse import urljoin
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

def login(site, username, password, session):
    """
    Log in to pwnable.tw.
    NOTE: pwnable.tw uses ReCaptcha. Automated login via script often fails 
    unless you already have a valid session cookie. 
    This function attempts a standard login but validates against the profile page.
    """
    login_url = urljoin(site, "/user/login")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": site + "/",
    }

    # Step 1: Get login page to set initial cookies
    try:
        session.get(login_url, headers=headers, timeout=10)
    except requests.RequestException as e:
        print(f"[!] GET {login_url} failed: {e}")
        return False

    # Step 2: Build login payload
    payload = {
        "username": username,
        "password": password,
        # 'g-recaptcha-response': '' # Captcha usually prevents this from working purely in python
    }

    # Step 3: POST login form
    try:
        print(f"[*] Attempting login (Note: If this fails, export your PHPSESSID cookie)")
        post_resp = session.post(login_url, data=payload, headers=headers, timeout=10)
    except requests.RequestException as e:
        print(f"[!] POST {login_url} failed: {e}")
        return False

    # Step 4: Check if login was successful
    # On pwnable.tw, a successful login usually shows "Logout" or the username in the navbar
    if "Logout" in post_resp.text or username in post_resp.text:
        print(f"[✓] Login successful!")
        return True
    
    # Fallback: Check if we are already authenticated via cookies passed externally
    if "Logout" in session.get(site, headers=headers).text:
         print(f"[✓] Session is already authenticated.")
         return True

    print("[-] Login failed. pwnable.tw requires ReCaptcha.")
    print("[-] TIP: Log in via browser, get the 'PHPSESSID' cookie, and edit this script to set it manually.")
    return False

def sanitize_filename(name):
    """Sanitize a string to be used as a filename/directory name"""
    name = str(name)
    name = re.sub(r'[^\w\-]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_').lower()

def download_file(site, session, file_url, dest_path):
    """Download a file from pwnable.tw"""
    full_url = urljoin(site, file_url)
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

def parse_and_download_challenges(site, session, event_name="pwnable_tw"):
    """Parse all challenges from the HTML and create the folder structure"""
    # pwnable.tw lists challenges on the main page or /challenge/
    challenges_url = urljoin(site, "/challenge/")
    print(f"[+] Fetching challenges from {challenges_url}")
    
    try:
        resp = session.get(challenges_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[!] Failed to fetch challenges page: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    
    # pwnable.tw uses a UL with ID 'challenge-list' and LI items
    chal_list = soup.find("ul", id="challenge-list")
    if not chal_list:
        print("[!] Could not find 'challenge-list' in HTML.")
        return

    entries = chal_list.find_all("li", class_="challenge-entry")
    
    if not entries:
        print("[!] No challenges found.")
        return
        
    print(f"[+] Found {len(entries)} challenges")
    
    # Create base directory
    base_dir = Path(event_name)
    base_dir.mkdir(exist_ok=True)
    
    master_json = {}
    
    for entry in entries:
        try:
            # --- Extract Data from Entry ---
            
            # 1. Info container
            info = entry.find("div", class_="challenge-info")
            if not info:
                continue

            # 2. Title and Score
            title_div = info.find("div", class_="title")
            chal_name = title_div.find("span", class_="tititle").get_text(strip=True)
            score_text = title_div.find("span", class_="score").get_text(strip=True)
            
            # Clean up score (e.g. "100 pts" -> 100)
            try:
                chal_points = int(re.search(r'\d+', score_text).group())
            except:
                chal_points = 0

            # 3. Description (Hidden div)
            desc_div = info.find("div", class_="description")
            # Get text preserving some structure for reading
            chal_desc = desc_div.get_text("\n", strip=True) if desc_div else "No description"
            
            # 4. Solves
            solves_span = info.find("span", class_="solved_times")
            solves_text = solves_span.get_text(strip=True) if solves_span else "0"
            try:
                chal_solves = int(re.search(r'\d+', solves_text).group())
            except:
                chal_solves = 0

            print(f"\n[*] Processing: {chal_name} ({chal_points} pts)")

            # --- Create Structure ---
            
            cat_safe = "pwn" # pwnable.tw is all pwn
            chal_safe = sanitize_filename(chal_name)
            
            chal_dir = base_dir / cat_safe / chal_safe
            chal_dir.mkdir(parents=True, exist_ok=True)
            
            # --- Files ---
            # Files are <a> tags inside the description div
            # Common extensions on pwnable.tw: no extension (binary), .so, .tgz, .tar.gz
            files_to_download = []
            links = info.find_all("a")
            
            for link in links:
                href = link.get('href')
                
                # Filter logic
                if href:
                    # Standard pwnable.tw static paths
                    if '/static/chall/' in href or '/static/libc/' in href:
                        if href not in files_to_download:
                            files_to_download.append(href)
                    
                    # Fallback for files without standard prefixes
                    # checking context to ensure we don't grab garbage links
                    elif not href.startswith(('http', 'mailto', '#', '/user', '/writeup', 'javascript')):
                         if href not in files_to_download:
                            files_to_download.append(href)
            challenge_meta = {
                "name": chal_name,
                "category": "pwn",
                "description": chal_desc,
                "value": chal_points,
                "solves": chal_solves,
                "files": []
            }
            
            # Download files
            for file_url in files_to_download:
                filename = os.path.basename(file_url.split('?')[0])
                dest_path = chal_dir / filename
                
                if download_file(site, session, file_url, str(dest_path)):
                    challenge_meta['files'].append(filename)
            
            # Save metadata
            meta_path = chal_dir / 'challenge.json'
            with open(meta_path, 'w') as f:
                json.dump(challenge_meta, f, indent=2)
            
            # Add to master JSON
            key = f"{cat_safe}-{chal_safe}"
            master_json[key] = {
                "event": event_name,
                "category": "pwn",
                "challenge": chal_name,
                "path": f"{cat_safe}/{chal_safe}"
            }
        
        except Exception as e:
            print(f"[!] Failed to parse entry {chal_name if 'chal_name' in locals() else 'unknown'}: {e}")

    # Save master JSON
    master_path = base_dir / 'challenges.json'
    with open(master_path, 'w') as f:
        json.dump(master_json, f, indent=2)
    print(f"\n[✓] Saved master index to {master_path}")

if __name__ == "__main__":
    # You can hardcode cookies here if login fails
    # session_cookies = {'PHPSESSID': 'your_cookie_here'}
    session_cookies = None

    if len(sys.argv) < 3 and not session_cookies:
        print("Usage: python3 pwnable_tw_parser.py <username> <password>")
        print("Or edit the script to hardcode PHPSESSID cookie.")
        sys.exit(1)

    site = normalize_site("https://pwnable.tw")
    sess = requests.Session()
    
    if session_cookies:
        sess.cookies.update(session_cookies)
        print("[*] Using provided session cookies.")
    else:
        username = sys.argv[1]
        password = sys.argv[2]
        print(f"[*] Logging into {site}...")
        login(site, username, password, sess)
        # We proceed even if login fails, as some info is public

    try:        
        print("\n" + "="*50)
        print(f"[*] Starting challenge download...")
        print("="*5)
        
        parse_and_download_challenges(site, sess)
        
    except Exception as e:
        print(f"[!] Unhandled Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)