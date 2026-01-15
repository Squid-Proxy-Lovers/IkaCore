#!/usr/bin/env python3
"""
MetaCTF Platform Parser
Scrapes challenges from app.metactf.com competitions
"""
import requests
import sys
import re
import os
import json
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Optional, Dict, List, Any


class MetaCTFParser:
    """Parser for MetaCTF platform (app.metactf.com and compete.metactf.com)"""
    
    def __init__(self, base_url: str = "https://app.metactf.com"):
        self.base_url = base_url.rstrip('/')
        self.compete_url = "https://compete.metactf.com"
        self.cache_dir = Path("metactf_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'X-Requested-With': 'XMLHttpRequest',
        })
    
    def login(self, email: str, password: str) -> bool:
        """
        Login to MetaCTF platform using email and password
        Uses the actual API endpoint: /api/meta?a=login
        Logs in to both app and compete subdomains
        """
        print(f"[*] Attempting login as {email}...")
        
        # Login to both subdomains to ensure session works everywhere
        login_urls = [
            (self.base_url, '/api/meta'),
            (self.compete_url, '/api/meta')
        ]
        
        success_count = 0
        
        for base_url, endpoint in login_urls:
            try:
                login_url = urljoin(base_url, endpoint)
                payload = {
                    'email': email,
                    'pass': password
                }
                
                resp = self.session.post(f"{login_url}?a=login", data=payload, timeout=10, allow_redirects=True)
                
                if resp.status_code == 200:
                    response_code = resp.text.strip()
                    
                    # Response code "201" means success
                    if response_code == "201":
                        success_count += 1
                        # Immediately fetch and save dashboard HTML while session is fresh
                        dashboard_resp = self.session.get(urljoin(base_url, '/dashboard'), timeout=10)
                        if 'login_form' not in dashboard_resp.text and len(dashboard_resp.text) > 10000:
                            print(f"[+] Session verified for {base_url}")
                            # Save the dashboard HTML for later use
                            if not hasattr(self, '_dashboards'):
                                self._dashboards = {}
                            self._dashboards[base_url] = dashboard_resp.text
                            
                            # Also cache challenges pages for all competitions found in dashboard
                            self._cache_challenges_pages(base_url)
                        else:
                            print(f"[!] Warning: Session test failed for {base_url}, dashboard size: {len(dashboard_resp.text)}")
                    elif response_code == "302":
                        print(f"[-] Login failed at {base_url}: Incorrect email or password")
                    elif response_code == "230":
                        print(f"[-] Login failed at {base_url}: Too many login attempts. Try again later.")
                    elif response_code == "366":
                        print(f"[-] Login failed at {base_url}: This account must login via SSO")
                    else:
                        print(f"[-] Login failed at {base_url} with code: {response_code}")
                        
            except Exception as e:
                print(f"[!] Login error at {base_url}: {e}")
                continue
        
        if success_count > 0:
            print(f"[+] Login successful! (logged into {success_count} subdomain(s))")
            print(f"[+] Session cookies: {dict(self.session.cookies)}")
            # Print cookie details for debugging
            for cookie in self.session.cookies:
                print(f"[DEBUG] Cookie: {cookie.name}={cookie.value[:10]}... domain={cookie.domain} path={cookie.path}")
            return True
        else:
            print("[-] Login failed on all subdomains")
            return False
    
    def _cache_challenges_pages(self, base_url: str):
        """Cache challenges pages for competitions while session is fresh"""
        if not hasattr(self, '_challenges_cache'):
            self._challenges_cache = {}
        if not hasattr(self, '_auth_tokens'):
            self._auth_tokens = {}
        
        # Extract competition IDs and auth tokens from dashboard
        dashboard_html = self._dashboards.get(base_url, '')
        
        # Find compete.metactf.com URLs with auth tokens
        # Pattern: href="https://compete.metactf.com/289/?auth=true&uid=1061133&tok=245548bbc8a4ea2a28ff426bab250cb6"
        auth_pattern = r'compete\.metactf\.com/(\d+)/\?auth=true&(uid=\d+&tok=[a-f0-9]+)'
        matches = re.findall(auth_pattern, dashboard_html)
        
        print(f"[*] Found {len(matches)} competitions with auth tokens, caching challenges...")
        
        for comp_id, auth_params in matches:
            try:
                # Save auth token
                self._auth_tokens[comp_id] = auth_params
                
                # First, visit the main competition page with auth to establish session
                auth_url = f'https://compete.metactf.com/{comp_id}/?auth=true&{auth_params}'
                print(f"[DEBUG] Authenticating to competition {comp_id}...")
                auth_resp = self.session.get(auth_url, timeout=10, allow_redirects=True)
                
                # Then try to fetch challenges page
                challenges_url = f'https://compete.metactf.com/{comp_id}/problems'
                print(f"[DEBUG] Fetching challenges from {challenges_url}...")
                resp = self.session.get(challenges_url, timeout=10)
                
                # Also try to fetch the API endpoint for problems
                api_url = f'https://compete.metactf.com/{comp_id}/api/problems_json.php'
                print(f"[DEBUG] Fetching API from {api_url}...")
                api_resp = self.session.get(api_url, timeout=10)
                
                # Debug: save first response
                if comp_id == '289':
                    with open('debug_challenges_289.html', 'w', encoding='utf-8') as f:
                        f.write(resp.text)
                    with open('debug_api_289.json', 'w', encoding='utf-8') as f:
                        f.write(api_resp.text)
                        print(f"[DEBUG] Saved API response to debug_api_289.json")
                
                print(f"[DEBUG] HTML Status: {resp.status_code}, Size: {len(resp.text)}")
                print(f"[DEBUG] API Status: {api_resp.status_code}, Size: {len(api_resp.text)}")
                
                # Prefer API response if available
                if api_resp.status_code == 200 and len(api_resp.text) > 100:
                    try:
                        data = api_resp.json()
                        if data:
                            # Save to file cache
                            api_cache_file = self.cache_dir / f"challenges_api_{comp_id}.json"
                            with open(api_cache_file, 'w', encoding='utf-8') as f:
                                f.write(api_resp.text)
                            print(f"[+] Cached challenges API for competition {comp_id} ({len(api_resp.text)} bytes)")
                            continue
                    except Exception as e:
                        print(f"[!] API parse error for {comp_id}: {e}")
                
                # Fall back to HTML
                if resp.status_code == 200 and 'login_form' not in resp.text and len(resp.text) > 5000:
                    html_cache_file = self.cache_dir / f"challenges_{comp_id}.html"
                    with open(html_cache_file, 'w', encoding='utf-8') as f:
                        f.write(resp.text)
                    print(f"[+] Cached challenges page for competition {comp_id} ({len(resp.text)} bytes)")
                else:
                    print(f"[!] Could not cache {comp_id}: redirected to login or failed (size: {len(resp.text)})")
            except Exception as e:
                print(f"[DEBUG] Could not cache challenges for {comp_id}: {e}")
    
    def _parse_dashboards(self) -> List[Dict[str, Any]]:
        """Parse competitions from cached dashboard HTML"""
        all_competitions = []
        seen_ids = set()
        
        for base_url, html in self._dashboards.items():
            print(f"[*] Parsing dashboard from {base_url}")
            
            # Pattern 1: Your competitions - compete.metactf.com links
            compete_pattern = r'card-title">([^<]+?)&nbsp;&nbsp;<a\s+href="https://compete\.metactf\.com/(\d+)'
            for match in re.finditer(compete_pattern, html):
                comp_id = match.group(2)
                if comp_id not in seen_ids:
                    seen_ids.add(comp_id)
                    name = match.group(1).strip()
                    name = re.sub(r'\s+', ' ', name).replace('&nbsp;', ' ').strip()
                    all_competitions.append({'id': comp_id, 'name': name, 'source': 'compete'})
            
            # Pattern 2: Available events - /signup?e=123
            signup_pattern = r'<h4[^>]*class="card-title"[^>]*>([^<]+)</h4>.*?/signup\?e=(\d+)'
            for match in re.finditer(signup_pattern, html, re.DOTALL):
                comp_id = match.group(2)
                if comp_id not in seen_ids:
                    seen_ids.add(comp_id)
                    name = match.group(1).strip()
                    name = re.sub(r'\s+', ' ', name).replace('&nbsp;', ' ').strip()
                    all_competitions.append({'id': comp_id, 'name': name, 'source': 'signup'})
        
        if all_competitions:
            all_competitions.sort(key=lambda x: int(x.get('id', 0)))
            print(f"[+] Found {len(all_competitions)} total competition(s)")
            return all_competitions
        
        return []
    
    
    def fetch_competitions(self) -> List[Dict[str, Any]]:
        """Fetch list of available competitions from both app and compete subdomains"""
        print("[*] Fetching competitions...")
        
        # Use cached dashboard HTML if available
        if hasattr(self, '_dashboards'):
            return self._parse_dashboards()
        
        
        # Check both app.metactf.com and compete.metactf.com
        base_urls = [self.base_url, self.compete_url]
        endpoints = [
            '/dashboard',
            '/api/competitions',
            '/api/v1/competitions',
            '/competitions',
            '/api/events',
            '/'
        ]
        
        all_competitions = []
        seen_ids = set()
        
        for base in base_urls:
            for endpoint in endpoints:
                try:
                    url = urljoin(base, endpoint)
                    # Debug: show cookies being sent
                    cookies_for_url = {c.name: c.value[:10] + '...' for c in self.session.cookies if c.domain in url}
                    print(f"[DEBUG] Fetching {url} with cookies: {cookies_for_url}")
                    resp = self.session.get(url, timeout=30)
                    
                    if resp.status_code == 200:
                        # Check if we got redirected to login page
                        if '/login' in resp.url or 'login_form' in resp.text:
                            print(f"[DEBUG] {url} redirected to login, skipping")
                            continue
                        
                        try:
                            data = resp.json()
                            if isinstance(data, list):
                                for comp in data:
                                    comp_id = str(comp.get('id', ''))
                                    if comp_id and comp_id not in seen_ids:
                                        seen_ids.add(comp_id)
                                        all_competitions.append(comp)
                                continue
                            elif isinstance(data, dict) and 'data' in data:
                                for comp in data['data']:
                                    comp_id = str(comp.get('id', ''))
                                    if comp_id and comp_id not in seen_ids:
                                        seen_ids.add(comp_id)
                                        all_competitions.append(comp)
                                continue
                        except json.JSONDecodeError:
                            # Try to parse HTML for competition links and names
                            # Pattern 1: Available events - /signup?e=123
                            # Pattern 2: Your competitions - compete.metactf.com/123
                            # Pattern 3: Direct links - /289/problems
                            
                            print(f"[DEBUG] Parsing HTML from {url}")
                            
                            # First, find compete.metactf.com links with names
                            # <h4 class="card-title">Name&nbsp;&nbsp;<a href="https://compete.metactf.com/123/...">
                            # HTML is minified, so pattern needs to be more flexible
                            compete_pattern = r'class="card-title">([^<]+?)&nbsp;&nbsp;<a\s+href="https://compete\.metactf\.com/(\d+)'
                            compete_matches = list(re.finditer(compete_pattern, resp.text))
                            print(f"[DEBUG] Found {len(compete_matches)} compete.metactf.com matches")
                            
                            for match in compete_matches:
                                comp_id = match.group(2)
                                if comp_id not in seen_ids:
                                    seen_ids.add(comp_id)
                                    name = match.group(1).strip()
                                    name = re.sub(r'\s+', ' ', name).replace('&nbsp;', ' ').strip()
                                    print(f"[DEBUG] Added competition {comp_id}: {name}")
                                    all_competitions.append({'id': comp_id, 'name': name, 'source': 'compete'})
                            
                            # Then find signup links with names
                            # <h4 class="card-title">Name</h4>...<a href="/signup?e=123">
                            signup_pattern = r'<h4[^>]*class="card-title"[^>]*>([^<]+)</h4>.*?/signup\?e=(\d+)'
                            signup_matches = list(re.finditer(signup_pattern, resp.text, re.DOTALL))
                            print(f"[DEBUG] Found {len(signup_matches)} signup matches")
                            
                            for match in signup_matches:
                                comp_id = match.group(2)
                                if comp_id not in seen_ids:
                                    seen_ids.add(comp_id)
                                    name = match.group(1).strip()
                                    name = re.sub(r'\s+', ' ', name).replace('&nbsp;', ' ').strip()
                                    print(f"[DEBUG] Added competition {comp_id}: {name}")
                                    all_competitions.append({'id': comp_id, 'name': name, 'source': 'signup'})
                            
                            # Fallback: Look for other patterns
                            fallback_patterns = [
                                r'/(\d+)/problems',
                                r'/event[s]?/(\d+)',
                                r'/competition[s]?/(\d+)',
                            ]
                            
                            comp_ids = set()
                            for pattern in fallback_patterns:
                                matches = re.findall(pattern, resp.text)
                                comp_ids.update(matches)
                            
                            for comp_id in comp_ids:
                                if comp_id not in seen_ids:
                                    seen_ids.add(comp_id)
                                    all_competitions.append({'id': comp_id, 'name': f'Competition {comp_id}', 'source': 'fallback'})
                            
                except Exception as e:
                    print(f"[!] Error fetching {url}: {e}")
                    continue
        
        if all_competitions:
            # Sort by ID
            all_competitions.sort(key=lambda x: int(x.get('id', 0)))
            print(f"[+] Found {len(all_competitions)} total competition(s)")
            return all_competitions
        
        print("[-] Could not fetch competitions")
        return []
    
    def fetch_challenges(self, competition_id: str) -> List[Dict[str, Any]]:
        """Fetch challenges for a specific competition"""
        print(f"[*] Fetching challenges for competition {competition_id}...")
        
        # Check if we have cached API data (preferred)
        api_cache_file = self.cache_dir / f"challenges_api_{competition_id}.json"
        if api_cache_file.exists():
            try:
                print(f"[*] Loading cached API data for competition {competition_id}")
                with open(api_cache_file, 'r', encoding='utf-8') as f:
                    api_data = json.load(f)
                
                print(f"[DEBUG] API data keys: {list(api_data.keys())}")
                if 'problems' in api_data:  # MetaCTF uses 'problems' not 'challenges'
                    challenges = self._parse_challenges_json(api_data, competition_id)
                    if challenges:
                        print(f"[+] Parsed {len(challenges)} challenges from API cache")
                        return challenges
                    else:
                        print(f"[!] _parse_challenges_json returned empty list")
            except Exception as e:
                import traceback
                print(f"[!] Error loading API cache: {e}")
                traceback.print_exc()
        
        # Fall back to HTML cache
        html_cache_file = self.cache_dir / f"challenges_{competition_id}.html"
        if html_cache_file.exists():
            try:
                print(f"[*] Loading cached HTML for competition {competition_id}")
                with open(html_cache_file, 'r', encoding='utf-8') as f:
                    html = f.read()
                challenges = self._parse_challenges_html(html, competition_id)
                if challenges:
                    print(f"[+] Parsed {len(challenges)} challenges from HTML cache")
                    return challenges
            except Exception as e:
                print(f"[!] Error loading HTML cache: {e}")
        
        # If nothing cached, try to fetch
        print(f"[-] No cached data found for competition {competition_id}")
        print(f"[!] Please run login first to cache challenge data")
        return []
    
    def _parse_challenges_json(self, data: Dict[str, Any], comp_id: str) -> List[Dict[str, Any]]:
        """Parse challenges from API JSON response"""
        challenges = []
        
        # MetaCTF uses 'problems' key
        problems_key = 'problems' if 'problems' in data else 'challenges'
        if problems_key not in data:
            print(f"[!] No 'problems' or 'challenges' key in API response")
            return challenges
        
        for chal in data[problems_key]:
            # Extract file URLs from description HTML
            files = []
            desc_html = chal.get('description', '')
            file_pattern = r'href=["\']([^"\']*(?:metaproblems\.com|mctf\.io|cyberlabhost\.com|static\.metaproblems\.com)[^"\']*)["\']'
            file_matches = re.findall(file_pattern, desc_html)
            for url in file_matches:
                # Skip non-download links
                if any(skip in url.lower() for skip in ['javascript:', 'mailto:', '#', 'api/get_jwt']):
                    continue
                files.append(url)
            
            challenge = {
                'id': str(chal.get('id', '')),
                'title': chal.get('title', ''),
                'category': chal.get('category', ''),
                'points': chal.get('points', 0),
                'description': desc_html,
                'solved_by': chal.get('solved_by', 0),
                'files': files,
                'competition_id': comp_id
            }
            challenges.append(challenge)
        
        return challenges
    
    def OLD_fetch_challenges_live(self, competition_id: str) -> List[Dict[str, Any]]:
        """OLD METHOD - Fetch challenges for a specific competition from both app and compete subdomains"""
        print(f"[*] Fetching challenges for competition {competition_id}...")
        
        # Try both base URLs (app and compete subdomains)
        base_urls = [self.base_url, self.compete_url]
        
        # Different endpoint patterns to try
        endpoints = [
            f'/{competition_id}/problems',  # compete.metactf.com uses this
            f'/api/competitions/{competition_id}/challenges',
            f'/api/v1/competitions/{competition_id}/challenges',
            f'/api/challenges?competition={competition_id}',
            f'/competitions/{competition_id}/challenges',
            f'/api/problems?competition={competition_id}',
        ]
        
        for base in base_urls:
            for endpoint in endpoints:
                try:
                    url = urljoin(base, endpoint)
                    print(f"[DEBUG] Trying {url}")
                    resp = self.session.get(url, timeout=30)
                    
                    if resp.status_code == 200:
                        # Check if redirected to login
                        if 'login_form' in resp.text:
                            print(f"[DEBUG] {url} requires authentication, skipping")
                            continue
                        
                        try:
                            data = resp.json()
                            if isinstance(data, list) and len(data) > 0:
                                print(f"[+] Found {len(data)} challenges via {endpoint}")
                                return data
                            elif isinstance(data, dict):
                                if 'data' in data and data['data']:
                                    print(f"[+] Found {len(data['data'])} challenges via {endpoint}")
                                    return data['data']
                                elif 'challenges' in data and data['challenges']:
                                    print(f"[+] Found {len(data['challenges'])} challenges via {endpoint}")
                                    return data['challenges']
                                elif 'problems' in data and data['problems']:
                                    print(f"[+] Found {len(data['problems'])} challenges via {endpoint}")
                                    return data['problems']
                        except json.JSONDecodeError:
                            # Try parsing HTML for challenges
                            if '/problems' in endpoint and 'challenge-card' in resp.text or 'problem-card' in resp.text:
                                print(f"[*] Found HTML challenges page, attempting to parse...")
                                challenges = self._parse_challenges_html(resp.text, competition_id)
                                if challenges:
                                    print(f"[+] Parsed {len(challenges)} challenges from HTML")
                                    return challenges
                            continue
                            
                except Exception as e:
                    print(f"[DEBUG] Error fetching {url}: {e}")
                    continue
        
        print(f"[-] Could not fetch challenges for competition {competition_id}")
        return []
    
    def _parse_challenges_html(self, html: str, competition_id: str) -> List[Dict[str, Any]]:
        """Parse challenges from HTML page"""
        challenges = []
        
        # Try to find challenge cards in HTML
        # Pattern: Look for challenge/problem cards with names and point values
        card_pattern = r'<div[^>]*(?:challenge-card|problem-card)[^>]*>.*?<h\d[^>]*>([^<]+)</h\d>.*?(\d+)\s*points?'
        matches = re.findall(card_pattern, html, re.DOTALL | re.IGNORECASE)
        
        for idx, (name, points) in enumerate(matches):
            challenges.append({
                'id': f'{competition_id}_{idx}',
                'name': name.strip(),
                'points': int(points),
                'description': '',
                'category': 'Unknown'
            })
        
        return challenges
    
    def download_file(self, file_url: str, dest_path: str) -> bool:
        """Download a challenge file"""
        full_url = urljoin(self.base_url, file_url)
        print(f"[+] Downloading {full_url}")
        
        try:
            resp = self.session.get(full_url, timeout=60, stream=True)
            if resp.status_code != 200:
                print(f"[!] Failed to download: {resp.status_code}")
                return False
            
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"[✓] Downloaded to {dest_path}")
            return True
            
        except Exception as e:
            print(f"[!] Download error: {e}")
            return False
    
    def sanitize_filename(self, name: str) -> str:
        """Sanitize string for filesystem"""
        name = re.sub(r'[^\w\-]', '_', name)
        name = re.sub(r'_+', '_', name)
        return name.strip('_').lower()
    
    def parse_competition(self, competition_id: str, output_dir: str = "metactf_challenges") -> bool:
        """Parse all challenges from a competition"""
        challenges = self.fetch_challenges(competition_id)
        
        if not challenges:
            print("[!] No challenges found")
            return False
        
        base_dir = Path(output_dir) / f"competition_{competition_id}"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        master_json = {}
        
        for chal in challenges:
            # Handle both dict and string formats
            if isinstance(chal, str):
                chal_name = chal
                chal_id = chal
                category = "misc"
            else:
                chal_id = chal.get('id', chal.get('challenge_id', 'unknown'))
                chal_name = chal.get('name', chal.get('title', f'challenge_{chal_id}'))
                category = chal.get('category', chal.get('type', 'misc'))
            
            print(f"\n[*] Processing: {chal_name} ({category})")
            
            cat_safe = self.sanitize_filename(str(category))
            chal_safe = self.sanitize_filename(str(chal_name))
            
            chal_dir = base_dir / cat_safe / chal_safe
            chal_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare metadata
            challenge_meta = {
                "name": chal_name,
                "category": category,
                "challenge_id": chal_id,
                "competition_id": competition_id,
                "files": []
            }
            
            # Add optional fields
            if isinstance(chal, dict):
                for key in ['description', 'points', 'value', 'tags', 'solves', 'author']:
                    if key in chal:
                        challenge_meta[key] = chal[key]
                
                # Download files if available
                file_fields = ['files', 'file', 'attachments', 'downloads']
                for field in file_fields:
                    if field in chal and chal[field]:
                        files = chal[field] if isinstance(chal[field], list) else [chal[field]]
                        
                        for file_info in files:
                            file_url = file_info
                            if isinstance(file_info, dict):
                                file_url = file_info.get('url', file_info.get('location', ''))
                            
                            if file_url:
                                filename = os.path.basename(file_url.split('?')[0])
                                dest_path = chal_dir / filename
                                
                                if self.download_file(file_url, str(dest_path)):
                                    challenge_meta['files'].append(filename)
            
            # Save challenge metadata
            meta_path = chal_dir / 'challenge.json'
            with open(meta_path, 'w') as f:
                json.dump(challenge_meta, f, indent=2)
            print(f"[✓] Saved metadata to {meta_path}")
            
            # Add to master index
            key = f"{cat_safe[:3]}-{chal_safe}"
            master_json[key] = {
                "year": "2024",  # Update as needed
                "event": f"metactf_{competition_id}",
                "category": category,
                "challenge": chal_name,
                "path": f"{cat_safe}/{chal_safe}"
            }
        
        # Save master JSON
        master_path = base_dir / 'challenges.json'
        with open(master_path, 'w') as f:
            json.dump(master_json, f, indent=2)
        print(f"\n[✓] Saved master index to {master_path}")
        print(f"[✓] Downloaded {len(master_json)} challenges to {base_dir}/")
        
        return True


def main():
    if len(sys.argv) < 2:
        print("MetaCTF Parser - Download challenges from app.metactf.com")
        print("\nUsage:")
        print("  python3 parsemetactf.py <competition_id> [email] [password]")
        print("  python3 parsemetactf.py --list [email] [password]")
        print("\nExamples:")
        print("  python3 parsemetactf.py 123")
        print("  python3 parsemetactf.py 123 user@email.com password")
        print("  python3 parsemetactf.py --list user@email.com password")
        print("\nNote: Some competitions may require authentication")
        print("      Use your MetaCTF email and password for login")
        sys.exit(1)
    
    parser = MetaCTFParser()
    
    # Handle --list flag
    if sys.argv[1] == '--list':
        if len(sys.argv) >= 4:
            email = sys.argv[2]
            password = sys.argv[3]
            if parser.login(email, password):
                competitions = parser.fetch_competitions()
                print("\n[*] Available competitions:")
                for comp in competitions:
                    comp_id = comp.get('id', 'unknown')
                    comp_name = comp.get('name', 'Unknown')
                    print(f"  [{comp_id}] {comp_name}")
        else:
            print("[*] Fetching public competitions...")
            competitions = parser.fetch_competitions()
            print("\n[*] Available competitions:")
            for comp in competitions:
                comp_id = comp.get('id', 'unknown')
                comp_name = comp.get('name', 'Unknown')
                print(f"  [{comp_id}] {comp_name}")
        sys.exit(0)
    
    competition_id = sys.argv[1]
    
    # Login if credentials provided
    if len(sys.argv) >= 4:
        email = sys.argv[2]
        password = sys.argv[3]
        if not parser.login(email, password):
            print("[!] Warning: Login failed, continuing without authentication...")
    
    # Parse competition
    try:
        print(f"\n[*] Parsing competition {competition_id}...")
        success = parser.parse_competition(competition_id)
        
        if success:
            print("\n[✓] Parsing complete!")
        else:
            print("\n[!] Parsing failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
