import requests
import json
import re
from typing import List, Dict, Optional, Union
from urllib.parse import urljoin

GP_BASE_URL = 'https://proxy.uscg.win'

def _request(method: str, path: str, *, error: str, **kwargs) -> Optional[requests.Response]:
    url = urljoin(GP_BASE_URL, path.rstrip('/'))
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        print(f"{error}: {exc}")
        return None
    
def get_info() -> Dict:
    response = _request("GET", '/info', error="Error fetching info")
    if not response:
        return {}
    return response.json()

def get_teams(id: Union[int, str] = "") -> List[Dict]:
    response = _request("GET", f'/teams/{id}', error="Error fetching teams")
    if not response:
        return []
    return response.json()

def get_services(id: Union[int, str] = "") -> List[Dict]:
    response = _request("GET", f'/services/{id}', error="Error fetching services")
    if not response:
        return []
    return response.json()

def get_flagids(service_id: Union[int, str] = "", team_id: Union[int, str] = "") -> List[Dict]:
    path = '/flagids'
    if service_id and team_id:
        path += f'/{service_id}/{team_id}'
    elif service_id:
        path += f'/{service_id}'
    elif team_id:
        raise ValueError(f"team_id [{team_id}] requires service_id")
    response = _request("GET", path, error="Error fetching flagids")
    if not response:
        return []
    return response.json()

def get_latest_nop_flagids(service_name: str) -> str:
    flag_regex = get_info().get("flag_regex", r"FLAG\{[A-Za-z0-9_]{31}=\}")

    def remove_suffix(s: str) -> str:
        return re.sub(r'-[0-9]+$', '', s)

    services = get_services()
    services = [s for s in services.values() if remove_suffix(s.get("name")) == service_name]
    
    if len(services) == 0:
        raise ValueError(f"Service with name '{service_name}' not found in proxy API.")
    
    if len(services) == 1 and not services[0].get('uses_flagids'):
        return {}
    
    service_ids = [s['id'] for s in services if s.get('uses_flagids')]
    
    team = get_teams(id="nop")
    service_ip = team.get("services", {}).get(service_ids[0]) if service_ids else None

    if not service_ip:
        raise ValueError(f"Service with id '{service_ids[0]}' not found for team 'nop'.")

    total_flagids = []
    for service_id in service_ids:
        flagids = get_flagids(service_id=service_id, team_id="nop")
        if not flagids:
            raise ValueError(f"No flagids found for service_id '{service_id}' and team_id 'nop'.")
        flagids = {int(k): v for k, v in flagids.items()}
        total_flagids.append(flagids[max(flagids.keys())])
    return flag_regex, service_ip, total_flagids