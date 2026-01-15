import requests
import json
import re
from typing import Optional, List, Dict, Any
from datetime import datetime

# ==========================================
# Original Class Definition (Preserved)
# ==========================================

class LeakTown:
    """Client for interacting with leak.town API"""
    
    def __init__(self, token: Optional[str] = None, host: str = "https://leak.town/"):
        """Initialize client. Creates new webhook if no token provided."""
        self.token = token
        self.session = requests.Session()
        
        # Set base URLs
        self.base = host.rstrip('/')
        self.api = f"{self.base}/api"
        self._url = None
        
        if not self.token:
            self.create_new_webhook()
            self.extend()
    
    def create_new_webhook(self) -> Dict[str, Any]:
        """Create a new webhook."""
        response = self.session.post(f"{self.api}/town/create",json={"duration":3600})
        response.raise_for_status()
        
        data = response.json()
        # print(data) # Commented out for tool usage to keep stdout clean
        assert data['success'], "token failed"
        self.token = data['data']['token']

        # Print statements removed/commented to rely on tool return values
        # print(f"✓ New webhook created!")
        # ...
        
        return data
    
    @property
    def url(self) -> str:
        """get base url"""
        if self._url is None:
            r = self.session.get(f"{self.api}/town/info", **self.args)
            data = r.json()
            assert data['success'], "info failed"
            self._url = data['data']['url']
        return self._url

    @property
    def args(self) -> Dict[str, Any]:
        """default arguments to pass to http requests"""
        return {
            "params":{
                'token': self.token
            },
            "json": {
                'token': self.token
            }
        }
        
    def extend(self) -> bool:
        """attempt webhook extension"""
        r = self.session.post(f"{self.api}/town/extend",**self.args)
        return r.json()['success']
    
    def get_requests(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all requests sent to this webhook."""
        response = self.session.get(f"{self.api}/request/list", **self.args)
        data = response.json()
        assert data['success'], data
        # for now we only return HTTP requests, we can work on email/dns later
        return [x for x in data['data'] if x['type'] == 'http'][:limit]

    def view_requests(self, limit: int = 10, raw: bool = False):
        """View requests (formatted or raw JSON)."""
        # Note: In tool context, we will use get_requests and format strings manually
        # to return to the LLM, rather than printing to stdout.
        requests_list = self.get_requests(limit=limit)
        return requests_list

    def delete_all_requests(self) -> bool:
        """Delete all requests for this webhook."""
        response = self.session.post(f"{self.api}/request/clear",**self.args)
        assert response.json()['success']
        return True

    def get_response_config(self) -> Dict[str, Any]:
        """get current configuration for response data"""
        response = self.session.get(f"{self.api}/endpoint/get",**self.args)
        data = response.json()
        assert data['success']
        return data['data']
    
    def view_response_config(self) -> None:
        """pretty-print current response config"""
        # Note: Replaced by tool logic to return string
        conf = self.get_response_config()
        return conf
    
    def update_response_config(self, content: str = None, headers: Dict[str,str] = None):
        """
        update current response config
        """
        current = self.get_response_config()[0]
        if content:
            current['handler']['data'] = content
        if headers:
            # the site requires headers to be lists: { "key": ["value"] }
            headers = {k:[v] for k,v in headers.items()}
            current['handler']['headers'].update(headers)
            
        r = self.session.post(f"{self.api}/endpoint/edit", json={
            "token": self.token,
            "id": current['id'],
            "endpoint": current
        })
        assert r.json()['success']