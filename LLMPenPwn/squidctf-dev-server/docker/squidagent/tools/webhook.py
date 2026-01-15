#!/usr/bin/env python3
"""
Webhook.site Python Tool

A simple tool to interact with webhook.site API.
"""

import requests
import json
from typing import Optional, List, Dict, Any


class WebhookSite:
    """Client for interacting with webhook.site API"""
    
    def __init__(self, token: Optional[str] = None, host: str = "https://webhook.site"):
        """Initialize client. Creates new webhook if no token provided."""
        self.token = token
        self.session = requests.Session()
        
        # Set base URLs
        self.base_url = host.rstrip('/')
        self.api_base = f"{self.base_url}/token"
        
        if not self.token:
            self.create_new_webhook()
    
    def create_new_webhook(self) -> Dict[str, Any]:
        """Create a new webhook."""
        response = self.session.post(f"{self.api_base}")
        response.raise_for_status()
        
        data = response.json()
        self.token = data['uuid']
        
        print(f"✓ New webhook created!")
        print(f"  UUID: {self.token}")
        print(f"  URL: {self.get_url()}")
        print(f"  Email: {self.get_email()}")
        
        return data
    
    def get_url(self) -> str:
        """Get the webhook URL"""
        return f"{self.base_url}/{self.token}"
    
    def get_email(self) -> str:
        """Get the webhook email address"""
        return f"{self.token}@email.webhook.site"
    
    def get_requests(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all requests sent to this webhook."""
        url = f"{self.api_base}/{self.token}/requests"
        params = {'per_page': limit}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data.get('data', [])
    
    def view_requests(self, limit: int = 10, raw: bool = False):
        """View requests (formatted or raw JSON)."""
        requests_list = self.get_requests(limit=limit)
        
        if not requests_list:
            if raw:
                print("[]")
            else:
                print("No requests received yet.")
            return []
        
        if raw:
            print(json.dumps(requests_list, indent=2))
        else:
            print(f"\n{'='*80}")
            print(f"Webhook Requests ({len(requests_list)} shown)")
            print(f"{'='*80}\n")
            
            for idx, req in enumerate(requests_list, 1):
                # Extract path from URL
                url = req.get('url', '')
                path = url.split('?')[0] if url else 'N/A'
                if '://' in path:
                    path = '/' + '/'.join(path.split('/')[3:])
                
                print(f"[{idx}] {req.get('method', 'N/A')} - {req.get('created_at', 'N/A')}")
                print(f"    Path: {path}")
                
                # URL params
                query = req.get('query', {})
                if query:
                    params_str = ', '.join([f"{k}={v}" for k, v in query.items()])
                    print(f"    Params: {params_str}")
                
                # User agent
                user_agent = req.get('user_agent')
                if user_agent:
                    print(f"    User-Agent: {user_agent}")
                
                # POST body
                content = req.get('content', '')
                if content:
                    print(f"    Body: {content[:200]}")
                
                print()
        
        return requests_list
    
    def delete_request(self, request_id: str) -> bool:
        """Delete a specific request by UUID."""
        url = f"{self.api_base}/{self.token}/request/{request_id}"
        response = self.session.delete(url)
        response.raise_for_status()
        
        print(f"✓ Request {request_id[:8]} deleted")
        return True
    
    def delete_request_by_index(self, index: int) -> bool:
        """Delete a request by its index number (1-based)."""
        requests_list = self.get_requests(limit=100)
        
        if index < 1 or index > len(requests_list):
            print(f"✗ Invalid index {index}. Valid range: 1-{len(requests_list)}")
            return False
        
        request_id = requests_list[index - 1]['uuid']
        return self.delete_request(request_id)
    
    def delete_all_requests(self) -> bool:
        """Delete all requests for this webhook."""
        url = f"{self.api_base}/{self.token}/requests"
        response = self.session.delete(url)
        response.raise_for_status()
        
        print(f"✓ All requests cleared")
        return True
    
    def delete_webhook(self) -> bool:
        """Delete the webhook itself."""
        url = f"{self.api_base}/{self.token}"
        response = self.session.delete(url)
        response.raise_for_status()
        
        print(f"✓ Webhook deleted")
        self.token = None
        return True


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Webhook.site Python Tool')
    parser.add_argument('--token', '-t', help='Existing webhook token/UUID')
    parser.add_argument('--host', default='https://webhook.site', help='Webhook host URL (default: https://webhook.site)')
    parser.add_argument('--create', '-c', action='store_true', help='Create a new webhook')
    parser.add_argument('--view', '-v', action='store_true', help='View webhook requests')
    parser.add_argument('--limit', '-l', type=int, default=10, help='Limit number of requests (default: 10)')
    parser.add_argument('--raw', '-r', action='store_true', help='Output raw JSON data')
    parser.add_argument('--delete-request', metavar='ID', help='Delete specific request(s) by UUID or index (comma-separated)')
    parser.add_argument('--clear', action='store_true', help='Clear all requests')
    parser.add_argument('--delete', action='store_true', help='Delete the webhook')
    
    args = parser.parse_args()
    
    # Create or use existing webhook
    if args.create or not args.token:
        webhook = WebhookSite(host=args.host)
    else:
        webhook = WebhookSite(token=args.token, host=args.host)
    
    # Execute actions
    if args.view or args.raw:
        webhook.view_requests(limit=args.limit, raw=args.raw)
    
    if args.delete_request:
        items = [x.strip() for x in args.delete_request.split(',')]
        for item in items:
            try:
                idx = int(item)
                webhook.delete_request_by_index(idx)
            except ValueError:
                webhook.delete_request(item)
    
    if args.clear:
        webhook.delete_all_requests()
    
    if args.delete:
        webhook.delete_webhook()


if __name__ == '__main__':
    main()