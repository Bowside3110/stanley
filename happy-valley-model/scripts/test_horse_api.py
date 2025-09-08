# scripts/test_horse_api.py
import requests
from requests.auth import HTTPBasicAuth
import sys
import json

BASE = "https://api.theracingapi.com/v1"
USERNAME = "mnLPvpPyIPk9NodfZOKdzfH0"  # your API username
PASSWORD = "XjLJwHmwsrAX6yco36zr3dsg"  # your API password

def fetch_horse_results(horse_id: str):
    url = f"{BASE}/horses/{horse_id}/results"
    resp = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=15)
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_horse_api.py <horse_id>")
        sys.exit(1)

    horse_id = sys.argv[1]
    data = fetch_horse_results(horse_id)

    print(f"Top-level keys: {list(data.keys())}")
    print(f"Number of results: {len(data.get('results', []))}")

    if data.get("results"):
        first = data["results"][0]
        print("\nFirst result sample:")
        print(json.dumps(first, indent=2)[:2000])
    else:
        print("No results found for this horse.")
