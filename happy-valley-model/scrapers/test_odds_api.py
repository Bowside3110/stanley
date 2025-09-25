#!/usr/bin/env python
import requests
import json
import time
from requests.auth import HTTPBasicAuth

BASE = "https://api.theracingapi.com/v1"
AUTH = HTTPBasicAuth("mnLPvpPyIPk9NodfZOKdzfH0", "XjLJwHmwsrAX6yco36zr3dsg")

def test_odds_endpoint(race_id, horse_id, timeout=10):
    """Test the odds endpoint with a specific race_id and horse_id."""
    url = f"{BASE}/odds/{race_id}/{horse_id}"
    print(f"Testing URL: {url}")
    
    try:
        r = requests.get(url, auth=AUTH, timeout=timeout)
        status = r.status_code
        print(f"Status code: {status}")
        
        if status == 200:
            data = r.json()
            print(f"Response data: {json.dumps(data, indent=2)}")
            return data
        elif status == 401:
            print("Authentication error - check your API credentials")
        elif status == 404:
            print("Resource not found - check race_id and horse_id")
        elif status == 429:
            print("Rate limited - try again later")
        else:
            print(f"Unexpected status code: {status}")
            print(f"Response text: {r.text[:500]}")
    except requests.exceptions.Timeout:
        print(f"Request timed out after {timeout} seconds")
    except Exception as e:
        print(f"Error: {e}")
    
    return None

def test_racecards_endpoint(timeout=10):
    """Test the racecards endpoint to get valid race_ids and horse_ids."""
    url = f"{BASE}/racecards/pro"
    print(f"Testing URL: {url}")
    
    try:
        r = requests.get(url, auth=AUTH, timeout=timeout)
        status = r.status_code
        print(f"Status code: {status}")
        
        if status == 200:
            data = r.json()
            racecards = data.get("racecards", [])
            print(f"Found {len(racecards)} racecards")
            
            if racecards:
                # Get the first race with runners
                for race in racecards:
                    race_id = race.get("race_id")
                    runners = race.get("runners", [])
                    if race_id and runners:
                        print(f"\nFound race: {race_id} - {race.get('race_name')}")
                        print(f"Date: {race.get('date')}")
                        print(f"Course: {race.get('course')}")
                        print(f"Number of runners: {len(runners)}")
                        
                        # Get the first runner
                        if runners:
                            runner = runners[0]
                            horse_id = runner.get("horse_id")
                            if horse_id:
                                print(f"\nFound runner: {horse_id} - {runner.get('horse')}")
                                
                                # Test the odds endpoint with this race_id and horse_id
                                print("\nTesting odds endpoint with these IDs:")
                                test_odds_endpoint(race_id, horse_id)
                                return
            
            print("No suitable race/runner found to test odds endpoint")
        elif status == 401:
            print("Authentication error - check your API credentials")
        else:
            print(f"Unexpected status code: {status}")
            print(f"Response text: {r.text[:500]}")
    except requests.exceptions.Timeout:
        print(f"Request timed out after {timeout} seconds")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=== Testing Racing API Odds Endpoint ===")
    
    # First test the racecards endpoint to get valid IDs
    print("\n1. Testing racecards endpoint to find valid race_id and horse_id:")
    test_racecards_endpoint()
    
    # If you know specific race_id and horse_id values, you can test them directly:
    # print("\n2. Testing with specific race_id and horse_id:")
    # test_odds_endpoint("race_123456", "horse_789012")
