#!/usr/bin/env python3
"""
Test script for API endpoints
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_endpoints():
    """Test all API endpoints"""
    
    print("=" * 60)
    print("Testing Stanley Racing Predictions API Endpoints")
    print("=" * 60)
    
    # Step 1: Login
    print("\n1. Testing Login...")
    session = requests.Session()
    login_data = {
        "username": "ben",
        "password": "password"
    }
    
    response = session.post(f"{BASE_URL}/login", data=login_data, allow_redirects=False)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 303:
        print("   ✓ Login successful (redirect received)")
        # Extract cookie
        cookies = session.cookies.get_dict()
        print(f"   Cookies: {list(cookies.keys())}")
    else:
        print(f"   ✗ Login failed: {response.text}")
        return
    
    # Step 2: Test /api/races
    print("\n2. Testing /api/races...")
    try:
        response = session.get(f"{BASE_URL}/api/races")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            races = response.json()
            print(f"   ✓ Found {len(races)} upcoming races")
            
            if races:
                print("\n   Sample race:")
                race = races[0]
                print(f"     - Race ID: {race.get('race_id')}")
                print(f"     - Name: {race.get('race_name')}")
                print(f"     - Course: {race.get('course')}")
                print(f"     - Post Time: {race.get('post_time')}")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Step 3: Test /api/predictions
    print("\n3. Testing /api/predictions...")
    try:
        response = session.get(f"{BASE_URL}/api/predictions")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            predictions = response.json()
            print(f"   ✓ Found predictions for {len(predictions)} races")
            
            if predictions:
                print("\n   Sample prediction:")
                pred = predictions[0]
                print(f"     - Race: {pred.get('race_name')}")
                print(f"     - Course: {pred.get('course')}")
                print(f"     - Horses with predictions: {len(pred.get('predictions', []))}")
                
                if pred.get('predictions'):
                    top_pick = pred['predictions'][0]
                    print(f"\n     Top pick:")
                    print(f"       Horse: {top_pick.get('horse')}")
                    print(f"       Draw: {top_pick.get('draw')}")
                    print(f"       Predicted Rank: {top_pick.get('predicted_rank')}")
                    print(f"       Predicted Score: {top_pick.get('predicted_score'):.4f}")
                    print(f"       Win Odds: {top_pick.get('win_odds')}")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Step 4: Test /api/predictions/{race_id}
    print("\n4. Testing /api/predictions/{{race_id}}...")
    try:
        # First get a race_id
        response = session.get(f"{BASE_URL}/api/races")
        if response.status_code == 200:
            races = response.json()
            if races:
                race_id = races[0]['race_id']
                print(f"   Testing with race_id: {race_id}")
                
                response = session.get(f"{BASE_URL}/api/predictions/{race_id}")
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    race_pred = response.json()
                    print(f"   ✓ Retrieved predictions for race")
                    print(f"     - Race: {race_pred.get('race_name')}")
                    print(f"     - Predictions: {len(race_pred.get('predictions', []))}")
                elif response.status_code == 404:
                    print(f"   ! No predictions found for this race")
                else:
                    print(f"   ✗ Error: {response.text}")
            else:
                print("   ! No upcoming races to test with")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Step 5: Test /api/past-predictions
    print("\n5. Testing /api/past-predictions...")
    try:
        response = session.get(f"{BASE_URL}/api/past-predictions?limit=5")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            past = response.json()
            print(f"   ✓ Found {len(past)} past races with predictions")
            
            if past:
                print("\n   Sample past race:")
                race = past[0]
                print(f"     - Race: {race.get('race_name')}")
                print(f"     - Date: {race.get('date')}")
                print(f"     - Course: {race.get('course')}")
                
                if race.get('predictions'):
                    # Check if we have results
                    has_results = any(p.get('actual_position') for p in race['predictions'])
                    if has_results:
                        print(f"     - Results available: Yes")
                        winner = next((p for p in race['predictions'] if p.get('actual_position') == 1), None)
                        if winner:
                            print(f"     - Winner: {winner.get('horse')} (predicted rank: {winner.get('predicted_rank')})")
                    else:
                        print(f"     - Results available: No")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Step 6: Test /api/scheduler/status
    print("\n6. Testing /api/scheduler/status...")
    try:
        response = session.get(f"{BASE_URL}/api/scheduler/status")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            status = response.json()
            print(f"   ✓ Scheduler Status:")
            print(f"     - Active: {status.get('active')}")
            print(f"     - Next Job: {status.get('next_job')}")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_endpoints()

