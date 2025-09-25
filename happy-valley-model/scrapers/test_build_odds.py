#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.build_future_dataset import fetch_odds, fetch

def test_odds_direct():
    """Test the odds endpoint directly with the same race_id and horse_id that worked in test_odds_api.py"""
    race_id = "rac_11720735"
    horse_id = "hrs_54494531"
    
    print(f"Testing direct fetch from odds endpoint for race {race_id}, horse {horse_id}")
    odds_data = fetch(f"odds/{race_id}/{horse_id}")
    print(f"Raw response: {odds_data}")
    
    print("\nTesting fetch_odds function from build_future_dataset.py")
    odds_value = fetch_odds(race_id, horse_id)
    print(f"Odds value: {odds_value}")

if __name__ == "__main__":
    test_odds_direct()
