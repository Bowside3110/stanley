#!/usr/bin/env python3
"""
Fetch race results for completed races using The Racing API.

This script fetches results for dates that have predictions but no results,
and updates the runners table with position data.
"""

import sqlite3
import requests
import time
from requests.auth import HTTPBasicAuth
from src.horse_matcher import normalize_horse_name

# The Racing API credentials
BASE = "https://api.theracingapi.com/v1"
AUTH = HTTPBasicAuth("mnLPvpPyIPk9NodfZOKdzfH0", "XjLJwHmwsrAX6yco36zr3dsg")

def fetch_results_for_date(date):
    """Fetch results for a specific date from The Racing API."""
    print(f"\n📥 Fetching results for {date}...")
    
    params = {
        "start_date": date,
        "end_date": date,
        "limit": 50
    }
    
    try:
        response = requests.get(
            f"{BASE}/results",
            auth=AUTH,
            params=params,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        races = data.get("results", [])
        
        # Filter for HK races only
        hk_races = [r for r in races if "sha tin" in r.get("course", "").lower() or 
                    "happy valley" in r.get("course", "").lower()]
        
        print(f"   Found {len(hk_races)} HK races")
        return hk_races
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []

def update_positions(date, races, db_path="data/historical/hkjc.db"):
    """Update runners table with position data from results."""
    if not races:
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    updated = 0
    
    for race in races:
        race_name = race.get("race_name", "")
        runners = race.get("runners", [])
        
        # Normalize race name
        race_name_norm = race_name.upper().strip().replace('  ', ' ')
        
        for runner in runners:
            horse_name = runner.get("horse", "")
            position = runner.get("position")
            
            # Skip if no valid position
            if not position or not str(position).isdigit():
                continue
            
            position = int(position)
            
            # Normalize horse name
            horse_norm = normalize_horse_name(horse_name)
            
            # Find matching runner in database
            query = '''
                SELECT run.rowid, run.horse
                FROM runners run
                JOIN races r ON run.race_id = r.race_id
                WHERE r.date = ?
                  AND UPPER(REPLACE(r.race_name, '  ', ' ')) LIKE ?
            '''
            
            # Try exact match first
            cursor.execute(query, (date, race_name_norm))
            results = cursor.fetchall()
            
            # If no exact match, try fuzzy match
            if not results:
                cursor.execute(query, (date, f'%{race_name_norm[:20]}%'))
                results = cursor.fetchall()
            
            # Filter by normalized horse name
            matching_rowid = None
            for result in results:
                rowid, db_horse = result
                if normalize_horse_name(db_horse) == horse_norm:
                    matching_rowid = rowid
                    break
            
            if matching_rowid:
                cursor.execute('''
                    UPDATE runners
                    SET position = ?
                    WHERE rowid = ?
                ''', (position, matching_rowid))
                updated += 1
    
    conn.commit()
    conn.close()
    
    return updated

def main():
    """Main function."""
    print("=" * 80)
    print("FETCHING RACE RESULTS FROM THE RACING API")
    print("=" * 80)
    
    # Get dates that need results
    conn = sqlite3.connect("data/historical/hkjc.db")
    cursor = conn.cursor()
    
    query = '''
        SELECT DISTINCT r.date
        FROM races r
        JOIN runners run ON r.race_id = run.race_id
        WHERE run.predicted_rank IS NOT NULL
          AND (run.position IS NULL OR run.position = 0)
        ORDER BY r.date
    '''
    
    cursor.execute(query)
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if not dates:
        print("\n✅ No dates need results!")
        return
    
    print(f"\n📋 Found {len(dates)} dates needing results:")
    for date in dates:
        print(f"   • {date}")
    
    # Fetch and update for each date
    total_updated = 0
    
    for date in dates:
        races = fetch_results_for_date(date)
        updated = update_positions(date, races)
        print(f"   ✅ Updated {updated} runners")
        total_updated += updated
        time.sleep(1)  # Be nice to the API
    
    print("\n" + "=" * 80)
    print(f"✅ COMPLETE: Updated {total_updated} runners with results")
    print("=" * 80)
    
    # Show summary
    conn = sqlite3.connect("data/historical/hkjc.db")
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN position IS NOT NULL AND position > 0 THEN 1 END) as with_results
        FROM runners
        WHERE predicted_rank IS NOT NULL
    ''')
    total, with_results = cursor.fetchone()
    conn.close()
    
    print(f"\n📊 Summary:")
    print(f"   Total predictions: {total}")
    print(f"   With results: {with_results} ({with_results/total*100:.1f}%)")
    print(f"   Still missing: {total - with_results}")
    
    if with_results > 0:
        print(f"\n🎯 Ready to analyze: python analyze_predictions.py")

if __name__ == "__main__":
    main()

