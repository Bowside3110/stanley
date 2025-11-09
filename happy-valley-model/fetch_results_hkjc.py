#!/usr/bin/env python3
"""
Fetch race results from HKJC API and update runners table with positions.

Uses the hkjc-api Node.js package to fetch historical race results
and updates the database with finalPosition data.
"""

import sqlite3
import json
import subprocess
from src.horse_matcher import normalize_horse_name

def get_dates_needing_results(db_path="data/historical/hkjc.db"):
    """Query database for dates with predictions but no results."""
    conn = sqlite3.connect(db_path)
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
    
    return dates

def fetch_results_from_hkjc(date):
    """
    Fetch race results for a specific date using HKJC API.
    Returns JSON data with race results including finalPosition.
    """
    print(f"\n📥 Fetching results for {date} from HKJC API...")
    
    node_script = f'''
const {{ HorseRacingAPI }} = require('hkjc-api');

async function fetchResults() {{
    const api = new HorseRacingAPI();
    
    try {{
        const meetings = await api.getAllRaces('{date}');
        
        // Filter for HK meetings only (ST and HV)
        const hkMeetings = meetings.filter(m => 
            m.venueCode === 'ST' || m.venueCode === 'HV'
        );
        
        console.log(JSON.stringify(hkMeetings, null, 2));
    }} catch (error) {{
        console.error('Error:', error.message);
        console.log('[]');
    }}
}}

fetchResults();
'''
    
    try:
        result = subprocess.run(
            ['node', '-e', node_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse JSON from stdout
        if result.returncode == 0 and result.stdout.strip():
            try:
                data = json.loads(result.stdout.strip())
                return data
            except json.JSONDecodeError:
                # Try to find JSON in output
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('['):
                        json_str = '\n'.join(lines[i:])
                        data = json.loads(json_str)
                        return data
        
        print(f"   ⚠️  No valid JSON data")
        return None
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ Timeout")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def update_results_in_db(date, meetings_data, db_path="data/historical/hkjc.db"):
    """
    Update the runners table with position data from HKJC results.
    
    Args:
        date: Race date (YYYY-MM-DD)
        meetings_data: JSON data from HKJC API with meetings and races
        db_path: Path to database
    """
    if not meetings_data:
        print(f"   ⚠️  No data to process")
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    updated = 0
    
    for meeting in meetings_data:
        venue = meeting.get('venueCode', '')
        races = meeting.get('races', [])
        
        print(f"   Processing {venue} - {len(races)} races")
        
        for race in races:
            race_name_en = race.get('raceName_en', '')
            runners = race.get('runners', [])
            
            if not race_name_en or not runners:
                continue
            
            # Normalize race name
            race_name_norm = race_name_en.upper().strip().replace('  ', ' ')
            
            for runner in runners:
                horse_name = runner.get('name_en', '')
                final_position = runner.get('finalPosition', 0)
                
                # Skip if no valid position
                if not final_position or final_position == 0:
                    continue
                
                # Normalize horse name
                horse_norm = normalize_horse_name(horse_name)
                
                # Find matching runner in database
                query = '''
                    SELECT run.rowid, run.horse
                    FROM runners run
                    JOIN races r ON run.race_id = r.race_id
                    WHERE r.date = ?
                      AND UPPER(REPLACE(r.race_name, '  ', ' ')) = ?
                '''
                
                cursor.execute(query, (date, race_name_norm))
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
                    ''', (final_position, matching_rowid))
                    updated += 1
    
    conn.commit()
    conn.close()
    
    return updated

def main():
    """Main function to fetch and update missing results."""
    print("=" * 80)
    print("FETCHING RACE RESULTS FROM HKJC API")
    print("=" * 80)
    
    # Get dates that need results
    dates = get_dates_needing_results()
    
    if not dates:
        print("\n✅ No dates need results - all predictions have results!")
        return
    
    print(f"\n📋 Found {len(dates)} dates needing results:")
    for date in dates:
        print(f"   • {date}")
    
    # Fetch and update results for each date
    total_updated = 0
    
    for date in dates:
        meetings_data = fetch_results_from_hkjc(date)
        
        if meetings_data:
            updated = update_results_in_db(date, meetings_data)
            print(f"   ✅ Updated {updated} runners")
            total_updated += updated
        else:
            print(f"   ⚠️  No results available")
    
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

