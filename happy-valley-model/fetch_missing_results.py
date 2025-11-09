#!/usr/bin/env python3
"""
Fetch missing race results for dates that have predictions but no results.

This script uses the HKJC API (via Node.js hkjc-api package) to fetch
results for completed races and updates the runners table with position data.
"""

import sqlite3
import json
import subprocess
from pathlib import Path
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
          AND run.position IS NULL
        ORDER BY r.date
    '''
    
    cursor.execute(query)
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return dates

def fetch_results_from_hkjc(date):
    """
    Fetch race results for a specific date using HKJC API via Node.js.
    Returns JSON data with race results.
    """
    print(f"\n📥 Fetching results for {date} from HKJC API...")
    
    # Use the same Node.js script pattern as make_predictions.py
    # but fetch results instead of racecards
    node_script = f'''
    const {{ getRaceResults }} = require('hkjc-api');
    
    async function fetchResults() {{
        try {{
            const date = '{date}';
            console.log(`Fetching results for ${{date}}...`);
            
            // The HKJC API might use getRaceResults or similar
            // We'll try to get results for the date
            const results = await getRaceResults(date);
            
            if (results && results.length > 0) {{
                console.log(`Found ${{results.length}} races`);
                console.log(JSON.stringify(results, null, 2));
            }} else {{
                console.log('No results found');
                console.log('[]');
            }}
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
        output_lines = result.stdout.strip().split('\n')
        json_start = None
        for i, line in enumerate(output_lines):
            if line.strip().startswith('[') or line.strip().startswith('{'):
                json_start = i
                break
        
        if json_start is not None:
            json_str = '\n'.join(output_lines[json_start:])
            data = json.loads(json_str)
            return data
        else:
            print(f"   ⚠️  No JSON data found in output")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ Timeout fetching results for {date}")
        return None
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def update_results_in_db(date, results_data, db_path="data/historical/hkjc.db"):
    """
    Update the runners table with position data from results.
    
    Args:
        date: Race date (YYYY-MM-DD)
        results_data: JSON data from HKJC API with race results
        db_path: Path to database
    """
    if not results_data:
        print(f"   ⚠️  No results data to process")
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    updated = 0
    
    for race in results_data:
        race_name = race.get('raceName', '')
        runners = race.get('runners', [])
        
        if not runners:
            continue
        
        # Normalize race name for matching
        race_name_norm = race_name.upper().strip().replace('  ', ' ')
        
        for runner in runners:
            horse_name = runner.get('horseName', '')
            final_position = runner.get('finalPosition', 0)
            
            # Skip if no position (scratched, etc.)
            if not final_position or final_position == 0:
                continue
            
            # Normalize horse name
            horse_norm = normalize_horse_name(horse_name)
            
            # Find matching runner in database
            query = '''
                SELECT run.rowid
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
                rowid = result[0]
                cursor.execute('SELECT horse FROM runners WHERE rowid = ?', (rowid,))
                db_horse = cursor.fetchone()[0]
                if normalize_horse_name(db_horse) == horse_norm:
                    matching_rowid = rowid
                    break
            
            if matching_rowid:
                # Update position
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
    print("FETCHING MISSING RACE RESULTS")
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
        results_data = fetch_results_from_hkjc(date)
        
        if results_data:
            updated = update_results_in_db(date, results_data)
            print(f"   ✅ Updated {updated} runners")
            total_updated += updated
        else:
            print(f"   ⚠️  No results available yet (race may not have run)")
    
    print("\n" + "=" * 80)
    print(f"✅ COMPLETE: Updated {total_updated} runners with results")
    print("=" * 80)
    
    # Show summary
    conn = sqlite3.connect("data/historical/hkjc.db")
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN position IS NOT NULL THEN 1 END) as with_results
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
        print(f"\n🎯 Ready to run: python analyze_predictions.py")

if __name__ == "__main__":
    main()

