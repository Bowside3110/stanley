#!/usr/bin/env python3
import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('data/historical/hkjc.db')

print('Horse ID formats by table:\n')

# Compare horse_id formats across tables
tables = ['runners', 'horse_results', 'racecard_pro_runners']
for table in tables:
    try:
        df = pd.read_sql(f"SELECT horse_id FROM {table} LIMIT 5", conn)
        print(f'{table}:')
        print(df)
        print()
    except Exception as e:
        print(f"Error querying {table}: {e}")
        print()

# Check for last_run field in racecard_pro_runners
print("Checking for last_run values in racecard_pro_runners...")
try:
    df = pd.read_sql("""
        SELECT COUNT(*) as total_count,
               SUM(CASE WHEN last_run IS NOT NULL THEN 1 ELSE 0 END) as with_last_run
        FROM racecard_pro_runners
    """, conn)
    total = df['total_count'].iloc[0]
    with_last_run = df['with_last_run'].iloc[0]
    print(f"Total records: {total}")
    print(f"Records with last_run: {with_last_run} ({with_last_run/total*100:.1f}%)")
    
    # Sample of records with last_run
    df = pd.read_sql("""
        SELECT race_id, horse_id, horse, last_run
        FROM racecard_pro_runners
        WHERE last_run IS NOT NULL
        LIMIT 5
    """, conn)
    print("\nSample records with last_run:")
    print(df)
    print()
except Exception as e:
    print(f"Error checking last_run: {e}")
    print()

# Check for horses that appear in both runners and horse_results
print("Checking for horses that appear in both runners and horse_results...")
try:
    # Get sample of horse names from runners table
    runners_df = pd.read_sql("""
        SELECT DISTINCT r.horse, r.horse_id as runner_id 
        FROM runners r 
        WHERE r.race_id LIKE 'RACE_20251019_%'
        LIMIT 10
    """, conn)
    
    # For each horse name, check if it exists in horse_results
    for idx, row in runners_df.iterrows():
        horse_name = row['horse']
        runner_id = row['runner_id']
        
        # Check if this horse name appears in runners with other race_ids
        other_runners_df = pd.read_sql(f"""
            SELECT r.race_id, r.horse_id, r.horse
            FROM runners r 
            WHERE r.horse = '{horse_name}'
            AND r.race_id NOT LIKE 'RACE_20251019_%'
            LIMIT 5
        """, conn)
        
        print(f"Horse: {horse_name}")
        print(f"  Future race ID: {runner_id}")
        
        if len(other_runners_df) > 0:
            print(f"  Other appearances in runners:")
            for _, other_row in other_runners_df.iterrows():
                print(f"    Race: {other_row['race_id']}, ID: {other_row['horse_id']}")
        else:
            print(f"  No other appearances in runners table")
            
        # Now try to find in racecard_pro_runners by name
        racecard_df = pd.read_sql(f"""
            SELECT rpr.race_id, rpr.horse_id, rpr.horse, rpr.last_run
            FROM racecard_pro_runners rpr
            WHERE rpr.horse = '{horse_name}'
            LIMIT 5
        """, conn)
        
        if len(racecard_df) > 0:
            print(f"  Appearances in racecard_pro_runners:")
            for _, rc_row in racecard_df.iterrows():
                print(f"    Race: {rc_row['race_id']}, ID: {rc_row['horse_id']}, Last run: {rc_row['last_run']}")
        else:
            print(f"  No appearances in racecard_pro_runners table")
        print()

except Exception as e:
    print(f"Error during cross-table check: {e}")

# Close the connection
conn.close()
