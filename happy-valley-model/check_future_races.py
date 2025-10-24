#!/usr/bin/env python3
import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('data/historical/hkjc.db')

# Check races for 2025-10-19
print("Checking races for 2025-10-19...")
df_races = pd.read_sql("SELECT * FROM races WHERE date = '2025-10-19'", conn)
print(f"Found {len(df_races)} races")
if len(df_races) > 0:
    print(df_races.head())

# Check runners for these races
print("\nChecking runners for these races...")
df_runners = pd.read_sql("SELECT r.*, rpr.last_run FROM runners r LEFT JOIN racecard_pro_runners rpr ON r.race_id = rpr.race_id AND r.horse_id = rpr.horse_id WHERE r.race_id LIKE 'RACE_20251019_%'", conn)
print(f"Found {len(df_runners)} runners")
if len(df_runners) > 0:
    print(df_runners[['race_id', 'horse_id', 'horse', 'last_run']].head(10))
    
    # Check how many have last_run values
    has_last_run = df_runners['last_run'].notnull().sum()
    print(f"\nRunners with last_run data: {has_last_run} out of {len(df_runners)} ({has_last_run/len(df_runners)*100:.1f}%)")

# Close the connection
conn.close()

