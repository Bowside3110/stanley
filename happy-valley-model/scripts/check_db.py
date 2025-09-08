import sqlite3, pandas as pd

conn = sqlite3.connect("data/historical/hkjc.db")

# Coverage per table
for tbl in ["races", "results", "racecard_pro_runners", "horse_results", "jockey_results", "trainer_results"]:
    if pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tbl}'", conn).shape[0]:
        cnt = pd.read_sql(f"SELECT COUNT(*) AS n FROM {tbl}", conn)["n"].iloc[0]
        print(f"{tbl}: {cnt:,} rows")
    else:
        print(f"{tbl}: MISSING")

# Check recent races
df_recent = pd.read_sql("""
    SELECT r.race_id, r.date, r.course, res.horse_id,
           res.sp_dec, res.sp, res.weight_lbs, res.weight
    FROM races r
    JOIN results res ON res.race_id = r.race_id
    WHERE r.date >= '2025-05-21'
    LIMIT 20
""", conn)
print("\nSample of recent results (should show odds/weight):")
print(df_recent.head(10))

# Check if racecard enrichment exists for same window
df_card = pd.read_sql("""
    SELECT race_id, horse_id, draw, headgear, wind_surgery
    FROM racecard_pro_runners
    WHERE race_id IN (SELECT race_id FROM races WHERE date >= '2025-05-21')
    LIMIT 20
""", conn)
print("\nSample of racecard_pro_runners (should show draw/headgear etc):")
print(df_card.head(10))

# Check if horse histories exist
df_hr = pd.read_sql("""
    SELECT horse_id, MIN(date) AS first_run, MAX(date) AS last_run, COUNT(*) AS n
    FROM horse_results
    GROUP BY horse_id
    ORDER BY last_run DESC
    LIMIT 5
""", conn)
print("\nSample horse_results coverage:")
print(df_hr)

conn.close()
