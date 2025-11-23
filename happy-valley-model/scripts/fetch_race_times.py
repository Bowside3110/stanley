#!/usr/bin/env python3
"""
fetch_race_times.py

Lightweight race schedule fetcher for Stanley scheduling system.
Fetches race times and inserts only race metadata (no runners, no predictions).

Usage:
    python scripts/fetch_race_times.py
"""

import subprocess
import sqlite3
import json
from pathlib import Path
import os

DB_PATH = "data/historical/hkjc.db"
PREDICTIONS_DIR = Path("data/predictions")


def run_node_fetch():
    """Run the Node fetcher to get the next meeting's races"""
    print("🔄 Fetching race schedule...")
    result = subprocess.run(
        ["node", "scripts/fetch_next_meeting.mjs"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("❌ Node fetcher failed:")
        print(result.stderr)
        raise RuntimeError("Node fetch failed")
    print(result.stdout)


def get_latest_json():
    """Find the newest racecard JSON file"""
    files = sorted(PREDICTIONS_DIR.glob("races_*.json"), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError("No race JSON files found in data/predictions/")
    return files[-1]


def import_race_times(json_path):
    """Insert races into hkjc.db (races table only, skip runners)"""
    print(f"📥 Importing race schedule from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "raceMeetings" in data:
        meetings = data["raceMeetings"]
    else:
        raise ValueError("Unexpected JSON structure: no 'raceMeetings' found")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    race_count = 0
    meeting_date = None

    for meeting in meetings:
        race_date = meeting["date"]
        meeting_date = race_date
        course = meeting["venueCode"]

        for race in meeting.get("races", []):
            race_id = race["id"]
            race_name = race.get("raceName_en", "")
            race_class = race.get("raceClass_en", "")
            distance = race.get("distance")
            going = race.get("raceTrack", {}).get("description_en")
            rail = race.get("raceCourse", {}).get("description_en")
            post_time = race.get("postTime", "")

            # Insert race (skip runners)
            cur.execute("""
                INSERT OR REPLACE INTO races
                (race_id, date, course, race_name, class, distance, going, rail, post_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id, race_date, course, race_name, race_class,
                distance, going, rail, post_time
            ))
            race_count += 1

    conn.commit()
    conn.close()
    
    print(f"✅ Fetched {race_count} races for {meeting_date}")


def main():
    """Main entry point"""
    # Ensure directories exist
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        run_node_fetch()
        latest_json = get_latest_json()
        import_race_times(latest_json)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())


