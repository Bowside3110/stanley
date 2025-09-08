import os
import time
import sqlite3
import requests
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth

BASE = "https://api.theracingapi.com/v1"
AUTH = HTTPBasicAuth("mnLPvpPyIPk9NodfZOKdzfH0", "XjLJwHmwsrAX6yco36zr3dsg")

HK_RACE_COUNT = 0


def fetch(endpoint, params=None, retries=5):
    """Fetch wrapper with retries + backoff for rate limits."""
    url = f"{BASE}/{endpoint.lstrip('/')}"
    for attempt in range(retries):
        try:
            r = requests.get(url, auth=AUTH, params=params or {}, timeout=15)
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limited. Sleeping {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Fetch error: {e}")
            return {}
    return {}


def fetch_results_day(target_date, limit=50):
    """Fetch all results for a single date with pagination."""
    all_races = []
    skip = 0
    while True:
        params = {
            "start_date": target_date,
            "end_date": target_date,
            "limit": limit,
            "skip": skip,
        }
        data = fetch("results", params=params)
        races = data.get("results", [])
        if not races:
            break
        all_races.extend(races)
        skip += limit
    return all_races


def reset_schema(conn):
    """Drop all tables if they exist (avoids schema drift errors)."""
    cur = conn.cursor()
    cur.executescript("""
    DROP TABLE IF EXISTS races;
    DROP TABLE IF EXISTS runners;
    DROP TABLE IF EXISTS results;
    DROP TABLE IF EXISTS backfill_log;
    """)
    conn.commit()


def create_schema(conn):
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS races (
        race_id TEXT PRIMARY KEY,
        date TEXT,
        course TEXT,
        race_name TEXT,
        class TEXT,
        distance REAL,
        going TEXT,
        rail TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS runners (
        race_id TEXT,
        horse_id TEXT,
        horse TEXT,
        draw TEXT,
        weight TEXT,
        jockey TEXT,
        jockey_id TEXT,
        trainer TEXT,
        trainer_id TEXT,
        win_odds REAL,
        PRIMARY KEY (race_id, horse_id)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
        race_id TEXT,
        horse_id TEXT,
        position INT,
        PRIMARY KEY (race_id, horse_id)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS backfill_log (
        date TEXT PRIMARY KEY,
        processed_at TEXT
    )""")

    conn.commit()


def insert_race_data(conn, races):
    cur = conn.cursor()
    for race in races:
        course = race.get("course", "")
        if "sha tin" not in course.lower() and "happy valley" not in course.lower():
            continue

        race_id = race.get("race_id")
        race_meta = (
            race_id,
            race.get("date"),
            course,
            race.get("race_name"),
            race.get("class"),
            race.get("dist_m"),
            race.get("going"),
            race.get("rail_movements"),
        )
        cur.execute("""
        INSERT OR REPLACE INTO races
        (race_id, date, course, race_name, class, distance, going, rail)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, race_meta)

        for r in race.get("runners", []):
            cur.execute("""
            INSERT OR REPLACE INTO runners
            (race_id, horse_id, horse, draw, weight, jockey, trainer, win_odds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id,
                r.get("horse_id"),
                r.get("horse"),
                r.get("draw"),
                r.get("weight"),
                r.get("jockey"),
                r.get("trainer"),
                r.get("sp_dec"),
            ))

            cur.execute("""
            INSERT OR REPLACE INTO results
            (race_id, horse_id, position)
            VALUES (?, ?, ?)
            """, (
                race_id,
                r.get("horse_id"),
                int(r.get("position")) if r.get("position") and str(r.get("position")).isdigit() else None,
            ))

    conn.commit()


def enrich_race_with_racecard(conn, race_id):
    """Call /racecards/{race_id}/pro to enrich runners with odds/jockey/trainer IDs."""
    global HK_RACE_COUNT
    try:
        card = fetch(f"racecards/{race_id}/pro")
    except Exception as e:
        print(f"  Could not fetch racecard for {race_id}: {e}")
        return

    cur = conn.cursor()
    for r in card.get("runners", []):
        cur.execute("""
        UPDATE runners
        SET jockey_id=?, trainer_id=?, win_odds=COALESCE(?, win_odds)
        WHERE race_id=? AND horse_id=?
        """, (
            r.get("jockey_id"),
            r.get("trainer_id"),
            (r.get("odds")[0] if r.get("odds") else None),
            race_id,
            r.get("horse_id"),
        ))

    conn.commit()
    HK_RACE_COUNT += 1
    if HK_RACE_COUNT % 50 == 0:
        print(f"  Progress: {HK_RACE_COUNT} HK races processed so far...")


def backfill(start_date, end_date, db_path="data/historical/hkjc.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    reset_schema(conn)
    create_schema(conn)

    d1 = datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.strptime(end_date, "%Y-%m-%d")

    current = d1
    while current <= d2:
        day_str = current.strftime("%Y-%m-%d")

        cur = conn.cursor()
        cur.execute("SELECT 1 FROM backfill_log WHERE date=?", (day_str,))
        if cur.fetchone():
            print(f"Skipping {day_str} (already processed).")
            current += timedelta(days=1)
            continue

        print(f"Fetching {day_str}...")
        try:
            races = fetch_results_day(day_str)
            if races:
                insert_race_data(conn, races)
                for race in races:
                    if "sha tin" in race.get("course", "").lower() or "happy valley" in race.get("course", "").lower():
                        enrich_race_with_racecard(conn, race["race_id"])
                print(f"  Inserted {len(races)} races (HK filtered).")
                cur.execute("INSERT OR REPLACE INTO backfill_log (date, processed_at) VALUES (?, ?)",
                            (day_str, datetime.now().isoformat()))
                conn.commit()
            else:
                print("  No races found.")
        except Exception as e:
            print(f"  Error on {day_str}: {e}")
        current += timedelta(days=1)
        time.sleep(1)

    conn.close()
    print(f"✅ Backfill complete. Data saved to {db_path}")
    print(f"✅ Total HK races processed: {HK_RACE_COUNT}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill HKJC results + racecards into SQLite (no histories)")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--db", default="data/historical/hkjc.db", help="Path to SQLite DB file")
    args = parser.parse_args()

    backfill(args.start_date, args.end_date, args.db)
