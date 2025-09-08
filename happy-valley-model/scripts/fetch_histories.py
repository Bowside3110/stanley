import sqlite3
import requests
import time
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth

BASE = "https://api.theracingapi.com/v1"
AUTH = HTTPBasicAuth("mnLPvpPyIPk9NodfZOKdzfH0", "XjLJwHmwsrAX6yco36zr3dsg")
DB_PATH = "data/historical/hkjc.db"


def fetch(endpoint, retries=3, sleep=2):
    """Wrapper for GET requests with retry + timeout."""
    url = f"{BASE}/{endpoint.lstrip('/')}"
    for attempt in range(retries):
        try:
            r = requests.get(url, auth=AUTH, timeout=15)
            if r.status_code == 429:
                wait = sleep * (2 ** attempt)
                print(f"Rate limited. Sleeping {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  Fetch error ({url}): {e}")
            time.sleep(sleep)
    return None


def fetch_horse_history(conn, horse_id):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM horse_results WHERE horse_id=? LIMIT 1", (horse_id,))
    if cur.fetchone():
        return
    data = fetch(f"horses/{horse_id}/results")
    if not data:
        return
    for res in data if isinstance(data, list) else []:
        pos = res.get("position")
        cur.execute("""
            INSERT OR IGNORE INTO horse_results (horse_id, race_id, date, position)
            VALUES (?, ?, ?, ?)
        """, (
            horse_id,
            res.get("race_id"),
            res.get("date"),
            int(pos) if pos and str(pos).isdigit() else None,
        ))
    conn.commit()


def fetch_jockey_history(conn, jockey_id):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM jockey_results WHERE jockey_id=? LIMIT 1", (jockey_id,))
    if cur.fetchone():
        return
    data = fetch(f"jockeys/{jockey_id}/results")
    if not data:
        return
    for race in data.get("results", []):
        for runner in race.get("runners", []):
            pos = runner.get("position")
            cur.execute("""
                INSERT OR IGNORE INTO jockey_results (jockey_id, race_id, date, position)
                VALUES (?, ?, ?, ?)
            """, (
                jockey_id,
                race.get("race_id"),
                race.get("date"),
                int(pos) if pos and str(pos).isdigit() else None,
            ))
    conn.commit()


def fetch_trainer_history(conn, trainer_id):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM trainer_results WHERE trainer_id=? LIMIT 1", (trainer_id,))
    if cur.fetchone():
        return
    data = fetch(f"trainers/{trainer_id}/results")
    if not data:
        return
    for race in data.get("results", []):
        for runner in race.get("runners", []):
            pos = runner.get("position")
            cur.execute("""
                INSERT OR IGNORE INTO trainer_results (trainer_id, race_id, date, position)
                VALUES (?, ?, ?, ?)
            """, (
                trainer_id,
                race.get("race_id"),
                race.get("date"),
                int(pos) if pos and str(pos).isdigit() else None,
            ))
    conn.commit()


def process_with_eta(items, fetch_func, label, conn, limit=None, log_every=50):
    total = len(items) if not limit else min(limit, len(items))
    start_time = datetime.now()

    print(f"Found {len(items)} {label.lower()}. Processing {total}...")

    for i, item_id in enumerate(items[:total], start=1):
        fetch_func(conn, item_id)

        if i % log_every == 0 or i == total:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = elapsed / i
            remaining = (total - i) * rate
            eta = datetime.now() + timedelta(seconds=remaining)
            print(f"  {label}: {i}/{total} processed "
                  f"(ETA {eta.strftime('%H:%M:%S')}, ~{int(remaining/60)} min left)")


def backfill_histories(db_path=DB_PATH, limit=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Horses
    cur.execute("SELECT DISTINCT horse_id FROM runners WHERE horse_id IS NOT NULL")
    horses = [r[0] for r in cur.fetchall()]
    process_with_eta(horses, fetch_horse_history, "Horses", conn, limit, log_every=50)

    # Jockeys
    cur.execute("SELECT DISTINCT jockey_id FROM runners WHERE jockey_id IS NOT NULL")
    jockeys = [r[0] for r in cur.fetchall()]
    process_with_eta(jockeys, fetch_jockey_history, "Jockeys", conn, limit, log_every=20)

    # Trainers
    cur.execute("SELECT DISTINCT trainer_id FROM runners WHERE trainer_id IS NOT NULL")
    trainers = [r[0] for r in cur.fetchall()]
    process_with_eta(trainers, fetch_trainer_history, "Trainers", conn, limit, log_every=20)

    conn.close()
    print("✅ History backfill complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch horse/jockey/trainer histories into SQLite")
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite DB file")
    parser.add_argument("--limit", type=int, help="Limit number of entities (for testing)")
    args = parser.parse_args()

    backfill_histories(args.db, args.limit)
