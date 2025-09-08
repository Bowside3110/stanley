import sqlite3
import requests
import time
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth

BASE = "https://api.theracingapi.com/v1"
AUTH = HTTPBasicAuth("mnLPvpPyIPk9NodfZOKdzfH0", "XjLJwHmwsrAX6yco36zr3dsg")
DB_PATH = "data/historical/hkjc.db"


def fetch(endpoint, retries=3, sleep=2):
    url = f"{BASE}/{endpoint.lstrip('/')}"
    for attempt in range(retries):
        try:
            r = requests.get(url, auth=AUTH, timeout=15)
            if r.status_code == 429:
                wait = sleep * (2 ** attempt)
                print(f"429 rate limited. Sleeping {wait}s…")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Fetch error ({url}): {e}")
            time.sleep(sleep)
    return {}


def ensure_history_tables(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS horse_results (
        horse_id   TEXT NOT NULL,
        race_id    TEXT,
        date       TEXT,
        position   INT,
        class      TEXT,
        course     TEXT,
        going      TEXT,
        dist_m     REAL,
        draw       TEXT,
        weight     TEXT,
        weight_lbs REAL,
        sp_dec     REAL,
        btn        REAL,
        time       TEXT,
        off_dt     TEXT,
        PRIMARY KEY (horse_id, race_id)
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hr_horse_date ON horse_results(horse_id, date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hr_race ON horse_results(race_id)")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS jockey_results (
        jockey_id TEXT,
        race_id   TEXT,
        date      TEXT,
        position  INT,
        PRIMARY KEY (jockey_id, race_id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trainer_results (
        trainer_id TEXT,
        race_id    TEXT,
        date       TEXT,
        position   INT,
        PRIMARY KEY (trainer_id, race_id)
    )
    """)
    conn.commit()


# ---------------- Horses (now with UPSERT) ----------------

def _fnum(x):
    try:
        return float(x)
    except Exception:
        return None

def fetch_horse_history(conn, horse_id, upsert: bool):
    """
    If upsert=True, always call API and upsert per race.
    If upsert=False, skip horses that already have any rows (fast path).
    """
    cur = conn.cursor()

    if not upsert:
        cur.execute("SELECT 1 FROM horse_results WHERE horse_id=? LIMIT 1", (horse_id,))
        if cur.fetchone():
            return  # quick skip

    data = fetch(f"horses/{horse_id}/results")
    if not data:
        return

    results = data.get("results", [])
    for race in results:
        # find this horse’s row within the race
        runner = None
        for r in race.get("runners", []) or []:
            if r.get("horse_id") == horse_id:
                runner = r
                break
        if runner is None:
            continue

        pos = runner.get("position")
        cur.execute("""
        INSERT INTO horse_results
        (horse_id, race_id, date, position, class, course, going, dist_m, draw,
         weight, weight_lbs, sp_dec, btn, time, off_dt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(horse_id, race_id) DO UPDATE SET
            date       = COALESCE(excluded.date,       horse_results.date),
            position   = COALESCE(excluded.position,   horse_results.position),
            class      = COALESCE(excluded.class,      horse_results.class),
            course     = COALESCE(excluded.course,     horse_results.course),
            going      = COALESCE(excluded.going,      horse_results.going),
            dist_m     = COALESCE(excluded.dist_m,     horse_results.dist_m),
            draw       = COALESCE(excluded.draw,       horse_results.draw),
            weight     = COALESCE(excluded.weight,     horse_results.weight),
            weight_lbs = COALESCE(excluded.weight_lbs, horse_results.weight_lbs),
            sp_dec     = COALESCE(excluded.sp_dec,     horse_results.sp_dec),
            btn        = COALESCE(excluded.btn,        horse_results.btn),
            time       = COALESCE(excluded.time,       horse_results.time),
            off_dt     = COALESCE(excluded.off_dt,     horse_results.off_dt)
        """, (
            horse_id,
            race.get("race_id"),
            race.get("date"),
            int(pos) if pos and str(pos).isdigit() else None,
            race.get("class"),
            race.get("course"),
            race.get("going"),
            _fnum(race.get("dist_m")),
            runner.get("draw"),
            runner.get("weight"),
            _fnum(runner.get("weight_lbs")),
            _fnum(runner.get("sp_dec")),
            _fnum(runner.get("btn")),
            runner.get("time"),
            race.get("off_dt"),
        ))
    conn.commit()


# ---------------- Jockey / Trainer (unchanged semantics) ----------------

def fetch_jockey_history(conn, jockey_id):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM jockey_results WHERE jockey_id=? LIMIT 1", (jockey_id,))
    if cur.fetchone():
        return
    data = fetch(f"jockeys/{jockey_id}/results")
    if not data:
        return
    results = data.get("results", [])
    for race in results:
        for runner in race.get("runners", []) or []:
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
    results = data.get("results", [])
    for race in results:
        for runner in race.get("runners", []) or []:
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


# ---------------- Orchestration ----------------

def process_with_eta(items, fetch_func, label, conn, limit=None, log_every=50, **kwargs):
    total = len(items) if not limit else min(limit, len(items))
    start = datetime.now()
    print(f"Found {len(items)} {label.lower()}. Processing {total}…")
    for i, item_id in enumerate(items[:total], 1):
        fetch_func(conn, item_id, **kwargs)
        if i % log_every == 0 or i == total:
            elapsed = (datetime.now() - start).total_seconds()
            rate = elapsed / i
            eta = datetime.now() + timedelta(seconds=(total - i) * rate)
            print(f"  {label}: {i}/{total} processed (ETA {eta:%H:%M:%S})")

def backfill_histories(db_path=DB_PATH, limit=None, upsert=False):
    conn = sqlite3.connect(db_path)
    ensure_history_tables(conn)

    cur = conn.cursor()

    # Horses (upsert if requested)
    cur.execute("SELECT DISTINCT horse_id FROM runners WHERE horse_id IS NOT NULL")
    horses = [r[0] for r in cur.fetchall()]
    process_with_eta(horses, fetch_horse_history, "Horses", conn, limit, log_every=50, upsert=upsert)

    # Jockeys / Trainers: no extra fields to upsert right now
    cur.execute("SELECT DISTINCT jockey_id FROM runners WHERE jockey_id IS NOT NULL")
    jockeys = [r[0] for r in cur.fetchall()]
    process_with_eta(jockeys, fetch_jockey_history, "Jockeys", conn, limit, log_every=20)

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
    parser.add_argument("--upsert", action="store_true", help="Refetch horses and upsert fields even if rows exist")
    args = parser.parse_args()

    backfill_histories(args.db, args.limit, upsert=args.upsert)
