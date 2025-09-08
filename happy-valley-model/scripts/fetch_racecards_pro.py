# scripts/fetch_racecards_pro.py
import os, time, sqlite3, argparse, re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth

DB_PATH = "data/historical/hkjc.db"

API_USER = os.getenv("RACING_API_USER", "mnLPvpPyIPk9NodfZOKdzfH0")
API_PASS = os.getenv("RACING_API_PASS", "XjLJwHmwsrAX6yco36zr3dsg")
AUTH = HTTPBasicAuth(API_USER, API_PASS)
BASE = "https://api.theracingapi.com/v1"


# ---------------- Helpers ----------------

def fetch(endpoint: str, retries: int = 3, base_sleep: float = 1.5) -> Optional[Dict[str, Any]]:
    url = f"{BASE}/{endpoint.lstrip('/')}"
    for i in range(retries):
        try:
            r = requests.get(url, auth=AUTH, timeout=20)
            if r.status_code == 429:
                wait = base_sleep * (2 ** i)
                print(f"[429] Sleeping {wait:.1f}s…")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = base_sleep * (2 ** i)
            print(f"[HTTP] Error {endpoint}: {e} (retry {i+1}/{retries})")
            time.sleep(wait)
    return None

def _fnum(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None

def _distance_to_meters(s: str) -> Optional[float]:
    if not s:
        return None
    st = str(s).lower().strip()
    if st.endswith("m"):
        try: return float(st[:-1])
        except: return None
    if st.endswith("f"):
        try: return float(st[:-1]) * 201.168
        except: return None
    try: return float(st)
    except: return None

def _parse_class_from_name(name: str) -> str:
    """Extract 'Class X' from a race_name string, e.g. '(Class 4)'."""
    if not name:
        return ""
    m = re.search(r"\(Class\s+(\d)\)", name, flags=re.IGNORECASE)
    if m:
        return f"Class {m.group(1)}"
    return ""


# ---------------- DB schema ----------------

def ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS racecard_pro (
        race_id     TEXT PRIMARY KEY,
        date        TEXT,
        course      TEXT,
        race_name   TEXT,
        race_class  TEXT,
        age_band    TEXT,
        rating_band TEXT,
        pattern     TEXT,
        going       TEXT,
        surface     TEXT,
        dist        TEXT,
        dist_m      REAL,
        rail        TEXT,
        off_time    TEXT,
        off_dt      TEXT,
        updated_utc TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS racecard_pro_runners (
        race_id          TEXT NOT NULL,
        horse_id         TEXT NOT NULL,
        horse            TEXT,
        number           TEXT,
        draw             TEXT,
        weight           TEXT,
        weight_lbs       REAL,
        headgear         TEXT,
        headgear_run     TEXT,
        wind_surgery     TEXT,
        wind_surgery_run TEXT,
        last_run         TEXT,
        form             TEXT,
        jockey           TEXT,
        jockey_id        TEXT,
        trainer          TEXT,
        trainer_id       TEXT,
        win_odds         REAL,
        updated_utc      TEXT,
        PRIMARY KEY (race_id, horse_id)
    )
    """)
    conn.commit()


# ---------------- Upserts ----------------

def upsert_race(conn, race: Dict[str, Any]) -> None:
    cur = conn.cursor()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    race_class = race.get("race_class") or _parse_class_from_name(race.get("race_name"))
    cur.execute("""
    INSERT INTO racecard_pro
    (race_id, date, course, race_name, race_class, age_band, rating_band, pattern,
     going, surface, dist, dist_m, rail, off_time, off_dt, updated_utc)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(race_id) DO UPDATE SET
      date        = excluded.date,
      course      = excluded.course,
      race_name   = excluded.race_name,
      race_class  = excluded.race_class,
      age_band    = excluded.age_band,
      rating_band = excluded.rating_band,
      pattern     = excluded.pattern,
      going       = excluded.going,
      surface     = excluded.surface,
      dist        = excluded.dist,
      dist_m      = excluded.dist_m,
      rail        = excluded.rail,
      off_time    = excluded.off_time,
      off_dt      = excluded.off_dt,
      updated_utc = excluded.updated_utc
    """, (
        race.get("race_id"),
        race.get("date"),
        race.get("course"),
        race.get("race_name"),
        race_class,
        race.get("age_band"),
        race.get("rating_band"),
        race.get("pattern"),
        race.get("going"),
        race.get("surface"),
        race.get("distance"),
        _distance_to_meters(race.get("distance")),
        race.get("rail_movements"),
        race.get("off_time"),
        race.get("off_dt"),
        now,
    ))
    conn.commit()

def upsert_runners(conn, race_id: str, runners: List[Dict[str, Any]]) -> None:
    cur = conn.cursor()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    if not runners:
        return
    for r in runners:
        cur.execute("""
        INSERT INTO racecard_pro_runners
        (race_id, horse_id, horse, number, draw, weight, weight_lbs,
         headgear, headgear_run, wind_surgery, wind_surgery_run,
         last_run, form, jockey, jockey_id, trainer, trainer_id,
         win_odds, updated_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(race_id, horse_id) DO UPDATE SET
            horse            = excluded.horse,
            number           = excluded.number,
            draw             = excluded.draw,
            weight           = excluded.weight,
            weight_lbs       = excluded.weight_lbs,
            headgear         = excluded.headgear,
            headgear_run     = excluded.headgear_run,
            wind_surgery     = excluded.wind_surgery,
            wind_surgery_run = excluded.wind_surgery_run,
            last_run         = excluded.last_run,
            form             = excluded.form,
            jockey           = excluded.jockey,
            jockey_id        = excluded.jockey_id,
            trainer          = excluded.trainer,
            trainer_id       = excluded.trainer_id,
            win_odds         = excluded.win_odds,
            updated_utc      = excluded.updated_utc
        """, (
            race_id,
            r.get("horse_id"),
            r.get("horse"),
            r.get("number"),
            r.get("draw"),
            r.get("weight"),
            _fnum(r.get("weight_lbs")),
            r.get("headgear"),
            r.get("headgear_run"),
            r.get("wind_surgery") or r.get("wind_surge"),
            r.get("wind_surgery_run"),
            r.get("last_run"),
            r.get("form"),
            r.get("jockey"),
            r.get("jockey_id"),
            r.get("trainer"),
            r.get("trainer_id"),
            _fnum(r.get("sp_dec") or r.get("win_odds")),
            now,
        ))
    conn.commit()


# ---------------- Orchestration ----------------

def get_race_ids(conn, start: str, end: str, all: bool) -> List[str]:
    cur = conn.cursor()
    if all:
        sql = "SELECT race_id FROM races WHERE date BETWEEN ? AND ? ORDER BY date"
    else:
        sql = """
        SELECT r.race_id
        FROM races r
        LEFT JOIN racecard_pro_runners pr ON pr.race_id = r.race_id
        WHERE r.date BETWEEN ? AND ? AND pr.race_id IS NULL
        ORDER BY r.date
        """
    cur.execute(sql, (start, end))
    return [row[0] for row in cur.fetchall()]

def run(db_path: str, start: str, end: str, all: bool, limit: Optional[int]):
    conn = sqlite3.connect(db_path)
    ensure_tables(conn)

    race_ids = get_race_ids(conn, start, end, all)
    if limit:
        race_ids = race_ids[:limit]

    if not race_ids:
        print("No races found for that window.")
        return

    print(f"Found {len(race_ids)} races.")
    for i, rid in enumerate(race_ids, 1):
        data = fetch(f"racecards/{rid}/pro")
        if not data:
            continue
        upsert_race(conn, data)
        upsert_runners(conn, rid, data.get("runners", []))
        if i % 10 == 0 or i == len(race_ids):
            print(f"  {i}/{len(race_ids)} processed…")

    conn.close()
    print("✅ Racecard enrichment complete.")


# ---------------- CLI ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--start_date", required=True)
    ap.add_argument("--end_date", required=True)
    ap.add_argument("--all", action="store_true", help="Include already-enriched races")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    run(args.db, args.start_date, args.end_date, args.all, args.limit)
