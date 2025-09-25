import os
import re
import sqlite3
import time
import pandas as pd
from datetime import date, datetime
import requests
from requests.auth import HTTPBasicAuth

BASE = "https://api.theracingapi.com/v1"
AUTH = HTTPBasicAuth("mnLPvpPyIPk9NodfZOKdzfH0", "XjLJwHmwsrAX6yco36zr3dsg")
DB_PATH = "data/historical/hkjc.db"

def fetch(path, params=None, retries=3, sleep_time=1):
    url = f"{BASE}/{path.lstrip('/')}"
    for attempt in range(retries):
        try:
            r = requests.get(url, auth=AUTH, params=params or {}, timeout=30)
            if r.status_code == 429:  # Rate limited
                wait = sleep_time * (2 ** attempt)
                print(f"Rate limited. Sleeping {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            if attempt < retries - 1:
                time.sleep(sleep_time)
            else:
                print(f"Failed after {retries} attempts")
                return None

def extract_class(name: str):
    if not name:
        return None
    m = re.search(r"\(Class\s*(\d+)\)", name)
    return f"Class {m.group(1)}" if m else None

def clean_weight(val):
    if not val:
        return None
    try:
        return float(re.sub(r"[^\d.]", "", str(val)))
    except Exception:
        return None

def compute_form_features(horse_id):
    """Compute avg_last3 and days_since_last from past results."""
    try:
        results = fetch(f"horses/{horse_id}/results")
    except Exception:
        return None, None

    if not results or not isinstance(results, list):
        return None, None

    df = pd.DataFrame(results)
    if "date" not in df.columns or "position" not in df.columns:
        return None, None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "position"]).sort_values("date", ascending=False)

    # Last 3 runs average finishing position
    try:
        avg_last3 = df.head(3)["position"].astype(float).mean()
    except Exception:
        avg_last3 = None

    # Days since last run
    try:
        last_date = df.iloc[0]["date"]
        days_since_last = (datetime.today() - last_date).days
    except Exception:
        days_since_last = None

    return avg_last3, days_since_last

def fetch_odds(race_id, horse_id):
    """Fetch odds data for a specific horse in a race using the upgraded API."""
    print(f"Fetching odds for race {race_id}, horse {horse_id}")
    path = f"odds/{race_id}/{horse_id}"
    try:
        data = fetch(path)
        if not data:
            print(f"  No data returned for race {race_id}, horse {horse_id}")
            return None
            
        # The API returns odds data as a list under the 'odds' key
        odds_data = data.get("odds", [])
        print(f"  Found {len(odds_data)} odds entries for race {race_id}, horse {horse_id}")
        
        # If we have odds data, use the first bookmaker's decimal odds
        # We could also average across bookmakers or use a specific one
        if odds_data and len(odds_data) > 0:
            odds_value = float(odds_data[0].get("decimal", 0))
            print(f"  Using odds value: {odds_value}")
            return odds_value
            
        print(f"  No odds data found for race {race_id}, horse {horse_id}")
        return None
    except Exception as e:
        print(f"Error fetching odds for race {race_id}, horse {horse_id}: {e}")
        return None


def normalize_course_name(course_name):
    """Normalize course names to match the database format."""
    if not course_name:
        return None
    if "happy valley" in course_name.lower():
        return "Happy Valley (HK)"
    elif "sha tin" in course_name.lower():
        return "Sha Tin (HK)"
    return course_name


def build_future_dataset(courses=("Sha Tin", "Happy Valley"), update_db=True, db_path=DB_PATH, fetch_odds_data=True):
    racecards = fetch("racecards/pro")  # Using pro endpoint for better data
    races = racecards if isinstance(racecards, list) else racecards.get("racecards", [])
    rows = []

    for race in races:
        course_name = race.get("course", "")
        if not any(c.lower() in course_name.lower() for c in courses):
            continue

        # Try to parse race number (API sometimes lacks explicit race_no)
        race_no = race.get("race_number") or (len(rows) + 1)

        # Normalize course name for database consistency
        normalized_course = normalize_course_name(course_name)
        
        race_meta = {
            "race_date": race.get("date"),
            "race_no": race_no,
            "course": normalized_course,
            "race_name": race.get("race_name"),
            "race_class": extract_class(race.get("race_name")),
            "distance": race.get("dist_m"),
            "going": race.get("going"),
            "race_id": race.get("race_id"),  # Store race_id for database and odds lookup
        }

        for i, runner in enumerate(race.get("runners", []), start=1):
            horse_id = runner.get("horse_id")
            avg_last3, days_since_last = compute_form_features(horse_id) if horse_id else (None, None)
            
            # Fetch odds data for this horse using the upgraded API
            win_odds = None
            if fetch_odds_data and horse_id and race.get("race_id"):
                try:
                    win_odds = fetch_odds(race.get("race_id"), horse_id)
                    # Small delay to avoid hitting rate limits
                    time.sleep(0.2)
                except Exception as e:
                    print(f"Error fetching odds for {horse_id}: {e}")
                    # Continue with None odds

            row = race_meta.copy()
            row.update({
                "horse": runner.get("horse"),
                "horse_id": horse_id,
                "horse_no": i,
                "draw_cat": runner.get("draw"),
                "act_wt": clean_weight(runner.get("weight")),
                "win_odds": win_odds,  # Using odds from the upgraded API
                "avg_last3": avg_last3,
                "days_since_last": days_since_last,
                # Placeholders for now:
                "jockey_win_rate": 0.0,
                "trainer_win_rate": 0.0,
                "jt_combo_win_rate": 0.0,
                "jockey": runner.get("jockey"),
                "trainer": runner.get("trainer"),
            })
            rows.append(row)

    df = pd.DataFrame(rows)
    
    # Update the database if requested
    if update_db and not df.empty:
        update_database(df, db_path)
        
    return df

def update_database(df, db_path=DB_PATH):
    """Update the database with race and runner information."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Group by race for easier processing
    for race_id, race_group in df.groupby("race_id"):
        if not race_id:  # Skip if race_id is missing
            continue
            
        # Get race info from the first row
        race_info = race_group.iloc[0]
        race_date = pd.to_datetime(race_info["race_date"]).strftime("%Y-%m-%d")
        
        # Insert into races table
        cur.execute(
            """
            INSERT OR REPLACE INTO races (race_id, date, course, race_name, class, distance, going, rail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                race_id,
                race_date,
                race_info["course"],
                race_info["race_name"],
                race_info["race_class"],
                race_info["distance"],
                race_info["going"],
                None,  # rail not available in this dataset
            ),
        )
        
        # Insert into racecard_pro table
        cur.execute(
            """
            INSERT OR REPLACE INTO racecard_pro (race_id, date, course, race_name, race_class, 
                                               going, dist_m, updated_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                race_id,
                race_date,
                race_info["course"],
                race_info["race_name"],
                race_info["race_class"],
                race_info["going"],
                race_info["distance"],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        
        # Insert runners
        for _, runner in race_group.iterrows():
            horse_id = runner["horse_id"]
            if not horse_id:  # Skip if horse_id is missing
                continue
                
            # Insert into runners table
            cur.execute(
                """
                INSERT OR REPLACE INTO runners (race_id, horse_id, horse, draw, weight, jockey, jockey_id, trainer, trainer_id, win_odds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    race_id,
                    horse_id,
                    runner["horse"],
                    runner["draw_cat"],
                    str(runner["act_wt"]) if pd.notna(runner["act_wt"]) else None,
                    runner["jockey"],
                    None,  # jockey_id not available
                    runner["trainer"],
                    None,  # trainer_id not available
                    runner["win_odds"],
                ),
            )
            
            # Insert into racecard_pro_runners table
            cur.execute(
                """
                INSERT OR REPLACE INTO racecard_pro_runners (race_id, horse_id, horse, draw, weight, weight_lbs,
                                                         jockey, trainer, win_odds, updated_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    race_id,
                    horse_id,
                    runner["horse"],
                    runner["draw_cat"],
                    str(runner["act_wt"]) if pd.notna(runner["act_wt"]) else None,
                    runner["act_wt"],
                    runner["jockey"],
                    runner["trainer"],
                    runner["win_odds"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    print(f"✅ Updated database with {len(df)} runners")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build future dataset and update database")
    parser.add_argument("--no-db", action="store_true", help="Skip updating the database")
    parser.add_argument("--no-odds", action="store_true", help="Skip fetching odds data")
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite DB")
    parser.add_argument("--course", default="Happy Valley", help="Course name (Happy Valley or Sha Tin)")
    args = parser.parse_args()
    
    today = str(date.today())
    df = build_future_dataset(courses=[args.course], update_db=not args.no_db, db_path=args.db, fetch_odds_data=not args.no_odds)
    
    # Save to CSV for backward compatibility
    os.makedirs("data/future", exist_ok=True)
    out_path = f"data/future/hkjc_future_{today}_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Built future dataset: {out_path} with {len(df)} rows")
