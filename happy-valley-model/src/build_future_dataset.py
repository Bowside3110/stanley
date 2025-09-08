import os
import re
import pandas as pd
from datetime import date, datetime
import requests
from requests.auth import HTTPBasicAuth

BASE = "https://api.theracingapi.com/v1"
AUTH = HTTPBasicAuth("mnLPvpPyIPk9NodfZOKdzfH0", "XjLJwHmwsrAX6yco36zr3dsg")

def fetch(path, params=None):
    url = f"{BASE}/{path.lstrip('/')}"
    r = requests.get(url, auth=AUTH, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

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

def build_future_dataset(courses=("Sha Tin", "Happy Valley")):
    racecards = fetch("racecards/free")
    races = racecards if isinstance(racecards, list) else racecards.get("racecards", [])
    rows = []

    for race in races:
        course_name = race.get("course", "")
        if not any(c.lower() in course_name.lower() for c in courses):
            continue

        # Try to parse race number (API sometimes lacks explicit race_no)
        race_no = race.get("race_number") or (len(rows) + 1)

        race_meta = {
            "race_date": race.get("date"),
            "race_no": race_no,
            "course": course_name,
            "race_name": race.get("race_name"),
            "race_class": extract_class(race.get("race_name")),
            "distance": race.get("dist_m"),
            "going": race.get("going"),
        }

        for i, runner in enumerate(race.get("runners", []), start=1):
            horse_id = runner.get("horse_id")
            avg_last3, days_since_last = compute_form_features(horse_id) if horse_id else (None, None)

            row = race_meta.copy()
            row.update({
                "horse": runner.get("horse"),
                "horse_id": horse_id,
                "horse_no": i,
                "draw_cat": runner.get("draw"),
                "act_wt": clean_weight(runner.get("weight")),
                "win_odds": runner.get("odds"),  # may be None at your plan level
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

    return pd.DataFrame(rows)

if __name__ == "__main__":
    today = str(date.today())
    df = build_future_dataset()
    os.makedirs("data/future", exist_ok=True)
    out_path = f"data/future/hkjc_future_{today}_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Built future dataset: {out_path} with {len(df)} rows")
