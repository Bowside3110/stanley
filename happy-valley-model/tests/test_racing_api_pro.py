import requests
import pandas as pd
from datetime import date
from requests.auth import HTTPBasicAuth

BASE = "https://api.theracingapi.com/v1"
AUTH = HTTPBasicAuth("mnLPvpPyIPk9NodfZOKdzfH0", "XjLJwHmwsrAX6yco36zr3dsg")

def fetch(endpoint, params=None):
    url = f"{BASE}/{endpoint.lstrip('/')}"
    r = requests.get(url, auth=AUTH, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def flatten_racecards(racecards):
    races = racecards.get("racecards", [])
    rows = []
    for race in races:
        race_meta = {
            "race_id": race.get("race_id"),
            "date": race.get("date"),
            "course": race.get("course"),
            "race_name": race.get("race_name"),
            "class": race.get("class"),
            "distance": race.get("dist_m"),
            "going": race.get("going"),
        }
        for r in race.get("runners", []):
            row = race_meta.copy()
            row.update({
                "horse_id": r.get("horse_id"),
                "horse": r.get("horse"),
                "draw": r.get("draw"),
                "weight": r.get("weight"),
                "jockey": r.get("jockey"),
                "trainer": r.get("trainer"),
                "win_odds": r.get("win_odds"),
            })
            rows.append(row)
    return pd.DataFrame(rows)

def flatten_results(results):
    races = results.get("results", [])
    rows = []
    for race in races:
        race_meta = {
            "race_id": race.get("race_id"),
            "date": race.get("date"),
            "course": race.get("course"),
            "race_name": race.get("race_name"),
            "class": race.get("class"),
            "distance": race.get("dist_m"),
            "going": race.get("going"),
        }
        for r in race.get("runners", []):
            row = race_meta.copy()
            row.update({
                "horse_id": r.get("horse_id"),
                "horse": r.get("horse"),
                "position": r.get("position"),
            })
            rows.append(row)
    return pd.DataFrame(rows)

def test_today_sha_tin():
    today = str(date.today())

    # Fetch racecards (basic)
    racecards = fetch("racecards/basic", params={"date": today})
    df_cards = flatten_racecards(racecards)
    print("\n=== Racecards (basic) sample ===")
    print(df_cards.head())

    # Fetch results
    results = fetch("results", params={"date": today})
    df_results = flatten_results(results)
    print("\n=== Results sample ===")
    print(df_results.head())

    # Merge by race_id + horse_id
    merged = pd.merge(
        df_cards,
        df_results,
        on=["race_id", "horse_id", "date", "course"],
        how="inner",
        suffixes=("_card", "_result"),
    )
    print("\n=== Joined Racecards + Results (sample) ===")
    print(merged.head(20))

if __name__ == "__main__":
    test_today_sha_tin()
