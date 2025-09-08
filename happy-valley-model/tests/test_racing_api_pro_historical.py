import requests
import pandas as pd
from requests.auth import HTTPBasicAuth

BASE = "https://api.theracingapi.com/v1"
AUTH = HTTPBasicAuth("mnLPvpPyIPk9NodfZOKdzfH0", "XjLJwHmwsrAX6yco36zr3dsg")


def fetch(endpoint, params=None):
    """Generic fetch wrapper with auth + error handling."""
    url = f"{BASE}/{endpoint.lstrip('/')}"
    r = requests.get(url, auth=AUTH, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def flatten_results_day(results, course_filter="Sha Tin"):
    races = results.get("results", [])
    rows = []
    for race in races:
        if course_filter.lower() not in race.get("course", "").lower():
            continue
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
                "draw": r.get("draw"),
                "weight": r.get("weight"),
                "jockey": r.get("jockey"),
                "trainer": r.get("trainer"),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_historical_workflow():
    # Use Sha Tin meeting on Sept 7, 2025
    yday = "2025-09-07"

    # Step 1: Fetch Sha Tin results for that date
    results = fetch("results", params={"start_date": yday, "end_date": yday})
    df_results = flatten_results_day(results, "Sha Tin")
    print("\n=== Sha Tin Results (2025-09-07) ===")
    if df_results.empty:
        print("No Sha Tin results found.")
        return
    print(df_results.head())

    # Step 2: Fetch Pro racecard for the first race_id
    sample_race_id = df_results["race_id"].iloc[0]
    racecard = fetch(f"racecards/{sample_race_id}/pro")
    print(f"\n=== Racecard Pro for race_id {sample_race_id} ===")
    if racecard.get("runners"):
        print(pd.DataFrame(racecard["runners"]).head())
    else:
        print("No runners found in racecard.")

    # Step 3: Fetch horse history for the first horse_id
    sample_horse_id = df_results["horse_id"].iloc[0]
    horse_history = fetch(f"horses/{sample_horse_id}/results")
    print(f"\n=== Past results for horse_id {sample_horse_id} ===")
    if isinstance(horse_history, list) and horse_history:
        print(pd.DataFrame(horse_history).head())
    else:
        print("No history returned for this horse.")


if __name__ == "__main__":
    test_historical_workflow()
