import requests
import pandas as pd
from requests.auth import HTTPBasicAuth

BASE = "https://api.theracingapi.com/v1"
AUTH = HTTPBasicAuth("mnLPvpPyIPk9NodfZOKdzfH0", "XjLJwHmwsrAX6yco36zr3dsg")

def fetch_results_day(target_date, course_filter="Sha Tin", limit=50):
    """Fetch all results for a single date, paginating with skip."""
    all_rows = []
    skip = 0
    while True:
        params = {
            "start_date": target_date,
            "end_date": target_date,
            "limit": limit,
            "skip": skip,
        }
        r = requests.get(f"{BASE}/results", auth=AUTH, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        races = data.get("results", [])
        if not races:
            break

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
            for runner in race.get("runners", []):
                row = race_meta.copy()
                row.update({
                    "horse_id": runner.get("horse_id"),
                    "horse": runner.get("horse"),
                    "position": runner.get("position"),
                    "draw": runner.get("draw"),
                    "weight": runner.get("weight"),
                    "jockey": runner.get("jockey"),
                    "trainer": runner.get("trainer"),
                })
                all_rows.append(row)

        skip += limit

    return pd.DataFrame(all_rows)


def test_results_pagination():
    target_date = "2025-09-07"  # Sha Tin meeting date
    df = fetch_results_day(target_date, "Sha Tin")
    print(f"\n=== Results for {target_date} (Sha Tin) ===")
    if df.empty:
        print("No Sha Tin results found.")
    else:
        print(df.head(20))


if __name__ == "__main__":
    test_results_pagination()
