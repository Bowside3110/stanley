# tests/test_racing_api.py
import re
import requests
import pandas as pd
from datetime import date
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

def clean_race_name(name: str) -> str:
    if not name:
        return ""
    # Drop "(Class X)" and normalise whitespace
    name = re.sub(r"\(Class\s*\d+\)", "", name)
    return re.sub(r"\s+", " ", name).strip()

def flatten_racecards(racecards, course_filter="Sha Tin"):
    races = racecards if isinstance(racecards, list) else racecards.get("racecards", [])
    rows = []
    for race in races:
        if course_filter.lower() not in race.get("course", "").lower():
            continue
        race_meta = {
            "date": race.get("date"),
            "course": race.get("course"),
            "race_name": race.get("race_name"),
            "race_class": extract_class(race.get("race_name")),
            "distance": race.get("dist_m"),
            "going": race.get("going"),
        }
        for runner in race.get("runners", []):
            row = race_meta.copy()
            row.update({
                "horse": runner.get("horse"),
                "horse_id": runner.get("horse_id"),
                "jockey": runner.get("jockey"),
                "trainer": runner.get("trainer"),
                "draw": runner.get("draw"),
                "weight": runner.get("weight"),
                "odds": runner.get("odds"),
            })
            rows.append(row)
    return pd.DataFrame(rows)

def flatten_results(results, course_filter="Sha Tin"):
    races = results if isinstance(results, list) else results.get("results", [])
    rows = []
    for race in races:
        if course_filter.lower() not in race.get("course", "").lower():
            continue
        race_meta = {
            "date": race.get("date"),
            "course": race.get("course"),
            "race_name": race.get("race_name"),
            "race_class": extract_class(race.get("race_name")),
            "distance": race.get("dist_m"),
            "going": race.get("going"),
        }
        for runner in race.get("runners", []):
            row = race_meta.copy()
            row.update({
                "horse": runner.get("horse"),
                "horse_id": runner.get("horse_id"),
                "jockey": runner.get("jockey"),
                "trainer": runner.get("trainer"),
                "draw": runner.get("draw"),
                "position": runner.get("position"),
                "odds_result": runner.get("odds"),
            })
            rows.append(row)
    return pd.DataFrame(rows)

def test_hong_kong_pipeline_with_horses():
    today = str(date.today())

    # 1. Racecards (Sha Tin)
    racecards = fetch("racecards/free")
    df_cards = flatten_racecards(racecards, course_filter="Sha Tin")
    if not df_cards.empty:
        df_cards["race_name_clean"] = df_cards["race_name"].apply(clean_race_name)
        cards_file = f"sha_tin_racecards_{today}.csv"
        df_cards.to_csv(cards_file, index=False)
        print(f"\nSaved {len(df_cards)} Sha Tin racecard rows to {cards_file}")
        print(df_cards.groupby("race_name_clean").size())
    else:
        print("\nNo Sha Tin racecards available.")
        return

    # 2. Results (Sha Tin)
    results = fetch("results/today")
    df_results = flatten_results(results, course_filter="Sha Tin")
    if not df_results.empty:
        df_results["race_name_clean"] = df_results["race_name"].apply(clean_race_name)
        results_file = f"sha_tin_results_{today}.csv"
        df_results.to_csv(results_file, index=False)
        print(f"\nSaved {len(df_results)} Sha Tin results rows to {results_file}")
        print(df_results.groupby("race_name_clean").size())
    else:
        print("\nNo Sha Tin results available.")

    # 3. Merge
    if not df_cards.empty and not df_results.empty:
        merged = pd.merge(
            df_cards,
            df_results,
            on=["date", "course", "race_name_clean", "horse", "jockey", "trainer", "draw", "race_class", "distance", "going", "horse_id"],
            how="inner",
            suffixes=("_card", "_result")
        )
        merged_file = f"sha_tin_merged_{today}.csv"
        merged.to_csv(merged_file, index=False)
        print(f"\nSaved merged Sha Tin data with {len(merged)} rows to {merged_file}")
        print(merged.head(10))

    # 4. Sample horse enrichment (first 2 horses in first race)
    first_race = df_cards[df_cards["race_name_clean"] == df_cards["race_name_clean"].iloc[0]]
    sample_horses = first_race.dropna(subset=["horse_id"]).head(2)

    for _, runner in sample_horses.iterrows():
        horse_id = runner["horse_id"]
        horse_name = runner["horse"]
        print(f"\n=== Enriching horse {horse_name} ({horse_id}) ===")

        try:
            past_results = fetch(f"horses/{horse_id}/results")
            if isinstance(past_results, list) and past_results:
                print(f"Past results for {horse_name} (showing 5):")
                print(pd.DataFrame(past_results).head())
            else:
                print(f"No past results for {horse_name}")

            dist_analysis = fetch(f"horses/{horse_id}/analysis/distance-times")
            print(f"Distance analysis for {horse_name}:")
            print(dist_analysis if isinstance(dist_analysis, dict) else str(dist_analysis)[:300])
        except requests.HTTPError as e:
            print(f"Error fetching data for {horse_name}: {e}")

if __name__ == "__main__":
    test_hong_kong_pipeline_with_horses()
