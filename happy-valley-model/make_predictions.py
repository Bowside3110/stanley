import subprocess
import sqlite3
import json
from pathlib import Path
import os
import pandas as pd

DB_PATH = "data/historical/hkjc.db"
PREDICTIONS_DIR = Path("data/predictions")

def run_node_fetch():
    """Run the Node fetcher to get the next meeting's races"""
    print("🔄 Running Node fetcher...")
    result = subprocess.run(
        ["node", "fetch_next_meeting.mjs"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("❌ Node fetcher failed:")
        print(result.stderr)
        raise RuntimeError("Node fetch failed")
    print(result.stdout)

def get_latest_json():
    """Find the newest racecard JSON file"""
    files = sorted(PREDICTIONS_DIR.glob("races_*.json"), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError("No race JSON files found in data/predictions/")
    return files[-1]

def import_races(json_path):
    """Insert races + runners into hkjc.db"""
    print(f"📥 Importing races from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "raceMeetings" in data:
        meetings = data["raceMeetings"]
    else:
        raise ValueError("Unexpected JSON structure: no 'raceMeetings' found")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    race_count, runner_count = 0, 0
    missing_odds = 0

    for meeting in meetings:
        race_date = meeting["date"]
        course = meeting["venueCode"]

        for race in meeting.get("races", []):
            race_id = race["id"]
            race_no = race["no"]
            race_name = race.get("raceName_en", "")
            race_class = race.get("raceClass_en", "")
            distance = race.get("distance")
            going = race.get("raceTrack", {}).get("description_en")
            rail = race.get("raceCourse", {}).get("description_en")

            # Insert race
            cur.execute("""
                INSERT OR REPLACE INTO races
                (race_id, date, course, race_name, class, distance, going, rail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id, race_date, course, race_name, race_class,
                distance, going, rail
            ))
            race_count += 1

            # Insert runners
            for runner in race.get("runners", []):
                horse_id = runner["id"]
                horse_name = runner.get("name_en", "")
                draw = runner.get("barrierDrawNumber")
                weight = runner.get("handicapWeight")
                win_odds = runner.get("winOdds")

                jockey_id = runner["jockey"]["code"] if runner.get("jockey") else None
                jockey_name = runner["jockey"]["name_en"] if runner.get("jockey") else None
                trainer_id = runner["trainer"]["code"] if runner.get("trainer") else None
                trainer_name = runner["trainer"]["name_en"] if runner.get("trainer") else None

                # NEW: capture runner status
                status = runner.get("status", "unknown")

                cur.execute("""
                    INSERT OR REPLACE INTO runners
                    (race_id, horse_id, horse, draw, weight, jockey, jockey_id,
                    trainer, trainer_id, win_odds, position, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """, (
                    race_id, horse_id, horse_name, draw, weight,
                    jockey_name, jockey_id, trainer_name, trainer_id, win_odds, status
                ))

                runner_count += 1

    conn.commit()
    conn.close()
    print(f"✅ Imported {race_count} races and {runner_count} runners into database")

    if missing_odds > 0:
        print(f"⚠️ Warning: {missing_odds} runners are missing win odds. "
              f"Predictions will be made without odds data.")

def run_predictions(date):
    """Run your predict_future.py with the given date"""
    print(f"🎯 Running predictions for {date}")
    out_csv = PREDICTIONS_DIR / f"predictions_{date}.csv"
    subprocess.run([
        "python", "-m", "src.predict_future",
        "--db", str(DB_PATH),
        "--date", date,
        "--box", "5",
        "--save_csv", str(out_csv)
    ])
    return out_csv

def show_cheat_sheet(csv_path: str):
    import pandas as pd
    df = pd.read_csv(csv_path)

    # If status column exists, filter to Declared only
    if "status" in df.columns:
        non_declared = df[df["status"].str.lower() != "declared"]
        if not non_declared.empty:
            print(f"⚠️ Note: {len(non_declared)} non-declared runners (scratched/standby) "
                  f"excluded from cheat sheet output.")
        df = df[df["status"].str.lower() == "declared"]

    # Group by race and display
    for rid, g in df.groupby("race_id"):
        race_name = g["race_name"].iloc[0] if "race_name" in g else "Unknown Race"
        print(f"\nRace: {race_name}")
        print("Ranked selections:")

        top_n = g.sort_values("rank").head(5)
        for _, row in top_n.iterrows():
            horse = row["horse"]
            score = row["score"]
            print(f" {row['rank']}. {horse} ({score:.3f})")

if __name__ == "__main__":
    run_node_fetch()
    latest_json = get_latest_json()
    import_races(latest_json)

    # Extract the date from filename e.g. races_2025-09-28_ST.json
    race_date = Path(latest_json).stem.split("_")[1]
    predictions_csv = run_predictions(race_date)
    show_cheat_sheet(predictions_csv)
