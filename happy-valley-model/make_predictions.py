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

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for meeting in data:
        race_date = meeting["date"]
        venue = meeting["venueCode"]

        for race in meeting.get("races", []):
            race_id = race["id"]
            race_no = race["no"]
            race_name = race.get("raceName_en", "")

            for runner in race.get("runners", []):
                horse_id = runner["id"]
                horse_name = runner.get("name_en", "")
                trainer_id = runner["trainer"]["code"] if runner.get("trainer") else None
                trainer_name = runner["trainer"]["name_en"] if runner.get("trainer") else None
                jockey_id = runner["jockey"]["code"] if runner.get("jockey") else None
                jockey_name = runner["jockey"]["name_en"] if runner.get("jockey") else None
                draw = runner.get("barrierDrawNumber")
                weight = runner.get("handicapWeight")
                win_odds = runner.get("winOdds")

                cur.execute("""
                    INSERT OR REPLACE INTO runners
                    (race_id, race_date, race_no, race_name,
                     venue, horse_id, horse, trainer_id, trainer,
                     jockey_id, jockey, draw, weight, win_odds, position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """, (
                    race_id, race_date, race_no, race_name,
                    venue, horse_id, horse_name, trainer_id, trainer_name,
                    jockey_id, jockey_name, draw, weight, win_odds
                ))

    conn.commit()
    conn.close()
    print("✅ Races imported into database")

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

def show_cheat_sheet(csv_path):
    """Print the predictions cheat sheet with venue info"""
    print(f"\n📊 Cheat sheet from {csv_path}")
    df = pd.read_csv(csv_path)
    for (race_id, venue), group in df.groupby(["race_id", "race_name"]):
        race_name = group["race_name"].iloc[0]
        # try to show venue if present
        venue_val = group["venue"].iloc[0] if "venue" in group else ""
        header = f"{race_name} ({venue_val})" if venue_val else race_name
        print(f"\n=== {header} ===")
        for _, row in group.sort_values("rank").iterrows():
            print(f"{int(row['rank'])}. {row['horse']} ({row['score']:.3f})")

if __name__ == "__main__":
    run_node_fetch()
    latest_json = get_latest_json()
    import_races(latest_json)

    # Extract the date from filename e.g. races_2025-09-28_ST.json
    race_date = Path(latest_json).stem.split("_")[1]
    predictions_csv = run_predictions(race_date)
    show_cheat_sheet(predictions_csv)
