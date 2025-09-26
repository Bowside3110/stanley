import json
import sqlite3
from pathlib import Path

DB_PATH = "data/historical/hkjc.db"
JSON_PATH = "races_2025_09_28.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def insert_races(db_path, races_json):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # adjust to your actual schema; assumes a "runners" table exists
    for meeting in races_json:
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

                # Insert into your schema
                cur.execute("""
                    INSERT OR REPLACE INTO runners
                    (race_id, race_date, race_no, race_name,
                     horse_id, horse, trainer_id, trainer,
                     jockey_id, jockey, draw, weight, win_odds, position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """, (
                    race_id, race_date, race_no, race_name,
                    horse_id, horse_name, trainer_id, trainer_name,
                    jockey_id, jockey_name, draw, weight, win_odds
                ))

    conn.commit()
    conn.close()
    print("✅ Races imported into database")

if __name__ == "__main__":
    data = load_json(JSON_PATH)
    insert_races(DB_PATH, data)
