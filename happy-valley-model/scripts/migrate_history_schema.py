# scripts/migrate_history_schema.py
import sqlite3

DB = "data/historical/hkjc.db"

# columns we want in horse_results -> {name: sqlite_type}
TARGET_COLS = {
    "race_id": "TEXT",
    "date": "TEXT",
    "position": "INT",
    "class": "TEXT",
    "course": "TEXT",
    "going": "TEXT",
    "dist_m": "REAL",
    "draw": "TEXT",
    "weight": "TEXT",
    "weight_lbs": "REAL",
    "sp_dec": "REAL",
    "btn": "REAL",
    "time": "TEXT",
    "off_dt": "TEXT",
}

ddl_create = """
CREATE TABLE IF NOT EXISTS horse_results (
  horse_id   TEXT NOT NULL,
  race_id    TEXT,
  date       TEXT,
  position   INT,
  class      TEXT,
  course     TEXT,
  going      TEXT,
  dist_m     REAL,
  draw       TEXT,
  weight     TEXT,
  weight_lbs REAL,
  sp_dec     REAL,
  btn        REAL,
  time       TEXT,
  off_dt     TEXT,
  PRIMARY KEY (horse_id, race_id)
)
"""

def ensure():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(ddl_create)
    # discover existing cols
    cols = {r[1] for r in cur.execute("PRAGMA table_info(horse_results)")}
    # add missing ones
    for col, typ in TARGET_COLS.items():
        if col not in cols:
            cur.execute(f"ALTER TABLE horse_results ADD COLUMN {col} {typ}")
    # helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hr_horse_date ON horse_results(horse_id, date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hr_race ON horse_results(race_id)")
    conn.commit()
    conn.close()
    print("✅ horse_results schema ensured.")

if __name__ == "__main__":
    ensure()
