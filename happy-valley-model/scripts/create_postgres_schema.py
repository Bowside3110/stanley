#!/usr/bin/env python3
"""
Create PostgreSQL schema for Stanley racing database.

This script creates all tables with PostgreSQL-compatible syntax:
- SERIAL instead of AUTOINCREMENT
- Uses proper PostgreSQL constraints
- Creates all necessary indexes
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Get database connection
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL environment variable not set")
    print("   Please add DATABASE_URL to your .env file")
    sys.exit(1)

try:
    import psycopg2
except ImportError:
    print("❌ ERROR: psycopg2 not installed")
    print("   Run: pip install psycopg2-binary")
    sys.exit(1)

# SQL statements to create all tables
CREATE_TABLES = """
-- Races table
CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY,
    date TEXT,
    course TEXT,
    race_name TEXT,
    class TEXT,
    distance REAL,
    going TEXT,
    rail TEXT,
    post_time TEXT
);

-- Runners table (main race participants data)
CREATE TABLE IF NOT EXISTS runners (
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    horse TEXT,
    draw TEXT,
    weight TEXT,
    jockey TEXT,
    jockey_id TEXT,
    trainer TEXT,
    trainer_id TEXT,
    win_odds REAL,
    position INTEGER,
    status TEXT,
    btn REAL,
    time TEXT,
    starting_price REAL,
    predicted_rank INTEGER,
    predicted_score REAL,
    prediction_date TEXT,
    model_version TEXT,
    PRIMARY KEY (race_id, horse_id)
);

-- Results table (minimal results tracking)
CREATE TABLE IF NOT EXISTS results (
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    position INT,
    PRIMARY KEY (race_id, horse_id)
);

-- Backfill log (tracking data imports)
CREATE TABLE IF NOT EXISTS backfill_log (
    date TEXT PRIMARY KEY,
    processed_at TEXT,
    year INTEGER
);

-- Jockey results history
CREATE TABLE IF NOT EXISTS jockey_results (
    jockey_id TEXT NOT NULL,
    race_id TEXT NOT NULL,
    date TEXT,
    position INT,
    PRIMARY KEY (jockey_id, race_id)
);

-- Trainer results history
CREATE TABLE IF NOT EXISTS trainer_results (
    trainer_id TEXT NOT NULL,
    race_id TEXT NOT NULL,
    date TEXT,
    position INT,
    PRIMARY KEY (trainer_id, race_id)
);

-- Horse results history (detailed)
CREATE TABLE IF NOT EXISTS horse_results (
    horse_id TEXT NOT NULL,
    race_id TEXT NOT NULL,
    date TEXT,
    position INT,
    class TEXT,
    course TEXT,
    going TEXT,
    dist_m REAL,
    draw TEXT,
    weight TEXT,
    weight_lbs REAL,
    sp_dec REAL,
    btn REAL,
    time TEXT,
    off_dt TEXT,
    surface TEXT,
    PRIMARY KEY (horse_id, race_id)
);

-- Racecard Pro API data
CREATE TABLE IF NOT EXISTS racecard_pro (
    race_id TEXT PRIMARY KEY,
    date TEXT,
    course TEXT,
    race_name TEXT,
    race_class TEXT,
    age_band TEXT,
    rating_band TEXT,
    pattern TEXT,
    going TEXT,
    surface TEXT,
    dist TEXT,
    dist_m REAL,
    rail TEXT,
    off_time TEXT,
    off_dt TEXT,
    updated_utc TEXT
);

-- Racecard Pro runners
CREATE TABLE IF NOT EXISTS racecard_pro_runners (
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    horse TEXT,
    number TEXT,
    draw TEXT,
    weight TEXT,
    weight_lbs REAL,
    headgear TEXT,
    headgear_run TEXT,
    wind_surgery TEXT,
    wind_surgery_run TEXT,
    last_run TEXT,
    form TEXT,
    jockey TEXT,
    jockey_id TEXT,
    trainer TEXT,
    trainer_id TEXT,
    win_odds REAL,
    updated_utc TEXT,
    PRIMARY KEY (race_id, horse_id)
);

-- Predictions table (multi-version tracking)
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    predicted_rank INTEGER,
    predicted_score REAL,
    prediction_timestamp TEXT NOT NULL,
    model_version TEXT,
    win_odds_at_prediction REAL,
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_hr_horse_date ON horse_results(horse_id, date);
CREATE INDEX IF NOT EXISTS idx_hr_race ON horse_results(race_id);
CREATE INDEX IF NOT EXISTS idx_backfill_log_year ON backfill_log(year);
CREATE INDEX IF NOT EXISTS idx_predictions_race ON predictions(race_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_horse ON predictions(race_id, horse_id);
"""

def create_schema():
    """Create PostgreSQL schema"""
    print("🔄 Connecting to PostgreSQL...")
    print(f"   Database: {DATABASE_URL.split('@')[1].split('/')[0]}")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        print("\n📊 Creating tables...")
        
        # Execute all CREATE TABLE statements
        cursor.execute(CREATE_TABLES)
        conn.commit()
        
        print("✅ All tables created successfully!")
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print(f"\n📋 Created {len(tables)} tables:")
        for table in tables:
            print(f"   • {table[0]}")
        
        cursor.close()
        conn.close()
        
        print("\n✅ Schema migration complete!")
        print("\n🔍 Next steps:")
        print("   1. Verify the schema: psql <connection_string> -c '\\dt'")
        print("   2. Run connection test: python scripts/test_db_connection.py")
        print("   3. Migrate data from SQLite (if needed)")
        
    except Exception as e:
        print(f"\n❌ Error creating schema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_schema()

