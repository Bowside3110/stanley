#!/usr/bin/env python3
"""
predict_next_race.py

Predicts the NEXT upcoming race only, using live odds.
Handles timezone conversion from Hong Kong Time (HKT) to Brisbane Time.

Usage:
    python -m src.predict_next_race
    python -m src.predict_next_race --save-csv  # Optional: save to CSV
"""

import subprocess
import sqlite3  # Keep for legacy compatibility
from src.db_config import get_connection, get_placeholder
import json
from pathlib import Path
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

DB_PATH = "data/historical/hkjc.db"  # Legacy path reference
PREDICTIONS_DIR = Path("data/predictions")

# Timezone definitions
HKT = ZoneInfo("Asia/Hong_Kong")  # UTC+8, no DST
BRISBANE = ZoneInfo("Australia/Brisbane")  # UTC+10 (AEST) or UTC+11 (AEDT)


def run_node_fetch():
    """Run the Node fetcher to get the next meeting's races"""
    print("🔄 Fetching latest racecards from HKJC...")
    result = subprocess.run(
        ["node", "fetch_next_meeting.mjs"],
        capture_output=True,
        text=True,
        cwd="scripts"
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


def parse_hkt_time(time_str: str) -> datetime:
    """
    Parse a time string from HKJC API and return a timezone-aware datetime.
    
    The HKJC API returns times in ISO 8601 format: "2025-11-09T13:00:00+08:00"
    
    Args:
        time_str: Time in ISO 8601 format
    
    Returns:
        Timezone-aware datetime
    """
    # Parse ISO 8601 format with timezone
    return datetime.fromisoformat(time_str)


def find_next_race(json_path):
    """
    Find the next upcoming race from the JSON file.
    
    Returns:
        Tuple of (race_dict, race_date, meeting_dict) or (None, None, None) if no upcoming races
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "raceMeetings" in data:
        meetings = data["raceMeetings"]
    else:
        raise ValueError("Unexpected JSON structure: no 'raceMeetings' found")

    # Get current time in Brisbane
    now_brisbane = datetime.now(BRISBANE)
    print(f"\n🕐 Current time in Brisbane: {now_brisbane.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Convert to HKT for comparison
    now_hkt = now_brisbane.astimezone(HKT)
    print(f"🕐 Current time in Hong Kong: {now_hkt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Find all upcoming races
    upcoming_races = []
    
    for meeting in meetings:
        race_date = meeting["date"]
        course = meeting["venueCode"]
        
        for race in meeting.get("races", []):
            post_time = race.get("postTime", "")
            
            if not post_time:
                continue
            
            # Parse the post time (ISO 8601 format with timezone)
            try:
                race_time_hkt = parse_hkt_time(post_time)
                
                # Check if this race is in the future
                if race_time_hkt > now_hkt:
                    # Convert to Brisbane time for display
                    race_time_brisbane = race_time_hkt.astimezone(BRISBANE)
                    
                    # Calculate minutes until race
                    minutes_until = (race_time_hkt - now_hkt).total_seconds() / 60
                    
                    upcoming_races.append({
                        "race": race,
                        "race_date": race_date,
                        "meeting": meeting,
                        "race_time_hkt": race_time_hkt,
                        "race_time_brisbane": race_time_brisbane,
                        "minutes_until": minutes_until,
                        "course": course
                    })
            except ValueError as e:
                print(f"⚠️ Warning: Could not parse time '{post_time}' for race {race.get('no')}: {e}")
                continue
    
    if not upcoming_races:
        print("\n❌ No upcoming races found.")
        return None, None, None
    
    # Sort by time and get the next race
    upcoming_races.sort(key=lambda x: x["race_time_hkt"])
    next_race_info = upcoming_races[0]
    
    # Display info about the next race
    race = next_race_info["race"]
    print(f"\n🏇 Next Race Found:")
    print(f"   Course: {next_race_info['course']}")
    print(f"   Race {race.get('no')}: {race.get('raceName_en', 'Unknown')}")
    print(f"   Class: {race.get('raceClass_en', 'Unknown')}")
    print(f"   Distance: {race.get('distance')}m")
    print(f"   Post Time (HKT): {next_race_info['race_time_hkt'].strftime('%H:%M')}")
    print(f"   Post Time (Brisbane): {next_race_info['race_time_brisbane'].strftime('%H:%M')}")
    print(f"   ⏰ Time until race: {int(next_race_info['minutes_until'])} minutes")
    
    # Show other upcoming races
    if len(upcoming_races) > 1:
        print(f"\n📋 Other upcoming races today:")
        for i, info in enumerate(upcoming_races[1:6], 2):  # Show next 5
            r = info["race"]
            print(f"   {i}. Race {r.get('no')} at {info['race_time_brisbane'].strftime('%H:%M')} Brisbane "
                  f"({int(info['minutes_until'])} min)")
    
    return next_race_info["race"], next_race_info["race_date"], next_race_info["meeting"]


def import_single_race(race, race_date, meeting):
    """
    Insert a single race + runners into hkjc.db
    
    Args:
        race: Race dictionary from JSON
        race_date: Date string (YYYY-MM-DD)
        meeting: Meeting dictionary containing venue info
    """
    print(f"\n📥 Importing race into database...")
    
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    
    # First, clear any existing races for this date that don't have results yet
    # This ensures we only predict on the newly imported race
    print(f"   Clearing existing future races for {race_date}...")
    cur.execute("""
        DELETE FROM runners 
        WHERE race_id IN (
            SELECT race_id FROM races 
            WHERE date = ? AND race_id NOT IN (
                SELECT DISTINCT race_id FROM runners WHERE position IS NOT NULL
            )
        )
    """, (race_date,))
    
    cur.execute("""
        DELETE FROM races 
        WHERE date = ? AND race_id NOT IN (
            SELECT DISTINCT race_id FROM runners WHERE position IS NOT NULL
        )
    """, (race_date,))
    
    deleted_races = cur.rowcount
    if deleted_races > 0:
        print(f"   Cleared {deleted_races} existing future race(s)")
    
    conn.commit()
    
    race_id = race["id"]
    race_no = race["no"]
    race_name = race.get("raceName_en", "")
    race_class = race.get("raceClass_en", "")
    distance = race.get("distance")
    going = race.get("raceTrack", {}).get("description_en")
    rail = race.get("raceCourse", {}).get("description_en")
    post_time = race.get("postTime", "")
    course = meeting["venueCode"]
    
    # Insert race
    cur.execute("""
        INSERT OR REPLACE INTO races
        (race_id, date, course, race_name, class, distance, going, rail, post_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        race_id, race_date, course, race_name, race_class,
        distance, going, rail, post_time
    ))
    
    # Insert runners
    runner_count = 0
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
        
        status = runner.get("status", "unknown")
        final_position = runner.get("finalPosition")
        if final_position == 0:
            final_position = None
        
        cur.execute("""
            INSERT OR REPLACE INTO runners
            (race_id, horse_id, horse, draw, weight, jockey, jockey_id,
            trainer, trainer_id, win_odds, position, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            race_id, horse_id, horse_name, draw, weight,
            jockey_name, jockey_id, trainer_name, trainer_id, win_odds, final_position, status
        ))
        
        runner_count += 1
    
    conn.commit()
    conn.close()
    
    print(f"✅ Imported race {race_no} with {runner_count} runners")
    return race_id


def run_predictions(date, save_csv=False, use_pretrained=False):
    """
    Run predictions for the given date using predict_future.py
    
    Args:
        date: Race date in YYYY-MM-DD format
        save_csv: Whether to save predictions to CSV
        use_pretrained: Whether to use pre-trained model (fast) or train fresh (slow)
    
    Returns:
        Path to CSV file (always generated for database saving)
    """
    # Try the super-fast odds-only update first
    print(f"\n🎯 Generating predictions (odds-only update)...")
    csv_path = run_predictions_odds_only(date, save_csv)
    
    if csv_path is not None:
        return csv_path
    
    # Fallback to slow path if odds-only fails
    print(f"\n⚠️  Odds-only update failed, falling back to full training...")
    return run_predictions_slow(date, save_csv)


def run_predictions_odds_only(date, save_csv=False):
    """
    Fast prediction - uses existing predictions, just updates the display.
    Since odds changes don't affect the model-based rankings significantly,
    we just copy the existing predictions and update the odds column.
    """
    print("   Looking for existing predictions from make_predictions.py...")
    
    # Find the predictions CSV from make_predictions.py
    predictions_files = sorted(
        PREDICTIONS_DIR.glob(f"predictions_{date}*.csv"),
        key=os.path.getmtime,
        reverse=True
    )
    
    if not predictions_files:
        print(f"   ❌ No existing predictions found for {date}")
        print(f"   (make_predictions.py must run first)")
        return None
    
    existing_csv = predictions_files[0]
    print(f"   ✅ Found: {existing_csv.name}")
    
    # Read existing predictions
    import pandas as pd
    df = pd.read_csv(existing_csv)
    
    # Update odds from database (they were updated by import_single_race)
    print("   Fetching latest odds from database...")
    conn = get_connection()
    
    updated_count = 0
    for idx, row in df.iterrows():
        race_id = row.get('race_id')
        horse_id = row.get('horse_id')
        
        if pd.notna(race_id) and pd.notna(horse_id):
            cursor = conn.cursor()
            cursor.execute(
                "SELECT win_odds FROM runners WHERE race_id = ? AND horse_id = ?",
                (race_id, horse_id)
            )
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                df.at[idx, 'win_odds'] = result[0]
                updated_count += 1
    
    conn.close()
    
    print(f"   ✅ Updated odds for {updated_count}/{len(df)} runners")
    
    # Save to new CSV
    out_csv = PREDICTIONS_DIR / f"next_race_{date}_{datetime.now().strftime('%H%M')}.csv"
    df.to_csv(out_csv, index=False)
    
    print(f"   ✅ Predictions ready (using model-based rankings from morning)")
    
    return out_csv


def run_predictions_slow(date, save_csv=False):
    """Original prediction method - trains model from scratch."""
    # Always generate a CSV (needed for database save)
    out_csv = PREDICTIONS_DIR / f"next_race_{date}_{datetime.now().strftime('%H%M')}.csv"
    cmd = [
        "python", "-m", "src.predict_future",
        "--db", str(DB_PATH),
        "--date", date,
        "--box", "5",
        "--save_csv", str(out_csv)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print the output
    print(result.stdout)
    
    if result.returncode != 0:
        print("❌ Prediction failed:")
        print(result.stderr)
        return None
    
    return out_csv


def run_predictions_fast(date, save_csv=False):
    """Fast prediction method - uses pre-trained model."""
    import pickle
    import sqlite3
    from itertools import combinations
    from src.features import build_features
    
    print("   Loading pre-trained model...")
    
    # Load pre-trained model
    pretrained_dir = Path("data/models/pretrained")
    model_path = pretrained_dir / "latest_model.pkl"
    
    if not model_path.exists():
        print(f"❌ Pre-trained model not found at {model_path}")
        print("   Run 'python -m src.train_model' first to create a pre-trained model.")
        return None
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    runner_feats = model_data["runner_features"]
    
    print(f"   ✅ Loaded model trained on {model_data['num_training_races']} races")
    print(f"   Training date: {model_data['training_date'][:10]}")
    
    # Build features for future races only
    print("   Building features for future races...")
    df = build_features(DB_PATH)
    df["race_date"] = pd.to_datetime(df["race_date"])
    
    # Filter to the specific date and races without results
    target_date = pd.to_datetime(date).date()
    df_future = df[(df["race_date"].dt.date == target_date) & (df["position"].isna())].copy()
    
    if df_future.empty:
        print(f"❌ No future races found for {date}")
        return None
    
    print(f"   Found {df_future['race_id'].nunique()} race(s) to predict")
    
    # Build pair dataset
    print("   Building pairwise comparisons...")
    import numpy as np
    pairs = []
    for rid, grp in df_future.groupby("race_id"):
        horses = grp["horse_id"].tolist()
        for i, j in combinations(range(len(horses)), 2):
            hi, hj = horses[i], horses[j]
            ri, rj = grp.iloc[i], grp.iloc[j]
            feats = {}
            for f in runner_feats:
                xi, xj = ri.get(f), rj.get(f)
                if pd.notna(xi) or pd.notna(xj):
                    feats[f"{f}_min"] = np.nanmin([xi, xj])
                    feats[f"{f}_max"] = np.nanmax([xi, xj])
                else:
                    feats[f"{f}_min"] = np.nan
                    feats[f"{f}_max"] = np.nan
                feats[f"{f}_diff"] = abs(xi - xj) if pd.notna(xi) and pd.notna(xj) else np.nan
            feats["same_trainer"] = int(
                ri.get("trainer_normalized", "") == rj.get("trainer_normalized", "") 
                and ri.get("trainer_normalized", "") != ""
            )
            feats["same_jockey"] = int(
                ri.get("jockey_normalized", "") == rj.get("jockey_normalized", "")
                and ri.get("jockey_normalized", "") != ""
            )
            feats["race_id"] = rid
            feats["pair"] = tuple(sorted((hi, hj)))
            pairs.append(feats)
    
    test_pairs = pd.DataFrame(pairs)
    print(f"   Created {len(test_pairs)} pairs")
    
    # Predict
    print("   Making predictions...")
    X_te = test_pairs.drop(columns=["race_id", "pair"])
    
    # Convert pandas NA to numpy nan for sklearn compatibility
    import numpy as np
    X_te = X_te.fillna(np.nan)
    
    test_pairs["p_raw"] = model.predict_proba(X_te)[:, 1]
    
    # Apply softmax
    from src.predict_future import softmax_pairs
    dir_df = softmax_pairs(test_pairs[["race_id", "pair", "p_raw"]], temperature=0.3)
    
    # Generate rankings
    print("   Generating rankings...")
    rows = []
    for rid, g in dir_df.groupby("race_id"):
        scores = {}
        for _, r in g.iterrows():
            h1, h2 = r["pair"]
            scores[h1] = scores.get(h1, 0) + r["p_pair"]
            scores[h2] = scores.get(h2, 0) + r["p_pair"]
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        race_df = df_future[df_future["race_id"] == rid].copy()
        
        # Get race metadata
        race_meta = race_df.iloc[0]
        
        for rank, (horse_id, score) in enumerate(ranked, 1):
            runner_row = race_df[race_df["horse_id"] == horse_id].iloc[0].to_dict()
            runner_row["score"] = score
            runner_row["rank"] = rank
            
            # Add confidence interpretation
            if score >= 0.17:
                runner_row["confidence"] = "Very High"
            elif score >= 0.15:
                runner_row["confidence"] = "High"
            elif score >= 0.13:
                runner_row["confidence"] = "Medium"
            else:
                runner_row["confidence"] = "Low"
            
            rows.append(runner_row)
    
    # Create output DataFrame
    out_df = pd.DataFrame(rows)
    
    # Format for CSV output (similar to predict_future.py)
    if "race_date" in out_df.columns:
        out_df["race_date"] = pd.to_datetime(out_df["race_date"]).dt.strftime('%Y-%m-%d')
    if "race_time" in out_df.columns:
        out_df["race_time"] = pd.to_datetime(out_df["race_time"], errors='coerce').apply(
            lambda x: x.strftime('%H:%M') if pd.notna(x) else ""
        )
    
    # Format numeric columns
    for col in ["score", "trainer_win30", "jockey_win30"]:
        if col in out_df.columns:
            out_df[col] = out_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
    
    # Save to CSV
    out_csv = PREDICTIONS_DIR / f"next_race_{date}_{datetime.now().strftime('%H%M')}.csv"
    out_df.to_csv(out_csv, index=False)
    
    print(f"   ✅ Predictions saved to {out_csv}")
    
    return out_csv


def save_predictions_to_db(predictions_csv):
    """
    Save predictions from CSV to the predictions table in the database.
    This allows tracking multiple predictions per race over time.
    """
    from src.horse_matcher import normalize_horse_name
    import re
    
    print("\n💾 Saving predictions to database...")
    
    # Read predictions CSV
    df = pd.read_csv(predictions_csv)
    
    # Extract prediction date from filename
    match = re.search(r'(\d{4}-\d{2}-\d{2})', str(predictions_csv))
    if not match:
        print("⚠️  Could not extract date from filename, skipping database save")
        return
    
    prediction_date = match.group(1)
    
    # Get model version from latest model file
    model_files = sorted(Path("data/models").glob(f"model_{prediction_date}_*.pkl"))
    model_version = model_files[-1].name if model_files else f"model_{prediction_date}"
    
    # Get current timestamp for this prediction run
    prediction_timestamp = datetime.now().isoformat()
    
    # Connect to database
    conn = get_connection()
    cursor = conn.cursor()
    placeholder = get_placeholder()
    
    # Normalize race names for matching
    def normalize_race_name(name):
        if not name or pd.isna(name):
            return ""
        return str(name).upper().strip().replace('  ', ' ')
    
    df['race_name_normalized'] = df['race_name'].apply(normalize_race_name)
    df['horse_normalized'] = df['horse'].apply(normalize_horse_name)
    
    # Parse score (handle percentage format)
    def parse_score(score):
        if pd.isna(score):
            return None
        score_str = str(score).strip()
        if score_str.endswith('%'):
            return float(score_str[:-1]) / 100.0
        return float(score_str)
    
    df['score_parsed'] = df['score'].apply(parse_score)
    
    # Insert into predictions table
    matched = 0
    unmatched = 0
    
    for _, row in df.iterrows():
        race_name_norm = row['race_name_normalized']
        horse_norm = row['horse_normalized']
        pred_rank = int(row['rank'])
        pred_score = row['score_parsed']
        win_odds = row.get('win_odds')  # Capture odds at prediction time
        
        # Find matching runner to get race_id and horse_id
        query = """
            SELECT run.race_id, run.horse_id, run.horse
            FROM runners run
            JOIN races r ON run.race_id = r.race_id
            WHERE r.date = ?
              AND UPPER(REPLACE(r.race_name, '  ', ' ')) = ?
        """
        
        cursor.execute(query, (prediction_date, race_name_norm))
        results = cursor.fetchall()
        
        # Filter by normalized horse name
        matching_results = []
        for result in results:
            db_horse_norm = normalize_horse_name(result[2])
            if db_horse_norm == horse_norm:
                matching_results.append(result)
        
        if len(matching_results) == 1:
            race_id, horse_id, _ = matching_results[0]
            
            # Insert prediction into predictions table
            insert_query = """
                INSERT INTO predictions
                (race_id, horse_id, predicted_rank, predicted_score, 
                 prediction_timestamp, model_version, win_odds_at_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_query, (
                race_id, horse_id, pred_rank, pred_score,
                prediction_timestamp, model_version, win_odds
            ))
            matched += 1
            
            # Also update runners table for backward compatibility
            update_query = """
                UPDATE runners
                SET predicted_rank = ?,
                    predicted_score = ?,
                    prediction_date = ?,
                    model_version = ?
                WHERE race_id = ? AND horse_id = ?
            """
            cursor.execute(update_query, (
                pred_rank, pred_score, prediction_date, model_version,
                race_id, horse_id
            ))
        else:
            unmatched += 1
    
    conn.commit()
    conn.close()
    
    print(f"✅ Saved {matched} predictions to database (predictions table)")
    print(f"   Prediction timestamp: {prediction_timestamp}")
    if unmatched > 0:
        print(f"⚠️  {unmatched} predictions could not be matched")


def display_predictions(date, race_id):
    """
    Display predictions in a clean, readable format for a specific race.
    
    Args:
        date: Race date in YYYY-MM-DD format
        race_id: The specific race ID to display
    """
    # Query the database for predictions for this specific race
    conn = get_connection()
    
    query = """
        SELECT 
            r.race_name,
            r.class,
            r.distance,
            r.post_time,
            run.horse,
            run.win_odds,
            run.predicted_score,
            run.predicted_rank,
            run.draw,
            run.jockey,
            run.trainer,
            run.status
        FROM runners run
        JOIN races r ON run.race_id = r.race_id
        WHERE r.race_id = ?
          AND run.predicted_rank IS NOT NULL
        ORDER BY run.predicted_rank
    """
    
    df = pd.read_sql_query(query, conn, params=(race_id,))
    conn.close()
    
    if df.empty:
        print("\n⚠️ No predictions found in database. The prediction may have failed.")
        return
    
    # Filter to declared runners only
    if "status" in df.columns:
        declared = df[df["status"].str.lower() == "declared"]
        if len(declared) < len(df):
            print(f"\n⚠️ Note: {len(df) - len(declared)} non-declared runners excluded from display")
        df = declared
    
    # Display the predictions
    print("\n" + "=" * 80)
    print("🏆 NEXT RACE PREDICTIONS")
    print("=" * 80)
    
    race_name = df.iloc[0]["race_name"]
    race_class = df.iloc[0]["class"]
    distance = df.iloc[0]["distance"]
    post_time = df.iloc[0]["post_time"]
    
    print(f"\nRace: {race_name}")
    print(f"Class: {race_class} | Distance: {distance}m | Post Time: {post_time}")
    print(f"\nTop Selections (ranked by model confidence):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Horse':<25} {'Odds':<8} {'Score':<10} {'Draw':<6} {'Jockey':<20}")
    print("-" * 80)
    
    for _, row in df.head(10).iterrows():
        rank = int(row["predicted_rank"])
        horse = row["horse"][:24]  # Truncate long names
        odds = f"{row['win_odds']:.1f}" if pd.notna(row["win_odds"]) else "N/A"
        score = f"{row['predicted_score']*100:.2f}%" if pd.notna(row["predicted_score"]) else "N/A"
        draw = int(row["draw"]) if pd.notna(row["draw"]) else "?"
        jockey = row["jockey"][:19] if pd.notna(row["jockey"]) else "Unknown"
        
        print(f"{rank:<6} {horse:<25} {odds:<8} {score:<10} {draw:<6} {jockey:<20}")
    
    print("-" * 80)
    print("\n💡 Tip: Higher scores indicate stronger model confidence")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Predict the next upcoming race with live odds"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save predictions to CSV file"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching new data, use existing JSON file"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="(Deprecated - always trains fresh to avoid pickle timeouts)"
    )
    args = parser.parse_args()
    
    # Always use fresh training (ignore pretrained flag to avoid pickle load timeouts)
    args.use_pretrained = False
    
    # Ensure directories exist
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Fetch latest racecards (unless skipped)
        if not args.skip_fetch:
            run_node_fetch()
        else:
            print("⏭️  Skipping fetch, using existing JSON file...")
        
        # Step 2: Find the next race
        latest_json = get_latest_json()
        next_race, race_date, meeting = find_next_race(latest_json)
        
        if next_race is None:
            print("\n✅ No upcoming races to predict.")
            return
        
        # Step 3: Import the race into database
        race_id = import_single_race(next_race, race_date, meeting)
        
        # Step 4: Run predictions
        csv_path = run_predictions(race_date, save_csv=True, use_pretrained=args.use_pretrained)
        
        if csv_path is None:
            print("\n❌ Prediction generation failed.")
            return 1
        
        # Step 5: Save predictions to database
        save_predictions_to_db(csv_path)
        
        # Step 6: Display predictions in a clean format
        display_predictions(race_date, race_id)
        
        if args.save_csv:
            print(f"\n📄 Predictions saved to: {csv_path}")
        else:
            # Clean up temporary CSV if user didn't request it
            csv_path.unlink()
            print(f"\n🗑️  Temporary CSV cleaned up")
        
        print("\n✅ Done! Predictions are ready.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

