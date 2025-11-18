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
            post_time = race.get("postTime", "")

            # Insert race
            cur.execute("""
                INSERT OR REPLACE INTO races
                (race_id, date, course, race_name, class, distance, going, rail, post_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id, race_date, course, race_name, race_class,
                distance, going, rail, post_time
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

                # Capture runner status
                status = runner.get("status", "unknown")
                
                # Capture final position if race is completed
                # finalPosition = 0 means race not finished, convert to NULL
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
    print(f"✅ Imported {race_count} races and {runner_count} runners into database")

    if missing_odds > 0:
        print(f"⚠️ Warning: {missing_odds} runners are missing win odds. "
              f"Predictions will be made without odds data.")

def run_predictions(date):
    """Run your predict_future.py with the given date"""
    print(f"🎯 Running predictions for {date}")
    out_csv = PREDICTIONS_DIR / f"predictions_{date}.csv"
    
    # Run the prediction with feature importance flag
    result = subprocess.run([
        "python", "-m", "src.predict_future",
        "--db", str(DB_PATH),
        "--date", date,
        "--box", "5",
        "--save_csv", str(out_csv)
    ], capture_output=True, text=True)
    
    # Print the standard output from the prediction script
    print(result.stdout)
    
    # Check for any errors
    if result.returncode != 0:
        print("❌ Prediction failed:")
        print(result.stderr)
        
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
    for race_name, g in df.groupby("race_name", sort=False):
        print(f"\nRace: {race_name}")
        print("Ranked selections:")

        top_n = g.sort_values("rank").head(5)
        for _, row in top_n.iterrows():
            horse = row["horse"]
            score = row["score"]
            print(f" {row['rank']}. {horse} ({score})")
    
    return df  # Return the DataFrame for further analysis


def save_predictions_to_db(predictions_csv, db_path="data/historical/hkjc.db"):
    """
    Save predictions from CSV to the predictions table in the database.
    This allows tracking multiple predictions per race over time.
    
    Args:
        predictions_csv: Path to the predictions CSV file
        db_path: Path to the database file
    """
    from src.horse_matcher import normalize_horse_name
    from datetime import datetime
    import re
    
    print("\n" + "=" * 80)
    print("💾 Saving predictions to database...")
    
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
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
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
    print("=" * 80)


def analyze_feature_importance(predictions_df):
    """
    Analyze and display feature importance and odds-history metrics summary.
    
    Args:
        predictions_df: DataFrame with predictions
    """
    print("\n=== Feature Influence Summary ===")
    
    # Check if we have the odds-history metrics in the DataFrame
    odds_features = ["horse_odds_efficiency", "horse_odds_trend", "trainer_odds_bias"]
    present_odds_features = [f for f in odds_features if f in predictions_df.columns]
    
    if present_odds_features:
        # We have odds-history metrics in the predictions DataFrame
        print(f"Found {len(present_odds_features)}/{len(odds_features)} odds-history metrics in predictions")
        
        # Calculate summary statistics for the odds-history metrics
        print("\nOdds-history metrics summary:")
        for feature in present_odds_features:
            values = predictions_df[feature].dropna()
            if len(values) > 0:
                mean = values.mean()
                std = values.std()
                min_val = values.min()
                max_val = values.max()
                print(f"  {feature:25s}: mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}")
            else:
                print(f"  {feature:25s}: No valid values")
    else:
        # No odds-history metrics found
        print("No odds-history metrics found in predictions DataFrame")
        
    # Try to extract feature importances from a model dump file if it exists
    try:
        import pickle
        import os
        
        # Check if we have a recent model dump
        model_files = sorted(Path("data/models").glob("*.pkl"), key=os.path.getmtime, reverse=True)
        
        if model_files:
            # Load the most recent model
            model_path = model_files[0]
            print(f"\nLoading model from {model_path} to extract feature importances...")
            
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                
            # Check if the model has feature importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_names = getattr(model, "feature_names_in_", None)
                
                if feature_names is not None and len(feature_names) == len(importances):
                    # Create a DataFrame with feature importances
                    import numpy as np
                    feat_imp = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    # Print top 10 features
                    print("\nTop 10 features by importance:")
                    for i, (feature, importance) in enumerate(zip(feat_imp['feature'].head(10), 
                                                                feat_imp['importance'].head(10))):
                        print(f"  {i+1:2d}. {feature:30s}: {importance:.4f}")
                    
                    # Check for odds-history metrics
                    odds_importances = feat_imp[feat_imp['feature'].str.contains('|'.join(odds_features), regex=True)]
                    if not odds_importances.empty:
                        print("\nOdds-history metrics importances:")
                        for i, (feature, importance) in enumerate(zip(odds_importances['feature'], 
                                                                    odds_importances['importance'])):
                            print(f"  {i+1:2d}. {feature:30s}: {importance:.4f}")
                        
                        # Calculate total importance of odds features
                        total_imp = odds_importances['importance'].sum()
                        print(f"\nTotal importance of odds-history metrics: {total_imp:.4f} "
                              f"({total_imp/importances.sum()*100:.2f}%)")
                else:
                    print("Model has feature importances but no feature names available")
            else:
                print("Model doesn't expose feature_importances_ attribute")
        else:
            print("No model files found in data/models directory")
    except Exception as e:
        print(f"Error extracting feature importances from model: {e}")
        
    # If we couldn't get feature importances from a model, use the predictions DataFrame
    # to provide some insight into the odds-history metrics
    if not present_odds_features:
        print("\nFallback analysis: Checking for pair-level features in predictions...")
        pair_features = [col for col in predictions_df.columns if any(f in col for f in 
                         ["_min", "_max", "_diff"] + odds_features)]
        
        if pair_features:
            print(f"Found {len(pair_features)} potential pair-level features:")
            for feature in sorted(pair_features)[:10]:  # Show top 10
                print(f"  - {feature}")
        else:
            print("No pair-level features found in predictions DataFrame")

if __name__ == "__main__":
    # Ensure the models directory exists
    Path("data/models").mkdir(parents=True, exist_ok=True)
    
    run_node_fetch()
    latest_json = get_latest_json()
    import_races(latest_json)

    # Extract the date from filename e.g. races_2025-09-28_ST.json
    race_date = Path(latest_json).stem.split("_")[1]
    predictions_csv = run_predictions(race_date)
    
    # Save predictions to database
    save_predictions_to_db(predictions_csv)
    
    # Show the cheat sheet and get the DataFrame for feature importance analysis
    predictions_df = show_cheat_sheet(predictions_csv)
    
    # Analyze and display feature importance
    analyze_feature_importance(predictions_df)
