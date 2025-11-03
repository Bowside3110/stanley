#!/usr/bin/env python3
"""
Backfill predictions from CSV files into the runners table.

This script reads prediction CSV files from data/predictions/ and updates
the runners table with predicted_rank, predicted_score, prediction_date,
and model_version for each horse.
"""

import sqlite3
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from src.horse_matcher import normalize_horse_name

def extract_date_from_filename(filename):
    """Extract date from filename like 'predictions_2025-11-02.csv'"""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return None

def normalize_race_name(name):
    """Normalize race name for matching"""
    if not name or pd.isna(name):
        return ""
    # Convert to uppercase and remove extra whitespace
    name = str(name).upper().strip()
    # Remove common variations
    name = re.sub(r'\s+', ' ', name)
    return name

def backfill_from_csv(csv_path, conn, dry_run=False):
    """
    Backfill predictions from a single CSV file.
    
    Returns: (matched, unmatched, updated)
    """
    print(f"\n📄 Processing: {csv_path.name}")
    
    # Extract prediction date from filename
    prediction_date = extract_date_from_filename(csv_path.name)
    if not prediction_date:
        print(f"   ⚠️  Could not extract date from filename, skipping")
        return 0, 0, 0
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"   ❌ Error reading CSV: {e}")
        return 0, 0, 0
    
    # Check required columns
    required_cols = ['race_name', 'horse', 'rank', 'score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   ⚠️  Missing required columns: {missing_cols}, skipping")
        return 0, 0, 0
    
    print(f"   Found {len(df)} predictions for {prediction_date}")
    
    # Normalize horse names
    df['horse_normalized'] = df['horse'].apply(normalize_horse_name)
    df['race_name_normalized'] = df['race_name'].apply(normalize_race_name)
    
    # Parse score (handle percentage format like "17.21%")
    def parse_score(score):
        if pd.isna(score):
            return None
        score_str = str(score).strip()
        if score_str.endswith('%'):
            return float(score_str[:-1]) / 100.0
        return float(score_str)
    
    df['score_parsed'] = df['score'].apply(parse_score)
    
    # Model version (use filename for now)
    model_version = f"predictions_{prediction_date}"
    
    # Match and update runners
    cursor = conn.cursor()
    matched = 0
    unmatched = 0
    updated = 0
    
    for _, row in df.iterrows():
        race_name_norm = row['race_name_normalized']
        horse_norm = row['horse_normalized']
        pred_rank = int(row['rank'])
        pred_score = row['score_parsed']
        
        if not race_name_norm or not horse_norm:
            unmatched += 1
            continue
        
        # Try to find matching runner in database
        # Match by: race date, race name (normalized), and horse name
        # Note: We normalize horse names in Python since runners table doesn't have normalized column
        query = """
            SELECT run.rowid, run.horse, r.race_name
            FROM runners run
            JOIN races r ON run.race_id = r.race_id
            WHERE r.date = ?
              AND UPPER(REPLACE(r.race_name, '  ', ' ')) = ?
        """
        
        cursor.execute(query, (prediction_date, race_name_norm))
        results = cursor.fetchall()
        
        # Filter results by normalized horse name in Python
        matching_results = []
        for result in results:
            db_horse_norm = normalize_horse_name(result[1])
            if db_horse_norm == horse_norm:
                matching_results.append(result)
        
        results = matching_results
        
        if len(results) == 0:
            # Try fuzzy match on race name (sometimes has extra spaces or variations)
            query_fuzzy = """
                SELECT run.rowid, run.horse, r.race_name
                FROM runners run
                JOIN races r ON run.race_id = r.race_id
                WHERE r.date = ?
            """
            cursor.execute(query_fuzzy, (prediction_date,))
            all_results = cursor.fetchall()
            
            # Filter by normalized horse name
            for result in all_results:
                db_horse_norm = normalize_horse_name(result[1])
                if db_horse_norm == horse_norm:
                    results.append(result)
            
            if len(results) == 0:
                unmatched += 1
                if unmatched <= 3:  # Only show first few
                    print(f"   ⚠️  No match: {row['horse']} in {row['race_name']}")
                continue
            elif len(results) > 1:
                # Multiple horses with same name on same day - try to match by race name
                matched_by_race = [r for r in results if normalize_race_name(r[2]) == race_name_norm]
                if len(matched_by_race) == 1:
                    results = matched_by_race
                else:
                    unmatched += 1
                    continue
        
        matched += 1
        rowid = results[0][0]
        
        # Update the runner with prediction data
        if not dry_run:
            update_query = """
                UPDATE runners
                SET predicted_rank = ?,
                    predicted_score = ?,
                    prediction_date = ?,
                    model_version = ?
                WHERE rowid = ?
            """
            cursor.execute(update_query, (pred_rank, pred_score, prediction_date, model_version, rowid))
            updated += 1
    
    if not dry_run:
        conn.commit()
    
    print(f"   ✅ Matched: {matched}, Unmatched: {unmatched}, Updated: {updated}")
    return matched, unmatched, updated

def main():
    """Main backfill process"""
    print("=" * 80)
    print("STEP 2: Backfilling predictions from CSV files")
    print("=" * 80)
    
    # Connect to database
    db_path = Path("data/historical/hkjc.db")
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    # Find all prediction CSV files
    predictions_dir = Path("data/predictions")
    csv_files = sorted(predictions_dir.glob("predictions_*.csv"))
    
    # Filter out test files and version files
    csv_files = [f for f in csv_files if 'test' not in f.name.lower() and ' v' not in f.name]
    
    print(f"\n📂 Found {len(csv_files)} prediction CSV files")
    for f in csv_files:
        print(f"   • {f.name}")
    
    # Process each CSV
    total_matched = 0
    total_unmatched = 0
    total_updated = 0
    
    for csv_file in csv_files:
        matched, unmatched, updated = backfill_from_csv(csv_file, conn, dry_run=False)
        total_matched += matched
        total_unmatched += unmatched
        total_updated += updated
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ BACKFILL COMPLETE")
    print("=" * 80)
    print(f"Total matched: {total_matched}")
    print(f"Total unmatched: {total_unmatched}")
    print(f"Total updated: {total_updated}")
    
    # Show sample of updated data
    print("\n📊 Sample of updated predictions:")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.date, r.race_name, run.horse, run.position, 
               run.predicted_rank, run.predicted_score, run.prediction_date
        FROM runners run
        JOIN races r ON run.race_id = r.race_id
        WHERE run.predicted_rank IS NOT NULL
        ORDER BY r.date DESC, run.predicted_rank
        LIMIT 10
    """)
    
    print(f"\n{'Date':<12} {'Race':<30} {'Horse':<20} {'Pos':<5} {'Pred':<5} {'Score':<7}")
    print("-" * 95)
    for row in cursor.fetchall():
        date, race, horse, pos, pred_rank, pred_score, pred_date = row
        race_short = race[:28] + ".." if len(race) > 30 else race
        horse_short = horse[:18] + ".." if len(horse) > 20 else horse
        pos_str = str(pos) if pos else "-"
        score_str = f"{pred_score:.4f}" if pred_score else "-"
        print(f"{date:<12} {race_short:<30} {horse_short:<20} {pos_str:<5} {pred_rank:<5} {score_str:<7}")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("Next: Step 3 - Update make_predictions.py to save predictions going forward")
    print("=" * 80)

if __name__ == "__main__":
    main()

