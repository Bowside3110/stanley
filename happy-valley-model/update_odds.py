#!/usr/bin/env python3
"""
Update odds for upcoming races and regenerate predictions.

This script:
1. Re-fetches live odds from the HKJC API
2. Updates the database with fresh odds
3. Automatically re-runs predictions with updated odds

Usage:
    python update_odds.py                    # Updates today's races
    python update_odds.py --date 2025-11-02  # Updates specific date
    python update_odds.py --show-changes     # Shows odds changes before/after
"""

import subprocess
import sqlite3
import json
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

DB_PATH = "data/historical/hkjc.db"
PREDICTIONS_DIR = Path("data/predictions")

def get_current_odds(race_date, db_path=DB_PATH):
    """Get current odds from database for comparison."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT 
            r.race_id,
            r.race_name,
            run.horse,
            run.win_odds as old_odds
        FROM races r
        JOIN runners run ON r.race_id = run.race_id
        WHERE r.date = ?
        AND run.status = 'declared'
        ORDER BY r.race_id, run.horse
    """
    df = pd.read_sql_query(query, conn, params=(race_date,))
    conn.close()
    return df

def fetch_live_odds(race_date):
    """Fetch live odds from HKJC API using Node.js script."""
    print(f"🔄 Fetching live odds for {race_date}...")
    
    # Run the Node.js fetcher
    result = subprocess.run(
        ["node", "fetch_next_meeting.mjs"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ Failed to fetch odds from HKJC API:")
        print(result.stderr)
        raise RuntimeError("Odds fetch failed")
    
    print(result.stdout)
    
    # Find the latest JSON file
    files = sorted(PREDICTIONS_DIR.glob("races_*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No race JSON files found after fetch")
    
    return files[-1]

def update_odds_in_db(json_path, db_path=DB_PATH):
    """Update win_odds in the database from the JSON file."""
    print(f"📥 Updating odds in database from {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "raceMeetings" in data:
        meetings = data["raceMeetings"]
    else:
        raise ValueError("Unexpected JSON structure: no 'raceMeetings' found")
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    updates_count = 0
    missing_odds = 0
    
    for meeting in meetings:
        for race in meeting.get("races", []):
            race_id = race["id"]
            
            for runner in race.get("runners", []):
                horse_id = runner["id"]
                win_odds = runner.get("winOdds")
                
                if win_odds is not None:
                    # Update the odds for this runner
                    cur.execute("""
                        UPDATE runners
                        SET win_odds = ?
                        WHERE race_id = ? AND horse_id = ?
                    """, (win_odds, race_id, horse_id))
                    
                    if cur.rowcount > 0:
                        updates_count += 1
                else:
                    missing_odds += 1
    
    conn.commit()
    conn.close()
    
    print(f"✅ Updated odds for {updates_count} runners")
    if missing_odds > 0:
        print(f"⚠️  {missing_odds} runners still have no odds (may not be betting yet)")
    
    return updates_count

def show_odds_changes(old_df, race_date, db_path=DB_PATH):
    """Show which odds changed and by how much."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT 
            r.race_id,
            r.race_name,
            run.horse,
            run.win_odds as new_odds
        FROM races r
        JOIN runners run ON r.race_id = run.race_id
        WHERE r.date = ?
        AND run.status = 'declared'
        ORDER BY r.race_id, run.horse
    """
    new_df = pd.read_sql_query(query, conn, params=(race_date,))
    conn.close()
    
    # Merge old and new
    comparison = old_df.merge(
        new_df,
        on=["race_id", "race_name", "horse"],
        how="outer"
    )
    
    # Calculate changes
    comparison["odds_change"] = comparison["new_odds"] - comparison["old_odds"]
    comparison["pct_change"] = (comparison["odds_change"] / comparison["old_odds"] * 100).round(1)
    
    # Filter to significant changes (>5% or >0.5 odds points)
    significant = comparison[
        (comparison["odds_change"].abs() > 0.5) | 
        (comparison["pct_change"].abs() > 5)
    ].copy()
    
    if len(significant) == 0:
        print("\n✅ No significant odds changes detected")
        return
    
    print("\n" + "=" * 80)
    print("📊 SIGNIFICANT ODDS CHANGES")
    print("=" * 80)
    
    for race_name, group in significant.groupby("race_name", sort=False):
        print(f"\n{race_name}:")
        for _, row in group.iterrows():
            direction = "📈" if row["odds_change"] > 0 else "📉"
            print(f"  {direction} {row['horse']:30s} {row['old_odds']:5.1f} → {row['new_odds']:5.1f} "
                  f"({row['pct_change']:+.1f}%)")
    
    print("\n" + "=" * 80)
    print(f"Total horses with significant changes: {len(significant)}")
    print("=" * 80 + "\n")

def regenerate_predictions(race_date):
    """Re-run make_predictions.py to generate fresh predictions."""
    print(f"🔮 Regenerating predictions for {race_date}...")
    
    # Import and run the prediction function
    from make_predictions import run_predictions
    
    predictions_csv = run_predictions(race_date)
    
    print(f"\n✅ Fresh predictions saved to {predictions_csv}")
    return predictions_csv

def main():
    parser = argparse.ArgumentParser(
        description="Update odds and regenerate predictions before race time"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Race date (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--show-changes",
        action="store_true",
        help="Show odds changes before/after update"
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Only update odds, don't regenerate predictions"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"🎯 UPDATING ODDS FOR {args.date}")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # Step 1: Get current odds for comparison (if requested)
    old_odds = None
    if args.show_changes:
        old_odds = get_current_odds(args.date)
    
    # Step 2: Fetch live odds from HKJC API
    json_path = fetch_live_odds(args.date)
    
    # Step 3: Update database with new odds
    updates_count = update_odds_in_db(json_path)
    
    if updates_count == 0:
        print("⚠️  No odds were updated. Check if races are available for betting.")
        return
    
    # Step 4: Show changes (if requested)
    if args.show_changes and old_odds is not None:
        show_odds_changes(old_odds, args.date)
    
    # Step 5: Regenerate predictions with fresh odds
    if not args.skip_predictions:
        predictions_csv = regenerate_predictions(args.date)
        
        print("\n" + "=" * 80)
        print("✅ ODDS UPDATE COMPLETE")
        print("=" * 80)
        print(f"Updated: {updates_count} runners")
        print(f"Fresh predictions: {predictions_csv}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    else:
        print("\n✅ Odds updated (predictions skipped)")

if __name__ == "__main__":
    main()

