#!/usr/bin/env python3
"""
Backfill btn (beaten by) and time data from horse_results table to runners table.

This script migrates historical race result data (btn and time) from the horse_results
table to the runners table, which is now the single source of truth for all race data.
"""

import sqlite3
import sys

DB_PATH = "data/historical/hkjc.db"

def backfill_btn_time():
    """Migrate btn and time data from horse_results to runners table."""
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    print("=" * 80)
    print("BACKFILLING BTN AND TIME DATA")
    print("=" * 80)
    
    # Check current state
    print("\n1. Checking current data availability...")
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(btn) as has_btn,
            COUNT(time) as has_time
        FROM runners
    """)
    before = cur.fetchone()
    print(f"   Runners table: {before[0]} total, {before[1]} with btn, {before[2]} with time")
    
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(btn) as has_btn,
            COUNT(time) as has_time
        FROM horse_results
    """)
    source = cur.fetchone()
    print(f"   Horse_results table: {source[0]} total, {source[1]} with btn, {source[2]} with time")
    
    # Find matching records that need backfill
    print("\n2. Finding records to backfill...")
    cur.execute("""
        SELECT COUNT(*)
        FROM runners run
        JOIN horse_results hr 
            ON run.horse_id = hr.horse_id 
            AND run.race_id = hr.race_id
        WHERE run.btn IS NULL 
            AND hr.btn IS NOT NULL
    """)
    to_backfill = cur.fetchone()[0]
    print(f"   Found {to_backfill} records that can be backfilled")
    
    if to_backfill == 0:
        print("\n✅ No records need backfilling. All data is up to date!")
        conn.close()
        return
    
    # Perform the backfill
    print("\n3. Backfilling data...")
    cur.execute("""
        UPDATE runners
        SET 
            btn = (
                SELECT hr.btn 
                FROM horse_results hr 
                WHERE hr.horse_id = runners.horse_id 
                    AND hr.race_id = runners.race_id
            ),
            time = (
                SELECT hr.time 
                FROM horse_results hr 
                WHERE hr.horse_id = runners.horse_id 
                    AND hr.race_id = runners.race_id
            )
        WHERE EXISTS (
            SELECT 1 
            FROM horse_results hr 
            WHERE hr.horse_id = runners.horse_id 
                AND hr.race_id = runners.race_id
                AND (hr.btn IS NOT NULL OR hr.time IS NOT NULL)
        )
    """)
    
    updated = cur.rowcount
    conn.commit()
    print(f"   ✅ Updated {updated} records")
    
    # Verify the results
    print("\n4. Verifying results...")
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(btn) as has_btn,
            COUNT(time) as has_time
        FROM runners
    """)
    after = cur.fetchone()
    print(f"   Runners table: {after[0]} total, {after[1]} with btn, {after[2]} with time")
    
    btn_added = after[1] - before[1]
    time_added = after[2] - before[2]
    print(f"\n   Added {btn_added} btn values and {time_added} time values")
    
    # Check coverage by date
    print("\n5. Data coverage by recent dates:")
    cur.execute("""
        SELECT 
            DATE(r.date) as race_date,
            COUNT(*) as total_runners,
            COUNT(run.btn) as has_btn,
            COUNT(run.time) as has_time,
            ROUND(100.0 * COUNT(run.btn) / COUNT(*), 1) as btn_pct
        FROM runners run
        JOIN races r ON run.race_id = r.race_id
        WHERE r.date >= '2025-09-01'
        GROUP BY DATE(r.date)
        ORDER BY race_date DESC
        LIMIT 10
    """)
    
    print(f"\n   {'Date':<12} {'Runners':>8} {'Has BTN':>8} {'Has Time':>9} {'BTN %':>7}")
    print("   " + "-" * 58)
    for row in cur.fetchall():
        print(f"   {row[0]:<12} {row[1]:>8} {row[2]:>8} {row[3]:>9} {row[4]:>6}%")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("✅ BACKFILL COMPLETE")
    print("=" * 80)
    print("\nNote: Races after 2025-09-07 will still have no btn/time data")
    print("because the HKJC API doesn't provide this information.")
    print("These fields are only available in the historical horse_results table.")
    print("=" * 80)

if __name__ == "__main__":
    try:
        backfill_btn_time()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

