#!/usr/bin/env python3
"""
Enhanced migration script to consolidate runner-specific data into the runners table.

This script:
- Adds new columns to runners table (btn, time, starting_price)
- Backfills ALL runner-specific data from horse_results
- Updates position, draw, weight where NULL in runners
- Shows detailed statistics and verification
- Is idempotent (safe to run multiple times)
"""

import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd

DB_PATH = "data/historical/hkjc.db"

def backup_database(db_path: str) -> str:
    """Create a backup of the database before migration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    
    print(f"📦 Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    print(f"✅ Backup created successfully")
    
    return backup_path

def check_schema(conn: sqlite3.Connection):
    """Check current schema and add missing columns if needed."""
    print("\n🔍 Checking schema...")
    
    # Get current columns
    cursor = conn.execute("PRAGMA table_info(runners)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    
    print(f"   Current columns: {', '.join(columns.keys())}")
    
    # Columns to add
    new_columns = {
        'btn': 'REAL',
        'time': 'TEXT',
        'starting_price': 'REAL'
    }
    
    added = []
    for col_name, col_type in new_columns.items():
        if col_name not in columns:
            print(f"   Adding column: {col_name} ({col_type})")
            conn.execute(f"ALTER TABLE runners ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
    
    if added:
        conn.commit()
        print(f"✅ Added {len(added)} new columns: {', '.join(added)}")
    else:
        print(f"✅ All columns already exist")
    
    return added

def check_current_state(conn: sqlite3.Connection):
    """Check the current state of data in both tables."""
    print("\n🔍 Checking current state...")
    
    # Check runners table
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(position) as has_position,
            COUNT(draw) as has_draw,
            COUNT(weight) as has_weight,
            COUNT(btn) as has_btn,
            COUNT(time) as has_time,
            COUNT(starting_price) as has_sp
        FROM runners
    """)
    runners_stats = cursor.fetchone()
    
    # Check horse_results table
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(position) as has_position,
            COUNT(draw) as has_draw,
            COUNT(weight) as has_weight,
            COUNT(btn) as has_btn,
            COUNT(time) as has_time,
            COUNT(sp_dec) as has_sp
        FROM horse_results
    """)
    results_stats = cursor.fetchone()
    
    # Check potential matches
    cursor = conn.execute("""
        SELECT COUNT(*)
        FROM runners r
        JOIN races rac ON r.race_id = rac.race_id
        JOIN horse_results hr ON r.horse_id = hr.horse_id AND rac.date = hr.date
    """)
    matchable = cursor.fetchone()[0]
    
    print(f"\n📊 Current State:")
    print(f"\n   Runners table ({runners_stats[0]:,} rows):")
    print(f"      position: {runners_stats[1]:,} ({runners_stats[1]/runners_stats[0]*100:.1f}%)")
    print(f"      draw: {runners_stats[2]:,} ({runners_stats[2]/runners_stats[0]*100:.1f}%)")
    print(f"      weight: {runners_stats[3]:,} ({runners_stats[3]/runners_stats[0]*100:.1f}%)")
    print(f"      btn: {runners_stats[4]:,} ({runners_stats[4]/runners_stats[0]*100:.1f}%)")
    print(f"      time: {runners_stats[5]:,} ({runners_stats[5]/runners_stats[0]*100:.1f}%)")
    print(f"      starting_price: {runners_stats[6]:,} ({runners_stats[6]/runners_stats[0]*100:.1f}%)")
    
    print(f"\n   Horse_results table ({results_stats[0]:,} rows):")
    print(f"      position: {results_stats[1]:,} ({results_stats[1]/results_stats[0]*100:.1f}%)")
    print(f"      draw: {results_stats[2]:,} ({results_stats[2]/results_stats[0]*100:.1f}%)")
    print(f"      weight: {results_stats[3]:,} ({results_stats[3]/results_stats[0]*100:.1f}%)")
    print(f"      btn: {results_stats[4]:,} ({results_stats[4]/results_stats[0]*100:.1f}%)")
    print(f"      time: {results_stats[5]:,} ({results_stats[5]/results_stats[0]*100:.1f}%)")
    print(f"      sp_dec: {results_stats[6]:,} ({results_stats[6]/results_stats[0]*100:.1f}%)")
    
    print(f"\n   Matchable rows: {matchable:,} (can be joined by horse_id + date)")
    
    return matchable

def show_sample_data(conn: sqlite3.Connection, before: bool = True):
    """Show sample data for verification."""
    label = "BEFORE" if before else "AFTER"
    print(f"\n📋 Sample Data ({label} migration):")
    
    query = """
        SELECT 
            r.race_id,
            r.horse_id,
            r.horse,
            r.position as r_pos,
            r.draw as r_draw,
            r.weight as r_weight,
            r.btn as r_btn,
            r.time as r_time,
            r.starting_price as r_sp,
            hr.position as hr_pos,
            hr.draw as hr_draw,
            hr.weight as hr_weight,
            hr.btn as hr_btn,
            hr.time as hr_time,
            hr.sp_dec as hr_sp
        FROM runners r
        JOIN races rac ON r.race_id = rac.race_id
        JOIN horse_results hr ON r.horse_id = hr.horse_id AND rac.date = hr.date
        LIMIT 5
    """
    
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

def migrate_data(conn: sqlite3.Connection, dry_run: bool = False):
    """Migrate all runner-specific data from horse_results to runners."""
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made")
        return
    
    print("\n🔄 Running migration...")
    
    # Update query - backfill all fields
    update_query = """
        UPDATE runners
        SET 
            position = COALESCE(runners.position, (
                SELECT hr.position 
                FROM horse_results hr
                JOIN races rac ON runners.race_id = rac.race_id
                WHERE hr.horse_id = runners.horse_id AND hr.date = rac.date
            )),
            draw = COALESCE(runners.draw, (
                SELECT hr.draw 
                FROM horse_results hr
                JOIN races rac ON runners.race_id = rac.race_id
                WHERE hr.horse_id = runners.horse_id AND hr.date = rac.date
            )),
            weight = COALESCE(runners.weight, (
                SELECT hr.weight 
                FROM horse_results hr
                JOIN races rac ON runners.race_id = rac.race_id
                WHERE hr.horse_id = runners.horse_id AND hr.date = rac.date
            )),
            btn = COALESCE(runners.btn, (
                SELECT hr.btn 
                FROM horse_results hr
                JOIN races rac ON runners.race_id = rac.race_id
                WHERE hr.horse_id = runners.horse_id AND hr.date = rac.date
            )),
            time = COALESCE(runners.time, (
                SELECT hr.time 
                FROM horse_results hr
                JOIN races rac ON runners.race_id = rac.race_id
                WHERE hr.horse_id = runners.horse_id AND hr.date = rac.date
            )),
            starting_price = COALESCE(runners.starting_price, (
                SELECT hr.sp_dec 
                FROM horse_results hr
                JOIN races rac ON runners.race_id = rac.race_id
                WHERE hr.horse_id = runners.horse_id AND hr.date = rac.date
            ))
        WHERE EXISTS (
            SELECT 1 
            FROM horse_results hr
            JOIN races rac ON runners.race_id = rac.race_id
            WHERE hr.horse_id = runners.horse_id AND hr.date = rac.date
        )
    """
    
    cursor = conn.execute(update_query)
    rows_updated = cursor.rowcount
    conn.commit()
    
    print(f"✅ Migration complete! Updated {rows_updated:,} rows")

def verify_migration(conn: sqlite3.Connection):
    """Verify the migration was successful."""
    print("\n🔍 Verifying migration...")
    
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(position) as has_position,
            COUNT(draw) as has_draw,
            COUNT(weight) as has_weight,
            COUNT(btn) as has_btn,
            COUNT(time) as has_time,
            COUNT(starting_price) as has_sp
        FROM runners
    """)
    stats = cursor.fetchone()
    
    print(f"\n📊 After Migration:")
    print(f"   Total rows: {stats[0]:,}")
    print(f"   position: {stats[1]:,} ({stats[1]/stats[0]*100:.1f}%)")
    print(f"   draw: {stats[2]:,} ({stats[2]/stats[0]*100:.1f}%)")
    print(f"   weight: {stats[3]:,} ({stats[3]/stats[0]*100:.1f}%)")
    print(f"   btn: {stats[4]:,} ({stats[4]/stats[0]*100:.1f}%)")
    print(f"   time: {stats[5]:,} ({stats[5]/stats[0]*100:.1f}%)")
    print(f"   starting_price: {stats[6]:,} ({stats[6]/stats[0]*100:.1f}%)")
    
    # Check for any remaining NULL values that could have been filled
    cursor = conn.execute("""
        SELECT COUNT(*)
        FROM runners r
        JOIN races rac ON r.race_id = rac.race_id
        JOIN horse_results hr ON r.horse_id = hr.horse_id AND rac.date = hr.date
        WHERE (r.position IS NULL AND hr.position IS NOT NULL)
           OR (r.btn IS NULL AND hr.btn IS NOT NULL)
           OR (r.time IS NULL AND hr.time IS NOT NULL)
           OR (r.starting_price IS NULL AND hr.sp_dec IS NOT NULL)
    """)
    remaining = cursor.fetchone()[0]
    
    if remaining > 0:
        print(f"\n⚠️  Warning: {remaining:,} rows still have NULL values but could be updated")
    else:
        print(f"\n✅ All available data has been migrated!")

def main():
    """Main migration function."""
    print("=" * 60)
    print("Runner Data Migration Script")
    print("=" * 60)
    
    # Check if database exists
    if not Path(DB_PATH).exists():
        print(f"❌ Error: Database not found at {DB_PATH}")
        return
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Check and update schema
        added_columns = check_schema(conn)
        
        # Check current state
        matchable = check_current_state(conn)
        
        if matchable == 0:
            print("\n⚠️  No matchable rows found between runners and horse_results!")
            return
        
        # Show sample data before migration
        show_sample_data(conn, before=True)
        
        # Ask for confirmation
        print(f"\n⚠️  This will update runner data for up to {matchable:,} rows")
        response = input("Do you want to proceed? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("❌ Migration cancelled")
            return
        
        # Create backup
        backup_path = backup_database(DB_PATH)
        
        # Run migration
        migrate_data(conn, dry_run=False)
        
        # Show sample data after migration
        show_sample_data(conn, before=False)
        
        # Verify results
        verify_migration(conn)
        
        print(f"\n✅ Migration completed successfully!")
        print(f"   Backup saved at: {backup_path}")
        
    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()

