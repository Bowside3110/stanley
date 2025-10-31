#!/usr/bin/env python3
"""
One-time migration script to copy position data from horse_results to runners table.

This script:
- Joins horse_results and runners tables by matching on horse_id and race_id
- Updates runners.position with values from horse_results.position where it's currently NULL
- Shows progress (how many rows updated)
- Is idempotent (safe to run multiple times)
"""

import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

DB_PATH = "data/historical/hkjc.db"

def backup_database(db_path: str) -> str:
    """Create a backup of the database before migration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    
    print(f"📦 Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    print(f"✅ Backup created successfully")
    
    return backup_path

def check_current_state(conn: sqlite3.Connection):
    """Check the current state of position data."""
    print("\n🔍 Checking current state...")
    
    # Check runners table
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total_runners,
            COUNT(position) as with_position,
            COUNT(*) - COUNT(position) as without_position
        FROM runners
    """)
    runners_stats = cursor.fetchone()
    
    # Check horse_results table
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total_results,
            COUNT(position) as with_position
        FROM horse_results
    """)
    results_stats = cursor.fetchone()
    
    # Check potential matches
    cursor = conn.execute("""
        SELECT COUNT(*)
        FROM runners r
        JOIN horse_results hr ON r.horse_id = hr.horse_id AND r.race_id = hr.race_id
        WHERE r.position IS NULL AND hr.position IS NOT NULL
    """)
    potential_updates = cursor.fetchone()[0]
    
    print(f"\n📊 Current State:")
    print(f"   Runners table:")
    print(f"      Total rows: {runners_stats[0]:,}")
    print(f"      With position: {runners_stats[1]:,}")
    print(f"      Without position: {runners_stats[2]:,}")
    print(f"\n   Horse_results table:")
    print(f"      Total rows: {results_stats[0]:,}")
    print(f"      With position: {results_stats[1]:,}")
    print(f"\n   Potential updates: {potential_updates:,} rows can be updated")
    
    return potential_updates

def migrate_positions(conn: sqlite3.Connection, dry_run: bool = False):
    """Migrate position data from horse_results to runners."""
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made")
    
    # Update query
    update_query = """
        UPDATE runners
        SET position = (
            SELECT hr.position 
            FROM horse_results hr
            WHERE hr.horse_id = runners.horse_id 
              AND hr.race_id = runners.race_id
        )
        WHERE runners.position IS NULL
          AND EXISTS (
              SELECT 1 
              FROM horse_results hr 
              WHERE hr.horse_id = runners.horse_id 
                AND hr.race_id = runners.race_id
                AND hr.position IS NOT NULL
          )
    """
    
    if not dry_run:
        print("\n🔄 Running migration...")
        cursor = conn.execute(update_query)
        rows_updated = cursor.rowcount
        conn.commit()
        print(f"✅ Migration complete! Updated {rows_updated:,} rows")
    else:
        print("\n✅ Dry run complete - no changes made")

def verify_migration(conn: sqlite3.Connection):
    """Verify the migration was successful."""
    print("\n🔍 Verifying migration...")
    
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total_runners,
            COUNT(position) as with_position,
            COUNT(*) - COUNT(position) as without_position
        FROM runners
    """)
    stats = cursor.fetchone()
    
    print(f"\n📊 After Migration:")
    print(f"   Total rows: {stats[0]:,}")
    print(f"   With position: {stats[1]:,}")
    print(f"   Without position: {stats[2]:,}")
    
    # Check for any remaining NULL positions that could have been filled
    cursor = conn.execute("""
        SELECT COUNT(*)
        FROM runners r
        JOIN horse_results hr ON r.horse_id = hr.horse_id AND r.race_id = hr.race_id
        WHERE r.position IS NULL AND hr.position IS NOT NULL
    """)
    remaining = cursor.fetchone()[0]
    
    if remaining > 0:
        print(f"\n⚠️  Warning: {remaining:,} rows still have NULL position but could be updated")
    else:
        print(f"\n✅ All available positions have been migrated!")

def main():
    """Main migration function."""
    print("=" * 60)
    print("Position Data Migration Script")
    print("=" * 60)
    
    # Check if database exists
    if not Path(DB_PATH).exists():
        print(f"❌ Error: Database not found at {DB_PATH}")
        return
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Check current state
        potential_updates = check_current_state(conn)
        
        if potential_updates == 0:
            print("\n✅ No updates needed - all positions are already migrated!")
            return
        
        # Ask for confirmation
        print(f"\n⚠️  This will update {potential_updates:,} rows in the runners table")
        response = input("Do you want to proceed? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("❌ Migration cancelled")
            return
        
        # Create backup
        backup_path = backup_database(DB_PATH)
        
        # Run migration
        migrate_positions(conn, dry_run=False)
        
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

