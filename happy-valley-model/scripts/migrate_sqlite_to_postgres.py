#!/usr/bin/env python3
"""
SQLite to PostgreSQL Data Migration Script

Migrates all data from SQLite (hkjc.db) to DigitalOcean PostgreSQL database.
Handles data type conversions, NULL values, and provides progress reporting.

Usage:
    python scripts/migrate_sqlite_to_postgres.py
    python scripts/migrate_sqlite_to_postgres.py --force  # Override safety checks
    python scripts/migrate_sqlite_to_postgres.py --tables races runners  # Migrate specific tables
"""

import sqlite3
import psycopg2
from psycopg2.extras import execute_batch
import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuration
SQLITE_DB = 'data/historical/hkjc.db'
POSTGRES_URL = os.getenv('DATABASE_URL')
BATCH_SIZE = 1000

# Define table schemas (columns to migrate)
TABLE_SCHEMAS = {
    'races': ['race_id', 'date', 'course', 'race_name', 'class', 'distance', 'going', 'rail', 'post_time'],
    'runners': ['race_id', 'horse_id', 'horse', 'draw', 'weight', 'jockey', 'jockey_id', 
                'trainer', 'trainer_id', 'win_odds', 'position', 'status', 'btn', 'time', 
                'starting_price', 'predicted_rank', 'predicted_score', 'prediction_date', 'model_version'],
    'results': ['race_id', 'horse_id', 'position'],
    'backfill_log': ['date', 'processed_at', 'year'],
    'jockey_results': ['jockey_id', 'race_id', 'date', 'position'],
    'trainer_results': ['trainer_id', 'race_id', 'date', 'position'],
    'horse_results': ['horse_id', 'race_id', 'date', 'position', 'class', 'course', 'going', 
                      'dist_m', 'draw', 'weight', 'weight_lbs', 'sp_dec', 'btn', 'time', 
                      'off_dt', 'surface'],
    'racecard_pro': ['race_id', 'date', 'course', 'race_name', 'race_class', 'age_band', 
                     'rating_band', 'pattern', 'going', 'surface', 'dist', 'dist_m', 'rail', 
                     'off_time', 'off_dt', 'updated_utc'],
    'racecard_pro_runners': ['race_id', 'horse_id', 'horse', 'number', 'draw', 'weight', 
                             'weight_lbs', 'headgear', 'headgear_run', 'wind_surgery', 
                             'wind_surgery_run', 'last_run', 'form', 'jockey', 'jockey_id', 
                             'trainer', 'trainer_id', 'win_odds', 'updated_utc'],
    'predictions': ['race_id', 'horse_id', 'predicted_rank', 'predicted_score', 
                    'prediction_timestamp', 'model_version', 'win_odds_at_prediction']
}

# Migration order (respects foreign key constraints)
MIGRATION_ORDER = [
    'races',
    'runners', 
    'results',
    'backfill_log',
    'jockey_results',
    'trainer_results',
    'horse_results',
    'racecard_pro',
    'racecard_pro_runners',
    'predictions'
]


def clean_value(value):
    """
    Clean value for PostgreSQL compatibility.
    Converts empty strings to None, handles type conversions.
    """
    if value == '':
        return None
    if value is None:
        return None
    return value


def check_prerequisites():
    """Check that all prerequisites are met before migration"""
    print("🔍 Checking prerequisites...")
    
    # Check SQLite database exists
    if not os.path.exists(SQLITE_DB):
        print(f"❌ Error: SQLite database not found at {SQLITE_DB}")
        return False
    
    print(f"   ✓ SQLite database found: {SQLITE_DB}")
    
    # Check PostgreSQL URL is configured
    if not POSTGRES_URL:
        print("❌ Error: DATABASE_URL environment variable not set")
        print("   Please add DATABASE_URL to your .env file")
        return False
    
    print(f"   ✓ PostgreSQL URL configured")
    
    # Test PostgreSQL connection
    try:
        pg_conn = psycopg2.connect(POSTGRES_URL)
        pg_cursor = pg_conn.cursor()
        pg_cursor.execute("SELECT version();")
        version = pg_cursor.fetchone()[0]
        print(f"   ✓ PostgreSQL connection successful")
        print(f"     Version: {version.split(',')[0]}")
        pg_conn.close()
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False
    
    return True


def get_table_count(conn, table_name, is_postgres=False):
    """Get row count for a table"""
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    cursor.close()
    return count


def table_exists(conn, table_name, is_postgres=False):
    """Check if table exists in database"""
    cursor = conn.cursor()
    
    if is_postgres:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """, (table_name,))
    else:
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?;
        """, (table_name,))
    
    result = cursor.fetchone()
    cursor.close()
    
    if is_postgres:
        return result[0] if result else False
    else:
        return result is not None


def migrate_table(table_name, columns, sqlite_conn, pg_conn, force=False):
    """
    Migrate a single table from SQLite to PostgreSQL
    
    Args:
        table_name: Name of the table to migrate
        columns: List of column names to migrate
        sqlite_conn: SQLite connection
        pg_conn: PostgreSQL connection
        force: Skip safety checks if True
    """
    print(f"\n{'='*60}")
    print(f"Migrating: {table_name}")
    print(f"{'='*60}")
    
    # Check if source table exists
    if not table_exists(sqlite_conn, table_name, is_postgres=False):
        print(f"⚠️  Skipping {table_name} - table doesn't exist in SQLite")
        return False
    
    # Check if target table exists
    if not table_exists(pg_conn, table_name, is_postgres=True):
        print(f"⚠️  Skipping {table_name} - table doesn't exist in PostgreSQL")
        print(f"   Run: python scripts/create_postgres_schema.py")
        return False
    
    # Get source row count
    sqlite_count = get_table_count(sqlite_conn, table_name, is_postgres=False)
    print(f"📊 Source rows (SQLite): {sqlite_count:,}")
    
    if sqlite_count == 0:
        print(f"⚠️  No data to migrate in {table_name}")
        return True
    
    # Check target table status
    pg_count = get_table_count(pg_conn, table_name, is_postgres=True)
    print(f"📊 Target rows (PostgreSQL): {pg_count:,}")
    
    if pg_count > 0 and not force:
        print(f"⚠️  WARNING: Target table already has {pg_count:,} rows")
        response = input(f"   Continue? This will add {sqlite_count:,} more rows. (y/N): ")
        if response.lower() != 'y':
            print(f"   Skipping {table_name}")
            return False
    
    # Start migration
    start_time = datetime.now()
    
    sqlite_cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()
    
    # For predictions table, don't include prediction_id (it's SERIAL in PostgreSQL)
    if table_name == 'predictions':
        # Read all columns from SQLite first
        sqlite_cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
        all_columns = [desc[0] for desc in sqlite_cursor.description]
        # Filter out prediction_id if it exists
        columns = [col for col in columns if col != 'prediction_id']
    
    # Build SQL queries
    select_sql = f"SELECT {', '.join(columns)} FROM {table_name}"
    placeholders = ', '.join(['%s'] * len(columns))
    insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    
    # Migrate in batches
    migrated = 0
    skipped = 0
    batch = []
    
    print(f"🔄 Migrating data...")
    
    for row in sqlite_cursor.execute(select_sql):
        # Clean values for PostgreSQL
        cleaned_row = tuple(clean_value(v) for v in row)
        
        # Skip rows with NULL in first column if it's a primary key table
        # (backfill_log, jockey_results, trainer_results, horse_results have composite keys)
        if cleaned_row[0] is None:
            skipped += 1
            continue
        
        batch.append(cleaned_row)
        
        # Insert batch when full
        if len(batch) >= BATCH_SIZE:
            execute_batch(pg_cursor, insert_sql, batch)
            pg_conn.commit()
            migrated += len(batch)
            batch = []
            
            # Progress indicator
            percent = ((migrated + skipped) / sqlite_count) * 100
            print(f"   Progress: {migrated:,}/{sqlite_count:,} ({percent:.1f}%)", end='\r')
    
    # Insert remaining rows
    if batch:
        execute_batch(pg_cursor, insert_sql, batch)
        pg_conn.commit()
        migrated += len(batch)
    
    print(f"   Progress: {migrated:,}/{sqlite_count:,} (100.0%)  ")
    
    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped:,} rows with NULL primary keys")
    
    # Verify migration
    pg_count_after = get_table_count(pg_conn, table_name, is_postgres=True)
    expected_count = pg_count + migrated  # Use migrated, not sqlite_count
    
    elapsed = datetime.now() - start_time
    print(f"⏱️  Time: {elapsed.total_seconds():.2f} seconds")
    print(f"📊 Final row count: {pg_count_after:,}")
    
    if pg_count_after == expected_count:
        print(f"✅ {table_name} migration successful!")
        return True
    else:
        print(f"⚠️  Warning: Expected {expected_count:,} rows, got {pg_count_after:,}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Migrate data from SQLite to PostgreSQL',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--force', action='store_true',
                       help='Force migration even if tables have data')
    parser.add_argument('--tables', nargs='+',
                       help='Specific tables to migrate (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be migrated without actually migrating')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stanley SQLite → PostgreSQL Data Migration")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Determine which tables to migrate
    if args.tables:
        tables_to_migrate = [t for t in args.tables if t in TABLE_SCHEMAS]
        if not tables_to_migrate:
            print(f"❌ Error: None of the specified tables are valid")
            print(f"   Valid tables: {', '.join(TABLE_SCHEMAS.keys())}")
            sys.exit(1)
    else:
        tables_to_migrate = [t for t in MIGRATION_ORDER if t in TABLE_SCHEMAS]
    
    print(f"\n📋 Tables to migrate: {', '.join(tables_to_migrate)}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No data will be migrated")
        
        # Show what would be migrated
        sqlite_conn = sqlite3.connect(SQLITE_DB)
        pg_conn = psycopg2.connect(POSTGRES_URL)
        
        print("\n" + "=" * 60)
        for table_name in tables_to_migrate:
            if table_exists(sqlite_conn, table_name, is_postgres=False):
                sqlite_count = get_table_count(sqlite_conn, table_name, is_postgres=False)
                pg_count = get_table_count(pg_conn, table_name, is_postgres=True)
                print(f"\n{table_name}:")
                print(f"  SQLite:     {sqlite_count:,} rows")
                print(f"  PostgreSQL: {pg_count:,} rows")
                print(f"  Would add:  {sqlite_count:,} rows")
        
        sqlite_conn.close()
        pg_conn.close()
        print("\n" + "=" * 60)
        return
    
    # Connect to both databases
    print(f"\n🔌 Connecting to databases...")
    sqlite_conn = sqlite3.connect(SQLITE_DB)
    pg_conn = psycopg2.connect(POSTGRES_URL)
    print(f"   ✓ Connected to both databases")
    
    # Migrate tables
    start_time = datetime.now()
    successful = []
    failed = []
    skipped = []
    
    for table_name in tables_to_migrate:
        columns = TABLE_SCHEMAS[table_name]
        try:
            result = migrate_table(table_name, columns, sqlite_conn, pg_conn, force=args.force)
            if result:
                successful.append(table_name)
            else:
                skipped.append(table_name)
        except Exception as e:
            print(f"❌ Error migrating {table_name}: {e}")
            failed.append(table_name)
    
    # Close connections
    sqlite_conn.close()
    pg_conn.close()
    
    # Summary
    elapsed = datetime.now() - start_time
    
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"⏱️  Total time: {elapsed.total_seconds():.2f} seconds")
    print(f"✅ Successful: {len(successful)} tables")
    if successful:
        for table in successful:
            print(f"   • {table}")
    
    if skipped:
        print(f"\n⚠️  Skipped: {len(skipped)} tables")
        for table in skipped:
            print(f"   • {table}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} tables")
        for table in failed:
            print(f"   • {table}")
    
    print("\n" + "=" * 60)
    
    if failed:
        print("❌ Migration completed with errors")
        sys.exit(1)
    else:
        print("✅ Migration Complete!")
        print("\n🔍 Next steps:")
        print("   1. Verify data: python scripts/verify_migration.py")
        print("   2. Test connection: python scripts/test_db_connection.py")
        print("   3. Update .env: Set USE_POSTGRES=true")


if __name__ == "__main__":
    main()

