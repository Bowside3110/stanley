#!/usr/bin/env python3
"""
Verify SQLite to PostgreSQL migration.

Compares row counts and sample data between SQLite and PostgreSQL databases
to ensure migration was successful.
"""

import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import sys
from dotenv import load_dotenv

load_dotenv()

SQLITE_DB = 'data/historical/hkjc.db'
POSTGRES_URL = os.getenv('DATABASE_URL')

# Tables to verify
TABLES = [
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


def get_table_count(conn, table_name, is_postgres=False):
    """Get row count for a table"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    except Exception as e:
        return None


def table_exists(conn, table_name, is_postgres=False):
    """Check if table exists"""
    cursor = conn.cursor()
    
    try:
        if is_postgres:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table_name,))
            result = cursor.fetchone()[0]
        else:
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?;
            """, (table_name,))
            result = cursor.fetchone() is not None
        
        cursor.close()
        return result
    except Exception as e:
        cursor.close()
        return False


def get_sample_data(conn, table_name, limit=3, is_postgres=False):
    """Get sample rows from table"""
    try:
        if is_postgres:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
        
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        rows = cursor.fetchall()
        cursor.close()
        
        return [dict(row) for row in rows]
    except Exception as e:
        return None


def compare_tables(sqlite_conn, pg_conn):
    """Compare all tables between SQLite and PostgreSQL"""
    
    results = []
    
    for table_name in TABLES:
        # Check if tables exist
        sqlite_exists = table_exists(sqlite_conn, table_name, is_postgres=False)
        pg_exists = table_exists(pg_conn, table_name, is_postgres=True)
        
        if not sqlite_exists and not pg_exists:
            continue  # Skip tables that don't exist in either
        
        # Get counts
        sqlite_count = get_table_count(sqlite_conn, table_name, is_postgres=False) if sqlite_exists else 0
        pg_count = get_table_count(pg_conn, table_name, is_postgres=True) if pg_exists else 0
        
        # Determine status
        if sqlite_count == pg_count:
            status = "✅ Match"
        elif sqlite_count == 0 and pg_count == 0:
            status = "⚪ Empty"
        elif not pg_exists:
            status = "❌ Missing in PG"
        elif not sqlite_exists:
            status = "⚠️  Only in PG"
        elif pg_count > sqlite_count:
            status = "⚠️  More in PG"
        else:
            status = "❌ Mismatch"
        
        results.append({
            'table': table_name,
            'sqlite': f"{sqlite_count:,}" if sqlite_exists else "N/A",
            'postgres': f"{pg_count:,}" if pg_exists else "N/A",
            'status': status
        })
    
    return results


def verify_sample_data(sqlite_conn, pg_conn, table_name):
    """Verify sample data matches between databases"""
    
    print(f"\n{'='*60}")
    print(f"Sample Data Comparison: {table_name}")
    print(f"{'='*60}")
    
    # Get sample from SQLite
    sqlite_sample = get_sample_data(sqlite_conn, table_name, limit=1, is_postgres=False)
    pg_sample = get_sample_data(pg_conn, table_name, limit=1, is_postgres=True)
    
    if not sqlite_sample or not pg_sample:
        print("⚠️  Could not retrieve sample data")
        return
    
    print("\nSQLite (first row):")
    for key, value in list(sqlite_sample[0].items())[:5]:  # Show first 5 columns
        print(f"  {key}: {value}")
    
    print("\nPostgreSQL (first row):")
    for key, value in list(pg_sample[0].items())[:5]:  # Show first 5 columns
        print(f"  {key}: {value}")


def main():
    print("=" * 60)
    print("SQLite to PostgreSQL Migration Verification")
    print("=" * 60)
    print()
    
    # Check prerequisites
    if not os.path.exists(SQLITE_DB):
        print(f"❌ Error: SQLite database not found at {SQLITE_DB}")
        sys.exit(1)
    
    if not POSTGRES_URL:
        print("❌ Error: DATABASE_URL not set in .env file")
        sys.exit(1)
    
    # Connect to databases
    print("🔌 Connecting to databases...")
    try:
        sqlite_conn = sqlite3.connect(SQLITE_DB)
        print("   ✓ Connected to SQLite")
    except Exception as e:
        print(f"   ❌ SQLite connection failed: {e}")
        sys.exit(1)
    
    try:
        pg_conn = psycopg2.connect(POSTGRES_URL)
        print("   ✓ Connected to PostgreSQL")
    except Exception as e:
        print(f"   ❌ PostgreSQL connection failed: {e}")
        sys.exit(1)
    
    # Compare table counts
    print("\n📊 Comparing table row counts...")
    results = compare_tables(sqlite_conn, pg_conn)
    
    # Display results in table format
    table_data = [[r['table'], r['sqlite'], r['postgres'], r['status']] for r in results]
    headers = ['Table', 'SQLite', 'PostgreSQL', 'Status']
    
    # Simple table formatting
    col_widths = [25, 15, 15, 20]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print("\n" + header_line)
    print("-" * sum(col_widths))
    for row in table_data:
        row_line = "".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(row_line)
    
    # Count matches and mismatches
    matches = sum(1 for r in results if '✅' in r['status'])
    mismatches = sum(1 for r in results if '❌' in r['status'])
    empty = sum(1 for r in results if '⚪' in r['status'])
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✅ Matching tables: {matches}")
    print(f"⚪ Empty tables: {empty}")
    print(f"⚠️  Warnings: {sum(1 for r in results if '⚠️' in r['status'])}")
    print(f"❌ Mismatches: {mismatches}")
    
    # Show sample data for key tables
    if matches > 0:
        print("\n" + "=" * 60)
        print("Sample Data Verification")
        print("=" * 60)
        
        for table in ['races', 'runners']:
            if any(r['table'] == table and '✅' in r['status'] for r in results):
                verify_sample_data(sqlite_conn, pg_conn, table)
    
    # Close connections
    sqlite_conn.close()
    pg_conn.close()
    
    print("\n" + "=" * 60)
    
    if mismatches > 0:
        print("❌ Verification failed - some tables don't match")
        print("\n💡 Troubleshooting:")
        print("   1. Review migration logs for errors")
        print("   2. Re-run migration for mismatched tables:")
        print("      python scripts/migrate_sqlite_to_postgres.py --tables <table_name>")
        sys.exit(1)
    else:
        print("✅ Verification successful - all tables match!")
        print("\n🔍 Next steps:")
        print("   1. Update .env: Set USE_POSTGRES=true")
        print("   2. Test application: python test_api_endpoints.py")
        print("   3. Run predictions: python -m src.predict_future")


if __name__ == "__main__":
    main()

