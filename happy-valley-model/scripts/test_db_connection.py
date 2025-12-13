#!/usr/bin/env python3
"""
Test database connection for both SQLite and PostgreSQL.

This script verifies that the database connection is working correctly
and displays information about the connected database.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.db_config import get_connection, get_db_type, USE_POSTGRES

def test_connection():
    """Test database connection"""
    print("=" * 80)
    print("DATABASE CONNECTION TEST")
    print("=" * 80)
    print()
    
    # Show configuration
    db_type = get_db_type()
    print(f"📊 Database Type: {db_type.upper()}")
    print(f"   USE_POSTGRES: {USE_POSTGRES}")
    
    if USE_POSTGRES:
        database_url = os.getenv('DATABASE_URL', '')
        if database_url:
            # Hide password in display
            url_parts = database_url.split('@')
            if len(url_parts) > 1:
                host_part = url_parts[1]
                print(f"   Host: {host_part.split('/')[0]}")
        else:
            print("   ❌ DATABASE_URL not set!")
            return False
    else:
        print(f"   Path: data/historical/hkjc.db")
    
    print()
    print("🔄 Testing connection...")
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Test query based on database type
        if USE_POSTGRES:
            cursor.execute("SELECT version();")
            result = cursor.fetchone()
            print("✅ PostgreSQL connection successful!")
            print(f"   Version: {result['version'][:80]}...")
            print()
            
            # Get table count
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            result = cursor.fetchone()
            table_count = result['count']
            
            # List tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = cursor.fetchall()
            
            print(f"📋 Found {table_count} tables:")
            for table in tables:
                print(f"   • {table['table_name']}")
            
        else:
            cursor.execute("SELECT sqlite_version();")
            result = cursor.fetchone()
            print("✅ SQLite connection successful!")
            print(f"   Version: {result[0]}")
            print()
            
            # Get table count
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            table_count = cursor.fetchone()[0]
            
            # List tables
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = cursor.fetchall()
            
            print(f"📋 Found {table_count} tables:")
            for table in tables:
                print(f"   • {table[0]}")
        
        print()
        
        # Test a sample query on races table if it exists
        try:
            cursor.execute("SELECT COUNT(*) FROM races")
            race_count = cursor.fetchone()
            if USE_POSTGRES:
                race_count = race_count['count']
            else:
                race_count = race_count[0]
            print(f"🏇 Races table: {race_count} races")
        except Exception:
            print("⚠️  Races table not found or empty")
        
        try:
            cursor.execute("SELECT COUNT(*) FROM runners")
            runner_count = cursor.fetchone()
            if USE_POSTGRES:
                runner_count = runner_count['count']
            else:
                runner_count = runner_count[0]
            print(f"🐴 Runners table: {runner_count} runners")
        except Exception:
            print("⚠️  Runners table not found or empty")
        
        try:
            cursor.execute("SELECT COUNT(*) FROM predictions")
            pred_count = cursor.fetchone()
            if USE_POSTGRES:
                pred_count = pred_count['count']
            else:
                pred_count = pred_count[0]
            print(f"🎯 Predictions table: {pred_count} predictions")
        except Exception:
            print("⚠️  Predictions table not found or empty")
        
        cursor.close()
        conn.close()
        
        print()
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check that .env file exists and has correct settings")
        print("  2. Verify DATABASE_URL is correct (if using PostgreSQL)")
        print("  3. Ensure PostgreSQL server is accessible")
        print("  4. Run: pip install psycopg2-binary (if using PostgreSQL)")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)

