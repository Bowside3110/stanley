#!/usr/bin/env python3
"""
wait_for_db.py

Wait for database to be ready before starting services.
This prevents startup failures when the database is still initializing.

Usage:
    python scripts/wait_for_db.py
    
Exit codes:
    0 - Database is ready
    1 - Database connection failed after max retries
"""

import time
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_config import get_connection

# Configuration
MAX_RETRIES = 30
RETRY_INTERVAL = 2  # seconds

def main():
    """Main entry point"""
    print("=" * 80)
    print("🔍 Waiting for database to be ready...")
    print("=" * 80)
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Try to connect to database
            conn = get_connection()
            
            # Test that we can actually query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            # Close connection
            conn.close()
            
            print(f"✅ Database connection successful on attempt {attempt}")
            print("=" * 80)
            return 0
            
        except Exception as e:
            print(f"⏳ Attempt {attempt}/{MAX_RETRIES} - Database not ready yet: {e}")
            
            if attempt < MAX_RETRIES:
                print(f"   Retrying in {RETRY_INTERVAL} seconds...")
                time.sleep(RETRY_INTERVAL)
            else:
                print("=" * 80)
                print("❌ Database connection failed after maximum retries")
                print("=" * 80)
                return 1
    
    return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

