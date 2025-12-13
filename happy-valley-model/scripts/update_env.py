#!/usr/bin/env python3
"""
Append PostgreSQL configuration to existing .env file
"""

import os
from pathlib import Path

ENV_FILE = Path("/Users/bendunn/Stanley/happy-valley-model/.env")

# PostgreSQL configuration to add
POSTGRES_CONFIG = """
# ============================================================================
# DATABASE CONFIGURATION (PostgreSQL Migration)
# ============================================================================
USE_POSTGRES=true
DATABASE_URL=postgresql://username:password@host:port/database?sslmode=require
"""

def update_env():
    """Update .env file with PostgreSQL settings"""
    if not ENV_FILE.exists():
        print(f"❌ .env file not found at {ENV_FILE}")
        return False
    
    # Read existing content
    with open(ENV_FILE, 'r') as f:
        content = f.read()
    
    # Check if USE_POSTGRES already exists
    if 'USE_POSTGRES' in content:
        print("✅ PostgreSQL settings already present in .env")
        print("\nTo change USE_POSTGRES, edit .env manually:")
        print("   USE_POSTGRES=true   # Use PostgreSQL")
        print("   USE_POSTGRES=false  # Use SQLite")
        return True
    
    # Append PostgreSQL configuration
    with open(ENV_FILE, 'a') as f:
        f.write(POSTGRES_CONFIG)
    
    print("✅ Successfully added PostgreSQL configuration to .env")
    print("\nAdded configuration:")
    print(POSTGRES_CONFIG)
    return True

if __name__ == "__main__":
    update_env()

