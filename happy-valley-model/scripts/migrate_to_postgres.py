#!/usr/bin/env python3
"""
Migration script to update all database connections from SQLite to PostgreSQL-compatible code.

This script:
1. Updates import statements
2. Replaces sqlite3.connect() with get_connection()
3. Replaces ? placeholders with get_placeholder()
4. Updates INSERT OR REPLACE to INSERT ... ON CONFLICT
5. Updates datetime('now') to CURRENT_TIMESTAMP
"""

import re
import sys
from pathlib import Path

# Files to migrate
FILES_TO_MIGRATE = [
    "src/predict_next_race.py",
    "src/update_odds.py",
    "src/analyze_predictions.py",
    "src/features.py",
    "src/model.py",
    "src/build_future_dataset.py",
]

def add_imports(content):
    """Add db_config imports if not present"""
    if 'from src.db_config import' in content:
        return content
    
    # Find import sqlite3 line and add our imports after it
    if 'import sqlite3' in content:
        content = content.replace(
            'import sqlite3',
            'import sqlite3  # Keep for type hints\nfrom src.db_config import get_connection, get_placeholder'
        )
    return content

def replace_sqlite_connect(content):
    """Replace sqlite3.connect() with get_connection()"""
    # Pattern: conn = sqlite3.connect(...)
    content = re.sub(
        r'conn\s*=\s*sqlite3\.connect\([^)]+\)',
        'conn = get_connection()',
        content
    )
    return content

def add_placeholder_variable(content):
    """Add placeholder variable after get_connection()"""
    # Find patterns like: conn = get_connection()\n    cur = conn.cursor()
    # And add: placeholder = get_placeholder()
    pattern = r'(conn\s*=\s*get_connection\(\)\s*\n\s*)(cur(?:sor)?\s*=\s*conn\.cursor\(\))'
    replacement = r'\1\2\n    placeholder = get_placeholder()'
    
    content = re.sub(pattern, replacement, content)
    return content

def main():
    """Run migration on all specified files"""
    print("🔄 Starting PostgreSQL migration...")
    print()
    
    for file_path in FILES_TO_MIGRATE:
        full_path = Path(file_path)
        if not full_path.exists():
            print(f"⚠️  Skipping {file_path} (not found)")
            continue
        
        print(f"📝 Processing {file_path}...")
        
        # Read file
        with open(full_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply transformations
        content = add_imports(content)
        content = replace_sqlite_connect(content)
        content = add_placeholder_variable(content)
        
        # Write back if changed
        if content != original_content:
            with open(full_path, 'w') as f:
                f.write(content)
            print(f"   ✅ Updated {file_path}")
        else:
            print(f"   ℹ️  No changes needed for {file_path}")
    
    print()
    print("✅ Migration phase 1 complete!")
    print()
    print("⚠️  MANUAL STEPS REQUIRED:")
    print("   1. Update SQL queries to use placeholder variable")
    print("   2. Replace INSERT OR REPLACE with INSERT ... ON CONFLICT")
    print("   3. Replace datetime('now') with CURRENT_TIMESTAMP")
    print("   4. Test each file individually")

if __name__ == "__main__":
    main()

