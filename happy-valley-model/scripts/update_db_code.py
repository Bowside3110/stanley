#!/usr/bin/env python3
"""
Comprehensive database migration script for PostgreSQL compatibility.

This script updates all Python source files to use the new db_config module
and PostgreSQL-compatible SQL syntax.
"""

import re
from pathlib import Path

# List of files that need updating
FILES_TO_UPDATE = [
    'src/predict_next_race.py',
    'src/update_odds.py',
    'src/analyze_predictions.py',
    'src/features.py',
    'src/model.py',
    'src/build_future_dataset.py',
]

# Track changes
changes_made = []

def update_file(file_path):
    """Update a single file with PostgreSQL-compatible changes."""
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️  Skipping {file_path} (not found)")
        return False
    
    print(f"\n📝 Processing {file_path}...")
    
    with open(path, 'r') as f:
        content = f.read()
    
    original_content = content
    file_changes = []
    
    # 1. Update imports
    if 'import sqlite3' in content and 'from src.db_config import' not in content:
        content = re.sub(
            r'import sqlite3',
            'import sqlite3  # Keep for legacy compatibility\nfrom src.db_config import get_connection, get_placeholder',
            content,
            count=1
        )
        file_changes.append("Added db_config imports")
    
    # 2. Replace all sqlite3.connect() calls with get_connection()
    old_connects = len(re.findall(r'sqlite3\.connect\([^)]+\)', content))
    content = re.sub(
        r'sqlite3\.connect\([^)]+\)',
        'get_connection()',
        content
    )
    if old_connects > 0:
        file_changes.append(f"Replaced {old_connects} sqlite3.connect() calls")
    
    # 3. Add placeholder variable after connection creation
    # Find patterns: conn = get_connection()
    # Add: placeholder = get_placeholder()
    def add_placeholder(match):
        conn_line = match.group(0)
        indent = match.group(1)
        # Check if placeholder already exists
        return conn_line + f'\n{indent}placeholder = get_placeholder()'
    
    # Match: conn = get_connection() followed by cursor creation
    pattern = r'([ \t]*)conn = get_connection\(\)\s*\n\1(?:cur(?:sor)? = conn\.cursor\(\))'
    if re.search(pattern, content):
        # Only add if not already present
        if 'placeholder = get_placeholder()' not in content:
            content = re.sub(
                pattern,
                lambda m: m.group(0) + f'\n{m.group(1)}placeholder = get_placeholder()',
                content
            )
            file_changes.append("Added placeholder variable initialization")
    
    # 4. Update datetime('now') to CURRENT_TIMESTAMP
    datetime_now_count = len(re.findall(r"datetime\('now'\)", content))
    content = re.sub(
        r"datetime\('now'\)",
        'CURRENT_TIMESTAMP',
        content
    )
    if datetime_now_count > 0:
        file_changes.append(f"Updated {datetime_now_count} datetime('now') to CURRENT_TIMESTAMP")
    
    # 5. Replace "INSERT OR REPLACE" with "INSERT ... ON CONFLICT" (manual review needed)
    # This is complex and needs manual intervention per case
    insert_or_replace_count = len(re.findall(r'INSERT OR REPLACE', content, re.IGNORECASE))
    if insert_or_replace_count > 0:
        file_changes.append(f"⚠️  Found {insert_or_replace_count} INSERT OR REPLACE statements (needs manual review)")
    
    # 6. Flag queries with ? placeholders for manual review
    queries_with_placeholders = len(re.findall(r'VALUES\s*\([^)]*\?', content))
    if queries_with_placeholders > 0:
        file_changes.append(f"⚠️  Found {queries_with_placeholders} queries with ? placeholders (needs manual review)")
    
    # Write changes if any were made
    if content != original_content:
        with open(path, 'w') as f:
            f.write(content)
        
        print(f"   ✅ Updated!")
        for change in file_changes:
            print(f"      • {change}")
        
        changes_made.append({
            'file': str(file_path),
            'changes': file_changes
        })
        return True
    else:
        print(f"   ℹ️  No changes needed")
        return False

def main():
    """Run migration on all files."""
    print("=" * 80)
    print("POSTGRESQL MIGRATION - CODE UPDATE SCRIPT")
    print("=" * 80)
    
    for file_path in FILES_TO_UPDATE:
        update_file(file_path)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✅ Updated {len(changes_made)} files")
    
    if changes_made:
        print("\nFiles changed:")
        for item in changes_made:
            print(f"\n📄 {item['file']}")
            for change in item['changes']:
                print(f"   • {change}")
    
    print("\n" + "=" * 80)
    print("⚠️  MANUAL STEPS STILL REQUIRED:")
    print("=" * 80)
    print("""
1. Review all files with INSERT OR REPLACE statements:
   - Change to: INSERT ... ON CONFLICT ... DO UPDATE SET

2. Review all files with ? placeholders:
   - Replace ? with {placeholder} using f-strings
   - Or use proper parameterized queries

3. Test each file individually:
   - Run: python scripts/test_db_connection.py
   - Then test: python -m src.make_predictions

4. Review special cases:
   - Row factory compatibility (dict vs tuple access)
   - Transaction handling
   - Error handling

Files that need the most attention:
   - src/predict_next_race.py (4 connection points)
   - src/analyze_predictions.py (4 connection points)
   - src/features.py (complex queries)

Recommendation:
   - Start with SQLite (USE_POSTGRES=false) to test
   - Fix any bugs introduced
   - Then switch to PostgreSQL (USE_POSTGRES=true)
   - Run schema creation: python scripts/create_postgres_schema.py
   - Test connection: python scripts/test_db_connection.py
""")

if __name__ == "__main__":
    main()

