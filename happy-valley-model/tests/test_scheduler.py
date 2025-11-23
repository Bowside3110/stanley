#!/usr/bin/env python3
"""
test_scheduler.py

Test the scheduler components to ensure they work correctly.
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = "data/historical/hkjc.db"
BRISBANE = ZoneInfo("Australia/Brisbane")
HKT = ZoneInfo("Asia/Hong_Kong")


def test_fetch_race_times():
    """Test that fetch_race_times.py runs successfully"""
    print("\n" + "=" * 80)
    print("TEST 1: fetch_race_times.py")
    print("=" * 80)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/fetch_race_times.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    if result.returncode != 0:
        print("❌ FAILED:")
        print(result.stderr)
        return False
    
    print("✅ PASSED: fetch_race_times.py executed successfully")
    return True


def test_database_schema():
    """Test that races table has post_time column"""
    print("\n" + "=" * 80)
    print("TEST 2: Database Schema")
    print("=" * 80)
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Check if post_time column exists
    cur.execute("PRAGMA table_info(races)")
    columns = [row[1] for row in cur.fetchall()]
    
    if "post_time" not in columns:
        print("❌ FAILED: post_time column not found in races table")
        print(f"Available columns: {columns}")
        conn.close()
        return False
    
    print("✅ PASSED: post_time column exists in races table")
    conn.close()
    return True


def test_query_upcoming_races():
    """Test querying upcoming races from database"""
    print("\n" + "=" * 80)
    print("TEST 3: Query Upcoming Races")
    print("=" * 80)
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Get current time in HKT
    now_brisbane = datetime.now(BRISBANE)
    now_hkt = now_brisbane.astimezone(HKT)
    now_iso = now_hkt.isoformat()
    
    query = """
        SELECT race_id, date, course, race_name, post_time
        FROM races
        WHERE post_time > ?
        ORDER BY post_time
        LIMIT 5
    """
    
    cur.execute(query, (now_iso,))
    races = cur.fetchall()
    conn.close()
    
    if not races:
        print("⚠️  No upcoming races found (this is OK if no future races in DB)")
        return True
    
    print(f"✅ Found {len(races)} upcoming races:")
    for race_id, date, course, race_name, post_time in races:
        print(f"   {date} {course}: {race_name[:40]}")
        print(f"   Post time: {post_time}")
    
    print("✅ PASSED: Successfully queried upcoming races")
    return True


def test_timezone_parsing():
    """Test parsing HKT times and converting to Brisbane"""
    print("\n" + "=" * 80)
    print("TEST 4: Timezone Parsing")
    print("=" * 80)
    
    # Test ISO 8601 time parsing
    test_time = "2025-11-20T13:00:00+08:00"
    
    try:
        dt_hkt = datetime.fromisoformat(test_time)
        dt_brisbane = dt_hkt.astimezone(BRISBANE)
        
        print(f"HKT time: {dt_hkt.strftime('%Y-%m-%d %H:%M %Z')}")
        print(f"Brisbane time: {dt_brisbane.strftime('%Y-%m-%d %H:%M %Z')}")
        
        # Verify the conversion is correct (Brisbane is UTC+10, HKT is UTC+8)
        # So Brisbane should be 2 hours ahead
        time_diff = (dt_brisbane - dt_hkt).total_seconds() / 3600
        
        if abs(time_diff) < 0.01:  # Should be 0 (same instant in time)
            print("✅ PASSED: Timezone conversion correct")
            return True
        else:
            print(f"❌ FAILED: Time difference is {time_diff} hours (expected 0)")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_scheduler_imports():
    """Test that scheduler.py can be imported without errors"""
    print("\n" + "=" * 80)
    print("TEST 5: Scheduler Imports")
    print("=" * 80)
    
    try:
        # Try importing the scheduler module
        import importlib.util
        spec = importlib.util.spec_from_file_location("scheduler", "scripts/scheduler.py")
        scheduler_module = importlib.util.module_from_spec(spec)
        
        # Check if required functions exist
        spec.loader.exec_module(scheduler_module)
        
        required_functions = [
            'load_and_schedule_all_races',
            'schedule_meeting_predictions',
            'schedule_race_prediction',
            'run_meeting_predictions',
            'run_race_prediction',
            'refresh_schedule'
        ]
        
        for func_name in required_functions:
            if not hasattr(scheduler_module, func_name):
                print(f"❌ FAILED: Function {func_name} not found")
                return False
        
        print("✅ PASSED: All required functions exist in scheduler.py")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 STANLEY SCHEDULER TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_database_schema,
        test_timezone_parsing,
        test_scheduler_imports,
        test_fetch_race_times,
        test_query_upcoming_races,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"❌ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())


