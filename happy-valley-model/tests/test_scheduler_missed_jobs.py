#!/usr/bin/env python3
"""
Test scheduler behavior with missed/past jobs.
Verifies that the scheduler correctly skips past races and cleans up stale jobs.
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.scheduler import parse_hkt_time, load_and_schedule_all_races
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

# Test constants
TEST_DB = "data/test/test_scheduler.db"
TEST_SCHEDULER_DB = "data/test/test_scheduler_jobs.db"
BRISBANE = ZoneInfo("Australia/Brisbane")


def setup_test_db():
    """Create test database with past and future races"""
    Path("data/test").mkdir(parents=True, exist_ok=True)
    
    # Remove existing test databases
    for db_path in [TEST_DB, TEST_SCHEDULER_DB]:
        if Path(db_path).exists():
            Path(db_path).unlink()
    
    conn = sqlite3.connect(TEST_DB)
    cur = conn.cursor()
    
    # Create races table
    cur.execute("""
        CREATE TABLE races (
            race_id TEXT PRIMARY KEY,
            date TEXT,
            course TEXT,
            race_name TEXT,
            class TEXT,
            distance INTEGER,
            post_time TEXT
        )
    """)
    
    # Get current time
    now = datetime.now(BRISBANE)
    
    # Add past races (should be skipped)
    past_race_time = now - timedelta(hours=2)
    past_race_hkt = past_race_time.astimezone(ZoneInfo("Asia/Hong_Kong"))
    
    cur.execute("""
        INSERT INTO races VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        "HV_2025-12-03_R1",
        "2025-12-03",
        "HV",
        "PAST RACE HANDICAP",
        "Class 3",
        1200,
        past_race_hkt.isoformat()
    ))
    
    # Add future races (should be scheduled)
    future_race_time = now + timedelta(hours=2)
    future_race_hkt = future_race_time.astimezone(ZoneInfo("Asia/Hong_Kong"))
    
    cur.execute("""
        INSERT INTO races VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        "HV_2025-12-04_R1",
        "2025-12-04",
        "HV",
        "FUTURE RACE HANDICAP",
        "Class 3",
        1200,
        future_race_hkt.isoformat()
    ))
    
    # Add race that's too soon (within 5 min buffer - should be skipped)
    soon_race_time = now + timedelta(minutes=3)
    soon_race_hkt = soon_race_time.astimezone(ZoneInfo("Asia/Hong_Kong"))
    
    cur.execute("""
        INSERT INTO races VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        "HV_2025-12-03_R2",
        "2025-12-03",
        "HV",
        "TOO SOON RACE HANDICAP",
        "Class 3",
        1200,
        soon_race_hkt.isoformat()
    ))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Created test database with 3 races:")
    print(f"   - Past race: {past_race_time.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"   - Too soon race: {soon_race_time.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"   - Future race: {future_race_time.strftime('%Y-%m-%d %H:%M %Z')}")


def test_scheduler_skips_past_races():
    """Test that scheduler correctly skips past races"""
    print("\n" + "=" * 80)
    print("🧪 Testing scheduler with past/future races")
    print("=" * 80)
    
    # Setup test database
    setup_test_db()
    
    # Create scheduler with test database
    jobstores = {
        'default': SQLAlchemyJobStore(url=f'sqlite:///{TEST_SCHEDULER_DB}')
    }
    
    scheduler = BackgroundScheduler(
        jobstores=jobstores,
        timezone=BRISBANE
    )
    
    # Temporarily patch DB_PATH for testing
    import scripts.scheduler as sched_module
    original_db_path = sched_module.DB_PATH
    sched_module.DB_PATH = TEST_DB
    
    try:
        # Load and schedule races
        load_and_schedule_all_races(scheduler)
        
        # Check scheduled jobs
        jobs = scheduler.get_jobs()
        
        print("\n📋 Scheduled Jobs:")
        print("-" * 80)
        
        meeting_jobs = [j for j in jobs if j.id.startswith('meeting_')]
        race_jobs = [j for j in jobs if j.id.startswith('race_')]
        
        print(f"Meeting jobs: {len(meeting_jobs)}")
        print(f"Race jobs: {len(race_jobs)}")
        
        for job in jobs:
            print(f"  {job.id}")
        
        # Assertions
        assert len(meeting_jobs) == 1, f"Expected 1 meeting job, got {len(meeting_jobs)}"
        assert len(race_jobs) == 1, f"Expected 1 race job (only future race), got {len(race_jobs)}"
        
        # Verify the correct race was scheduled
        race_job = race_jobs[0]
        assert "2025-12-04" in race_job.id, f"Expected future race job, got {race_job.id}"
        
        print("\n✅ Test passed: Scheduler correctly skipped past races")
        print(f"   - Scheduled: 1 future race")
        print(f"   - Skipped: 2 past/too-soon races")
        
    finally:
        # Restore original DB_PATH
        sched_module.DB_PATH = original_db_path
        
        # Cleanup
        try:
            scheduler.shutdown(wait=False)
        except:
            pass  # Scheduler wasn't started, that's fine
        
        for db_path in [TEST_DB, TEST_SCHEDULER_DB]:
            if Path(db_path).exists():
                Path(db_path).unlink()
        
        print("\n🧹 Cleaned up test databases")


def test_parse_hkt_time():
    """Test HKT time parsing"""
    print("\n" + "=" * 80)
    print("🧪 Testing HKT time parsing")
    print("=" * 80)
    
    # Test ISO 8601 format with timezone
    time_str = "2025-12-03T21:10:00+08:00"
    result = parse_hkt_time(time_str)
    
    print(f"Input: {time_str}")
    print(f"Output: {result.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Timezone: {result.tzinfo}")
    
    # Verify it's in Brisbane timezone
    assert result.tzinfo.key == "Australia/Brisbane", f"Expected Brisbane timezone, got {result.tzinfo}"
    
    # Verify time conversion (HKT is UTC+8, Brisbane is UTC+10)
    # 21:10 HKT should be 23:10 Brisbane
    assert result.hour == 23, f"Expected hour 23, got {result.hour}"
    assert result.minute == 10, f"Expected minute 10, got {result.minute}"
    
    print("✅ Test passed: HKT time correctly converted to Brisbane time")


if __name__ == "__main__":
    test_parse_hkt_time()
    test_scheduler_skips_past_races()
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)

