#!/usr/bin/env python3
"""
scheduler.py

Main scheduling daemon for Stanley racing predictions.
Schedules meeting predictions (30min before first race) and individual race predictions (2min before each race).

Usage:
    python scripts/scheduler.py
"""

import os
import sys
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
ALERT_EMAIL = "adamsalistair1978@gmail.com"
ALERT_PHONE = "+61417676973"
DB_PATH = "data/historical/hkjc.db"
SCHEDULER_DB_PATH = "data/historical/scheduler_jobs.db"

# Timezone definitions
HKT = ZoneInfo("Asia/Hong_Kong")  # UTC+8
BRISBANE = ZoneInfo("Australia/Brisbane")  # UTC+10 (AEST) or UTC+11 (AEDT)

# Global scheduler instance for refresh job
_scheduler = None


def parse_hkt_time(time_str: str) -> datetime:
    """
    Parse a time string from HKJC API and return a timezone-aware datetime in Brisbane time.
    
    Args:
        time_str: Time in ISO 8601 format (e.g., "2025-11-09T13:00:00+08:00")
    
    Returns:
        Timezone-aware datetime in Brisbane timezone
    """
    # Parse ISO 8601 format with timezone
    dt_hkt = datetime.fromisoformat(time_str)
    # Convert to Brisbane time
    return dt_hkt.astimezone(BRISBANE)


def load_and_schedule_all_races(scheduler):
    """
    Query database for all upcoming races and schedule all jobs.
    
    Args:
        scheduler: APScheduler instance
    """
    print("\n" + "=" * 80)
    print("📅 Loading upcoming race schedule from database...")
    print("=" * 80)
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Query all races where post_time > NOW()
    # Get current time in HKT for comparison
    now_brisbane = datetime.now(BRISBANE)
    now_hkt = now_brisbane.astimezone(HKT)
    now_iso = now_hkt.isoformat()
    
    query = """
        SELECT race_id, date, course, race_name, class, distance, post_time
        FROM races
        WHERE post_time > ?
        ORDER BY post_time
    """
    
    cur.execute(query, (now_iso,))
    races = cur.fetchall()
    conn.close()
    
    if not races:
        print("⚠️  No upcoming races found in database.")
        print("   Run 'python scripts/fetch_race_times.py' to fetch the latest schedule.")
        return
    
    print(f"\n✅ Found {len(races)} upcoming races")
    
    # Group races by meeting date to schedule meeting predictions
    meetings = {}
    for race_id, date, course, race_name, race_class, distance, post_time in races:
        if date not in meetings:
            meetings[date] = []
        meetings[date].append({
            'race_id': race_id,
            'course': course,
            'race_name': race_name,
            'class': race_class,
            'distance': distance,
            'post_time': post_time
        })
    
    # Schedule meeting predictions (30min before first race)
    print("\n📋 Scheduling meeting predictions:")
    print("-" * 80)
    for meeting_date, meeting_races in meetings.items():
        # Sort races by post_time to find first race
        meeting_races.sort(key=lambda x: x['post_time'])
        first_race = meeting_races[0]
        first_race_time = parse_hkt_time(first_race['post_time'])
        
        # Schedule 30 minutes before first race
        meeting_job_time = first_race_time - timedelta(minutes=30)
        
        # Only schedule if in the future (with 5 min buffer to avoid edge cases)
        if meeting_job_time > now_brisbane + timedelta(minutes=5):
            job_id = f"meeting_{meeting_date}"
            
            # Remove existing job if it exists
            if scheduler.get_job(job_id):
                scheduler.remove_job(job_id)
            
            scheduler.add_job(
                run_meeting_predictions,
                'date',
                run_date=meeting_job_time,
                args=[meeting_date],
                id=job_id,
                replace_existing=True
            )
            
            print(f"✓ Meeting {meeting_date} ({first_race['course']})")
            print(f"  First race: {first_race_time.strftime('%H:%M %Z')}")
            print(f"  Prediction job: {meeting_job_time.strftime('%H:%M %Z')}")
        else:
            print(f"⏭️  Skipping past meeting: {meeting_date} (job time was {meeting_job_time.strftime('%H:%M %Z')})")
    
    # Schedule individual race predictions (2min before each race)
    print("\n🏇 Scheduling individual race predictions:")
    print("-" * 80)
    scheduled_count = 0
    skipped_count = 0
    
    for race_id, date, course, race_name, race_class, distance, post_time in races:
        race_time = parse_hkt_time(post_time)
        
        # Schedule 2 minutes before race
        race_job_time = race_time - timedelta(minutes=2)
        
        # Extract race number from race_name or race_id (needed for both branches)
        race_no = race_id.split('_')[-1] if '_' in race_id else race_id
        
        # Only schedule if in the future (with 5 min buffer to avoid edge cases)
        if race_job_time > now_brisbane + timedelta(minutes=5):
            job_id = f"race_{race_id}"
            
            # Remove existing job if it exists
            if scheduler.get_job(job_id):
                scheduler.remove_job(job_id)
            
            scheduler.add_job(
                run_race_prediction,
                'date',
                run_date=race_job_time,
                args=[race_id, race_no],
                id=job_id,
                replace_existing=True
            )
            
            scheduled_count += 1
            print(f"✓ Race {race_no}: {race_name[:40]}")
            print(f"  Post time: {race_time.strftime('%H:%M %Z')}")
            print(f"  Prediction job: {race_job_time.strftime('%H:%M %Z')}")
        else:
            skipped_count += 1
            if skipped_count <= 3:  # Only print first few to avoid spam
                print(f"⏭️  Skipping past race: {race_no} (job time was {race_job_time.strftime('%H:%M %Z')})")
    
    print("-" * 80)
    print(f"✅ Scheduled {scheduled_count} race predictions")
    if skipped_count > 0:
        print(f"⏭️  Skipped {skipped_count} past races")
    print("=" * 80 + "\n")


def schedule_meeting_predictions(meeting_date: str, first_race_time: datetime, scheduler):
    """
    Schedule a meeting prediction job (30min before first race).
    
    Args:
        meeting_date: Date in YYYY-MM-DD format
        first_race_time: Datetime of first race (Brisbane timezone)
        scheduler: APScheduler instance
    """
    job_time = first_race_time - timedelta(minutes=30)
    job_id = f"meeting_{meeting_date}"
    
    # Remove existing job if it exists
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)
    
    scheduler.add_job(
        run_meeting_predictions,
        'date',
        run_date=job_time,
        args=[meeting_date],
        id=job_id,
        replace_existing=True
    )
    
    print(f"✓ Scheduled meeting predictions for {meeting_date} at {job_time.strftime('%H:%M %Z')}")


def schedule_race_prediction(race_id: str, race_no: str, post_time: datetime, scheduler):
    """
    Schedule a race prediction job (2min before race).
    
    Args:
        race_id: Unique race identifier
        race_no: Race number for display
        post_time: Race post time (Brisbane timezone)
        scheduler: APScheduler instance
    """
    job_time = post_time - timedelta(minutes=2)
    job_id = f"race_{race_id}"
    
    # Remove existing job if it exists
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)
    
    scheduler.add_job(
        run_race_prediction,
        'date',
        run_date=job_time,
        args=[race_id, race_no],
        id=job_id,
        replace_existing=True
    )
    
    print(f"✓ Scheduled race {race_no} prediction at {job_time.strftime('%H:%M %Z')}")


def run_meeting_predictions(meeting_date: str):
    """
    Execute make_predictions.py and email CSV to ALERT_EMAIL.
    
    Args:
        meeting_date: Date in YYYY-MM-DD format
    """
    print("\n" + "=" * 80)
    print(f"🏇 Running meeting predictions for {meeting_date}")
    print("=" * 80)
    
    try:
        # Run make_predictions.py (as module from project root)
        result = subprocess.run(
            [sys.executable, "-m", "src.make_predictions"],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=Path(__file__).parent.parent  # Run from project root
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print("❌ Prediction failed:")
            print(result.stderr)
            return
        
        # Note: Email is sent by make_predictions.py itself, no need to send again
        print("\n✅ Meeting predictions completed (email sent by make_predictions.py)")
        
    except subprocess.TimeoutExpired:
        print("❌ Prediction timed out after 10 minutes")
    except Exception as e:
        print(f"❌ Error running meeting predictions: {e}")
        import traceback
        traceback.print_exc()


def run_race_prediction(race_id: str, race_no: str):
    """
    Execute predict_next_race.py and SMS top pick to ALERT_PHONE.
    
    Args:
        race_id: Unique race identifier
        race_no: Race number for display
    """
    print("\n" + "=" * 80)
    print(f"🏇 Running prediction for Race {race_no} at {datetime.now(BRISBANE).strftime('%H:%M:%S')}")
    print("=" * 80)
    
    try:
        import time
        start_time = time.time()
        # Run predict_next_race.py with --skip-fetch (race should already be in DB)
        result = subprocess.run(
            [sys.executable, "-m", "src.predict_next_race", "--skip-fetch"],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout (training fresh model takes 5-8 minutes)
            cwd=Path(__file__).parent.parent  # Run from project root
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️  Prediction completed in {elapsed:.1f} seconds")
        print(result.stdout)
        
        if result.returncode != 0:
            print("❌ Prediction failed:")
            print(result.stderr)
            return
        
        # Query database for top 5 predictions
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        query = """
            SELECT horse, predicted_rank, predicted_score, win_odds, draw
            FROM runners
            WHERE race_id = ? AND predicted_rank <= 5
            ORDER BY predicted_rank
        """
        
        cur.execute(query, (race_id,))
        results = cur.fetchall()
        conn.close()
        
        if not results:
            print(f"⚠️  No predictions found for race {race_id}")
            return
        
        # Build SMS message with top 5
        sms_lines = [f"Stanley - Race {race_no}"]
        
        for horse, rank, score, odds, draw in results:
            score_pct = score * 100 if score else 0
            odds_str = f"{odds:.1f}" if odds else "N/A"
            draw_str = f"#{draw}" if draw else "?"
            
            # Shortened format to fit more picks
            sms_lines.append(f"{rank}. {horse[:20]} ({odds_str}) {draw_str}")
        
        sms_body = "\n".join(sms_lines)
        
        # Send SMS
        print(f"\n📱 Sending SMS to {ALERT_PHONE}...")
        from twilio.rest import Client
        
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        from_number = os.getenv('TWILIO_FROM_PHONE')
        
        if not all([account_sid, auth_token, from_number]):
            print("⚠️  Missing Twilio credentials, skipping SMS")
            return
        
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=sms_body,
            from_=from_number,
            to=ALERT_PHONE
        )
        
        print(f"✅ SMS sent successfully. SID: {message.sid}")
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"❌ Prediction timed out after {elapsed:.1f} seconds (limit: 600s)")
        print("   This may indicate:")
        print("   - Model training taking too long")
        print("   - Database lock or corruption")
        print("   - System resource exhaustion (memory/CPU)")
    except Exception as e:
        print(f"❌ Error running race prediction: {e}")
        import traceback
        traceback.print_exc()


def refresh_schedule():
    """
    Re-fetch race times and reload schedule.
    This is called automatically by the daily cron job.
    """
    print("\n🔄 Refreshing schedule...")
    
    try:
        # Run fetch_race_times.py from project root
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            [sys.executable, "scripts/fetch_race_times.py"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print("❌ Failed to fetch race times:")
            print(result.stderr)
            return
        
        # Get the scheduler from the global context
        # APScheduler makes the scheduler available via get_scheduler()
        from apscheduler.schedulers import SchedulerNotRunningError
        try:
            # Access scheduler through the module-level variable
            # We'll need to store it globally
            global _scheduler
            if _scheduler:
                load_and_schedule_all_races(_scheduler)
        except NameError:
            print("⚠️  Scheduler not available in refresh context")
        
    except Exception as e:
        print(f"❌ Error refreshing schedule: {e}")


def main():
    """Main entry point"""
    global _scheduler
    
    print("\n" + "=" * 80)
    print("🏇 Stanley Racing Scheduler")
    print("=" * 80)
    print(f"Alert Email: {ALERT_EMAIL}")
    print(f"Alert Phone: {ALERT_PHONE}")
    print(f"Database: {DB_PATH}")
    print(f"Timezone: {BRISBANE}")
    print("=" * 80)
    
    # Ensure directories exist
    Path("data/historical").mkdir(parents=True, exist_ok=True)
    Path("data/predictions").mkdir(parents=True, exist_ok=True)
    
    # Configure APScheduler
    jobstores = {
        'default': SQLAlchemyJobStore(url=f'sqlite:///{SCHEDULER_DB_PATH}')
    }
    executors = {
        'default': ThreadPoolExecutor(max_workers=2)
    }
    job_defaults = {
        'coalesce': False,
        'max_instances': 1,
        'misfire_grace_time': None  # Don't run missed jobs - races are time-sensitive
    }
    
    scheduler = BlockingScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone=BRISBANE
    )
    
    # Store globally for refresh job
    _scheduler = scheduler
    
    # Clean up any stale jobs from previous runs
    print("\n🧹 Cleaning up stale jobs from database...")
    stale_jobs = []
    for job in scheduler.get_jobs():
        # Skip the daily refresh job
        if job.id == 'daily_refresh':
            continue
        # Remove jobs that would have run in the past
        next_run = job.trigger.get_next_fire_time(None, datetime.now(BRISBANE))
        if next_run is None or next_run < datetime.now(BRISBANE):
            stale_jobs.append(job.id)
            scheduler.remove_job(job.id)
    
    if stale_jobs:
        print(f"✓ Removed {len(stale_jobs)} stale job(s)")
    else:
        print("✓ No stale jobs found")
    
    # Step 1: Fetch latest race times
    print("\n📥 Fetching latest race schedule...")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/fetch_race_times.py"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        if result.returncode != 0:
            print("⚠️  Warning: Failed to fetch race times")
            print(result.stderr)
    except Exception as e:
        print(f"⚠️  Warning: Error fetching race times: {e}")
    
    # Step 2: Load and schedule all races
    load_and_schedule_all_races(scheduler)
    
    # Step 3: Schedule daily refresh at 6am Brisbane time
    print("\n🔄 Scheduling daily refresh...")
    scheduler.add_job(
        refresh_schedule,
        'cron',
        hour=6,
        minute=0,
        id='daily_refresh',
        replace_existing=True
    )
    print("✓ Daily refresh scheduled for 6:00 AM Brisbane time")
    
    # Print scheduled jobs
    jobs = scheduler.get_jobs()
    if jobs:
        print("\n📋 Scheduled Jobs:")
        print("-" * 80)
        # Get job details with next run time from scheduler
        job_list = []
        for job in jobs:
            next_run = scheduler.get_job(job.id).trigger.get_next_fire_time(None, datetime.now(BRISBANE))
            if next_run:
                job_list.append((job.id, next_run))
        
        # Sort by next run time
        for job_id, next_run in sorted(job_list, key=lambda x: x[1]):
            print(f"{job_id}: {next_run.strftime('%Y-%m-%d %H:%M %Z')}")
        print("-" * 80)
    
    # Start scheduler
    print("\n✅ Scheduler started. Daily refresh at 6:00 AM.")
    print("Press Ctrl+C to exit.\n")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n\n👋 Scheduler stopped.")


if __name__ == "__main__":
    main()


