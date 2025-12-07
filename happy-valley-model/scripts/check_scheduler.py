#!/usr/bin/env python3
"""
Check scheduler status - shows running schedulers and upcoming jobs.

Usage:
    python scripts/check_scheduler.py
"""

import sqlite3
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

BRISBANE = ZoneInfo("Australia/Brisbane")
SCHEDULER_DB = "data/historical/scheduler_jobs.db"

def check_running_schedulers():
    """Check for running scheduler processes"""
    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True,
        text=True
    )
    
    schedulers = [line for line in result.stdout.split('\n') 
                  if 'scheduler.py' in line and 'grep' not in line]
    
    return schedulers

def check_scheduled_jobs():
    """Check jobs in scheduler database"""
    conn = sqlite3.connect(SCHEDULER_DB)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id, next_run_time 
        FROM apscheduler_jobs 
        ORDER BY next_run_time 
        LIMIT 10
    """)
    
    jobs = cur.fetchall()
    conn.close()
    
    return jobs

def main():
    print("\n" + "=" * 80)
    print("🔍 Scheduler Status Check")
    print("=" * 80)
    
    # Check running processes
    print("\n📊 Running Scheduler Processes:")
    print("-" * 80)
    schedulers = check_running_schedulers()
    
    if schedulers:
        for sched in schedulers:
            # Extract PID and start time
            parts = sched.split()
            pid = parts[1]
            start_time = parts[8]
            print(f"  PID {pid} - Started: {start_time}")
        
        if len(schedulers) > 1:
            print(f"\n⚠️  WARNING: {len(schedulers)} schedulers running!")
            print("   Only one scheduler should be running at a time.")
            print("   Kill extras with: kill <PID>")
    else:
        print("  ❌ No scheduler processes found")
    
    # Check scheduled jobs
    print("\n📅 Upcoming Jobs:")
    print("-" * 80)
    jobs = check_scheduled_jobs()
    
    if jobs:
        now = datetime.now(BRISBANE)
        
        for job_id, next_run_ts in jobs:
            next_run = datetime.fromtimestamp(next_run_ts, tz=BRISBANE)
            time_until = (next_run - now).total_seconds() / 60
            
            if time_until < 0:
                status = f"⏰ OVERDUE by {abs(time_until):.0f} min"
            elif time_until < 5:
                status = f"🔥 IMMINENT ({time_until:.0f} min)"
            elif time_until < 60:
                status = f"⏱️  Soon ({time_until:.0f} min)"
            else:
                status = f"⏳ Later ({time_until/60:.1f} hrs)"
            
            print(f"  {job_id:30s} {next_run.strftime('%H:%M %Z')}  {status}")
    else:
        print("  ❌ No jobs scheduled")
    
    print("-" * 80)
    print(f"\nCurrent time: {datetime.now(BRISBANE).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

