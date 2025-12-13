# Stanley Project - DigitalOcean Deployment Guide

**Project:** Stanley Horse Racing Prediction System  
**Type:** Python + Node.js scheduled prediction service  
**Primary Use:** Hong Kong horse racing predictions with automated scheduling

---

## 📁 Project Structure

```
happy-valley-model/
├── src/                          # Core Python modules
│   ├── make_predictions.py       # Generate predictions for entire meeting
│   ├── predict_next_race.py      # Predict single upcoming race
│   ├── analyze_predictions.py    # Analyze prediction accuracy vs results
│   ├── train_model.py            # Train ML model (HistGradientBoosting)
│   ├── model.py                  # Model definition and training logic
│   ├── features.py               # Feature engineering from database
│   ├── feature_engineer.py       # Advanced feature creation
│   ├── email_utils.py            # Email sending via SMTP
│   ├── sms_utils.py              # SMS notifications via Twilio
│   ├── update_odds.py            # Re-fetch odds and regenerate predictions
│   ├── horse_matcher.py          # Name normalization
│   └── pair_model.py             # Quinella/exacta predictions
│
├── scripts/                      # Operational scripts
│   ├── scheduler.py              # ⭐ MAIN ENTRY POINT - APScheduler daemon
│   ├── fetch_race_times.py       # Fetch upcoming race schedule
│   ├── fetch_next_meeting.mjs    # Node.js HKJC API fetcher
│   ├── backfill_predictions.py   # Backfill historical predictions
│   ├── backfill_odds.py          # Backfill odds data
│   └── package.json              # Node.js dependencies (hkjc-api)
│
├── scrapers/                     # Data collection
│   ├── hkjc_scraper.py           # Scrape HKJC website
│   ├── hkjc_future_scraper.py    # Scrape future races
│   └── hkjc_results_all_scraper.py # Scrape results
│
├── tests/                        # Testing suite
│   └── test_*.py                 # Various test modules
│
├── data/                         # Data storage (NOT in git)
│   ├── historical/
│   │   ├── hkjc.db               # ⭐ Main SQLite database
│   │   └── scheduler_jobs.db     # APScheduler job store
│   ├── predictions/              # JSON racecards & CSV predictions
│   ├── processed/                # Processed datasets
│   ├── raw/                      # Raw scraped data
│   └── models/                   # Trained model files
│
├── logs/                         # Application logs (NOT in git)
│   └── scheduler.log             # Scheduler activity log
│
├── docs/                         # Documentation
│   ├── HANDOFF_DOCUMENT.md       # System overview and workflows
│   ├── FAST_PREDICTIONS_GUIDE.md # Quick prediction guide
│   └── ODDS_DRIFT_ANALYSIS_GUIDE.md # Odds drift tracking
│
├── requirements.txt              # ⭐ Python dependencies
├── .env                          # ⭐ Environment variables (NOT in git)
└── .gitignore                    # Ignore data/, logs/, .env, etc.
```

---

## 📦 Dependencies

### Python Dependencies (requirements.txt)

```txt
# Core scientific stack
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2

# Gradient boosting for tabular racing data
xgboost==2.1.1
lightgbm==4.5.0

# Web + PDF data ingestion
requests==2.32.3
beautifulsoup4==4.12.3
pdfplumber==0.11.0

# Utilities
tqdm==4.66.5
joblib==1.4.2
python-dotenv==1.0.0
twilio==9.3.7                    # SMS notifications
apscheduler==3.10.4              # Job scheduling
sqlalchemy==2.0.36               # APScheduler job store

# Exploration & visualization
jupyter==1.0.0
matplotlib==3.9.2
seaborn==0.13.2
```

**Install:**
```bash
pip install -r requirements.txt
```

### Node.js Dependencies

**File:** `scripts/package.json`

```json
{
  "name": "happy-valley-model",
  "version": "1.0.0",
  "type": "commonjs",
  "dependencies": {
    "hkjc-api": "^1.0.3"           # HKJC race data API wrapper
  }
}
```

**Install:**
```bash
cd scripts && npm install
```

---

## 🚀 Main Entry Point

### Primary Application: Scheduler Daemon

**File:** `scripts/scheduler.py`

**Purpose:** 
- Automated scheduling system for race predictions
- Schedules meeting predictions 30 mins before first race
- Schedules individual race predictions 2 mins before each race
- Daily refresh at 10am Brisbane time (8am Hong Kong time)
- Sends predictions via email and SMS

**Start Command:**
```bash
python scripts/scheduler.py
```

**What It Does:**
1. Fetches latest race schedule from HKJC API
2. Schedules prediction jobs based on race times
3. Runs `src.make_predictions` for meeting predictions (email CSV)
4. Runs `src.predict_next_race` for individual races (SMS top pick)
5. Auto-refreshes schedule daily to catch new races

**Requirements:**
- Must run continuously (use systemd, supervisor, or screen)
- Requires environment variables (see below)
- Creates `data/historical/scheduler_jobs.db` for persistent jobs

---

## 🌐 External Service Dependencies

### 1. **HKJC API (Hong Kong Jockey Club)**
- **Purpose:** Fetch race schedules, racecards, odds
- **Access:** Public API via `hkjc-api` npm package
- **Usage:** `scripts/fetch_next_meeting.mjs`
- **No authentication required**

### 2. **SMTP Email Server**
- **Purpose:** Send prediction CSVs to recipients
- **Usage:** `src/email_utils.py` via `src/make_predictions.py`
- **Required Env Vars:**
  - `SMTP_SERVER` (e.g., smtp.gmail.com)
  - `SMTP_PORT` (e.g., 587)
  - `SMTP_USERNAME`
  - `SMTP_PASSWORD`
  - `ALERT_EMAIL` (recipient)

### 3. **Twilio SMS**
- **Purpose:** Send race predictions via SMS
- **Usage:** `src/sms_utils.py` via `src/predict_next_race.py`
- **Required Env Vars:**
  - `TWILIO_ACCOUNT_SID`
  - `TWILIO_AUTH_TOKEN`
  - `TWILIO_FROM_PHONE`
  - `ALERT_PHONE` (recipient, in scripts/scheduler.py)

### 4. **SQLite Database**
- **Purpose:** Store race data, predictions, results
- **File:** `data/historical/hkjc.db`
- **No external server required** (file-based)

---

## 🔐 Environment Variables

### DigitalOcean App Platform Configuration

Configure these environment variables in DigitalOcean's App Platform dashboard under **Settings > App-Level Environment Variables**:

```bash
# Database Configuration (PostgreSQL - Managed by DigitalOcean)
USE_POSTGRES=true
DATABASE_URL=${db.DATABASE_URL}  # Auto-populated by DigitalOcean when you attach a managed database

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=recipient@example.com

# Twilio SMS Configuration
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your-auth-token
TWILIO_FROM_PHONE=+1234567890

# JWT Secret for Web Dashboard
JWT_SECRET_KEY=your-secure-random-string-here

# Alert Phone (hardcoded in scheduler.py)
# ALERT_PHONE=+61417676973  # Set in scripts/scheduler.py:29
```

### Local Development (.env file)

For local development, create a `.env` file in project root with the same variables.

**Security Notes:**
- `.env` is in `.gitignore` - **never commit it to git**
- Use app-specific passwords for Gmail
- DATABASE_URL is automatically provided by DigitalOcean when you attach a managed PostgreSQL database
- In DigitalOcean, use `${db.DATABASE_URL}` to reference the attached database
- Store all secrets in DigitalOcean's environment variables, not in code

---

## 🗄️ Database Schema

**Database:** SQLite (`data/historical/hkjc.db`)

### Key Tables

#### `races` table
```sql
race_id TEXT PRIMARY KEY
date TEXT
course TEXT
race_name TEXT
class TEXT
distance INTEGER
going TEXT
rail TEXT
post_time TEXT  -- ISO 8601 format with timezone
```

#### `runners` table
```sql
race_id TEXT
horse_id TEXT
horse TEXT
draw INTEGER
weight REAL
win_odds REAL
jockey TEXT
trainer TEXT
status TEXT
position INTEGER
btn REAL  -- Beaten by lengths
time REAL  -- Race time
predicted_rank INTEGER
predicted_score REAL
prediction_date TEXT
model_version TEXT
```

#### `predictions` table (multi-version tracking)
```sql
prediction_id INTEGER PRIMARY KEY
race_id TEXT
horse_id TEXT
predicted_rank INTEGER
predicted_score REAL
prediction_timestamp TEXT
model_version TEXT
win_odds_at_prediction REAL
```

---

## 🔄 Typical Workflows

### 1. Scheduled Prediction (Automated)
```bash
# Runs automatically via scheduler.py
# 30 mins before first race: email meeting predictions
# 2 mins before each race: SMS top pick
```

### 2. Manual Meeting Predictions
```bash
python -m src.make_predictions
# Fetches races, generates predictions, emails CSV
```

### 3. Manual Single Race Prediction
```bash
python -m src.predict_next_race
# Predicts only next upcoming race with live odds
```

### 4. Analyze Results After Race
```bash
python -m src.analyze_predictions --fetch-results
# Fetches results and analyzes prediction accuracy
```

### 5. Update Odds Before Race
```bash
python -m src.update_odds
# Re-fetches live odds and regenerates predictions
```

---

## 🐳 DigitalOcean Deployment Recommendations

### Option 1: Droplet (VPS)

**Recommended:** Basic Droplet ($6-12/month)

**Setup:**
```bash
# 1. Create Ubuntu 22.04 droplet
# 2. Install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install python3.11 python3-pip nodejs npm sqlite3 -y

# 3. Clone/upload project
cd /opt
git clone <your-repo> stanley
cd stanley/happy-valley-model

# 4. Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Install Node.js dependencies
cd scripts && npm install && cd ..

# 6. Create .env file with credentials
nano .env

# 7. Initialize database structure (if needed)
mkdir -p data/historical data/predictions logs

# 8. Run scheduler with systemd
sudo nano /etc/systemd/system/stanley-scheduler.service
```

**systemd Service File:**
```ini
[Unit]
Description=Stanley Racing Scheduler
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/stanley/happy-valley-model
Environment="PATH=/opt/stanley/happy-valley-model/venv/bin"
ExecStart=/opt/stanley/happy-valley-model/venv/bin/python scripts/scheduler.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable stanley-scheduler
sudo systemctl start stanley-scheduler
sudo systemctl status stanley-scheduler

# View logs
journalctl -u stanley-scheduler -f
```

### Option 2: App Platform (PaaS)

**Not Recommended** - App Platform expects web servers, not long-running schedulers.

---

## 📊 Resource Requirements

- **CPU:** 1 vCPU (minimal, predictions run quickly)
- **RAM:** 1-2 GB (model training + pandas operations)
- **Storage:** 5-10 GB (database, logs, models)
- **Network:** Outbound only (HKJC API, SMTP, Twilio)

---

## 🔧 Environment Setup Summary

### 1. Python Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Node.js Setup
```bash
cd scripts
npm install
cd ..
```

### 3. Database Setup
```bash
mkdir -p data/historical data/predictions logs
# Database will be created automatically on first run
```

### 4. Configuration
```bash
cp .env.example .env  # If you create one
nano .env  # Add SMTP, Twilio credentials
```

### 5. Test Run
```bash
# Test single prediction
python -m src.predict_next_race --skip-fetch

# Start scheduler
python scripts/scheduler.py
```

---

## 🚨 Important Notes

1. **Timezone:** App uses Brisbane time (AEST/AEDT) for scheduling
2. **Database:** SQLite is file-based - ensure `data/` persists on server
3. **Logs:** Check `logs/scheduler.log` for job execution history
4. **Race Times:** Scheduler auto-refreshes at 10am daily
5. **Missed Jobs:** Scheduler skips past races (misfire_grace_time=None)
6. **SMS Costs:** Twilio charges per SMS - monitor usage
7. **Email Limits:** Gmail has daily send limits (~500/day)

---

## 🐛 Troubleshooting

### Scheduler won't start
```bash
# Check Python dependencies
pip list

# Check Node.js dependencies
cd scripts && npm list

# Check environment variables
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('SMTP_SERVER'))"
```

### No predictions generated
```bash
# Check if races exist in database
sqlite3 data/historical/hkjc.db "SELECT COUNT(*) FROM races WHERE post_time > datetime('now');"

# Manually fetch races
python scripts/fetch_race_times.py
```

### SMS not sending
```bash
# Test Twilio credentials
python -c "from twilio.rest import Client; import os; from dotenv import load_dotenv; load_dotenv(); c = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN')); print('OK')"
```

---

## 📚 Additional Documentation

- `docs/HANDOFF_DOCUMENT.md` - Comprehensive system overview
- `docs/FAST_PREDICTIONS_GUIDE.md` - Quick prediction workflows
- `docs/ODDS_DRIFT_ANALYSIS_GUIDE.md` - Odds tracking methodology
- `docs/PREDICTIONS_TABLE_GUIDE.md` - Database schema details

---

**Last Updated:** December 13, 2025  
**Contact:** See ALERT_EMAIL in scheduler.py

