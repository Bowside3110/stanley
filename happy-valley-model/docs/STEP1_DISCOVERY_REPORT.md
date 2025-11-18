# Step 1: Discovery Report - Stanley Database Schema & Patterns

**Date:** November 18, 2025  
**Task:** Integrate HKJC Results Scraper into Stanley

---

## 🗄️ Database Schema

Stanley uses **SQLite** (`hkjc.db`) with the following relevant tables:

### **races** table
Primary table for race metadata:
- `race_id` (TEXT, PRIMARY KEY)
- `date` (TEXT)
- `course` (TEXT) - venue code (e.g., "ST", "HV")
- `race_name` (TEXT)
- `class` (TEXT)
- `distance` (REAL)
- `going` (TEXT)
- `rail` (TEXT)
- `post_time` (TEXT)

### **runners** table
Primary table for horse/runner data and results:
- `race_id` (TEXT, PRIMARY KEY part 1)
- `horse_id` (TEXT, PRIMARY KEY part 2)
- `horse` (TEXT) - horse name
- `draw` (TEXT)
- `weight` (TEXT)
- `jockey` (TEXT)
- `jockey_id` (TEXT)
- `trainer` (TEXT)
- `trainer_id` (TEXT)
- `win_odds` (REAL) - **odds at prediction time**
- **`position` (INTEGER)** - ⭐ **ACTUAL FINISHING POSITION** (this is where results go!)
- `status` (TEXT) - runner status (e.g., "Declared", "Scratched")
- `btn` (REAL) - beaten by (lengths)
- `time` (TEXT) - finishing time
- `starting_price` (REAL) - final odds
- **`predicted_rank` (INTEGER)** - Stanley's prediction
- **`predicted_score` (REAL)** - Stanley's confidence score
- `prediction_date` (TEXT)
- `model_version` (TEXT)

### **predictions** table
Separate table for tracking predictions (newer schema):
- `prediction_id` (INTEGER, PRIMARY KEY, autoincrement)
- `race_id` (TEXT, NOT NULL)
- `horse_id` (TEXT, NOT NULL)
- `predicted_rank` (INTEGER)
- `predicted_score` (REAL)
- `prediction_timestamp` (TEXT, NOT NULL)
- `model_version` (TEXT)
- `win_odds_at_prediction` (REAL)

### **results** table
Lightweight results tracking:
- `race_id` (TEXT, PRIMARY KEY part 1)
- `horse_id` (TEXT, PRIMARY KEY part 2)
- `position` (INT)

---

## 🔌 Database Connection Pattern

Stanley uses **raw `sqlite3`** (not SQLAlchemy):

```python
import sqlite3

DB_PATH = "data/historical/hkjc.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Execute queries
cursor.execute("SELECT ...")
results = cursor.fetchall()

# Insert/update
cursor.execute("INSERT OR REPLACE INTO ...")
conn.commit()

conn.close()
```

**Key patterns:**
- Uses `INSERT OR REPLACE` for upsert logic
- No ORM - raw SQL queries
- Connection opened/closed per operation (no connection pooling)
- Database path: `data/historical/hkjc.db` (relative to project root)

---

## 📁 Codebase Structure

### Existing scrapers location:
`/Users/bendunn/Stanley/happy-valley-model/scrapers/`
- `hkjc_scraper.py` - Scrapes individual race results (runners + dividends)
- `hkjc_future_scraper.py` - Scrapes future racecards

### Existing results fetching:
- `scripts/fetch_results_hkjc.py` - Fetches results via HKJC API (Node.js wrapper)
- Uses `hkjc-api` npm package to get results
- Updates `runners.position` field directly

### Key scripts:
- `make_predictions.py` - Main prediction generation
- `analyze_predictions.py` - Compares predictions vs actual results
- `update_odds.py` - Updates odds data

---

## 🪵 Logging System

Stanley uses **simple print statements** (no formal logging framework):
- Emoji-based status indicators: ✅ ❌ ⚠️ 📥 🔄
- Print statements for progress/errors
- No `logging` module configuration found
- Some scripts have basic print-based logging

**Example pattern:**
```python
print("=" * 80)
print("PREDICTION ACCURACY ANALYSIS")
print("=" * 80)
print(f"✅ Updated {updated} runners")
print(f"⚠️ No results available")
```

---

## 🎯 Key Findings

### 1. **Results are already stored in `runners.position`**
The `runners` table already has a `position` column for actual finishing positions. This is the primary location for race results.

### 2. **Existing scraper exists**
`scrapers/hkjc_scraper.py` already scrapes HKJC results, but uses a different approach (scrapes individual race pages, extracts runners + dividends + metadata).

### 3. **Existing results fetcher exists**
`scripts/fetch_results_hkjc.py` already fetches results using the HKJC API (via Node.js), which is more reliable than HTML scraping.

### 4. **Performance analysis already implemented**
`analyze_predictions.py` already compares predictions vs actual results and calculates metrics like:
- Top-1/Top-3 accuracy
- MAE (Mean Absolute Error)
- Correlation
- Odds drift analysis

---

## 📊 Column Mapping: HKJC HTML → Stanley Schema

Based on the provided scraper code and Stanley's schema:

| HKJC HTML Column | Stanley Column | Notes |
|------------------|----------------|-------|
| `RaceDate` | `races.date` | Format: YYYY/MM/DD |
| `Venue` | `races.course` | Map "Happy Valley" → "HV", "Sha Tin" → "ST" |
| `RaceNumber` | Part of `race_id` | Need to construct race_id or match by race_name |
| `Horse No.` | `runners.draw` or match logic | Used to identify runner |
| `Horse Name` | `runners.horse` | Use `normalize_horse_name()` for matching |
| `Jockey` | `runners.jockey` | Already in schema |
| `Trainer` | `runners.trainer` | Already in schema |
| `Pla.` (Position/Place) | **`runners.position`** | ⭐ Main target column |
| `Win Odds` | `runners.starting_price` | Final odds (vs `win_odds` at prediction time) |

---

## ⚠️ Integration Considerations

### Simplicity First Analysis:

**Question:** Do we actually need the new HKJC results scraper?

**Current state:**
1. ✅ Stanley already has `scripts/fetch_results_hkjc.py` that fetches results via API
2. ✅ Results are already stored in `runners.position`
3. ✅ Performance analysis is already implemented in `analyze_predictions.py`
4. ✅ An HTML scraper already exists in `scrapers/hkjc_scraper.py`

**Potential issues with existing system:**
- API-based fetcher (`scripts/fetch_results_hkjc.py`) requires Node.js and `hkjc-api` package
- May have reliability issues if API is down
- HTML scraper in `scrapers/` uses different URL pattern (individual race pages)

**Benefits of new scraper:**
- ✅ Scrapes from "ResultsAll" page (all races at once)
- ✅ More robust error handling and retry logic
- ✅ Better HTML structure validation
- ✅ Python-only (no Node.js dependency)
- ✅ Production-ready with status enums and logging

**Recommendation:**
- Integrate the new scraper as a **backup/alternative** to the API-based fetcher
- Keep existing `scripts/fetch_results_hkjc.py` as primary method
- Use new scraper when API fails or for historical backfilling

---

## 🎯 Next Steps (for Step 2)

1. Understand Stanley's codebase structure and where to place new scraper
2. Identify coding conventions and patterns
3. Determine integration strategy (replace vs augment existing scrapers)

---

**Status:** ✅ Step 1 Complete - Ready for Step 2

