# Handoff Document - Horse Racing Prediction System

**Date:** November 9, 2025  
**Project:** Stanley - Happy Valley Model  
**Last Session Summary:** Prediction tracking, analysis system, and result fetching workflows

---

## 🎯 Current State

### What's Working
✅ **Prediction Pipeline** - Fully operational
- `make_predictions.py` fetches racecards, generates predictions, saves to CSV and database
- Predictions include confidence ratings, jockey/trainer evaluations
- CSV output is formatted and ready for sharing

✅ **Prediction Tracking** - Implemented and tested
- `runners` table stores: `predicted_rank`, `predicted_score`, `prediction_date`, `model_version`
- `backfill_predictions.py` successfully backfilled historical predictions from CSV files
- Predictions are automatically saved to database when `make_predictions.py` runs

✅ **Analysis System** - Working with Odds Drift Tracking
- `analyze_predictions.py` analyzes prediction accuracy vs actual results
- Generates CSV (`analysis_YYYY-MM-DD.csv`) and text summary for latest race meeting
- Metrics: Top-1/Top-3 accuracy, MAE, correlation, confidence-level breakdown
- **NEW:** Tracks odds drift when multiple predictions exist (morning vs pre-race)
- Shows rank changes, odds changes, and accuracy comparison by timing
- See `ODDS_DRIFT_ANALYSIS_GUIDE.md` for details

✅ **Odds Update Workflow** - Available
- `update_odds.py` re-fetches live odds and optionally re-runs predictions
- Useful for mitigating odds drift between prediction time and race time

---

## 📊 Key Files and Their Purpose

### Main Scripts
| File | Purpose | When to Use |
|------|---------|-------------|
| `python -m src.make_predictions` | Fetch racecards, generate predictions, save to DB/CSV | Before each race meeting (all races) |
| `python -m src.predict_next_race` | Predict ONLY the next upcoming race (uses pre-trained model) | 5-15 mins before a specific race |
| `python -m src.analyze_predictions` | Analyze prediction accuracy after races complete | After race results are available |
| `python -m src.analyze_predictions --fetch-results` | Auto-fetch missing results then analyze | When you want one-step analysis |
| `python -m src.update_odds` | Re-fetch live odds and regenerate predictions | 5-10 mins before race start (all races) |
| `python scripts/backfill_predictions.py` | Backfill predictions from CSV to database | One-time or when recovering data |

### Data Flow Scripts (Experimental)
| File | Purpose | Status |
|------|---------|--------|
| `scripts/fetch_missing_results.py` | Fetch results for dates with predictions but no results | Experimental |
| `scripts/fetch_results_hkjc.py` | Fetch results using HKJC API (Node.js) | Experimental |
| `scripts/fetch_results_simple.py` | Fetch results using The Racing API | Experimental (auth issues) |
| `scripts/backfill_btn_time.py` | Migrate btn/time from horse_results to runners | Completed, archived |

### Core Modules
| File | Purpose |
|------|---------|
| `src/features.py` | Feature engineering from database |
| `src/predict_future.py` | Prediction logic and CSV formatting |
| `src/model.py` | Model training (HistGradientBoostingClassifier) |
| `src/horse_matcher.py` | Name normalization for horses/jockeys/trainers |

---

## 🗄️ Database Schema

### `races` table
- Stores race metadata: `race_id`, `date`, `course`, `race_name`, `class`, `distance`, `going`, `rail`, **`post_time`**

### `runners` table
- Stores runner data: `race_id`, `horse_id`, `horse`, `draw`, `weight`, `win_odds`, `jockey`, `trainer`, `status`, `position`, `btn`, `time`
- **Prediction fields (latest only):** `predicted_rank`, `predicted_score`, `prediction_date`, `model_version`

### `predictions` table ⭐ NEW
- Stores **all predictions** over time (not just latest)
- Allows tracking odds drift and comparing multiple prediction runs
- Fields: `prediction_id`, `race_id`, `horse_id`, `predicted_rank`, `predicted_score`, `prediction_timestamp`, `model_version`, `win_odds_at_prediction`
- See `PREDICTIONS_TABLE_GUIDE.md` for usage examples

---

## 📁 CSV Output Format

**File:** `data/predictions/predictions_YYYY-MM-DD.csv`

**Column Order:**
1. `race_date`, `race_time` (HH:MM format)
2. `course`, `race_name`, `race_class`, `dist_m`
3. `horse`, `win_odds`, `score` (%), `confidence` (interpretation), `rank`, `draw`
4. `horse_pos_avg3` (avg position last 3 races)
5. `weight`, `jockey`, `jockey_win30` (%), `jockey_rating` (interpretation)
6. `trainer`, `trainer_win30` (%), `trainer_rating` (interpretation), `status`
7. Additional features: `market_prob`, `market_logit`, `horse_odds_efficiency`, `horse_odds_trend`, `trainer_odds_bias`, etc.

**Removed Columns** (can't be reliably populated from current data sources):
- `btn_last3`, `time_last3`, `last_run`, `form`, `form_close`, `class_move`, `dist_delta`
- `headgear`, `wind_surgery` and related flags
- `going`, `rail`, `horse_last_placed`, `match_confidence`
- All ID columns: `race_id`, `horse_id`, `jockey_id`, `trainer_id`

---

## 🔧 Known Issues and Workarounds

### Issue 1: Odds Drift
**Problem:** Betting odds change significantly between prediction time (when CSV is generated) and race start time.

**Workaround:** Run `update_odds.py` 5-10 minutes before race start:
```bash
python -m src.update_odds --race-date 2025-11-10 --show-changes --rerun-predictions
```

### Issue 2: Results Fetching
**Problem:** No single reliable method to fetch historical results automatically.

**Current Solution:** Re-run `make_predictions.py` after races complete. The HKJC API returns `finalPosition` for completed races, which updates the `runners` table.

**Alternative (not yet implemented):** The Racing API has results endpoint, but requires valid credentials.

### Issue 3: Empty Feature Columns
**Problem:** Some historical features (`btn_last3`, `time_last3`, etc.) were empty.

**Resolution:** These columns have been removed from CSV output as they cannot be reliably populated from current data sources (HKJC API + runners table).

### Issue 4: Jockey/Trainer Name Matching
**Problem:** Historical data uses full names ("Zac Purton"), HKJC API uses abbreviated names ("Z Purton").

**Resolution:** `normalize_jockey_name()` in `src/horse_matcher.py` now extracts only the last name for consistent matching.

---

## 📈 Analysis Output

### Two Analyses in `analyze_predictions.py`

1. **Overall Historical Analysis** (`analyze_prediction_accuracy()`)
   - Shows ALL predictions across all dates
   - Useful for tracking model performance over time
   - Output: Console summary with trends, accuracy by class, etc.

2. **Latest Race Meeting** (`create_race_meeting_csv()`)
   - Shows ONLY the most recent race meeting
   - Output: CSV + text summary files
   - Files: `data/predictions/analysis_YYYY-MM-DD.csv` and `analysis_YYYY-MM-DD_summary.txt`

**Note:** When you run `analyze_predictions.py`, it runs BOTH analyses. The console shows all historical data, but the CSV/text files show only the latest meeting.

---

## 🚀 Typical Workflow

### Before Race Meeting (Day Before or Morning Of)
1. Run `make_predictions.py` to fetch racecards and generate predictions
   ```bash
   python -m src.make_predictions
   ```
2. This automatically:
   - ✅ Trains the model and saves it for later use
   - ✅ Generates predictions for all races
   - ✅ Saves CSV: `data/predictions/predictions_YYYY-MM-DD.csv`
   - ✅ Creates pre-trained model for fast predictions later
3. Review the CSV or use the console "cheat sheet" output

### Just Before a Specific Race (5-15 mins before post time) ⭐ NEW
1. Run `predict_next_race.py` for **fast predictions** (~5 seconds):
   ```bash
   python -m src.predict_next_race
   ```
2. This automatically:
   - ✅ Uses pre-trained model (fast!)
   - ✅ Fetches the latest odds
   - ✅ Predicts ONLY the next race
   - ✅ Saves to `predictions` table with timestamp
   - ✅ Displays clean console output with countdown
3. **Speed:** ~5 seconds (uses pre-trained model by default)

**Note:** The pre-trained model is automatically created when you run `make_predictions.py`.  
**Advanced:** Use `--no-pretrained` to force training a fresh model (~72 seconds).

### Alternative: Update All Races (Optional)
1. Update odds for all races to account for market movements:
   ```bash
   python -m src.update_odds --race-date 2025-11-10 --show-changes --rerun-predictions
   ```

### After Races Complete
1. Re-run `make_predictions.py` to fetch results (updates `position` in `runners` table)
   ```bash
   python -m src.make_predictions
   ```
2. Run analysis to evaluate prediction accuracy:
   ```bash
   python -m src.analyze_predictions
   ```
3. Review:
   - Console output: Overall historical performance
   - CSV: `data/predictions/analysis_YYYY-MM-DD.csv` (detailed latest meeting)
   - Text: `data/predictions/analysis_YYYY-MM-DD_summary.txt` (summary)

---

## 🎓 Key Learnings from This Session

### 1. Feature Engineering Challenges
- Historical features require careful data source management
- The `horse_results` table is deprecated; all data should come from `runners` table
- Some features (form, class moves, equipment changes) are not available via HKJC API

### 2. Name Matching is Critical
- Inconsistent name formats between data sources cause feature calculation failures
- Using only last names for jockeys improved historical matching significantly
- Always normalize names before matching

### 3. Odds Drift is Real
- Odds can change 20-30% between prediction time and race start
- Consider implementing a "refresh odds" workflow for live betting
- `update_odds.py` provides a quick fix but adds operational overhead

### 4. Prediction Tracking is Essential
- Storing predictions in the database enables systematic accuracy analysis
- The `model_version` field allows comparing different model iterations
- Backfilling historical predictions from CSV files was successful

### 5. Result Fetching is Complex
- HKJC API returns results for completed races (via `getAllRaces()`)
- The Racing API has a results endpoint but requires valid credentials
- Simplest solution: Re-run `make_predictions.py` after races complete

---

## 🔮 Potential Next Steps

### High Priority
- [ ] **Automate the workflow** - Create a cron job or scheduled task to run predictions daily
- [ ] **Improve odds refresh** - Implement automatic odds updates closer to race time
- [ ] **Validate model performance** - Run backtests on more historical data to quantify accuracy

### Medium Priority
- [ ] **Implement staking strategy** - Use confidence levels to determine bet sizing
- [ ] **Add exotic bet predictions** - Quinella, exacta, trifecta predictions
- [ ] **Create web dashboard** - Display predictions and analysis in a user-friendly interface

### Low Priority
- [ ] **Optimize feature engineering** - Investigate which features have the most predictive power
- [ ] **Experiment with different models** - Try neural networks, XGBoost, etc.
- [ ] **Add more data sources** - Incorporate weather, track conditions, etc.

---

## 📝 Important Notes

### Database Transactions (from User Rules)
- When debugging stateful test failures or flaky DB behavior, check if `set_session` is applied inside the correct transaction scope
- Use helper functions to safely wrap `set_session` usage and avoid cross-request leaks
- See `POSTGRES_TRANSACTION_FIX_SUMMARY.md` for context (if it exists in the project)

### Git Workflow
- Always commit logical changes separately
- Use clear, imperative commit messages
- Never force push to main/master
- The `scripts/` directory is in `.gitignore` (intentional)

### Code Style
- Follow existing patterns in the codebase
- Prioritize working code over perfect code
- Make small, incremental changes
- Test after each change

---

## 🆘 Troubleshooting

### "No such column: race_time"
**Solution:** The `post_time` column was added to the `races` table. If you see this error, run:
```sql
ALTER TABLE races ADD COLUMN post_time TEXT;
```

### "Empty btn_last3 / time_last3 columns"
**Solution:** These columns have been removed from the CSV output. They cannot be reliably populated from current data sources.

### "Jockey/Trainer win rates are all zero"
**Solution:** Check name normalization in `src/horse_matcher.py`. Ensure `normalize_jockey_name()` extracts only the last name.

### "Analysis shows multiple race dates"
**Solution:** `analyze_predictions.py` runs two analyses: one for all historical data (console), one for the latest meeting (CSV/text files). The CSV and text files show only the latest meeting.

### "Can't find analysis summary file"
**Solution:** Check `data/predictions/` directory. Files are named `analysis_YYYY-MM-DD_summary.txt` and `analysis_YYYY-MM-DD.csv`.

---

## 📚 Related Documentation

- `WORKFLOW_SUMMARY.md` - ⭐ **START HERE:** Quick reference guide for daily workflow
- `FAST_PREDICTIONS_GUIDE.md` - ⚡ Detailed guide to 5-second predictions with pre-trained models
- `PREDICTIONS_TABLE_GUIDE.md` - How to use the predictions table for tracking odds drift
- `ODDS_DRIFT_ANALYSIS_GUIDE.md` - 📊 How to analyze odds changes and prediction timing accuracy
- `CSV_COLUMN_GUIDE.txt` - Detailed explanation of each CSV column (if it exists)
- `README.md` - Project overview and setup instructions
- `POSTGRES_TRANSACTION_FIX_SUMMARY.md` - Database transaction handling (if it exists)
- `POSTGRES_TRANSACTION_HELPER_UTILITIES.md` - Transaction helper functions (if it exists)

---

## 🎬 Quick Start for Next Session

1. **Check what's in the database:**
   ```bash
   sqlite3 data/historical/hkjc.db "SELECT COUNT(*) FROM runners WHERE predicted_rank IS NOT NULL;"
   ```

2. **Run predictions for upcoming races:**
   ```bash
   python -m src.make_predictions
   ```

3. **Analyze latest results (with auto-fetch):**
   ```bash
   python -m src.analyze_predictions --fetch-results
   ```
   
   Or without fetching:
   ```bash
   python -m src.analyze_predictions
   ```

4. **Check git status:**
   ```bash
   git status
   ```

---

## 🆕 Recent Updates (November 9, 2025)

### New Features Added:
1. **`predict_next_race.py` script** - Predict only the next upcoming race
   - Handles timezone conversion (HKT → Brisbane)
   - Shows countdown to race start
   - **Uses pre-trained model by default** for ultra-fast predictions (~5s)
   - Clears existing future races to ensure only predicting on the new race
   - Optional: `--no-pretrained` flag to train fresh model (~72s)

2. **Pre-trained model system** ⚡ NEW - Fast predictions with saved models
   - `make_predictions.py` now automatically saves a pre-trained model
   - Reuse for multiple fast predictions throughout the day
   - **Speed improvement: 72 seconds → 5 seconds** (93% faster)
   - Odds are only 5 seconds stale instead of 72 seconds
   - Optional: `train_model.py` for training without predictions
   - See `FAST_PREDICTIONS_GUIDE.md` for detailed usage

3. **`predictions` table** - Track multiple predictions per race over time
   - Stores ALL predictions with timestamps
   - Captures odds at prediction time
   - Enables odds drift analysis
   - Allows comparing predictions made at different times
   - See `PREDICTIONS_TABLE_GUIDE.md` for detailed usage

4. **Improved prediction tracking** - Both `make_predictions.py` and `predict_next_race.py` now:
   - Save to `predictions` table (all predictions with timestamps)
   - Update `runners` table (latest prediction only, for backward compatibility)

5. **Enhanced `analyze_predictions.py`** 📊 - Now includes odds drift analysis AND auto-fetch
   - Compares morning predictions vs pre-race predictions
   - Shows odds changes, rank changes, and accuracy by timing
   - CSV includes: `morning_odds`, `prerace_odds`, `odds_drift_pct`, `rank_change`
   - Text summary includes dedicated ODDS DRIFT ANALYSIS section
   - **NEW:** `--fetch-results` flag automatically fetches missing results from HKJC API
   - One-step workflow: fetch results + analyze in single command
   - See `ODDS_DRIFT_ANALYSIS_GUIDE.md` for detailed examples

---

**Last Updated:** November 9, 2025  
**Session Focus:** Single-race prediction script, predictions table for odds drift tracking  
**Status:** ✅ All systems operational, ready for production use

