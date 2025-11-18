# Odds Drift Analysis Guide

## Overview

The enhanced `analyze_predictions.py` script now tracks **odds drift** when multiple predictions are made for the same race. This allows you to:

1. See how odds changed between morning predictions and pre-race predictions
2. Compare prediction accuracy at different times
3. Identify horses with significant odds movement
4. Understand if fresher odds lead to better predictions

## How It Works

### Data Flow

1. **Morning Prediction** (`make_predictions.py`):
   - Runs early in the day (e.g., 10:00 AM)
   - Fetches racecards with current odds
   - Generates predictions
   - Saves to `predictions` table with timestamp and odds

2. **Pre-Race Predictions** (`predict_next_race.py`):
   - Runs close to race time (e.g., 5 minutes before)
   - Fetches latest odds
   - Generates new predictions
   - Saves to `predictions` table with new timestamp and updated odds

3. **Analysis** (`analyze_predictions.py`):
   - After races complete and results are available
   - Compares first prediction (morning) vs last prediction (pre-race)
   - Calculates odds drift and rank changes
   - Generates enhanced CSV and text reports

### Database Schema

The `predictions` table stores multiple predictions per horse:

```sql
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    predicted_rank INTEGER,
    predicted_score REAL,
    prediction_timestamp TEXT NOT NULL,
    model_version TEXT,
    win_odds_at_prediction REAL
);
```

## Enhanced CSV Output

When multiple predictions exist, the CSV includes these additional columns:

| Column | Description |
|--------|-------------|
| `num_predictions` | How many times this race was predicted |
| `morning_predicted_rank` | Rank from first prediction |
| `morning_predicted_score` | Score from first prediction |
| `morning_odds` | Odds at time of first prediction |
| `prerace_predicted_rank` | Rank from last prediction |
| `prerace_predicted_score` | Score from last prediction |
| `prerace_odds` | Odds at time of last prediction |
| `rank_change` | Change in predicted rank (positive = improved) |
| `odds_change` | Change in odds (positive = drifted out) |
| `odds_drift_pct` | Percentage change in odds |

### Example CSV Row

```csv
date,race_name,horse,predicted_rank,actual_position,num_predictions,
morning_predicted_rank,morning_odds,prerace_predicted_rank,prerace_odds,
rank_change,odds_change,odds_drift_pct

2025-11-09,SUISSE PROGRAMME HANDICAP,INCREDIBLE MOMENT,1,2,2,
2,19.0,1,16.0,
1,-3.0,-15.8
```

**Interpretation**: 
- Horse improved from rank 2 to rank 1 between predictions
- Odds shortened from 19.0 to 16.0 (-15.8% drift)
- Finished 2nd (close to prediction)

## Enhanced Text Summary

The text report includes a new **ODDS DRIFT ANALYSIS** section:

```
================================================================================
ODDS DRIFT ANALYSIS
================================================================================
Races with multiple predictions: 8/10
Average time between predictions: 4.2 hours
Average odds drift: 12.3%
Maximum odds drift: 35.2%

Accuracy by prediction timing:
  Morning predictions (10:00 AM): 60% top-3 accuracy
  Pre-race predictions (2:45 PM): 70% top-3 accuracy
  → Pre-race predictions were 10% more accurate

Horses with significant odds drift (>20%):
  • PERFECT PEACH: 3.5 → 3.2 (↓8.6%)
    Predicted winner, finished 1
  • ONLY U: 6.0 → 7.0 (↑16.7%)
    Predicted winner, finished 5
```

## Usage

### Daily Workflow

```bash
# Morning: Generate initial predictions
python -m src.make_predictions

# Before each race: Update predictions with fresh odds
python -m src.predict_next_race

# After race day: Analyze results
python -m src.analyze_predictions
```

### Interpreting Results

**Positive Indicators:**
- Pre-race predictions more accurate than morning
- Lower odds drift on successful predictions
- Rank improvements correlate with odds shortening

**Warning Signs:**
- Large odds drift (>30%) suggests market disagreement
- Predicted winners with odds drifting out often miss
- Inconsistent rank changes indicate model instability

## SQL Queries for Manual Analysis

### Find horses with biggest odds drift:

```sql
SELECT 
    r.date,
    r.race_name,
    run.horse,
    p1.win_odds_at_prediction AS morning_odds,
    p2.win_odds_at_prediction AS prerace_odds,
    ((p2.win_odds_at_prediction - p1.win_odds_at_prediction) / p1.win_odds_at_prediction * 100) AS drift_pct,
    run.position
FROM predictions p1
JOIN predictions p2 ON p1.race_id = p2.race_id AND p1.horse_id = p2.horse_id
JOIN runners run ON p1.race_id = run.race_id AND p1.horse_id = run.horse_id
JOIN races r ON p1.race_id = r.race_id
WHERE p1.prediction_timestamp < p2.prediction_timestamp
  AND ABS((p2.win_odds_at_prediction - p1.win_odds_at_prediction) / p1.win_odds_at_prediction * 100) > 20
  AND run.position IS NOT NULL
ORDER BY ABS(drift_pct) DESC;
```

### Compare accuracy by prediction timing:

```sql
-- Morning predictions
SELECT 
    COUNT(*) AS total,
    SUM(CASE WHEN p.predicted_rank = 1 AND run.position = 1 THEN 1 ELSE 0 END) AS correct
FROM predictions p
JOIN runners run ON p.race_id = run.race_id AND p.horse_id = run.horse_id
WHERE p.prediction_timestamp IN (
    SELECT MIN(prediction_timestamp)
    FROM predictions
    GROUP BY race_id, horse_id
)
AND run.position IS NOT NULL;

-- Pre-race predictions
SELECT 
    COUNT(*) AS total,
    SUM(CASE WHEN p.predicted_rank = 1 AND run.position = 1 THEN 1 ELSE 0 END) AS correct
FROM predictions p
JOIN runners run ON p.race_id = run.race_id AND p.horse_id = run.horse_id
WHERE p.prediction_timestamp IN (
    SELECT MAX(prediction_timestamp)
    FROM predictions
    GROUP BY race_id, horse_id
)
AND run.position IS NOT NULL;
```

## Limitations

1. **Requires Multiple Predictions**: Odds drift analysis only works when you've run predictions multiple times for the same race.

2. **Results Must Be Available**: The analysis script only processes completed races with position data.

3. **Backward Compatibility**: If no predictions table data exists, the script falls back to using only the `runners` table (legacy behavior).

## Troubleshooting

**Q: Why don't I see odds drift columns in my CSV?**

A: Check:
- Have you run `predict_next_race.py` multiple times for the same race?
- Are results available for that race date?
- Query: `SELECT COUNT(*) FROM predictions WHERE race_id LIKE 'RACE_20251109%' GROUP BY race_id;`

**Q: The odds drift seems too large/small**

A: This is normal! Hong Kong betting markets are highly liquid and odds can move significantly based on:
- Late betting activity
- Scratched horses
- Weather changes
- Insider information

**Q: Should I trust morning or pre-race predictions more?**

A: Generally, pre-race predictions are more accurate because:
- Fresher odds reflect more market information
- Model uses latest form/conditions
- Closer to actual race conditions

However, morning predictions can be valuable for:
- Early betting opportunities
- Identifying value before odds shorten
- Understanding market movements

## Future Enhancements

Potential additions to odds drift analysis:
- Intraday odds tracking (multiple predictions throughout the day)
- Odds velocity (rate of change)
- Market sentiment indicators
- Correlation between odds drift and track conditions
- Automated alerts for significant drift on top picks


