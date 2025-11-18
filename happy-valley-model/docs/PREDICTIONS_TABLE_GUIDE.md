# Predictions Table Guide

**Date:** November 9, 2025  
**Purpose:** Track multiple predictions per race over time to analyze odds drift and model performance

---

## 📊 Database Schema

### `predictions` Table

Stores **all predictions** ever made, allowing you to track how predictions change over time.

```sql
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    predicted_rank INTEGER,
    predicted_score REAL,
    prediction_timestamp TEXT NOT NULL,  -- ISO 8601 format
    model_version TEXT,
    win_odds_at_prediction REAL,
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);
```

**Indexes:**
- `idx_predictions_race` - Fast queries by race_id
- `idx_predictions_timestamp` - Fast queries by time
- `idx_predictions_horse` - Fast queries by race_id + horse_id

---

## 🎯 Use Cases

### 1. Track Odds Drift
Compare predictions made at different times to see how odds changes affect the model:

```sql
SELECT 
    p1.horse_id,
    r.horse,
    p1.predicted_rank as rank_2hrs_before,
    p1.win_odds_at_prediction as odds_2hrs_before,
    p2.predicted_rank as rank_10min_before,
    p2.win_odds_at_prediction as odds_10min_before,
    (p2.win_odds_at_prediction - p1.win_odds_at_prediction) as odds_change
FROM predictions p1
JOIN predictions p2 ON p1.race_id = p2.race_id AND p1.horse_id = p2.horse_id
JOIN runners r ON p1.race_id = r.race_id AND p1.horse_id = r.horse_id
WHERE p1.race_id = 'RACE_20251109_0001'
  AND p1.prediction_timestamp < p2.prediction_timestamp
ORDER BY p1.predicted_rank;
```

### 2. Compare Multiple Prediction Runs
See how your predictions evolved as you ran the script multiple times:

```sql
SELECT 
    horse_id,
    predicted_rank,
    predicted_score,
    win_odds_at_prediction,
    prediction_timestamp
FROM predictions
WHERE race_id = 'RACE_20251109_0001'
ORDER BY horse_id, prediction_timestamp;
```

### 3. Analyze Prediction Accuracy Over Time
Join with actual results to see which prediction timing was most accurate:

```sql
SELECT 
    p.prediction_timestamp,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN p.predicted_rank <= 3 AND r.position <= 3 THEN 1 ELSE 0 END) as top3_correct,
    AVG(ABS(p.predicted_rank - r.position)) as avg_rank_error
FROM predictions p
JOIN runners r ON p.race_id = r.race_id AND p.horse_id = r.horse_id
WHERE r.position IS NOT NULL
GROUP BY DATE(p.prediction_timestamp)
ORDER BY p.prediction_timestamp DESC;
```

### 4. Find Races with Significant Odds Movement
Identify races where odds changed dramatically between predictions:

```sql
SELECT 
    p1.race_id,
    COUNT(DISTINCT p1.horse_id) as horses_with_big_moves,
    AVG(ABS(p2.win_odds_at_prediction - p1.win_odds_at_prediction)) as avg_odds_change
FROM predictions p1
JOIN predictions p2 ON p1.race_id = p2.race_id 
    AND p1.horse_id = p2.horse_id 
    AND p1.prediction_timestamp < p2.prediction_timestamp
WHERE ABS(p2.win_odds_at_prediction - p1.win_odds_at_prediction) > 2.0
GROUP BY p1.race_id
HAVING COUNT(DISTINCT p1.horse_id) >= 3
ORDER BY avg_odds_change DESC;
```

---

## 🔄 How It Works

### When You Run `make_predictions.py` or `predict_next_race.py`:

1. **Predictions are generated** using the current odds from HKJC API
2. **New row inserted** into `predictions` table with:
   - Current timestamp
   - Predicted rank and score
   - Odds at prediction time
   - Model version used
3. **Runners table also updated** (for backward compatibility)
   - Only stores the LATEST prediction
   - Used by existing analysis scripts

### Multiple Runs = Multiple Records

If you run predictions multiple times for the same race:
- Each run creates NEW rows in `predictions` table
- The `runners` table is OVERWRITTEN with latest prediction
- You can compare all predictions using `prediction_timestamp`

---

## 📈 Example Workflow

### Morning Prediction (10:00 AM)
```bash
python -m src.make_predictions
```
- Fetches racecards with morning odds
- Saves predictions to `predictions` table with timestamp `2025-11-09T10:00:00`

### Pre-Race Update (2:45 PM, 15 mins before first race)
```bash
python -m src.predict_next_race
```
- Fetches latest odds (may have drifted significantly)
- Saves NEW predictions to `predictions` table with timestamp `2025-11-09T14:45:00`
- Now you have TWO sets of predictions for comparison

### After Races Complete
```bash
python -m src.analyze_predictions
```
- Compares predictions against actual results
- Can analyze which timing (morning vs pre-race) was more accurate

---

## 🔍 Querying Tips

### Get Latest Prediction for a Race
```sql
SELECT p.*, r.horse
FROM predictions p
JOIN runners r ON p.race_id = r.race_id AND p.horse_id = r.horse_id
WHERE p.race_id = 'RACE_20251109_0001'
  AND p.prediction_timestamp = (
      SELECT MAX(prediction_timestamp) 
      FROM predictions 
      WHERE race_id = p.race_id
  )
ORDER BY p.predicted_rank;
```

### Get All Predictions for a Specific Horse
```sql
SELECT 
    p.prediction_timestamp,
    p.predicted_rank,
    p.predicted_score,
    p.win_odds_at_prediction,
    r.position as actual_position
FROM predictions p
JOIN runners r ON p.race_id = r.race_id AND p.horse_id = r.horse_id
WHERE r.horse = 'PERFECT PEACH'
  AND p.race_id LIKE 'RACE_20251109%'
ORDER BY p.prediction_timestamp;
```

### Count Predictions Per Race
```sql
SELECT 
    race_id,
    COUNT(DISTINCT prediction_timestamp) as num_prediction_runs,
    MIN(prediction_timestamp) as first_prediction,
    MAX(prediction_timestamp) as last_prediction
FROM predictions
GROUP BY race_id
HAVING num_prediction_runs > 1
ORDER BY race_id DESC;
```

---

## 🛠️ Maintenance

### Clean Up Old Predictions (Optional)
If the table gets too large, you can delete old predictions:

```sql
-- Keep only the last 3 months
DELETE FROM predictions 
WHERE prediction_timestamp < DATE('now', '-3 months');
```

### Check Table Size
```sql
SELECT 
    COUNT(*) as total_predictions,
    COUNT(DISTINCT race_id) as unique_races,
    COUNT(DISTINCT DATE(prediction_timestamp)) as days_with_predictions
FROM predictions;
```

---

## 📝 Notes

1. **Backward Compatibility:** The `runners` table still has prediction fields (`predicted_rank`, `predicted_score`, etc.) for compatibility with existing scripts. These are updated with the LATEST prediction.

2. **No Duplicates in Results:** The `runners.position` field (actual race results) is never duplicated. Each race has only ONE set of results, but can have MANY predictions.

3. **Timestamp Format:** All timestamps are in ISO 8601 format (e.g., `2025-11-09T14:45:23.123456`) for easy sorting and comparison.

4. **Model Version:** The `model_version` field stores the filename of the model used (e.g., `model_2025-11-09_130958.pkl`), allowing you to compare different model versions.

---

## 🚀 Future Enhancements

Potential additions to the predictions table:
- `prediction_source` - Track whether prediction came from `make_predictions.py` or `predict_next_race.py`
- `confidence_level` - Store the confidence interpretation (Very High, High, Medium, Low)
- `notes` - Optional field for manual notes about the prediction run
- `api_response_time` - Track how long it took to fetch odds from HKJC API

---

**Last Updated:** November 9, 2025  
**Status:** ✅ Implemented and tested


