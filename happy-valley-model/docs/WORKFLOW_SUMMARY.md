# Workflow Summary - Horse Racing Predictions

**Quick Reference Guide**

---

## 🎯 The Simple Workflow

### Morning (Once)
```bash
python -m src.make_predictions
```
**Does everything:**
- ✅ Trains the model
- ✅ Saves pre-trained model for later
- ✅ Generates predictions for all races
- ✅ Saves CSV for review

**Time:** ~72 seconds

---

### Before Each Race (Fast)
```bash
python -m src.predict_next_race
```
**Fast predictions (uses pre-trained model by default):**
- ✅ Loads pre-trained model
- ✅ Fetches fresh odds
- ✅ Predicts next race only
- ✅ Saves to predictions table

**Time:** ~5 seconds ⚡

---

## 📊 What Each Script Does

| Script | Purpose | When to Use | Speed |
|--------|---------|-------------|-------|
| `python -m src.make_predictions` | Train model + predict all races | Morning (once) | ~72s |
| `python -m src.predict_next_race` | Fast single-race prediction (default) | Before each race | ~5s |
| `python -m src.predict_next_race --no-pretrained` | Train + predict single race | If you want fresh model | ~72s |
| `python -m src.train_model` | Train model only (no predictions) | Rarely needed | ~60s |
| `python -m src.analyze_predictions` | Analyze accuracy after races | After races complete | ~5s |

---

## 🔄 Complete Daily Workflow

### 1. Morning Setup (10:00 AM)
```bash
# One command does everything
python -m src.make_predictions
```

**Output:**
- `predictions_2025-11-09.csv` - All race predictions
- `pretrained_model_20251109_100530.pkl` - Saved model
- `latest_model.pkl` - Symlink to model

**Review the CSV and share with others**

---

### 2. Before Race 1 (2:45 PM - 15 mins before)
```bash
# Fast prediction with fresh odds (uses pre-trained model automatically)
python -m src.predict_next_race
```

**Output:**
```
🏇 Next Race Found:
   Race 1: L'OREAL PARIS HANDICAP
   ⏰ Time until race: 15 minutes

🏆 NEXT RACE PREDICTIONS
1. PERFECT PEACH (3.2 odds, 14.87%)
2. ONLY U (7.0 odds, 14.58%)
3. CELESTIAL HARMONY (10.0 odds, 14.53%)
```

**Time:** ~5 seconds
**Odds freshness:** 5 seconds old

---

### 3. Before Race 2 (3:15 PM)
```bash
# Run again for next race
python -m src.predict_next_race
```

**Repeat for each race throughout the day**

---

### 4. After All Races Complete (6:00 PM)
```bash
# Fetch results and analyze
python -m src.make_predictions  # Updates results
python -m src.analyze_predictions  # Analyzes accuracy
```

**Output:**
- `analysis_2025-11-09.csv` - Detailed analysis
- `analysis_2025-11-09_summary.txt` - Summary

---

## 💡 Key Benefits

### Morning Predictions (make_predictions.py)
- ✅ Overview of all races
- ✅ Early odds (may drift later)
- ✅ Creates pre-trained model
- ✅ Good for planning

### Pre-Race Predictions (predict_next_race.py)
- ✅ Fresh odds (5 seconds old)
- ✅ Fast (~5 seconds)
- ✅ Can run multiple times
- ✅ Perfect for live betting

---

## 🎓 Advanced Usage

### Compare Odds Drift
Run predictions multiple times and compare:

```bash
# Morning
python -m src.make_predictions

# 2 hours before race
python -m src.predict_next_race

# 15 mins before race
python -m src.predict_next_race
```

Then query the `predictions` table to see how odds changes affected predictions.

---

### Retrain Model Mid-Day
If you want to include today's completed races:

```bash
# Option 1: Retrain + predict all remaining races
python -m src.make_predictions

# Option 2: Just retrain (no predictions)
python -m src.train_model

# Then use new model
python -m src.predict_next_race
```

---

### Skip Fetching (Testing)
If races already happened or you're testing:

```bash
python -m src.predict_next_race --skip-fetch
```

---

### Save Predictions CSV
Keep a copy of single-race predictions:

```bash
python -m src.predict_next_race --save-csv
```

---

## 🔍 Troubleshooting

### "Pre-trained model not found"
**Solution:** Run `make_predictions.py` first to create the model.

### Predictions seem outdated
**Solution:** Run `make_predictions.py` to retrain with latest data.

### Want to force fresh model training
**Solution:** Use `predict_next_race.py --no-pretrained` to train a fresh model.

### Terminal hanging
**Solution:** Try a fresh terminal session. The venv activation might be stuck.

---

## 📈 Performance Comparison

| Scenario | Method | Time | Odds Age |
|----------|--------|------|----------|
| **Morning overview** | `python -m src.make_predictions` | 72s | Varies |
| **Pre-race (slow)** | `python -m src.predict_next_race --no-pretrained` | 72s | 72s old |
| **Pre-race (fast, default)** | `python -m src.predict_next_race` | 5s | 5s old ⚡ |

**Result:** 93% faster, 14x fresher odds!

---

## 📚 Documentation

- `HANDOFF_DOCUMENT.md` - Complete system overview
- `FAST_PREDICTIONS_GUIDE.md` - Detailed guide to pre-trained models
- `PREDICTIONS_TABLE_GUIDE.md` - How to query and analyze predictions
- `WORKFLOW_SUMMARY.md` - This file (quick reference)

---

**Last Updated:** November 9, 2025  
**Status:** ✅ Production ready  
**Note:** Pre-trained model is used by default for fast predictions

