#!/usr/bin/env python3
"""
train_model.py

Train the prediction model on all historical data and save it to disk.
This allows for fast predictions without retraining.

NOTE: make_predictions.py now automatically trains and saves the model!
      You only need this script if you want to retrain WITHOUT generating predictions.

Usage:
    python -m src.train_model
    python -m src.train_model --output-dir data/models/pretrained

Typical workflow:
    Morning:  python -m src.make_predictions  (trains model + generates predictions)
    Pre-race: python -m src.predict_next_race --use-pretrained  (fast, ~5s)
"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import build_features, _pick_features
from src.backtest import _actual_top2

DB_PATH = "data/historical/hkjc.db"
DEFAULT_OUTPUT_DIR = Path("data/models/pretrained")


def build_pair_dataset(df: pd.DataFrame, runner_feats: list[str], gold=None):
    """Build pairwise comparison dataset from runner features."""
    rows = []
    gold_map = None
    if gold is not None:
        gold_map = dict(zip(gold["race_id"], gold["actual_top2"]))

    for rid, grp in df.groupby("race_id"):
        horses = grp["horse_id"].tolist()
        g = gold_map.get(rid) if gold_map else None
        for i, j in combinations(range(len(horses)), 2):
            hi, hj = horses[i], horses[j]
            ri, rj = grp.iloc[i], grp.iloc[j]
            feats = {}
            for f in runner_feats:
                xi, xj = ri[f], rj[f]
                if pd.notna(xi) or pd.notna(xj):
                    feats[f"{f}_min"] = np.nanmin([xi, xj])
                    feats[f"{f}_max"] = np.nanmax([xi, xj])
                else:
                    feats[f"{f}_min"] = np.nan
                    feats[f"{f}_max"] = np.nan
                feats[f"{f}_diff"] = (
                    abs(xi - xj) if pd.notna(xi) and pd.notna(xj) else np.nan
                )
            feats["same_trainer"] = int(
                ri.get("trainer_normalized", "") == rj.get("trainer_normalized", "") 
                and ri.get("trainer_normalized", "") != ""
            )
            feats["same_jockey"] = int(
                ri.get("jockey_normalized", "") == rj.get("jockey_normalized", "")
                and ri.get("jockey_normalized", "") != ""
            )
            feats["race_id"] = rid
            feats["pair"] = tuple(sorted((hi, hj)))
            if gold_map:
                feats["y"] = int(g is not None and set((hi, hj)) == set(g))
            rows.append(feats)
    return pd.DataFrame(rows)


def train_and_save_model(db_path: str, output_dir: Path):
    """
    Train the model on all historical data and save to disk.
    
    Args:
        db_path: Path to SQLite database
        output_dir: Directory to save the trained model
    """
    print("=" * 80)
    print("🎓 TRAINING PREDICTION MODEL")
    print("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load features
    print("\n[1/5] Building features from historical data...")
    df = build_features(db_path)
    print(f"    Loaded {len(df)} rows from {df['race_id'].nunique()} races")
    
    # 2. Filter to training data only (races with results)
    print("\n[2/5] Filtering to completed races...")
    df["race_date"] = pd.to_datetime(df["race_date"])
    df_train = df[df["position"].notna()].copy()
    print(f"    Training set: {df_train['race_id'].nunique()} races, {len(df_train)} runners")
    
    # 3. Select features
    print("\n[3/5] Selecting features...")
    runner_feats = _pick_features(df_train)
    print(f"    Selected {len(runner_feats)} features")
    print(f"    Features: {', '.join(runner_feats[:10])}{'...' if len(runner_feats) > 10 else ''}")
    
    # 4. Build pair dataset
    print("\n[4/5] Building pairwise comparison dataset...")
    gold = _actual_top2(df_train)
    train_pairs = build_pair_dataset(df_train, runner_feats, gold)
    print(f"    Created {len(train_pairs)} training pairs")
    
    # 5. Train model
    print("\n[5/5] Training model...")
    X_tr = train_pairs.drop(columns=["race_id", "pair", "y"])
    y_tr = train_pairs["y"]
    
    model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.05,
        max_iter=300,
        validation_fraction=0.1,
        random_state=42
    )
    
    print("    Fitting model (this may take a minute)...")
    model.fit(X_tr, y_tr)
    
    # Calculate training accuracy
    train_score = model.score(X_tr, y_tr)
    print(f"    ✅ Training accuracy: {train_score:.4f}")
    
    # 6. Save model and metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"pretrained_model_{timestamp}.pkl"
    model_path = output_dir / model_filename
    
    # Save model with metadata
    model_data = {
        "model": model,
        "runner_features": runner_feats,
        "training_date": datetime.now().isoformat(),
        "num_training_races": df_train['race_id'].nunique(),
        "num_training_pairs": len(train_pairs),
        "training_accuracy": train_score,
        "model_params": {
            "max_depth": 5,
            "learning_rate": 0.05,
            "max_iter": 300
        }
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved to: {model_path}")
    
    # Create a symlink to "latest"
    latest_link = output_dir / "latest_model.pkl"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(model_filename)
    print(f"✅ Symlink created: {latest_link} → {model_filename}")
    
    # Save summary
    summary_path = output_dir / f"model_summary_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINED MODEL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Database: {db_path}\n")
        f.write(f"Model File: {model_filename}\n\n")
        f.write(f"Training Data:\n")
        f.write(f"  - Races: {df_train['race_id'].nunique()}\n")
        f.write(f"  - Runners: {len(df_train)}\n")
        f.write(f"  - Pairs: {len(train_pairs)}\n\n")
        f.write(f"Features ({len(runner_feats)}):\n")
        for feat in runner_feats:
            f.write(f"  - {feat}\n")
        f.write(f"\nModel Parameters:\n")
        f.write(f"  - max_depth: 5\n")
        f.write(f"  - learning_rate: 0.05\n")
        f.write(f"  - max_iter: 300\n\n")
        f.write(f"Performance:\n")
        f.write(f"  - Training accuracy: {train_score:.4f}\n\n")
        f.write("=" * 80 + "\n")
        f.write("USAGE\n")
        f.write("=" * 80 + "\n\n")
        f.write("To use this pre-trained model for fast predictions:\n\n")
        f.write("  python -m src.predict_next_race --use-pretrained\n\n")
        f.write("This will load the model from disk instead of retraining,\n")
        f.write("reducing prediction time from ~72 seconds to ~5 seconds.\n\n")
    
    print(f"✅ Summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTo use this model for fast predictions, run:")
    print(f"  python -m src.predict_next_race --use-pretrained")
    print(f"\nExpected speedup: ~72s → ~5s per prediction")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train and save prediction model for fast predictions"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DB_PATH,
        help="Path to SQLite database (default: data/historical/hkjc.db)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save trained model (default: data/models/pretrained)"
    )
    args = parser.parse_args()
    
    train_and_save_model(args.db, Path(args.output_dir))


if __name__ == "__main__":
    main()

