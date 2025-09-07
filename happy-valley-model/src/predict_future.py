import os
import argparse
import pandas as pd

from src.model import prepare_dataset
from src.backtest import train_model


def predict_future(future_csv, out_dir="data/predictions", n_top=3, missing_odds_threshold=0.8):
    # Load tomorrow’s runners (already merged with race metadata)
    df_future = pd.read_csv(future_csv)

    # --- Warn if odds are missing ---
    if "odds" in df_future.columns:
        n_missing = df_future["odds"].isna().sum()
        total = len(df_future)
        if total > 0:
            frac_missing = n_missing / total
            if frac_missing >= missing_odds_threshold:
                print(f"⚠️ Warning: {n_missing}/{total} runners ({frac_missing:.0%}) have no live odds yet.")
                print("   Probabilities may be artificially low. Re-scrape closer to racetime for better predictions.\n")

    # Train model on historical dataset
    df_hist = prepare_dataset()
    model, features = train_model(df_hist)

    # Add any missing columns to future data
    for col in features:
        if col not in df_future.columns:
            df_future[col] = 0

    # Convert known numeric fields
    for col in ["horse_no", "draw", "act_wt", "odds"]:
        if col in df_future.columns:
            df_future[col] = pd.to_numeric(df_future[col], errors="coerce")

    # Predict probabilities for each runner
    X_future = df_future[features]
    df_future["prob_place"] = model.predict_proba(X_future)[:, 1]

    # Build predicted quinella pairs
    results = []
    grouped = df_future.groupby(["race_date", "race_no"])
    for (race_date, race_no), group in grouped:
        horse_nos = group["horse_no"].astype("Int64").tolist()
        probs = group["prob_place"].tolist()

        race_class = group["race_class"].iloc[0] if "race_class" in group.columns else None
        distance = group["distance"].iloc[0] if "distance" in group.columns else None
        going = group["going"].iloc[0] if "going" in group.columns else None
        course = group["course"].iloc[0] if "course" in group.columns else None

        pair_probs = []
        for i in range(len(horse_nos)):
            for j in range(i + 1, len(horse_nos)):
                pair_prob = probs[i] * probs[j]
                num_pair = tuple(sorted([horse_nos[i], horse_nos[j]]))
                pair_probs.append((num_pair, pair_prob))

        pair_probs.sort(key=lambda x: x[1], reverse=True)

        for pair, prob in pair_probs[:n_top]:
            results.append({
                "race_date": race_date,
                "race_no": race_no,
                "race_class": race_class,
                "distance": distance,
                "going": going,
                "course": course,
                "pair": "-".join(map(str, pair)),
                "predicted_prob": prob
            })

    results_df = pd.DataFrame(results)

    # Save predictions
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "future_quinella_predictions.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved predictions -> {out_path}")

    # Print per race
    print("\n=== Top Quinella Picks ===")
    for (race_date, race_no), group in results_df.groupby(["race_date", "race_no"]):
        rc = group["race_class"].iloc[0]
        dist = group["distance"].iloc[0]
        going = group["going"].iloc[0]
        course = group["course"].iloc[0]
        print(f"\nRace {race_no} ({race_date}) - Class {rc}, {dist}M, {going}, {course}")
        for _, row in group.iterrows():
            print(f"  Quinella {row['pair']} (predicted prob: {row['predicted_prob']:.3f})")

    # Per-class summary
    class_summary = results_df.groupby("race_class").agg(
        races=("race_no", "nunique"),
        avg_pred_prob=("predicted_prob", "mean"),
        max_pred_prob=("predicted_prob", "max")
    ).reset_index()

    print("\n=== Per-Class Summary (Predicted) ===")
    print(class_summary)

    return results_df, class_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict quinella picks for future races")
    parser.add_argument("--future_csv", type=str,
                        default="data/future/hkjc_future_2025-09-07_merged.csv",
                        help="Path to merged future runners CSV")
    parser.add_argument("--n_top", type=int, default=3,
                        help="Number of top quinella pairs to print per race")
    args = parser.parse_args()

    predict_future(args.future_csv, n_top=args.n_top)
