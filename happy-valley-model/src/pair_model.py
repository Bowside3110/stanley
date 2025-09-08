# src/pair_model.py
import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import build_features
from src.backtest import (
    _train_test_split_by_date,
    _pick_features,
    _actual_top2,
)

# ---------------- Helpers ----------------

def build_pair_dataset(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Construct pairwise dataset with min/max/diff transforms and relational flags."""
    gold = _actual_top2(df)
    gold_map = dict(zip(gold["race_id"], gold["actual_top2"]))

    rows = []
    for rid, grp in df.groupby("race_id"):
        horse_ids = grp["horse_id"].tolist()
        g = gold_map.get(rid, None)
        for i, j in combinations(range(len(horse_ids)), 2):
            hi, hj = horse_ids[i], horse_ids[j]
            row_i, row_j = grp.iloc[i], grp.iloc[j]

            feats = {}
            for f in features:
                xi, xj = row_i[f], row_j[f]
                feats[f"{f}_min"]  = np.nanmin([xi, xj])
                feats[f"{f}_max"]  = np.nanmax([xi, xj])
                feats[f"{f}_diff"] = abs(xi - xj)

            feats["same_trainer"] = int(row_i.get("trainer_id") == row_j.get("trainer_id"))
            feats["same_jockey"]  = int(row_i.get("jockey_id") == row_j.get("jockey_id"))
            try:
                feats["draw_diff"] = abs((row_i.get("draw") or 0) - (row_j.get("draw") or 0))
            except Exception:
                feats["draw_diff"] = np.nan

            feats["race_id"] = rid
            feats["pair"]    = tuple(sorted([hi, hj]))
            feats["y"]       = int(g is not None and set([hi, hj]) == set(g))

            rows.append(feats)
    return pd.DataFrame(rows)


def top_k_pairs(pair_df: pd.DataFrame, k: int) -> pd.DataFrame:
    out_rows = []
    for rid, grp in pair_df.groupby("race_id"):
        grp_sorted = grp.sort_values("p_pair", ascending=False)
        sel = grp_sorted.head(k)["pair"].tolist()
        out_rows.append({"race_id": rid, "top_pairs": sel})
    return pd.DataFrame(out_rows)


def coverage_mass_selection(pair_df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    out_rows = []
    for rid, grp in pair_df.groupby("race_id"):
        total = grp["p_pair"].sum()
        grp_sorted = grp.sort_values("p_pair", ascending=False)
        sel, cum = [], 0
        for _, row in grp_sorted.iterrows():
            sel.append(row["pair"])
            cum += row["p_pair"]
            if total > 0 and cum / total >= threshold:
                break
        out_rows.append({"race_id": rid, "sel_pairs": sel})
    return pd.DataFrame(out_rows)


def hit_rate(pred_df: pd.DataFrame, gold_df: pd.DataFrame, col: str) -> float:
    joined = pred_df.merge(gold_df, on="race_id", how="inner")
    return joined.apply(lambda row: row["actual_top2"] in row[col], axis=1).mean()


def box_n_horses(pair_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select top-n horses by marginal 'strength' (sum of pair probs involving that horse)."""
    out_rows = []
    for rid, grp in pair_df.groupby("race_id"):
        scores = {}
        for _, row in grp.iterrows():
            h1, h2 = row["pair"]
            scores[h1] = scores.get(h1, 0) + row["p_pair"]
            scores[h2] = scores.get(h2, 0) + row["p_pair"]
        top_horses = sorted(scores, key=scores.get, reverse=True)[:n]
        pairs = [tuple(sorted(p)) for p in combinations(top_horses, 2)]
        out_rows.append({"race_id": rid, "box_pairs": pairs})
    return pd.DataFrame(out_rows)

# ---------------- Main ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--coverage", type=float, default=0.3,
                        help="Coverage mass threshold (0–1)")
    parser.add_argument("--box", type=int, nargs="*", default=[4,5],
                        help="List of N values for box-N horses (e.g. 4 5)")
    args = parser.parse_args()

    # 1) Features
    df_full = build_features()
    df_full["is_place"] = (pd.to_numeric(df_full["position"], errors="coerce") <= 3).astype(int)
    base_features = _pick_features(df_full)

    # 2) Split
    df_train, df_test = _train_test_split_by_date(df_full, test_frac=args.test_frac)

    # 3) Build pairwise dataset
    print("Building pair dataset…")
    train_pairs = build_pair_dataset(df_train, base_features)
    test_pairs  = build_pair_dataset(df_test,  base_features)

    X_train = train_pairs.drop(columns=["race_id", "pair", "y"])
    y_train = train_pairs["y"]
    X_test  = test_pairs.drop(columns=["race_id", "pair", "y"])
    y_test  = test_pairs["y"]

    # 4) Train
    print(f"Training on {len(train_pairs)} pairs")
    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300,
        validation_fraction=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    # 5) Predict
    test_pairs = test_pairs.copy()
    test_pairs["p_pair"] = model.predict_proba(X_test)[:, 1]

    # 6) Gold + race classes
    gold_df = _actual_top2(df_test)
    race_classes = df_test[["race_id", "race_class"]].drop_duplicates()

    # --- Global metrics ---
    top_df = top_k_pairs(test_pairs, k=args.top_k)
    top_hit = hit_rate(top_df, gold_df, "top_pairs")
    print(f"\nTop-{args.top_k} pair hit rate: {top_hit*100:.2f}%")

    cov_df = coverage_mass_selection(test_pairs, threshold=args.coverage)
    cov_hit = hit_rate(cov_df.rename(columns={"sel_pairs":"top_pairs"}), gold_df, "top_pairs")
    print(f"Coverage-mass ({args.coverage*100:.0f}%) hit rate: {cov_hit*100:.2f}%")

    for n in args.box:
        box_df = box_n_horses(test_pairs, n=n)
        joined = box_df.merge(gold_df, on="race_id", how="inner")
        hit = joined.apply(lambda row: row["actual_top2"] in row["box_pairs"], axis=1).mean()
        print(f"Box-{n} horses hit rate: {hit*100:.2f}%")

    # --- Per-class metrics ---
    per_race = race_classes.copy()

    # add top-k hits per race
    joined = top_df.merge(gold_df, on="race_id", how="inner")
    joined["top_hit"] = joined.apply(lambda row: row["actual_top2"] in row["top_pairs"], axis=1)
    per_race = per_race.merge(joined[["race_id","top_hit"]], on="race_id", how="left")

    # add coverage hits per race
    joined = cov_df.merge(gold_df, on="race_id", how="inner")
    joined["cov_hit"] = joined.apply(lambda row: row["actual_top2"] in row["sel_pairs"], axis=1)
    per_race = per_race.merge(joined[["race_id","cov_hit"]], on="race_id", how="left")

    # add box-N hits per race
    for n in args.box:
        box_df = box_n_horses(test_pairs, n=n)
        joined = box_df.merge(gold_df, on="race_id", how="inner")
        joined[f"box{n}_hit"] = joined.apply(lambda row: row["actual_top2"] in row["box_pairs"], axis=1)
        per_race = per_race.merge(joined[["race_id", f"box{n}_hit"]], on="race_id", how="left")

    summary = (
        per_race.groupby("race_class")
        .agg(
            races=("race_id","count"),
            top_hit=("top_hit","mean"),
            cov_hit=("cov_hit","mean"),
            **{f"box{n}_hit":(f"box{n}_hit","mean") for n in args.box}
        )
        .reset_index()
    )

    print("\nPer-class quinella hit rates (pair model):\n")
    print(summary.sort_values("race_class"))

    # --- Samples ---
    merged = (
        top_df.merge(gold_df, on="race_id", how="inner")
              .merge(df_test[["race_id", "race_date"]].drop_duplicates(),
                     on="race_id", how="left")
              .sort_values("race_date")
    )
    print("\nSamples:")
    print(merged.head(10)[["race_date","race_id","top_pairs","actual_top2"]])
