# scripts/box5_by_class.py
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import build_features
from src.backtest import (
    _train_test_split_by_date,
    _pick_features,
    _actual_top2,
    _top_k_pairs,
    _box_n_horses,
    _runner_market_prob,
)

DB_PATH = "data/historical/hkjc.db"
ALPHA = 0.25
TEST_FRAC = 0.2


def load_dividends():
    """Try to load quinella dividends if present; else return empty DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql("SELECT race_id, quinella_dividend FROM results", conn)
    except Exception:
        # Column not available
        return pd.DataFrame(columns=["race_id", "quinella_dividend"])
    finally:
        conn.close()


def main():
    # 1. Features + label
    df = build_features(DB_PATH)
    df["is_place"] = (pd.to_numeric(df["position"], errors="coerce") <= 3).astype(int)

    # 2. Select features
    features = _pick_features(df)

    # 3. Split
    df_train, df_test = _train_test_split_by_date(df, test_frac=TEST_FRAC)

    # 4. Train model
    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=400,
        validation_fraction=0.1, random_state=42
    )
    model.fit(df_train[features], df_train["is_place"])

    # 5. Predict
    p_model = pd.Series(model.predict_proba(df_test[features])[:, 1],
                        index=df_test.index, name="p_model")
    p_mkt = _runner_market_prob(df_test)
    p_blend = (ALPHA * p_model) + ((1 - ALPHA) * p_mkt)
    p_blend = p_blend.groupby(df_test["race_id"]).transform(
        lambda s: s / s.sum() if s.sum() else 1.0 / len(s)
    )
    df_test = df_test.assign(p_blend=p_blend.values)

    # 6. Gold outcomes
    gold_df = _actual_top2(df_test)

    # 7. Per-race metrics
    per_race = df_test[["race_id", "race_class"]].drop_duplicates()

    for k in [1, 2, 3, 4]:
        top_df = _top_k_pairs(df_test, prob_col="p_blend", k=k)
        joined = top_df.merge(gold_df, on="race_id", how="inner")
        joined["hit"] = joined.apply(
            lambda row: row["actual_top2"] in row["top_pairs"], axis=1
        )
        per_race = per_race.merge(
            joined[["race_id", "hit"]].rename(columns={"hit": f"top{k}_hit"}),
            on="race_id", how="left"
        )

    box5_df = _box_n_horses(df_test, prob_col="p_blend", n=5)
    joined = box5_df.merge(gold_df, on="race_id", how="inner")
    joined["hit"] = joined.apply(
        lambda row: row["actual_top2"] in row["box_pairs"], axis=1
    )
    per_race = per_race.merge(
        joined[["race_id", "hit"]].rename(columns={"hit": "box5_hit"}),
        on="race_id", how="left"
    )

    # 8. Try dividends (skip if missing)
    divs = load_dividends()
    if not divs.empty:
        per_race = per_race.merge(divs, on="race_id", how="left")
    else:
        per_race["quinella_dividend"] = np.nan

    # 9. Group by class
    summary = (
        per_race.groupby("race_class")
        .agg(
            races=("race_id", "count"),
            top1_hit=("top1_hit", "mean"),
            top2_hit=("top2_hit", "mean"),
            top3_hit=("top3_hit", "mean"),
            top4_hit=("top4_hit", "mean"),
            box5_hit=("box5_hit", "mean"),
        )
        .reset_index()
    )

    print("\nPer-class quinella strike rates (test set):\n")
    print(summary.sort_values("race_class"))


if __name__ == "__main__":
    main()
