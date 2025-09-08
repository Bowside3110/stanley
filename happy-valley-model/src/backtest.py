# src/backtest.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import build_features


# ------------------------ config ------------------------

DEFAULT_ALPHA_BLEND = float(os.environ.get("ALPHA_BLEND", "0.25"))
DEFAULT_TEST_FRAC   = float(os.environ.get("TEST_FRACTION", "0.20"))

JITTER_SEED = 123
JITTER_EPS  = 1e-12

LEAKY_COLS = {"position", "is_place"}
ID_COLS    = {"race_id", "race_date", "horse_id"}
META_EXCLUDE = ID_COLS | {
    "race_name", "race_class", "distance_bucket", "rail_tag", "going", "rail", "horse"
}


# ------------------------ helpers ------------------------

def _pick_features(df: pd.DataFrame) -> list[str]:
    feats = []
    for c in df.columns:
        if c in META_EXCLUDE or c in LEAKY_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    if not feats:
        raise RuntimeError("No numeric features found.")
    return feats


def _train_test_split_by_date(df: pd.DataFrame, test_frac: float):
    dates = np.array(sorted(df["race_date"].dropna().dt.normalize().unique()))
    cutoff_idx = int(len(dates) * (1 - test_frac)) if len(dates) > 4 else max(len(dates) - 1, 0)
    cutoff = dates[cutoff_idx]

    train_races = df[df["race_date"].dt.normalize() <= cutoff]["race_id"].unique()
    test_races  = df[df["race_date"].dt.normalize() >  cutoff]["race_id"].unique()

    df_train = df[df["race_id"].isin(train_races)].copy()
    df_test  = df[df["race_id"].isin(test_races)].copy()

    tr_rng = (df_train["race_date"].min(), df_train["race_date"].max())
    te_rng = (df_test["race_date"].min(),  df_test["race_date"].max())
    print(f"\nTraining on {df_train['race_id'].nunique()} races ({tr_rng[0].date()} → {tr_rng[1].date()})")
    print(f"Testing on  {df_test['race_id'].nunique()} races ({te_rng[0].date()} → {te_rng[1].date()})\n")
    return df_train, df_test


def _runner_market_prob(df: pd.DataFrame) -> pd.Series:
    implied = 1.0 / df["win_odds"].replace({0: np.nan})
    implied = implied.replace([np.inf, -np.inf], np.nan)

    def _norm(group):
        s = group.sum(skipna=True)
        if s and np.isfinite(s):
            return group / s
        n = group.shape[0]
        return pd.Series([1.0 / n] * n, index=group.index)

    return implied.groupby(df["race_id"]).transform(_norm).fillna(0.0)


def _pair_scores_for_race(horse_ids, probs):
    rng = np.random.default_rng(JITTER_SEED)
    s = dict(zip(horse_ids, probs))
    ids_sorted = sorted(s.keys())
    pairs = []
    for i in range(len(ids_sorted)):
        for j in range(i + 1, len(ids_sorted)):
            a, b = ids_sorted[i], ids_sorted[j]
            score = float(s[a] * s[b]) + float(rng.random() * JITTER_EPS)
            pairs.append(((a, b), score))
    pairs.sort(key=lambda t: (-t[1], t[0]))
    return pairs


def _top_k_pairs(df: pd.DataFrame, prob_col: str, k: int):
    out_rows = []
    for rid, grp in df.groupby("race_id"):
        horse_ids = grp["horse_id"].tolist()
        probs = grp[prob_col].tolist()
        scored = _pair_scores_for_race(horse_ids, probs)
        topk = [p for p, _ in scored[:k]]
        out_rows.append({"race_id": rid, "top_pairs": topk})
    return pd.DataFrame(out_rows)


def _actual_top2(df: pd.DataFrame) -> pd.DataFrame:
    gold = (
        df[pd.to_numeric(df["position"], errors="coerce").isin([1, 2])]
        .sort_values(["race_id", "position"])
        .groupby("race_id")["horse_id"]
        .apply(lambda s: tuple(sorted(s.tolist())))
        .rename("actual_top2")
        .reset_index()
    )
    return gold


def _hit_rate(top_df: pd.DataFrame, gold_df: pd.DataFrame) -> float:
    joined = top_df.merge(gold_df, on="race_id", how="inner")
    def _hit(row):
        return any(tuple(sorted(p)) == row["actual_top2"] for p in row["top_pairs"])
    return float(joined.apply(_hit, axis=1).mean())


def _random_baseline(df_test: pd.DataFrame, k: int) -> float:
    vals = []
    for _, grp in df_test.groupby("race_id"):
        n = grp.shape[0]
        total_pairs = n * (n - 1) / 2
        vals.append(min(k / total_pairs, 1.0) if total_pairs > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


# ---------------- Box-N horses ----------------

def _box_n_horses(df: pd.DataFrame, prob_col: str, n: int):
    out_rows = []
    for rid, grp in df.groupby("race_id"):
        grp_sorted = grp.sort_values(prob_col, ascending=False)
        top_horses = grp_sorted["horse_id"].head(n).tolist()
        pairs = []
        for i in range(len(top_horses)):
            for j in range(i + 1, len(top_horses)):
                pairs.append(tuple(sorted((top_horses[i], top_horses[j]))))
        out_rows.append({"race_id": rid, "box_pairs": pairs})
    return pd.DataFrame(out_rows)


# ------------------------ main ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA_BLEND,
                        help="Weight on model (vs market).")
    parser.add_argument("--test_frac", type=float, default=DEFAULT_TEST_FRAC,
                        help="Fraction of dates to reserve for test split.")
    args = parser.parse_args()

    # 1) Features & label
    df_full = build_features()
    df_full["is_place"] = (pd.to_numeric(df_full["position"], errors="coerce") <= 3).astype(int)

    # 2) Feature selection
    features = _pick_features(df_full)
    assert "is_place" not in features and "position" not in features, \
        f"Leak detected in features: {set(features) & {'is_place','position'}}"
    print(f"[Sanity] Using {len(features)} numeric features")

    # 3) Split
    df_train, df_test = _train_test_split_by_date(df_full, test_frac=args.test_frac)

    # 4) Model
    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=400,
        validation_fraction=0.1, random_state=42
    )
    model.fit(df_train[features], df_train["is_place"])

    # 5) Predict
    p_model = pd.Series(model.predict_proba(df_test[features])[:, 1],
                        index=df_test.index, name="p_model")

    if "win_odds" in df_test.columns:
        p_mkt = _runner_market_prob(df_test)
    else:
        p_mkt = df_test.groupby("race_id")["horse_id"].transform(
            lambda s: pd.Series([1.0 / len(s)] * len(s), index=s.index)
        )

    p_blend = (args.alpha * p_model) + ((1 - args.alpha) * p_mkt)
    p_blend = p_blend.groupby(df_test["race_id"]).transform(
        lambda s: s / s.sum() if s.sum() else 1.0 / len(s)
    )
    df_test = df_test.assign(p_model=p_model.values,
                             p_market=p_mkt.values,
                             p_blend=p_blend.values)

    # 6) Metrics
    gold_df = _actual_top2(df_test)
    for k in [1, 2, 3, 4]:
        top_df = _top_k_pairs(df_test, prob_col="p_blend", k=k)
        hit = _hit_rate(top_df, gold_df)
        rnd = _random_baseline(df_test, k)
        lift = (hit / rnd) if rnd > 0 else float("inf")
        print(f"Model (α={args.alpha:.2f}) — top-{k} quinella hit rate: {hit*100:.2f}% "
              f"(random: {rnd*100:.2f}%, lift: {lift:.1f}×)")

    # Box-4 and Box-5 strike rates
    for n in [4, 5]:
        box_df = _box_n_horses(df_test, prob_col="p_blend", n=n)
        joined = box_df.merge(gold_df, on="race_id", how="inner")
        hits = joined.apply(lambda row: row["actual_top2"] in row["box_pairs"], axis=1)
        print(f"Model (α={args.alpha:.2f}) — box-{n} horses quinella hit rate: {hits.mean()*100:.2f}%")


    # 7) Sample output
    merged = (
        _top_k_pairs(df_test, prob_col="p_blend", k=3)
        .merge(gold_df, on="race_id", how="inner")
        .merge(df_test[["race_id", "race_date"]].drop_duplicates(),
               on="race_id", how="left")
        .sort_values("race_date")
    )
    print("\nSamples:")
    print(merged.head(10)[["race_date", "race_id", "top_pairs", "actual_top2"]])
