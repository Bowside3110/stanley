# src/backtest.py
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features import build_features


# ---------------- Utilities ----------------
def _actual_top2(df: pd.DataFrame) -> pd.DataFrame:
    """Return the actual quinella pairs (top 2 finishers) for each race."""
    out = []
    for rid, g in df.groupby("race_id"):
        g = g.sort_values("position")
        top2 = g["horse_id"].head(2).tolist()
        if len(top2) == 2:
            out.append({"race_id": rid, "actual_top2": tuple(sorted(top2))})
    return pd.DataFrame(out)


def _train_test_split_by_date(df: pd.DataFrame, test_frac: float = 0.2):
    """Split by race_date (time-based split)."""
    dates = sorted(df["race_date"].unique())
    cutoff = int(len(dates) * (1 - test_frac))
    train_dates = dates[:cutoff]
    test_dates = dates[cutoff:]
    df_train = df[df["race_date"].isin(train_dates)].copy()
    df_test = df[df["race_date"].isin(test_dates)].copy()
    return df_train, df_test


# ---------------- Pair dataset helpers ----------------
def build_pair_dataset(df: pd.DataFrame, runner_feats: list[str], gold=None):
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
                xi, xj = ri.get(f, np.nan), rj.get(f, np.nan)
                if pd.notna(xi) or pd.notna(xj):
                    feats[f"{f}_min"] = np.nanmin([xi, xj])
                    feats[f"{f}_max"] = np.nanmax([xi, xj])
                else:
                    feats[f"{f}_min"] = np.nan
                    feats[f"{f}_max"] = np.nan
                feats[f"{f}_diff"] = (
                    abs(xi - xj) if pd.notna(xi) and pd.notna(xj) else np.nan
                )
            feats["same_trainer"] = int(ri.get("trainer_id") == rj.get("trainer_id"))
            feats["same_jockey"] = int(ri.get("jockey_id") == rj.get("jockey_id"))
            feats["race_id"] = rid
            feats["pair"] = tuple(sorted((hi, hj)))
            if gold_map:
                feats["y"] = int(g is not None and set((hi, hj)) == set(g))
            rows.append(feats)
    return pd.DataFrame(rows)


def softmax_pairs(pair_df: pd.DataFrame, score_col="p_raw") -> pd.DataFrame:
    out = []
    for rid, g in pair_df.groupby("race_id"):
        s = g[score_col].to_numpy()
        z = np.exp(s - np.max(s))
        p = z / z.sum() if z.sum() > 0 else np.ones_like(z) / len(z)
        gg = g.copy()
        gg["p_pair"] = p
        out.append(gg)
    return pd.concat(out, ignore_index=True)


# ---------------- Evaluation helpers ----------------
def _hit_rate(top_df: pd.DataFrame, gold_df: pd.DataFrame) -> float:
    j = top_df.merge(gold_df, on="race_id", how="inner")
    return j.apply(lambda r: r["actual_top2"] in r["top_pairs"], axis=1).mean()


def _random_baseline(df: pd.DataFrame, k: int) -> float:
    rows = []
    for rid, g in df.groupby("race_id"):
        n = len(g)
        if n < 2:
            continue
        total_pairs = n * (n - 1) // 2
        rows.append(min(1.0, k / total_pairs))
    return np.mean(rows)


# ---------------- Main ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="data/historical/hkjc.db")
    ap.add_argument("--test_frac", type=float, default=0.2)
    args = ap.parse_args()

    # 1. Build features
    df_full = build_features(args.db)

    # pick numeric runner-level features
    runner_feats = [
        c for c in df_full.columns
        if c not in ["race_id", "race_date", "race_name", "horse_id", "horse",
                     "trainer_id", "jockey_id", "position"]
           and pd.api.types.is_numeric_dtype(df_full[c])
    ]
    print(f"\n[Sanity] Using {len(runner_feats)} numeric features")

    # 2. Train/test split
    df_train, df_test = _train_test_split_by_date(df_full, test_frac=args.test_frac)
    print(f"\nTraining on {df_train['race_id'].nunique()} races "
          f"({df_train['race_date'].min().date()} → {df_train['race_date'].max().date()})")
    print(f"Testing on  {df_test['race_id'].nunique()} races "
          f"({df_test['race_date'].min().date()} → {df_test['race_date'].max().date()})")

    # 3. Build pair datasets
    gold = _actual_top2(df_train)
    train_pairs = build_pair_dataset(df_train, runner_feats, gold)
    test_pairs  = build_pair_dataset(df_test, runner_feats)

    # 4. Train DirectPair model
    X_tr = train_pairs.drop(columns=["race_id", "pair", "y"])
    y_tr = train_pairs["y"]
    X_te = test_pairs.drop(columns=["race_id", "pair"], errors="ignore")

    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300,
        validation_fraction=0.1, random_state=42
    )
    model.fit(X_tr, y_tr)

    # 5. Score test pairs
    test_pairs = test_pairs.copy()
    test_pairs["p_raw"] = model.predict_proba(X_te)[:, 1]
    dir_df = softmax_pairs(test_pairs[["race_id", "pair", "p_raw"]])

    # 6. Evaluate hit rates
    gold_df = _actual_top2(df_test)

    for k in [1, 2, 3]:
        top_df = (
            dir_df.sort_values(["race_id", "p_pair"], ascending=[True, False])
                  .groupby("race_id").head(k)
                  .groupby("race_id")["pair"].apply(list).reset_index(name="top_pairs")
        )
        hit = _hit_rate(top_df, gold_df)
        rnd = _random_baseline(df_test, k)
        lift = (hit / rnd) if rnd > 0 else float("inf")
        print(f"\nDirectPair — top-{k} quinella hit rate: {hit*100:.2f}% "
              f"(random: {rnd*100:.2f}%, lift: {lift:.1f}×)")

    # Box-4/5 evaluations
    for n in [4, 5]:
        out_rows = []
        for rid, g in dir_df.groupby("race_id"):
            scores = {}
            for _, r in g.iterrows():
                h1, h2 = r["pair"]
                scores[h1] = scores.get(h1, 0) + r["p_pair"]
                scores[h2] = scores.get(h2, 0) + r["p_pair"]
            top_horses = sorted(scores, key=scores.get, reverse=True)[:n]
            sel_pairs = [tuple(sorted(p)) for p in combinations(top_horses, 2)]
            out_rows.append({"race_id": rid, "box_pairs": sel_pairs})
        box_df = pd.DataFrame(out_rows)
        joined = box_df.merge(gold_df, on="race_id", how="inner")
        hits = joined.apply(lambda row: row["actual_top2"] in row["box_pairs"], axis=1)
        print(f"DirectPair — box-{n} horses quinella hit rate: {hits.mean()*100:.2f}%")
