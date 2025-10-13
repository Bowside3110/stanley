# src/backtest.py
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from src.features import build_features, _pick_features


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

    # pick numeric runner-level features (centralized)
    runner_feats = _pick_features(df_full)
    print(f"\n[Sanity] Using {len(runner_feats)} numeric features:")
    print(runner_feats)

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

   # 5b. Feature importance
    print("\n=== Feature importance (by predictive value) ===")
    result = permutation_importance(model, X_tr, y_tr, n_repeats=5, random_state=42, n_jobs=1)
    feat_imp = pd.DataFrame({
        "feature": X_tr.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    print(feat_imp.to_string(index=False))

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

    # 7. Assessment by race class
    print("\n=== Prediction quality by race class ===")
    test_with_class = df_test[["race_id", "race_class"]].drop_duplicates()
    dir_df_with_class = dir_df.merge(test_with_class, on="race_id", how="left")
    gold_df_with_class = gold_df.merge(test_with_class, on="race_id", how="left")

    for cls, cls_pairs in dir_df_with_class.groupby("race_class"):
        print(f"\nRace class: {cls}")
        cls_gold = gold_df_with_class[gold_df_with_class["race_class"] == cls]

        for k in [1, 2, 3]:
            top_df = (
                cls_pairs.sort_values(["race_id", "p_pair"], ascending=[True, False])
                        .groupby("race_id").head(k)
                        .groupby("race_id")["pair"].apply(list).reset_index(name="top_pairs")
            )
            hit = _hit_rate(top_df, cls_gold)
            rnd = _random_baseline(
                df_test[df_test["race_class"] == cls], k
            )
            lift = (hit / rnd) if rnd > 0 else float("inf")
            print(f"  Top-{k} hit rate: {hit*100:.2f}% "
                  f"(random: {rnd*100:.2f}%, lift: {lift:.1f}×)")

        # Optional: box-4/5 per class
        for n in [4, 5]:
            out_rows = []
            for rid, g in cls_pairs.groupby("race_id"):
                scores = {}
                for _, r in g.iterrows():
                    h1, h2 = r["pair"]
                    scores[h1] = scores.get(h1, 0) + r["p_pair"]
                    scores[h2] = scores.get(h2, 0) + r["p_pair"]
                top_horses = sorted(scores, key=scores.get, reverse=True)[:n]
                sel_pairs = [tuple(sorted(p)) for p in combinations(top_horses, 2)]
                out_rows.append({"race_id": rid, "box_pairs": sel_pairs})
            box_df = pd.DataFrame(out_rows)
            joined = box_df.merge(cls_gold, on="race_id", how="inner")
            hits = joined.apply(lambda row: row["actual_top2"] in row["box_pairs"], axis=1)
            print(f"  Box-{n} hit rate: {hits.mean()*100:.2f}%")

    # 8. Summary table by race class
    rows = []
    for cls, cls_pairs in dir_df_with_class.dropna(subset=["race_class"]).groupby("race_class"):
        cls_gold = gold_df_with_class[gold_df_with_class["race_class"] == cls]

        # top-1/2/3
        for k in [1, 2, 3]:
            top_df = (
                cls_pairs.sort_values(["race_id", "p_pair"], ascending=[True, False])
                        .groupby("race_id").head(k)
                        .groupby("race_id")["pair"].apply(list).reset_index(name="top_pairs")
            )
            hit = _hit_rate(top_df, cls_gold)
            rnd = _random_baseline(df_test[df_test["race_class"] == cls], k)
            rows.append({
                "race_class": cls,
                "metric": f"Top-{k}",
                "hit_rate": round(hit*100, 2),
                "random": round(rnd*100, 2),
                "lift": round((hit/rnd) if rnd > 0 else float("inf"), 1)
            })

        # box-4/5
        for n in [4, 5]:
            out_rows = []
            for rid, g in cls_pairs.groupby("race_id"):
                scores = {}
                for _, r in g.iterrows():
                    h1, h2 = r["pair"]
                    scores[h1] = scores.get(h1, 0) + r["p_pair"]
                    scores[h2] = scores.get(h2, 0) + r["p_pair"]
                top_horses = sorted(scores, key=scores.get, reverse=True)[:n]
                sel_pairs = [tuple(sorted(p)) for p in combinations(top_horses, 2)]
                out_rows.append({"race_id": rid, "box_pairs": sel_pairs})
            box_df = pd.DataFrame(out_rows)
            joined = box_df.merge(cls_gold, on="race_id", how="inner")
            hits = joined.apply(lambda row: row["actual_top2"] in row["box_pairs"], axis=1)
            rows.append({
                "race_class": cls,
                "metric": f"Box-{n}",
                "hit_rate": round(hits.mean()*100, 2),
                "random": None,
                "lift": None
            })

    summary_df = pd.DataFrame(rows)
    print("\n=== Summary by race class ===")
    print(summary_df.pivot(index="race_class", columns="metric", values="hit_rate"))

    # 9. Overlay analysis: model vs market disagreement
    print("\n=== Overlay analysis (model vs market) ===")

    # Get market probabilities at runner level
    runner_probs = df_test[["race_id", "horse_id", "market_prob"]]

    # Expand to pairs, align with test_pairs
    pairs_with_market = []
    for rid, grp in df_test.groupby("race_id"):
        horses = grp["horse_id"].tolist()
        probs = dict(zip(grp["horse_id"], grp["market_prob"]))
        for i, j in combinations(horses, 2):
            pi, pj = probs.get(i, np.nan), probs.get(j, np.nan)
            if pd.notna(pi) and pd.notna(pj):
                # naive market probability = product of individual probs
                market_p = pi * pj
            else:
                market_p = np.nan
            pairs_with_market.append({"race_id": rid, "pair": tuple(sorted((i, j))), "market_p": market_p})

    market_df = pd.DataFrame(pairs_with_market)

    # Merge with model predictions
    overlay_df = dir_df.merge(market_df, on=["race_id", "pair"], how="left")

    # Compute overlay ratio (model vs market)
    overlay_df["overlay"] = overlay_df["p_pair"] / overlay_df["market_p"]

    # Attach gold outcomes
    gold_pairs = gold_df.rename(columns={"actual_top2": "gold_pair"})
    overlay_df = overlay_df.merge(gold_pairs, on="race_id", how="left")
    overlay_df["hit"] = overlay_df.apply(lambda r: r["pair"] == r["gold_pair"], axis=1)

    # Bucket by overlay size
    bins = [0, 0.5, 1.0, 1.5, 2.0, np.inf]
    labels = ["<0.5x", "0.5–1.0x", "1.0–1.5x", "1.5–2.0x", "2.0x+"]
    overlay_df["bucket"] = pd.cut(overlay_df["overlay"], bins=bins, labels=labels, include_lowest=True)

    # Evaluate hit rates by bucket
    summary = overlay_df.groupby("bucket")["hit"].mean().reset_index()
    summary["hit_rate_%"] = summary["hit"] * 100

    print(summary.to_string(index=False))
