import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features import build_features, _pick_features
from src.backtest import _actual_top2


# ----- Build pair dataset -----
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


# ----- Main prediction -----
def predict_future(db_path: str, race_date: str, top_box: int = 5, save_csv: str | None = None):
    # 1. Load features
    df = build_features(db_path)

    # 2. Split train/future
    df_future = df[df["position"].isna()].copy()
    df_train = df[df["position"].notna()].copy()

    if race_date:
        df_future = df_future[df_future["race_date"] == race_date]

    if df_future.empty:
        print(f"No future races found for {race_date}")
        return

    # 3. Runner-level features
    runner_feats = _pick_features(df_train)

    # 4. Build pair datasets
    gold = _actual_top2(df_train)
    train_pairs = build_pair_dataset(df_train, runner_feats, gold)
    test_pairs = build_pair_dataset(df_future, runner_feats)

    # 5. Train DirectPair model
    X_tr = train_pairs.drop(columns=["race_id", "pair", "y"])
    y_tr = train_pairs["y"]
    X_te = test_pairs.drop(columns=["race_id", "pair"], errors="ignore")

    pair_model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300,
        validation_fraction=0.1, random_state=42
    )
    pair_model.fit(X_tr, y_tr)

    test_pairs = test_pairs.copy()
    test_pairs["p_raw"] = pair_model.predict_proba(X_te)[:, 1]
    dir_df = softmax_pairs(test_pairs[["race_id", "pair", "p_raw"]])

    # 6. Cheat sheet: Ranked horses by pairwise strength
    cheat_rows = []
    print("\n=== Happy Valley Quinella Cheat Sheet (DirectPair) ===")
    for rid, g in dir_df.groupby("race_id"):
        race_name = df_future[df_future["race_id"] == rid]["race_name"].iloc[0]
        scores = {}
        for _, r in g.iterrows():
            h1, h2 = r["pair"]
            scores[h1] = scores.get(h1, 0) + r["p_pair"]
            scores[h2] = scores.get(h2, 0) + r["p_pair"]
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_box]
        names = df_future[df_future["race_id"] == rid].set_index("horse_id").loc[[h for h, _ in ranked], "horse"].tolist()

        print(f"\nRace: {race_name}")
        print("Ranked selections:")
        for i, (h, score) in enumerate(ranked, 1):
            horse_name = df_future[df_future["horse_id"] == h]["horse"].iloc[0]
            print(f" {i}. {horse_name} ({score:.3f})")
            cheat_rows.append({"race_id": rid, "race_name": race_name, "rank": i, "horse": horse_name, "score": score})

    # 7. Optional CSV export
    if save_csv:
        out_df = pd.DataFrame(cheat_rows)
        out_df.to_csv(save_csv, index=False)
        print(f"\n✅ Cheat sheet saved to {save_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="data/historical/hkjc.db",
                    help="Path to SQLite database with race data")
    ap.add_argument("--date", type=str, required=True,
                    help="Race date in YYYY-MM-DD format (e.g. 2025-09-10)")
    ap.add_argument("--box", type=int, default=5,
                    help="Number of horses to include in ranking (default=5)")
    ap.add_argument("--save_csv", type=str, default=None,
                    help="Optional path to save cheat sheet as CSV")
    args = ap.parse_args()

    predict_future(args.db, args.date, top_box=args.box, save_csv=args.save_csv)