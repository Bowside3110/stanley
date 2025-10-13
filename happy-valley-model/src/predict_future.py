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
    print(f"Predicting races for {race_date} using database {db_path}...")

    # 1. Load features
    print("[1/6] Building features...")
    df = build_features(db_path)
    print(f"    Loaded {len(df)} rows")
    print("    Available columns:", df.columns.tolist())


    # 2. Split train/future
    print("[2/6] Splitting train/future sets...")

    df["race_date"] = pd.to_datetime(df["race_date"])

    # Train = any race with results (position not null)
    df_train = df[df["position"].notna()].copy()

    # Future = any race with no results yet
    df_future = df[df["position"].isna()].copy()

    # If a specific race date is provided, filter to that date
    if race_date:
        race_date = pd.to_datetime(race_date).date()
        df_future = df_future[df_future["race_date"].dt.date == race_date]

    if df_future.empty:
        print(f"No future races found for {race_date}")
        return

    print(f"    Train races: {df_train['race_id'].nunique()}, Future races: {df_future['race_id'].nunique()}")


    # 3. Runner-level features
    print("[3/6] Selecting features...")
    runner_feats = _pick_features(df_train)
    print(f"    Using {len(runner_feats)} features")

    # 4. Build pair datasets
    print("[4/6] Building pair datasets...")
    gold = _actual_top2(df_train)
    train_pairs = build_pair_dataset(df_train, runner_feats, gold)
    test_pairs = build_pair_dataset(df_future, runner_feats)
    print(f"    Train pairs: {len(train_pairs)}, Future pairs: {len(test_pairs)}")

    # 5. Train DirectPair model
    print("[5/6] Training model...")
    X_tr = train_pairs.drop(columns=["race_id", "pair", "y"])
    y_tr = train_pairs["y"]
    X_te = test_pairs.drop(columns=["race_id", "pair"], errors="ignore")

    pair_model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300,
        validation_fraction=0.1, random_state=42
    )
    pair_model.fit(X_tr, y_tr)

    print("    Scoring future races...")
    test_pairs = test_pairs.copy()
    test_pairs["p_raw"] = pair_model.predict_proba(X_te)[:, 1]
    dir_df = softmax_pairs(test_pairs[["race_id", "pair", "p_raw"]])

    # 6. Cheat sheet: Ranked horses by pairwise strength
    print("[6/6] Generating cheat sheet...")
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
        rows = []
        used_meta_cols = set()

        for rid, g in dir_df.groupby("race_id"):
            # collect scores for every horse
            scores = {}
            for _, r in g.iterrows():
                h1, h2 = r["pair"]
                scores[h1] = scores.get(h1, 0) + r["p_pair"]
                scores[h2] = scores.get(h2, 0) + r["p_pair"]

            # pick whatever race metadata exists in df_future
            candidate_meta_cols = [
                "race_id", "race_name", "class", "race_class",
                "distance", "dist_m", "going"
            ]
            meta_cols = [c for c in candidate_meta_cols if c in df_future.columns]
            used_meta_cols.update(meta_cols)

            if meta_cols:
                race_meta = (
                    df_future[df_future["race_id"] == rid][meta_cols]
                    .drop_duplicates()
                    .iloc[0]
                    .to_dict()
                )
            else:
                race_meta = {"race_id": rid}

            # join with runner features + add score, rank, and status
            race_df = df_future[df_future["race_id"] == rid].copy()
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            for rank, (horse_id, score) in enumerate(ranked, 1):
                runner_row = race_df[race_df["horse_id"] == horse_id].iloc[0].to_dict()
                runner_row.update(race_meta)
                runner_row["score"] = score
                runner_row["rank"] = rank
                # NEW: carry status into the output
                if "status" in runner_row:
                    runner_row["status"] = runner_row["status"]
                rows.append(runner_row)

        out_df = pd.DataFrame(rows)
        out_df = out_df.sort_values(["race_id", "rank"])
        out_df.to_csv(save_csv, index=False)

        print(f"\n✅ Detailed predictions saved to {save_csv}")
        print(f"   Included race metadata columns: {sorted(list(used_meta_cols))}")



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
