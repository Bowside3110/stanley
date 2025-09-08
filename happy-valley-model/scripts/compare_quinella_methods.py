# scripts/compare_quinella_methods.py
import argparse, os, sqlite3, numpy as np, pandas as pd
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from src.features import build_features
from src.backtest import (
    _pick_features,
    _train_test_split_by_date,
    _actual_top2,
    _runner_market_prob,
)

ALPHA = 0.25
TEST_FRAC = 0.20
SEG_COLS = ["course", "distance_bucket", "race_class"]

# ------------------- calibration helpers -------------------
def ensure_segment_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "distance_bucket" not in df.columns:
        if "distance" in df.columns:
            dist = df["distance"]
            if dist.dtype == object:
                dist_num = pd.to_numeric(dist.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
            else:
                dist_num = pd.to_numeric(dist, errors="coerce")
            bins   = [0, 1100, 1300, 1500, 1700, 10_000]
            labels = ["1000", "1200", "1400", "1600", "1800+"]
            df["distance_bucket"] = pd.cut(dist_num, bins=bins, labels=labels, include_lowest=True).astype(str)
        else:
            df["distance_bucket"] = "unknown"
    for c in ("course", "race_class"):
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    return df

def fit_segmented_isotonic(df_train: pd.DataFrame, prob_col: str, y_col: str,
                           min_pos=20, min_rows=200):
    cals = {}
    for seg, g in df_train.groupby(SEG_COLS):
        y = pd.to_numeric(g[y_col], errors="coerce")
        x = pd.to_numeric(g[prob_col], errors="coerce")
        mask = (x.notna()) & (y.notna())
        x, y = x[mask], y[mask]
        if len(x) >= min_rows and y.sum() >= min_pos:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(x.to_numpy(), y.to_numpy())
            cals[seg] = ir
    return cals

def apply_segmented_isotonic(df: pd.DataFrame, cals: dict, prob_col: str, out_col: str):
    def _one(row):
        seg = tuple(row.get(c) for c in SEG_COLS)
        p = float(row[prob_col])
        if seg in cals:
            return float(cals[seg].transform([p])[0])
        return p
    df[out_col] = df.apply(_one, axis=1)
    return df

def _safe_logit(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.log(p) - np.log(1-p)

# ------------------- helper: dividends -------------------
def load_dividends(db_path="data/historical/hkjc.db"):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("""
            SELECT race_id, tote_csf AS quinella_dividend
            FROM results
            WHERE tote_csf IS NOT NULL
        """, conn)
    except Exception:
        df = pd.DataFrame(columns=["race_id","quinella_dividend"])
    conn.close()
    df["quinella_dividend"] = pd.to_numeric(df["quinella_dividend"], errors="coerce")
    return df

# ------------------- composition methods -------------------
def harville_pairs_from_pwin(df_runners: pd.DataFrame, p_col="p_win_cal", lam=None):
    rows = []
    for rid, g in df_runners.groupby("race_id"):
        g = g[["horse_id", p_col]].dropna()
        if len(g) < 2: continue
        pmap = dict(zip(g["horse_id"], g[p_col]))
        pairs, scores = [], []
        for a, b in combinations(pmap.keys(), 2):
            pa, pb = pmap[a], pmap[b]
            if lam is None:
                s = pa * (pb/max(1e-12, 1-pa)) + pb * (pa/max(1e-12, 1-pb))
            else:
                s = (pa*pb/max(1e-12, 1-lam*pa)) + (pb*pa/max(1e-12, 1-lam*pb))
            pairs.append(tuple(sorted((a, b)))); scores.append(s)
        scores = np.array(scores, float); scores = np.clip(scores, 0, None)
        scores = scores / scores.sum() if scores.sum() > 0 else np.ones_like(scores)/len(scores)
        for pr, s in zip(pairs, scores):
            rows.append({"race_id": rid, "pair": pr, "p_pair": float(s)})
    return pd.DataFrame(rows)

def pl_pairs_from_scores(df_runners: pd.DataFrame, logit_col="win_logit"):
    rows = []
    for rid, g in df_runners.groupby("race_id"):
        g = g[["horse_id", logit_col]].dropna()
        if len(g) < 2: continue
        w = np.exp(g[logit_col].to_numpy())
        ids = g["horse_id"].tolist()
        W = w.sum()
        pairs, probs = [], []
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                pij = (w[i]/W) * (w[j]/max(1e-12, W-w[i])) + (w[j]/W) * (w[i]/max(1e-12, W-w[j]))
                pairs.append((ids[i], ids[j])); probs.append(pij)
        probs = np.array(probs); probs = probs / probs.sum()
        for pr, p in zip(pairs, probs):
            rows.append({"race_id": rid, "pair": tuple(sorted(pr)), "p_pair": float(p)})
    return pd.DataFrame(rows)

# ------------------- direct pair model -------------------
def build_pair_dataset(df: pd.DataFrame, runner_feats: list[str]) -> pd.DataFrame:
    gold = _actual_top2(df); gold_map = dict(zip(gold["race_id"], gold["actual_top2"]))
    rows = []
    for rid, grp in df.groupby("race_id"):
        horses = grp["horse_id"].tolist()
        g = gold_map.get(rid)
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
                feats[f"{f}_diff"] = abs(xi - xj) if pd.notna(xi) and pd.notna(xj) else np.nan
            feats["same_trainer"] = int(ri.get("trainer_id") == rj.get("trainer_id"))
            feats["same_jockey"]  = int(ri.get("jockey_id") == rj.get("jockey_id"))
            try:
                di = int(ri.get("draw")) if str(ri.get("draw")).isdigit() else np.nan
                dj = int(rj.get("draw")) if str(rj.get("draw")).isdigit() else np.nan
                feats["draw_diff"] = abs(di - dj) if (pd.notna(di) and pd.notna(dj)) else np.nan
            except Exception:
                feats["draw_diff"] = np.nan
            feats["race_id"] = rid
            feats["pair"] = tuple(sorted((hi, hj)))
            feats["y"] = int(g is not None and set((hi, hj)) == set(g))
            rows.append(feats)
    return pd.DataFrame(rows)

def softmax_pairs(pair_df: pd.DataFrame, score_col="p_raw") -> pd.DataFrame:
    out = []
    for rid, g in pair_df.groupby("race_id"):
        s = g[score_col].to_numpy()
        z = np.exp(s - np.max(s))
        p = z / z.sum() if z.sum() > 0 else np.ones_like(z)/len(z)
        gg = g.copy(); gg["p_pair"] = p
        out.append(gg)
    return pd.concat(out, ignore_index=True)

# ------------------- evaluation helpers -------------------
def top_k_hit(pair_df: pd.DataFrame, gold_df: pd.DataFrame, k=3) -> float:
    out = []
    for rid, g in pair_df.groupby("race_id"):
        sel = g.sort_values("p_pair", ascending=False).head(k)["pair"].tolist()
        out.append({"race_id": rid, "sel": sel})
    sel_df = pd.DataFrame(out)
    j = sel_df.merge(gold_df, on="race_id", how="inner")
    return j.apply(lambda r: r["actual_top2"] in r["sel"], axis=1).mean()

def mass_coverage_hit(pair_df: pd.DataFrame, gold_df: pd.DataFrame, threshold=0.30) -> tuple[float, float]:
    out = []
    for rid, g in pair_df.groupby("race_id"):
        gg = g.sort_values("p_pair", ascending=False)
        sel, cum = [], 0.0
        for _, r in gg.iterrows():
            sel.append(r["pair"]); cum += r["p_pair"]
            if cum >= threshold: break
        out.append({"race_id": rid, "sel": sel})
    sel_df = pd.DataFrame(out)
    j = sel_df.merge(gold_df, on="race_id", how="inner")
    hit = j.apply(lambda r: r["actual_top2"] in r["sel"], axis=1).mean()
    return hit, sel_df["sel"].str.len().mean() if len(sel_df) else np.nan

def box_n_hit(pair_df, gold_df, df_test, n=4):
    out = []
    for rid, g in pair_df.groupby("race_id"):
        scores = {}
        for _, r in g.iterrows():
            h1, h2 = r["pair"]
            scores[h1] = scores.get(h1, 0) + r["p_pair"]
            scores[h2] = scores.get(h2, 0) + r["p_pair"]
        top_horses = sorted(scores, key=scores.get, reverse=True)[:n]
        sel_pairs = [tuple(sorted(p)) for p in combinations(top_horses, 2)]
        out.append({"race_id": rid, "sel_pairs": sel_pairs})
    df_out = pd.DataFrame(out)
    j = df_out.merge(gold_df, on="race_id", how="inner")
    j["box_hit"] = j.apply(lambda r: r["actual_top2"] in r["sel_pairs"], axis=1)
    return j[["race_id","box_hit"]]

def per_class_hits(method_name, pdf, gold_df, df_test, k, coverage, divs=None):
    if pdf.empty:
        return pd.DataFrame()

    # top-k
    top = []
    for rid, g in pdf.groupby("race_id"):
        sel = g.sort_values("p_pair", ascending=False).head(k)["pair"].tolist()
        top.append({"race_id": rid, "sel": sel})
    top_df = pd.DataFrame(top).merge(gold_df, on="race_id", how="inner")
    top_df["top_hit"] = top_df.apply(lambda r: r["actual_top2"] in r["sel"], axis=1)

    # coverage
    cov = []
    for rid, g in pdf.groupby("race_id"):
        gg = g.sort_values("p_pair", ascending=False)
        sel, cum = [], 0.0
        for _, r in gg.iterrows():
            sel.append(r["pair"]); cum += r["p_pair"]
            if cum >= coverage: break
        cov.append({"race_id": rid, "sel": sel})
    cov_df = pd.DataFrame(cov).merge(gold_df, on="race_id", how="inner")
    cov_df["cov_hit"] = cov_df.apply(lambda r: r["actual_top2"] in r["sel"], axis=1)

    # box-4/5
    box4 = box_n_hit(pdf, gold_df, df_test, n=4).rename(columns={"box_hit":"box4_hit"})
    box5 = box_n_hit(pdf, gold_df, df_test, n=5).rename(columns={"box_hit":"box5_hit"})

    merged = df_test[["race_id","race_class"]].drop_duplicates()
    merged = merged.merge(top_df[["race_id","top_hit"]], on="race_id", how="left")
    merged = merged.merge(cov_df[["race_id","cov_hit"]], on="race_id", how="left")
    merged = merged.merge(box4, on="race_id", how="left")
    merged = merged.merge(box5, on="race_id", how="left")

    # attach dividends if present
    if divs is not None and not divs.empty:
        merged = merged.merge(divs, on="race_id", how="left")
        for col in ["top_hit","cov_hit","box4_hit","box5_hit"]:
            if col in merged:
                merged[f"{col}_roi"] = np.where(merged[col]==1, merged["quinella_dividend"], 0) - 1.0

    # dynamic aggregation depending on columns present
    agg_dict = {
        "races": ("race_id","count"),
        "top_hit": ("top_hit","mean"),
        "cov_hit": ("cov_hit","mean"),
        "box4_hit": ("box4_hit","mean"),
        "box5_hit": ("box5_hit","mean"),
    }
    for roi_col in ["top_hit_roi","cov_hit_roi","box4_hit_roi","box5_hit_roi"]:
        if roi_col in merged.columns:
            agg_dict[roi_col] = (roi_col,"mean")

    summary = merged.groupby("race_class").agg(**agg_dict).reset_index()
    summary["method"] = method_name
    return summary

# ------------------- main -------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--test_frac", type=float, default=TEST_FRAC)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--coverage", type=float, default=0.30)
    ap.add_argument("--henery_lambda", type=float, default=0.2)
    ap.add_argument("--save_dir", type=str, default="")
    args = ap.parse_args()

    df = build_features("data/historical/hkjc.db")
    df["is_win"] = (pd.to_numeric(df["position"], errors="coerce") == 1).astype(int)
    runner_feats = _pick_features(df)

    df_train, df_test = _train_test_split_by_date(df, test_frac=args.test_frac)
    df_train = ensure_segment_cols(df_train)
    df_test  = ensure_segment_cols(df_test)
    gold = _actual_top2(df_test)

    # win model
    win_model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=400,
                                               validation_fraction=0.1, random_state=42)
    win_model.fit(df_train[runner_feats], df_train["is_win"])

    # calibration
    p_win_model_tr = pd.Series(win_model.predict_proba(df_train[runner_feats])[:,1], index=df_train.index)
    p_win_mkt_tr   = _runner_market_prob(df_train)
    p_win_blend_tr = (args.alpha * p_win_model_tr) + ((1-args.alpha) * p_win_mkt_tr)
    p_win_blend_tr = p_win_blend_tr.groupby(df_train["race_id"]).transform(lambda s: s/s.sum() if s.sum() else 1/len(s))
    cal_df = df_train.copy(); cal_df["p_win_blend"] = p_win_blend_tr
    cal_map = fit_segmented_isotonic(cal_df, prob_col="p_win_blend", y_col="is_win", min_pos=10, min_rows=150)

    # test probs
    p_win_model_te = pd.Series(win_model.predict_proba(df_test[runner_feats])[:,1], index=df_test.index)
    p_win_mkt_te   = _runner_market_prob(df_test)
    p_win_blend_te = (args.alpha * p_win_model_te) + ((1-args.alpha) * p_win_mkt_te)
    p_win_blend_te = p_win_blend_te.groupby(df_test["race_id"]).transform(lambda s: s/s.sum() if s.sum() else 1/len(s))
    df_test = df_test.assign(p_win=p_win_blend_te.values)
    df_test = apply_segmented_isotonic(df_test, cal_map, prob_col="p_win", out_col="p_win_cal")
    df_test["win_logit"] = _safe_logit(df_test["p_win_cal"].to_numpy())

    # methods
    har_df = harville_pairs_from_pwin(df_test, p_col="p_win_cal", lam=None)
    hen_df = harville_pairs_from_pwin(df_test, p_col="p_win_cal", lam=args.henery_lambda)
    pl_df  = pl_pairs_from_scores(df_test, logit_col="win_logit")

    train_pairs = build_pair_dataset(df_train, runner_feats)
    test_pairs  = build_pair_dataset(df_test,  runner_feats)
    X_tr = train_pairs.drop(columns=["race_id","pair","y"]); y_tr = train_pairs["y"]
    X_te = test_pairs.drop(columns=["race_id","pair","y"])
    pair_model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                                validation_fraction=0.1, random_state=42)
    pair_model.fit(X_tr, y_tr)
    test_pairs = test_pairs.copy()
    test_pairs["p_raw"] = pair_model.predict_proba(X_te)[:,1]
    dir_df = softmax_pairs(test_pairs[["race_id","pair","p_raw"]])

    methods = {
        "DirectPair": dir_df,
        "Harville":   har_df,
        f"Henery(λ={args.henery_lambda:g})": hen_df,
        "PlackettLuce": pl_df,
    }

    print("\n=== Global quinella hit rates ===")
    for name, pdf in methods.items():
        if pdf.empty: continue
        top = top_k_hit(pdf, gold, k=args.k)
        cov, avg_n = mass_coverage_hit(pdf, gold, threshold=args.coverage)
        print(f"{name:14s}  top-{args.k}: {top*100:5.2f}%   mass {int(args.coverage*100)}%: {cov*100:5.2f}%")

    print("\n=== Per-class breakdowns ===")
    divs = load_dividends()
    all_summaries = []
    for name, pdf in methods.items():
        summ = per_class_hits(name, pdf, gold, df_test, args.k, args.coverage, divs)
        if not summ.empty: all_summaries.append(summ)
    if all_summaries:
        perclass = pd.concat(all_summaries, ignore_index=True)
        print(perclass.sort_values(["race_class","method"]))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for nm, pdf in methods.items():
            if not pdf.empty:
                pdf.to_csv(f"{args.save_dir.rstrip('/')}/pairs_{nm}.csv", index=False)
        df_test[["race_id","horse_id","p_win_cal","win_logit"]].to_csv(
            f"{args.save_dir.rstrip('/')}/runner_win_probs.csv", index=False
        )
