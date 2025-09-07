import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_ingestion import load_hkjc_runners, load_hkjc_dividends
from src.feature_engineer import filter_to_hk_races, add_basic_features, add_jockey_trainer_features
from src.model import get_preprocessor, prepare_dataset


def select_features(df):
    base_features = ["avg_last3", "days_since_last", "horse_no", "draw_cat", "odds", "act_wt"]
    extra_features = ["jockey_win_rate", "trainer_win_rate", "jt_combo_win_rate"]

    features = [f for f in base_features if f in df.columns]
    for col in extra_features:
        if col in df.columns and df[col].notna().any():
            features.append(col)
    return features


def train_model(df):
    features = select_features(df)
    X = df[features]
    y = df["is_place"]

    preprocessor = get_preprocessor(X)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    return model, features


def prepare_dividends():
    divs = load_hkjc_dividends().copy()
    divs["race_date"] = pd.to_datetime(divs["race_date"]).dt.strftime("%Y-%m-%d")
    divs["race_no"] = divs["race_no"].astype(int)
    divs = divs[divs["pool"].str.upper() == "QUINELLA"]
    return divs


def backtest_quinellas(df, model, dividends_df, features, n_top=3, stake_per_pair=10.0, max_races=None):
    """Backtest quinella predictions with fixed stake per pair."""
    results = []
    total_staked = 0
    total_return = 0

    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d")
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").astype("Int64")

    grouped = df.groupby(["race_date", "race_no"])
    if max_races:
        grouped = list(grouped)[:max_races]

    for (race_date, race_no), group in grouped:
        group = group.dropna(subset=["horse_no"])
        if group.empty:
            continue

        X = group[features]
        if X.empty:
            continue

        probs = model.predict_proba(X)[:, 1]
        horse_nos = pd.to_numeric(group["horse_no"], errors="coerce").dropna().astype(int).tolist()
        if len(horse_nos) < 2:
            continue

        # predicted pairs
        pair_probs = []
        for i in range(len(horse_nos)):
            for j in range(i + 1, len(horse_nos)):
                pair_prob = probs[i] * probs[j]
                num_pair = tuple(sorted([horse_nos[i], horse_nos[j]]))
                pair_probs.append((num_pair, pair_prob))
        pair_probs.sort(key=lambda x: x[1], reverse=True)

        # actual dividends
        race_divs = dividends_df[
            (dividends_df["race_date"] == race_date) &
            (dividends_df["race_no"] == race_no)
        ]

        race_staked = n_top * stake_per_pair
        race_return = 0.0
        hit = False

        for num_pair, prob in pair_probs[:n_top]:
            total_staked += stake_per_pair
            combo_str = "-".join(map(str, num_pair))
            match = race_divs[race_divs["combo"] == combo_str]
            if not match.empty:
                div_value = float(str(match["dividend"].iloc[0]).replace(",", ""))
                payout = div_value * stake_per_pair
                race_return += payout
                hit = True

        total_return += race_return
        results.append({
            "race_date": race_date,
            "race_no": race_no,
            "hit": hit,
            "staked": race_staked,
            "return": race_return,
            "net": race_return - race_staked
        })

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    print("Loading dataset...")
    df = prepare_dataset()
    dividends_df = prepare_dividends()

    # --- Merge race metadata ---
    races = pd.read_csv("data/raw/hkjc_2023_24_races.csv")

    # normalise race_date formats for merge
    df["race_date"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d")
    races["race_date"] = pd.to_datetime(races["race_date"]).dt.strftime("%Y-%m-%d")

    df = df.merge(
        races[["race_date", "race_no", "race_class"]],
        on=["race_date", "race_no"],
        how="left"
    )

    # Train model
    model, features = train_model(df)

    # Backtest
    results = backtest_quinellas(df, model, dividends_df, features, n_top=3, stake_per_pair=10.0)

    # Attach race_class to results
    race_classes = df[["race_date", "race_no", "race_class"]].drop_duplicates()
    results = results.merge(race_classes, on=["race_date", "race_no"], how="left")

    # Per-class summary
    class_summary = results.groupby("race_class").agg(
        races=("race_no", "count"),
        hits=("hit", "sum"),
        hit_rate=("hit", "mean"),
        total_staked=("staked", "sum"),
        total_return=("return", "sum")
    ).reset_index()

    class_summary["roi"] = (
        (class_summary["total_return"] - class_summary["total_staked"])
        / class_summary["total_staked"]
    )

    print("\n=== Per-class summary ===")
    print(class_summary)

    # Save detailed results
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, "quinella_backtest_results.csv"), index=False)
    class_summary.to_csv(os.path.join(out_dir, "quinella_backtest_class_summary.csv"), index=False)
    print(f"\nSaved detailed race results and per-class summary to {out_dir}")
