import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.features import build_features


def train_model(df, features, categorical):
    X = df[features + categorical].copy()
    y = df["is_place"]

    # One-hot encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X, y)
    return model


def backtest_quinellas(df, model, features, categorical, n_top=3):
    X = df[features + categorical].copy()
    df = df.copy()
    df["prob_place"] = model.predict_proba(X)[:, 1]

    results = []
    grouped = df.groupby(["race_date", "race_id"])
    for (race_date, race_id), group in grouped:
        horses = group["horse_id"].tolist()
        probs = group["prob_place"].tolist()

        pair_probs = []
        for i in range(len(horses)):
            for j in range(i + 1, len(horses)):
                prob = probs[i] * probs[j]
                pair = tuple(sorted([horses[i], horses[j]]))
                pair_probs.append((pair, prob))

        pair_probs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = pair_probs[:n_top]

        actual_top2 = set(group.sort_values("position").head(2)["horse_id"].tolist())
        hit = any(actual_top2 == set(pair) for pair, _ in top_pairs)

        results.append({
            "race_date": race_date,
            "race_id": race_id,
            "predicted_pairs": [p for p, _ in top_pairs],
            "predicted_probs": [pr for _, pr in top_pairs],
            "actual_top2": list(actual_top2),
            "hit": hit
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Build features from DB
    df = build_features()
    df["race_date"] = pd.to_datetime(df["race_date"])

    # Chronological split
    cutoff = df["race_date"].quantile(0.8)
    train_df = df[df["race_date"] <= cutoff]
    test_df = df[df["race_date"] > cutoff]

    print(f"Training on {train_df['race_id'].nunique()} races "
          f"({train_df['race_date'].min().date()} → {train_df['race_date'].max().date()})")
    print(f"Testing on {test_df['race_id'].nunique()} races "
          f"({test_df['race_date'].min().date()} → {test_df['race_date'].max().date()})")

    # Expanded feature set
    features = [
        "win_odds", "log_win_odds", "draw", "weight",
        "days_since_last_run", "avg_pos_last3",
        "career_runs", "career_win_pct", "career_place_pct",
        "jockey_runs", "jockey_win_pct", "jockey_place_pct",
        "trainer_runs", "trainer_win_pct", "trainer_place_pct",
        "combo_runs", "combo_win_pct", "combo_place_pct"
    ]

    categorical = ["race_class", "going", "distance_bucket"]

    # Train model
    model = train_model(train_df, features, categorical)

    # Run backtest
    results = backtest_quinellas(test_df, model, features, categorical, n_top=3)

    # Summary
    hit_rate = results["hit"].mean()
    print(f"\nBacktest hit rate (top 3 quinella picks): {hit_rate:.2%}")
    print(f"Races tested: {results['race_id'].nunique()}")
    print(results.head(10))
