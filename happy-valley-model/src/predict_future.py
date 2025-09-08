import sqlite3
import pandas as pd
from itertools import combinations
from .model import train_model


def fetch_future_racecards(db_path="data/historical/hkjc.db", target_date=None):
    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT r.date as race_date,
           r.course,
           r.race_name,
           r.class as race_class,
           r.distance,
           r.going,
           ru.race_id,
           ru.horse_id,
           ru.horse,
           ru.draw,
           ru.weight,
           ru.jockey,
           ru.trainer,
           ru.win_odds
    FROM runners ru
    JOIN races r ON ru.race_id = r.race_id
    WHERE r.course IN ('Sha Tin (HK)', 'Happy Valley (HK)')
      {"AND r.date = '" + target_date + "'" if target_date else ""}
    """
    df_future = pd.read_sql(query, conn)
    conn.close()
    return df_future


def predict_future(db_path="data/historical/hkjc.db", target_date=None, n_top=3):
    model = train_model(db_path)
    df_future = fetch_future_racecards(db_path, target_date)

    results = []

    for race_id, group in df_future.groupby("race_id"):
        race_name = group["race_name"].iloc[0]

        X_future = group[["win_odds", "draw", "weight"]].fillna(0)
        group["place_prob"] = model.predict_proba(X_future)[:, 1]

        # Quinella pairs
        pairs = []
        for h1, h2 in combinations(group.index, 2):
            prob = group.loc[h1, "place_prob"] * group.loc[h2, "place_prob"]
            pairs.append((group.loc[h1, "horse"], group.loc[h2, "horse"], prob))

        top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:n_top]

        for horse1, horse2, prob in top_pairs:
            results.append({
                "race_id": race_id,
                "race_name": race_name,
                "pair": f"{horse1} + {horse2}",
                "predicted_prob": prob
            })

    results_df = pd.DataFrame(results)
    print("\n=== Quinella Predictions ===")
    print(results_df)
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict future quinellas from SQLite DB")
    parser.add_argument("--db", default="data/historical/hkjc.db", help="Path to SQLite DB")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD)")
    parser.add_argument("--n_top", type=int, default=3, help="Number of top quinella pairs per race")
    args = parser.parse_args()

    predict_future(db_path=args.db, target_date=args.date, n_top=args.n_top)
