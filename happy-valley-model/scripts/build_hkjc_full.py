import os
import sqlite3
import pandas as pd

DB_PATH = "data/historical/hkjc.db"
OUT_PATH = "data/historical/hkjc_full.csv"


def build_training_set(db_path=DB_PATH, out_path=OUT_PATH):
    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        ra.date as race_date,
        ra.course,
        ra.race_id,
        ra.race_name,
        ra.class as race_class,
        ra.distance,
        ra.going,
        ra.rail,
        run.horse_id,
        run.horse,
        run.draw,
        run.weight,
        run.jockey,
        run.jockey_id,
        run.trainer,
        run.trainer_id,
        run.win_odds,
        run.age,
        run.sex,
        run.colour,
        run.form,
        run.rpr,
        run.ts,
        run.silk_url,
        res.position
    FROM races ra
    JOIN runners run ON ra.race_id = run.race_id
    LEFT JOIN results res ON run.race_id = res.race_id AND run.horse_id = res.horse_id
    WHERE ra.course LIKE '%Sha Tin%' OR ra.course LIKE '%Happy Valley%'
    ORDER BY ra.date, ra.race_id, run.horse_id
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Label: did horse place in top 2?
    df["is_place"] = df["position"].apply(
        lambda x: 1 if pd.notnull(x) and str(x).isdigit() and int(x) <= 2 else 0
    )

    # Convert odds to numeric
    df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce")

    # Assign race_no within each date+course
    df["race_no"] = df.groupby(["race_date", "course"]).cumcount() + 1

    # Save CSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Training set built with {len(df)} rows → {out_path}")

    return df


if __name__ == "__main__":
    build_training_set()
