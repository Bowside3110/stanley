import sqlite3
import pandas as pd
import numpy as np


def build_features(db_path="data/historical/hkjc.db"):
    conn = sqlite3.connect(db_path)

    # Base: runners joined with race info
    query = """
    SELECT r.date as race_date,
           r.race_id,
           r.course,
           r.race_name,
           r.class as race_class,
           r.distance,
           r.going,
           ru.horse_id,
           ru.horse,
           ru.draw,
           ru.weight,
           ru.jockey,
           ru.jockey_id,
           ru.trainer,
           ru.trainer_id,
           ru.win_odds,
           re.position
    FROM runners ru
    JOIN races r ON ru.race_id = r.race_id
    JOIN results re ON ru.race_id = re.race_id AND ru.horse_id = re.horse_id
    WHERE r.course IN ('Sha Tin (HK)', 'Happy Valley (HK)')
    """
    df = pd.read_sql(query, conn)

    # Horse history features
    horse_hist = pd.read_sql("SELECT * FROM horse_results", conn)
    horse_hist["date"] = pd.to_datetime(horse_hist["date"])

    agg = (horse_hist
           .groupby("horse_id")
           .agg(career_runs=("race_id", "count"),
                career_win_pct=("position", lambda x: (x == 1).mean()),
                career_place_pct=("position", lambda x: (x <= 2).mean()))
           .reset_index())
    df = df.merge(agg, on="horse_id", how="left")

    # Days since last run + avg pos last 3
    last_runs = (horse_hist
                 .sort_values("date")
                 .groupby("horse_id")
                 .tail(3))
    avg3 = (last_runs
            .groupby("horse_id")
            .agg(avg_pos_last3=("position", "mean"),
                 last_run_date=("date", "max"))
            .reset_index())
    df = df.merge(avg3, on="horse_id", how="left")
    df["days_since_last_run"] = (pd.to_datetime(df["race_date"]) - df["last_run_date"]).dt.days

    # Jockey features
    jockey_hist = pd.read_sql("SELECT * FROM jockey_results", conn)
    jockey_agg = (jockey_hist
                  .groupby("jockey_id")
                  .agg(jockey_runs=("race_id", "count"),
                       jockey_win_pct=("position", lambda x: (x == 1).mean()),
                       jockey_place_pct=("position", lambda x: (x <= 2).mean()))
                  .reset_index())
    df = df.merge(jockey_agg, on="jockey_id", how="left")

    # Trainer features
    trainer_hist = pd.read_sql("SELECT * FROM trainer_results", conn)
    trainer_agg = (trainer_hist
                   .groupby("trainer_id")
                   .agg(trainer_runs=("race_id", "count"),
                        trainer_win_pct=("position", lambda x: (x == 1).mean()),
                        trainer_place_pct=("position", lambda x: (x <= 2).mean()))
                   .reset_index())
    df = df.merge(trainer_agg, on="trainer_id", how="left")

    conn.close()

    # Clean numeric fields
    df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce")
    df["draw"] = pd.to_numeric(df["draw"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"].astype(str).str.replace("-", ""), errors="coerce")

    # Log odds
    df["log_win_odds"] = -np.log(df["win_odds"].replace(0, np.nan))

    # Distance buckets
    df["distance_bucket"] = pd.cut(df["distance"].astype(float),
                                   bins=[0, 1200, 1400, 1600, 2000, 3000],
                                   labels=["sprint", "1400", "mile", "middle", "staying"])

    # Target variable
    df["is_place"] = (df["position"].astype(float) <= 2).astype(int)

    # Jockey–trainer combo strike rates
    combos = (df.groupby(["jockey_id", "trainer_id"])
                .agg(combo_runs=("race_id", "count"),
                     combo_win_pct=("is_place", lambda x: (x == 1).mean()),
                     combo_place_pct=("is_place", "mean"))
                .reset_index())
    df = df.merge(combos, on=["jockey_id", "trainer_id"], how="left")

    # Final cleanup: silence FutureWarning about mixed dtypes
    df = df.infer_objects(copy=False)

    return df
