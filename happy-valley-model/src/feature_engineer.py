import pandas as pd


def coerce_numeric(df, cols):
    """Convert specified columns to numeric, setting invalid entries to NaN."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def filter_to_hk_races(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ST/HV races (if venue is present)."""
    if "venue" in df.columns:
        return df[df["venue"].isin(["ST", "HV"])].copy()
    return df.copy()


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to HKJC runners.
    - is_win / is_place
    - days_since_last
    - avg_last3
    - draw categories
    """
    df = df.copy()

    # --- Clean numeric columns ---
    numeric_cols = ["placing", "horse_no", "odds", "act_wt", "decl_wt"]
    df = coerce_numeric(df, numeric_cols)

    # --- Targets ---
    df["is_win"] = (df["placing"] == 1).astype(int)
    df["is_place"] = df["placing"].between(1, 3).astype(int)

    # --- Dates ---
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    # --- Days since last run ---
    df = df.sort_values(["horse", "race_date"])
    df["days_since_last"] = df.groupby("horse")["race_date"].diff().dt.days

    # --- Rolling average placing (last 3) ---
    df["avg_last3"] = (
        df.groupby("horse")["placing"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # --- Barrier categories ---
    df["draw_cat"] = pd.cut(
        df["horse_no"],
        bins=[0, 3, 7, 14],
        labels=["inside", "middle", "wide"]
    )

    return df


def add_jockey_trainer_features(df: pd.DataFrame, track_specific: bool = True) -> pd.DataFrame:
    """
    Add jockey and trainer strike-rate features.
    """
    df = df.copy()

    if "jockey" not in df.columns or "trainer" not in df.columns:
        df["jockey_win_rate"] = None
        df["trainer_win_rate"] = None
        df["jt_combo_win_rate"] = None
        return df

    if track_specific and "venue" in df.columns:
        # By track (ST vs HV)
        jockey_win_rate = df.groupby(["venue", "jockey"])["is_win"].mean()
        df["jockey_win_rate"] = df.set_index(["venue", "jockey"]).index.map(jockey_win_rate)

        trainer_win_rate = df.groupby(["venue", "trainer"])["is_win"].mean()
        df["trainer_win_rate"] = df.set_index(["venue", "trainer"]).index.map(trainer_win_rate)

        combo_win_rate = df.groupby(["venue", "jockey", "trainer"])["is_win"].mean()
        df["jt_combo_win_rate"] = df.set_index(["venue", "jockey", "trainer"]).index.map(combo_win_rate)
    else:
        # Global rates
        jockey_win_rate = df.groupby("jockey")["is_win"].mean()
        df["jockey_win_rate"] = df["jockey"].map(jockey_win_rate)

        trainer_win_rate = df.groupby("trainer")["is_win"].mean()
        df["trainer_win_rate"] = df["trainer"].map(trainer_win_rate)

        combo_win_rate = df.groupby(["jockey", "trainer"])["is_win"].mean()
        df["jt_combo_win_rate"] = df.set_index(["jockey", "trainer"]).index.map(combo_win_rate)

    return df
