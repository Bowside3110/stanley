from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd


# ---------------- Helpers ----------------

def _read_sql(conn: sqlite3.Connection, sql: str, params: tuple | None = None) -> pd.DataFrame:
    if params is not None:
        return pd.read_sql_query(sql, conn, params=params)
    else:
        return pd.read_sql_query(sql, conn)

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None

def _parse_weight_lbs(raw) -> float:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    s = str(raw).strip()
    if "-" in s:
        try:
            a, b = s.split("-", 1)
            return int(a) * 14 + int(b)
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def _parse_frac_odds_to_decimal(s) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    st = str(s).strip().lower()
    if st in {"evs", "evens", "even"}:
        return 2.0
    if "/" in st:
        try:
            a, b = st.split("/", 1)
            return 1.0 + float(a) / float(b)
        except Exception:
            return np.nan
    try:
        return float(st)
    except Exception:
        return np.nan

# ---------------- Base pull ----------------
def _base_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = """
    SELECT
        r.race_id,
        r.date AS race_date,
        r.course,
        r.race_name,
        r.class AS race_class,
        r.going,
        r.distance AS dist_m,
        r.rail,
        run.horse_id,
        run.horse,
        run.draw,
        run.weight,
        run.win_odds,
        run.jockey_id,
        run.trainer_id,
        run.status,
        res.position
    FROM races r
    JOIN runners run 
        ON run.race_id = r.race_id
    LEFT JOIN results res 
        ON res.race_id = run.race_id
       AND res.horse_id = run.horse_id
    WHERE r.course IN ('ST', 'HV', 'Sha Tin (HK)', 'Happy Valley (HK)')
      AND r.race_id IS NOT NULL
    """
    df = _read_sql(conn, sql)

    # standard parsing
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["draw"] = pd.to_numeric(df["draw"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["weight"] = df["weight"].apply(_parse_weight_lbs)
    df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce")

    # ✅ Only filter by status for future races (no result yet)
    if "status" in df.columns:
        mask_future = df["position"].isna()
        df.loc[mask_future] = df.loc[mask_future].loc[
            df.loc[mask_future, "status"].str.lower() == "declared"
        ]

    return df


# ---------------- Racecard runner enrich ----------------

def _attach_racecard_runner_fields(conn: sqlite3.Connection, df: pd.DataFrame) -> pd.DataFrame:
    if not _table_exists(conn, "racecard_pro_runners"):
        for c in ["headgear", "headgear_run", "wind_surgery", "wind_surgery_run", "last_run", "form"]:
            df[c] = np.nan
        return df
    extra = _read_sql(conn, """
        SELECT race_id, horse_id,
               headgear, headgear_run, wind_surgery, wind_surgery_run,
               last_run, form
        FROM racecard_pro_runners
    """)
    return df.merge(extra, on=["race_id", "horse_id"], how="left")


# ---------------- Equipment flags ----------------

def _equipment_flags(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["headgear", "headgear_run", "wind_surgery", "wind_surgery_run"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    df["has_headgear"] = df.get("headgear", pd.Series([""] * len(df))).str.strip().ne("").astype(int)

    def _changed(series: pd.Series) -> pd.Series:
        s = series.fillna("").astype(str).str.upper()
        return s.str.contains(r"\b1\b|^1$|Y", regex=True).astype(int)

    df["headgear_changed"] = _changed(df.get("headgear_run", pd.Series([""] * len(df))))
    df["has_windsurg"]     = df.get("wind_surgery", pd.Series([""] * len(df))).str.strip().ne("").astype(int)
    df["windsurg_changed"] = _changed(df.get("wind_surgery_run", pd.Series([""] * len(df))))
    return df


# ---------------- Margins & Times ----------------

def _add_margins_and_times(conn, df):
    if not _table_exists(conn, "horse_results"):
        return df
    hr = _read_sql(conn, """
        SELECT horse_id, date, btn, time, dist_m, class
        FROM horse_results
        WHERE date IS NOT NULL
        ORDER BY horse_id, date
    """)
    hr["date"] = pd.to_datetime(hr["date"], errors="coerce")
    hr["btn"] = pd.to_numeric(hr["btn"], errors="coerce")
    hr["dist_m"] = pd.to_numeric(hr["dist_m"], errors="coerce")
    hr["time_sec"] = pd.to_timedelta(hr["time"], errors="coerce").dt.total_seconds()

    feats = []
    for hid, g in hr.groupby("horse_id"):
        g = g.sort_values("date")
        g["btn_last3"] = g["btn"].rolling(3, min_periods=1).mean()
        g["time_last3"] = g["time_sec"].rolling(3, min_periods=1).mean()
        g["form_close"] = (g["btn"] <= 1).astype(int).rolling(3, min_periods=1).mean()
        feats.append(g[["horse_id","date","btn_last3","time_last3","form_close"]])
    feats = pd.concat(feats)

    feats["date"] = pd.to_datetime(feats["date"], errors="coerce")
    df = df.merge(feats, left_on=["horse_id","race_date"],
                  right_on=["horse_id","date"], how="left")
    df = df.drop(columns="date", errors="ignore")
    return df


# ---------------- Class / Distance Moves ----------------

def _add_class_distance_moves(conn, df):
    if not _table_exists(conn, "horse_results"):
        return df
    hr = _read_sql(conn, """
        SELECT horse_id, date, dist_m, class
        FROM horse_results
        WHERE date IS NOT NULL
        ORDER BY horse_id, date
    """)
    hr["date"] = pd.to_datetime(hr["date"], errors="coerce")
    hr["dist_m"] = pd.to_numeric(hr["dist_m"], errors="coerce")

    feats = []
    for hid, g in hr.groupby("horse_id"):
        g = g.sort_values("date")
        g["prev_class"] = g["class"].shift(1)
        g["prev_dist_m"] = g["dist_m"].shift(1)
        g["class_move"] = (g["prev_class"] != g["class"]).astype(int)
        g["dist_delta"] = g["dist_m"] - g["prev_dist_m"]
        feats.append(g[["horse_id","date","class_move","dist_delta"]])
    feats = pd.concat(feats)

    feats["date"] = pd.to_datetime(feats["date"], errors="coerce")
    df = df.merge(feats, left_on=["horse_id","race_date"],
                  right_on=["horse_id","date"], how="left")
    df = df.drop(columns="date", errors="ignore")
    return df


# ---------------- Phase 1 Features ----------------

def _add_relative_draw_and_weight(df: pd.DataFrame) -> pd.DataFrame:
    df["field_size"] = df.groupby("race_id")["horse_id"].transform("count")
    df["rel_draw"] = df["draw"].astype(float) / df["field_size"]

    df["avg_weight"] = df.groupby("race_id")["weight"].transform("mean")
    df["rel_weight"] = df["weight"] - df["avg_weight"]
    return df

def _add_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    df["market_prob"] = 1 / df["win_odds"].replace(0, np.nan)
    df["market_logit"] = np.log(df["market_prob"] / (1 - df["market_prob"]))
    return df


# ---------------- Odds-based metrics ----------------

def horse_odds_efficiency(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the difference between actual results and market expectation for each horse.
    
    Formula: efficiency = mean(position <= 2) - mean(1/win_odds)
    
    Parameters:
        df: DataFrame with horse_id, position, and win_odds columns
        
    Returns:
        Series aligned with df index containing efficiency scores
    """
    # Ensure numeric types
    win_odds = pd.to_numeric(df['win_odds'], errors='coerce')
    position = pd.to_numeric(df['position'], errors='coerce')
    
    # Calculate market probability (1/odds)
    market_prob = 1 / win_odds.replace(0, np.nan)
    
    # Calculate actual performance (placed in top 2)
    actual_perf = position.fillna(999) <= 2
    
    # Group by horse_id and calculate efficiency - using a two-step approach to avoid warnings
    # 1. Calculate actual performance (placed in top 2) for each horse
    actual_perf_by_horse = df.groupby('horse_id')['position'].apply(
        lambda x: (x.fillna(999) <= 2).mean()
    )
    # 2. Calculate market expectation for each horse
    market_exp_by_horse = df.groupby('horse_id')['win_odds'].apply(
        lambda x: (1 / x.replace(0, np.nan)).mean()
    )
    # 3. Calculate efficiency as the difference
    horse_eff = actual_perf_by_horse - market_exp_by_horse
    
    # Map back to original DataFrame index
    result = df['horse_id'].map(horse_eff)
    
    # Log stats for debugging
    min_val = result.min()
    max_val = result.max()
    mean_val = result.mean()
    print(f"[horse_odds_efficiency] min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
    
    return result

def horse_odds_trend(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the ratio of current odds to the 3-race rolling average of win_odds.
    
    Formula: win_odds / rolling_avg_last3_odds
    
    Parameters:
        df: DataFrame with horse_id, race_date, and win_odds columns
        
    Returns:
        Series aligned with df index containing odds trend ratios
    """
    # Ensure numeric types and sort
    df_sorted = df.copy()
    df_sorted['win_odds'] = pd.to_numeric(df_sorted['win_odds'], errors='coerce')
    df_sorted = df_sorted.sort_values(['horse_id', 'race_date'])
    
    # Calculate rolling average of win_odds for each horse
    rolling_odds = df_sorted.groupby('horse_id')['win_odds'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )
    
    # Calculate the ratio (current odds / rolling average)
    # Fill missing or zero denominators with 1
    ratio = df_sorted['win_odds'] / rolling_odds.replace(0, 1).fillna(1)
    
    # Log stats for debugging
    min_val = ratio.min()
    max_val = ratio.max()
    mean_val = ratio.mean()
    print(f"[horse_odds_trend] min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
    
    return ratio

def trainer_odds_bias(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the average difference between actual outcomes and market probabilities for each trainer.
    
    Formula: bias = mean(position <= 2) - mean(1/win_odds)
    
    Parameters:
        df: DataFrame with trainer_id, position, and win_odds columns
        
    Returns:
        Series aligned with df index containing bias scores
    """
    # Ensure numeric types
    win_odds = pd.to_numeric(df['win_odds'], errors='coerce')
    position = pd.to_numeric(df['position'], errors='coerce')
    
    # Calculate market probability (1/odds)
    market_prob = 1 / win_odds.replace(0, np.nan)
    
    # Calculate actual performance (placed in top 2)
    actual_perf = position.fillna(999) <= 2
    
    # Group by trainer_id and calculate bias - using a two-step approach to avoid warnings
    # 1. Calculate actual performance (placed in top 2) for each trainer
    actual_perf_by_trainer = df.groupby('trainer_id')['position'].apply(
        lambda x: (x.fillna(999) <= 2).mean()
    )
    # 2. Calculate market expectation for each trainer
    market_exp_by_trainer = df.groupby('trainer_id')['win_odds'].apply(
        lambda x: (1 / x.replace(0, np.nan)).mean()
    )
    # 3. Calculate bias as the difference
    trainer_bias = actual_perf_by_trainer - market_exp_by_trainer
    
    # Map back to original DataFrame index
    result = df['trainer_id'].map(trainer_bias)
    
    # Log stats for debugging
    min_val = result.min()
    max_val = result.max()
    mean_val = result.mean()
    print(f"[trainer_odds_bias] min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
    
    return result

# ---------------- Rolling form ----------------

def _add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    # Horse rolling form
    df = df.sort_values(['horse_id','race_date'])
    df['horse_pos_avg3'] = (
        df.groupby('horse_id')['position']
          .transform(lambda x: x.rolling(3, 1).mean().shift(1))
    )
    df['horse_last_placed'] = (
        df.groupby('horse_id')['position']
          .shift(1)
          .between(1, 3).astype(int)
    )

    # Trainer rolling strike rate
    df = df.sort_values(['trainer_id','race_date'])
    df['trainer_win30'] = (
        df.assign(win=(df['position'] == 1).astype(int))
          .groupby('trainer_id')['win']
          .transform(lambda x: x.rolling(window=50, min_periods=5).mean().shift(1))
    )

    # Jockey rolling strike rate
    df = df.sort_values(['jockey_id','race_date'])
    df['jockey_win30'] = (
        df.assign(win=(df['position'] == 1).astype(int))
          .groupby('jockey_id')['win']
          .transform(lambda x: x.rolling(window=50, min_periods=5).mean().shift(1))
    )

    return df


# ---------------- Odds-history metrics ----------------

def _add_odds_history_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add odds-based historical metrics to the DataFrame.
    
    Adds four columns:
    - is_first_time_runner: Indicator for horses with no previous runs
    - horse_odds_efficiency: Difference between actual results and market expectation for each horse
    - horse_odds_trend: Ratio of current odds to 3-race rolling average of win_odds
    - trainer_odds_bias: Difference between actual outcomes and market probabilities for each trainer
    
    Parameters:
        df: DataFrame with required columns (horse_id, trainer_id, position, win_odds, race_date)
        
    Returns:
        DataFrame with added columns
    """
    # Add first-time runner indicator
    df['is_first_time_runner'] = df['last_run'].isna().astype(int)
    
    # Calculate horse odds efficiency
    df['horse_odds_efficiency'] = horse_odds_efficiency(df)
    
    # Calculate horse odds trend
    df['horse_odds_trend'] = horse_odds_trend(df)
    
    # Calculate trainer odds bias
    df['trainer_odds_bias'] = trainer_odds_bias(df)
    
    # Recalibrate default values for missing odds-history metrics using population statistics
    # instead of fixed values (0, 1, 0)
    df['horse_odds_efficiency'] = df['horse_odds_efficiency'].fillna(df['horse_odds_efficiency'].mean())
    df['horse_odds_trend'] = df['horse_odds_trend'].fillna(df['horse_odds_trend'].median())
    df['trainer_odds_bias'] = df['trainer_odds_bias'].fillna(df['trainer_odds_bias'].mean())
    
    # Apply a strong penalty for first-time runners (uncertainty discount)
    # This significantly reduces the model's bias toward first-time runners
    df.loc[df['is_first_time_runner'] == 1, 'horse_odds_efficiency'] -= 0.25  # Stronger penalty
    
    # Create a derived feature that explicitly penalizes first-time runners in model scoring
    # This helps the model learn that first-time runners should generally be ranked lower
    df['first_time_penalty'] = df['is_first_time_runner'] * -0.5
    
    # Print debug summary
    print("✅ Added odds-history features:")
    print(df[['is_first_time_runner', 'first_time_penalty', 'horse_odds_efficiency', 'horse_odds_trend', 'trainer_odds_bias']].describe())
    
    # Print first-time runner stats
    first_timers = df['is_first_time_runner'].sum()
    total_horses = len(df)
    print(f"First-time runners: {first_timers} ({first_timers/total_horses*100:.1f}% of total)")
    
    return df


# ---------------- Public entrypoint ----------------

def build_features(db_path: str = "data/historical/hkjc.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = _base_frame(conn)
        df = _attach_racecard_runner_fields(conn, df)
        df = _equipment_flags(df)

        df = _add_margins_and_times(conn, df)
        df = _add_class_distance_moves(conn, df)

        # --- Phase 1 new features ---
        df = _add_relative_draw_and_weight(df)
        df = _add_market_probs(df)

        # --- Phase 2 rolling form ---
        df = _add_rolling_form(df)
        
        # --- Phase 3 odds-history metrics ---
        df = _add_odds_history_metrics(df)

        df["__ord"] = df["draw"].fillna(9999)
        df = df.sort_values(["race_id", "__ord", "horse_id"]).drop(columns="__ord")

        return df
    finally:
        conn.close()


# ---------------- Feature picker ----------------

def _pick_features(df: pd.DataFrame) -> list[str]:
    base_exclude = [
        "race_id", "race_date", "race_name",
        "horse_id", "horse",
        "trainer_id", "jockey_id", "position",
        # newly excluded race metadata
        "race_class", "dist_m", "going", "rail"
    ]

    # Features we explicitly want to drop (redundant or flatlined)
    drop_features = {
        # redundant transforms
        "market_logit_min", "market_logit_max", "market_logit_diff",
        "avg_weight_min", "avg_weight_max", "avg_weight_diff",
        # sparse / noisy equipment flags
        "has_windsurg_min", "has_windsurg_max", "has_windsurg_diff",
        "windsurg_changed_min", "windsurg_changed_max", "windsurg_changed_diff",
        "headgear_changed_min", "headgear_changed_max", "headgear_changed_diff",
        "has_headgear_min", "has_headgear_max",
        # weak pair flags
        "same_trainer", "same_jockey",
        # flatlined numerics
        "time_last3_min", "time_last3_max", "time_last3_diff",
        "dist_m_min", "dist_m_max", "dist_m_diff",
        "field_size_diff",  # already captured in rel_draw
    }

    # Standard numeric selection
    feats = [
        c for c in df.columns
        if c not in base_exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and c not in drop_features
    ]

    # Always include core Phase 1 features explicitly
    must_have = [
        # Core Phase 1 features
        "rel_draw", "rel_weight", "market_prob", "market_logit",
        # Phase 3 odds-history metrics
        "horse_odds_efficiency", "horse_odds_trend", "trainer_odds_bias",
        # First-time runner features
        "is_first_time_runner", "first_time_penalty"
    ]
    for col in must_have:
        if col in df.columns and col not in feats:
            feats.append(col)

    # Debug print
    print(f"[pick_features] Selected {len(feats)} features:")
    print("   " + ", ".join(feats))

    return feats


# ---------------- Test code ----------------

if __name__ == "__main__":
    # Simple test for the odds-based metrics functions
    import pandas as pd
    import numpy as np
    
    # Create a test DataFrame
    test_data = {
        'race_id': ['R1', 'R1', 'R1', 'R2', 'R2', 'R2', 'R3', 'R3', 'R3'],
        'race_date': pd.to_datetime(['2025-01-01', '2025-01-01', '2025-01-01', 
                                     '2025-01-08', '2025-01-08', '2025-01-08',
                                     '2025-01-15', '2025-01-15', '2025-01-15']),
        'horse_id': ['H1', 'H2', 'H3', 'H1', 'H2', 'H3', 'H1', 'H2', 'H3'],
        'position': [1, 3, 2, 5, 1, 4, 2, 6, 1],
        'win_odds': [5.0, 10.0, 3.0, 4.0, 8.0, 12.0, 3.5, 15.0, 6.0],
        'trainer_id': ['T1', 'T2', 'T1', 'T1', 'T2', 'T3', 'T1', 'T2', 'T3']
    }
    
    test_df = pd.DataFrame(test_data)
    
    print("\n----- Testing odds-based metrics functions -----")
    
    # Test horse_odds_efficiency
    print("\nTesting horse_odds_efficiency:")
    horse_eff = horse_odds_efficiency(test_df)
    print(f"Horse efficiency values:\n{horse_eff.to_dict()}")
    
    # Test horse_odds_trend
    print("\nTesting horse_odds_trend:")
    horse_trend = horse_odds_trend(test_df)
    print(f"Horse odds trend values:\n{horse_trend.to_dict()}")
    
    # Test trainer_odds_bias
    print("\nTesting trainer_odds_bias:")
    trainer_bias = trainer_odds_bias(test_df)
    print(f"Trainer bias values:\n{trainer_bias.to_dict()}")
    
    # Test integration in feature building
    print("\n----- Testing integration in build_features() -----")
    try:
        print("\nTesting with small test DataFrame:")
        # Add required columns for build_features pipeline
        test_df['draw'] = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        test_df['weight'] = [120, 118, 122, 121, 119, 123, 119, 117, 121]
        test_df['jockey_id'] = ['J1', 'J2', 'J3', 'J1', 'J2', 'J3', 'J1', 'J2', 'J3']
        test_df['course'] = 'ST'
        
        # Apply odds history metrics
        result_df = _add_odds_history_metrics(test_df)
        
        # Check if columns were added
        new_cols = ['horse_odds_efficiency', 'horse_odds_trend', 'trainer_odds_bias']
        for col in new_cols:
            if col in result_df.columns:
                print(f"✓ Column '{col}' successfully added")
            else:
                print(f"✗ Column '{col}' missing!")
        
        print("\nTrying to load real data (if available):")
        # Try to load real data if available (will fail gracefully if not)
        try:
            import os
            if os.path.exists("data/historical/hkjc.db"):
                print("Real database found, testing with actual data...")
                real_df = build_features()
                
                # Check if columns were added
                for col in new_cols:
                    if col in real_df.columns:
                        print(f"✓ Column '{col}' successfully added to real data")
                        print(f"  Sample values: {real_df[col].head(3).tolist()}")
                    else:
                        print(f"✗ Column '{col}' missing from real data!")
            else:
                print("Real database not found, skipping test with actual data")
        except Exception as e:
            print(f"Error testing with real data: {e}")
            
    except Exception as e:
        print(f"Error during integration test: {e}")


