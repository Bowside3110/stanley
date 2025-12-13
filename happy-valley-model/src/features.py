from __future__ import annotations

import sqlite3  # Keep for legacy compatibility
from src.db_config import get_connection, get_placeholder
import numpy as np
import pandas as pd
from src.horse_matcher import (
    normalize_horse_name, fuzzy_match_horse, build_horse_name_index,
    normalize_jockey_name, normalize_trainer_name
)


# ---------------- Global cache for horse matching ----------------
_HORSE_MATCH_CACHE = {}
_HORSE_NAME_INDEX = None


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
        r.post_time AS race_time,
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
        run.jockey,
        run.trainer_id,
        run.trainer,
        run.status,
        run.position
    FROM races r
    JOIN runners run 
        ON run.race_id = r.race_id
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
        
        # For future races (no position), only keep declared horses
        # Accept: "declared", "Declared", "DECLARED" (case-insensitive)
        # Exclude: "Scratched", "STOP_SELL", "REFUND_BEFORE_CLOSE", etc.
        mask_declared = df["status"].str.lower() == "declared"
        # Keep all historical races OR future declared races
        df = df[~mask_future | mask_declared].copy()

    # Add normalized names for consistent matching across different IDs
    df["horse_normalized"] = df["horse"].apply(normalize_horse_name)
    df["jockey_normalized"] = df["jockey"].apply(normalize_jockey_name)
    df["trainer_normalized"] = df["trainer"].apply(normalize_trainer_name)
    
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
    # Get data from runners table
    hr = _read_sql(conn, """
        SELECT 
            run.horse_id, 
            r.date, 
            run.btn, 
            run.time, 
            r.distance AS dist_m, 
            r.class, 
            run.horse
        FROM runners run
        JOIN races r ON run.race_id = r.race_id
        WHERE r.date IS NOT NULL
            AND run.position IS NOT NULL
        ORDER BY run.horse_id, r.date
    """)
    
    print(f"[DEBUG margins_times] Loaded {len(hr)} rows from runners table")
    
    hr["date"] = pd.to_datetime(hr["date"], errors="coerce")
    hr["btn"] = pd.to_numeric(hr["btn"], errors="coerce")
    hr["dist_m"] = pd.to_numeric(hr["dist_m"], errors="coerce")
    
    # Parse race time format "1:49.52" (minutes:seconds.centiseconds)
    def parse_race_time(time_str):
        if pd.isna(time_str) or time_str == '':
            return np.nan
        try:
            parts = str(time_str).split(":")
            if len(parts) != 2:
                return np.nan
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        except:
            return np.nan
    
    hr["time_sec"] = hr["time"].apply(parse_race_time)
    
    # Add normalized horse name for matching
    hr["horse_normalized"] = hr["horse"].apply(normalize_horse_name)
    
    print(f"[DEBUG margins_times] Unique normalized horses: {hr['horse_normalized'].nunique()}")

    feats = []
    # Group by normalized name instead of horse_id
    for norm_name, g in hr.groupby("horse_normalized"):
        if not norm_name:  # Skip empty names
            continue
        g = g.sort_values("date")
        g["btn_last3"] = g["btn"].rolling(3, min_periods=1).mean()
        g["time_last3"] = g["time_sec"].rolling(3, min_periods=1).mean()
        g["form_close"] = (g["btn"] <= 1).astype(int).rolling(3, min_periods=1).mean()
        feats.append(g[["horse_normalized","date","btn_last3","time_last3","form_close"]])
    feats = pd.concat(feats) if feats else pd.DataFrame()

    if not feats.empty:
        feats["date"] = pd.to_datetime(feats["date"], errors="coerce")
        print(f"[DEBUG margins_times] Created features for {len(feats)} horse-date combinations")
        
        # Merge by normalized horse name instead of horse_id
        before_merge = len(df)
        df = df.merge(feats, left_on=["horse_normalized","race_date"],
                      right_on=["horse_normalized","date"], how="left")
        df = df.drop(columns="date", errors="ignore")
        
        # Report coverage
        btn_coverage = df["btn_last3"].notna().sum()
        form_coverage = df["form_close"].notna().sum()
        print(f"[DEBUG margins_times] After merge: btn_last3={btn_coverage}/{before_merge} ({btn_coverage/before_merge*100:.1f}%), form_close={form_coverage}/{before_merge} ({form_coverage/before_merge*100:.1f}%)")
    else:
        print("[DEBUG margins_times] No features created (empty feats)")
        
    return df


# ---------------- Class / Distance Moves ----------------

def _add_class_distance_moves(conn, df):
    # Get data from runners table
    hr = _read_sql(conn, """
        SELECT 
            run.horse_id, 
            r.date, 
            r.distance AS dist_m, 
            r.class, 
            run.horse
        FROM runners run
        JOIN races r ON run.race_id = r.race_id
        WHERE r.date IS NOT NULL
            AND run.position IS NOT NULL
        ORDER BY run.horse_id, r.date
    """)
    
    print(f"[DEBUG class_dist] Loaded {len(hr)} rows from runners table")
    
    hr["date"] = pd.to_datetime(hr["date"], errors="coerce")
    hr["dist_m"] = pd.to_numeric(hr["dist_m"], errors="coerce")
    
    # Add normalized horse name for matching
    hr["horse_normalized"] = hr["horse"].apply(normalize_horse_name)
    
    print(f"[DEBUG class_dist] Unique normalized horses: {hr['horse_normalized'].nunique()}")

    feats = []
    # Group by normalized name instead of horse_id
    for norm_name, g in hr.groupby("horse_normalized"):
        if not norm_name:  # Skip empty names
            continue
        g = g.sort_values("date")
        g["prev_class"] = g["class"].shift(1)
        g["prev_dist_m"] = g["dist_m"].shift(1)
        g["class_move"] = (g["prev_class"] != g["class"]).astype(int)
        g["dist_delta"] = g["dist_m"] - g["prev_dist_m"]
        feats.append(g[["horse_normalized","date","class_move","dist_delta"]])
    feats = pd.concat(feats) if feats else pd.DataFrame()

    if not feats.empty:
        feats["date"] = pd.to_datetime(feats["date"], errors="coerce")
        print(f"[DEBUG class_dist] Created features for {len(feats)} horse-date combinations")
        
        # Merge by normalized horse name instead of horse_id
        before_merge = len(df)
        df = df.merge(feats, left_on=["horse_normalized","race_date"],
                      right_on=["horse_normalized","date"], how="left")
        df = df.drop(columns="date", errors="ignore")
        
        # Report coverage
        class_coverage = df["class_move"].notna().sum()
        dist_coverage = df["dist_delta"].notna().sum()
        print(f"[DEBUG class_dist] After merge: class_move={class_coverage}/{before_merge} ({class_coverage/before_merge*100:.1f}%), dist_delta={dist_coverage}/{before_merge} ({dist_coverage/before_merge*100:.1f}%)")
    else:
        print("[DEBUG class_dist] No features created (empty feats)")
        
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


# ---------------- Horse Matching ----------------

def _match_horses_to_historical(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Match horses in current DataFrame to historical horses using name-based fuzzy matching.
    Adds columns: matched_historical_id (normalized name), match_confidence
    """
    global _HORSE_MATCH_CACHE, _HORSE_NAME_INDEX
    
    # Build horse name index if not already built
    if _HORSE_NAME_INDEX is None:
        _HORSE_NAME_INDEX = build_horse_name_index(conn)
    
    # Initialize columns
    df['matched_historical_id'] = None
    df['match_confidence'] = 0.0
    
    match_stats = {'exact': 0, 'fuzzy': 0, 'no_match': 0}
    
    for idx, row in df.iterrows():
        current_horse_id = row['horse_id']
        current_horse_name = row.get('horse', '')
        
        # Check cache first
        if current_horse_id in _HORSE_MATCH_CACHE:
            match_name, confidence = _HORSE_MATCH_CACHE[current_horse_id]
            df.at[idx, 'matched_historical_id'] = match_name
            df.at[idx, 'match_confidence'] = confidence
            continue
        
        # Try to match by normalizing the current horse name
        if current_horse_name:
            normalized_name = normalize_horse_name(current_horse_name)
            
            # Check if this normalized name exists in our index
            if normalized_name and normalized_name in _HORSE_NAME_INDEX:
                _HORSE_MATCH_CACHE[current_horse_id] = (normalized_name, 1.0)
                df.at[idx, 'matched_historical_id'] = normalized_name
                df.at[idx, 'match_confidence'] = 1.0
                match_stats['exact'] += 1
                continue
            
            # Fuzzy match by name
            if _HORSE_NAME_INDEX:
                match_name, confidence = fuzzy_match_horse(
                    current_horse_name, 
                    _HORSE_NAME_INDEX,
                    threshold=0.85
                )
                
                # Cache the result
                _HORSE_MATCH_CACHE[current_horse_id] = (match_name, confidence)
                df.at[idx, 'matched_historical_id'] = match_name
                df.at[idx, 'match_confidence'] = confidence
                
                if match_name:
                    if confidence >= 0.95:
                        match_stats['exact'] += 1
                    else:
                        match_stats['fuzzy'] += 1
                        print(f"  ⚠️  Fuzzy matched '{current_horse_name}' → '{match_name}' ({confidence:.2f})")
                else:
                    match_stats['no_match'] += 1
            else:
                match_stats['no_match'] += 1
        else:
            match_stats['no_match'] += 1
    
    # Print matching statistics
    total = sum(match_stats.values())
    if total > 0:
        print(f"\n✅ Horse Matching Results:")
        print(f"   Exact matches: {match_stats['exact']} ({match_stats['exact']/total*100:.1f}%)")
        print(f"   Fuzzy matches: {match_stats['fuzzy']} ({match_stats['fuzzy']/total*100:.1f}%)")
        print(f"   No matches (first-time): {match_stats['no_match']} ({match_stats['no_match']/total*100:.1f}%)")
    
    return df


# ---------------- Odds-based metrics (with matching) ----------------

def _calculate_horse_odds_efficiency(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.Series:
    """
    Calculate odds efficiency using normalized horse names and all matching IDs.
    Returns efficiency score for each horse.
    """
    global _HORSE_NAME_INDEX
    
    # Get unique normalized names for horses that have been matched
    matched_normalized_names = df[df['matched_historical_id'].notna()]['matched_historical_id'].unique().tolist()
    
    if not matched_normalized_names or conn is None or _HORSE_NAME_INDEX is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    # Collect all horse IDs that match these normalized names
    all_horse_ids = []
    for norm_name in matched_normalized_names:
        if norm_name in _HORSE_NAME_INDEX:
            all_horse_ids.extend(_HORSE_NAME_INDEX[norm_name])
    
    if not all_horse_ids:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    # Query historical performance using horse IDs
    placeholders = ','.join('?' * len(all_horse_ids))
    query = f"""
    SELECT r.horse_id, r.horse, r.position, r.win_odds
    FROM runners r
    WHERE r.horse_id IN ({placeholders})
      AND r.position IS NOT NULL
      AND r.win_odds IS NOT NULL
      AND r.win_odds > 0
    """
    
    hist_df = pd.read_sql_query(query, conn, params=all_horse_ids)
    
    print(f"\n[DEBUG efficiency] Queried {len(all_horse_ids)} horse IDs from {len(matched_normalized_names)} normalized names, got {len(hist_df)} historical rows")
    
    # Convert win_odds to numeric and handle non-numeric values
    hist_df['win_odds'] = pd.to_numeric(hist_df['win_odds'], errors='coerce')
    
    # Filter out rows where win_odds is NaN
    hist_df = hist_df[hist_df['win_odds'].notna()]
    
    if hist_df.empty:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    # Convert position to numeric, coercing non-numeric values (DSQ, PU, etc.) to NaN
    hist_df['position'] = pd.to_numeric(hist_df['position'], errors='coerce')
    
    # Filter out rows where position couldn't be converted
    hist_df = hist_df[hist_df['position'].notna()]
    
    # Check if we still have data after filtering
    if hist_df.empty:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    # Normalize horse names in historical data to group by normalized name
    hist_df['normalized_name'] = hist_df['horse'].apply(normalize_horse_name)
    
    print(f"[DEBUG efficiency] After filtering: {len(hist_df)} rows, {hist_df['normalized_name'].nunique()} unique normalized names")
    
    # Calculate efficiency per normalized horse name
    hist_df['placed_top2'] = (hist_df['position'] <= 2).astype(int)
    hist_df['market_prob'] = 1 / hist_df['win_odds']
    
    efficiency = hist_df.groupby('normalized_name').agg({
        'placed_top2': 'mean',
        'market_prob': 'mean'
    })
    efficiency['efficiency'] = efficiency['placed_top2'] - efficiency['market_prob']
    
    print(f"[DEBUG efficiency] Calculated efficiency for {len(efficiency)} normalized names")
    print(f"[DEBUG efficiency] Sample efficiency values:\n{efficiency['efficiency'].head(10)}")
    
    # Map back to original DataFrame using matched_historical_id (which is the normalized name)
    result = df['matched_historical_id'].map(efficiency['efficiency'])
    
    return result


def _calculate_horse_odds_trend(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.Series:
    """
    Calculate odds trend (current odds / rolling avg) using normalized horse names and all matching IDs.
    """
    global _HORSE_NAME_INDEX
    
    # Get unique normalized names for horses that have been matched
    matched_normalized_names = df[df['matched_historical_id'].notna()]['matched_historical_id'].unique().tolist()
    
    if not matched_normalized_names or conn is None or _HORSE_NAME_INDEX is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    # Collect all horse IDs that match these normalized names
    all_horse_ids = []
    for norm_name in matched_normalized_names:
        if norm_name in _HORSE_NAME_INDEX:
            all_horse_ids.extend(_HORSE_NAME_INDEX[norm_name])
    
    if not all_horse_ids:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    # Query historical odds using horse IDs
    placeholders = ','.join('?' * len(all_horse_ids))
    query = f"""
    SELECT r.horse_id, r.horse, r.race_id, rac.date as race_date, r.win_odds
    FROM runners r
    JOIN races rac ON r.race_id = rac.race_id
    WHERE r.horse_id IN ({placeholders})
      AND r.win_odds IS NOT NULL
      AND r.win_odds > 0
    ORDER BY rac.date
    """
    
    hist_df = pd.read_sql_query(query, conn, params=all_horse_ids)
    
    print(f"\n[DEBUG trend] Queried {len(all_horse_ids)} horse IDs from {len(matched_normalized_names)} normalized names, got {len(hist_df)} historical rows")
    
    # Convert win_odds to numeric and handle non-numeric values
    hist_df['win_odds'] = pd.to_numeric(hist_df['win_odds'], errors='coerce')
    
    # Filter out rows where win_odds is NaN
    hist_df = hist_df[hist_df['win_odds'].notna()]
    
    if hist_df.empty:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    # Normalize horse names in historical data to group by normalized name
    hist_df['normalized_name'] = hist_df['horse'].apply(normalize_horse_name)
    
    print(f"[DEBUG trend] After filtering: {len(hist_df)} rows, {hist_df['normalized_name'].nunique()} unique normalized names")
    
    # Calculate rolling average of odds for each normalized horse name
    hist_df['race_date'] = pd.to_datetime(hist_df['race_date'])
    hist_df = hist_df.sort_values(['normalized_name', 'race_date'])
    
    hist_df['rolling_avg_odds'] = hist_df.groupby('normalized_name')['win_odds'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Get the most recent rolling average for each normalized name
    latest_avg = hist_df.groupby('normalized_name')['rolling_avg_odds'].last()
    
    print(f"[DEBUG trend] Calculated trends for {len(latest_avg)} normalized names")
    print(f"[DEBUG trend] Sample trend values:\n{latest_avg.head(10)}")
    
    # Map back to original DataFrame using matched_historical_id and calculate trend
    result = df.apply(lambda row: 
        row['win_odds'] / latest_avg.get(row['matched_historical_id'], 1.0) 
        if pd.notna(row['matched_historical_id']) and pd.notna(row['win_odds']) and row['matched_historical_id'] in latest_avg
        else np.nan,
        axis=1
    )
    
    return result


def trainer_odds_bias(df: pd.DataFrame) -> pd.Series:
    """
    Calculate trainer odds bias using normalized trainer names.
    Now uses names instead of IDs for consistent matching.
    """
    win_odds = pd.to_numeric(df['win_odds'], errors='coerce')
    position = pd.to_numeric(df['position'], errors='coerce')
    
    # Calculate market probability
    market_prob = 1 / win_odds.replace(0, np.nan)
    
    # Calculate actual performance (placed in top 2) - use normalized name
    actual_perf_by_trainer = df.groupby('trainer_normalized')['position'].apply(
        lambda x: (x.fillna(999) <= 2).mean()
    )
    market_exp_by_trainer = df.groupby('trainer_normalized')['win_odds'].apply(
        lambda x: (1 / x.replace(0, np.nan)).mean()
    )
    trainer_bias = actual_perf_by_trainer - market_exp_by_trainer
    
    result = df['trainer_normalized'].map(trainer_bias)
    
    return result


# ---------------- Rolling form ----------------

def _add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[DEBUG rolling_form] Starting with {len(df)} rows")
    print(f"[DEBUG rolling_form] Rows with position: {df['position'].notna().sum()}")
    
    # Horse rolling form - use normalized name instead of horse_id
    df = df.sort_values(['horse_normalized','race_date'])
    df['horse_pos_avg3'] = (
        df.groupby('horse_normalized')['position']
          .transform(lambda x: x.rolling(3, 1).mean().shift(1))
    )
    df['horse_last_placed'] = (
        df.groupby('horse_normalized')['position']
          .shift(1)
          .between(1, 3).astype(int)
    )
    
    horse_pos_coverage = df['horse_pos_avg3'].notna().sum()
    horse_placed_coverage = df['horse_last_placed'].notna().sum()
    print(f"[DEBUG rolling_form] Horse features: horse_pos_avg3={horse_pos_coverage}/{len(df)} ({horse_pos_coverage/len(df)*100:.1f}%), horse_last_placed={horse_placed_coverage}/{len(df)} ({horse_placed_coverage/len(df)*100:.1f}%)")

    # Trainer rolling strike rate - use normalized name instead of ID
    df = df.sort_values(['trainer_normalized','race_date'])
    df['trainer_win30'] = (
        df.assign(win=(df['position'] == 1).astype(int))
          .groupby('trainer_normalized')['win']
          .transform(lambda x: x.rolling(window=50, min_periods=5).mean().shift(1))
    )
    
    trainer_coverage = df['trainer_win30'].notna().sum()
    print(f"[DEBUG rolling_form] Trainer features: trainer_win30={trainer_coverage}/{len(df)} ({trainer_coverage/len(df)*100:.1f}%)")

    # Jockey rolling strike rate - use normalized name instead of ID
    df = df.sort_values(['jockey_normalized','race_date'])
    df['jockey_win30'] = (
        df.assign(win=(df['position'] == 1).astype(int))
          .groupby('jockey_normalized')['win']
          .transform(lambda x: x.rolling(window=50, min_periods=5).mean().shift(1))
    )
    
    jockey_coverage = df['jockey_win30'].notna().sum()
    print(f"[DEBUG rolling_form] Jockey features: jockey_win30={jockey_coverage}/{len(df)} ({jockey_coverage/len(df)*100:.1f}%)")

    return df


# ---------------- Odds-history metrics (main entry point) ----------------

def _add_odds_history_metrics(df: pd.DataFrame, conn: sqlite3.Connection = None) -> pd.DataFrame:
    """
    Add odds-based historical metrics to the DataFrame.
    Now uses horse name matching to handle different horse IDs across data sources.
    """
    if conn is None:
        print("⚠️  Warning: No database connection provided for odds-history metrics")
        # Add placeholder columns
        df['is_first_time_runner'] = 1
        df['horse_odds_efficiency'] = 0.0
        df['horse_odds_trend'] = 1.0
        df['trainer_odds_bias'] = 0.0
        df['first_time_penalty'] = -0.5
        return df
    
    print("\n🔍 Matching horses to historical data...")
    
    # Only match future races (where position is NaN)
    # Historical races already have horse_id, so use that as matched_historical_id
    future_mask = df['position'].isna()
    historical_mask = ~future_mask
    
    # For historical races, use existing horse_id as matched_historical_id
    df.loc[historical_mask, 'matched_historical_id'] = df.loc[historical_mask, 'horse_id']
    df.loc[historical_mask, 'match_confidence'] = 1.0
    
    # Only match future horses
    if future_mask.sum() > 0:
        df_future = df[future_mask].copy()
        df_future = _match_horses_to_historical(df_future, conn)
        
        # Merge back the matching results
        df.loc[future_mask, 'matched_historical_id'] = df_future['matched_historical_id']
        df.loc[future_mask, 'match_confidence'] = df_future['match_confidence']
    else:
        # No future races to match
        df['matched_historical_id'] = df['horse_id']
        df['match_confidence'] = 1.0
    
    # Add first-time runner indicator (only for future races)
    df['is_first_time_runner'] = 0
    df.loc[future_mask & df['matched_historical_id'].isna(), 'is_first_time_runner'] = 1
    
    # Calculate horse odds efficiency
    print("\n📊 Calculating horse odds efficiency...")
    df['horse_odds_efficiency'] = _calculate_horse_odds_efficiency(df, conn)
    
    # Calculate horse odds trend
    print("📈 Calculating horse odds trend...")
    df['horse_odds_trend'] = _calculate_horse_odds_trend(df, conn)
    
    # Calculate trainer odds bias (doesn't need matching)
    print("👔 Calculating trainer odds bias...")
    df['trainer_odds_bias'] = trainer_odds_bias(df)
    
    # Fill missing values with population statistics
    df['horse_odds_efficiency'] = df['horse_odds_efficiency'].fillna(df['horse_odds_efficiency'].mean())
    df['horse_odds_trend'] = df['horse_odds_trend'].fillna(df['horse_odds_trend'].median())
    df['trainer_odds_bias'] = df['trainer_odds_bias'].fillna(df['trainer_odds_bias'].mean())
    
    # Apply penalty for first-time runners
    df.loc[df['is_first_time_runner'] == 1, 'horse_odds_efficiency'] -= 0.25
    df['first_time_penalty'] = df['is_first_time_runner'] * -0.5
    
    # Print summary (for all data, but highlight future races if any)
    print("\n✅ Odds-history features summary:")
    total_first_time = df['is_first_time_runner'].sum()
    if future_mask.sum() > 0:
        future_first_time = df.loc[future_mask, 'is_first_time_runner'].sum()
        print(f"   First-time runners (future races): {future_first_time}/{future_mask.sum()} ({future_first_time/future_mask.sum()*100:.1f}%)")
    print(f"   First-time runners (total): {total_first_time} ({df['is_first_time_runner'].mean()*100:.1f}%)")
    print(f"   Horse odds efficiency: mean={df['horse_odds_efficiency'].mean():.3f}, std={df['horse_odds_efficiency'].std():.3f}")
    print(f"   Horse odds trend: mean={df['horse_odds_trend'].mean():.3f}, std={df['horse_odds_trend'].std():.3f}")
    print(f"   Trainer odds bias: mean={df['trainer_odds_bias'].mean():.3f}, std={df['trainer_odds_bias'].std():.3f}")
    
    return df


# ---------------- Public entrypoint ----------------

def build_features(db_path: str = "data/historical/hkjc.db") -> pd.DataFrame:
    conn = get_connection()
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
        
        # --- Phase 3 odds-history metrics (with horse matching) ---
        df = _add_odds_history_metrics(df, conn)

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
        "race_class", "dist_m", "going", "rail",
        # exclude matching metadata
        "matched_historical_id", "match_confidence"
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