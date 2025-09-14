from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd


# ---------------- Helpers ----------------

def _read_sql(conn: sqlite3.Connection, sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Wrapper around pd.read_sql_query that avoids passing params=None."""
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
    """
    Base = races + runners + results, enriched with racecard_pro if available.
    """
    sql = """
    SELECT
        r.race_id,
        r.date AS race_date,
        r.course,
        rc.race_name,
        rc.race_class,
        rc.going,
        rc.dist_m,
        rc.rail,
        run.horse_id,
        run.horse,
        run.draw,
        run.weight,
        run.win_odds,
        run.jockey_id,
        run.trainer_id,
        res.position
    FROM races r
    JOIN runners run ON run.race_id = r.race_id
    LEFT JOIN results res ON res.race_id = run.race_id AND res.horse_id = run.horse_id
    LEFT JOIN racecard_pro rc ON rc.race_id = r.race_id
    WHERE r.course LIKE '%(HK)' AND r.race_id IS NOT NULL
    """
    df = _read_sql(conn, sql)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["draw"] = pd.to_numeric(df["draw"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["weight"] = df["weight"].apply(_parse_weight_lbs)
    df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce")
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


# ---------------- New: Margins & Times ----------------

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


# ---------------- New: Class / Distance Moves ----------------

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


# ---------------- Public entrypoint ----------------

def build_features(db_path: str = "data/historical/hkjc.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = _base_frame(conn)
        df = _attach_racecard_runner_fields(conn, df)
        df = _equipment_flags(df)

        # New features
        df = _add_margins_and_times(conn, df)
        df = _add_class_distance_moves(conn, df)

        # Deterministic order
        df["__ord"] = df["draw"].fillna(9999)
        df = df.sort_values(["race_id", "__ord", "horse_id"]).drop(columns="__ord")

        return df
    finally:
        conn.close()


# ---------------- Feature picker ----------------

def _pick_features(df: pd.DataFrame) -> list[str]:
    """Pick numeric runner-level features from the feature frame."""
    return [
        c for c in df.columns
        if c not in ["race_id", "race_date", "race_name", "horse_id", "horse",
                     "trainer_id", "jockey_id", "position"]
           and pd.api.types.is_numeric_dtype(df[c])
    ]