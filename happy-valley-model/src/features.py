# src/features.py
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd


# ---------------- Helpers ----------------

def _read_sql(conn: sqlite3.Connection, sql: str, params: tuple | None = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)

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


# ---------------- Public entrypoint ----------------

def build_features(db_path: str = "data/historical/hkjc.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = _base_frame(conn)
        df = _attach_racecard_runner_fields(conn, df)
        df = _equipment_flags(df)

        # TODO: re-attach all your other feature builders (days_since_last_run, career stats, etc.)
        # They can remain unchanged; this is just the racecard patch.

        # Deterministic order
        df["__ord"] = df["draw"].fillna(9999)
        df = df.sort_values(["race_id", "__ord", "horse_id"]).drop(columns="__ord")

        return df
    finally:
        conn.close()
