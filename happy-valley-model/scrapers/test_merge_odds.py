import pandas as pd
import os

def merge_runners_and_odds(
    runners_path="data/future/hkjc_future_2025-09-07_runners.csv",
    odds_path="data/future/StaticOdds.csv",
    out_path="data/future/hkjc_future_2025-09-07_merged.csv"
):
    # Load runners
    runners = pd.read_csv(runners_path)
    print(f"Loaded runners: {len(runners)} rows, columns = {list(runners.columns)}")

    # Load odds
    odds = pd.read_csv(odds_path)
    print(f"Loaded odds: {len(odds)} rows, columns = {list(odds.columns)}")

    # Coerce merge keys to numeric
    for df in (runners, odds):
        if "race_no" in df.columns:
            df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").astype("Int64")
        if "horse_no" in df.columns:
            df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")

    # Merge
    merge_keys = ["race_no", "horse_no"]
    merged = runners.merge(odds, on=merge_keys, how="left", suffixes=("", "_odds"))

    # If duplicate horse column exists, drop the odds one
    if "horse_odds" in merged.columns:
        merged = merged.drop(columns=["horse_odds"])

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"✅ Saved merged dataset with odds -> {out_path}")
    print(f"Summary: {len(merged)} rows after merge")

    # Show sample
    cols_to_show = ["race_no", "horse_no"]
    if "horse" in merged.columns:
        cols_to_show.append("horse")
    if "win_odds" in merged.columns:
        cols_to_show.append("win_odds")
    if "place_odds" in merged.columns:
        cols_to_show.append("place_odds")

    print("\nSample merged rows:")
    print(merged.head(10)[cols_to_show])

    return merged


if __name__ == "__main__":
    merge_runners_and_odds()
