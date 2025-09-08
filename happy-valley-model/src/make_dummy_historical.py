import os
import pandas as pd
import numpy as np

def make_dummy_historical(out_path="data/historical/hkjc_full.csv", n_rows=200):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "avg_last3": rng.integers(1, 10, size=n_rows),          # finishing pos average
        "days_since_last": rng.integers(5, 90, size=n_rows),    # days since last run
        "horse_no": rng.integers(1, 14, size=n_rows),           # runner number
        "draw_cat": rng.integers(1, 14, size=n_rows),           # barrier draw
        "win_odds": rng.uniform(2, 50, size=n_rows).round(2),   # odds
        "act_wt": rng.integers(110, 135, size=n_rows),          # weight
        "jockey_win_rate": rng.uniform(0, 0.2, size=n_rows),    # jockey strike %
        "trainer_win_rate": rng.uniform(0, 0.2, size=n_rows),   # trainer strike %
        "jt_combo_win_rate": rng.uniform(0, 0.15, size=n_rows), # combo %
        "is_place": rng.choice([0, 1], size=n_rows, p=[0.7, 0.3]) # label: placed or not
    })

    df.to_csv(out_path, index=False)
    print(f"✅ Dummy historical dataset created: {out_path} with {len(df)} rows")

if __name__ == "__main__":
    make_dummy_historical()
