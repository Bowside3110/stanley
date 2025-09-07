import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def load_dataset(filename: str) -> pd.DataFrame:
    """Generic CSV loader from data/raw."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


# === Kaggle / old loaders (keep if you want backwards compatibility) ===
def load_sectionals():
    return load_dataset("sectional_times.csv")


# === New HKJC scrapes (preferred for ROI) ===
def load_hkjc_runners(path=None):
    """Load HKJC scraped runner-level data (one row per horse)."""
    if path is None:
        path = os.path.join(DATA_DIR, "hkjc_2023_24_runners.csv")
    return pd.read_csv(path, low_memory=False)


def load_hkjc_dividends(path=None):
    """Load HKJC scraped dividends (race-level, multiple pools)."""
    if path is None:
        path = os.path.join(DATA_DIR, "hkjc_2023_24_dividends.csv")
    return pd.read_csv(path, low_memory=False)


def load_hkjc_dataset():
    """Convenience: return both runners and dividends as DataFrames."""
    return load_hkjc_runners(), load_hkjc_dividends()
