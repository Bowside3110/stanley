# tests/test_hkjc_fetch.py
import pytest
import pandas as pd
from src.data import hkjc_client  # assumes you wrap HKJC GraphQL calls here

def test_hkjc_runners():
    date = "2025-09-07"
    venue = "ST"  # Sha Tin
    race_no = 2

    runners = hkjc_client.fetch_runners(date, venue, race_no)
    df = pd.DataFrame(runners)
    print(df.head())

    assert "jockey" in df.columns
    assert "trainer" in df.columns
    assert not df.empty
