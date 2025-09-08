# tests/test_stitch_tabnz_hkjc.py
import pytest
import pandas as pd
from src.data import tab_nz, hkjc_client

def test_stitch():
    date = "2025-09-07"
    meetings = tab_nz.list_hk_meetings(date)
    assert meetings, "No HK meetings"

    # Take first race of first meeting
    m = meetings[0]
    meet_no = int(m["meetNo"])
    race_no = int(m["races"][0]["raceNo"])

    # TAB odds
    odds = tab_nz.load_odds(date)[(meet_no, race_no)]
    odds_df = pd.DataFrame([{
        "runner_no": r.get("number"),
        "runner_name": r.get("name"),
        "win_odds": r.get("win") or r.get("fixedOdds")
    } for r in odds.get("runners", [])])

    # HKJC runners
    venue_code = "ST" if "Sha Tin" in m["meetingName"] else "HV"
    hkjc_df = pd.DataFrame(hkjc_client.fetch_runners(date, venue_code, race_no))

    # Attempt merge on runner_no first, fallback to runner_name
    merged = pd.merge(odds_df, hkjc_df,
                      on="runner_no",
                      how="outer",
                      suffixes=("_tab", "_hkjc"))

    print(merged.head())

    assert not merged.empty
