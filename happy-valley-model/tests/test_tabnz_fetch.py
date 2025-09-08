# tests/test_tabnz_fetch.py
import requests
import pandas as pd

BASE = "https://json.tab.co.nz"

def fetch(path: str):
    url = f"{BASE}/{path}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

def test_tabnz_day(date="2025-09-07"):
    # 1. Schedule: what meetings/races are on
    sched = fetch(f"schedule/{date}")
    meetings = sched.get("meetings", sched)

    print(f"\nAll meetings on {date}:")
    for m in meetings:
        print("-", m.get("meetingName"))

    # Looser filter: look for "sha" or "happy" in name
    hk_meetings = [
        m for m in meetings
        if m.get("meetingName") and any(
            kw in m["meetingName"].lower() for kw in ["sha", "happy"]
        )
    ]
    print("\nHK Meetings (filtered):", [m["meetingName"] for m in hk_meetings])

    # 2. Odds: win/place for each runner
    odds = fetch(f"odds/{date}")
    odds_meetings = odds.get("meetings", odds)
    for m in odds_meetings:
        if m.get("meetingName") in [hm["meetingName"] for hm in hk_meetings]:
            for race in m.get("races", []):
                rows = []
                for r in race.get("runners", []):
                    rows.append({
                        "race_no": race.get("raceNo"),
                        "runner_no": r.get("number"),
                        "runner_name": r.get("name"),
                        "win_odds": r.get("win") or r.get("fixedOdds"),
                        "place_odds": r.get("place") or r.get("placeFixedOdds"),
                    })
                df = pd.DataFrame(rows)
                print(f"\nRace {race.get('raceNo')} odds:")
                print(df.head())

    # 3. Results: finishing order
    results = fetch(f"results/{date}")
    for m in results.get("meetings", results):
        if m.get("meetingName") in [hm["meetingName"] for hm in hk_meetings]:
            for race in m.get("races", []):
                print(f"\nRace {race.get('raceNo')} results:", race.get("results"))

    # 4. Quinella dividends
    quinella = fetch(f"qla/{date}")
    for m in quinella.get("meetings", quinella):
        if m.get("meetingName") in [hm["meetingName"] for hm in hk_meetings]:
            for race in m.get("races", []):
                print(f"\nRace {race.get('raceNo')} quinella dividends:", race.get("dividends"))

if __name__ == "__main__":
    test_tabnz_day()
