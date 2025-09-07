import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import time


def parse_race_header(soup, race_date, race_no):
    """Extract race metadata from the header block for a given race page."""
    header = soup.find("div", class_=lambda c: c and "f_fs13" in c)
    text = header.get_text(" ", strip=True) if header else ""
    race_meta = {
        "race_date": race_date,
        "race_no": race_no,
        "race_name": None,
        "race_class": None,
        "distance": None,
        "going": None,
        "course": None,
    }

    m = re.search(r"Race\s+\d+\s*-\s*(.+?)(?=Sunday|Turf|Prize|$)", text)
    if m:
        race_meta["race_name"] = m.group(1).strip()

    m = re.search(r"(\d+)\s*M", text)
    if m:
        race_meta["distance"] = int(m.group(1))

    m = re.search(r"(Good|Yielding|Firm|Sloppy|Wet Slow|Fast)", text, flags=re.I)
    if m:
        race_meta["going"] = m.group(1).title()

    m = re.search(r"Course[,:\s]+([A-Za-z0-9\"'\-\s]+)", text)
    if m:
        race_meta["course"] = m.group(1).strip()

    m = re.search(r"Class\s+(\d+)", text)
    if m:
        race_meta["race_class"] = int(m.group(1))

    return race_meta


def clean_odds(value):
    """Return odds as string if numeric, otherwise None."""
    if not value:
        return None
    v = value.strip().upper()
    if v in ("PPG", "---", "N/A"):
        return None
    # numeric-like (e.g. "3.2")
    try:
        float(v)
        return v
    except ValueError:
        return None


def fetch_race(race_date, race_no):
    """Fetch one race page and return runners + metadata."""
    url = f"https://racing.hkjc.com/racing/information/English/racing/RaceCard.aspx?RaceDate={race_date}&RaceNo={race_no}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Failed to fetch race {race_no} for {race_date}")
        return [], None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Metadata
    race_meta = parse_race_header(soup, race_date, race_no)

    # Runners
    runners = []
    table = soup.find("table", class_=lambda c: c and "starter" in c)
    if not table:
        print(f"⚠️  No runners table found for Race {race_no}")
        return [], race_meta

    for row in table.find_all("tr")[1:]:
        cols = [td.get_text(" ", strip=True) for td in row.find_all("td")]
        if len(cols) < 6:
            continue
        odds_val = clean_odds(cols[-1])
        runner = {
            "race_date": race_date,
            "race_no": race_no,
            "horse_no": cols[0],
            "horse": cols[1],
            "jockey": cols[2],
            "trainer": cols[3],
            "act_wt": cols[4],
            "draw": cols[5],
            "odds": odds_val,
        }
        runners.append(runner)

    missing = sum(r["odds"] is None for r in runners)
    if missing:
        print(f"⚠️  Race {race_no}: {missing} runners have no live odds yet (PPG/blank)")

    print(f"Race {race_no}: {race_meta['race_name']} "
          f"(Class {race_meta['race_class']}, {race_meta['distance']}M, "
          f"{race_meta['going']}, {race_meta['course']}) "
          f"→ {len(runners)} runners scraped")

    return runners, race_meta


def fetch_racecard_meeting(race_date="2025/09/07", n_races=10):
    """Scrape all races for a meeting by looping RaceNo=1..n_races."""
    all_runners = []
    all_races = []
    for race_no in range(1, n_races + 1):
        runners, meta = fetch_race(race_date, race_no)
        all_runners.extend(runners)
        if meta:
            all_races.append(meta)
        time.sleep(0.5)  # polite pause
    return all_runners, all_races


def scrape_future_meeting_to_csv(race_date="2025/09/07", n_races=10,
                                 out_dir="data/future", out_prefix="hkjc_future"):
    runners, races = fetch_racecard_meeting(race_date, n_races)
    if not runners:
        print(f"No runners found for {race_date}")
        return

    os.makedirs(out_dir, exist_ok=True)
    base_name = f"{out_prefix}_{race_date.replace('/', '-')}"
    runners_path = os.path.join(out_dir, f"{base_name}_runners.csv")
    races_path = os.path.join(out_dir, f"{base_name}_races.csv")
    merged_path = os.path.join(out_dir, f"{base_name}_merged.csv")

    # Save runners and races separately
    pd.DataFrame(runners).to_csv(runners_path, index=False)
    pd.DataFrame(races).to_csv(races_path, index=False)

    # Merge into one model-ready file
    runners_df = pd.DataFrame(runners)
    races_df = pd.DataFrame(races)
    runners_df["race_date"] = pd.to_datetime(runners_df["race_date"]).dt.strftime("%Y-%m-%d")
    races_df["race_date"] = pd.to_datetime(races_df["race_date"]).dt.strftime("%Y-%m-%d")

    merged = runners_df.merge(
        races_df[["race_date", "race_no", "race_class", "distance", "going", "course"]],
        on=["race_date", "race_no"],
        how="left"
    )
    merged.to_csv(merged_path, index=False)

    print(f"✅ Saved future runners -> {runners_path}")
    print(f"✅ Saved future races   -> {races_path}")
    print(f"✅ Saved merged dataset -> {merged_path}")
    print(f"Summary: {len(races)} races, {len(runners)} runners scraped")


if __name__ == "__main__":
    scrape_future_meeting_to_csv("2025/09/07", n_races=10)
