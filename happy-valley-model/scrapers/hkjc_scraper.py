import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime, timedelta
import time
import re

def normalize_combo(combo_str):
    """Standardize combo strings (e.g. '8,5' -> '5-8')."""
    try:
        nums = [int(x) for x in combo_str.replace('-', ',').split(',') if x.strip().isdigit()]
        return "-".join(str(x) for x in sorted(nums))
    except Exception:
        return combo_str

def parse_race_meta(soup, race_date, race_no):
    """Extract race-level metadata (class, distance, going, course)."""
    page_text = soup.get_text(" ", strip=True)

    class_match = re.search(r"(Class \d+)", page_text)
    dist_match = re.search(r"(\d+)\s*M", page_text)
    going_match = re.search(r"Going\s*:\s*([A-Za-z]+)", page_text)
    course_match = re.search(r"Course\s*:\s*([^ ]+.*?Course)", page_text)

    return {
        "race_date": race_date,
        "race_no": race_no,
        "race_class": class_match.group(1) if class_match else None,
        "distance": int(dist_match.group(1)) if dist_match else None,
        "going": going_match.group(1).strip() if going_match else None,
        "course": course_match.group(1).strip() if course_match else None,
    }

def fetch_race_data(race_date="2023/09/13", race_no=1):
    """Scrape one race (runners + dividends + race metadata)."""
    url = f"https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={race_date}&RaceNo={race_no}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None, None, None
    soup = BeautifulSoup(resp.text, "html.parser")

    tables = soup.find_all("table")
    if len(tables) < 4:
        return None, None, None

    # --- Race Meta ---
    race_meta = parse_race_meta(soup, race_date, race_no)

    # --- Runners (Table 2) ---
    runners = []
    runner_table = tables[2]
    for row in runner_table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cols) >= 3:
            runners.append({
                "race_date": race_date,
                "race_no": race_no,
                "placing": cols[0],
                "horse_no": cols[1],
                "horse": cols[2],
                "jockey": cols[3],
                "trainer": cols[4],
                "act_wt": cols[5],
                "decl_wt": cols[6],
                "draw": cols[7],
                "odds": cols[-1],
            })

    # --- Dividends (Table 3) ---
    dividends = []
    div_table = tables[3]
    for row in div_table.find_all("tr")[2:]:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cols) == 3:
            pool, combo, div = cols
            dividends.append({
                "race_date": race_date,
                "race_no": race_no,
                "pool": pool,
                "combo": normalize_combo(combo),
                "dividend": div
            })

    return runners, dividends, race_meta

def scrape_meeting(race_date="2023/09/13", max_races=12):
    """Scrape all races for a meeting date and return runners+dividends+meta."""
    all_runners, all_dividends, all_races = [], [], []
    for race_no in range(1, max_races + 1):
        runners, dividends, race_meta = fetch_race_data(race_date, race_no)
        if not runners:
            break
        all_runners.extend(runners)
        all_dividends.extend(dividends)
        all_races.append(race_meta)
    return all_runners, all_dividends, all_races

def scrape_season(start_date="2023-09-01", end_date="2024-07-14",
                  out_dir="data/raw", out_prefix="hkjc_2023_24"):
    """Scrape all meetings between start_date and end_date into three CSVs."""
    d0 = datetime.strptime(start_date, "%Y-%m-%d")
    d1 = datetime.strptime(end_date, "%Y-%m-%d")
    cur = d0

    all_runners, all_dividends, all_races = [], [], []

    while cur <= d1:
        date_str = cur.strftime("%Y/%m/%d")
        runners, dividends, races = scrape_meeting(date_str)
        if runners:
            all_runners.extend(runners)
            all_dividends.extend(dividends)
            all_races.extend(races)
            print(f"[{date_str}] {len(runners)} runners, {len(dividends)} dividends, {len(races)} races scraped")
        else:
            print(f"[{date_str}] No races")
        cur += timedelta(days=1)
        time.sleep(1)

    if not all_runners:
        print("No data scraped for this season.")
        return

    os.makedirs(out_dir, exist_ok=True)
    runners_path = os.path.join(out_dir, f"{out_prefix}_runners.csv")
    divs_path = os.path.join(out_dir, f"{out_prefix}_dividends.csv")
    races_path = os.path.join(out_dir, f"{out_prefix}_races.csv")

    pd.DataFrame(all_runners).to_csv(runners_path, index=False)
    pd.DataFrame(all_dividends).to_csv(divs_path, index=False)
    pd.DataFrame(all_races).to_csv(races_path, index=False)

    print(f"Saved season runners -> {runners_path}")
    print(f"Saved season dividends -> {divs_path}")
    print(f"Saved season races -> {races_path}")

def scrape_season_races_only(start_date="2023-09-01", end_date="2024-07-14",
                             out_dir="data/raw", out_prefix="hkjc_2023_24"):
    """Scrape only race-level metadata (no runners/dividends)."""
    d0 = datetime.strptime(start_date, "%Y-%m-%d")
    d1 = datetime.strptime(end_date, "%Y-%m-%d")
    cur = d0

    all_races = []

    while cur <= d1:
        date_str = cur.strftime("%Y/%m/%d")
        _, _, races = scrape_meeting(date_str)
        if races:
            all_races.extend(races)
            print(f"[{date_str}] {len(races)} races scraped (meta only)")
        else:
            print(f"[{date_str}] No races")
        cur += timedelta(days=1)
        time.sleep(1)

    if not all_races:
        print("No metadata scraped for this season.")
        return

    os.makedirs(out_dir, exist_ok=True)
    races_path = os.path.join(out_dir, f"{out_prefix}_races.csv")
    pd.DataFrame(all_races).to_csv(races_path, index=False)
    print(f"Saved season races -> {races_path}")

if __name__ == "__main__":
    # Option A: scrape everything
    # scrape_season("2023-09-01", "2024-07-14")

    # Option B: scrape only race-level metadata
    scrape_season_races_only("2023-09-01", "2024-07-14")
