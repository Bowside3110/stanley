import requests
from bs4 import BeautifulSoup
import re

date = "2023/09/13"
race_no = 1

url = f"https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={date}&RaceNo={race_no}"
resp = requests.get(url)
soup = BeautifulSoup(resp.text, "html.parser")

# Collapse the whole page text into a single string
page_text = soup.get_text(" ", strip=True)

# Regex patterns
class_match = re.search(r"(Class \d+)", page_text)
dist_match = re.search(r"(\d+)\s*M", page_text)
going_match = re.search(r"Going\s*:\s*([A-Za-z]+)", page_text)

race_meta = {
    "race_date": date,
    "race_no": race_no,
    "race_class": class_match.group(1) if class_match else None,
    "distance": int(dist_match.group(1)) if dist_match else None,
    "going": going_match.group(1).strip() if going_match else None,
}

print("Parsed race meta:", race_meta)
