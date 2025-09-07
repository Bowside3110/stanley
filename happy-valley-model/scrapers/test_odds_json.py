import requests
import json

def test_fetch_odds_json(race_date="2025-09-07", race_no=1, course_code="ST"):
    url = (
        f"https://bet.hkjc.com/racing/getJSON.aspx?"
        f"type=winplaodds&date={race_date}&venue={course_code}&raceno={race_no}&lang=en"
    )
    print(f"Fetching: {url}")
    resp = requests.get(url)

    print("Status code:", resp.status_code)
    print("Content type:", resp.headers.get("Content-Type"))

    # Show first 500 characters of the response
    print("\nRaw text (first 500 chars):")
    print(resp.text[:500])

    # Try to decode JSON safely
    try:
        data = resp.json()
        print("\n✅ Parsed JSON keys:", list(data.keys()))
        with open("data/future/test_odds.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Saved JSON -> data/future/test_odds.json")
    except Exception as e:
        print("\n⚠️ Could not parse JSON:", e)

if __name__ == "__main__":
    test_fetch_odds_json("2025-09-07", 1, "ST")
