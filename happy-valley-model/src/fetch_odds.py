import requests

def fetch_quinella_odds(race_no=1, venue="HV"):
    url = (
        f"https://bet.hkjc.com/racing/getJSON.aspx?"
        f"type=QIN&venue={venue}&raceno={race_no}&lang=en"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://bet.hkjc.com/racing/pages/odds_wp.aspx?lang=EN"
    }
    resp = requests.get(url, headers=headers)

    print("Status:", resp.status_code)
    print("First 200 chars of response:")
    print(resp.text[:200])

    # Try JSON only if it looks like JSON
    try:
        return resp.json()
    except Exception:
        return {}
