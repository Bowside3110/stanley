import requests
import json

GRAPHQL_URL = "https://consvc.hkjc.com/JCBW/api/graph"

def fetch_odds(race_date="2025-09-07", race_no=1, course_code="ST"):
    payload = {
        "operationName": "racing",
        "variables": {
            "date": race_date,
            "venueCode": course_code,
            "raceNo": race_no,
            "oddsTypes": ["WIN", "PLA"],
        },
        "query": """
        query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
          raceMeetings(date: $date, venueCode: $venueCode) {
            pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
              oddsType
              oddsNodes {
                combString
                oddsValue
              }
            }
          }
        }
        """,
    }

    resp = requests.post(GRAPHQL_URL, json=payload, headers={"Content-Type": "application/json"})
    print("Status:", resp.status_code)
    try:
        data = resp.json()
        print(json.dumps(data, indent=2)[:1000])  # show first 1000 chars
        return data
    except Exception as e:
        print("⚠️ Could not parse JSON:", e)
        print("Raw text snippet:", resp.text[:500])
        return None

if __name__ == "__main__":
    fetch_odds("2025-09-07", race_no=1, course_code="ST")
