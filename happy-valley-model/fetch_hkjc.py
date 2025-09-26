import requests, json

url = "https://info.cld.hkjc.com/graphql/base/"

query = """
query raceMeetings($date: String, $venueCode: String) {
  raceMeetings(date: $date, venueCode: $venueCode) {
    id
    venueCode
    date
    races {
      id
      no
      raceName_en
      distance
      raceClass_en
      raceTrack {
        description_en
      }
      raceCourse {
        displayCode
      }
      runners {
        id
        no
        name_en
        barrierDrawNumber
        handicapWeight
        winOdds
        jockey {
          code
          name_en
        }
        trainer {
          code
          name_en
        }
      }
    }
  }
}
"""

variables = {
    "date": "2025-09-28",   # tonight’s date
    "venueCode": "HV"       # HV = Happy Valley
}

resp = requests.post(url, json={"query": query, "variables": variables})
data = resp.json()

print(json.dumps(data, indent=2))
