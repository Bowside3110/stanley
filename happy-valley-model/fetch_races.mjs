import { HorseRacingAPI } from "hkjc-api";
import fs from "fs";
import path from "path";

async function main() {
  const api = new HorseRacingAPI();

  try {
    // Step 1: get all active meetings
    const meetings = await api.getActiveMeetings();
    if (!meetings || meetings.length === 0) {
      console.error("No upcoming meetings found.");
      process.exit(1);
    }

    // Step 2: pick the earliest one
    const nextMeeting = meetings[0]; // meetings are usually sorted by date
    const { date, venueCode } = nextMeeting;

    console.log(`Fetching races for next meeting: ${venueCode} on ${date}`);

    // Step 3: fetch full races
    const races = await api.getRaceMeetings({ date, venueCode });

    // Step 4: save to file
    const dir = path.join("data", "predictions");
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    
    const filename = `races_${date}_${venueCode}.json`;
    fs.writeFileSync(filename, JSON.stringify(races, null, 2));

    console.log(`✅ Saved racecard to ${filename}`);
  } catch (err) {
    console.error("Error fetching races:", err);
  }
}

main();

