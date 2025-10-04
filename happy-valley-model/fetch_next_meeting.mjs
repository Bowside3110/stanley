import { HorseRacingAPI } from "hkjc-api";
import fs from "fs";
import path from "path";

const api = new HorseRacingAPI();

api.getActiveMeetings()
  .then(meetings => {
    if (!meetings || meetings.length === 0) {
      console.error("No upcoming meetings found.");
      process.exit(1);
    }

    const nextMeeting = meetings[0];
    const { date, venueCode } = nextMeeting;
    console.log(`Fetching races for next meeting: ${venueCode} on ${date}`);

    return api.getRaceMeetings({ date, venueCode })
      .then(races => {
        // For each race, fetch odds
        const oddsPromises = [];
        races.raceMeetings.forEach(meeting => {
          meeting.races.forEach(race => {
            const oddsPromise = api.getRaceOdds(race.no, ["WIN"])
              .then(oddsResult => {
                // Check if we have an array of odds data
                if (oddsResult && Array.isArray(oddsResult)) {
                  // Find the WIN odds data
                  const winOddsData = oddsResult.find(item => item.oddsType === "WIN");

                  if (winOddsData && winOddsData.oddsNodes) {
                    // Process each runner in the race
                    race.runners.forEach(runner => {
                      // Find the odds for this runner by matching runner.no with combString
                      const runnerNo = runner.no.toString().padStart(2, '0');
                      const oddsNode = winOddsData.oddsNodes.find(node =>
                        node.combString === runnerNo || node.combString === String(runner.no)
                      );

                      if (oddsNode && oddsNode.oddsValue) {
                        runner.winOdds = oddsNode.oddsValue;
                        console.log(`Found odds for horse #${runner.no}: ${runner.winOdds}`);
                      }
                    });
                  }
                }
              })
              .catch(err => {
                console.warn(`⚠️ Could not fetch odds for race ${race.no}:`, err.message);
              });

            oddsPromises.push(oddsPromise);
          });
        });

        return Promise.all(oddsPromises).then(() => races);
      })
      .then(races => {
        // Save file
        const dir = path.join("data", "predictions");
        if (!fs.existsSync(dir)) {
          fs.mkdirSync(dir, { recursive: true });
        }
        const filename = path.join(dir, `races_${date}_${venueCode}.json`);
        fs.writeFileSync(filename, JSON.stringify(races, null, 2));
        console.log(`✅ Saved racecard (with odds) to ${filename}`);
      });
  })
  .catch(err => {
    console.error("Error fetching races:", err);
  });
