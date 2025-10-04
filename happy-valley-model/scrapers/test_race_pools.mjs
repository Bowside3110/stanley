import { HorseRacingAPI } from "hkjc-api";

const api = new HorseRacingAPI();

// Get active meetings
api.getActiveMeetings()
    .then(meetings => {
        if (!meetings || meetings.length === 0) {
            console.error("No upcoming meetings found.");
            process.exit(1);
        }

        const nextMeeting = meetings[0];
        const { date, venueCode } = nextMeeting;
        console.log(`Testing pools for next meeting: ${venueCode} on ${date}`);

        // Get race meetings to find race numbers
        return api.getRaceMeetings({ date, venueCode })
            .then(races => {
                if (!races || !races.raceMeetings || !races.raceMeetings[0] || !races.raceMeetings[0].races) {
                    console.error("No races found in meeting.");
                    process.exit(1);
                }

                // Get the first race number
                const firstRace = races.raceMeetings[0].races[0];
                const raceNo = firstRace.no;
                console.log(`Testing race #${raceNo}`);

                // Try both methods from the API documentation
                return Promise.all([
                    // Method 1: getRaceOdds
                    api.getRaceOdds(raceNo, ['WIN', 'PLA', 'QIN'])
                        .then(oddsResult => {
                            console.log("\n=== getRaceOdds Result ===");
                            console.log(JSON.stringify(oddsResult, null, 2));
                            return oddsResult;
                        })
                        .catch(err => {
                            console.error("Error fetching odds:", err.message);
                            return null;
                        }),

                    // Method 2: getRacePools
                    api.getRacePools(raceNo, ['WIN', 'PLA'])
                        .then(poolsResult => {
                            console.log("\n=== getRacePools Result ===");
                            console.log(JSON.stringify(poolsResult, null, 2));
                            return poolsResult;
                        })
                        .catch(err => {
                            console.error("Error fetching pools:", err.message);
                            return null;
                        })
                ]);
            });
    })
    .catch(err => {
        console.error("Error:", err);
    });
