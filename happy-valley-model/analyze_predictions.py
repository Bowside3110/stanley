#!/usr/bin/env python3
import pandas as pd

# Load the predictions
df = pd.read_csv('data/predictions/test_predictions_updated.csv')

# Print summary statistics
print(f'Total horses: {len(df)}')
if 'is_first_time_runner' in df.columns:
    print(f'First-time runners: {df["is_first_time_runner"].sum()}')
    print(f'Percentage: {df["is_first_time_runner"].mean()*100:.1f}%')
else:
    print('is_first_time_runner column not found in predictions')

# Print top 3 horses per race
print('\nTop 3 horses per race:')
for rid, g in df.groupby('race_id'):
    print(f'\nRace: {g["race_name"].iloc[0]}')
    top3 = g.sort_values('rank').head(3)
    for _, row in top3.iterrows():
        if 'is_first_time_runner' in df.columns:
            first_time = "Yes" if row["is_first_time_runner"] == 1 else "No"
            print(f'  {row["rank"]}. {row["horse"]} (First-time: {first_time}, Score: {row["score"]:.3f})')
        else:
            print(f'  {row["rank"]}. {row["horse"]} (Score: {row["score"]:.3f})')
