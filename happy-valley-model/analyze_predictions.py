#!/usr/bin/env python3
"""
Analyze prediction accuracy by comparing predicted ranks vs actual results.

This script queries the database for races where both predictions and results
are available, then calculates various accuracy metrics.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_prediction_accuracy(db_path="data/historical/hkjc.db"):
    """
    Analyze prediction accuracy across all completed races with predictions.
    """
    print("=" * 80)
    print("PREDICTION ACCURACY ANALYSIS")
    print("=" * 80)
    
    conn = sqlite3.connect(db_path)
    
    # Query races with both predictions and results
    query = """
        SELECT 
            r.date,
            r.race_name,
            r.class AS race_class,
            run.horse,
            run.jockey,
            run.trainer,
            run.win_odds,
            run.position,
            run.predicted_rank,
            run.predicted_score,
            run.prediction_date,
            run.model_version
        FROM runners run
        JOIN races r ON run.race_id = r.race_id
        WHERE run.predicted_rank IS NOT NULL
          AND run.position IS NOT NULL
          AND run.position > 0
        ORDER BY r.date, r.race_name, run.predicted_rank
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        print("\n⚠️  No completed races with predictions found.")
        print("   Predictions are available, but races haven't been run yet.")
        print("   Run this script again after race results are available.")
        return
    
    print(f"\n📊 Dataset: {len(df)} runners across {df['date'].nunique()} race days")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Total races: {df.groupby(['date', 'race_name']).ngroups}")
    
    # Calculate accuracy metrics
    print("\n" + "=" * 80)
    print("OVERALL ACCURACY METRICS")
    print("=" * 80)
    
    # Top-1 accuracy (predicted rank 1 = actual position 1)
    top1_correct = len(df[(df['predicted_rank'] == 1) & (df['position'] == 1)])
    top1_total = len(df[df['predicted_rank'] == 1])
    top1_accuracy = (top1_correct / top1_total * 100) if top1_total > 0 else 0
    
    print(f"\n🥇 Top-1 Accuracy: {top1_accuracy:.1f}%")
    print(f"   Predicted winners that actually won: {top1_correct}/{top1_total}")
    
    # Top-3 accuracy (predicted top 3 finished in top 3)
    top3_pred = df[df['predicted_rank'] <= 3]
    top3_correct = len(top3_pred[top3_pred['position'] <= 3])
    top3_total = len(top3_pred)
    top3_accuracy = (top3_correct / top3_total * 100) if top3_total > 0 else 0
    
    print(f"\n🥉 Top-3 Accuracy: {top3_accuracy:.1f}%")
    print(f"   Predicted top-3 that finished in top-3: {top3_correct}/{top3_total}")
    
    # Mean Absolute Error
    mae = np.mean(np.abs(df['predicted_rank'] - df['position']))
    print(f"\n📏 Mean Absolute Error: {mae:.2f} positions")
    print(f"   Average difference between predicted and actual position")
    
    # Correlation
    correlation = df['predicted_rank'].corr(df['position'])
    print(f"\n📈 Correlation: {correlation:.3f}")
    print(f"   Correlation between predicted rank and actual position")
    print(f"   (1.0 = perfect prediction, 0.0 = no correlation, -1.0 = inverse)")
    
    # Score vs Position correlation
    score_correlation = df['predicted_score'].corr(df['position'])
    print(f"\n🎯 Score-Position Correlation: {score_correlation:.3f}")
    print(f"   Higher scores should predict lower (better) positions")
    print(f"   (Negative correlation is good here)")
    
    # Analyze by confidence level
    print("\n" + "=" * 80)
    print("ACCURACY BY CONFIDENCE LEVEL")
    print("=" * 80)
    
    # Define confidence buckets based on predicted_score
    df['confidence_bucket'] = pd.cut(df['predicted_score'], 
                                      bins=[0, 0.15, 0.17, 0.20, 1.0],
                                      labels=['Low (<15%)', 'Medium (15-17%)', 
                                             'High (17-20%)', 'Very High (>20%)'])
    
    print("\nTop-1 accuracy by confidence level:")
    for bucket in ['Low (<15%)', 'Medium (15-17%)', 'High (17-20%)', 'Very High (>20%)']:
        bucket_df = df[(df['confidence_bucket'] == bucket) & (df['predicted_rank'] == 1)]
        if len(bucket_df) > 0:
            correct = len(bucket_df[bucket_df['position'] == 1])
            total = len(bucket_df)
            accuracy = (correct / total * 100)
            print(f"   {bucket:20s}: {accuracy:5.1f}% ({correct}/{total})")
    
    # Analyze by race class
    print("\n" + "=" * 80)
    print("ACCURACY BY RACE CLASS")
    print("=" * 80)
    
    print("\nTop-1 accuracy by race class:")
    for race_class in sorted(df['race_class'].unique()):
        class_df = df[(df['race_class'] == race_class) & (df['predicted_rank'] == 1)]
        if len(class_df) > 0:
            correct = len(class_df[class_df['position'] == 1])
            total = len(class_df)
            accuracy = (correct / total * 100)
            print(f"   {race_class:20s}: {accuracy:5.1f}% ({correct}/{total})")
    
    # Show best and worst predictions
    print("\n" + "=" * 80)
    print("BEST PREDICTIONS (Predicted Rank 1, Actually Won)")
    print("=" * 80)
    
    winners = df[(df['predicted_rank'] == 1) & (df['position'] == 1)].head(10)
    if len(winners) > 0:
        print(f"\n{'Date':<12} {'Race':<30} {'Horse':<20} {'Score':<8} {'Odds':<6}")
        print("-" * 80)
        for _, row in winners.iterrows():
            race_short = row['race_name'][:28] + ".." if len(row['race_name']) > 30 else row['race_name']
            horse_short = row['horse'][:18] + ".." if len(row['horse']) > 20 else row['horse']
            print(f"{row['date']:<12} {race_short:<30} {horse_short:<20} "
                  f"{row['predicted_score']:.4f}   {row['win_odds']:.1f}")
    else:
        print("\n   No perfect predictions yet (predicted winner that actually won)")
    
    print("\n" + "=" * 80)
    print("BIGGEST MISSES (Predicted Rank 1, Finished Outside Top 3)")
    print("=" * 80)
    
    misses = df[(df['predicted_rank'] == 1) & (df['position'] > 3)].head(10)
    if len(misses) > 0:
        print(f"\n{'Date':<12} {'Race':<30} {'Horse':<20} {'Pred':<6} {'Actual':<6} {'Odds':<6}")
        print("-" * 85)
        for _, row in misses.iterrows():
            race_short = row['race_name'][:28] + ".." if len(row['race_name']) > 30 else row['race_name']
            horse_short = row['horse'][:18] + ".." if len(row['horse']) > 20 else row['horse']
            print(f"{row['date']:<12} {race_short:<30} {horse_short:<20} "
                  f"{row['predicted_rank']:<6} {row['position']:<6} {row['win_odds']:.1f}")
    else:
        print("\n   No major misses (all predicted winners finished in top 3)")
    
    # Analyze by date to see if model is improving
    print("\n" + "=" * 80)
    print("ACCURACY OVER TIME")
    print("=" * 80)
    
    print("\nTop-1 accuracy by race date:")
    for date in sorted(df['date'].unique()):
        date_df = df[(df['date'] == date) & (df['predicted_rank'] == 1)]
        if len(date_df) > 0:
            correct = len(date_df[date_df['position'] == 1])
            total = len(date_df)
            accuracy = (correct / total * 100)
            races = df[df['date'] == date].groupby('race_name').ngroups
            print(f"   {date}: {accuracy:5.1f}% ({correct}/{total} races, {races} total races)")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nPredicted Rank 1 horses:")
    rank1 = df[df['predicted_rank'] == 1]
    print(f"   Finished 1st: {len(rank1[rank1['position'] == 1])} ({len(rank1[rank1['position'] == 1])/len(rank1)*100:.1f}%)")
    print(f"   Finished 2nd: {len(rank1[rank1['position'] == 2])} ({len(rank1[rank1['position'] == 2])/len(rank1)*100:.1f}%)")
    print(f"   Finished 3rd: {len(rank1[rank1['position'] == 3])} ({len(rank1[rank1['position'] == 3])/len(rank1)*100:.1f}%)")
    print(f"   Finished 4th+: {len(rank1[rank1['position'] > 3])} ({len(rank1[rank1['position'] > 3])/len(rank1)*100:.1f}%)")
    
    print(f"\nPredicted Top-3 horses:")
    top3 = df[df['predicted_rank'] <= 3]
    print(f"   Finished in top 3: {len(top3[top3['position'] <= 3])} ({len(top3[top3['position'] <= 3])/len(top3)*100:.1f}%)")
    print(f"   Finished 4th+: {len(top3[top3['position'] > 3])} ({len(top3[top3['position'] > 3])/len(top3)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return df

def create_race_meeting_csv(db_path="data/historical/hkjc.db", output_dir="data/predictions"):
    """
    Create a CSV file analyzing the most recent race meeting with predictions and results.
    """
    import os
    from datetime import datetime
    
    conn = sqlite3.connect(db_path)
    
    # Get the most recent date with both predictions and results
    query = """
        SELECT DISTINCT r.date
        FROM races r
        JOIN runners run ON r.race_id = run.race_id
        WHERE run.predicted_rank IS NOT NULL
          AND run.position IS NOT NULL
          AND run.position > 0
        ORDER BY r.date DESC
        LIMIT 1
    """
    
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    
    if not result:
        print("\n⚠️  No race meetings with both predictions and results found.")
        return None
    
    latest_date = result[0]
    
    # Get full data for that date
    query = """
        SELECT 
            r.date,
            r.race_name,
            r.class AS race_class,
            r.distance AS dist_m,
            run.horse,
            run.jockey,
            run.trainer,
            run.draw,
            run.win_odds,
            run.position AS actual_position,
            run.predicted_rank,
            run.predicted_score,
            CASE 
                WHEN run.predicted_rank = 1 AND run.position = 1 THEN 'Winner'
                WHEN run.predicted_rank <= 3 AND run.position <= 3 THEN 'Top 3'
                WHEN run.predicted_rank = 1 AND run.position > 3 THEN 'Miss'
                ELSE 'Other'
            END as prediction_result
        FROM runners run
        JOIN races r ON run.race_id = r.race_id
        WHERE r.date = ?
          AND run.predicted_rank IS NOT NULL
          AND run.position IS NOT NULL
          AND run.position > 0
        ORDER BY r.race_name, run.predicted_rank
    """
    
    df = pd.read_sql_query(query, conn, params=(latest_date,))
    conn.close()
    
    # Format the predicted_score as percentage
    df['predicted_score_pct'] = (df['predicted_score'] * 100).round(2).astype(str) + '%'
    
    # Reorder columns for better readability
    output_df = df[[
        'date', 'race_name', 'race_class', 'dist_m',
        'horse', 'jockey', 'trainer', 'draw', 'win_odds',
        'predicted_rank', 'predicted_score_pct', 
        'actual_position', 'prediction_result'
    ]]
    
    # Create output filename
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"analysis_{latest_date}.csv")
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    
    print(f"\n📄 Race meeting analysis saved to: {output_file}")
    print(f"   Date: {latest_date}")
    print(f"   Races: {df['race_name'].nunique()}")
    print(f"   Runners: {len(df)}")
    
    # Show summary stats
    winners = len(df[(df['predicted_rank'] == 1) & (df['actual_position'] == 1)])
    total_races = df['race_name'].nunique()
    top3_correct = len(df[(df['predicted_rank'] <= 3) & (df['actual_position'] <= 3)])
    top3_total = len(df[df['predicted_rank'] <= 3])
    
    print(f"\n   Summary:")
    print(f"   • Predicted winners that won: {winners}/{total_races} ({winners/total_races*100:.1f}%)")
    print(f"   • Predicted top-3 in actual top-3: {top3_correct}/{top3_total} ({top3_correct/top3_total*100:.1f}%)")
    
    # Create text summary
    summary_file = os.path.join(output_dir, f"analysis_{latest_date}_summary.txt")
    create_text_summary(df, latest_date, summary_file)
    
    return output_file

def create_text_summary(df, date, output_file):
    """Create a text summary of the race meeting analysis."""
    
    # Calculate statistics
    total_races = df['race_name'].nunique()
    total_runners = len(df)
    
    # Winners
    winners_df = df[df['predicted_rank'] == 1]
    winners_correct = len(winners_df[winners_df['actual_position'] == 1])
    
    # Top-3
    top3_df = df[df['predicted_rank'] <= 3]
    top3_correct = len(top3_df[top3_df['actual_position'] <= 3])
    
    # Mean absolute error
    mae = (df['predicted_rank'] - df['actual_position']).abs().mean()
    
    # Get successful predictions
    successful = df[(df['predicted_rank'] == 1) & (df['actual_position'] == 1)].sort_values('predicted_score', ascending=False)
    
    # Get biggest misses
    misses = df[(df['predicted_rank'] == 1) & (df['actual_position'] > 3)].sort_values('actual_position', ascending=False)
    
    # By race class
    class_stats = []
    for race_class in sorted(df['race_class'].unique()):
        class_df = df[(df['race_class'] == race_class) & (df['predicted_rank'] == 1)]
        if len(class_df) > 0:
            correct = len(class_df[class_df['actual_position'] == 1])
            class_stats.append((race_class, correct, len(class_df), correct/len(class_df)*100))
    
    # Write summary
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RACE MEETING PREDICTION ANALYSIS - {date}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Date: {date}\n")
        f.write(f"Total Races: {total_races}\n")
        f.write(f"Total Runners Analyzed: {total_runners}\n\n")
        
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Top-1 Accuracy: {winners_correct}/{total_races} ({winners_correct/total_races*100:.1f}%)\n")
        f.write(f"  → Predicted winners that actually won\n")
        f.write(f"  → Random chance would be ~{100/14:.1f}% (1 in 14 horses)\n\n")
        
        f.write(f"Top-3 Accuracy: {top3_correct}/{len(top3_df)} ({top3_correct/len(top3_df)*100:.1f}%)\n")
        f.write(f"  → Predicted top-3 horses that finished in actual top-3\n")
        f.write(f"  → Random chance would be ~20%\n\n")
        
        f.write(f"Mean Absolute Error: {mae:.2f} positions\n")
        f.write(f"  → Average difference between predicted and actual position\n\n")
        
        if len(successful) > 0:
            f.write("SUCCESSFUL PREDICTIONS (Predicted #1 and Won)\n")
            f.write("-" * 80 + "\n")
            for _, row in successful.iterrows():
                f.write(f"✅ {row['horse']}\n")
                f.write(f"   Race: {row['race_name']}\n")
                f.write(f"   Jockey: {row['jockey']} | Trainer: {row['trainer']}\n")
                f.write(f"   Odds: {row['win_odds']:.1f} | Confidence: {row['predicted_score_pct']}\n\n")
        else:
            f.write("SUCCESSFUL PREDICTIONS: None\n\n")
        
        if len(misses) > 0:
            f.write("BIGGEST MISSES (Predicted #1, Finished 4th or Worse)\n")
            f.write("-" * 80 + "\n")
            for _, row in misses.iterrows():
                f.write(f"❌ {row['horse']} - Finished {int(row['actual_position'])}\n")
                f.write(f"   Race: {row['race_name']}\n")
                f.write(f"   Jockey: {row['jockey']} | Trainer: {row['trainer']}\n")
                f.write(f"   Odds: {row['win_odds']:.1f} | Confidence: {row['predicted_score_pct']}\n\n")
        
        if class_stats:
            f.write("PERFORMANCE BY RACE CLASS\n")
            f.write("-" * 80 + "\n")
            for race_class, correct, total, pct in class_stats:
                f.write(f"{race_class:20s}: {correct}/{total} ({pct:.1f}%)\n")
            f.write("\n")
        
        f.write("RACE-BY-RACE SUMMARY\n")
        f.write("-" * 80 + "\n")
        for race_name in df['race_name'].unique():
            race_df = df[df['race_name'] == race_name]
            predicted_winner = race_df[race_df['predicted_rank'] == 1].iloc[0]
            actual_winner = race_df[race_df['actual_position'] == 1].iloc[0]
            
            f.write(f"\n{race_name}\n")
            f.write(f"  Class: {predicted_winner['race_class']} | Distance: {int(predicted_winner['dist_m'])}m\n")
            f.write(f"  Predicted Winner: {predicted_winner['horse']} (odds {predicted_winner['win_odds']:.1f})\n")
            f.write(f"  Actual Winner: {actual_winner['horse']} (odds {actual_winner['win_odds']:.1f})\n")
            
            if predicted_winner['horse'] == actual_winner['horse']:
                f.write(f"  Result: ✅ CORRECT\n")
            else:
                f.write(f"  Result: ❌ Predicted winner finished {int(predicted_winner['actual_position'])}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Calculate some insights
        avg_winning_odds = df[df['actual_position'] == 1]['win_odds'].mean()
        avg_predicted_odds = df[df['predicted_rank'] == 1]['win_odds'].mean()
        
        f.write(f"• Average odds of actual winners: {avg_winning_odds:.1f}\n")
        f.write(f"• Average odds of predicted winners: {avg_predicted_odds:.1f}\n")
        
        if winners_correct > 0:
            successful_odds = successful['win_odds'].mean()
            f.write(f"• Average odds of successful predictions: {successful_odds:.1f}\n")
        
        f.write(f"\n• Prediction accuracy ({winners_correct/total_races*100:.1f}%) is ")
        if winners_correct/total_races > 0.25:
            f.write("EXCELLENT (>25%)\n")
        elif winners_correct/total_races > 0.15:
            f.write("GOOD (15-25%)\n")
        elif winners_correct/total_races > 0.10:
            f.write("AVERAGE (10-15%)\n")
        else:
            f.write("BELOW AVERAGE (<10%)\n")
        
        f.write(f"\n• Top-3 accuracy ({top3_correct/len(top3_df)*100:.1f}%) is ")
        if top3_correct/len(top3_df) > 0.40:
            f.write("EXCELLENT (>40%)\n")
        elif top3_correct/len(top3_df) > 0.30:
            f.write("GOOD (30-40%)\n")
        else:
            f.write("NEEDS IMPROVEMENT (<30%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Analysis generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    print(f"📄 Text summary saved to: {output_file}")

if __name__ == "__main__":
    # Run the full analysis
    analyze_prediction_accuracy()
    
    # Create CSV for the latest race meeting
    create_race_meeting_csv()
