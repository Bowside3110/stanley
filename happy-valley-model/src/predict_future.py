import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from src.features import build_features, _pick_features, _add_odds_history_metrics
from src.backtest import _actual_top2


def build_features_baseline(db_path: str) -> pd.DataFrame:
    """
    Build features without the odds-history metrics.
    This is a modified version of build_features that skips _add_odds_history_metrics.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        DataFrame with features (without odds-history metrics)
    """
    # Get the full features first
    df = build_features(db_path)
    
    # Drop the odds-history metrics columns
    odds_cols = ['horse_odds_efficiency', 'horse_odds_trend', 'trainer_odds_bias']
    for col in odds_cols:
        if col in df.columns:
            df = df.drop(columns=col)
    
    print("✅ Baseline features (without odds-history metrics)")
    return df


# ----- Build pair dataset -----
def build_pair_dataset(df: pd.DataFrame, runner_feats: list[str], gold=None):
    rows = []
    gold_map = None
    if gold is not None:
        gold_map = dict(zip(gold["race_id"], gold["actual_top2"]))

    for rid, grp in df.groupby("race_id"):
        horses = grp["horse_id"].tolist()
        g = gold_map.get(rid) if gold_map else None
        for i, j in combinations(range(len(horses)), 2):
            hi, hj = horses[i], horses[j]
            ri, rj = grp.iloc[i], grp.iloc[j]
            feats = {}
            for f in runner_feats:
                xi, xj = ri[f], rj[f]
                if pd.notna(xi) or pd.notna(xj):
                    feats[f"{f}_min"] = np.nanmin([xi, xj])
                    feats[f"{f}_max"] = np.nanmax([xi, xj])
                else:
                    feats[f"{f}_min"] = np.nan
                    feats[f"{f}_max"] = np.nan
                feats[f"{f}_diff"] = (
                    abs(xi - xj) if pd.notna(xi) and pd.notna(xj) else np.nan
                )
            feats["same_trainer"] = int(ri.get("trainer_id") == rj.get("trainer_id"))
            feats["same_jockey"] = int(ri.get("jockey_id") == rj.get("jockey_id"))
            feats["race_id"] = rid
            feats["pair"] = tuple(sorted((hi, hj)))
            if gold_map:
                feats["y"] = int(g is not None and set((hi, hj)) == set(g))
            rows.append(feats)
    return pd.DataFrame(rows)


def softmax_pairs(pair_df: pd.DataFrame, score_col="p_raw", temperature=1.0) -> pd.DataFrame:
    """
    Apply softmax normalization to pair scores within each race.
    
    Args:
        pair_df: DataFrame with race_id and score_col
        score_col: Column containing raw scores to normalize
        temperature: Temperature parameter for softmax (lower values increase contrast)
        
    Returns:
        DataFrame with added p_pair column containing normalized probabilities
    """
    out = []
    for rid, g in pair_df.groupby("race_id"):
        s = g[score_col].to_numpy()
        
        # Check for identical values
        if np.std(s) < 1e-6:  # All values essentially identical
            # Use uniform distribution but with tiny variations
            p = np.ones_like(s) / len(s)
            # Add small random variations to break perfect ties
            p = p + np.random.normal(0, 0.001, size=len(p))
            # Renormalize
            p = p / p.sum()
        else:
            # Apply temperature scaling to increase/decrease contrast
            s = s / temperature
            # Standard softmax
            z = np.exp(s - np.max(s))  # Subtract max for numerical stability
            p = z / z.sum() if z.sum() > 0 else np.ones_like(z) / len(z)
        
        gg = g.copy()
        gg["p_pair"] = p
        out.append(gg)
    
    return pd.concat(out, ignore_index=True)


# ----- Performance metrics -----
def calculate_performance_metrics(df_test, predictions, gold):
    """
    Calculate performance metrics for a set of predictions.
    
    Args:
        df_test: Test DataFrame with race_id and horse_id
        predictions: Dictionary mapping race_id to list of (horse_id, score) tuples
        gold: DataFrame with race_id and actual_top2
        
    Returns:
        Dictionary with performance metrics
    """
    # Convert gold to a dictionary for easier lookup
    gold_map = dict(zip(gold["race_id"], gold["actual_top2"]))
    
    # Calculate top-2 accuracy
    top2_correct = 0
    total_races = 0
    
    # For Spearman correlation
    all_predicted_ranks = []
    all_actual_ranks = []
    
    for race_id, horse_scores in predictions.items():
        if race_id not in gold_map:
            continue
            
        total_races += 1
        actual_top2 = gold_map[race_id]
        
        # Sort horses by predicted score
        ranked_horses = sorted(horse_scores, key=lambda x: x[1], reverse=True)
        pred_top2 = tuple(sorted([h for h, _ in ranked_horses[:2]]))
        
        # Check if at least one horse in predicted top-2 is in actual top-2
        if set(pred_top2).intersection(set(actual_top2)):
            top2_correct += 1
            
        # Get actual positions for Spearman correlation
        race_df = df_test[df_test["race_id"] == race_id].copy()
        if "position" not in race_df.columns or race_df["position"].isna().all():
            continue
            
        # Create rank dictionaries
        actual_ranks = {}
        for _, row in race_df.iterrows():
            if pd.notna(row["position"]):
                actual_ranks[row["horse_id"]] = row["position"]
                
        pred_ranks = {h: i+1 for i, (h, _) in enumerate(ranked_horses)}
        
        # Only include horses that have both predicted and actual ranks
        common_horses = set(actual_ranks.keys()).intersection(set(pred_ranks.keys()))
        if len(common_horses) < 3:  # Need at least 3 points for meaningful correlation
            continue
            
        for h in common_horses:
            all_predicted_ranks.append(pred_ranks[h])
            all_actual_ranks.append(actual_ranks[h])
    
    # Calculate metrics
    top2_accuracy = top2_correct / total_races if total_races > 0 else 0
    
    # Calculate Spearman correlation
    if len(all_predicted_ranks) > 5:
        spearman_corr, _ = spearmanr(all_predicted_ranks, all_actual_ranks)
    else:
        spearman_corr = 0
        
    return {
        "top2_accuracy": top2_accuracy,
        "spearman_corr": spearman_corr,
        "total_races": total_races
    }


# ----- Main prediction -----
def predict_future(db_path: str, race_date: str, top_box: int = 5, save_csv: str | None = None, 
                   compare_baseline: bool = False, max_validation_races: int = 500, save_model: bool = True,
                   future_mode: bool = True):
    print(f"Predicting races for {race_date} using database {db_path}...")

    if compare_baseline:
        print("\n=== COMPARISON MODE: Baseline vs. Enhanced Model ===")
        print(f"Using max {max_validation_races} races for validation\n")
    
    # 1. Load features
    print("[1/6] Building features...")
    df = build_features(db_path)
    print(f"    Loaded {len(df)} rows")
    print("    Available columns:", df.columns.tolist())


    # 2. Split train/future
    print("[2/6] Splitting train/future sets...")

    df["race_date"] = pd.to_datetime(df["race_date"])

    # Train = any race with results (position not null)
    df_train = df[df["position"].notna()].copy()

    # Future = any race with no results yet
    df_future = df[df["position"].isna()].copy()

    # If a specific race date is provided, filter to that date
    if race_date:
        race_date = pd.to_datetime(race_date).date()
        df_future = df_future[df_future["race_date"].dt.date == race_date]
        
    # Special handling for future races where all horses are first-time runners
    if future_mode and 'is_first_time_runner' in df_future.columns:
        first_time_pct = df_future['is_first_time_runner'].mean() * 100
        if first_time_pct > 95:  # If more than 95% are first-time runners
            print(f"    ⚠️ Warning: {first_time_pct:.1f}% of future horses are first-time runners!")
            print("    Adjusting predictions to rely more on market odds and less on historical metrics...")
            
            # For future races, adjust the first_time_penalty to be less severe
            # This helps when all horses are first-time runners
            if 'first_time_penalty' in df_future.columns:
                df_future['first_time_penalty'] = df_future['first_time_penalty'] * 0.5  # Reduce penalty by half
                
            # Make market odds a stronger factor for these races
            if 'market_prob' in df_future.columns:
                # Scale market probability to have more influence
                df_future['market_prob_scaled'] = df_future['market_prob'] * 1.5
                print("    Added market_prob_scaled feature to increase odds influence")
                
            # Add a random component based on draw for variety
            if 'draw' in df_future.columns:
                # Convert draw to numeric if it's not already
                if not pd.api.types.is_numeric_dtype(df_future['draw']):
                    df_future['draw'] = pd.to_numeric(df_future['draw'], errors='coerce')
                
                # Create a small random factor based on draw
                np.random.seed(42)  # For reproducibility
                df_future['draw_factor'] = df_future['draw'].apply(
                    lambda x: np.random.normal(0, 0.05) if pd.isna(x) else np.random.normal(0, 0.05, 1)[0]
                )
                print("    Added draw_factor for slight randomization in all-first-time fields")

    if df_future.empty:
        print(f"No future races found for {race_date}")
        return

    print(f"    Train races: {df_train['race_id'].nunique()}, Future races: {df_future['race_id'].nunique()}")


    # 3. Runner-level features
    print("[3/6] Selecting features...")
    
    # If in comparison mode, we need to create a validation set
    if compare_baseline:
        # Use a time-based split for validation
        from sklearn.model_selection import train_test_split
        
        # Sort by date to ensure chronological split
        df_sorted = df_train.sort_values("race_date")
        
        # Limit to a reasonable number of races for validation
        race_ids = df_sorted["race_id"].unique()
        if len(race_ids) > max_validation_races:
            validation_races = race_ids[-max_validation_races:]
            train_races = race_ids[:-max_validation_races]
        else:
            # If we don't have enough races, use 20% for validation
            train_races, validation_races = train_test_split(
                race_ids, test_size=0.2, random_state=42
            )
            
        df_train_subset = df_train[df_train["race_id"].isin(train_races)].copy()
        df_validation = df_train[df_train["race_id"].isin(validation_races)].copy()
        
        print(f"    Created validation set with {len(validation_races)} races")
        print(f"    Training set: {len(train_races)} races, {len(df_train_subset)} rows")
        print(f"    Validation set: {len(validation_races)} races, {len(df_validation)} rows")
    else:
        df_train_subset = df_train
    
    # Select features for the enhanced model
    runner_feats = _pick_features(df_train_subset)
    
    # Add the new market_prob_scaled and draw_factor features if they exist in df_future
    if 'market_prob_scaled' in df_future.columns and 'market_prob_scaled' not in runner_feats:
        runner_feats.append('market_prob_scaled')
        print("    Added market_prob_scaled to feature list")
        
    if 'draw_factor' in df_future.columns and 'draw_factor' not in runner_feats:
        runner_feats.append('draw_factor')
        print("    Added draw_factor to feature list")
    
    # Verify inclusion of odds-history metrics
    odds_features = ["horse_odds_efficiency", "horse_odds_trend", "trainer_odds_bias"]
    included_odds_features = [f for f in odds_features if f in runner_feats]
    print(f"    Using {len(runner_feats)} features (including {len(included_odds_features)}/{len(odds_features)} odds-history metrics)")
    if len(included_odds_features) < len(odds_features):
        missing = set(odds_features) - set(included_odds_features)
        print(f"    ⚠️ Warning: Missing odds features: {', '.join(missing)}")
    else:
        print(f"    ✅ All odds-history metrics included: {', '.join(included_odds_features)}")

    # 4. Build pair datasets
    print("[4/6] Building pair datasets...")
    gold = _actual_top2(df_train)
    
    if compare_baseline:
        # For validation metrics
        validation_gold = _actual_top2(df_validation)
        
        # Build datasets for both models
        print("\n    Building datasets for baseline model...")
        # For baseline model (without odds-history metrics)
        baseline_runner_feats = [f for f in runner_feats if f not in odds_features]
        baseline_train_pairs = build_pair_dataset(df_train_subset, baseline_runner_feats, gold)
        baseline_validation_pairs = build_pair_dataset(df_validation, baseline_runner_feats, validation_gold)
        baseline_test_pairs = build_pair_dataset(df_future, baseline_runner_feats)
        
        print(f"    Baseline train pairs: {len(baseline_train_pairs)}")
        print(f"    Baseline validation pairs: {len(baseline_validation_pairs)}")
        print(f"    Baseline future pairs: {len(baseline_test_pairs)}")
        
        print("\n    Building datasets for enhanced model...")
        # For enhanced model (with odds-history metrics)
        enhanced_train_pairs = build_pair_dataset(df_train_subset, runner_feats, gold)
        enhanced_validation_pairs = build_pair_dataset(df_validation, runner_feats, validation_gold)
        enhanced_test_pairs = build_pair_dataset(df_future, runner_feats)
        
        print(f"    Enhanced train pairs: {len(enhanced_train_pairs)}")
        print(f"    Enhanced validation pairs: {len(enhanced_validation_pairs)}")
        print(f"    Enhanced future pairs: {len(enhanced_test_pairs)}")
        
        train_pairs = enhanced_train_pairs
        test_pairs = enhanced_test_pairs
    else:
        # Standard mode - just build one set of datasets
        train_pairs = build_pair_dataset(df_train, runner_feats, gold)
        test_pairs = build_pair_dataset(df_future, runner_feats)
        print(f"    Train pairs: {len(train_pairs)}, Future pairs: {len(test_pairs)}")

    # 5. Train DirectPair model
    print("[5/6] Training model...")
    
    if compare_baseline:
        # ----- Train both models and compare -----
        print("\n    Training baseline model (without odds-history metrics)...")
        X_baseline_tr = baseline_train_pairs.drop(columns=["race_id", "pair", "y"])
        y_baseline_tr = baseline_train_pairs["y"]
        X_baseline_val = baseline_validation_pairs.drop(columns=["race_id", "pair", "y"])
        y_baseline_val = baseline_validation_pairs["y"]
        X_baseline_te = baseline_test_pairs.drop(columns=["race_id", "pair"], errors="ignore")
        
        # Train baseline model
        baseline_model = HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.05, max_iter=300,
            validation_fraction=0.1, random_state=42
        )
        baseline_model.fit(X_baseline_tr, y_baseline_tr)
        
        # Score validation set with baseline model
        baseline_validation_pairs = baseline_validation_pairs.copy()
        baseline_validation_pairs["p_raw"] = baseline_model.predict_proba(X_baseline_val)[:, 1]
        baseline_dir_df = softmax_pairs(baseline_validation_pairs[["race_id", "pair", "p_raw"]], temperature=0.3)
        
        # Score future races with baseline model
        baseline_test_pairs = baseline_test_pairs.copy()
        baseline_test_pairs["p_raw"] = baseline_model.predict_proba(X_baseline_te)[:, 1]
        baseline_future_df = softmax_pairs(baseline_test_pairs[["race_id", "pair", "p_raw"]], temperature=0.3)
        
        # ----- Now train enhanced model -----
        print("\n    Training enhanced model (with odds-history metrics)...")
        X_enhanced_tr = enhanced_train_pairs.drop(columns=["race_id", "pair", "y"])
        y_enhanced_tr = enhanced_train_pairs["y"]
        X_enhanced_val = enhanced_validation_pairs.drop(columns=["race_id", "pair", "y"])
        y_enhanced_val = enhanced_validation_pairs["y"]
        X_enhanced_te = enhanced_test_pairs.drop(columns=["race_id", "pair"], errors="ignore")
        
        # Train enhanced model
        enhanced_model = HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.05, max_iter=300,
            validation_fraction=0.1, random_state=42
        )
        enhanced_model.fit(X_enhanced_tr, y_enhanced_tr)
        
        # Score validation set with enhanced model
        enhanced_validation_pairs = enhanced_validation_pairs.copy()
        enhanced_validation_pairs["p_raw"] = enhanced_model.predict_proba(X_enhanced_val)[:, 1]
        enhanced_dir_df = softmax_pairs(enhanced_validation_pairs[["race_id", "pair", "p_raw"]], temperature=0.3)
        
        # Score future races with enhanced model
        enhanced_test_pairs = enhanced_test_pairs.copy()
        enhanced_test_pairs["p_raw"] = enhanced_model.predict_proba(X_enhanced_te)[:, 1]
        enhanced_future_df = softmax_pairs(enhanced_test_pairs[["race_id", "pair", "p_raw"]], temperature=0.3)
        
        # ----- Analyze feature importance -----
        print("\n=== Feature Impact Analysis ===")
        
        # For baseline model
        baseline_feat_imp = None
        if hasattr(baseline_model, 'feature_importances_'):
            baseline_importances = baseline_model.feature_importances_
            baseline_feat_imp = pd.DataFrame({
                'feature': X_baseline_tr.columns,
                'importance': baseline_importances
            }).sort_values('importance', ascending=False)
            
            print("\nBaseline model - Top 10 feature importances:")
            for i, (feature, importance) in enumerate(zip(baseline_feat_imp['feature'].head(10), 
                                                         baseline_feat_imp['importance'].head(10))):
                print(f"  {i+1:2d}. {feature:30s}: {importance:.4f}")
        else:
            # Try using permutation importance for baseline model
            try:
                from sklearn.inspection import permutation_importance
                print("\nCalculating permutation importance for baseline model...")
                # Use a small subset for speed
                sample_size = min(5000, X_baseline_val.shape[0])
                X_sample = X_baseline_val.sample(sample_size, random_state=42)
                y_sample = y_baseline_val.loc[X_sample.index]
                
                perm_importance = permutation_importance(
                    baseline_model, X_sample, y_sample, 
                    n_repeats=3, random_state=42
                )
                
                baseline_importances = perm_importance.importances_mean
                baseline_feat_imp = pd.DataFrame({
                    'feature': X_baseline_tr.columns,
                    'importance': baseline_importances
                }).sort_values('importance', ascending=False)
                
                print("\nBaseline model - Top 10 feature importances (permutation):")
                for i, (feature, importance) in enumerate(zip(baseline_feat_imp['feature'].head(10), 
                                                             baseline_feat_imp['importance'].head(10))):
                    print(f"  {i+1:2d}. {feature:30s}: {importance:.4f}")
            except Exception as e:
                print(f"\nError calculating permutation importance: {e}")
        
        # For enhanced model
        enhanced_feat_imp = None
        if hasattr(enhanced_model, 'feature_importances_'):
            enhanced_importances = enhanced_model.feature_importances_
            enhanced_feat_imp = pd.DataFrame({
                'feature': X_enhanced_tr.columns,
                'importance': enhanced_importances
            }).sort_values('importance', ascending=False)
            
            print("\nEnhanced model - Top 10 feature importances:")
            for i, (feature, importance) in enumerate(zip(enhanced_feat_imp['feature'].head(10), 
                                                         enhanced_feat_imp['importance'].head(10))):
                print(f"  {i+1:2d}. {feature:30s}: {importance:.4f}")
            
            # Check odds-history metrics
            odds_features_expanded = []
            for f in ["horse_odds_efficiency", "horse_odds_trend", "trainer_odds_bias"]:
                odds_features_expanded.extend([f"{f}_min", f"{f}_max", f"{f}_diff"])
            
            odds_importances = enhanced_feat_imp[enhanced_feat_imp['feature'].isin(odds_features_expanded)]
            if not odds_importances.empty:
                print("\nOdds-history metrics importances:")
                for i, (feature, importance) in enumerate(zip(odds_importances['feature'], odds_importances['importance'])):
                    print(f"  {i+1:2d}. {feature:30s}: {importance:.4f}")
                
                # Calculate total importance of odds features
                total_imp = odds_importances['importance'].sum()
                print(f"\nTotal importance of odds-history metrics: {total_imp:.4f} ({total_imp/sum(enhanced_importances)*100:.2f}%)")
            else:
                print("\n⚠️ No odds-history metrics found in feature importances!")
        else:
            # Try using permutation importance for enhanced model
            try:
                from sklearn.inspection import permutation_importance
                print("\nCalculating permutation importance for enhanced model...")
                # Use a small subset for speed
                sample_size = min(5000, X_enhanced_val.shape[0])
                X_sample = X_enhanced_val.sample(sample_size, random_state=42)
                y_sample = y_enhanced_val.loc[X_sample.index]
                
                perm_importance = permutation_importance(
                    enhanced_model, X_sample, y_sample, 
                    n_repeats=3, random_state=42
                )
                
                enhanced_importances = perm_importance.importances_mean
                enhanced_feat_imp = pd.DataFrame({
                    'feature': X_enhanced_tr.columns,
                    'importance': enhanced_importances
                }).sort_values('importance', ascending=False)
                
                print("\nEnhanced model - Top 10 feature importances (permutation):")
                for i, (feature, importance) in enumerate(zip(enhanced_feat_imp['feature'].head(10), 
                                                             enhanced_feat_imp['importance'].head(10))):
                    print(f"  {i+1:2d}. {feature:30s}: {importance:.4f}")
                
                # Check odds-history metrics
                odds_features_expanded = []
                for f in ["horse_odds_efficiency", "horse_odds_trend", "trainer_odds_bias"]:
                    odds_features_expanded.extend([f"{f}_min", f"{f}_max", f"{f}_diff"])
                
                odds_importances = enhanced_feat_imp[enhanced_feat_imp['feature'].isin(odds_features_expanded)]
                if not odds_importances.empty:
                    print("\nOdds-history metrics importances (permutation):")
                    for i, (feature, importance) in enumerate(zip(odds_importances['feature'], odds_importances['importance'])):
                        print(f"  {i+1:2d}. {feature:30s}: {importance:.4f}")
                    
                    # Calculate total importance of odds features
                    total_imp = odds_importances['importance'].sum()
                    print(f"\nTotal importance of odds-history metrics: {total_imp:.4f} ({total_imp/sum(enhanced_importances)*100:.2f}%)")
                else:
                    print("\n⚠️ No odds-history metrics found in feature importances!")
            except Exception as e:
                print(f"\nError calculating permutation importance: {e}")
        
        # ----- Calculate performance metrics -----
        print("\n=== Model Performance Comparison ===")
        
        # Convert pair predictions to horse scores
        baseline_horse_scores = {}
        for rid, g in baseline_dir_df.groupby("race_id"):
            scores = {}
            for _, r in g.iterrows():
                h1, h2 = r["pair"]
                scores[h1] = scores.get(h1, 0) + r["p_pair"]
                scores[h2] = scores.get(h2, 0) + r["p_pair"]
            baseline_horse_scores[rid] = [(h, s) for h, s in scores.items()]
        
        enhanced_horse_scores = {}
        for rid, g in enhanced_dir_df.groupby("race_id"):
            scores = {}
            for _, r in g.iterrows():
                h1, h2 = r["pair"]
                scores[h1] = scores.get(h1, 0) + r["p_pair"]
                scores[h2] = scores.get(h2, 0) + r["p_pair"]
            enhanced_horse_scores[rid] = [(h, s) for h, s in scores.items()]
        
        # Calculate metrics
        baseline_metrics = calculate_performance_metrics(df_validation, baseline_horse_scores, validation_gold)
        enhanced_metrics = calculate_performance_metrics(df_validation, enhanced_horse_scores, validation_gold)
        
        # Print comparison
        baseline_top2 = baseline_metrics["top2_accuracy"]
        enhanced_top2 = enhanced_metrics["top2_accuracy"]
        top2_diff = enhanced_top2 - baseline_top2
        top2_pct = (top2_diff / baseline_top2) * 100 if baseline_top2 > 0 else float('inf')
        
        baseline_spearman = baseline_metrics["spearman_corr"]
        enhanced_spearman = enhanced_metrics["spearman_corr"]
        spearman_diff = enhanced_spearman - baseline_spearman
        
        print(f"Baseline top-2 accuracy: {baseline_top2:.2f}")
        print(f"Enhanced top-2 accuracy: {enhanced_top2:.2f} ({'+' if top2_diff >= 0 else ''}{top2_diff:.2f}, {'+' if top2_pct >= 0 else ''}{top2_pct:.1f}%)")
        print(f"Baseline Spearman ρ: {baseline_spearman:.2f}")
        print(f"Enhanced Spearman ρ: {enhanced_spearman:.2f} ({'+' if spearman_diff >= 0 else ''}{spearman_diff:.2f})")
        print(f"Total races evaluated: {baseline_metrics['total_races']}")
        
        # Use the enhanced model for future predictions
        pair_model = enhanced_model
        test_pairs = enhanced_test_pairs
        X_te = X_enhanced_te
        
    else:
        # ----- Standard mode - just train one model -----
        X_tr = train_pairs.drop(columns=["race_id", "pair", "y"])
        y_tr = train_pairs["y"]
        X_te = test_pairs.drop(columns=["race_id", "pair"], errors="ignore")
        
        # Check if our new features are present in the training data
        odds_features_expanded = []
        for f in ["horse_odds_efficiency", "horse_odds_trend", "trainer_odds_bias"]:
            odds_features_expanded.extend([f"{f}_min", f"{f}_max", f"{f}_diff"])
        
        odds_cols_present = [col for col in odds_features_expanded if col in X_tr.columns]
        print(f"\n    Odds-history features in training data: {len(odds_cols_present)}/{len(odds_features_expanded)}")
        if len(odds_cols_present) < len(odds_features_expanded):
            print(f"    ⚠️ Missing odds features: {set(odds_features_expanded) - set(odds_cols_present)}")
        else:
            print(f"    ✅ All odds features present in training data")
        
        # Check for zero variance in features
        zero_var_cols = []
        for col in X_tr.columns:
            if X_tr[col].nunique() <= 1:
                zero_var_cols.append(col)
        
        if zero_var_cols:
            print(f"\n    ⚠️ Warning: {len(zero_var_cols)} features have zero variance!")
            print(f"    Zero variance features: {zero_var_cols[:5]}{'...' if len(zero_var_cols) > 5 else ''}")
        
        # Train the model with slightly adjusted parameters for better differentiation
        pair_model = HistGradientBoostingClassifier(
            max_depth=5,  # Increased from 4 to 5 for more complex patterns
            learning_rate=0.05, 
            max_iter=300,
            validation_fraction=0.1, 
            random_state=42
        )
        pair_model.fit(X_tr, y_tr)
        
        # Use built-in feature importances if available
        if hasattr(pair_model, 'feature_importances_'):
            importances = pair_model.feature_importances_
            feat_imp = pd.DataFrame({
                'feature': X_tr.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Print top 15 features
            print("\n    Top 15 feature importances:")
            for i, (feature, importance) in enumerate(zip(feat_imp['feature'].head(15), feat_imp['importance'].head(15))):
                print(f"      {i+1:2d}. {feature:30s}: {importance:.4f}")
            
            # Check odds-history metrics
            odds_importances = feat_imp[feat_imp['feature'].isin(odds_features_expanded)]
            if not odds_importances.empty:
                print("\n    Odds-history metrics importances:")
                for i, (feature, importance) in enumerate(zip(odds_importances['feature'], odds_importances['importance'])):
                    print(f"      {i+1:2d}. {feature:30s}: {importance:.4f}")
                
                # Calculate total importance of odds features
                total_imp = odds_importances['importance'].sum()
                print(f"\n    Total importance of odds-history metrics: {total_imp:.4f} ({total_imp/sum(importances)*100:.2f}%)")
            else:
                print("\n    ⚠️ No odds-history metrics found in feature importances!")
        else:
            print("\n    ⚠️ Model doesn't expose feature_importances_ attribute")

    print("\n    Scoring future races...")
    test_pairs = test_pairs.copy()
    test_pairs["p_raw"] = pair_model.predict_proba(X_te)[:, 1]
    
    # Debug: Check if raw predictions have variation
    print("\n    Checking raw prediction scores:")
    raw_stats = test_pairs.groupby("race_id")["p_raw"].agg(["min", "max", "mean", "std"])
    print(f"    Raw prediction stats by race:\n{raw_stats}")
    
    # Check if there are any races with identical predictions
    identical_races = []
    for rid, g in test_pairs.groupby("race_id"):
        if g["p_raw"].nunique() == 1:
            identical_races.append(rid)
    
    if identical_races:
        print(f"\n    ⚠️ WARNING: {len(identical_races)} races have identical raw predictions for all pairs!")
        for rid in identical_races[:3]:  # Show first 3 races with issues
            print(f"      Race ID: {rid}, value: {test_pairs[test_pairs['race_id'] == rid]['p_raw'].iloc[0]}")
    else:
        print("\n    ✅ All races have variation in raw prediction scores")
    
    # Add small random noise to break ties if needed
    if identical_races:
        print("\n    Adding small random noise to break ties...")
        np.random.seed(42)
        test_pairs["p_raw"] = test_pairs["p_raw"] + np.random.normal(0, 0.0001, size=len(test_pairs))
    
    # Use a lower temperature (0.3) to increase contrast between scores
    print("\n    Applying softmax with temperature=0.3 for better differentiation...")
    dir_df = softmax_pairs(test_pairs[["race_id", "pair", "p_raw"]], temperature=0.3)

    # 6. Cheat sheet: Ranked horses by pairwise strength
    print("[6/6] Generating cheat sheet...")
    cheat_rows = []
    print("\n=== Happy Valley Quinella Cheat Sheet (DirectPair) ===")
    for rid, g in dir_df.groupby("race_id"):
        race_name = df_future[df_future["race_id"] == rid]["race_name"].iloc[0]
        scores = {}
        for _, r in g.iterrows():
            h1, h2 = r["pair"]
            scores[h1] = scores.get(h1, 0) + r["p_pair"]
            scores[h2] = scores.get(h2, 0) + r["p_pair"]
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_box]
        names = df_future[df_future["race_id"] == rid].set_index("horse_id").loc[[h for h, _ in ranked], "horse"].tolist()

        print(f"\nRace: {race_name}")
        print("Ranked selections:")
        for i, (h, score) in enumerate(ranked, 1):
            horse_name = df_future[df_future["horse_id"] == h]["horse"].iloc[0]
            print(f" {i}. {horse_name} ({score:.3f})")
            cheat_rows.append({"race_id": rid, "race_name": race_name, "rank": i, "horse": horse_name, "score": score})

    # 7. Optional CSV export
    if save_csv:
        rows = []
        used_meta_cols = set()

        for rid, g in dir_df.groupby("race_id"):
            # collect scores for every horse
            scores = {}
            for _, r in g.iterrows():
                h1, h2 = r["pair"]
                scores[h1] = scores.get(h1, 0) + r["p_pair"]
                scores[h2] = scores.get(h2, 0) + r["p_pair"]

            # pick whatever race metadata exists in df_future
            candidate_meta_cols = [
                "race_id", "race_name", "class", "race_class",
                "distance", "dist_m", "going"
            ]
            meta_cols = [c for c in candidate_meta_cols if c in df_future.columns]
            used_meta_cols.update(meta_cols)

            if meta_cols:
                race_meta = (
                    df_future[df_future["race_id"] == rid][meta_cols]
                    .drop_duplicates()
                    .iloc[0]
                    .to_dict()
                )
            else:
                race_meta = {"race_id": rid}

            # join with runner features + add score, rank, and status
            race_df = df_future[df_future["race_id"] == rid].copy()
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            for rank, (horse_id, score) in enumerate(ranked, 1):
                runner_row = race_df[race_df["horse_id"] == horse_id].iloc[0].to_dict()
                runner_row.update(race_meta)
                runner_row["score"] = score
                runner_row["rank"] = rank
                # NEW: carry status into the output
                if "status" in runner_row:
                    runner_row["status"] = runner_row["status"]
                rows.append(runner_row)

        out_df = pd.DataFrame(rows)
        out_df = out_df.sort_values(["race_id", "rank"])
        out_df.to_csv(save_csv, index=False)

        print(f"\n✅ Detailed predictions saved to {save_csv}")
        print(f"   Included race metadata columns: {sorted(list(used_meta_cols))}")
        
    # Save the model for later analysis if requested
    if save_model:
        try:
            from pathlib import Path
            import pickle
            import os
            
            # Create models directory if it doesn't exist
            models_dir = Path("data/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the model with a timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = models_dir / f"model_{race_date}_{timestamp}.pkl"
            
            with open(model_path, "wb") as f:
                pickle.dump(pair_model, f)
                
            print(f"\n✅ Model saved to {model_path}")
        except Exception as e:
            print(f"\n⚠️ Failed to save model: {e}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="data/historical/hkjc.db",
                    help="Path to SQLite database with race data")
    ap.add_argument("--date", type=str, required=True,
                    help="Race date in YYYY-MM-DD format (e.g. 2025-09-10)")
    ap.add_argument("--box", type=int, default=5,
                    help="Number of horses to include in ranking (default=5)")
    ap.add_argument("--save_csv", type=str, default=None,
                    help="Optional path to save cheat sheet as CSV")
    ap.add_argument("--compare_baseline", action="store_true",
                    help="Run comparison between baseline model (without odds-history metrics) and enhanced model")
    ap.add_argument("--max_validation_races", type=int, default=500,
                    help="Maximum number of races to use for validation (default=500)")
    ap.add_argument("--save_model", action="store_true", default=True,
                    help="Save the trained model for later analysis (default=True)")
    ap.add_argument("--no_future_mode", action="store_false", dest="future_mode", default=True,
                    help="Disable special handling for future races (default: enabled)")
    args = ap.parse_args()

    predict_future(
        args.db, 
        args.date, 
        top_box=args.box, 
        save_csv=args.save_csv, 
        compare_baseline=args.compare_baseline,
        max_validation_races=args.max_validation_races,
        save_model=args.save_model,
        future_mode=args.future_mode
    )
