import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np

from src.data_ingestion import load_hkjc_runners
from src.feature_engineer import (
    filter_to_hk_races,
    add_basic_features,
    add_jockey_trainer_features,
)


def prepare_dataset():
    """Load and engineer features for HKJC scraped runners."""
    df = load_hkjc_runners()
    df = filter_to_hk_races(df)
    df = add_basic_features(df)
    df = add_jockey_trainer_features(df)

    # Drop rows without target
    df = df.dropna(subset=["is_place"])
    return df


def get_preprocessor(X):
    """Build preprocessing pipeline (imputers + one-hot encoder)."""
    categorical = ["draw_cat"]
    numeric = [col for col in X.columns if col not in categorical]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("pass", "passthrough"),
                ]),
                numeric,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ]
    )


def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test, preprocessor):
    """Train a given model and print results."""
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n=== {model_name} ===")
    print("ROC AUC:", round(auc, 3))
    print(classification_report(y_test, (y_pred_proba > 0.5).astype(int)))

    # Feature importance (if supported)
    if model_name in ["Random Forest", "XGBoost"]:
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                ohe = trans.named_steps["onehot"]
                cats = ohe.categories_[0]
                feature_names.extend([f"{cols[0]}_{cat}" for cat in cats])

        importances = clf.named_steps["classifier"].feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:10]

        print("\nTop 10 Feature Importances:")
        for idx in sorted_idx:
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

    return clf, auc


if __name__ == "__main__":
    print("Loading and preparing HKJC dataset...")
    df = prepare_dataset()

    # Features: adjust to scraped schema
    features = [
        "avg_last3", "days_since_last", "horse_no", "draw_cat",
        "odds", "act_wt",
        "jockey_win_rate", "trainer_win_rate", "jt_combo_win_rate"
    ]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df["is_place"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = get_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, eval_metric="logloss"
        ),
    }

    results = {}
    for name, model in models.items():
        clf, auc = train_and_evaluate(name, model, X_train, X_test, y_train, y_test, preprocessor)
        results[name] = auc

    print("\n=== Model Comparison ===")
    for name, auc in results.items():
        print(f"{name}: ROC AUC = {auc:.3f}")
