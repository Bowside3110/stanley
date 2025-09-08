import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def prepare_dataset(db_path="data/historical/hkjc.db"):
    conn = sqlite3.connect(db_path)

    query = """
    SELECT r.date as race_date,
           r.course,
           r.race_name,
           r.class as race_class,
           r.distance,
           r.going,
           ru.race_id,
           ru.horse_id,
           ru.horse,
           ru.draw,
           ru.weight,
           ru.jockey,
           ru.trainer,
           ru.win_odds,
           re.position
    FROM runners ru
    JOIN races r ON ru.race_id = r.race_id
    JOIN results re ON ru.race_id = re.race_id AND ru.horse_id = re.horse_id
    WHERE r.course IN ('Sha Tin (HK)', 'Happy Valley (HK)')
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def train_model(db_path="data/historical/hkjc.db"):
    df = prepare_dataset(db_path)

    # Simple features: odds + draw + weight
    X = df[["win_odds", "draw", "weight"]].fillna(0)
    y = (df["position"] <= 2).astype(int)  # 1 if horse placed in top 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.2f}")

    return model
