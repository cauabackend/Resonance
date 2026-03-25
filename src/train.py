import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "loudness", "speechiness", "acousticness",
    "instrumentalness", "duration_ms",
]


def train_and_save(
    data_path: str = "data/processed/tracks_clean.csv",
    model_path: str = "model/model.pkl",
):
    """Treina XGBoost no dataset processado e salva o modelo."""
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df["is_hit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(model, model_path)
    print(f"\nModelo salvo em {model_path}")

    return model


if __name__ == "__main__":
    train_and_save()