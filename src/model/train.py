"""
XGBoost training pipeline for cart abandonment prediction.

Run once to generate the model artifact:
    python -m src.model.train

Saves model to: models/cart_abandonment_model.json
"""

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.generate import generate
from src.features.engineering import extract_features, features_to_array

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "cart_abandonment_model.json"
TRAINING_VOLUME = 50_000
SEED = 42


def build_training_data(
    volume: int = TRAINING_VOLUME, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    records = generate("mixed", volume, seed)
    X = np.array([
        features_to_array(extract_features(r.__dict__))
        for r in records
    ])
    y = np.array([r.abandoned for r in records], dtype=int)
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=SEED,
        eval_metric="auc",
        verbosity=0,
    )
    model.fit(X, y)
    return model


def main() -> None:
    print(f"Building training data ({TRAINING_VOLUME:,} sessions)...", file=sys.stderr)
    X, y = build_training_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    print("Training XGBoost classifier...", file=sys.stderr)
    model = train_model(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"Test AUC-ROC: {auc:.4f}", file=sys.stderr)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
