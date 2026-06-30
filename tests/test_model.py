import numpy as np
import xgboost as xgb

from src.features.engineering import extract_features
from src.model.predict import Incentive, Prediction, predict, recommend_incentive
from src.model.train import build_training_data, train_model

VALID_SESSION = {
    "cart_value": 89.99,
    "item_count": 3,
    "cart_adds": 4,
    "cart_removes": 1,
    "time_since_last_event_seconds": 45.0,
    "session_duration_seconds": 420.0,
    "page_views": 8,
    "checkout_start_attempts": 1,
    "device_type": "mobile",
    "referrer_source": "organic",
    "time_on_checkout_seconds": 60.0,
}


# --- Incentive threshold tests ---

def test_recommend_incentive_none_for_low_risk():
    assert recommend_incentive(0.0) == Incentive.NONE
    assert recommend_incentive(0.29) == Incentive.NONE


def test_recommend_incentive_urgency_for_mild_risk():
    assert recommend_incentive(0.3) == Incentive.URGENCY
    assert recommend_incentive(0.59) == Incentive.URGENCY


def test_recommend_incentive_free_shipping_for_moderate_risk():
    assert recommend_incentive(0.6) == Incentive.FREE_SHIPPING
    assert recommend_incentive(0.79) == Incentive.FREE_SHIPPING


def test_recommend_incentive_discount_for_high_risk():
    assert recommend_incentive(0.8) == Incentive.DISCOUNT_10
    assert recommend_incentive(1.0) == Incentive.DISCOUNT_10


# --- Training pipeline tests ---

def test_build_training_data_returns_x_y():
    X, y = build_training_data(volume=200, seed=42)
    assert X.shape == (200, 10)
    assert y.shape == (200,)


def test_build_training_data_y_is_binary():
    _, y = build_training_data(volume=200, seed=42)
    assert set(y.tolist()) == {0, 1}


def test_build_training_data_reproducible():
    X1, y1 = build_training_data(volume=100, seed=99)
    X2, y2 = build_training_data(volume=100, seed=99)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_train_model_returns_xgb_classifier():
    X, y = build_training_data(volume=500, seed=42)
    model = train_model(X, y)
    assert isinstance(model, xgb.XGBClassifier)


def test_train_model_predict_proba_shape():
    X, y = build_training_data(volume=500, seed=42)
    model = train_model(X, y)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 2)


def test_train_model_auc_above_threshold():
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    X, y = build_training_data(volume=2000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    model = train_model(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    assert auc >= 0.75, f"AUC {auc:.3f} below minimum threshold 0.75"


# --- predict() function tests ---

def test_predict_returns_prediction():
    X, y = build_training_data(volume=500, seed=42)
    model = train_model(X, y)
    features = extract_features(VALID_SESSION)
    result = predict(model, features)
    assert isinstance(result, Prediction)


def test_predict_risk_score_in_range():
    X, y = build_training_data(volume=500, seed=42)
    model = train_model(X, y)
    features = extract_features(VALID_SESSION)
    result = predict(model, features)
    assert 0.0 <= result.risk_score <= 1.0


def test_predict_incentive_consistent_with_score():
    X, y = build_training_data(volume=500, seed=42)
    model = train_model(X, y)
    features = extract_features(VALID_SESSION)
    result = predict(model, features)
    assert result.incentive == recommend_incentive(result.risk_score)
