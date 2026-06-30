import numpy as np
import pytest
from src.features.engineering import SessionFeatures, extract_features, features_to_array


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


def test_extract_features_returns_session_features():
    features = extract_features(VALID_SESSION)
    assert isinstance(features, SessionFeatures)


def test_extract_features_cart_value():
    features = extract_features(VALID_SESSION)
    assert features.cart_value == 89.99


def test_extract_features_cart_remove_ratio():
    # cart_removes=1, cart_adds=4 -> ratio = 1 / (4+1) = 0.2
    features = extract_features(VALID_SESSION)
    assert abs(features.cart_remove_ratio - 0.2) < 1e-6


def test_extract_features_remove_ratio_zero_adds():
    # cart_removes=0, cart_adds=0 -> ratio = 0 / (0+1) = 0 (no division by zero)
    session = {**VALID_SESSION, "cart_adds": 0, "cart_removes": 0}
    features = extract_features(session)
    assert features.cart_remove_ratio == 0.0


def test_extract_features_device_mobile():
    features = extract_features(VALID_SESSION)
    assert features.device_type_mobile == 1


def test_extract_features_device_desktop():
    session = {**VALID_SESSION, "device_type": "desktop"}
    features = extract_features(session)
    assert features.device_type_mobile == 0


def test_extract_features_device_tablet_not_mobile():
    session = {**VALID_SESSION, "device_type": "tablet"}
    features = extract_features(session)
    assert features.device_type_mobile == 0


def test_extract_features_referrer_organic():
    features = extract_features(VALID_SESSION)
    assert features.referrer_organic == 1


def test_extract_features_referrer_paid_not_organic():
    session = {**VALID_SESSION, "referrer_source": "paid"}
    features = extract_features(session)
    assert features.referrer_organic == 0


def test_features_to_array_shape():
    features = extract_features(VALID_SESSION)
    arr = features_to_array(features)
    assert arr.shape == (10,)


def test_features_to_array_dtype():
    features = extract_features(VALID_SESSION)
    arr = features_to_array(features)
    assert arr.dtype == np.float64


def test_features_to_array_values_match():
    features = extract_features(VALID_SESSION)
    arr = features_to_array(features)
    assert arr[0] == features.cart_value
    assert arr[1] == features.item_count
    assert abs(arr[2] - features.cart_remove_ratio) < 1e-6
    assert arr[7] == features.device_type_mobile
    assert arr[8] == features.referrer_organic
    assert arr[9] == features.time_on_checkout_seconds
