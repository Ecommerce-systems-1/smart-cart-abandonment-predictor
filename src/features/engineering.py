from dataclasses import dataclass

import numpy as np


@dataclass
class SessionFeatures:
    cart_value: float
    item_count: int
    cart_remove_ratio: float
    time_since_last_event_seconds: float
    session_duration_seconds: float
    page_views: int
    checkout_start_attempts: int
    device_type_mobile: int
    referrer_organic: int
    time_on_checkout_seconds: float


def extract_features(session: dict) -> SessionFeatures:
    cart_adds = session.get("cart_adds", 0)
    cart_removes = session.get("cart_removes", 0)
    return SessionFeatures(
        cart_value=float(session.get("cart_value", 0.0)),
        item_count=int(session.get("item_count", 0)),
        cart_remove_ratio=cart_removes / (cart_adds + 1),
        time_since_last_event_seconds=float(session.get("time_since_last_event_seconds", 0.0)),
        session_duration_seconds=float(session.get("session_duration_seconds", 0.0)),
        page_views=int(session.get("page_views", 1)),
        checkout_start_attempts=int(session.get("checkout_start_attempts", 0)),
        device_type_mobile=1 if session.get("device_type") == "mobile" else 0,
        referrer_organic=1 if session.get("referrer_source") == "organic" else 0,
        time_on_checkout_seconds=float(session.get("time_on_checkout_seconds", 0.0)),
    )


def features_to_array(features: SessionFeatures) -> np.ndarray:
    return np.array([
        features.cart_value,
        features.item_count,
        features.cart_remove_ratio,
        features.time_since_last_event_seconds,
        features.session_duration_seconds,
        features.page_views,
        features.checkout_start_attempts,
        features.device_type_mobile,
        features.referrer_organic,
        features.time_on_checkout_seconds,
    ], dtype=np.float64)
