from unittest.mock import MagicMock

import numpy as np
import pytest
import xgboost as xgb

from src.api.dependencies import get_model
from src.api.main import app


@pytest.fixture
def mock_model():
    model = MagicMock(spec=xgb.XGBClassifier)
    model.predict_proba = MagicMock(return_value=np.array([[0.25, 0.75]]))
    return model


@pytest.fixture
def client(mock_model):
    from fastapi.testclient import TestClient
    app.dependency_overrides[get_model] = lambda: mock_model
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def valid_payload():
    return {
        "session_id": "test-session-123",
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
