def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_200(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_predict_response_contains_session_id(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.json()["session_id"] == "test-session-123"


def test_predict_response_risk_score_in_range(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    score = response.json()["risk_score"]
    assert 0.0 <= score <= 1.0


def test_predict_response_high_risk_true_when_score_gte_06(client, valid_payload):
    # mock returns 0.75 risk → high_risk should be True
    response = client.post("/predict", json=valid_payload)
    assert response.json()["high_risk"] is True


def test_predict_response_incentive_is_valid_enum(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    valid_incentives = {"none", "urgency_message", "free_shipping", "discount_10_percent"}
    assert response.json()["incentive"] in valid_incentives


def test_predict_missing_session_id_returns_422(client, valid_payload):
    del valid_payload["session_id"]
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_negative_cart_value_returns_422(client, valid_payload):
    valid_payload["cart_value"] = -10.0
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_zero_page_views_returns_422(client, valid_payload):
    valid_payload["page_views"] = 0
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_empty_session_id_still_returns_200(client, valid_payload):
    valid_payload["session_id"] = ""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
