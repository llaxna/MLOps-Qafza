from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_ORDER = {
    "order_purchase_timestamp": "2018-06-21T08:29:29",
    "order_estimated_delivery_date": "2018-07-17T00:00:00",
    "item_count": 1,
    "product_count": 1,
    "seller_count": 1,
    "total_price": 46,
    "total_freight": 18.42,
    "payment_count": 2,
    "payment_value": 64.42,
    "max_installments": 1,
    "customer_state": "MG",
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_endpoint():
    response = client.get("/model")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "delivery_late_prediction"
    assert data["version"] == "1"


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json=VALID_ORDER,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "On Time"
    assert data["probability"] == 0.3665408921031306
    assert data["model"] == "delivery_late_prediction"
    assert data["version"] == "1"


def test_batch_predict_endpoint():
    response = client.post(
        "/predict/batch",
        json={
            "orders": [
                VALID_ORDER,
                VALID_ORDER,
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["prediction"] == "On Time"
    assert data["predictions"][1]["prediction"] == "On Time"


def test_predict_rejects_invalid_input():
    invalid_order = VALID_ORDER.copy()
    invalid_order["item_count"] = -1

    response = client.post(
        "/predict",
        json=invalid_order,
    )

    assert response.status_code == 422