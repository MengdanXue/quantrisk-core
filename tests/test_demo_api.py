from fastapi.testclient import TestClient

from demo.app import app

client = TestClient(app)


def test_index_served():
    assert client.get("/").status_code == 200


def test_backtest_endpoint():
    res = client.post("/api/backtest", json={"strategy": "ma_cross", "days": 300})
    assert res.status_code == 200
    body = res.json()
    assert len(body["equity"]) == len(body["dates"]) == 300
    assert "sharpe_ratio" in body["metrics"]


def test_unknown_strategy_rejected():
    assert client.post("/api/backtest", json={"strategy": "nope"}).status_code == 400


def test_indicator_endpoint():
    res = client.post("/api/indicator", json={"name": "RSI", "days": 200})
    assert res.status_code == 200
    assert res.json()["series"]
