"""Pipeline baseline smoke test (TestClient, 모델 로드 포함)."""
from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.main import app, lifespan


def test_pipeline_smoke():
    async def boot():
        async with lifespan(app):
            pass

    asyncio.run(boot())
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    no_key = client.post("/predict", json={"text": "실적 호조"})
    assert no_key.status_code == 401

    bad = client.post(
        "/predict",
        json={"text": ""},
        headers={"X-API-Key": "test-key-001"},
    )
    assert bad.status_code == 422

    ok = client.post(
        "/predict",
        json={"text": "오늘 실적 발표가 기대 이상이라 주가가 크게 올랐다."},
        headers={"X-API-Key": "test-key-001"},
    )
    assert ok.status_code == 200
    data = ok.json()
    assert data["success"] and data["label"] and data["score"] > 0
    print("pipeline smoke OK:", data)


if __name__ == "__main__":
    test_pipeline_smoke()
