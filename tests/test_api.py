"""API tests."""

from api.main import health


def test_api_health_endpoint():
    assert health()["status"] == "ok"
