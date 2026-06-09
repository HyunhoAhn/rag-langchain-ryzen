import requests

from rag_app.config import AppConfig
from rag_app.lemonade_client import check_lemonade_models


def _config() -> AppConfig:
    return AppConfig(
        lemonade_base_url="http://localhost:13305/v1",
        lemonade_chat_model="Qwen3-8B-GGUF",
        chroma_dir="./chroma_db",
        collection_name="ryzen_ai_max_rag",
        embedding_model="BAAI/bge-m3",
    )


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_check_lemonade_models_finds_configured_model(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://localhost:13305/v1/models"
        assert timeout == 5.0
        return FakeResponse({"data": [{"id": "Qwen3-8B-GGUF"}]})

    monkeypatch.setattr("rag_app.lemonade_client.requests.get", fake_get)

    result = check_lemonade_models(_config())

    assert result.reachable is True
    assert result.model_found is True
    assert result.model_ids == ["Qwen3-8B-GGUF"]


def test_check_lemonade_models_handles_unreachable_server(monkeypatch):
    def fake_get(url, timeout):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("rag_app.lemonade_client.requests.get", fake_get)

    result = check_lemonade_models(_config())

    assert result.reachable is False
    assert result.model_found is None
    assert "connection refused" in result.error
