from rag_app.config import load_config


def test_load_config_uses_defaults(tmp_path, monkeypatch):
    for key in (
        "LEMONADE_BASE_URL",
        "LEMONADE_CHAT_MODEL",
        "CHROMA_DIR",
        "COLLECTION_NAME",
        "EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    config = load_config(tmp_path / ".env")

    assert config.lemonade_base_url == "http://localhost:13305/v1"
    assert config.lemonade_chat_model == "Qwen3-8B-GGUF"
    assert config.chroma_dir == "./chroma_db"
    assert config.collection_name == "ryzen_ai_max_rag"
    assert config.embedding_model == "BAAI/bge-m3"


def test_load_config_prefers_environment_over_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("COLLECTION_NAME=from_file\n", encoding="utf-8")
    monkeypatch.setenv("COLLECTION_NAME", "from_env")

    config = load_config(env_file)

    assert config.collection_name == "from_env"
