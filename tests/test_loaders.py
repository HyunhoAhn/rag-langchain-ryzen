from rag_app.loaders import load_documents


def test_load_documents_loads_txt_files(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Local RAG notes", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].page_content == "Local RAG notes"
    assert documents[0].metadata["source"] == str(source)


def test_load_documents_loads_markdown_files_recursively(tmp_path):
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    source = nested_dir / "guide.md"
    source.write_text("# Guide\n\nUse the local documents.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("unsupported", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].page_content == "# Guide\n\nUse the local documents."
    assert documents[0].metadata["source"] == str(source)


def test_load_documents_returns_empty_list_without_supported_files(tmp_path):
    (tmp_path / "ignored.csv").write_text("unsupported", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert documents == []
