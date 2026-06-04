from rag_app.__main__ import main


def test_cli_help_lists_subcommands(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    output = capsys.readouterr().out

    assert "check" in output
    assert "ingest" in output
    assert "retrieve" in output
    assert "ask" in output
    assert "eval" in output
