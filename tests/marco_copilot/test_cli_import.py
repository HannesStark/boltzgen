import importlib.util


def test_cli_entrypoint_registered_in_pyproject():
    pyproject = open("pyproject.toml", encoding="utf-8").read()
    assert 'marco-copilot = "marco_copilot.cli:app"' in pyproject


def test_cli_module_file_exists():
    spec = importlib.util.find_spec("marco_copilot")
    assert spec is not None
