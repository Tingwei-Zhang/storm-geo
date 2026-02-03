"""Unit tests for infra parse_secrets_and_run (parse_secrets_json helper)."""
import importlib.util
import sys
from pathlib import Path

# Load parse_secrets_and_run from infra/ without requiring infra to be a package
REPO_ROOT = Path(__file__).resolve().parent.parent
PARSE_SECRETS_PATH = REPO_ROOT / "infra" / "parse_secrets_and_run.py"  # noqa: E501


def _load_parse_secrets_json():
    spec = importlib.util.spec_from_file_location(
        "parse_secrets_and_run",
        PARSE_SECRETS_PATH,
        submodule_search_locations=[str(REPO_ROOT)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["parse_secrets_and_run"] = mod
    spec.loader.exec_module(mod)
    return mod.parse_secrets_json


parse_secrets_json = _load_parse_secrets_json()


class TestParseSecretsJson:
    def test_none_returns_empty(self):
        assert parse_secrets_json(None) == {}

    def test_empty_string_returns_empty(self):
        assert parse_secrets_json("") == {}

    def test_valid_json_string_values(self):
        raw = '{"FOO":"bar","NUM":123}'
        out = parse_secrets_json(raw)
        assert out == {"FOO": "bar"}
        assert "NUM" not in out

    def test_valid_json_all_strings(self):
        raw = '{"OPENAI_API_KEY":"sk-x","BUCKET":"b1"}'
        out = parse_secrets_json(raw)
        assert out == {"OPENAI_API_KEY": "sk-x", "BUCKET": "b1"}

    def test_invalid_json_returns_empty(self):
        out = parse_secrets_json("not json")
        assert out == {}

    def test_non_dict_json_returns_empty(self):
        out = parse_secrets_json("[1,2,3]")
        assert out == {}
