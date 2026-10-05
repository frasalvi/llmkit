import pytest

from llmkit.credentials import credentials_for, find_dotenv, load_env
from llmkit.errors import MissingCredential


def test_dotenv_wins_over_environment(tmp_path):
    (tmp_path / ".env").write_text("AZURE_ENDPOINT=https://from-file\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    env = load_env(
        start=sub,
        environ={"AZURE_ENDPOINT": "https://from-shell", "AZURE_API_KEY": "shell-key"},
    )
    assert env["AZURE_ENDPOINT"] == "https://from-file"
    assert env["AZURE_API_KEY"] == "shell-key"


def test_no_dotenv_falls_back_to_environment(tmp_path):
    assert find_dotenv(tmp_path) != tmp_path / ".env"
    env = load_env(env_file=None, start=tmp_path, environ={"LLMKIT_TEST_ONLY": "1"})
    assert env["LLMKIT_TEST_ONLY"] == "1"


def test_explicit_env_file(tmp_path):
    path = tmp_path / "creds.env"
    path.write_text("OPENROUTER_API_KEY=abc\n")
    assert load_env(env_file=path, environ={})["OPENROUTER_API_KEY"] == "abc"
    with pytest.raises(FileNotFoundError):
        load_env(env_file=tmp_path / "missing.env", environ={})


def test_credentials_name_missing_variables():
    with pytest.raises(MissingCredential) as info:
        credentials_for("foundry", {"AZURE_API_KEY": "k", "AZURE_ENDPOINT": ""})
    assert info.value.missing == ["AZURE_ENDPOINT"]


def test_vertex_location_defaults_to_global():
    creds = credentials_for("vertex", {"GOOGLE_CLOUD_PROJECT": "p"})
    assert creds == {"GOOGLE_CLOUD_PROJECT": "p", "GOOGLE_CLOUD_LOCATION": "global"}
    creds = credentials_for(
        "vertex",
        {
            "GOOGLE_CLOUD_PROJECT": "p",
            "GOOGLE_CLOUD_LOCATION": "us-east5",
            "GOOGLE_APPLICATION_CREDENTIALS": "/k.json",
        },
    )
    assert creds["GOOGLE_CLOUD_LOCATION"] == "us-east5"
    assert creds["GOOGLE_APPLICATION_CREDENTIALS"] == "/k.json"
