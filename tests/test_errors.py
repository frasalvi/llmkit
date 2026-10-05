from llmkit.errors import (
    ContentFiltered,
    FatalRequest,
    LLMKitError,
    MissingCredential,
    RequestError,
    RequestTimeout,
    RetriesExhausted,
    SchemaError,
    TransientError,
    UnknownModel,
    UnsupportedEffort,
    UnsupportedFeature,
)


def test_hierarchy():
    for cls in (
        UnknownModel,
        MissingCredential,
        UnsupportedEffort,
        UnsupportedFeature,
        RequestError,
        SchemaError,
    ):
        assert issubclass(cls, LLMKitError)
    for cls in (
        ContentFiltered,
        FatalRequest,
        RequestTimeout,
        TransientError,
        RetriesExhausted,
    ):
        assert issubclass(cls, RequestError)


def test_missing_credential_names_variables():
    err = MissingCredential("foundry", ["AZURE_API_KEY", "AZURE_ENDPOINT"])
    assert err.provider == "foundry"
    assert err.missing == ["AZURE_API_KEY", "AZURE_ENDPOINT"]
    assert "AZURE_API_KEY" in str(err) and "AZURE_ENDPOINT" in str(err)


def test_request_error_fields():
    err = ContentFiltered(
        "blocked",
        provider="foundry",
        status=400,
        request_id="r1",
        body={"x": 1},
        categories=["hate"],
    )
    assert (err.provider, err.status, err.request_id, err.body) == (
        "foundry",
        400,
        "r1",
        {"x": 1},
    )
    assert err.categories == ["hate"]
    assert err.attempts == 1
    assert err.retry_after is None


def test_retries_exhausted_wraps_last():
    last = TransientError("503", provider="vertex", status=503)
    err = RetriesExhausted(last, 4)
    assert err.last is last and err.attempts == 4
    assert err.status == 503 and err.provider == "vertex"
    assert "4 attempts" in str(err)


def test_schema_error_keeps_raw():
    assert SchemaError("bad", raw="{").raw == "{"
