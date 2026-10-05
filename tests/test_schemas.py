import pytest
from pydantic import BaseModel

from llmkit.errors import SchemaError
from llmkit.schemas import (
    all_required,
    close_objects,
    parse_output,
    schema_name,
    to_json_schema,
)


class Verdict(BaseModel):
    answer: str
    confidence: float


class Optional_(BaseModel):
    answer: str
    note: str = ""


def test_close_objects_sets_additional_properties_everywhere():
    schema = {
        "type": "object",
        "properties": {
            "inner": {"type": "object", "properties": {"x": {"type": "integer"}}}
        },
    }
    closed = close_objects(schema)
    assert closed["additionalProperties"] is False
    assert closed["properties"]["inner"]["additionalProperties"] is False
    assert "additionalProperties" not in schema


def test_all_required():
    assert all_required(to_json_schema(Verdict))
    assert not all_required(to_json_schema(Optional_))


def test_schema_name():
    assert schema_name(Verdict) == "Verdict"
    assert schema_name({"title": "my schema!"}) == "my_schema_"
    assert schema_name({}) == "output"


def test_parse_output_validates_models_and_json():
    parsed = parse_output(Verdict, '{"answer": "yes", "confidence": 0.9}')
    assert parsed == Verdict(answer="yes", confidence=0.9)
    assert parse_output({"type": "object"}, '{"a": 1}') == {"a": 1}
    with pytest.raises(SchemaError) as info:
        parse_output(Verdict, '{"answer": "yes"}')
    assert info.value.raw == '{"answer": "yes"}'
    with pytest.raises(SchemaError):
        parse_output({"type": "object"}, "not json")
