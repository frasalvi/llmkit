from llmkit.types import (
    ROLE_TOOL,
    Image,
    Message,
    Text,
    ToolCall,
    Usage,
    parts,
    text_of,
)


def test_parts_and_text_of():
    assert parts("hi") == [Text("hi")]
    assert parts("") == []
    img = Image(b"\x89PNG", "image/png")
    mixed = [Text("a"), img, Text("b")]
    assert parts(mixed) == mixed
    assert text_of(mixed) == "ab"
    assert text_of("plain") == "plain"


def test_tool_result_carries_call_identity():
    call = ToolCall("c1", "get_weather", {"city": "Paris"})
    msg = Message.tool_result(call, "sunny", is_error=False)
    assert msg.role == ROLE_TOOL
    assert (msg.tool_call_id, msg.tool_name, msg.content) == (
        "c1",
        "get_weather",
        "sunny",
    )


def test_usage_add_pools_and_poisons_cost():
    a = Usage(
        model="m",
        input_tokens=10,
        output_tokens=5,
        cached_input_tokens=2,
        cache_write_tokens=1,
        latency_ms=100,
        cost=0.5,
    )
    b = Usage(
        model="m", effort="low", input_tokens=1, output_tokens=1, latency_ms=10, cost=0.25
    )
    total = a + b
    assert (total.input_tokens, total.output_tokens, total.cached_input_tokens) == (
        11,
        6,
        2,
    )
    assert total.cache_write_tokens == 1 and total.latency_ms == 110
    assert total.cost == 0.75 and total.effort == "low"
    assert (a + Usage(cost=None)).cost is None
