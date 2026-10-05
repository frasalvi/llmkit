import hashlib
import json
import threading

import pytest

from llmkit.records import JsonlLog, build_record, message_to_dict, read
from llmkit.registry import resolve
from llmkit.transports.base import Call
from llmkit.types import Image, Message, Result, Text, ToolCall, Usage


def make_call():
    return Call(
        route=resolve("glm-5.3"),
        messages=[Message.user("hi")],
        system="sys",
        effort="low",
    )


def make_result():
    msg = Message("assistant", "hello")
    return Result(
        "hello",
        "th",
        [ToolCall("c", "t", {"a": 1})],
        "end",
        Usage(model="glm-5.3", input_tokens=3, cost=0.1),
        msg,
    )


def test_build_record_success_and_failure():
    record = build_record(
        make_call(),
        result=make_result(),
        error=None,
        attempts=2,
        latency_ms=50,
        tags={"run": "r1"},
    )
    assert (record.model, record.provider, record.deployment) == (
        "glm-5.3",
        "foundry",
        "FW-GLM-5.3",
    )
    assert record.text == "hello" and record.usage["input_tokens"] == 3
    assert record.attempts == 2 and record.tags == {"run": "r1"} and record.error is None
    failed = build_record(
        make_call(),
        result=None,
        error=ValueError("boom"),
        attempts=1,
        latency_ms=5,
        tags={},
    )
    assert (failed.error_type, failed.error, failed.usage) == ("ValueError", "boom", None)


def test_images_are_hashed():
    data = b"\x89PNG"
    out = message_to_dict(Message.user([Text("see"), Image(data, "image/png")]))
    assert out["content"][1] == {
        "type": "image",
        "media_type": "image/png",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_jsonl_log_and_content_switch(tmp_path):
    path = tmp_path / "runs" / "calls.jsonl"
    record = build_record(
        make_call(), result=make_result(), error=None, attempts=1, latency_ms=1, tags={}
    )
    JsonlLog(path)(record)
    JsonlLog(path, content=False)(record)
    rows = read(path)
    assert rows[0]["text"] == "hello" and rows[0]["messages"][0]["content"] == "hi"
    assert rows[1]["text"] is None and rows[1]["messages"] is None
    assert rows[1]["usage"]["input_tokens"] == 3


def test_content_false_keeps_text_out_of_the_file(tmp_path):
    path = tmp_path / "calls.jsonl"
    record = build_record(
        make_call(), result=make_result(), error=None, attempts=1, latency_ms=1, tags={}
    )
    JsonlLog(path, content=False)(record)
    raw = path.read_text(encoding="utf-8")
    for secret in ("hello", '"hi"', '"sys"', '"th"'):
        assert secret not in raw
    row = json.loads(raw)
    for name in ("system", "messages", "text", "thinking", "tool_calls"):
        assert row[name] is None


def test_concurrent_writes_stay_line_atomic(tmp_path):
    path = tmp_path / "calls.jsonl"
    log = JsonlLog(path)
    record = build_record(
        make_call(), result=make_result(), error=None, attempts=1, latency_ms=1, tags={}
    )
    threads = [
        threading.Thread(target=lambda: [log(record) for _ in range(50)])
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(read(path)) == 400


def test_read_skips_torn_last_line_only(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps({"a": 1}) + "\n" + '{"a": 2')
    assert read(path) == [{"a": 1}]
    path.write_text('{"a": \n' + json.dumps({"a": 1}) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        read(path)
