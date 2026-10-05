# llmkit

> **Living document** — current state only. No history, decision-rationale, or which-subtask-did-what notes; keep it current.

One Python client for the LLMs reachable through Azure Foundry, Vertex and OpenRouter:
GPT and Claude on Foundry, Gemini and Claude on Vertex, and other models over chat
completions on any of the three.

## Install

```bash
uv add "llmkit @ git+https://github.com/frasalvi/llmkit@v0.2.0"
```

## Credentials

Copy `.env.example` to `.env` at the project root and fill in the providers you use.
Values in the nearest `.env` win over shell variables of the same name.

| Provider | Variables |
|---|---|
| Foundry | `AZURE_API_KEY`, `AZURE_ENDPOINT` (e.g. `https://<resource>.services.ai.azure.com`) |
| Vertex | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (default `global`), and `gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS` |
| OpenRouter | `OPENROUTER_API_KEY` |

## Quickstart

```python
from llmkit import LLM, JsonlLog, Message, Tool

llm = LLM("claude-opus-5", on_call=JsonlLog("runs/calls.jsonl"))
r = llm.complete("Summarize this paragraph: ...", system="Be brief.", effort="low")
print(r.text, r.usage.cost)
```

Structured output, tools, streaming and async:

```python
from pydantic import BaseModel


class Verdict(BaseModel):
    answer: str
    confidence: float


r = llm.complete("Is the claim supported?", schema=Verdict)
r.parsed  # Verdict

weather = Tool(
    "get_weather",
    "Weather for a city",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
first = llm.complete("Weather in Paris?", tools=[weather])
call = first.tool_calls[0]
history = [
    Message.user("Weather in Paris?"),
    first.message,
    Message.tool_result(call, "Sunny, 24 C"),
]
second = llm.complete(history, tools=[weather])

for event in llm.stream("Tell me a story"):
    ...  # TextDelta, ThinkingDelta, ToolCallDelta, then Done(result)

result = await llm.acomplete("...")
```

## Routing

| Model names | Provider · wire format |
|---|---|
| `gpt-*` | Foundry · Responses API |
| `gemini-*` | Vertex · generateContent |
| `claude-*` | Foundry · Messages API (default) or Vertex with `provider="vertex"` |
| anything else | registered models (`deepseek-v4-pro`, `glm-5.3`, `kimi-k3`), or any name with `provider="foundry" / "vertex" / "openrouter"` over chat completions |

Names must be exact; there are no aliases.

## Effort

`effort` is `off`, `low`, `medium`, `high`, `max`, or `None` (default: send nothing,
provider default applies). A rung a model does not serve raises `UnsupportedEffort`;
`llm.efforts` lists the served rungs.

## Cost

Costs are LiteLLM list prices (cached at `~/.cache/llmkit/prices.json`, refreshed
monthly), a lower bound on an invoice: tiered long-context rates are not applied. A
model with no price emits `UnpricedModelWarning` once and reports `cost=None`. To make
that fatal:
`warnings.simplefilter("error", llmkit.UnpricedModelWarning)`.

## Errors and retries

llmkit retries 408, 429, 5xx, dropped connections and timeouts (`max_retries=3`,
honouring `Retry-After`); `timeout` is per attempt. All errors derive from
`LLMKitError`; request errors carry `provider`, `status`, `request_id` and `body`.
Refusals and truncation are not errors: check `result.stop_reason`.

## Call records

`on_call` receives a `CallRecord` after every call that reaches the provider, success or
failure, and after every replay from a cache (`record.cache == "hit"`). Errors raised
before sending (`UnknownModel`, `UnsupportedEffort`, `BudgetExceeded`, ...) and streams
the caller stops reading early write no record.
`JsonlLog(path, content=False)` drops prompts and responses. `llmkit.records.read(path)`
loads a log. Use one log file per process.

## Batch queries

Give an `LLM` a cache and run a function over many items with `query`. Every call is
stored under a hash of its full request, so a rerun replays finished calls for free and
the results file is rebuilt from the cache and your current code.

```python
from llmkit import LLM, Cache, Message, query

cache = Cache(".llmkit/calls.sqlite")
llm = LLM("gpt-5.6-luna", cache=cache)


def interview(item):
    first = llm.complete(item["question"], sample=item["rep"])
    follow_up = [Message.user(item["question"]), first.message, Message.user("Why?")]
    return {"answer": first.text, "why": llm.complete(follow_up, sample=item["rep"]).text}


report = query(
    items,
    interview,
    key=lambda it: f"{it['id']}/{it['rep']}",
    out="runs/interviews.jsonl",
    max_cost=5.00,
)
print(report)  # 412 ok · 3 failed · $1.84 new · $9.10 replayed
```

- **What the key covers.** The cache key covers the model, the prompt, every setting
  and `sample`; repeats are `sample=0, 1, 2…`. Keep volatile text such as today's date
  out of prompts, or nothing will replay.
- **What is stored.** Every reply is stored, refusals and filtered replies included.
  `Cache(retry_on=…)` lists the stop reasons asked again on later runs (default
  `{"max_tokens"}`), up to `max_attempts` outcomes per request (default 3). Errors are
  never stored.
- **The results file.** `out` is rewritten in input order when the run finishes, and a
  run that replays everything writes the same file. `runs/interviews.meta.json` beside
  it holds the git commit, llmkit version, counts, new and replayed spend, and served-model
  drift. An interrupted run leaves the previous file untouched.
- **Spending and trials.** `max_cost` caps this run's new spend; replays are always
  served. `limit=N` tries the first N items and writes nothing.
- **Async and streaming.** `aquery` is the same for code already inside an event loop.
  `stream` does not use the cache.
- **Privacy.** A cache is local: llmkit puts a `.gitignore` in the folder it creates
  for it.

## Development

```bash
uv sync --group dev
uv run pre-commit install
uv run pytest            # unit tests, no network
uv run pytest -m live    # every route; needs .env, costs money
uv run python scripts/capture_fixtures.py
```
