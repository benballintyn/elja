# elja

*Elja* (Icelandic): relentless drive — the quality of working at something without letting up.

A fully-customizable LLM agent harness built on [Pydantic AI](https://ai.pydantic.dev), designed
for local-first models (LM Studio / OpenAI-compatible endpoints) with first-class support for
tools, skills, sub-agents, MCP servers, and custom context management.

**Status: under active development.**

## Installation

```bash
pip install elja
```

## Usage

Start LM Studio serving a model (default expectation: `qwen/qwen3.8-27b` at
`http://localhost:1234/v1`), then:

```bash
elja chat                      # interactive REPL (session persisted + resumed)
elja chat --once "list the files here and summarize them"
elja chat --config path/to/elja.toml --session mychat
```

Or from Python:

```python
from elja import EljaDeps, build_agent, build_usage_limits, load_settings

settings = load_settings()
agent = build_agent(settings)
result = agent.run_sync(
    "What's in this directory?",
    deps=EljaDeps.from_settings(settings),
    usage_limits=build_usage_limits(settings),
)
print(result.output)
```

Configuration lives in `elja.toml` (all keys optional; `ELJA_*` env vars
override, e.g. `ELJA_MODEL__BASE_URL`):

```toml
[model]
name = "qwen/qwen3.8-27b"
base_url = "http://localhost:1234/v1"
temperature = 0.2

[limits]
request_limit = 25

[workspace]
root = "."

[tools]
run_shell = true

[agent]
instructions = "Optional: replace the default system instructions."
```

## Development

```bash
poetry install --with dev
poetry run pre-commit install
```

Run checks:

```bash
poetry run ruff check src tests
poetry run mypy
poetry run pytest -m "not integration"
```

Integration tests (`-m integration`) require a running LM Studio server at
`http://localhost:1234/v1`.

## License

MIT
