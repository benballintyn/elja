# elja

*Elja* (Icelandic): relentless drive — the quality of working at something without letting up.

A fully-customizable LLM agent harness built on [Pydantic AI](https://ai.pydantic.dev), designed
for local-first models (LM Studio / OpenAI-compatible endpoints) with first-class support for
tools, skills, sub-agents, MCP servers, and custom context management.

**Status: under active development.** The first usable version is coming.

## Installation

```bash
pip install elja
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
