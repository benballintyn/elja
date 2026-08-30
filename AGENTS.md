# elja

LLM agent harness built on Pydantic AI. *Elja* (Icelandic): relentless drive — the quality of
working at something without letting up.

Fully-customizable agent harness targeting local-first models (LM Studio / OpenAI-compatible
endpoints, primary target: Qwen3.8-27B), with first-class tools, skills, sub-agents, MCP client
support, and custom context management (compaction).

## Architecture decisions

- **Built on Pydantic AI 2.x** (`pydantic-ai-slim[openai,mcp]`), not from scratch and not on vox.
  Decision recorded in the knowledge graph (project scope `agent-harness`); don't re-litigate.
  Pin-watch: 2.x moves fast — check changelogs on dependency bumps.
- Model layer: `OpenAIChatModel` + `OpenAIProvider(base_url=...)` for LM Studio
  (default `http://localhost:1234/v1`), with `OpenAIModelProfile` quirk flags.
- Persistence is DIY (Pydantic AI has no checkpointer) — history is carried explicitly.
- Logging via loguru; config via pydantic-settings.

## Setup

```bash
poetry install --with dev
poetry run pre-commit install
```

## Verification

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy
poetry run pytest -m "not integration"   # unit tests (CI-safe)
poetry run pytest -m integration          # requires LM Studio running locally
```

## Conventions

- Git flow: feature branches (`feat/`, `fix/`, ...), PRs squash-merged into main.
  Conventional Commit PR titles are enforced (pr_title.yml) and drive release-please versioning.
- Releases: merge the standing release-please PR → tag + GitHub Release + PyPI publish via
  OIDC trusted publishing (pypi-publish.yml, environment `pypi`). Never publish manually.
- Tests requiring a live LLM endpoint get the `integration` marker.
