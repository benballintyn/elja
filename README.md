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
elja chat --once "what is in this screenshot?" --image shot.png
elja chat --config path/to/elja.toml --session mychat
# In the REPL, attach an image to a message with: /img <path> <prompt>
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

### Embedding elja in an application

`build_agent` is the convenience factory: it reads settings and assembles a
complete local agent — workspace tools, filesystem skills, sub-agents, MCP
clients, compaction, permission gate. A server that owns its own tools,
dependencies and persistence wants the other door, which assembles **nothing
you did not ask for**:

```python
import asyncio
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from elja import build_application_agent, build_model, load_settings


@dataclass
class AppDeps:            # your own type — no EljaDeps, no workspace
    tenant: str
    records: list[str]


toolset: FunctionToolset[AppDeps] = FunctionToolset()


@toolset.tool
async def record_fact(ctx: RunContext[AppDeps], fact: str) -> str:
    """A domain tool: it gets your deps object, untouched."""
    ctx.deps.records.append(fact)
    return f"recorded for {ctx.deps.tenant}"


# Your Model, passed through untouched. build_model(load_settings()) is one way
# to get one from elja.toml — that read is YOURS, not something this path does.
agent = build_application_agent(
    build_model(load_settings()),
    deps_type=AppDeps,
    instructions="You keep a household's records.",
    toolsets=[toolset],
)


async def main() -> None:
    result = await agent.run(
        "remember the vet is tuesday",
        deps=AppDeps(tenant="acme", records=[]),
    )
    print(result.output)


asyncio.run(main())
```

No file/shell/web-search tools, no skills directory scan, no MCP subprocess, no
`.elja` writes, no workspace. The return value is a native `pydantic_ai.Agent`,
so `run_stream_events`, `message_history`, `output_type`, per-run
`model_settings`, `usage` and `usage_limits` all behave as they do upstream, and
the standard agent options (`name`, `retries`, `end_strategy`,
`max_concurrency`) are all reachable. A wall-clock cap per tool goes on your
toolset (`FunctionToolset(timeout=...)`), not on the agent — see the module
docstring for why.

elja's own capabilities stay available by opting in, e.g.
`capabilities=build_compaction(load_settings(), cleared_placeholder=...)` — pass
your own placeholder, because the default tells the model to *re-run* a cleared
tool and names `.elja/spill/`, and neither is right for a host with
side-effecting tools and no workspace. `elja/application.py` covers how `None`
and empty differ between the two paths; `elja/compaction.py` covers composing
the policy and why reporting goes last; and
`docs/EMBEDDING.md` covers metering, admission control, events, checkpoints,
cancellation, persistence, and the limits of what elja enforces, and
`examples/server_agent.py` is all of it as a running server-shaped host, driven
by the test suite so it cannot rot. Both live in the repository rather than the
published package, so they are linked from
[github.com/benballintyn/elja](https://github.com/benballintyn/elja/blob/main/docs/EMBEDDING.md)
for anyone reading this on PyPI.

Configuration lives in `elja.toml` (all keys optional; `ELJA_*` env vars
override, e.g. `ELJA_MODEL__BASE_URL`):

```toml
[model]
provider = "openai"      # "openai" (any OpenAI-compatible endpoint — the default,
                         # aimed at LM Studio), "anthropic", or "google".
                         # Native providers need: pip install 'elja[anthropic]' / 'elja[google]'
name = "qwen/qwen3.8-27b"
base_url = "http://localhost:1234/v1"   # unset with provider="openai" = local LM Studio.
                         # provider="openai" NEVER means api.openai.com on its own —
                         # a hosted app must name the endpoint and model it intends.
temperature = 0.2        # omit the parameter entirely with temperature = ""
max_tokens = 4096        # (same: "" means "send no max_tokens"). Works in TOML
                         # and as ELJA_MODEL__TEMPERATURE="" in the environment.
# api_key: set for cloud endpoints; native providers also honor
# ANTHROPIC_API_KEY / GOOGLE_API_KEY from the environment.

# Any other native pydantic-ai ModelSettings key. Keys AND values are checked
# against the selected provider's own dialect, at every depth, so an unsupported
# key or a wrongly typed value is a config error naming its path rather than a
# 400 at request time (or, worse, a typo that silently disables a setting).
[model.settings]
top_p = 0.9
# Reasoning controls are the provider's own — no elja-side enum:
#   openai_reasoning_effort / anthropic_thinking / google_thinking_config,
#   or the portable `thinking` (true/false or "minimal".."xhigh").
# NB: pair a reasoning setting with temperature = "" rather than a value.
# anthropic DROPS temperature/top_p when reasoning is on, with a warning; openai
# and google send them and the API rejects the request with a 400.

[limits]                 # every field maps to pydantic-ai's UsageLimits
request_limit = 25
total_tokens_limit = 120000
cost_limit = 2.50        # USD, and only for models pydantic-ai can price —
                         # on an unpriced model (the local default included) it
                         # is unenforced and warns. The warning is raised every
                         # request; Python's default filter shows it once.
tool_calls_limit = 40
input_tokens_limit = 100000
output_tokens_limit = 20000
per_request_input_tokens_limit = 30000   # enforced on every provider
count_tokens_before_request = false   # an extra count-tokens round trip, and
                         # ONLY supported by provider "anthropic" or "google";
                         # elja refuses it with "openai" rather than letting
                         # every request fail with NotImplementedError.

[workspace]
root = "."

[tools]
run_shell = true
web_search = true  # keyless DuckDuckGo search via ddgs (network egress!)

[agent]
instructions = "Optional: replace the default system instructions."

[permissions]          # per-tool policy: allow | ask | deny (any tool name,
default = "allow"      # incl. MCP tools and delegate_*). ask prompts y/N in the
                       # REPL and FAILS CLOSED when non-interactive.
[permissions.tools]
run_shell = "ask"      # the default: shell commands need a nod

[compaction]           # evidence-based tiered compaction (see elja/compaction.py)
                       # NB: 24000 is tuned for a local 27B; raise it for large-window
                       # cloud providers to avoid early masking + paid summarizer calls
enabled = true
target_tokens = 24000  # load the LM Studio model with at least this + headroom
keep_tool_pairs = 10   # recent tool results kept verbatim by the masking tier
keep_messages = 20     # verbatim tail if the summarization fallback fires

# Skills: markdown files in <workspace>/skills/ (or [skills] dir = "...") with
# YAML frontmatter (id, description) + an instructions body. The model loads
# them on demand, so a large skill library costs ~no context until used.

# Sub-agents: delegate tools with isolated context (results-only return).
[subagents.researcher]
description = "Researches a question and reports key facts."
instructions = "Answer tersely with sources."
tools = ["read_file", "web_search"]  # subset of ENABLED built-ins; omit for all enabled
request_limit = 8                     # per-delegation request budget (optional)

# Attach MCP servers; their tools become available to the agent.
[mcp.servers.mytools]
command = "python3"         # stdio: launched as a subprocess
args = ["my_mcp_server.py"]
env = { API_KEY = "..." }   # optional; NB the subprocess gets a minimal env
                            # (HOME/PATH/USER...) plus these — not your full shell env

[mcp.servers.remote]
transport = "http"          # streamable-HTTP endpoint
url = "http://localhost:9000/mcp"
headers = { Authorization = "Bearer ..." }  # optional auth headers
tool_prefix = "remote"      # optional: tools appear as remote_<name>; NB [permissions.tools]
                            # entries must then use the prefixed name
init_timeout = 30           # optional: seconds for slow (npx/uvx) server startup
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
