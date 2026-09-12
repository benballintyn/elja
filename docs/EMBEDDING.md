# Embedding elja in an application

elja has two construction paths. `build_agent(settings)` is the convenience
factory behind `elja chat`: it reads settings and assembles a complete local
agent — workspace tools, filesystem skills, sub-agents, MCP clients,
compaction, permission gate. `build_application_agent(...)` is the embedded
path: it assembles **nothing you did not ask for** and hands back a native
`pydantic_ai.Agent` bound to *your* dependency type.

This document covers what elja guarantees for a host that meters paid
requests, and — just as important — what it does not.

Versions these statements were verified against: pydantic-ai-slim **2.36.0**,
pydantic-ai-harness **0.27.0**, Python 3.12. The pins are ranges
(`pydantic-ai-slim >=2.36,<3`, `pydantic-ai-harness >=0.27,<0.28`), so the
behaviors below are pinned by `tests/test_metering.py` rather than assumed.

## Metering and admission control

The mechanism is a model wrapper. Subclass `pydantic_ai.models.wrapper.WrapperModel`,
run your admission check, then delegate.

**The rule, not a list: `WrapperModel` forwards every provider-reaching method it
does not override.** There are four, and each one dispatches to the provider:

| method | when it fires |
| --- | --- |
| `request` | the ordinary non-streamed model call |
| `request_stream` | the streamed call — `request` does **not** cover it |
| `count_tokens` | before *every* request when `UsageLimits.count_tokens_before_request` is set. A real network call on the providers that implement it, and pydantic-ai routes it through `check_allow_model_requests()` like any other model request |
| `compact_messages` | provider-side compaction, reachable if you attach a capability that uses it |

A guard on only `request` spends ungated on the other three.
`tests/test_metering.py` pins `request`, `request_stream` and `count_tokens`,
each with a positive control so the denial assertions cannot pass vacuously.

```python
class GuardedModel(WrapperModel):
    async def request(self, messages, model_settings, model_request_parameters):
        await self.budget.reserve(self.attribution)      # raises on denial
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, params, run_context=None):
        await self.budget.reserve(self.attribution)
        async with super().request_stream(messages, model_settings, params, run_context) as s:
            yield s

    async def count_tokens(self, messages, model_settings, model_request_parameters):
        await self.budget.reserve(self.attribution)
        return await super().count_tokens(messages, model_settings, model_request_parameters)
```

What elja guarantees:

- **Your model object is never rebuilt.** Both construction paths pass a `Model`
  instance straight to `Agent`, so the wrapper, its provider client, endpoint,
  timeout and retry configuration survive. (On the convenience path there is no
  caller-supplied model to preserve — `build_agent` builds one from settings.)
- **The summarizer goes through your guard too.** `build_compaction(settings)`
  leaves the summarization tier's `model=None`, which makes
  `SummarizingCompaction` use the running agent's own model *object*, so
  compaction's private request is admitted by the same wrapper as the main
  request. Verified, not inferred.
- **A separately supplied summarizer stays separate.**
  `build_compaction(settings, summarizer_model=my_other_guard)` hands that
  instance to the strategy untouched. It takes a `Model`, not a model *name*, on
  purpose: a name makes the summarizer build its own provider client from
  environment credentials, with nothing you wrapped in the path.
- **A denial is terminal.** Raising from a guarded hook propagates out of
  `agent.run`; it is not converted into a tool-retry suggestion, and a pending
  `ModelRetry` from a tool does not outlive it.
- **The summary attempt is accounted once.** `SummarizingCompaction` runs its
  private agent with `usage=ctx.usage`, so the summary's tokens roll up into the
  parent run's totals exactly once. Reconcile against the token totals, not
  `usage.requests`, which counts committed request *steps* rather than dispatches.

## Telling a compaction request from a main one

A `Model.request` call receives no run context at all, and the summarizer's
private agent is constructed as `Agent[None, str]` and run without `deps`, so
`RunContext.deps` is unavailable there even in principle. `request_stream` *is*
passed a `run_context`, but relying on it would give attribution on the streamed
path only.

Two channels work today, with no upstream patch:

1. **Tag the summarizer's own settings.** `SummarizingCompaction` forwards
   agent-level `model_settings` into its private agent, and those arrive at
   `Model.request`/`request_stream` as the `model_settings` argument. So one
   guard instance can distinguish the phases:
   `build_compaction(settings, summarizer_model_settings={"extra_headers": {"x-phase": "compaction"}})`.
   `extra_headers` is a base `ModelSettings` field, so this is typed and
   provider-neutral.
2. **Use a distinct instance.** Build one guarded model per run, closing over the
   run identity, and pass a separate one as `summarizer_model`.

**Proposed narrow upstream patch**, scoped to *deps-carried* attribution
specifically: `SummarizingCompaction._summarize` could pass the parent's `deps`
to the private agent it builds, so a host that carries identity in `deps` rather
than on the model instance gets it there too. Not needed for either channel
above; recorded rather than pretended to be covered.

## What elja does not provide

- **No spend ledger.** `UsageLimits.cost_limit` is a per-run, in-process ceiling
  evaluated against pydantic-ai's own pricing for models it can price. It is
  **not** a durable, multi-worker dollar cap, it does not survive a restart, and
  two workers each hold their own counter. A real budget needs reservations in a
  store the host owns; elja deliberately does not implement one. On the embedded
  path you set it per run on the native agent —
  `agent.run(..., usage_limits=UsageLimits(cost_limit=Decimal("2.50")))` — and on
  an unpriced model (the local default included) the run's cost is `None`, so the
  limit does nothing and pydantic-ai warns.
- **Sub-agent delegates are outside your guard.** A configured delegate builds
  its **own** model and its own compaction from `elja.toml`, never from the host.
  A guard passed to `build_application_agent` does not cover a delegation, and a
  per-delegation request budget is not a shared dollar budget. Keep sub-agents off
  the embedded path until elja propagates them explicitly.
- **No enforcement via event callbacks.** Status sinks in elja are display
  telemetry, and a raising sink is suppressed on purpose — one shared helper,
  `elja.deps.notify` — so a broken display cannot abort a run or lose the turn's
  history. Never put admission or accounting in a UI subscriber; put it in the
  model wrapper.
- **A wrapper is skippable by the host itself.** `agent.override(model=...)` and a
  per-run `model=` replace it, and nesting the guard *inside* a `FallbackModel`
  lets a denial trigger an attempt against the next model. Keep the guard
  outermost.
- **`count_tokens_before_request` is not self-evidently enforcement.** It only
  does something on providers that implement a count-tokens call, and it is
  itself a guarded dispatch (see the table above).
- **`PermissionGate` is not usable on the embedded path.** It reads
  `ctx.deps.confirm` and is therefore typed to `EljaDeps`. A host with its own
  approval UX should gate inside its own toolset.

## Persistence

History is caller-owned on the embedded path. Pass `message_history=` and
serialize with pydantic-ai's own message adapter. elja's named JSON `Session`
remains for CLI users and is not involved here; the embedded path writes no
files at all.
