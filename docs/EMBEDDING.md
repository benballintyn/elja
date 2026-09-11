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
run your admission check, then delegate:

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
```

**Override both.** `request` does not cover the streamed path. A guard on only
one of them leaves the other ungated; `tests/test_metering.py` pins both,
including a positive control so the denial assertions cannot pass vacuously.

What elja guarantees:

- **Your model object is never rebuilt.** Both construction paths pass a `Model`
  instance straight to `Agent`, so the wrapper, its provider client, endpoint,
  timeout and retry configuration survive. elja never infers a model from a
  display name you gave it as an object.
- **The summarizer goes through your guard too.** `build_compaction(settings)`
  leaves the summarization tier's `model=None`, which makes
  `SummarizingCompaction` use the running agent's own model *object*. So
  compaction's private request is admitted by the same wrapper as the main
  request. Verified, not inferred.
- **A separately supplied summarizer stays separate.**
  `build_compaction(settings, summarizer_model=my_other_guard)` hands that
  instance to the strategy untouched — for a different provider, a cheaper
  model, or separate attribution.
- **A denial is terminal.** Raising from `request`/`request_stream` propagates
  out of `agent.run`; it is not converted into a tool-retry suggestion, and a
  pending `ModelRetry` from a tool does not outlive it.
- **The summary attempt is counted once.** `SummarizingCompaction` runs its
  private agent with `usage=ctx.usage`, so the summary's tokens roll up into the
  parent run's totals exactly once. Do not post it again from your own recorder.

## Attribution: use per-instance state, not `RunContext`

A `Model.request` call receives no run context at all, and the summarizer's
private agent is constructed as `Agent[None, str]` and run without `deps`, so
`RunContext.deps` is unavailable there even in principle. `request_stream` *is*
passed a `run_context`, but relying on it would give you attribution on the
streamed path only.

So carry attribution on the wrapper instance — build one guarded model per run,
closing over the run/tenant identity — and use a *distinct* instance for the
summarizer when you need to tell a main request from a compaction request.
That is the only mechanism that works uniformly across both hooks and across
both of elja's paths.

**Proposed narrow upstream patch** (not worked around here, and not pretended
to be covered): `SummarizingCompaction._summarize` could pass the parent's
`deps` — or expose the parent `RunContext` — to the private summarizer agent it
builds. That would let a host attribute compaction through the same
deps-carried channel as everything else. Until then, per-instance state is the
documented mechanism.

## What elja does not provide

- **No spend ledger.** `UsageLimits.cost_limit` (exposed via `[limits]`) is a
  per-run, in-process ceiling evaluated against pydantic-ai's own pricing for
  models it can price. It is **not** a durable, multi-worker dollar cap, it does
  not survive a restart, and two workers each hold their own counter. A real
  budget needs reservations in a store the host owns. elja deliberately does not
  implement one.
- **No enforcement via event callbacks.** Events and status sinks in elja are
  display telemetry; `EljaDeps.on_status` failures are suppressed on purpose so
  a broken sink cannot abort a run. Never put admission or accounting in a UI
  subscriber — put it in the model wrapper, which cannot be skipped.
- **`count_tokens_before_request`** is forwarded to `UsageLimits` but only does
  something on providers that implement a count-tokens call. Setting it is not
  by itself proof that a per-request input ceiling was enforced.
- **No `FallbackModel` interaction guarantees.** If you wrap a guarded model in
  a fallback chain, check what your denial exception does to that chain: a
  denial must not cause an attempt against the next model. Keep the guard
  *outside* the fallback, or make the denial an exception the chain does not
  catch.
- **`PermissionGate` is not usable on the embedded path.** It reads
  `ctx.deps.confirm` and is therefore typed to `EljaDeps`. A host with its own
  approval UX should gate inside its own toolset.

## Persistence

History is caller-owned on the embedded path. Pass `message_history=` and
serialize with pydantic-ai's own message adapter. elja's named JSON `Session`
remains for CLI users and is not involved here; the embedded path writes no
files at all.
