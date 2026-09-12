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
pydantic-ai-harness **0.27.0**, Python 3.12. Every claim below was measured at
those versions against the installed packages.

The pins are ranges (`pydantic-ai-slim >=2.36,<3`, `pydantic-ai-harness
>=0.27,<0.28`), so *measured* and *regression-guarded* are not the same thing and
this document distinguishes them. `tests/test_metering.py` guards the subset named
under "Metering and admission control" below — `request`, `request_stream` and
`count_tokens`, each with a positive control. Everything else here is marked
**measured, not guarded**: true at these versions, and a minor release could
change it with the suite still green. `merge_model_settings`' own source carries
`# Note: we may want merge recursively if/when we add non-primitive values`, which
is exactly the kind of drift to expect.

## Metering and admission control

The mechanism is a model wrapper. Subclass `pydantic_ai.models.wrapper.WrapperModel`,
run your admission check, then delegate.

**The rule, not a list: any of these that your subclass does not override goes
straight to the provider.** There are five, and the last one is the odd case:

| method | when it fires |
| --- | --- |
| `request` | the ordinary non-streamed model call |
| `request_stream` | the streamed call — `request` does **not** cover it |
| `count_tokens` | before *every* request when `UsageLimits.count_tokens_before_request` is set. A real network call on the providers that implement it, routed through `check_allow_model_requests()` like any other model request. On one that does **not** implement it — `OpenAIChatModel`, elja's default — the flag raises `NotImplementedError` on every request rather than no-opping |
| `compact_messages` | provider-side compaction, reachable if you attach a capability that uses it |
| `cancel_suspended_response` | cancelling a suspended background response. On `OpenAIResponsesModel` this issues a real `responses.cancel` HTTP call, and it is reached from the ordinary agent path on the run's outermost model |

`cancel_suspended_response` needs **recording, not admission**: a cancel bills no
tokens, so reserving budget on it would be perverse. Two things make it worth
knowing anyway. It is the one dispatch that does **not** call
`check_allow_model_requests()`, so a host whose test suite proves "no egress
without admission" by setting `ALLOW_MODEL_REQUESTS=False` is not covered on this
path. And E2 asks that provider cancellation stay available to an
application-owned recorder, which means seeing it.

A guard on only `request` spends ungated on the other four.
`tests/test_metering.py` pins `request`, `request_stream` and `count_tokens`,
each with a positive control so the denial assertions cannot pass vacuously.

**One admission is one *logical* request, not one network attempt.** The provider
SDKs retry underneath: `openai` and `anthropic` both default to
`max_retries=2`, and elja's `build_model` does not override it, so a single
admission can front up to three HTTP attempts — and a retry after a timeout on a
request the server already began generating is billable. E2 is explicit that a
callback firing once while the provider makes several hidden requests does not
satisfy strict admission. A host that needs per-attempt admission must pass its
own client with `max_retries=0` (`AsyncOpenAI(max_retries=0)` /
`AsyncAnthropic(max_retries=0)`) and own the retry loop above the guard.

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

    # compact_messages is NOT overridden here: nothing in pydantic-ai-harness calls
    # Model.compact_messages (the name appears only as an OTel span), so no elja
    # host reaches it today. That is a fact about the installed version, not a
    # guarantee — the rule above still applies, so add it if you attach a
    # capability that uses provider-side compaction.
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

The compaction helper lives in its own module: `from elja.compaction import
build_compaction`.

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

   **It replaces, it does not merge.** `merge_model_settings` is a shallow
   `base | overrides`, so an `extra_headers` you pass here wipes whatever your
   model already carries — for the summarizer request only. Measured: a model
   built with `extra_headers={"authorization": ..., "x-tenant": ...}` sends the
   agent request with both and the compaction request with only `x-phase`. If
   your model carries gateway or tenant headers, restate them inside
   `summarizer_model_settings`. The same shallowness means a host carrying run
   identity in agent-level `extra_headers` loses it on the compaction request —
   identity belongs on the guard instance, which is what channel 2 is for.
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
- **No enforcement via event callbacks.** Never put admission or accounting in a
  UI subscriber; put it in the model wrapper. elja's *status* sink is display
  telemetry and a raising one is suppressed — one shared helper,
  `elja.deps.notify` — so a broken status display cannot abort a run or lose the
  turn's history. The CLI's **text-delta** sink is deliberately *not* suppressed:
  silently swallowing the model's own output would be worse than failing, so a
  raising delta sink does abort the turn (a closed stdout, for instance) and that
  turn's history is not saved. Two sinks, two different answers, on purpose.
- **`notify` lets a `BaseException` through.** `asyncio.CancelledError`,
  `KeyboardInterrupt` and `SystemExit` pass it untouched, because cancellation is
  the host's and a telemetry helper must not eat it.
- **A wrapper is skippable by the host itself.** `agent.override(model=...)` and a
  per-run `model=` replace it. Keep the guard outermost.
- **Nesting the guard inside a `FallbackModel` is unsafe regardless of your
  exception type.** Whether a *denial* falls through depends on the chain:
  `FallbackModel` defaults to `fallback_on=(ModelAPIError,)`, so a plain
  `Exception` denial aborts the run (good) while a denial raised as a
  `ModelHTTPError` silently dispatches to the next model (bad) — both measured.
  But the deeper problem holds either way: every *other* model in the chain sits
  outside your guard, so any genuine `ModelAPIError` from the first model
  produces an un-admitted dispatch to the second. Put the `FallbackModel` inside
  the guard, not the other way round.
- **`count_tokens_before_request` breaks the run on a provider that does not
  implement it.** It does not degrade quietly. `Model.count_tokens` raises
  `NotImplementedError`, and `OpenAIChatModel` — which is what `build_model`
  returns, and what every LM Studio / OpenAI-compatible endpoint goes through —
  does not override it, so **every request raises** with the flag on. Measured:
  `NotImplementedError: Token counting ahead of the request is not supported by
  OpenAIChatModel`. Turn it on only against a model you have confirmed implements
  `count_tokens`: of the three elja builds, `AnthropicModel` and `GoogleModel` do,
  `OpenAIChatModel` does not (`OpenAIResponsesModel` does, but elja's `openai`
  dialect builds the chat model). Where it *is* supported it is a real network
  call and a guarded dispatch, so it is also not free.
- **`PermissionGate` is not usable on the embedded path.** It reads
  `ctx.deps.confirm` and is therefore typed to `EljaDeps`. A host with its own
  approval UX should gate inside its own toolset.

## Persistence

History is caller-owned on the embedded path. Pass `message_history=` and
serialize with pydantic-ai's own message adapter. elja's named JSON `Session`
remains for CLI users and is not involved here; the embedded path writes no
files at all.
