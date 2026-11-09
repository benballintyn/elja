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
pydantic-ai-harness **0.27.0**, Python 3.12 (and 3.13 for the bounded-compaction
section, added in 0.6). Every claim below was measured at those versions against
the installed packages; CI runs the suite on 3.11, 3.12 and 3.13.

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
| `count_tokens` | before every request when `UsageLimits.count_tokens_before_request` is set — with one exception: it is called from `_prepare_request` only, so a *resumed* suspended turn skips it (fails safe, no un-admitted dispatch). A real network call on the providers that implement it, routed through `check_allow_model_requests()` like any other model request. On one that does **not** implement it — `OpenAIChatModel`, elja's default — the flag raises `NotImplementedError` on every request rather than no-opping |
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

    # compact_messages is NOT overridden here, and that is a gap you may need to
    # close. Nothing ELJA attaches calls it — the harness package uses the name only
    # as an OTel span — but pydantic-ai core's own OpenAICompaction does
    # (models/openai.py, `request_context.model.compact_messages(...)`), on the run's
    # outermost model, i.e. your guard. A host that attaches OpenAICompaction and
    # leaves this hook alone gets a real /responses/compact dispatch with no
    # admission. Override it if you attach anything that uses provider-side
    # compaction.
```

What elja guarantees:

- **Your model object is never rebuilt.** Both construction paths pass a `Model`
  instance straight to `Agent`, so the wrapper, its provider client, endpoint,
  timeout and retry configuration survive. (On the convenience path there is no
  caller-supplied model to preserve — `build_agent` builds one from settings.)
- **The summarizer goes through your guard too.** `build_compaction(settings)`
  leaves the summarization tier's `model=None`, which makes
  `SummarizingCompaction` inherit `ModelRequestContext.model`, so compaction's
  private request is admitted by the same wrapper as the main request. Verified,
  not inferred. To enforce its per-run budget, elja wraps that model — yours, or
  the one you pass as `summarizer_model` — in a counting `WrapperModel` for the
  duration of one summary. It sits *outside* your guard and delegates to it: every
  summarizer request that is dispatched still goes through your guard with the
  same messages and settings, and one refused at elja's budget never reaches your
  guard at all. One caveat upstream states and this inherits: that context model
  *starts* as the run's model and differs only where a capability replaced it. The
  embedded path invites `capabilities=[...]`, so if you attach one that swaps the
  model, this guarantee is the thing it swaps away.
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

   **Two separate facts here, and conflating them costs money. Measured on the
   same turn, every row:**

   | where the settings live | what the compaction request receives |
   | --- | --- |
   | agent-level `model_settings=` | **nothing at all** — `model_settings=None` |
   | per-run `agent.run(..., model_settings=)` | **nothing at all** |
   | a capability's `get_model_settings()` | **nothing at all** |
   | `agent.override(model_settings=)` | **nothing at all** |
   | the model object's own `settings=` | all of them |
   | `summarizer_model_settings=` | these, shallow-replacing the model's per key |

   **Nothing that lives on the parent agent or its run reaches the compaction
   request** — which is four of the six rows, including the per-request one a
   multi-tenant host reaches for first.
   `SummarizingCompaction._summarize` builds a *separate*
   `Agent(model, instructions=…, model_settings=self.model_settings)`, so the
   parent agent's `model_settings` are never handed to it — not `extra_headers`,
   not `max_tokens`, not `temperature`, and this has nothing to do with merging.
   Measured: a parent agent carrying
   `{"temperature": 0.1, "max_tokens": 77, "extra_headers": {"authorization":
   "Bearer gw"}}` with `summarizer_model_settings` unset sends the main request
   with all three and the compaction request with `model_settings=None`. A host
   behind a gateway that puts its auth header at the agent level therefore sends
   an **unauthenticated and unbounded** compaction request, deep in a long
   conversation, after a paid main turn. `elja/application.py` says the same thing
   in one line: settings on one agent never reach another.

   So anything the compaction request needs — gateway auth, a token cap, run
   identity — must live on the **model object** (`settings=ModelSettings(...)`,
   which does survive; measured) or be restated in `summarizer_model_settings`.

   **And what you restate replaces rather than merges.**
   `merge_model_settings` is a shallow `base | overrides`, so an `extra_headers`
   in `summarizer_model_settings` wipes the model's own wholesale. Measured: a
   model carrying `{"authorization": ...}` plus
   `summarizer_model_settings={"extra_headers": {"x-phase": "compaction"}}` sends
   the compaction request with `x-phase` and **no** `authorization`, while
   `max_tokens` and `temperature` from the model survive untouched. Restate every
   header you still need.
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
  turn's history is not saved. Two sinks, two different answers, on purpose. A
  third caller-supplied callback, `EljaDeps.confirm`, is also unsuppressed — but
  "unsuppressed" is as far as the claim goes. A raise there ends the run only on the
  parent's own tool calls; inside a delegation, `subagents.py`'s `except Exception`
  converts it into a `ModelRetry`, so the model re-delegates until the tool's retry
  budget is spent. Measured on one configured sub-agent: a raising approver was
  invoked four times and the run ended as `UnexpectedModelBehavior: exceeded max
  retries`, having spent six extra paid requests. It only applies to the CLI deps
  type, which the `PermissionGate` note below says is not usable on this path
  anyway.
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

## Composing compaction

`build_compaction(settings, ...)` takes every part of the policy as an argument,
because the defaults are tuned for a local workspace:

| argument | why a host changes it |
| --- | --- |
| `cleared_placeholder` | The default says "re-run the tool if you need it again" and names `.elja/spill/`. Safe for idempotent reads in a workspace; an invitation to double-write anywhere else. Point it at your own store. |
| `summary_prompt` | The default carries a note about reloading elja skills, which a host without skills has no reason to ship. This is the summarizer's *user* turn; upstream's `instructions` (its system prompt) is not exposed. Must *substitute* `{messages}`, checked at construction by rendering it twice — a doubled `{{messages}}` is a literal and is refused, while `{messages!r}` and `{messages:>10}` are fine. |
| `summarizer_model` | A different provider, a cheaper model, or separate budget attribution (see above). |
| `receipts` | Leaves a deterministic note where history was summarized away. With a capability implementing the harness's `TranscriptHandleProvider` protocol attached, the receipt carries a handle to your persisted transcript. Note one accumulates per compaction across a long caller-owned history, each with its own dropped-message count. |
| `on_unfit` | What happens to a request compaction cannot bring under the target. `None` (the default) raises `TargetUnreachableError`; a gate that returns sends the request. See "The summarizer is bounded" below. |

Replacing the policy wholesale is always available: build your own
`BoundedTieredCompaction` (from `elja.compaction`) with your own tiers and pass it
as a capability instead of calling the factory. Upstream's plain `TieredCompaction`
works too, and is unbounded.

**Put `ReportContextUsage` last.** None of these capabilities declare an
ordering, so list order decides what reporting measures. Measured on the same
turn: ~1k tokens with reporting after compaction, ~6k with it before, where the
second number describes a request that was never sent. Exact figures move with
the harness's estimator, so treat the six-fold gap as the finding, not the
numbers.

**What survives, measured not assumed.** Pinned parts
(`pydantic_ai_harness.compaction.pin`) survive every tier. Tool call/result
pairing stays valid across both tiers. So do the current request, the first user
message when it is preserved, leading system policy, your placeholder text,
receipts, and each part's own metadata (a `TextContent.metadata`, a
`ToolReturnPart.metadata`) — pinned on the path where compaction fits and on the
one where it cannot (`tests/test_compaction_bounds.py`). Message-level `metadata`
is a different matter and not compaction's doing: pydantic-ai drops it when it
merges consecutive requests on replay, by design (`_merge_consecutive_messages`).

## The summarizer is bounded

Some content no tier may remove. elja calls it the **protected floor**:
instructions and tool definitions (sent on every request), leading system policy,
the first user message when `preserve_first_user_message` is on, pins, and the
current request. Upstream re-injects or preserves all of it *after* the tail is
trimmed, so `keep_tokens` cannot bound it — and when the floor alone exceeds
`target_tokens`, the estimate can never reach target. In 0.5.0 the summarizing
tier then fired before **every** model request for the rest of the run: measured
over a six-step turn, six paid summarizer calls where an ordinary history needs
one.

`build_compaction` now returns a `BoundedTieredCompaction`, which is upstream's
`TieredCompaction` (an `isinstance` check still holds) with four additions:

1. **A budget.** At most `CompactionConfig.max_summarizer_requests` summarizer
   requests per run (default 3; 0 means masking only). Counted at the model, not
   per attempt, because a summarizer reply with no text makes pydantic-ai's
   private summary agent retry once — one attempt, two paid requests. A request
   past the budget is refused before it reaches your guard.
2. **A prediction.** Before paying for a summary, the floor is measured in the
   same units as the target. If it alone exceeds the target, no summary is paid
   for, and the request is reported as `protected_floor`.
3. **A latch.** A summary that ends over the target puts the run in *overflow
   mode*: from then on the summarizer runs only once the summarizable remainder
   (estimate minus floor) exceeds a whole `target_tokens` — real progress, not a
   re-summary of the summary on every request. Reported as `no_progress`. The
   floor-exceeded case behaves the same way for a host that chose to proceed.
4. **A gate.** Every request that would go out over target is put to
   `on_unfit(ctx, report)` with a typed `TargetUnreachable`. With no gate (the
   default) elja raises `TargetUnreachableError(report)` and **nothing is sent**.
   A gate that returns lets the request go as compacted; one that raises stops
   the run with its own exception.

**Soft target, not hard limit.** `TargetUnreachable` is about `target_tokens`,
elja's compaction target, set well below any real window. The provider's input
limit is reported separately as `window_tokens` when pydantic-ai can resolve it,
and is `None` otherwise — elja never guesses it. All numbers are estimates in the
units the target is compared against: provider-reported usage where the history
carries it (`anchored=True`), the characters-per-token heuristic where it does not.

| field | meaning |
| --- | --- |
| `reason` | `protected_floor`, `no_progress` or `budget_exhausted` |
| `estimated_tokens` | the request when compaction gave up on it. For `protected_floor` that is before the summarizing tier, since the report comes before any payment; a host that proceeds may still get one summary that shrinks it, so a `ReportContextUsage` placed after compaction reads the exact figure |
| `protected_tokens` | the floor; a summary cannot go below it |
| `summarizer_requests` / `max_summarizer_requests` | spent this run, and the budget |
| `occurrence` | 1 for the first report in a run, then 2, 3, … — check for 1 to act once per run |
| `anchored`, `window_tokens` | the estimate's reliability, and the hard limit when known |

**When the state resets.** All of it — budget, latch, report count — belongs to
one run. `for_run` gives each run a fresh copy, so a new `agent.run` starts with
the whole budget again, and two concurrent runs of one agent do not share one. A
manual `compact_now` call is its own run. Within a run:

- the floor is measured afresh on every request, so a huge tool result is
  protected only while it *is* the current request; a step later masking may
  clear it and the run fits again, with nothing paid for;
- overflow mode ends on the first request that compacts under the target again,
  and from there the run is back to ordinary compaction;
- the budget does not refill.

So a host that proceeds past `protected_floor` pays nothing more for unchanged
content: a summary only when the reducible part has grown by a whole target since
the last one, and never more than the budget. One that raises pays nothing at
all. The one case with a cost per run is `no_progress` on a fresh run: the floor
fits but floor plus summary does not, which the prediction cannot know before the
summary exists — at most one summary per run, then the outcome.

**Handling it.** `examples/server_agent.py`'s `run_turn_within_capacity` turns the
error into a typed `OverCapacity` value with the checkpoint beside it. A write
that completed earlier in the turn is in that checkpoint, receipt and all, so a
replay does not repeat it:

```python
try:
    output, history = await run_turn(agent, deps, prompt, history=history, recorder=recorder)
except TargetUnreachableError as error:
    if error.report.reason == "protected_floor":
        ...  # the brief, pin, policy or current request is itself too big: ask for less
    else:
        ...  # this turn outgrew its compaction budget: start fresh, or raise the budget
    history = recorder.partial  # what completed, repaired for replay
```

`elja chat` and configured sub-agents pass `proceed_over_target` instead: one
status line per run, then send. Their target is a conservative slice of a local
model's window, so a dead turn would be the worse outcome there — and it is the
wrong gate for a host that must account for every over-target request.

**Protected content no longer grows.** A `SystemPromptPart` in caller-owned
history — what `Agent(system_prompt=...)` produces — used to multiply: the harness
copies leading system parts into each new summary message, first-message
preservation keeps the original as well, and pydantic-ai merges consecutive
requests on replay, so each compacting turn left one more policy copy and one more
superseded summary for the next to preserve. elja now removes exactly those after
each summary: a system part seen twice by exact equality (content, timestamp and
dynamic ref, so two parts that merely say the same thing are both kept; the
leading copy stays), and summaries a newer one replaced. User prompts are never
touched. What still accumulates is the user's own prompts — the same merge folds
each compacted turn's prompt into the preserved first request — and since those
are real messages they stay, and the floor counts them.

Prefer `instructions=` for policy anyway: it is still protected (it goes on every
request), but it lives outside history, where nothing can accumulate.

**`cleared_placeholder` covers the compaction placeholder only.** If you also use
elja's built-in toolset, its output-capping message separately names
`.elja/spill/` and suggests paging with `run_shell`. On the embedded path with
your own tools that never arises; if you mix the two, that text is still there.

## Persistence

History is caller-owned on the embedded path. Pass `message_history=` and
serialize with pydantic-ai's own message adapter (`ModelMessagesTypeAdapter`).
elja's named JSON `Session` remains for CLI users and is not involved here; the
embedded path writes no files at all.

Use the adapter rather than a representation of your own. Opaque provider state —
thinking signatures, provider ids, reasoning item ids — round-trips because the
adapter owns the schema, and hand-rolling a second one is how that state gets
dropped.

**What `run_stream_events` hands back is the whole conversation**, history
included, not just the messages this turn produced. Persist that; a host saving
`new_messages()` alone drops every earlier turn on each save.

**Switching providers mid-history is a decision, not a no-op.** pydantic-ai will
not replay one provider's thinking *signature* to another. It does replay the
reasoning *content*, on every mapping checked: OpenAI's Responses path sends a
foreign thinking part either as an assistant message wrapped in the profile's
thinking tags or as a reasoning summary with no encrypted content, and the Anthropic
and Google mappings forward the content too. So a filter keyed on the signature
hides exactly the parts that cross. Decide explicitly —
log it, drop it, or start a fresh conversation. `foreign_thinking_parts` in the
example shows what to look at, including the two things that make "foreign"
subtle: a part carrying no `provider_name` inherits its message's, and provider
families alias (`google`/`google-gla`, `google-vertex`/`google-cloud`).

## Events, checkpoints and cancellation

**The stream does not label the answer while it is arriving.**
`FinalResultEvent` means "a part that could be the final output has started", so
a turn that narrates and then calls a tool emits it too — twice in one turn,
measured. Take the answer from `AgentRunResultEvent` and nowhere else. A host
that accumulates text deltas into its answer ships the narration with it. For the
same reason, a display sink fed from the stream sees the narration *and* the
answer; nothing at that point can separate them.

Two smaller facts a UI will meet: an empty (`""`) text delta is a real event
rather than something the framework filters, and a thinking part can arrive as
deltas like any other part.

**A turn that dies still owes the host its completed work.** This is the one that
bites: a tool that already ran has changed the host's world, so a failure that
discards the run's messages leaves the host with an effect and no record of it —
and the next turn, replaying a history with no trace of the write, writes again.
Everything the run completed is reachable on the handle, synchronously, from
inside the `except`:

```python
async with agent.run_stream_events(prompt, deps=deps, message_history=history) as events:
    try:
        async for event in events:
            ...
    except BaseException:
        try:
            # `all_messages()` is the run's own live list, so copy it — and copy it
            # through a repair, or this history cannot be replayed (next paragraph).
            partial = close_interrupted_calls(events.all_messages())
            spent = events.usage
        except UserError:
            pass          # the run never bound: nothing completed, nothing to keep
        raise
```

`except BaseException`, not `except Exception`: an external cancellation — a SIGTERM,
an enclosing timeout — is a `BaseException`, and it is the case where losing the
checkpoint costs the most. Both accessors raise `UserError` until the first
iteration binds the run, and that window is real: the background run task is created
but not awaited, so a cancellation landing in the few event-loop steps before the
binding arrives here unbound. Nothing completed then, so the checkpoint is empty —
and the guard is what stops a framework complaint about iteration order replacing
the failure the host has to see.

**A checkpoint with an unanswered tool call cannot be replayed, and that is exactly
the case you are checkpointing for.** Not because a provider sees it — pydantic-ai
repairs dangling calls at send time *unconditionally*, last response included, so one
never reaches the wire. The refusal is earlier, in the pass that decides how to resume
a history you hand back: replaying a dangling call with a new user prompt raises
`UserError('Cannot provide a new user prompt when the message history contains
unprocessed tool calls.')`, and replaying it *without* a prompt re-executes the call —
the duplicate side effect you were avoiding, one level down. That resumption pass is
the one that leaves the last response alone, because its calls are the live frontier
`deferred_tool_results` may still answer. A turn that dies *inside* a tool leaves its
dangling call exactly there. The one shape that self-heals is a response with two
calls where one returned, which is why a single-tool turn fails where a parallel one
does not.

The distinction matters for what you do with it: repair a history you are about to
**persist and replay**, never one you are about to hand straight back to the model,
and never a successful `DeferredToolRequests` history — upstream preserves that
frontier on purpose, and closing it is the defect below.

So close the frontier yourself, the way pydantic-ai would — a `ToolReturnPart` with
`outcome='interrupted'`, the response's own timestamp, and the synthesized marker so
upstream's pass sees nothing left to do. `close_interrupted_calls` in
`examples/server_agent.py` is twenty lines of exactly that, and the measured
difference is a resumable conversation versus a wedged one.

This is the host-selected checkpoint: where the save goes is the host's call, and the
contract is that the evidence is neither thrown away nor handed back in a shape
nothing will accept.

**Cancellation is pydantic-ai's, not elja's.** Prefer
`run_stream_events(cancellation_token=...)` or `AgentRunEvents.cancel()` over
wrapping the call in `asyncio.timeout`:

- A first-party cancellation arrives as `RunCancelled`, an ordinary catchable
  outcome, and carries the run's history on the exception. But **that history is
  not already resumable**: it closes out an interrupted call only where a sibling
  call in the same response already returned, so a tool stopped inside its own body
  leaves its call dangling there and replaying it with a new prompt raises
  `UserError`. Measured. Persist the checkpoint you repaired, not
  `RunCancelled.all_messages()`.
- An external asyncio cancellation must keep propagating for the enclosing
  timeout scope to unwind, so a host on that path has to re-raise and recover the
  state from the exception chain with `RunCancelled.from_cancellation`.

**Compaction that works tells the host nothing.** There is no "history was
rewritten" callback: the only one the harness package offers is
`ReportContextUsage.on_usage`, which carries a reading and no messages. A host
that wants to surface "history compacted" infers it from a reading that drops, or
from its own copy of the transcript. Compaction that *cannot* reach its target is
the exception, and it is loud and typed — see "The summarizer is bounded". Related, and a trap for anything billing on
it: readings are taken in `before_model_request`, which runs *upstream* of the
usage-limit check, so a turn that trips its limit reports one more reading than
it made requests.

`examples/server_agent.py` is all of the above as running code, driven by
`tests/test_server_contracts.py` so it cannot rot.

## Upgrading from 0.5.0

Release notes for the bounded-compaction change, verified against pydantic-ai-slim
2.36.0 and pydantic-ai-harness 0.27.0 (Python 3.13 locally; CI runs 3.11–3.13).

- **Intentional behavior change, for safety.** A request compaction cannot bring
  under `target_tokens` is no longer sent after a summary paid for on every
  request. With `build_compaction(settings)` and no `on_unfit`, the run raises
  `TargetUnreachableError` — before any summary is paid for when the protected
  floor alone is too big. A host that relied on the old behavior, and can afford
  over-target requests, passes a gate that returns (`proceed_over_target`, or its
  own). `elja chat` and sub-agents do this already, so the CLI keeps answering and
  shows one status line per run instead.
- **New budget.** `[compaction] max_summarizer_requests` (default 3) caps paid
  summarizer requests per run. A run that legitimately needs more than that many
  summaries now gets `budget_exhausted` instead of a fourth; raise it if your runs
  are that long.
- **New types**, all in `elja.compaction`: `BoundedTieredCompaction`,
  `TargetUnreachable`, `TargetUnreachableError` (an `AgentRunError`), `OnUnfit`,
  `UnfitReason`, `proceed_over_target`.
- **Type name.** The capability is `BoundedTieredCompaction`, a subclass of
  `TieredCompaction`. `isinstance` checks are unaffected; a check on
  `type(cap).__name__` is not. The `compact_messages` span still says
  `TieredCompaction`.
- **Histories shrink, not grow.** Copies of system policy and superseded
  summaries that 0.5.0 left in caller-owned histories are removed at the next
  compaction. Several old summaries are folded into one anchor first, so nothing
  they said is lost.
