"""The application construction path: elja embedded in a host application.

:func:`elja.build_agent` is the *convenience* factory. It reads
:class:`~elja.settings.EljaSettings` and assembles a complete local agent:
workspace tools, filesystem skills, configured sub-agents, MCP clients,
compaction and the permission gate. That is exactly right for the CLI and for
a local script, and exactly wrong for a server that owns its own tools,
dependencies and persistence.

:func:`build_application_agent` is the other door. It assembles **nothing you
did not ask for** — no file, shell or web-search tools, no skills directory
scan, no sub-agents, no MCP subprocess, no ``.elja`` writes, no workspace at
all — and hands back a native :class:`pydantic_ai.Agent` parameterised by the
caller's own dependency type. Everything after that is ordinary pydantic-ai:
``run``, ``run_stream``, ``run_stream_events``, ``message_history``,
``output_type``, per-run ``model_settings``, ``usage`` and ``usage_limits``.

elja's own pieces remain available, by opting in rather than by default:

- **Compaction**: ``capabilities=build_compaction(settings)`` — the harness
  strategies are dependency-agnostic, so they attach to any ``deps_type``.
- **Skills**: ``capabilities=load_skills(settings)`` if you want markdown
  skills; that one does read a directory, which is why it is not implicit.
- **Permissions**: :class:`~elja.permissions.PermissionGate` reads
  ``ctx.deps.confirm`` and is therefore typed to :class:`~elja.deps.EljaDeps`.
  A host with its own approval UX should gate inside its own toolset instead.

``None`` versus empty, which differ between the two paths:

- ``instructions``: on this path ``None`` and ``""`` are equivalent — nothing
  is injected, because an application owns its own prompt. On the convenience
  path ``None`` selects :data:`elja.agent.DEFAULT_INSTRUCTIONS` and only ``""``
  means "no system prompt".
- ``toolsets``/``capabilities``: ``None`` and ``[]`` are equivalent here, and
  both mean *empty*. This path never substitutes a default for an explicit
  empty choice.

``Agent`` stores its own copy of each sequence, so two agents built from one
list cannot alias each other and a later mutation of the caller's list is not
retroactive. ``tests/test_application.py`` pins that, because it is a property
a host relies on and pydantic-ai is a pinned range, not a frozen version.

This is also the single construction site: :func:`elja.build_agent` collects
its settings-derived pieces and then calls straight through to here, so the two
paths can never drift apart in how an agent is actually assembled.
"""

from collections.abc import Sequence
from typing import TypeVar, overload

from pydantic_ai import Agent, AgentCapability, AgentModelSettings, AgentToolset

# pydantic_ai re-exports this generic alias implicitly, which strict mypy's
# no_implicit_reexport rejects; the alias in pydantic_ai.agent.abstract is a plain
# assignment and loses its type parameter, so this is the one that stays precise.
from pydantic_ai.agent import AgentInstructions  # type: ignore[attr-defined]
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputSpec

DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")


@overload
def build_application_agent(
    model: Model | KnownModelName | str,
    *,
    deps_type: type[DepsT],
    instructions: AgentInstructions[DepsT] = None,
    toolsets: Sequence[AgentToolset[DepsT]] | None = None,
    capabilities: Sequence[AgentCapability[DepsT]] | None = None,
    model_settings: AgentModelSettings[DepsT] | None = None,
) -> Agent[DepsT, str]: ...


@overload
def build_application_agent(
    model: Model | KnownModelName | str,
    *,
    deps_type: type[DepsT],
    output_type: OutputSpec[OutputT],
    instructions: AgentInstructions[DepsT] = None,
    toolsets: Sequence[AgentToolset[DepsT]] | None = None,
    capabilities: Sequence[AgentCapability[DepsT]] | None = None,
    model_settings: AgentModelSettings[DepsT] | None = None,
) -> Agent[DepsT, OutputT]: ...


def build_application_agent(
    model: Model | KnownModelName | str,
    *,
    deps_type: type[DepsT],
    output_type: OutputSpec[OutputT] = str,  # type: ignore[assignment]
    instructions: AgentInstructions[DepsT] = None,
    toolsets: Sequence[AgentToolset[DepsT]] | None = None,
    capabilities: Sequence[AgentCapability[DepsT]] | None = None,
    model_settings: AgentModelSettings[DepsT] | None = None,
) -> Agent[DepsT, OutputT]:
    """Build an agent for a host application, with no implicit elja machinery.

    See the module docstring for what this deliberately does *not* do, and for
    the ``None``-versus-empty semantics.

    Args:
        model: A model instance, or a pydantic-ai model name. An instance is
            passed through untouched — never rebuilt from its display name —
            so a wrapper that meters or gates requests survives, and so does
            its provider client, endpoint and retry configuration.
        deps_type: The caller's dependency type, injected into every tool and
            capability as ``RunContext.deps``. No elja type is required or
            mixed in.
        output_type: Native pydantic-ai output specification; defaults to
            ``str``. Structured outputs, unions and deferred-tool outputs all
            behave as they do upstream.
        instructions: Instructions, or a (possibly async) callable receiving
            the run context. ``None`` injects nothing.
        toolsets: The toolsets to register. ``None`` or ``[]`` means no tools.
        capabilities: Capabilities to attach, e.g. elja's compaction. ``None``
            or ``[]`` means none.
        model_settings: Agent-level default settings. A per-run
            ``model_settings=`` argument still overrides these, and settings on
            one agent never reach another.

    Returns:
        A native ``Agent`` bound to ``deps_type`` and ``output_type``.
    """
    return Agent(
        model,
        deps_type=deps_type,
        output_type=output_type,
        instructions=instructions,
        toolsets=toolsets,
        capabilities=capabilities,
        model_settings=model_settings,
    )
