"""The endpoint configuration a task runs against, frozen when the task starts.

A translation task resolves its backend once (``backend_lease``) and then keeps
resolving *everything else* per request, straight out of ``os.environ`` - which
the settings page writes to while jobs are running. So the model a batch was
sent to and the model its cache key names were read at different moments, and
nothing required them to agree:

    engine takes the cache identity for model A
    -> settings switch to model B
    -> the worker's request is built with model B
    -> B's answer is written under A's cache key
    -> settings switch back to A
    -> the next run "hits cache" for A and never sends a request at all.

Freezing the Python object in the registry does not fix that, because the
object was never what varied.

This module is the seam: :func:`getenv` has exactly ``os.getenv`` semantics,
against the task's snapshot rather than against the live environment. Request
construction, the client factories, the preflight checks and the cache identity
all read through it, so they cannot disagree about which endpoint this task is
talking to - whatever the settings page does meanwhile.

Two deliberate limits:

* Only the keys in :data:`SNAPSHOT_KEYS` are frozen: the ones that name the
  endpoint, the credential and the model, i.e. the ones that decide both where
  a request goes and what its answer may be cached as. Deployment-wide values
  (``LLM_STRUCTURED_OUTPUT``) are not per-job choices and are not settings-page
  fields, so they stay ordinary module constants. Asking here for a key that
  was never snapshotted raises rather than silently falling back to the live
  environment, because a silent fallback is the bug this module exists for.
* A snapshot binds to one thread. The batch pool hands work to threads that
  have none of their own, so the dispatcher rebinds it exactly where it rebinds
  the backend lease - the two travel together or neither is worth having.

The prompt profile rides along for the same reason, even though it is not an
environment value. It is *derived* from several (backend, model name, the pin)
and it was being derived twice per run - once for the request and once, later,
for the cache key. A model switched in between gave `json@v3.4` output filed
under `hymt2@hymt2-line-v1`. Resolved once at the top of a task and bound here,
there is nothing left to re-derive: the request, the token budget, the batch
cache key and the repair pass's own cache writes all read the same object.

Nothing is bound outside a task. Then :func:`getenv` *is* ``os.getenv`` and
:func:`current_profile` is None, which is what keeps the CLI, the preflight
checks run from the queue and every direct transport test reading live
configuration the way they always have.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# Settings-page fields that decide where a request goes, who it authenticates
# as, which model answers it, and how hard that model thinks. `TARGET_LANG` and
# `TRANSLATION_GLOSSARY` are absent on purpose: they reach the pipeline as
# arguments resolved when the job is built, not through the transport.
# `TRANSLATION_BACKEND` is absent for the same kind of reason - the lease
# already froze it.
SNAPSHOT_KEYS: tuple[str, ...] = (
    "API_KEY",
    "OPENAI_COMPATIBILITY_BASE_URL",
    "LLM_MODEL_NAME",
    "LLM_REASONING_EFFORT",
)

_ACTIVE = threading.local()


@dataclass(frozen=True)
class TranslationConfig:
    """One task's endpoint configuration. Immutable, and compared by value."""

    values: Mapping[str, str | None]

    def get(self, name: str, default: str = "") -> str:
        """``os.getenv`` semantics: absent yields the default, empty stays empty."""
        if name not in SNAPSHOT_KEYS:
            raise ValueError(
                f"{name} is not part of the frozen translation configuration; "
                f"add it to SNAPSHOT_KEYS or read it from os.environ directly."
            )
        value = self.values.get(name)
        return default if value is None else value

    def identity(self) -> str:
        """Short, log-safe description. Never the key - only whether there is one."""
        base_url = (self.get("OPENAI_COMPATIBILITY_BASE_URL") or "").strip()
        model = (self.get("LLM_MODEL_NAME") or "").strip()
        keyed = "key" if (self.get("API_KEY") or "").strip() else "nokey"
        return f"{base_url or 'default'}|{model or 'unset'}|{keyed}"


def capture() -> TranslationConfig:
    """Freeze the current environment for the keys a request is built from."""
    return TranslationConfig(
        values=MappingProxyType(
            {name: os.environ.get(name) for name in SNAPSHOT_KEYS}
        )
    )


def current() -> TranslationConfig | None:
    """The snapshot bound to this thread, or None outside a task."""
    return getattr(_ACTIVE, "config", None)


def bind(config: TranslationConfig | None) -> None:
    """Carry a snapshot into a worker thread.

    The batch pool dispatches from threads that hold no configuration of their
    own; without this they would fall back to the live environment, which is
    the read this module replaces.
    """
    _ACTIVE.config = config


@contextmanager
def bound(config: TranslationConfig) -> Iterator[TranslationConfig]:
    previous = current()
    bind(config)
    try:
        yield config
    finally:
        bind(previous)


def getenv(name: str, default: str = "") -> str:
    """``os.getenv`` against this task's frozen configuration.

    Outside a task there is no snapshot and this is ``os.getenv``.
    """
    config = current()
    if config is None:
        return os.getenv(name, default)
    return config.get(name, default)


def current_profile():
    """The prompt profile this task resolved, or None outside a task."""
    return getattr(_ACTIVE, "profile", None)


def bind_profile(profile) -> None:
    """Carry the resolved profile into a worker thread, as with :func:`bind`."""
    _ACTIVE.profile = profile


@contextmanager
def bound_profile(profile) -> Iterator[object]:
    previous = current_profile()
    bind_profile(profile)
    try:
        yield profile
    finally:
        bind_profile(previous)
