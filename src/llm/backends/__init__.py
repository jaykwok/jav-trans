"""Translation backend registry and process-wide instance lifecycle."""

from __future__ import annotations

from core.typed_config import FALL_BACK, env_float

import os
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

from core import resources
from core.cancellation import raise_if_cancelled
from llm import request_config
from llm.errors import BackendLeaseInvalidatedError, BackendUnavailableError
from llm.settings import normalize_backend_name as _normalize_name
from llm.settings import selected_backend_name


class TranslationBackend(Protocol):
    def chat_completion(self, messages: list[dict], **kwargs) -> str: ...
    def cache_identity(self) -> str: ...
    def name(self) -> str: ...


BackendFactory = Callable[[], TranslationBackend]

_BACKEND_REGISTRY: dict[str, BackendFactory] = {}
_BACKEND_INSTANCES: dict[str, TranslationBackend] = {}
# One entry per live claim, holding the *instance* it was granted on: a lease is
# ownership of a concrete resource, not of a backend name. The name alone let a
# release close whatever instance happened to answer to it later.
_BACKEND_LEASES: dict[str, list[TranslationBackend]] = {}
# Instances whose close is in flight. The close blocks (it waits for a process
# to exit), so it runs *outside* the registry lock; this marks the name as taken
# meanwhile, which is what used to be expressed by holding the lock across it.
_CLOSING: dict[str, TranslationBackend] = {}


@dataclass
class _CloseOperation:
    instance: TranslationBackend
    done: threading.Event = field(default_factory=threading.Event)
    confirmed: bool = False


_CLOSE_OPERATIONS: dict[str, _CloseOperation] = {}
_REGISTRY_LOCK = threading.RLock()
# The instance the calling thread leased, as (name, instance). Requests resolve
# through this instead of by name, so a task cannot end up translating through
# an instance it holds no claim on.
_ACTIVE = threading.local()
# How long a task waits for a retiring generation before it says so in the log.
# It keeps waiting afterwards: attaching to a generation that is on its way out
# is exactly what retirement exists to prevent, and killing the task that still
# holds it is a different, explicit operation.
_RETIRE_WAIT_S = env_float("LOCAL_RETIRE_WAIT_S", 600.0, minimum=0.001, out_of_range=FALL_BACK)


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a backend factory.

    Instances are shared process-wide. Re-registering requires ``replace=True``
    and invalidates the old instance, making extension mistakes visible rather
    than silently overriding a backend.
    """
    normalized = _normalize_name(name)
    if not callable(factory):
        raise TypeError("Backend factory must be callable")
    previous = None
    with _REGISTRY_LOCK:
        if normalized in _BACKEND_REGISTRY and not replace:
            raise ValueError(f"Translation backend already registered: {normalized}")
        candidate = _BACKEND_INSTANCES.get(normalized)
        if candidate is not None and _begin_close_locked(normalized, candidate):
            previous = candidate
        _BACKEND_REGISTRY[normalized] = factory
    if previous is not None:
        _finish_close(normalized, previous)


def _get_backend_locked(normalized: str) -> TranslationBackend:
    if normalized in _CLOSING:
        raise BackendUnavailableError(f"翻译后端 {normalized} 正在清理，不能创建替代实例。")
    factory = _BACKEND_REGISTRY.get(normalized)
    if factory is None:
        available = ", ".join(sorted(_BACKEND_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown translation backend: {normalized}. Available: {available}"
        )
    instance = _BACKEND_INSTANCES.get(normalized)
    if instance is None:
        instance = factory()
        _validate_lifecycle(instance)
        _BACKEND_INSTANCES[normalized] = instance
    return instance


def _validate_lifecycle(instance) -> None:
    """Stateless adapters need no hooks; a cleanup-capable adapter needs all.

    Factories must not start resources: startup belongs to the request path,
    after validation. ManagedTranslationBackend enforces this at construction
    for built-ins; this check also covers structurally typed extensions.
    """
    hooks = ("close", "invalidate", "is_invalidated", "cleanup_status", "generation_retiring")
    if not any(hasattr(instance, hook) for hook in hooks):
        return
    missing = [hook for hook in hooks if not callable(getattr(instance, hook, None))]
    if missing:
        raise TypeError(f"Managed backend lifecycle incomplete: {', '.join(missing)}")


def _admission_timeout_s() -> float:
    # Read when a task starts, after application configuration has been loaded.
    return env_float("LOCAL_BACKEND_WAIT_TIMEOUT_S", 600.0, minimum=0.001, out_of_range=FALL_BACK)


def get_backend(name: str | None = None) -> TranslationBackend:
    """Return the shared backend instance for ``name``."""
    normalized = selected_backend_name(name)
    with _REGISTRY_LOCK:
        return _get_backend_locked(normalized)


@contextmanager
def backend_lease(name: str, cancel_event=None) -> Iterator[TranslationBackend]:
    """Claim the shared instance of ``name`` for the duration of one task.

    Instances are process-wide, so the task that finishes first used to close
    the local server another running task was translating through: two parallel
    videos, one `reset_backend("llamacpp")`, and the survivor's next request hit
    a dead server. A lease counts users instead - the instance is closed (and
    its VRAM released) when the last one leaves, which for a single job is still
    the moment that job ends.

    The claim names the instance it was granted on, and binds it for this thread
    so that the requests made under the lease go to *that* instance: a release
    bound to an object while every request re-resolved the backend by name meant
    a task could hold one instance and translate through another. Claiming can
    block (a retiring generation has to go first) and therefore takes the task's
    cancel event: this runs before the request layer exists, so it is the only
    thing that can stop the wait.

    The endpoint configuration is frozen here for the same reason and at the
    same moment. Naming the instance settles *which process or adapter* answers;
    it settles nothing about which model or endpoint the adapter is pointed at,
    and the settings page rewrites those while a job runs. One instance can
    serve two different models over a task's lifetime, which is how a reply from
    one ended up cached under the other's key - see `llm.request_config`.
    """
    normalized = selected_backend_name(name)
    instance = _claim_backend(normalized, cancel_event)
    previous = getattr(_ACTIVE, "lease", None)
    _ACTIVE.lease = (normalized, instance)
    previous_config = request_config.current()
    request_config.bind(request_config.capture())
    try:
        yield instance
    finally:
        request_config.bind(previous_config)
        _ACTIVE.lease = previous
        _release_backend_lease(normalized, instance)


def current_lease() -> tuple[str, TranslationBackend] | None:
    """The (name, instance) this thread holds, if any."""
    return getattr(_ACTIVE, "lease", None)


def bind_lease(lease: tuple[str, TranslationBackend] | None) -> None:
    """Carry a lease into a worker thread.

    Batch translation dispatches from a thread pool, and a fresh worker thread
    has no binding of its own - without this it would fall back to resolving the
    backend by name, which is the lookup the lease exists to replace.
    """
    _ACTIVE.lease = lease


def _refuse_unleased_lookup(caller: str) -> None:
    """Refuse to resolve a backend by name while the process is running tasks.

    Resolving by name is a legitimate answer for a caller that never took a
    lease - the CLI, a preflight check, a test. It is never the right answer for
    a thread *inside* a task that lost its binding: that thread would translate
    through whatever the settings panel currently names, which is the substitution
    the lease exists to prevent, and it would do it silently. The two cases are
    told apart by whether anything in this process holds a lease at all.
    """
    with _REGISTRY_LOCK:
        leased = any(_BACKEND_LEASES.values())
    if not leased:
        return
    raise BackendLeaseInvalidatedError(
        f"{caller}：当前线程没有任务租约，但本进程有任务正在翻译。"
        "按名称解析后端会让这个线程用上另一套设置，因此拒绝。"
        "线程池派发前请携带 llm.run_context.RunContext。"
    )


def task_backend() -> TranslationBackend:
    """The instance this task leased; the selected one when there is no lease.

    Raises if the leased instance is no longer the registered one - the settings
    were reset mid-task. Silently switching to the replacement is how a task
    ended up translating through an instance it held no claim on, which another
    task was then free to close underneath it.
    """
    lease = getattr(_ACTIVE, "lease", None)
    if lease is None:
        _refuse_unleased_lookup("task_backend")
        return get_backend()
    normalized, instance = lease
    with _REGISTRY_LOCK:
        if _BACKEND_INSTANCES.get(normalized) is instance and not _instance_invalidated(
            instance
        ):
            return instance
        if _CLOSING.get(normalized) is instance:
            raise BackendLeaseInvalidatedError(
                f"翻译后端 {normalized} 正在关闭，本次任务持有的实例已失效。"
            )
    raise BackendLeaseInvalidatedError(
        f"翻译后端 {normalized} 已在任务运行中被重置，本次任务持有的实例已失效；"
        "请重新运行该任务。"
    )


def task_backend_name() -> str:
    """The backend this task is actually translating through."""
    lease = getattr(_ACTIVE, "lease", None)
    if lease is not None:
        return lease[0]
    _refuse_unleased_lookup("task_backend_name")
    return selected_backend_name()


def _instance_invalidated(instance) -> bool:
    """Has this instance been taken out of service for good?"""
    invalidated = getattr(instance, "is_invalidated", None)
    if not callable(invalidated):
        return False
    try:
        return bool(invalidated())
    except Exception as exc:
        # "Unknown" must not become "invalidated and already cleaned": that
        # would evict a healthy live owner whose cleanup status is not pending.
        raise BackendUnavailableError("后端失效状态不可读；保留实例，拒绝领取。") from exc


def _instance_cleanup_pending(instance) -> bool:
    """Is this instance still holding a resource nobody has confirmed released?

    A read failure counts as "yes": "we could not ask" is not "there is nothing
    there", and the answer decides whether the instance may be dropped.
    """
    status = getattr(instance, "cleanup_status", None)
    if not callable(status):
        return False
    try:
        pending = status()["stuck"]
        return pending if isinstance(pending, bool) else True
    except Exception:
        return True


def _instance_retiring(instance) -> bool:
    retiring = getattr(instance, "generation_retiring", None)
    if not callable(retiring):
        return False
    try:
        return bool(retiring())
    except Exception as exc:
        raise BackendUnavailableError("后端退役状态不可读；保留实例，拒绝领取。") from exc


def _holders_locked(normalized: str, instance) -> int:
    return sum(1 for held in _BACKEND_LEASES.get(normalized, ()) if held is instance)


def _claim_step_locked(normalized: str, cancel_event) -> tuple[str, TranslationBackend | None]:
    """One attempt at a claim. Returns ("granted"|"retire"|"wait", instance).

    Deciding and granting happen in the same lock section: a check followed by a
    separate increment can be overtaken by a retirement starting in between, and
    the newcomer joins the generation it was supposed to stay out of. The cancel
    check is here too, for the same reason - acquiring this lock can take as long
    as a close, and a task cancelled while it waited must not come back to a
    fresh lease.
    """
    if normalized in _CLOSING:
        return ("wait", None)  # a close is in flight; no replacement until it ends
    instance = _BACKEND_INSTANCES.get(normalized)
    if instance is not None and _instance_invalidated(instance):
        # It is in the slot for one reason only: a close that did not confirm
        # left it the owner of a process nobody else can reach.
        if _instance_cleanup_pending(instance):
            return ("wait", None)  # waiting on that cleanup, cancellably
        # The cleanup has since succeeded - the background retry confirmed the
        # process gone - so its job is done. Evicting it here is what turns a
        # recovered resource back into a usable one: leaving it in the slot
        # meant the next task leased an instance that can only raise. Its
        # invalidation is never lifted; a new instance takes its place.
        _BACKEND_INSTANCES.pop(normalized, None)
        instance = None
    if instance is not None and _instance_retiring(instance):
        if _holders_locked(normalized, instance) > 0:
            return ("wait", None)  # someone is still translating through it
        # Nobody holds it, so retire it instead of attaching to it. The close
        # itself happens outside this lock; until it commits, the name is marked
        # closing and nothing may take its place.
        _begin_close_locked(normalized, instance)
        return ("retire", instance)
    raise_if_cancelled(cancel_event)
    instance = _get_backend_locked(normalized)
    _BACKEND_LEASES.setdefault(normalized, []).append(instance)
    return ("granted", instance)


def _claim_backend(normalized: str, cancel_event=None):
    """Wait for a claim on ``normalized``, cancellably.

    Cancelling a local request cannot prove the server stopped generating, so
    that generation is retired instead of trusted: the tasks already holding it
    finish normally, and no new task attaches. Without this the unknown never
    ends - each arriving task keeps the same server alive.

    A deadline ends admission, not cleanup: ownership stays with the old
    instance, and a timed-out task neither joins nor replaces that generation.
    """
    warn_at = time.monotonic() + _RETIRE_WAIT_S
    timeout_s = _admission_timeout_s()
    deadline = time.monotonic() + timeout_s
    while True:
        raise_if_cancelled(cancel_event)
        if time.monotonic() >= deadline:
            raise BackendUnavailableError(
                f"翻译后端 {normalized} 等待退役或清理超过 {timeout_s:g}s；"
                "本任务已停止等待，旧实例仍由清理流程持有。请查看 GPU 诊断并在清理完成后重试。"
            )
        with _REGISTRY_LOCK:
            action, instance = _claim_step_locked(normalized, cancel_event)
        if action == "granted":
            return instance
        if action == "retire":
            # Started, not awaited: the close blocks until a process exits, and
            # this thread has a cancel event to stay responsive to. It falls
            # through to the same `_CLOSING` wait every other claimer uses.
            _start_close(normalized, instance)
            continue
        if time.monotonic() >= warn_at:
            print(
                "[WARN] 本地翻译服务仍在退役中（上一个被取消的请求可能还在生成），"
                f"已等待 {_RETIRE_WAIT_S:.0f}s，继续等待前一个任务结束（可取消）。",
                flush=True,
            )
            warn_at = time.monotonic() + _RETIRE_WAIT_S
        _wait_briefly(cancel_event, min(0.25, max(0.0, deadline - time.monotonic())))


def _wait_briefly(cancel_event, timeout: float = 0.25) -> None:
    """Sleep, but wake immediately when the task is cancelled."""
    wait = getattr(cancel_event, "wait", None)
    if callable(wait):
        if wait(timeout):
            raise_if_cancelled(cancel_event)
        return
    time.sleep(timeout)


def _close_instance(instance) -> bool:
    """Close an instance. False means it did not confirm and is still ours.

    A backend that manages a process may fail to stop it. Forgetting the
    instance then loses the only reference to that process - and the next
    `get_backend()` builds a fresh one that starts a second server next to it.

    A backend that defines `close()` must say whether it worked. `None` used to
    count as success, which is the same "silence means fine" that let a failed
    teardown look like a clean one - so it is now read as *not confirmed*.
    Backends with nothing to stop simply do not define `close()`.
    """
    close = getattr(instance, "close", None)
    if not callable(close):
        return True
    try:
        result = close()
    except Exception as exc:
        print(f"[WARN] 后端 close() 抛出异常，按未确认处理：{exc!r}", flush=True)
        return False
    if not isinstance(result, bool):
        print(
            f"[WARN] 后端 {type(instance).__name__}.close() 未返回布尔值"
            f"（{result!r}），按未确认处理。",
            flush=True,
        )
        return False
    return result


def _begin_close_locked(normalized: str, instance) -> bool:
    """Take the instance out of service and reserve the name for its close.

    The instance is invalidated *here*, before the close starts and before the
    lock is released, because removing it from the dictionary only stops the
    *next* lookup. A caller that already holds the object - it passed its lease
    check a moment ago - would otherwise go on to use it, and for a local
    backend "use it" means `_ensure_server()`: it would start a replacement
    server for the one being closed. Invalidation is one-way: an instance that
    has been taken out of service never comes back, whether or not its close
    confirmed.
    """
    if normalized in _CLOSING:
        return False
    current = _BACKEND_INSTANCES.get(normalized)
    if current is not None and current is not instance:
        return False
    _CLOSING[normalized] = instance
    if _BACKEND_INSTANCES.get(normalized) is instance:
        _BACKEND_INSTANCES.pop(normalized, None)
    invalidate = getattr(instance, "invalidate", None)
    if callable(invalidate):
        try:
            invalidate()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] 后端实例失效标记失败：{exc!r}", flush=True)
    return True


def _start_close(normalized: str, instance) -> _CloseOperation:
    """Run a reserved close on its own thread.

    The claimer that discovers a retiring generation is the one that starts its
    close, but it must not be the one that *waits* for it: a close blocks until a
    process exits, and that wait ignores cancel events. Every claimer - the one
    that triggered this close included - goes back to the same cancellable wait
    on `_CLOSING`. Cancelling that wait leaves the close running: the resource
    still has to be released.
    """
    with _REGISTRY_LOCK:
        existing = _CLOSE_OPERATIONS.get(normalized)
        if existing is not None:
            if existing.instance is not instance:
                raise RuntimeError("Close operation identity mismatch")
            return existing
        if _CLOSING.get(normalized) is not instance:
            raise RuntimeError("Close requires a reserved instance")
        operation = _CloseOperation(instance)
        _CLOSE_OPERATIONS[normalized] = operation
        try:
            threading.Thread(
                target=_perform_close,
                args=(normalized, operation),
                name=f"backend-close-{normalized}",
                daemon=True,
            ).start()
        except BaseException:
            _CLOSE_OPERATIONS.pop(normalized, None)
            # No close ran, so cleanup_status() may still say "not stuck".
            # Keep the reservation: restoring an invalidated live instance to
            # the normal slot would let a claimer evict it as already cleaned.
            raise
        return operation


def _finish_close(normalized: str, instance) -> bool:
    """Bound the caller's wait, never the lifetime of the resource owner."""
    # A typo in configuration must not prevent cleanup from starting.
    timeout = env_float("LOCAL_BACKEND_CLOSE_WAIT_S", 15.0, minimum=0.001, out_of_range=FALL_BACK)
    operation = _start_close(normalized, instance)
    if not operation.done.wait(timeout):
        print(f"[WARN] 后端 {normalized} 清理超过 {timeout:g}s；后台继续清理，不会启动替代实例。", flush=True)
        return False
    return operation.confirmed


def _perform_close(normalized: str, operation: _CloseOperation) -> None:
    """One blocking close per name; commit ownership and completion together."""
    instance = operation.instance
    confirmed = False
    try:
        confirmed = _close_instance(instance)
    finally:
        with _REGISTRY_LOCK:
            if _CLOSING.get(normalized) is instance:
                _CLOSING.pop(normalized, None)
            if not confirmed and _BACKEND_INSTANCES.get(normalized) is None:
                # It still owns a process, refuses to start a replacement, and
                # is the only thing that can retry the cleanup.
                _BACKEND_INSTANCES[normalized] = instance
            operation.confirmed = confirmed
            if _CLOSE_OPERATIONS.get(normalized) is operation:
                _CLOSE_OPERATIONS.pop(normalized, None)
            operation.done.set()


def _release_backend_lease(normalized: str, instance=None) -> bool:
    """Drop one claim; close the instance when it was the last claim *on it*.

    Returns whether this release closed anything. The instance is matched by
    identity: a claim taken on one server generation must not be settled against
    the instance that replaced it.
    """
    with _REGISTRY_LOCK:
        held = _BACKEND_LEASES.get(normalized)
        if not held:
            # Never leased: another task's instance is not ours to close.
            return False
        if instance is None:
            instance = _BACKEND_INSTANCES.get(normalized)
        for index, candidate in enumerate(held):
            if candidate is instance:
                del held[index]
                break
        else:
            return False
        if not held:
            _BACKEND_LEASES.pop(normalized, None)
        if instance is None or any(other is instance for other in held):
            return False  # somebody else is still using this very instance
        # Reserve the name, then close outside the lock: the close waits for a
        # process to exit, and holding the lock across it makes every claim -
        # cancelled or not - queue on a mutex until it finishes.
        # A reset may already have closed this lease's old instance, or be
        # closing it now. A late release must not close it twice or reserve the
        # replacement's name on behalf of that old instance.
        if _BACKEND_INSTANCES.get(normalized) is not instance:
            return False
        if not _begin_close_locked(normalized, instance):
            return False
    return _finish_close(normalized, instance)


def backend_lease_count(name: str | None = None) -> int:
    with _REGISTRY_LOCK:
        return len(_BACKEND_LEASES.get(selected_backend_name(name), ()))


def _visible_instance(normalized: str):
    """The instance a status read should describe.

    An instance whose close is in flight is still the owner of its process, so a
    status read has to see it - reporting "nothing here" for the seconds a close
    takes is the same all-clear-while-resident that these statuses exist to
    prevent.
    """
    return _BACKEND_INSTANCES.get(normalized) or _CLOSING.get(normalized)


def local_backend_drain_status() -> dict:
    """Is a local server still working on a request nobody is waiting for?

    Aborting the HTTP request proves this process stopped reading, not that
    llama-server stopped generating - and the difference is VRAM the user can
    see. Reported rather than acted on: the process is shared, so cancelling one
    job must never kill it out from under another.
    """
    # No registry lock: this runs on the web event loop, and waiting on a lock a
    # lifecycle operation holds would stall every request including 取消. A plain
    # dict read is atomic and the worst it can return is the instance from a
    # moment ago.
    instance = _visible_instance("llamacpp")
    status = getattr(instance, "drain_status", None)
    if not callable(status):
        return {"draining": False, "since": 0.0}
    try:
        return dict(status())
    except Exception:
        return {"draining": False, "since": 0.0}


_NO_LOCAL_CLEANUP = {
    "stuck": False,
    "pid": None,
    "attempts": 0,
    "last_error": "",
    "generation": 0,
    "process_state": "gone",
    "handle_open": False,
    "blocks_gpu": False,
}


def local_backend_cleanup_status() -> dict:
    """A local server we tried to stop and could not confirm."""
    # Same reason as above: no registry lock on a status read.
    instance = _visible_instance("llamacpp")
    status = getattr(instance, "cleanup_status", None)
    if not callable(status):
        return dict(_NO_LOCAL_CLEANUP)
    try:
        return dict(status())
    except Exception as exc:
        # "We could not read it" is not "there is nothing there". Report the
        # unknown instead of an all-clear the caller would act on.
        unknown = dict(_NO_LOCAL_CLEANUP)
        unknown.update(stuck=True, last_error=f"状态读取失败：{exc!r}", process_state="unknown", blocks_gpu=True)
        return unknown


def retry_local_backend_cleanup() -> bool:
    """Retry a *recorded stuck* local server. Blocking - call from a thread.

    Goes through the resource registry, which only ever retries records already
    waiting for cleanup. This used to call the backend's cleanup directly and
    unconditionally, so pressing 重试清理 - for anything, including an unrelated
    media orphan - shut down the healthy shared server another job was still
    translating through.
    """
    instance = _BACKEND_INSTANCES.get("llamacpp")
    summary = resources.retry_pending("local_server")
    if resources.pending("local_server"):
        return False
    if instance is not None and summary["retried"]:
        # Confirmed gone: now it may leave the registry, if nobody is using it.
        with _REGISTRY_LOCK:
            if not _BACKEND_LEASES.get("llamacpp"):
                if _BACKEND_INSTANCES.get("llamacpp") is instance:
                    _BACKEND_INSTANCES.pop("llamacpp", None)
    return True


def reset_backend(name: str | None = None) -> None:
    """Drop cached instances, primarily for configuration changes and tests.

    Unconditional by design: it answers an explicit user action (switching the
    backend or the server path) or a test teardown, both of which mean "this
    instance must not be used again" even while a task still holds a lease.
    Task-scoped cleanup goes through ``backend_lease`` instead.
    """
    with _REGISTRY_LOCK:
        if name is None:
            instances = list(_BACKEND_INSTANCES.items())
        else:
            normalized = selected_backend_name(name)
            instance = _BACKEND_INSTANCES.get(normalized)
            instances = [(normalized, instance)] if instance is not None else []
        # Reserve every name first: no replacement may be loaded while these are
        # closing. The close itself waits for a process to exit, so it happens
        # after the lock is released - a task claiming a lease meanwhile waits on
        # the reservation, where its cancel event can still reach it.
        for key, instance in instances:
            _begin_close_locked(key, instance)
    for key, instance in instances:
        # A close that does not confirm puts the instance back: "must not be
        # used again" still holds - the instance itself refuses to serve or
        # restart once its process is unconfirmed - but dropping it would strand
        # that process with no owner.
        _finish_close(key, instance)


def list_backends() -> list[str]:
    with _REGISTRY_LOCK:
        return sorted(_BACKEND_REGISTRY)


def _register_builtin_backends() -> None:
    from llm.backends.llamacpp_server import LlamaCppServerBackend
    from llm.backends.openai_compat import OpenAICompatBackend

    # Two backends, matching the two prompt contracts: an OpenAI-compatible API
    # for the JSON batch contract, and llama.cpp for the local per-line one. The
    # in-process Transformers backend was removed on 2026-08-05 - it was a third
    # way to run a local model that no shipped model targeted.
    register_backend("openai", OpenAICompatBackend)
    register_backend("llamacpp", LlamaCppServerBackend)


_register_builtin_backends()


__all__ = [
    "BackendLeaseInvalidatedError",
    "TranslationBackend",
    "backend_lease",
    "backend_lease_count",
    "bind_lease",
    "current_lease",
    "get_backend",
    "list_backends",
    "local_backend_cleanup_status",
    "local_backend_drain_status",
    "register_backend",
    "reset_backend",
    "retry_local_backend_cleanup",
    "selected_backend_name",
    "task_backend",
    "task_backend_name",
]
