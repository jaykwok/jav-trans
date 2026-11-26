"""Refresh Hub connections between downloads without closing an active transfer."""

from contextlib import contextmanager
import os
import threading

from core.config import NETWORK_PROXY_ENV_KEYS


_DOWNLOAD_LOCK = threading.RLock()
_proxy_signature: tuple[str, ...] | None = None


@contextmanager
def download_session():
    """Use current proxy settings for the next complete download operation.

    snapshot_download owns its internal parallel requests. Hold this lock for
    the whole operation so a later download cannot close that shared client
    halfway through a transfer after the user changes the proxy.
    """
    global _proxy_signature
    from huggingface_hub.utils import close_session
    from huggingface_hub.utils._xet import abort_xet_session

    with _DOWNLOAD_LOCK:
        signature = tuple(os.environ.get(key, "") for key in NETWORK_PROXY_ENV_KEYS)
        if signature != _proxy_signature:
            # Xet keeps a separate Rust connection pool. The download lock
            # ensures aborting that idle session cannot interrupt a transfer.
            abort_xet_session()
            close_session()
            _proxy_signature = signature
        yield
