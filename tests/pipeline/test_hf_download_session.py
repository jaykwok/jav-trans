import os
import threading

import httpx
from huggingface_hub.utils import close_session, get_session
from huggingface_hub.utils import _xet
import pytest

from core import config
from helpers_socks import socks_origin
from utils import hf_download


@pytest.fixture(autouse=True)
def isolate_clients(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in config.NETWORK_PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    monkeypatch.setattr(hf_download, "_proxy_signature", None)
    monkeypatch.setattr(_xet, "_GLOBAL_XET_HOLDER", _xet.XetSessionHolder())
    close_session()
    yield
    close_session()
    _xet.abort_xet_session()


def test_next_hub_download_uses_changed_proxy_and_disabling_clears_it():
    with hf_download.download_session():
        old_client = get_session()
        old_xet = _xet.get_xet_session()
    with socks_origin({"/check": ("200 OK", "", b"new-proxy")}) as (port, requests):
        config.apply_network_proxy_environment(f"socks5://127.0.0.1:{port}")
        with hf_download.download_session():
            current_client = get_session()
            current_xet = _xet.get_xet_session()
            assert current_client is not old_client
            assert current_xet is not old_xet
            assert current_client.get("http://proxy-test.invalid/check", timeout=3).text == "new-proxy"
        assert len(requests) == 1
    config.apply_network_proxy_environment("", clear_existing=True)
    with hf_download.download_session():
        direct = get_session()
        assert direct is not current_client
        assert _xet.get_xet_session() is not current_xet
        transport = direct._transport_for_url(httpx.URL("https://proxy-test.invalid/"))
        assert getattr(transport._pool, "_proxy_url", None) is None


def test_proxy_change_does_not_close_a_download_that_is_still_running():
    started, entered = threading.Event(), threading.Event()
    seen = []

    def next_download():
        started.set()
        with hf_download.download_session():
            seen.append(get_session())
            entered.set()

    with hf_download.download_session():
        original = get_session()
        original_xet = _xet.get_xet_session()
        config.apply_network_proxy_environment("socks5://127.0.0.1:1080")
        thread = threading.Thread(target=next_download, daemon=True)
        thread.start()
        assert started.wait(2)
        assert not entered.wait(0.05)
        assert not original.is_closed
        assert _xet.get_xet_session() is original_xet
    thread.join(3)
    assert not thread.is_alive()
    assert entered.is_set()
    assert seen[0] is not original


def test_concurrent_downloads_with_same_signature_overlap_without_blocking():
    both_inside = threading.Barrier(2, timeout=3.0)
    errors: list[BaseException] = []

    def worker():
        try:
            with hf_download.download_session():
                both_inside.wait()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    # Seed the signature first so both workers share the reader path.
    with hf_download.download_session():
        pass

    t1 = threading.Thread(target=worker, daemon=True)
    t2 = threading.Thread(target=worker, daemon=True)
    t1.start()
    t2.start()
    t1.join(3)
    t2.join(3)
    assert errors == []
    assert not t1.is_alive()
    assert not t2.is_alive()


def test_hf_endpoint_and_token_path_changes_refresh_download_session():
    with hf_download.download_session():
        first = get_session()

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.example"
    with hf_download.download_session():
        second = get_session()
    assert second is not first

    os.environ["HF_TOKEN_PATH"] = "C:/tmp/custom-hf-token"
    with hf_download.download_session():
        third = get_session()
    assert third is not second


def test_reentrant_download_session_does_not_deadlock():
    with hf_download.download_session():
        outer = get_session()
        with hf_download.download_session():
            inner = get_session()
        assert inner is outer


def test_missing_xet_module_is_handled_defensively(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._xet", None)
    with hf_download.download_session():
        assert get_session() is not None
