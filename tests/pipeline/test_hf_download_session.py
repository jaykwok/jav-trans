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
