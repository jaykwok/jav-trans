from __future__ import annotations

import asyncio

import httpx
import pytest

from core import config


def _clear_proxy_environment(monkeypatch):
    for key in config._PROXY_ENV_KEYS + config._NO_PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("client_kind", ["sync", "async", "openai"])
def test_socks5_environment_carries_http_request(monkeypatch, client_kind):
    """Exercise real SOCKS negotiation without contacting an external host."""
    _clear_proxy_environment(monkeypatch)
    asyncio.run(_request_through_socks5(monkeypatch, client_kind))


async def _request_through_socks5(monkeypatch, client_kind):
    completed = asyncio.get_running_loop().create_future()

    async def serve_proxy(reader, writer):
        try:
            version, method_count = await reader.readexactly(2)
            assert version == 5
            assert 0 in await reader.readexactly(method_count)
            writer.write(b"\x05\x00")
            await writer.drain()
            # The hostname must reach the proxy without a local DNS lookup.
            assert await reader.readexactly(4) == b"\x05\x01\x00\x03"
            hostname_length = (await reader.readexactly(1))[0]
            assert await reader.readexactly(hostname_length) == b"proxy-test.invalid"
            assert int.from_bytes(await reader.readexactly(2), "big") == 80
            writer.write(b"\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x50")
            await writer.drain()
            request = await reader.readuntil(b"\r\n\r\n")
            expected_path = b"/v1/check" if client_kind == "openai" else b"/check"
            assert request.startswith(b"GET " + expected_path + b" HTTP/1.1\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n"
                b"Connection: close\r\n\r\nsocks-ok"
            )
            await writer.drain()
            completed.set_result(None)
        except Exception as exc:
            completed.set_exception(exc)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve_proxy, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("PROXY_PROTOCOL", "socks5")
        monkeypatch.setenv("PROXY_HOST", "127.0.0.1")
        monkeypatch.setenv("PROXY_PORT", str(port))
        # Register changes with monkeypatch before config exports the values.
        for key in config._PROXY_ENV_KEYS + config._NO_PROXY_ENV_KEYS:
            monkeypatch.setenv(key, "")
        config.sync_network_proxy_environment()

        url = "http://proxy-test.invalid/check"
        if client_kind == "openai":
            from llm.backends.openai_compat import _make_async_client

            monkeypatch.setenv("API_KEY", "test-only-key")
            monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "http://proxy-test.invalid")
            async with _make_async_client() as client:
                assert await client.with_options(max_retries=0, timeout=3).get("/check", cast_to=str) == "socks-ok"
            await asyncio.wait_for(completed, timeout=3.0)
            return
        if client_kind == "async":
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(url)
        else:
            def request_sync():
                with httpx.Client(timeout=3.0) as client:
                    return client.get(url)

            response = await asyncio.to_thread(request_sync)
        await asyncio.wait_for(completed, timeout=3.0)
        assert response.status_code == 200
        assert response.text == "socks-ok"


def test_socks5_environment_keeps_local_translation_direct(monkeypatch):
    _clear_proxy_environment(monkeypatch)
    for key in config._PROXY_ENV_KEYS + config._NO_PROXY_ENV_KEYS:
        monkeypatch.setenv(key, "")
    config.apply_network_proxy_environment("socks5://127.0.0.1:1080")
    with httpx.Client() as client:
        for host in ("127.0.0.1", "localhost", "[::1]"):
            local = client._transport_for_url(httpx.URL(f"http://{host}:8080/v1"))
            assert getattr(local._pool, "_proxy_url", None) is None
        remote = client._transport_for_url(httpx.URL("https://huggingface.co/"))
        assert getattr(remote._pool, "_proxy_url", None) is not None
