import io
import ssl
import urllib.error
import zipfile

import pytest

import bootstrap
from core import config
from helpers_socks import CERTIFICATE, socks_origin


@pytest.fixture(autouse=True)
def isolated_proxy_environment(monkeypatch):
    monkeypatch.setattr(bootstrap.os, "environ", dict(bootstrap.os.environ))
    for key in config.NETWORK_PROXY_ENV_KEYS:
        bootstrap.os.environ.pop(key, None)


@pytest.mark.parametrize("tls", [False, True])
@pytest.mark.parametrize("credentials", [None, ("test-user", "test-password")])
def test_bootstrap_socks_transfers_http_and_verified_https(monkeypatch, tls, credentials):
    trusted = ssl.create_default_context(cafile=str(CERTIFICATE))
    monkeypatch.setattr(ssl, "_create_default_https_context", lambda **_kwargs: trusted)
    responses = {"/check": ("200 OK", "", b"socks-ok")}
    with socks_origin(responses, tls=tls, credentials=credentials) as (port, requests):
        auth = "test-user:test-password@" if credentials else ""
        bootstrap.apply_proxy(f"socks5://{auth}127.0.0.1:{port}")
        scheme = "https" if tls else "http"
        with bootstrap._opener().open(f"{scheme}://proxy-test.invalid/check", timeout=3) as response:
            assert response.read() == b"socks-ok"
        assert len(requests) == 1
        assert b"test-password" not in requests[0][1]


def test_socks_connection_refusal_is_reported_without_direct_fallback():
    with socks_origin({}, reject=True) as (port, requests):
        bootstrap.apply_proxy(f"socks5://127.0.0.1:{port}")
        with pytest.raises(urllib.error.URLError, match="状态 5"):
            bootstrap._opener().open("http://proxy-test.invalid/check", timeout=3)
        assert requests == []


def test_socks_bad_credentials_fail_without_forwarding_a_request():
    with socks_origin({}, credentials=("test-user", "correct-password")) as (port, requests):
        bootstrap.apply_proxy(f"socks5://test-user:wrong-password@127.0.0.1:{port}")
        with pytest.raises(urllib.error.URLError, match="认证失败"):
            bootstrap._opener().open("http://proxy-test.invalid/check", timeout=3)
        assert requests == []


def test_socks_https_rejects_an_untrusted_certificate(monkeypatch):
    untrusted = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    monkeypatch.setattr(ssl, "_create_default_https_context", lambda **_kwargs: untrusted)
    with socks_origin({}, tls=True, expect_tls_failure=True) as (port, requests):
        bootstrap.apply_proxy(f"socks5://127.0.0.1:{port}")
        with pytest.raises(urllib.error.URLError) as caught:
            bootstrap._opener().open("https://proxy-test.invalid/check", timeout=3)
        assert isinstance(caught.value.reason, ssl.SSLCertVerificationError)
        assert requests == []


def test_first_run_uv_download_uses_socks_for_index_redirect_and_wheel(tmp_path, monkeypatch):
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("uv-1.0.0.data/scripts/uv.exe", b"synthetic-uv-executable")
    responses = {
        "/simple/uv/": ("302 Found", "Location: /index\r\n", b""),
        "/index": ("200 OK", "", b'<a href="https://proxy-test.invalid/uv-1.0.0-py3-none-win_amd64.whl">uv</a>'),
        "/uv-1.0.0-py3-none-win_amd64.whl": ("200 OK", "", wheel.getvalue()),
    }
    trusted = ssl.create_default_context(cafile=str(CERTIFICATE))
    monkeypatch.setattr(ssl, "_create_default_https_context", lambda **_kwargs: trusted)
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(bootstrap, "UV_PYPI_INDEX", "https://proxy-test.invalid/simple/uv/")
    with socks_origin(responses, tls=True) as (port, requests):
        bootstrap.apply_proxy(f"socks5://127.0.0.1:{port}")
        installed = bootstrap.download_uv()
        assert installed is not None
        assert installed.read_bytes() == b"synthetic-uv-executable"
        assert [path for path, _ in requests] == list(responses)


def test_command_line_proxy_survives_config_load_and_is_visible_to_settings(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    previous = 'PROXY_PROTOCOL=http\nPROXY_HOST=127.0.0.1\nPROXY_PORT=7890\n'
    path.write_text(previous, encoding="utf-8")
    monkeypatch.setattr(config, "PRIVATE_ENV_PATH", path)
    bootstrap.apply_proxy("socks5://127.0.0.1:1080")
    config.load_config()
    assert config.network_proxy_url_from_env() == "socks5://127.0.0.1:1080"
    assert config.os.environ["HTTPS_PROXY"] == "socks5://127.0.0.1:1080"
    assert path.read_text(encoding="utf-8") == previous


@pytest.mark.parametrize("url", ["socks4://localhost:1080", "http://localhost:0", "http://localhost:99999", "http://localhost:bad", "http://localhost/path"])
def test_invalid_command_line_proxy_does_not_change_the_environment(url):
    before = dict(bootstrap.os.environ)
    with pytest.raises(ValueError):
        bootstrap.apply_proxy(url)
    assert dict(bootstrap.os.environ) == before
