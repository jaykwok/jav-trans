"""Local SOCKS5 peer and HTTPS origin; never forwards traffic off the machine."""

from contextlib import contextmanager
from pathlib import Path
import socket
import ssl
import threading


CERTIFICATE = Path(__file__).parent / "fixtures" / "proxy-test.pem"


def receive(sock, size):
    result = b""
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise EOFError("peer closed")
        result += chunk
    return result


@contextmanager
def socks_origin(responses, *, tls=False, credentials=None, reject=False, expect_tls_failure=False):
    errors, requests = [], []
    stopped = threading.Event()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.1)
    port = listener.getsockname()[1]

    def serve():
        while not stopped.is_set():
            try:
                sock, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                sock.settimeout(3)
                version, count = receive(sock, 2)
                assert version == 5
                methods = receive(sock, count)
                method = 2 if credentials else 0
                assert method in methods
                sock.sendall(bytes([5, method]))
                if credentials:
                    assert receive(sock, 1) == b"\x01"
                    username = receive(sock, receive(sock, 1)[0]).decode()
                    password = receive(sock, receive(sock, 1)[0]).decode()
                    if (username, password) != credentials:
                        sock.sendall(b"\x01\x01")
                        continue
                    sock.sendall(b"\x01\x00")
                assert receive(sock, 4) == b"\x05\x01\x00\x03"
                hostname = receive(sock, receive(sock, 1)[0])
                assert hostname == b"proxy-test.invalid"
                assert int.from_bytes(receive(sock, 2), "big") == (443 if tls else 80)
                sock.sendall(bytes([5, 5 if reject else 0, 0, 1, 127, 0, 0, 1, 0, 80]))
                if reject:
                    continue
                if tls:
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    context.load_cert_chain(CERTIFICATE)
                    try:
                        sock = context.wrap_socket(sock, server_side=True)
                    except ssl.SSLError:
                        if expect_tls_failure:
                            continue
                        raise
                with sock.makefile("rb") as stream:
                    request = stream.readline().decode("ascii")
                    path = request.split()[1]
                    headers = []
                    while line := stream.readline():
                        if line == b"\r\n":
                            break
                        headers.append(line)
                    requests.append((path, b"".join(headers)))
                    status, headers, body = responses[path]
                    sock.sendall(
                        f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n{headers}\r\n".encode()
                        + body
                    )
            except Exception as exc:
                errors.append(exc)
            finally:
                sock.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield port, requests
    finally:
        stopped.set()
        listener.close()
        thread.join(4)
        assert not thread.is_alive(), "SOCKS test peer did not stop"
        assert not errors, errors
