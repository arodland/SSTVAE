"""RigctldClient against a fake rigctld speaking the real line protocol."""

import socket
import threading

import pytest

from sstvae.rig import RigError, RigctldClient


class FakeRigctld:
    """Minimal rigctld: one connection at a time, canned replies."""

    def __init__(self, replies=None, freq=14340000.0):
        self.replies = replies or {}
        self.freq = freq
        self.commands = []
        self.ptt = False
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                buf = b""
                while not self._stop.is_set():
                    try:
                        chunk = conn.recv(4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        cmd = line.decode().strip()
                        self.commands.append(cmd)
                        try:
                            conn.sendall(self._reply(cmd).encode())
                        except OSError:
                            break

    def _reply(self, cmd: str) -> str:
        if cmd in self.replies:
            return self.replies[cmd]
        if cmd.startswith("T "):
            self.ptt = cmd.split()[1] == "1"
            return "RPRT 0\n"
        if cmd == "t":
            return f"{1 if self.ptt else 0}\n"
        if cmd == "f":
            return f"{self.freq:.0f}\n"
        if cmd == "m":
            return "USB\n2400\n"
        return "RPRT -1\n"

    def close(self):
        self._stop.set()
        self._sock.close()


@pytest.fixture
def server():
    s = FakeRigctld()
    yield s
    s.close()


def test_ptt_and_frequency_roundtrip(server):
    with RigctldClient(port=server.port) as rig:
        rig.set_ptt(True)
        assert server.ptt is True
        assert rig.get_ptt() is True
        rig.set_ptt(False)
        assert server.ptt is False
        assert rig.get_frequency_hz() == pytest.approx(14340000.0)
        assert rig.get_mode() == "USB"
    assert server.commands[:2] == ["T 1", "t"]


def test_hamlib_error_code_raises_but_keeps_the_connection(server):
    """A rig that refuses a command is not a dead link -- the next
    command must still work on the same socket rather than forcing a
    reconnect."""
    server.replies["T 1"] = "RPRT -9\n"
    with RigctldClient(port=server.port) as rig:
        with pytest.raises(RigError, match="error -9"):
            rig.set_ptt(True)
        assert rig.connected
        assert rig.get_frequency_hz() == pytest.approx(14340000.0)


def test_reconnects_after_the_daemon_restarts(server):
    rig = RigctldClient(port=server.port)
    rig.connect()
    assert rig.get_frequency_hz() == pytest.approx(14340000.0)

    # Drop the socket underneath the client, as a rigctld restart would.
    rig._sock.close()

    # The retry path should redial transparently.
    assert rig.get_frequency_hz() == pytest.approx(14340000.0)
    rig.close()


def test_refused_connection_is_a_clear_error():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing is listening there now
    rig = RigctldClient(port=port, timeout=0.5)
    with pytest.raises(RigError, match="cannot reach rigctld"):
        rig.get_frequency_hz()
