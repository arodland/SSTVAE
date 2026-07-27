"""RigctldClient against a fake rigctld speaking the real line protocol."""

import socket
import subprocess
import threading

import pytest

from sstvae.rig import RigError, RigctldClient
from sstvae.rig import rigctld


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


# --- the model list -------------------------------------------------------

# Real `rigctld -l` output, trimmed to the rows that break naive parsing:
# single spaces inside Mfg and Model, and one Model ("Digital World
# Traveller") that fills its column exactly, leaving a *single* space
# before the Version field.
SAMPLE_LIST = """\
 Rig #  Mfg                    Model                   Version         Status      Macro
     1  Hamlib                 Dummy                   20240709.0      Stable      RIG_MODEL_DUMMY
     2  Hamlib                 NET rigctl              20250211.0      Stable      RIG_MODEL_NETRIGCTL
     4  FLRig                  FLRig                   20260130.0      Stable      RIG_MODEL_FLRIG
    10  N2ADR James Ahlstrom   Quisk                   20230709.0      Stable      RIG_MODEL_QUISK
    11  GQRX                   GQRX                    20250718.2      Untested    RIG_MODEL_GQRX
  1001  Yaesu                  FT-847                  20230512.0      Stable      RIG_MODEL_FT847
 25003  Coding Technologies    Digital World Traveller 20200112.0      Beta        RIG_MODEL_DWT
 42001  Harris                 PRC-138                 1.0.6           Alpha       RIG_MODEL_PRC138
"""


def test_model_list_columns_survive_embedded_and_missing_spaces():
    models = {m.number: m for m in rigctld._parse_model_list(SAMPLE_LIST)}
    assert len(models) == 8

    # Multi-word manufacturer: splitting on whitespace runs would eat it.
    assert models[10].mfg == "N2ADR James Ahlstrom"
    assert models[10].model == "Quisk"
    # Multi-word model.
    assert models[2].model == "NET rigctl"
    # Model field flush against Version, separated by one space only.
    assert models[25003].mfg == "Coding Technologies"
    assert models[25003].model == "Digital World Traveller"
    assert models[25003].version == "20200112.0"
    assert models[25003].macro == "RIG_MODEL_DWT"


def test_model_label_marks_only_non_stable_backends():
    models = {m.number: m for m in rigctld._parse_model_list(SAMPLE_LIST)}
    assert models[1001].label() == "Yaesu FT-847 (1001)"
    assert models[42001].label() == "Harris PRC-138 (42001) [Alpha]"
    assert models[11].label() == "GQRX GQRX (11) [Untested]"


def test_unparseable_output_raises_rather_than_returning_nothing():
    with pytest.raises(RigError, match="no header row"):
        rigctld._parse_model_list("rigctld: command not found\n")
    with pytest.raises(RigError, match="no usable models"):
        rigctld._parse_model_list(SAMPLE_LIST.splitlines()[0] + "\n")


def test_list_models_floats_shared_backends_and_sorts_the_rest(monkeypatch):
    monkeypatch.setattr(rigctld.shutil, "which", lambda _: "/usr/bin/rigctld")
    monkeypatch.setattr(
        rigctld.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, SAMPLE_LIST, ""),
    )
    models = rigctld.list_models()
    # NET rigctl and FLRig first: what you pick when another program owns
    # the serial port.
    assert [m.number for m in models[:2]] == [2, 4]
    rest = [(m.mfg.lower(), m.model.lower()) for m in models[2:]]
    assert rest == sorted(rest)


def test_missing_hamlib_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(rigctld.shutil, "which", lambda _: None)
    with pytest.raises(RigError, match="Install Hamlib"):
        rigctld.list_models()


def test_failed_rigctld_reports_its_own_stderr(monkeypatch):
    monkeypatch.setattr(rigctld.shutil, "which", lambda _: "/usr/bin/rigctld")
    monkeypatch.setattr(
        rigctld.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "no such option\n"),
    )
    with pytest.raises(RigError, match="no such option"):
        rigctld.list_models()
