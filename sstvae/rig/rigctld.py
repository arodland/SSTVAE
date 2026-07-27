"""Talk to Hamlib's `rigctld` over TCP.

Why the daemon rather than Hamlib's Python bindings: the bindings are a
SWIG extension installed by the distro into the system site-packages,
which a virtualenv cannot see. Requiring them would mean telling every
user to build Hamlib or recreate their venv with --system-site-packages.
`rigctld` ships with every Hamlib install, speaks a stable line
protocol, and can already be running and shared with other software
(WSJT-X, fldigi) instead of fighting them for the serial port.

The protocol used here is the short-command form:

    T 1\\n      set PTT on          -> "RPRT 0"
    T 0\\n      set PTT off         -> "RPRT 0"
    t\\n        get PTT             -> "0" or "1"
    f\\n        get frequency (Hz)  -> "14340000"

A non-zero RPRT is Hamlib's error code and is raised as `RigError`.
"""

import shutil
import socket
import subprocess
import threading

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4532


class RigError(RuntimeError):
    """Rig control failed: not connected, refused, or Hamlib returned an
    error code."""


class RigctldClient:
    """Small, reconnecting, thread-safe rigctld client.

    Thread-safety matters because PTT is keyed from the transmit worker
    while a UI timer is polling the frequency; one socket with one lock
    keeps their request/response pairs from interleaving.

    Reconnection is lazy: any command on a dead socket transparently
    redials once before giving up, so restarting rigctld doesn't require
    restarting the application.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    # --- connection ---------------------------------------------------
    def connect(self) -> None:
        with self._lock:
            self._connect_locked()

    def _connect_locked(self) -> None:
        self._close_locked()
        try:
            self._sock = socket.create_connection((self.host, self.port), self.timeout)
            self._sock.settimeout(self.timeout)
        except OSError as e:
            self._sock = None
            raise RigError(
                f"cannot reach rigctld at {self.host}:{self.port} ({e}). "
                "Is it running? Try: rigctld -m <model> -r <device>"
            ) from e

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def interrupt(self) -> None:
        """Abort whatever request is in flight, from another thread.

        Deliberately does *not* take the lock: the point is to break a
        thread that is blocked in `recv` and holding it. Shutting the
        socket down makes that call return at once instead of waiting out
        the timeout, which is what lets a caller tear this client down
        without inheriting its remaining wait.
        """
        sock = self._sock
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # already closed, or never connected

    def _close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # --- protocol -----------------------------------------------------
    def _command(self, cmd: str) -> str:
        """Send one command, return its (possibly empty) response body.

        A dead socket is retried once (rigctld may have been restarted
        since the last command); a Hamlib error code is *not* retried and
        leaves the connection up, since the link is fine and only the
        operation failed.
        """
        with self._lock:
            last: OSError | None = None
            for _ in range(2):
                if self._sock is None:
                    self._connect_locked()  # raises RigError if refused
                try:
                    self._sock.sendall((cmd + "\n").encode())
                    return self._read_reply_locked()
                except RigError:
                    raise  # rig said no; connection is still good
                except OSError as e:
                    last = e
                    self._close_locked()
            raise RigError(f"rigctld command {cmd!r} failed: {last}") from last

    def _read_reply_locked(self) -> str:
        """Read one line. `RPRT n` is a status line: 0 means success with
        no value, non-zero is an error. Anything else is the value."""
        buf = b""
        while b"\n" not in buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise OSError("rigctld closed the connection")
            buf += chunk
        line = buf.split(b"\n", 1)[0].decode(errors="replace").strip()
        if line.startswith("RPRT "):
            code = int(line[5:])
            if code != 0:
                raise RigError(f"rigctld returned error {code} (Hamlib RIG_E code)")
            return ""
        return line

    # --- operations ---------------------------------------------------
    def set_ptt(self, on: bool) -> None:
        self._command(f"T {1 if on else 0}")

    def get_ptt(self) -> bool:
        return self._command("t").strip() == "1"

    def get_frequency_hz(self) -> float:
        raw = self._command("f").strip()
        try:
            return float(raw)
        except ValueError as e:
            raise RigError(f"unexpected frequency reply {raw!r}") from e

    def get_mode(self) -> str:
        """Radio mode ("USB", "LSB", ...). `m` replies with mode then
        passband on separate lines; only the first is wanted."""
        return self._command("m").strip()

    def describe(self) -> str:
        """One-line status for a UI, never raising."""
        try:
            f = self.get_frequency_hz()
        except RigError as e:
            return f"rig: {e.args[0].splitlines()[0]}"
        return f"rig: {f / 1e6:.4f} MHz"


def spawn_rigctld(model: str, device: str, baud: int | None = None,
                  port: int = DEFAULT_PORT) -> subprocess.Popen:
    """Start a private rigctld. Only for users who aren't already running
    one; sharing an existing daemon is preferable, since two processes
    cannot both hold the serial port.

    The caller owns the returned process and must terminate it.
    """
    exe = shutil.which("rigctld")
    if exe is None:
        raise RigError(
            "rigctld not found on PATH. Install Hamlib "
            "(Debian/Ubuntu: apt install libhamlib-utils; Arch: hamlib; "
            "macOS: brew install hamlib)."
        )
    argv = [exe, "-m", str(model), "-r", device, "-t", str(port)]
    if baud:
        argv += ["-s", str(baud)]
    try:
        return subprocess.Popen(
            argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except OSError as e:
        raise RigError(f"could not start rigctld: {e}") from e
