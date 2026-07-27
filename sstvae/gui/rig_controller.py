"""Rig control for the GUI, kept off the GUI thread.

Every rigctld operation is blocking socket I/O with a timeout, so none of
it may happen on the thread that paints the window. A rigctld that is
missing or wedged costs up to the socket timeout on connect plus another
two on the send/recv retry — several seconds per poll, every poll, which
made the application unusable rather than merely un-rig-controlled.

Two decisions worth keeping:

**Polling runs on its own thread and backs off.** A rig that isn't
answering is usually not going to start answering in five seconds, so the
interval doubles on each consecutive failure up to `MAX_BACKOFF_S`. That
keeps a misconfigured setup from generating a permanent stream of
timeouts, and makes recovery cost at most one wasted interval.

**PTT gets its own connection.** rigctld is perfectly happy with several
clients, and a separate socket means a transmission can never wait behind
a frequency poll that is mid-timeout — the poll holds its own client's
lock, not one PTT needs. Keying is the one operation with a real deadline
(`ptt_lead_s`), so it does not queue behind status chatter.
"""

import threading

from PySide6.QtCore import QObject, Signal

from ..rig import RigctldClient, RigError, spawn_rigctld

MAX_BACKOFF_S = 60.0


def _reap(thread, *clients) -> None:
    """Wait out a stopped worker and close its sockets, off the GUI thread."""
    thread.join(timeout=10.0)
    for client in clients:
        if client is not None:
            client.close()


def next_interval(current: float, base: float, ok: bool) -> float:
    """How long to wait before the next poll.

    Success returns to the configured rate; failure doubles the wait, up
    to `MAX_BACKOFF_S`. A rig that isn't answering rarely starts within
    one interval, and retrying at full rate turns a misconfigured setup
    into a permanent stream of multi-second timeouts.
    """
    return base if ok else min(current * 2, MAX_BACKOFF_S)


class RigController(QObject):
    """Owns the rig connection and a polling thread.

    The GUI reads `current_frequency_hz` (a cached value, never a
    request) and listens to the signals; it never calls into rigctld.
    """

    frequencyChanged = Signal(object)  # float, or None when unknown
    statusChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._poll_client: RigctldClient | None = None
        self._ptt_client: RigctldClient | None = None
        self._process = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._frequency: float | None = None
        self._interval = 5.0
        self._last_status = ""

    # --- state the GUI may read -----------------------------------------
    @property
    def current_frequency_hz(self) -> float | None:
        return self._frequency

    def ptt(self):
        """The object `TxEngine` keys, or None when rig control is off.

        Deliberately a different client from the polling one; see the
        module docstring.
        """
        return self._ptt_client

    # --- lifecycle --------------------------------------------------------
    def apply_config(self, rig_cfg) -> None:
        """(Re)start from a `settings.RigConfig`. Safe to call repeatedly."""
        self.stop()
        if not rig_cfg.enabled:
            self._emit_status("Rig control off")
            return

        if rig_cfg.spawn_local:
            try:
                self._process = spawn_rigctld(
                    rig_cfg.model, rig_cfg.device, rig_cfg.baud, rig_cfg.port
                )
            except RigError as e:
                self._emit_status(str(e).splitlines()[0])
                return

        self._poll_client = RigctldClient(rig_cfg.host, rig_cfg.port)
        self._ptt_client = RigctldClient(rig_cfg.host, rig_cfg.port)
        self._interval = max(1.0, rig_cfg.poll_interval_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(self._interval, self._stop, self._poll_client),
            daemon=True, name="sstvae-rig",
        )
        self._emit_status("Rig: connecting...")
        self._thread.start()

    def stop(self) -> None:
        """Tear down the connection. Returns immediately.

        Called from the GUI thread (closing the window, or applying new
        settings), so it must not wait for anything. Joining the worker
        would inherit its in-flight socket timeout, and `close()` would
        block on the lock that worker is holding -- either way the window
        freezes for seconds against a wedged rigctld, which is the bug
        this whole class exists to avoid.
        """
        self._stop.set()
        self._resume.set()  # so a paused worker can notice and exit
        thread, self._thread = self._thread, None
        poll, ptt = self._poll_client, self._ptt_client
        self._poll_client = self._ptt_client = None
        process, self._process = self._process, None
        self._frequency = None

        for client in (poll, ptt):
            if client is not None:
                client.interrupt()  # unblock a stuck recv without the lock
        if process is not None:
            process.terminate()
        if thread is not None:
            # Joining and closing happen on a throwaway thread so this
            # call stays instant; the worker exits once its current
            # request unwinds.
            threading.Thread(
                target=_reap, args=(thread, poll, ptt), daemon=True,
                name="sstvae-rig-reap",
            ).start()

    # --- transmit interlock ------------------------------------------------
    def pause(self) -> None:
        """Stop polling while transmitting: the answer is not interesting
        mid-over, and some rigs dislike CAT traffic while keyed."""
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    # --- worker -------------------------------------------------------------
    def _run(self, base_interval: float, stop: threading.Event,
             client: RigctldClient) -> None:
        """Poll until told to stop.

        Takes its stop event and client as arguments rather than reading
        them off `self`: `stop()` returns before this thread has finished
        unwinding, so by the time a request completes the controller may
        already have been reconfigured with a new client. Working from
        locals means a departing worker cannot publish a stale frequency
        or status over the new one's.
        """
        interval = base_interval
        while not stop.is_set():
            self._resume.wait()
            if stop.is_set():
                break
            try:
                freq = client.get_frequency_hz()
            except RigError as e:
                ok, value, status = False, None, str(e).splitlines()[0]
            except Exception as e:  # never let the thread die silently
                ok, value, status = False, None, f"rig: {e}"
            else:
                ok, value, status = True, freq, f"Rig: {freq / 1e6:.4f} MHz"

            if stop.is_set():
                break  # superseded while the request was in flight
            self._frequency = value
            self.frequencyChanged.emit(value)
            self._emit_status(status)
            interval = next_interval(interval, base_interval, ok=ok)
            stop.wait(interval)

    def _emit_status(self, text: str) -> None:
        if text != self._last_status:
            self._last_status = text
            self.statusChanged.emit(text)
