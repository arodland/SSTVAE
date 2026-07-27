"""Rig polling must never block the GUI thread.

The bug: `poll_frequency` ran on a QTimer on the GUI thread and did
blocking socket I/O. A rigctld that accepts connections but doesn't
answer costs the full socket timeout on the recv, twice over the retry —
several seconds of frozen window, every poll interval.

The nasty case is not "connection refused" (that fails instantly) but a
daemon that is *up and wedged*, so that is what these tests simulate.
"""

import socket
import threading
import time

import pytest

pytest.importorskip("PySide6")

from sstvae.gui.rig_controller import MAX_BACKOFF_S, RigController, next_interval  # noqa: E402
from sstvae.gui.settings import RigConfig  # noqa: E402

# Well under the client's 2 s socket timeout: if any of these operations
# actually waited on the network they would blow through this.
RESPONSIVE_S = 0.5


class BlackHoleServer:
    """Accepts connections and then says nothing, ever."""

    def __init__(self):
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self.port = self._sock.getsockname()[1]
        self._held = []
        self._stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self._held.append(conn)  # keep it open, never reply

    def close(self):
        self._stop.set()
        self._sock.close()
        for c in self._held:
            c.close()


@pytest.fixture
def black_hole():
    server = BlackHoleServer()
    yield server
    server.close()


@pytest.fixture
def controller():
    ctrl = RigController()
    yield ctrl
    ctrl.stop()


def config_for(port, **kw):
    cfg = RigConfig()
    cfg.enabled = True
    cfg.host = "127.0.0.1"
    cfg.port = port
    cfg.poll_interval_s = kw.pop("poll_interval_s", 1.0)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# --- the reported bug ----------------------------------------------------

def test_starting_against_a_wedged_rigctld_returns_immediately(controller, black_hole):
    start = time.perf_counter()
    controller.apply_config(config_for(black_hole.port))
    elapsed = time.perf_counter() - start
    assert elapsed < RESPONSIVE_S, (
        f"apply_config blocked for {elapsed:.2f}s -- rig I/O is back on the "
        "calling thread"
    )


def test_reading_the_frequency_never_touches_the_network(controller, black_hole):
    """The GUI reads this on every status refresh and to name saved
    images; it has to be a cached value, not a request."""
    controller.apply_config(config_for(black_hole.port))
    time.sleep(0.2)  # let the worker get stuck in its recv

    start = time.perf_counter()
    for _ in range(100):
        controller.current_frequency_hz
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05, f"100 reads took {elapsed:.2f}s"
    assert controller.current_frequency_hz is None


def test_stopping_while_a_poll_is_stuck_is_prompt(controller, black_hole):
    controller.apply_config(config_for(black_hole.port))
    time.sleep(0.2)

    start = time.perf_counter()
    controller.stop()
    elapsed = time.perf_counter() - start
    # Must not wait at all: joining the worker inherits its in-flight
    # socket timeout, and close() blocks on the lock that worker holds.
    # Both freeze the window, which is the bug this class exists to fix.
    assert elapsed < RESPONSIVE_S, f"stop() took {elapsed:.2f}s"


def test_reconfiguring_against_a_wedged_rigctld_is_instant(controller, black_hole):
    """Applying new settings calls stop() then starts again; from the
    GUI thread that is one Settings-dialog OK click."""
    controller.apply_config(config_for(black_hole.port))
    time.sleep(0.2)

    start = time.perf_counter()
    controller.apply_config(config_for(black_hole.port))
    elapsed = time.perf_counter() - start
    assert elapsed < RESPONSIVE_S, f"reconfigure took {elapsed:.2f}s"


def test_a_superseded_worker_does_not_publish_stale_state(controller, black_hole):
    """stop() returns before the old worker unwinds, so it must not
    overwrite the new configuration's frequency or status."""
    controller.apply_config(config_for(black_hole.port))
    time.sleep(0.2)
    controller.stop()

    seen = []
    controller.frequencyChanged.connect(seen.append)
    controller._frequency = 14_340_000.0  # stand in for a fresh reading
    time.sleep(2.5)  # long enough for the old worker's recv to time out

    assert controller.current_frequency_hz == 14_340_000.0, (
        "a departing worker clobbered the current value"
    )
    assert seen == []


def test_a_disabled_rig_starts_no_thread_and_no_sockets(controller):
    cfg = RigConfig()
    cfg.enabled = False
    controller.apply_config(cfg)
    assert controller.ptt() is None
    assert controller.current_frequency_hz is None


# --- PTT isolation --------------------------------------------------------

def test_ptt_has_its_own_connection(controller, black_hole):
    """Keying has a real deadline (ptt_lead_s). It must not be able to
    queue behind a frequency poll sitting in a socket timeout, so the two
    do not share a client -- and therefore do not share its lock."""
    controller.apply_config(config_for(black_hole.port))
    ptt = controller.ptt()
    assert ptt is not None
    assert ptt is not controller._poll_client


def test_ptt_is_not_blocked_by_a_stuck_poll(controller, black_hole):
    """Acquiring the PTT client's lock must not wait on the poll."""
    controller.apply_config(config_for(black_hole.port))
    time.sleep(0.3)  # worker is now blocked in recv, holding *its* lock

    ptt = controller.ptt()
    start = time.perf_counter()
    acquired = ptt._lock.acquire(timeout=0.5)
    elapsed = time.perf_counter() - start
    if acquired:
        ptt._lock.release()
    assert acquired, "PTT client's lock was held by the polling thread"
    assert elapsed < 0.1


# --- backoff --------------------------------------------------------------

def test_backoff_doubles_on_failure():
    assert next_interval(5.0, 5.0, ok=False) == 10.0
    assert next_interval(10.0, 5.0, ok=False) == 20.0


def test_backoff_is_capped():
    assert next_interval(MAX_BACKOFF_S, 5.0, ok=False) == MAX_BACKOFF_S
    assert next_interval(1e6, 5.0, ok=False) == MAX_BACKOFF_S


def test_success_returns_to_the_configured_rate():
    assert next_interval(MAX_BACKOFF_S, 5.0, ok=True) == 5.0


def test_a_wedged_rig_does_not_poll_at_full_rate_forever(controller, black_hole):
    """End to end: after a few failures the worker should be waiting far
    longer than the configured interval rather than hammering."""
    statuses = []
    controller.statusChanged.connect(statuses.append)
    controller.apply_config(config_for(black_hole.port, poll_interval_s=1.0))
    # Two socket timeouts (~2 s each) plus backoff waits; in that window a
    # non-backing-off poller would have made many more attempts.
    time.sleep(6.0)
    controller.stop()
    # "connecting..." plus at most a couple of distinct failure messages;
    # _emit_status suppresses repeats, so this mostly proves the worker
    # did not die and did not spin.
    assert len(statuses) <= 4, statuses
