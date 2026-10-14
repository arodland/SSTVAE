"""Statistical acceptance of the C++ receiver (cpp-port-plan.md, stage 2:
"Statistical acceptance" and "Staleness pin").

The C++ receive (native/core/qrss/rx_api.hpp's receive_slot, through
sstvae_native.qrss.rx_api: the entry point an application calls, with no
Python behind it) on about a dozen seeded slots, against the Python
receiver's committed reference summaries
(native/tests/golden_qrss/rx_reference/, written by
tools/gen_qrss_rx_reference.py, whose scenario code this file imports
rather than copies). The release bar is the result -- the decoded picture
and header -- so what is compared is what a pass delivered, to the plan's
bars (`compare` in the tool): the same passes in the same order, waveform
and frame exactly; header and callsign read exactly; f_hz within 0.01 Hz,
tau0 within 0.5 CH samples, latent SNR against the simulator's truth
latents within 0.1 dB; and, the tool's own, Z_ref within 1% and mean W
within 0.1 dB. Measured (2026-10-14), the worst over every scenario: f_hz
2.5e-12 Hz, tau0 6.6e-12 CH samples, latent SNR 2.4e-10 dB, Z_ref 2.1e-8
relative, mean W 0 (w bit for bit) -- the stages' rounding, nine or more
orders of magnitude inside the bars, which are sized for another platform
or a later C++ change, not for this machine.

The scenarios (`--list` prints them): SHORT and TINY, CE and L, -6, -12
and -20 dB, a 1 Hz/min drift, a transmitter 100 ppm fast and a receiver
100 ppm slow, a CE and an L signal in one slot received jointly, a 10 s
lead-in, and a CE and an L signal each received with the default joint
frames (CE and L). Every decision they reach sits away from its
threshold (the tool's run prints Z_ref; the callsign reads' margins are
2.9 to 14 over READ_LLR_MIN where a call is read, under 0.2 where none
is, and Z_match at least 18 against Z_ACCEPT 6), so a rounding-level
difference cannot flip one. Each test first checks that its simulation
is the generator's (`capture` in the summary: length, t0, power, six
samples): a numpy, scipy or simulator change shows as that, not as a
receiver failure.

The staleness pin: rx_pins.json holds the SHA-256 of every Python source
the C++ receiver mirrors or the references run, and of the tool's scenario
section; `test_pins_are_current` fails with "Python receiver changed;
regenerate references" when one moves, and `test_pins_cover_the_imports`
keeps the list honest (every sstvae module the receiver, render and the
simulator import is pinned or excused in the tool's NOT_PINNED).

Times, C++ receive against the Python receive of the same capture (the
tool prints the latter), 4 shared CPUs: TINY 0.7-1.2 s against 1.0-1.7 s,
SHORT 1.5-2.0 s against 3.7-4.5 s, the joint SHORT neighbour slot 3.2 s
against 7.0 s. The file runs in about 25 s.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import pytest
from qrss_native import _rx, require

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "gen_qrss_rx_reference.py"
STALE = "Python receiver changed; regenerate references: python tools/gen_qrss_rx_reference.py --all"


def _load_tool():
    name = "gen_qrss_rx_reference"
    mod = sys.modules.get(name)
    if mod is not None and Path(mod.__file__).resolve() == TOOL:
        return mod
    spec = importlib.util.spec_from_file_location(name, TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses look the module up while the class is made
    spec.loader.exec_module(mod)
    return mod


G = _load_tool()
NAMES = [sc.name for sc in G.SCENARIOS]


@pytest.fixture(scope="module")
def rx_api(native):
    return require(native, "rx_api")


def _reference(name: str) -> dict:
    return json.loads(G.reference_path(name).read_text())


# --- the staleness pin --------------------------------------------------------------------------

def test_pins_are_current(native):
    """rx_pins.json against the sources and scenarios as they are now."""
    stale = G.stale_pins(json.loads(G.PINS.read_text()))
    assert not stale, f"{STALE} ({'; '.join(stale)})"


def test_pins_cover_the_imports(native):
    """Every sstvae source the receiver, render and the simulator import
    (statically, function-level imports too) is pinned or excused, and the
    pins file lists exactly the tool's PINNED."""
    assert not G.unaccounted_imports(), G.unaccounted_imports()
    assert sorted(json.loads(G.PINS.read_text())["sources"]) == sorted(G.PINNED)
    assert not set(G.PINNED) & set(G.NOT_PINNED)
    assert {"sstvae/qrss/receiver.py", "sstvae/qrss/track.py", "sstvae/qrss/ltrack.py",
            "sstvae/qrss/channel.py", "sstvae/qrss/frontend.py", "sstvae/qrss/acquire.py"} <= set(G.PINNED)


def test_stale_pins_name_what_moved():
    """The pin test's message says which source moved (or the scenarios),
    and CRLF line endings (a Windows checkout) are not a change."""
    now = G.pins()
    assert G.stale_pins(now) == []
    old = json.loads(json.dumps(now))
    old["sources"]["sstvae/qrss/track.py"] = "0" * 64
    old["sources"]["sstvae/qrss/gone.py"] = "0" * 64
    del old["sources"]["sstvae/qrss/ltrack.py"]
    old["scenarios"] = "0" * 64
    assert G.stale_pins(old) == [
        "sstvae/qrss/gone.py removed", "sstvae/qrss/ltrack.py added",
        "sstvae/qrss/track.py changed",
        "the scenario definitions (tools/gen_qrss_rx_reference.py) changed"]
    data = (ROOT / "sstvae/qrss/rules.py").read_bytes()
    assert b"\r\n" not in data
    assert G._sha256(data.replace(b"\n", b"\r\n").replace(b"\r\n", b"\n")) == \
        now["sources"]["sstvae/qrss/rules.py"]


def test_every_scenario_has_its_reference():
    """One summary per scenario, none left over from a renamed one, each
    naming its scenario with the parameters the tool has now."""
    on_disk = sorted(p.stem for p in G.OUT_DIR.glob("*.json"))
    assert on_disk == sorted(NAMES)
    assert len(NAMES) == len(set(NAMES))
    for sc in G.SCENARIOS:
        ref = _reference(sc.name)
        assert ref["scenario"] == sc.name
        assert json.dumps(ref["parameters"], sort_keys=True) == \
            json.dumps(json.loads(json.dumps(G.asdict(sc))), sort_keys=True)
        assert ref["n_passes"] == len(ref["passes"]) >= 1, sc.name


def test_the_scenarios_cover_the_plan():
    """What the plan asks the scenarios to hold, read from the table."""
    scs = G.SCENARIOS
    frames = {sc.frame for sc in scs}
    assert {"tiny", "short", "l-tiny", "l-short"} <= frames
    assert {-6.0, -12.0, -20.0} <= {sc.snr_db for sc in scs}
    assert any(sc.drift_hz_per_min == 1.0 for sc in scs)
    assert any(sc.tx_ppm == 100.0 for sc in scs) and any(sc.rx_ppm == -100.0 for sc in scs)
    assert any(sc.lead_in_s == 10.0 for sc in scs)
    joint = [sc for sc in scs if len(sc.receive) == 2]
    assert any(sc.neighbours and {sc.neighbours[0].waveform, "CE" if sc.frame in ("tiny", "short") else "L"}
               == {"CE", "L"} for sc in joint)
    single = [sc for sc in joint if not sc.neighbours]
    assert {sc.frame[:2] == "l-" for sc in single} == {True, False}     # a CE and an L signal
    # and the references hold what the names say: headers on SHORT, a
    # second pass beside the neighbour
    for sc in scs:
        ref = _reference(sc.name)
        if sc.frame in ("short", "l-short"):
            assert ref["passes"][0]["header"] is not None or len(ref["passes"]) > 1, sc.name
        if sc.neighbours:
            assert sorted(p["signal"] for p in ref["passes"]) == [0, 1], sc.name


# --- the comparison's bars ----------------------------------------------------------------------

def test_compare_holds_each_bar():
    """`compare` passes a difference inside each bar and fails one past it,
    and fails any change in a discrete field: the bars are the ones named
    in the module docstring, and they are enforced."""
    ref = _reference("ce-l-neighbours")
    assert G.compare(ref, json.loads(json.dumps(ref))) == ([], [])

    def moved(path, f):
        got = json.loads(json.dumps(ref))
        node = got
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = f(node[path[-1]])
        return G.compare(ref, got)

    for key, bar, rel in (("f_hz", 0.01, False), ("tau0_ch", 0.5, False),
                          ("latent_snr_db", 0.1, False), ("z_ref", 0.01, True),
                          ("mean_w_db", 0.1, False)):
        for i in (0, 1):
            step = bar * (abs(ref["passes"][i][key]) if rel else 1.0)
            problems, noise = moved(("passes", i, key), lambda v, s=step: v + 0.6 * s)
            assert problems == [] and len(noise) == 1, (key, i, problems, noise)
            problems, _ = moved(("passes", i, key), lambda v, s=step: v - 1.4 * s)
            assert len(problems) == 1 and key in problems[0], (key, i, problems)
    for path, f in ((("passes", 1, "header", "callsign"), lambda v: "K1ABD"),
                    (("passes", 1, "header", "picture_id"), lambda v: v + 1),
                    (("passes", 0, "cw", "text"), lambda v: None),
                    (("passes", 1, "cw", "agrees"), lambda v: False),
                    (("passes", 0, "frame"), lambda v: "l-tiny"),
                    (("passes", 0, "waveform"), lambda v: "CE"),
                    (("passes", 0, "signal"), lambda v: None),
                    (("passes", 1, "latent_snr_db"), lambda v: None),
                    (("capture", "power"), lambda v: v * (1 + 1e-5)),
                    (("capture", "n"), lambda v: v + 1),
                    (("parameters", "seed"), lambda v: v + 1)):
        problems, _ = moved(path, f)
        assert problems, path
    got = json.loads(json.dumps(ref))
    got["passes"].pop()
    got["n_passes"] -= 1
    assert G.compare(ref, got)[0]


# --- the scenarios --------------------------------------------------------------------------------

_PYTHON_RECEIVER = (("sstvae.qrss.receiver", ("receive_slot", "detect", "candidates", "receive_pass",
                                              "prepare", "channel_for")),
                    ("sstvae.qrss.acquire", ("acquire",)),
                    ("sstvae.qrss.track", ("track", "verify", "verify_track")),
                    ("sstvae.qrss.ltrack", ("track_l", "verify_l")),
                    ("sstvae.qrss.demod", ("extract",)),
                    ("sstvae.qrss.ldemod", ("extract_l",)))


@pytest.mark.parametrize("name", NAMES)
def test_scenario(rx_api, monkeypatch, name):
    """Scenario `name`: the generator's capture, received by the C++, gives
    the reference summary to the bars (module docstring). The Python
    receiver is booby-trapped for the call, so what is compared is the
    C++'s."""
    import importlib

    sc = G.scenario(name)
    ref = _reference(name)
    t = time.perf_counter()
    cap, signals = G.simulate(sc)
    t_sim = time.perf_counter() - t
    capture = G.capture_fingerprint(cap)
    problems, _ = G.compare(ref, dict(ref, capture=capture))
    assert not problems, (name, problems)       # the generator's own simulation, or stop here

    calls = []
    with monkeypatch.context() as mp:
        for modname, attrs in _PYTHON_RECEIVER:
            mod = importlib.import_module(modname)
            for attr in attrs:
                def trap(*a, _w=f"{modname}.{attr}", **k):
                    calls.append(_w)
                    raise AssertionError(f"the Python receiver was called: {_w}")
                mp.setattr(mod, attr, trap)
        t = time.perf_counter()
        raw = rx_api.receive_slot(cap, sc.receive, budget_s=None)
        t_cpp = time.perf_counter() - t
    assert calls == []
    passes = _rx.to_ref(raw)
    got = G.summarise(sc, cap, signals, passes)
    problems, noise = G.compare(ref, got)
    print(f"{name}: simulate {t_sim:.1f} s, C++ receive {t_cpp:.2f} s, {len(passes)} pass(es)")
    for line in noise:
        print(f"    (within the bars) {line}")
    assert not problems, "\n".join([name] + problems)
    for p, r in zip(got["passes"], ref["passes"]):
        assert p["latent_snr_db"] is not None and math.isfinite(p["latent_snr_db"]), (name, r)
