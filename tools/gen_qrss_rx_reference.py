#!/usr/bin/env python3
"""Reference summaries of the Python QRSSTVAE receiver, for the C++ port's
statistical acceptance (cpp-port-plan.md, stage 2: "Statistical
acceptance" and "Staleness pin").

    python tools/gen_qrss_rx_reference.py --list
    python tools/gen_qrss_rx_reference.py --scenario ce-short-12
    python tools/gen_qrss_rx_reference.py --all
    python tools/gen_qrss_rx_reference.py --all --check     # diff, write nothing

A scenario is a seeded simulation of one slot (`sstvae.qrss.channel`, the
CE and L paths), received by the PYTHON receiver
(`receiver.receive_slot`, budget_s=None so nothing depends on the
machine's speed). What came out is written as a summary to
native/tests/golden_qrss/rx_reference/<scenario>.json: the scenario's
parameters, a fingerprint of the capture, the signals in it, and per pass
the waveform, frame, header fields (or null), the callsign read, f_hz,
tau0 in CH samples, the latent SNR against the simulator's truth latents
(dB; null for a pass that is none of the signals), Z_ref and mean W.

tests/test_qrss_native_acceptance.py simulates the same captures (it
imports the scenario code from here, never a copy), receives them with
the C++ (`rx_api.hpp`'s receive_slot) and compares with these summaries
to the plan's bars (`compare`): the same pass count, waveform and frame;
header and callsign read exactly; f_hz within 0.01 Hz, tau0 within 0.5 CH
samples, latent SNR within 0.1 dB; and, beyond the plan, Z_ref within 1%
and mean W within 0.1 dB (both agree to about 1e-7 measured).

native/tests/golden_qrss/rx_pins.json pins what the references depend on:
the SHA-256 of every Python source the C++ receiver mirrors or the
simulation runs (`PINNED`), and of the scenario definitions themselves
(the source between the "scenarios: begin/end" markers below). The
acceptance file's pin test fails with "Python receiver changed; regenerate
references" when one moves, and `import_closure` keeps `PINNED` honest:
every sstvae module the receiver, render and the simulator import
(function-level imports and importlib.import_module included) is pinned
or listed in `NOT_PINNED` with its reason.

A tool, never a test: each scenario takes seconds to a minute of Python
(its time is printed). Run it when the pin test trips, or after changing
a scenario; --check re-runs and reports every difference from the
committed summaries (exit 1 if any lies past the bars, or a discrete
field differs; floating-point noise inside the bars is listed but passes).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

OUT_DIR = ROOT / "native" / "tests" / "golden_qrss" / "rx_reference"
PINS = ROOT / "native" / "tests" / "golden_qrss" / "rx_pins.json"
REGENERATE = "python tools/gen_qrss_rx_reference.py --all"

# Every Python source the C++ receiver mirrors or the references run, by
# path from the repository root: the receiver (receive_slot and all under
# it), render (decoder_planes, mirrored in C++) with the picture and air-map
# modules under it, the transmit side the simulation assembles with, the rx
# config sources (tools/gen_qrss_rx_config.py's modules, live.py among
# them), the simulator (channel.py: its CE and L paths), and the SSTVAE
# modules they import (config, dsp, beacon's CRC and alphabet, framing's air
# maps, golay under both). Three of them feed no reference and are pinned
# because stage 2's C++ mirrors them: live.py (StreamFE, frontend.cpp's
# streaming front end, and the `live` constants rx_config.hpp generates
# from it), wavio.py (read_wav, wavread.cpp) and qrss_receive.py (the
# `sstvae-qrss receive` CLI and its capture loading, receive.cpp).
PINNED = (
    "sstvae/qrss/acquire.py",
    "sstvae/qrss/beaconfile.py",
    "sstvae/qrss/ce.py",
    "sstvae/qrss/channel.py",
    "sstvae/qrss/constants.py",
    "sstvae/qrss/cwid.py",
    "sstvae/qrss/demod.py",
    "sstvae/qrss/frame.py",
    "sstvae/qrss/frontend.py",
    "sstvae/qrss/header.py",
    "sstvae/qrss/ldemod.py",
    "sstvae/qrss/lframe.py",
    "sstvae/qrss/ltrack.py",
    "sstvae/qrss/lwave.py",
    "sstvae/qrss/morse.py",
    "sstvae/qrss/picture.py",
    "sstvae/qrss/polar.py",
    "sstvae/qrss/precoder.py",
    "sstvae/qrss/receiver.py",
    "sstvae/qrss/render.py",
    "sstvae/qrss/rules.py",
    "sstvae/qrss/sequences.py",
    "sstvae/qrss/track.py",
    "sstvae/qrss/tx.py",
    "sstvae/qrss/types.py",
    "sstvae/qrss/live.py",
    "sstvae/config.py",
    "sstvae/wavio.py",
    "sstvae/modem/beacon.py",
    "sstvae/modem/dsp.py",
    "sstvae/modem/framing.py",
    "sstvae/modem/golay.py",
    "qrss_receive.py",
)

# Imported by the receiver, render or the simulator and deliberately not
# pinned, each with its reason. `import_closure` does not follow them.
NOT_PINNED = {
    "sstvae/__init__.py": "package marker",
    "sstvae/qrss/__init__.py": "package marker",
    "sstvae/modem/__init__.py": ("SSTVAE's modem package, which imports SSTVAE's Modem; QRSS uses "
                                 "dsp, beacon, framing and golay (pinned) and never the Modem"),
    "sstvae/hfchannel.py": "channel.simulate_audio only; the scenarios use channel.simulate",
    "sstvae/codec.py": "render.render only; the C++ mirrors decoder_planes, which needs no codec",
    "sstvae/checkpoint.py": "picture.codec_id_from_onnx only (the codec glue); no receiver reads a model",
}

ROOTS = ("sstvae/qrss/receiver.py", "sstvae/qrss/channel.py", "sstvae/qrss/render.py")

BEGIN = "# --- scenarios: begin"
END = "# --- scenarios: end"


# --- scenarios: begin ------------------------------------------------------------------------
# Everything from here to the end marker is hashed into rx_pins.json
# ("scenarios"): the table, how a scenario becomes a capture and its truth,
# and how a pass becomes a summary. Edit it and the pin test asks for the
# references to be regenerated.

Q = 1954752                          # the canonical slot, 2025-10-01T00:00Z (tests' Q_TEST)
CALL = "K1ABC"
HEADER = dict(callsign=CALL, grid="FN42", picture_id=0x12345678, mode=0, segment=0,
              codec_id=0xD1D8)
WAVEFORMS = {"CE": 0, "L": 1}

# the plan's bars, and two of this tool's own (Z_ref, mean W)
BARS = {"f_hz": 0.01, "tau0_ch": 0.5, "latent_snr_db": 0.1, "z_ref_rel": 0.01,
        "mean_w_db": 0.1}
# the capture's fingerprint: what the test's simulation must reproduce
CAPTURE_POWER_REL = 1e-6
CAPTURE_PROBE_REL = 1e-5


@dataclass(frozen=True)
class Nb:
    """A neighbour (`channel.Neighbour`): its named frequency df_hz above the
    wanted one's, rel_db relative to its power, its waveform's twin of the
    wanted frame, its own random latents, header and keying from `seed`."""
    df_hz: float
    rel_db: float
    waveform: str = "CE"
    seed: int = 7


@dataclass(frozen=True)
class Scenario:
    """One slot: the wanted signal (`frame` at `carrier_hz`, a CE carrier or
    an L pilot, latents and channel from `seed`), received with the frames
    `receive` (one per waveform; both is the joint receiver, what
    qrss_receive.py and the listener use by default)."""
    name: str
    frame: str
    snr_db: float
    receive: tuple
    seed: int = 1
    carrier_hz: float = 1500.0
    lead_in_s: float = 0.0
    drift_hz_per_min: float = 0.0
    tx_ppm: float = 0.0
    rx_ppm: float = 0.0
    neighbours: tuple = ()
    what: str = ""


SCENARIOS = (
    Scenario("ce-tiny-6", "tiny", -6.0, ("tiny",), seed=2,
             what="CE TINY at -6 dB, CE alone"),
    Scenario("ce-tiny-12-joint", "tiny", -12.0, ("tiny", "l-tiny"), seed=1,
             what="CE TINY at -12 dB, received with the default joint frames (CE and L)"),
    Scenario("l-tiny-6", "l-tiny", -6.0, ("l-tiny",), seed=1,
             what="L l-tiny at -6 dB, L alone"),
    Scenario("l-tiny-12", "l-tiny", -12.0, ("l-tiny",), seed=3,
             what="L l-tiny at -12 dB, L alone"),
    Scenario("ce-short-12", "short", -12.0, ("short",), seed=1,
             what="CE SHORT at -12 dB: header and callsign"),
    Scenario("ce-short-20", "short", -20.0, ("short",), seed=4,
             what="CE SHORT at -20 dB"),
    Scenario("l-short-12-joint", "l-short", -12.0, ("short", "l-short"), seed=1,
             what="L l-short at -12 dB, received with the default joint frames (CE and L)"),
    Scenario("l-short-20", "l-short", -20.0, ("l-short",), seed=5,
             what="L l-short at -20 dB"),
    Scenario("ce-short-drift", "short", -12.0, ("short",), seed=6, drift_hz_per_min=1.0,
             what="CE SHORT at -12 dB drifting 1 Hz/min"),
    Scenario("ce-short-tx-ppm", "short", -12.0, ("short",), seed=7, tx_ppm=100.0,
             what="CE SHORT at -12 dB, the transmitter's clock 100 ppm fast"),
    Scenario("l-short-rx-ppm", "l-short", -12.0, ("l-short",), seed=8, rx_ppm=-100.0,
             what="L l-short at -12 dB, the receiver's clock 100 ppm slow"),
    Scenario("ce-l-neighbours", "short", -12.0, ("short", "l-short"), seed=9,
             neighbours=(Nb(60.0, 0.0, "L", 7),),
             what="CE SHORT at -12 dB and an L l-short as strong 60 Hz above, received jointly"),
    Scenario("ce-short-lead-in", "short", -12.0, ("short",), seed=10, lead_in_s=10.0,
             what="CE SHORT at -12 dB after a 10 s lead-in"),
)


def scenario(name: str) -> Scenario:
    for sc in SCENARIOS:
        if sc.name == name:
            return sc
    raise KeyError(f"no scenario {name!r}; one of {', '.join(s.name for s in SCENARIOS)}")


def latents(n: int, seed: int) -> np.ndarray:
    """float64[n], RMS exactly 1: Gaussian air-order latents
    (tests/qrss_helpers.unit_rms_latents's recipe)."""
    a = np.random.default_rng(seed).standard_normal(n)
    return a / np.sqrt(np.mean(a ** 2))


@dataclass
class Signal:
    """A transmission in the slot, what a pass of it should deliver."""
    frame: str
    waveform: int
    f_hz: float                      # its named frequency: CE carrier, L pilot
    truth: np.ndarray = field(repr=False)   # air-order latents, float64[n_data]


def _neighbour_truth(nb, nspec, q: int) -> np.ndarray:
    """A neighbour's air-order latents: `channel._neighbour_symbols` draws
    its header bits and data symbols from default_rng(seed) and sends the
    draws as precoded symbols, so the receiver unprecodes them. The draw is
    checked against the simulator's own assembly."""
    from sstvae.qrss import channel, frame, lframe
    from sstvae.qrss.constants import WAVEFORM_L
    from sstvae.qrss.precoder import unprecode

    rng = np.random.default_rng(nb.seed)
    hdr = rng.integers(0, 2, frame.N_HDR_BITS, dtype=np.uint8) if nspec.has_header else None
    x = rng.standard_normal(nspec.n_data)
    assemble = lframe.assemble if nspec.waveform == WAVEFORM_L else frame.assemble
    sym, _ = channel._neighbour_symbols(nb, nspec)
    if not np.array_equal(assemble(nspec, hdr, x), sym):
        raise AssertionError("channel._neighbour_symbols draws differently from this tool's copy")
    return unprecode(x, q)


def simulate(sc: Scenario):
    """(frontend.Capture, [Signal]) of scenario sc: the slot through
    `channel.simulate`, the wanted signal first."""
    from sstvae.qrss import channel, frame, frontend, header, lframe
    from sstvae.qrss.channel import ChannelConfig, Neighbour
    from sstvae.qrss.constants import WAVEFORM_L
    from sstvae.qrss.morse import keying_units
    from sstvae.qrss.precoder import precode

    spec = lframe.get_any(sc.frame)
    a = latents(spec.n_data, sc.seed)
    bits = header.encode(header.HeaderFields(**HEADER)) if spec.has_header else None
    assemble = lframe.assemble if spec.waveform == WAVEFORM_L else frame.assemble
    sym = assemble(spec, bits, precode(a, Q))
    nbs = tuple(Neighbour(df_hz=n.df_hz, rel_db=n.rel_db, seed=n.seed,
                          waveform=WAVEFORMS[n.waveform]) for n in sc.neighbours)
    cfg = ChannelConfig(snr_db=sc.snr_db, seed=sc.seed, drift_hz_per_min=sc.drift_hz_per_min,
                        tx_ppm=sc.tx_ppm, rx_ppm=sc.rx_ppm, neighbours=nbs)
    sim = channel.simulate(sym, spec, Q, cfg, carrier_hz=sc.carrier_hz, lead_in_s=sc.lead_in_s,
                           keying=keying_units(CALL) if spec.n_win else None)
    signals = [Signal(spec.name, spec.waveform, sc.carrier_hz, a)]
    for nb in nbs:
        nspec = lframe.twin(spec, nb.waveform)
        signals.append(Signal(nspec.name, nspec.waveform, sc.carrier_hz + nb.df_hz,
                              _neighbour_truth(nb, nspec, Q)))
    return frontend.Capture(sim.fe, Q, sim.t0_index), signals


def receive_frames(sc: Scenario) -> tuple:
    """The frames sc is received with, as receiver.receive_slot takes them."""
    from sstvae.qrss import lframe
    return tuple(lframe.get_any(n) for n in sc.receive)


def capture_fingerprint(cap) -> dict:
    """What the test checks its own simulation against before comparing
    passes: the length, t0, mean power, and the samples at six places."""
    fe = np.asarray(cap.fe)
    n = len(fe)
    probe = [fe[(k * n) // 7] for k in range(1, 7)]
    return {"n": n, "t0_index": float(cap.t0_index),
            "power": float(np.mean(np.abs(fe.astype(np.complex128)) ** 2)),
            "probe": [[float(z.real), float(z.imag)] for z in probe]}


def latent_snr_db(est, truth) -> float:
    """10 log10(mean(truth^2) / mean((est - truth)^2)): tests/qrss_helpers'."""
    est = np.asarray(est, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    err = float(np.mean((est - truth) ** 2))
    return float(10 * np.log10(np.mean(truth ** 2) / err)) if err > 0 else math.inf


def _centre(waveform: int, f_hz: float) -> float:
    from sstvae.qrss import receiver
    return receiver.centre_hz(Signal("", waveform, f_hz, np.zeros(0)))


def signal_of(p, signals) -> int | None:
    """The index of the signal pass p received: the same waveform and frame,
    its band centre within receiver.MERGE_HZ of the signal's (the nearest);
    None for a pass that is none of them."""
    from sstvae.qrss import receiver
    c = receiver.centre_hz(p)
    best = None
    for i, s in enumerate(signals):
        if s.waveform != int(p.waveform) or s.frame != p.frame:
            continue
        d = abs(c - _centre(s.waveform, s.f_hz))
        if d < receiver.MERGE_HZ and (best is None or d < best[0]):
            best = (d, i)
    return None if best is None else best[1]


def summarise_pass(p, signals) -> dict:
    """What a pass delivered, as the references hold it."""
    from sstvae.qrss import receiver
    from sstvae.qrss.constants import WAVEFORM_NAMES

    i = signal_of(p, signals)
    snr = None
    if i is not None and len(signals[i].truth) == len(p.z):
        snr = latent_snr_db(p.z, signals[i].truth)
    cw = None
    if p.cw is not None:
        cw = {"text": p.cw.text, "keying": p.cw.keying, "agrees": p.cw.agrees}
    return {
        "waveform": WAVEFORM_NAMES[int(p.waveform)],
        "frame": p.frame,
        "signal": i,
        "header": None if p.header is None else asdict(p.header),
        "cw": cw,
        "f_hz": float(p.f_hz),
        "tau0_ch": float(p.timing.tau0),
        "latent_snr_db": snr,
        "z_ref": float(p.report.z_ref),
        "mean_w_db": receiver.mean_w_db(p),
    }


def summarise(sc: Scenario, cap, signals, passes) -> dict:
    """The reference summary of scenario sc: its parameters, the capture's
    fingerprint, the signals, and every pass in the receiver's order."""
    from sstvae.qrss.constants import WAVEFORM_NAMES
    return {
        "scenario": sc.name,
        "parameters": asdict(sc),
        "capture": capture_fingerprint(cap),
        "signals": [{"frame": s.frame, "waveform": WAVEFORM_NAMES[s.waveform], "f_hz": s.f_hz}
                    for s in signals],
        "n_passes": len(passes),
        "passes": [summarise_pass(p, signals) for p in passes],
    }


def _num_diff(name, ref, got, bar, rel=False):
    """(within, |difference| or relative difference) of two numbers."""
    if ref is None or got is None:
        return ref is None and got is None, None if ref is got else math.inf
    d = abs(got - ref)
    if rel:
        d = d / abs(ref) if ref != 0 else (0.0 if got == 0 else math.inf)
    return d <= bar, d


def compare(ref: dict, got: dict) -> tuple[list[str], list[str]]:
    """(problems, noise): the differences of a summary `got` from the
    reference `ref` past the bars (or in a discrete field), and those
    within them. Both are lists of readable lines."""
    problems, noise = [], []
    if json.dumps(ref["parameters"], sort_keys=True) != json.dumps(got["parameters"], sort_keys=True):
        problems.append(f"parameters differ: {ref['parameters']} against {got['parameters']}")
    rc, gc = ref["capture"], got["capture"]
    if rc["n"] != gc["n"] or rc["t0_index"] != gc["t0_index"]:
        problems.append(f"capture: {gc['n']} samples at t0 {gc['t0_index']}, the reference's "
                        f"{rc['n']} at {rc['t0_index']}")
    else:
        e = abs(gc["power"] - rc["power"]) / rc["power"]
        rms = math.sqrt(rc["power"])
        ep = max(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(gc["probe"], rc["probe"])) / rms
        if e > CAPTURE_POWER_REL or ep > CAPTURE_PROBE_REL:
            problems.append(f"capture: not the reference's simulation (power {e:.3g} relative, "
                            f"samples {ep:.3g} of the rms): numpy, scipy or the simulator moved")
        elif e or ep:
            noise.append(f"capture: power {e:.3g} relative, samples {ep:.3g} of the rms")
    if ref["signals"] != got["signals"]:
        problems.append(f"signals: {got['signals']} against {ref['signals']}")
    if ref["n_passes"] != got["n_passes"]:
        problems.append(f"{got['n_passes']} passes, the reference {ref['n_passes']}: "
                        f"{[p['f_hz'] for p in got['passes']]} against "
                        f"{[p['f_hz'] for p in ref['passes']]}")
        return problems, noise
    for i, (r, g) in enumerate(zip(ref["passes"], got["passes"])):
        tag = f"pass {i} ({r['waveform']} {r['f_hz']:.3f} Hz)"
        for k in ("waveform", "frame", "signal", "header", "cw"):
            if r[k] != g[k]:
                problems.append(f"{tag}: {k} {g[k]!r}, the reference {r[k]!r}")
        for k, bar, rel in (("f_hz", BARS["f_hz"], False), ("tau0_ch", BARS["tau0_ch"], False),
                            ("latent_snr_db", BARS["latent_snr_db"], False),
                            ("z_ref", BARS["z_ref_rel"], True),
                            ("mean_w_db", BARS["mean_w_db"], False)):
            ok, d = _num_diff(k, r[k], g[k], bar, rel)
            if not ok:
                problems.append(f"{tag}: {k} {g[k]!r}, the reference {r[k]!r} "
                                f"({'relative ' if rel else ''}difference {d:.3g} > {bar})")
            elif d:
                noise.append(f"{tag}: {k} differs by {d:.3g}{' relative' if rel else ''} "
                             f"(bar {bar})")
    return problems, noise

# --- scenarios: end --------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scenario_hash() -> str:
    """SHA-256 of this file's scenario section (between the markers)."""
    text = Path(__file__).read_text(encoding="utf-8")
    a = text.index("\n" + BEGIN)
    b = text.index("\n" + END)
    return _sha256(text[a:b].encode("utf-8"))


def source_hashes() -> dict:
    """SHA-256 of each pinned source with CRLF read as LF: Git for Windows
    checks .py files out with CRLF, which must not read as a change."""
    return {p: _sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")) for p in PINNED}


def pins() -> dict:
    """rx_pins.json's content: the pinned sources' hashes, the scenarios'."""
    return {
        "comment": ("SHA-256 of every Python source the C++ receiver mirrors or the receiver "
                    "references run, and of tools/gen_qrss_rx_reference.py's scenario section. "
                    f"Written by tools/gen_qrss_rx_reference.py; regenerate with: {REGENERATE}"),
        "sources": source_hashes(),
        "scenarios": scenario_hash(),
    }


def stale_pins(committed: dict) -> list[str]:
    """What moved since `committed` (rx_pins.json) was written, readable."""
    now = pins()
    out = []
    old = committed.get("sources", {})
    for p in sorted(set(old) | set(now["sources"])):
        if old.get(p) != now["sources"].get(p):
            what = "added" if p not in old else "removed" if p not in now["sources"] else "changed"
            out.append(f"{p} {what}")
    if committed.get("scenarios") != now["scenarios"]:
        out.append("the scenario definitions (tools/gen_qrss_rx_reference.py) changed")
    return out


def _module_path(name: str) -> str | None:
    """sstvae.x.y -> its source path from ROOT, '/'-separated as PINNED
    spells it on every OS (str() of a Windows path would use '\\', and
    then match nothing in PINNED), or None outside the repo."""
    if name != "sstvae" and not name.startswith("sstvae."):
        return None
    base = "/".join(name.split("."))
    for p in (f"{base}/__init__.py", f"{base}.py"):
        if (ROOT / p).is_file():
            return p
    return None


def _imports(path: str) -> set[str]:
    """Every sstvae module source `path` imports, anywhere in the file
    (function-level imports too), resolved to paths from ROOT."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    pkg = path[:-len(".py")].replace("/", ".")
    if pkg.endswith(".__init__"):
        pkg = pkg[:-len(".__init__")]
    else:
        pkg = pkg.rpartition(".")[0]
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = pkg.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module
            names.add(mod)
            names.update(f"{mod}.{a.name}" for a in node.names)   # from pkg import module
        elif (isinstance(node, ast.Call) and node.args
              and getattr(node.func, "attr", getattr(node.func, "id", None)) == "import_module"):
            # importlib.import_module("sstvae.x") or (f"{__package__}.x"), as
            # channel.py reaches ce and frontend
            a = node.args[0]
            if isinstance(a, ast.JoinedStr) and a.values and isinstance(a.values[-1], ast.Constant):
                a = a.values[-1]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                names.add(pkg + a.value if a.value.startswith(".") else a.value)
    out = set()
    for n in names:
        parts = n.split(".")
        for k in range(1, len(parts) + 1):                     # its packages' __init__ too
            p = _module_path(".".join(parts[:k]))
            if p is not None:
                out.add(p)
    return out


def import_closure() -> set[str]:
    """The sstvae sources the receiver, render and the simulator import,
    transitively and statically, not following `NOT_PINNED`."""
    seen, todo = set(), list(ROOTS)
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        if p in NOT_PINNED:
            continue
        todo.extend(_imports(p) - seen)
    return seen


def unaccounted_imports() -> list[str]:
    """Sources in the import closure neither pinned nor in NOT_PINNED."""
    return sorted(import_closure() - set(PINNED) - set(NOT_PINNED))


def reference_path(name: str) -> Path:
    return OUT_DIR / f"{name}.json"


def _dump(obj) -> str:
    """The committed JSON's text (written with LF line ends on every OS, so
    a regeneration on Windows is no diff)."""
    return json.dumps(obj, indent=1, allow_nan=False) + "\n"


def run(sc: Scenario) -> tuple[dict, float, float]:
    """(summary, simulation seconds, receive seconds) of scenario sc, by the
    Python receiver."""
    from sstvae.qrss import receiver

    t = time.perf_counter()
    cap, signals = simulate(sc)
    t_sim = time.perf_counter() - t
    t = time.perf_counter()
    passes = receiver.receive_slot(cap, receive_frames(sc), budget_s=None)
    t_rx = time.perf_counter() - t
    return summarise(sc, cap, signals, passes), t_sim, t_rx


def _clean_rules_env() -> None:
    """The references use the current rules: QRSS_OLD_* unset (rules.old)."""
    from sstvae.qrss import rules
    for name in (rules.EDGE_NOISE, rules.GATING):
        os.environ.pop(name, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--scenario", action="append", metavar="NAME",
                      help="this scenario (repeatable)")
    pick.add_argument("--all", action="store_true", help="every scenario")
    pick.add_argument("--list", action="store_true", help="list the scenarios")
    ap.add_argument("--check", action="store_true",
                    help="re-run and diff against the committed summaries; write nothing")
    args = ap.parse_args(argv)

    if args.list:
        for sc in SCENARIOS:
            print(f"{sc.name:18s} {sc.what}")
        return 0
    _clean_rules_env()
    names = [s.name for s in SCENARIOS] if args.all else args.scenario
    try:
        todo = [scenario(n) for n in names]
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2
    missing = unaccounted_imports()
    if missing:
        print("imported by the receiver or the simulator and neither in PINNED nor in "
              f"NOT_PINNED: {', '.join(missing)}", file=sys.stderr)
        return 1

    bad = 0
    for sc in todo:
        summary, t_sim, t_rx = run(sc)
        line = (f"{sc.name:18s} {summary['n_passes']} pass(es)  simulate {t_sim:5.1f} s  "
                f"receive (Python) {t_rx:5.1f} s")
        if args.check:
            path = reference_path(sc.name)
            if not path.exists():
                print(f"{line}  NO REFERENCE ({path.relative_to(ROOT)})")
                bad += 1
            else:
                problems, noise = compare(json.loads(path.read_text()), summary)
                print(f"{line}  {'DIFFERS' if problems else 'ok' if not noise else 'ok (noise)'}")
                for s in problems:
                    print(f"    {s}")
                for s in noise:
                    print(f"    (within the bars) {s}")
                bad += bool(problems)
        else:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            reference_path(sc.name).write_text(_dump(summary), encoding="utf-8", newline="\n")
            print(f"{line}  -> {reference_path(sc.name).relative_to(ROOT)}")
        for p in summary["passes"]:
            hdr = "header " + (p["header"]["callsign"] if p["header"] else "-")
            cw = "cw " + (repr(p["cw"]["text"]) if p["cw"] else "-")
            snr = "-" if p["latent_snr_db"] is None else f"{p['latent_snr_db']:.2f} dB"
            print(f"    {p['waveform']:2s} {p['frame']:8s} {p['f_hz']:9.3f} Hz  tau0 "
                  f"{p['tau0_ch']:9.2f}  Z_ref {p['z_ref']:7.1f}  mean W {p['mean_w_db']:6.2f} dB"
                  f"  latent SNR {snr}  {hdr}  {cw}")

    if args.check:
        stale = stale_pins(json.loads(PINS.read_text())) if PINS.exists() else ["no rx_pins.json"]
        if stale:
            print("rx_pins.json is stale: " + "; ".join(stale))
            bad += 1
        else:
            print("rx_pins.json is current")
        return 1 if bad else 0
    if args.all:
        PINS.write_text(_dump(pins()), encoding="utf-8", newline="\n")
        print(f"pins -> {PINS.relative_to(ROOT)}")
    else:
        print("pins not written (only --all writes them, so every reference is current)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
