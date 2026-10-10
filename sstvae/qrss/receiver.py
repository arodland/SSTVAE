"""The single-pass receiver: `receive_pass`, `receive_slot`, `receive_wav`
(design 6.1, 6.8; WP6).

Not a format module.

    8 kHz audio --audio_to_fe--> FE 4 kHz --blank, normalise--> FE'
    FE' --A1 lines, A2 preamble, A3 track-before-detect--> candidates
    candidate --channelise--> CH 250 Hz --gate V--> Detection
    Detection --round A: track, extract, header, callsign windows-->
              --round B (header decoded): header and keying known, re-track,
                re-extract--> PassResult

Round A tracks on the preamble, references, spare, lead-in, the plain
units of the callsign windows and the carrier under the unknown data.
Round B, after a CRC-valid header, adds the 2,474 header symbols as
known, and the callsign windows' Morse units when the keying matches the
header's callsign (Z > 6); both rounds' timing comes from the known
symbols' likelihood given the gain track. `receive_slot` runs the
acquisition over a stored slot, passes every candidate through gate V,
merges detections within 5 Hz, drops symbol-rate aliases and ghosts of
a stronger one, and receives each.

**Ghosts.** A strong CE signal passes gate V a second time about 12 Hz
either side of its carrier, at nearly its own timing: the reference
symbols recur every 16 positions (callsign windows are 24 x 16), so a
frequency error of k/(16 T) = 2.06 k Hz turns every reference by the
same phase and their correlation stays coherent; measured, k = 6
(12.4 Hz), with tau0 within 1.6 T of the true one and Z_ref up to 0.54
of the real pass's (FULL, -12 dB). A detection within GHOST_HZ of a
stronger one and within GHOST_T symbols of its timing is therefore taken
to be that transmission, before and again after tracking. The price is
a second station that is both under 20 Hz from a stronger one (the CE
plan spaces stations 50 Hz apart, and at 20 Hz their spectra overlap
anyway) and slot-aligned to within 3 symbols: it is not received.

A second kind got past that rule (integration review, 2026-10-09): 2 of
15 FULL passes at about -11 dB also returned a pass 8-14 Hz away with
its timing 6-10 symbols off, Z_ref 6.1-6.4 (at the gate, against 104
and 125 for the real pass), kappa at its 2.0 cap and flagged SUSPECT,
with an all-zero callsign read. After tracking, a SUSPECT pass within
GHOST_HZ of a sound pass and under GHOST_Z_RATIO of its Z_ref is
dropped too: whatever it is, its data are not usable, and a genuine
second station that close is already given up above.

Everything time-like comes from the `FrameSpec`: nothing here assumes a
slot length.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import acquire, cwid, demod, frontend, header, track
from .constants import CH_FS, FE_FS, LEAD_IN_MAX_S, SPAN, T_SYM, WAVEFORM_CE, Z_ACCEPT
from .frame import FULL, FrameSpec
from .morse import check_callsign, keying_units
from .types import CwIdResult, Detection, EmPrior, FreqPath, PassResult, Timing, pass_uid

MERGE_HZ = 5.0
GHOST_HZ = 20.0            # a weaker detection this close to a stronger one...
GHOST_T = 3.0              # ...with tau0 within this many symbols of it is a ghost
GHOST_Z_RATIO = 0.1        # a SUSPECT pass this much weaker than a sound one within GHOST_HZ
ALIAS_HZ = 0.5
ALIAS_RATIO = 3.0
MARGIN_S = 3.0


@dataclass
class Prepared:
    """A capture after the blanker and inverse AGC (FE'), with the keep indicator."""
    fe: np.ndarray
    keep: np.ndarray | None
    q: int
    t0_index: float
    fs: int = FE_FS


def prepare(cap) -> Prepared:
    """Blank and normalise a raw `frontend.Capture` (a `Prepared` passes through)."""
    if isinstance(cap, Prepared):
        return cap
    if cap.fs == CH_FS:
        return Prepared(fe=np.asarray(cap.fe), keep=None, q=cap.q, t0_index=cap.t0_index,
                        fs=CH_FS)
    fe, keep = frontend.blank(cap.fe, fs=cap.fs)
    fe, _ = frontend.normalise(fe, keep, fs=cap.fs)
    return Prepared(fe=fe, keep=keep, q=cap.q, t0_index=cap.t0_index, fs=cap.fs)


def channel_for(prep: Prepared, spec: FrameSpec, det: Detection, exclude_hz=(),
                f_mix_hz=None) -> track.ChanCapture:
    """The detection's 250 Hz channel over its whole keyed span plus margins.

    Mixed at the path's frequency at the middle of the frame, so a
    drifting carrier stays near 0 Hz in CH. A capture already at the CH
    rate (EM on a stored pass) is used as it is, mixed at `f_mix_hz`
    (Hz inside FE; default the detection's frequency on the mix grid).
    """
    if prep.fs == CH_FS:
        fm = (frontend.quantise_mix(frontend.fe_hz(det.f_hz)) if f_mix_hz is None
              else Fraction(f_mix_hz))
        return track.ChanCapture(ch=np.asarray(prep.fe), fs=CH_FS, t0_index=prep.t0_index,
                                 f_mix=fm, keep=None, exclude_hz=tuple(exclude_hz))
    fn = track.freq_path_spline(det.path)
    dur = spec.keyed_end_pos * T_SYM
    slip = 260e-6 * dur
    lo = -(LEAD_IN_MAX_S + SPAN * T_SYM) - MARGIN_S - slip
    if det.timing is not None:
        lo += det.timing.tau0 / CH_FS
    hi = dur + MARGIN_S + slip + (det.timing.tau0 / CH_FS if det.timing is not None else 2.5)
    return track.make_chan(prep.fe, prep.keep, prep.t0_index, prep.fs,
                           float(fn(dur / 2)), lo, hi, exclude_hz)


def _psi_frame(tr: track.TrackResult) -> np.ndarray:
    g = tr.gi(np.arange(tr.spec.n_pos))
    return tr.psi[g].astype(np.float32)


def _cw_result(tr: track.TrackResult, spec: FrameSpec, hdr) -> tuple[CwIdResult | None, tuple]:
    """(CwIdResult, cw_known for round B) from round A's tracker."""
    if spec.n_win == 0:
        return None, None
    call = hdr.callsign.rstrip(" ") if hdr is not None else None
    if call is not None:
        try:
            check_callsign(call)
        except ValueError:
            call = None
    kt = cwid.keying_type(None, tr, spec, call)
    ook = kt == "ook"
    soft = cwid.window_soft(None, CH_FS, tr, spec, ook=ook)
    text = cwid.read(cwid.window_llr(None, CH_FS, tr, spec, ook=ook), ook)
    if call is not None:
        z = cwid.match(soft, call)
        # a confident read of another call disagrees, whatever the match
        agrees = (text == call) or (text is None and z > Z_ACCEPT)
        known = (keying_units(call), ook) if z > Z_ACCEPT else None
    else:
        z = cwid.match(soft, text) if text else 0.0
        agrees = None
        known = None
    return CwIdResult(text=text, z_match=float(z), keying=kt, agrees=agrees), known


def _erase(tr: track.TrackResult, spans) -> None:
    """Erase (psi = 0) tr's symbols within T of any span (a, b), s after t0."""
    if not spans:
        return
    t = tr.t_pos_s()
    gone = np.zeros(len(t), dtype=bool)
    for a, b in spans:
        gone |= (t > float(a) - T_SYM) & (t < float(b) + T_SYM)
    tr.psi = np.where(gone, 0.0, tr.psi)


def receive_pass(cap, spec: FrameSpec, det: Detection, prior: EmPrior | None = None,
                 estimator: str = "joint", *, exclude_hz=(), f_mix_hz=None,
                 round_b: bool = True, return_tracks: bool = False,
                 erase_s=()):
    """Receive one detected pass (design 6.8).

    `cap` is a `frontend.Capture` (raw; blanked and normalised here), a
    `Prepared` one, or a CH-rate capture of a stored pass (fs = 250, for
    EM). A detection that has not been through gate V (z_ref nan) is
    verified first and its timing fitted; None is returned if it fails.

    erase_s, spans (a, b) in seconds after t0, receives a pass still in
    progress or with holes in its audio (the live view, `sstvae.qrss.live`):
    every symbol within one symbol of a span is erased after tracking
    (psi = 0), so the latents carry only what was heard. The capture in
    those spans should be noise (`live.slot_capture` fills it), not
    zeros: the tracker follows a fade, but a stream of zeros sends its
    gain estimate off to infinity and every heard symbol is then erased
    against it.
    """
    prep = prepare(cap)
    chan = channel_for(prep, spec, det, exclude_hz, f_mix_hz)
    if not np.isfinite(det.z_ref) or det.timing is None:
        det = track.verify(chan, CH_FS, spec, det)
        if det is None:
            return None
    q = prep.q
    # round A
    tr = track.track(chan, spec, det, track.make_classes(spec, prior=prior),
                     timing=det.timing)
    tr.z_ref = det.z_ref
    _erase(tr, erase_s)
    z, w, llr, diag = demod.extract(None, tr, spec, q, estimator)
    hdr = header.decode(llr.astype(np.float64)) if spec.n_hdr else None
    cw, cw_known = _cw_result(tr, spec, hdr)
    tracks = [tr]
    # round B
    if round_b and hdr is not None:
        tr2 = track.track(chan, spec, det, track.make_classes(spec, header.encode(hdr), prior),
                          cw_known=cw_known, timing=tr.timing, freq=tr.freq, outer=1)
        tr2.z_ref = det.z_ref
        _erase(tr2, erase_s)
        z, w, _, diag = demod.extract(None, tr2, spec, q, estimator)
        tr = tr2
        tracks.append(tr2)
    rep = track.report(tr, diag["kappa"], diag["suspect"])
    f_hz = float(tr.freq(0.0))
    pr = PassResult(
        uid=pass_uid(q, f_hz, z), q=int(q), frame=spec.name, waveform=WAVEFORM_CE,
        f_hz=f_hz, timing=tr.timing, report=rep, z=z, w=w, hdr_llr=llr, header=hdr, cw=cw,
        ch=chan.ch.astype(np.complex64), ch_fs=CH_FS, ch_t0_index=float(chan.t0_index),
        f_mix_hz=Fraction(chan.f_mix), psi=_psi_frame(tr), estimator=estimator)
    return (pr, tracks) if return_tracks else pr


def detection_from_pass(p: PassResult, lead_in_s: float = 0.0) -> Detection:
    """A detection reproducing pass p's frequency and timing (for EM re-receives)."""
    t = np.array([-10.0, 0.0, 1800.0])
    r = p.report
    path = FreqPath(t_s=t, f_hz=r.offset_hz + r.drift_hz_per_min * t / 60.0,
                    weight=np.ones(3))
    return Detection(f_hz=p.f_hz, path=path, timing=p.timing, z_ref=p.report.z_ref,
                     method="template", lead_in_s=lead_in_s)


def capture_from_pass(p: PassResult) -> Prepared:
    """The CH-rate capture a pass keeps, as `receive_pass` input (pass
    f_mix_hz=p.f_mix_hz with it)."""
    return Prepared(fe=np.asarray(p.ch), keep=None, q=p.q, t0_index=p.ch_t0_index, fs=p.ch_fs)


def _aliases(dets: list[Detection]) -> list[Detection]:
    """Drop detections at a stronger one's f +- k/T (k = 1, 2) with its timing
    to within k + 1/2 symbols (a 1/T alias can sit a whole symbol off)."""
    out: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.z_ref):
        if any(abs(abs(d.f_hz - g.f_hz) - k / T_SYM) < ALIAS_HZ
               and abs(d.timing.tau0 - g.timing.tau0) < CH_FS * T_SYM * (k + 0.5)
               and g.z_ref > ALIAS_RATIO * d.z_ref
               for g in out for k in (1, 2)):
            continue
        out.append(d)
    return out


def _ghost_of(f_hz: float, tau0: float, g_f_hz: float, g_tau0: float) -> bool:
    """Whether a detection at (f_hz, tau0) is a ghost of a stronger one at (g_f_hz, g_tau0)."""
    return abs(f_hz - g_f_hz) < GHOST_HZ and abs(tau0 - g_tau0) < GHOST_T * T_SYM * CH_FS


def _suspect_neighbour(p: PassResult, out: list[PassResult]) -> bool:
    """Whether p is a SUSPECT pass at the gate beside a sound, much stronger one
    (module docstring, "Ghosts"): within GHOST_HZ, Z_ref under GHOST_Z_RATIO of it."""
    return bool(p.report.suspect) and any(
        abs(p.f_hz - g.f_hz) < GHOST_HZ and not g.report.suspect
        and p.report.z_ref < GHOST_Z_RATIO * g.report.z_ref for g in out)


def _ghosts(dets: list[Detection]) -> list[Detection]:
    """Drop detections that are ghosts of a stronger one (module docstring)."""
    out: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.z_ref):
        if not any(_ghost_of(d.f_hz, d.timing.tau0, g.f_hz, g.timing.tau0) for g in out):
            out.append(d)
    return out


def detect(prep: Prepared, spec: FrameSpec, live_only: bool = False,
           max_candidates: int | None = None, budget_s: float | None = None) -> list[Detection]:
    """Acquisition plus gate V over a prepared capture: verified detections.

    max_candidates bounds how many candidates go to the gate (preamble
    hits first, then the strongest of the rest), and budget_s stops
    verifying new ones after that many seconds; a candidate that is not
    a signal costs most (about a minute over a whole FULL slot, measured
    on voice-like audio).
    """
    t_end = None if budget_s is None else time.monotonic() + budget_s
    acq = acquire.acquire(frontend.Capture(prep.fe, prep.q, prep.t0_index, prep.fs), spec,
                          live_only=live_only, preprocessed=True)
    cands = acq.detections(spec)
    if max_candidates is not None:
        cands = cands[:max_candidates]
    found: list[Detection] = []
    for d in cands:
        if t_end is not None and time.monotonic() > t_end:
            break
        if any(abs(d.f_hz - g.f_hz) < MERGE_HZ for g in found):
            continue
        chan = channel_for(prep, spec, d)
        v = track.verify(chan, CH_FS, spec, d)
        if v is None:
            continue
        if v.method != "preamble" and v.lead_in_s == 0.0:
            v.lead_in_s = acquire.lead_in_span(prep.fe, prep.t0_index, v.f_hz, 0.0,
                                               v.timing.tau0 / CH_FS, prep.fs)
        found.append(v)
    merged: list[Detection] = []
    for d in sorted(found, key=lambda d: -d.z_ref):
        if all(abs(d.f_hz - g.f_hz) >= MERGE_HZ for g in merged):
            merged.append(d)
    return _ghosts(_aliases(merged))


def receive_slot(cap, spec: FrameSpec = FULL, live_only: bool = False,
                 estimator: str = "joint", *, erase_s=(), round_b: bool = True,
                 dets: list[Detection] | None = None,
                 max_candidates: int | None = None,
                 budget_s: float | None = None) -> list[PassResult]:
    """A1 -> A2 (+ A3) -> V -> receive_pass for every signal in a stored slot.

    erase_s and round_b go to every `receive_pass`; `dets` skips the
    acquisition and receives those detections instead (the live view
    reuses the ones it found on the preamble). max_candidates and
    budget_s go to `detect`.
    """
    prep = prepare(cap)
    if dets is None:
        dets = detect(prep, spec, live_only, max_candidates, budget_s)
    out: list[PassResult] = []
    for d in dets:
        others = tuple(g.f_hz for g in dets if g is not d)
        p = receive_pass(prep, spec, d, estimator=estimator, exclude_hz=others,
                         round_b=round_b, erase_s=erase_s)
        if p is None:
            continue
        # tracking moves a pass's frequency: a ghost that got past detect()
        # can land on the stronger pass it came from
        if any(_ghost_of(p.f_hz, p.timing.tau0, g.f_hz, g.timing.tau0) for g in out):
            continue
        if _suspect_neighbour(p, out):
            continue
        out.append(p)
    return out


def capture_from_wav(path, q: int, start_unix: float | None = None) -> frontend.Capture:
    """A slot capture from an 8 kHz WAV; start_unix None means it starts at
    t0 - 12 s (as `qrss_transmit.py` writes it)."""
    from sstvae import wavio

    x = wavio.read_wav(str(path))
    fe = frontend.audio_to_fe(x)
    t0 = 900.0 * int(q) + 1.0
    lead = 12.0 if start_unix is None else t0 - float(start_unix)
    return frontend.Capture(fe=fe, q=int(q), t0_index=lead * FE_FS, fs=FE_FS)


def receive_wav(path, q: int, start_unix: float | None = None, spec: FrameSpec = FULL,
                estimator: str = "joint") -> list[PassResult]:
    """`receive_slot` on a WAV file."""
    return receive_slot(capture_from_wav(path, q, start_unix), spec, estimator=estimator)


def mean_w_db(p: PassResult) -> float:
    w = np.asarray(p.w, dtype=np.float64)
    return float(10 * math.log10(max(float(np.mean(w)), 1e-30)))


__all__ = ["Prepared", "prepare", "channel_for", "receive_pass", "receive_slot",
           "receive_wav", "capture_from_wav", "detect", "detection_from_pass",
           "capture_from_pass", "mean_w_db", "Timing"]
