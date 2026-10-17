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

Those ghosts, and the L copy of a strong CE carrier (A3 cannot tell
which waveform the carrier's line is, so it is gated as L too: its
pilot at the carrier, its band 9.5 Hz below), are A3 lines that
`detect` used to gate one by one, only to refuse them or drop them as
ghosts: up to 8 CPU s each on SHORT at -12 and -6 dB, and on FULL about
a minute each of the end-of-slot search's 180 s. So a whole-slot (A3)
candidate inside the own band of a signal found already on its
preamble (`_owned`: within GHOST_HZ of a CE carrier, or from
L_OWN_BELOW_HZ below an L pilot to L_OWN_ABOVE_HZ above it), and no
stronger in A3 than that signal's own line, is not gated, nor counted
against max_candidates; preamble hits and hints always are, and the
hits come first. A ghost or an L copy is no stronger than the line it
comes from, but a station whose preamble faded can be stronger than a
weaker one whose preamble was heard, and it is gated. The price is
the ghost rule's, without its timing condition: a station heard only by
the whole-slot search inside a preamble-found one's band, and weaker
there than it, is not received, timing or not; at that distance their
spectra overlap anyway. `rules.GATING` set back gates every candidate
as before.

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

**Waveform L, and both at once.** `receive_pass` dispatches on the
frame's waveform: an L frame (`lframe.LFrameSpec`, f_hz its pilot) is
gated, tracked and demodulated by `ltrack` and `ldemod` in the same two
rounds, its header decoded as format version 2 only (an L frame has no
version 1, so never `frame.plain`). `candidates`, `detect` and
`receive_slot` take a frame or a tuple of frames, at most one per
waveform (`frames`). A selection of CE alone is the CE receiver above,
call for call; one holding an L frame is the joint receiver
(`is_joint`): the waveform-aware acquisition (`acquire.acquire` with the
tuple), each candidate gated with its own waveform's frame, aliases and
ghosts per waveform at that waveform's symbol, and each detection
received with its own frame.

One signal has one identity, its band centre (`centre_hz`): a CE
carrier's frequency, or L_EXCLUDE_OFFSET_HZ below an L pilot (L's data
run from 22.07 to 3.10 Hz below it, so the 45 Hz guard a neighbour gets
in another signal's noise floor, about that centre, covers [f - 54.5,
f + 35.5]). Merging, "already found", the noise exclusion and the merge
after tracking compare centres within MERGE_HZ, across waveforms; where
a CE and an L result share a centre the larger Z_ref stands for the
signal, and its waveform with it.

CE's gate V passes an L signal (Z_ref 6-11 on l-short at -12 and -6 dB:
L's references correlate with CE's in the data band). So in the joint
receiver a CE accept is checked again on the known symbols' residual
against the noise model (`_ce_resid_ratio`, R; measured: CE short
1.02-1.07 at -12 dB and 1.29-1.35 at -6, CE's own distortion not being
in the model; an L signal under the CE gate 3.6-4.3 at -12 dB and
12.7-15.4 at -6) and refused when R > CE_RESID_MIN and
Z_ref / sqrt(R) <= Z_ACCEPT. L's gate makes the converse check itself
(`ltrack.verify_track_l`). The CE-only receiver does neither: it is
unchanged.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import acquire, cwid, demod, frontend, header, ldemod, lframe, ltrack, rules, track
from .constants import (CH_FS, FE_FS, FORMAT_VERSIONS, LEAD_IN_MAX_S, SPAN, T_SYM,
                        WAVEFORM_CE, WAVEFORM_L, WAVEFORM_NAMES, Z_ACCEPT)
from .frame import FULL, FrameSpec, plain
from .frame import layout as frame_layout
from .morse import check_callsign, keying_units
from .types import CwIdResult, Detection, EmPrior, FreqPath, PassResult, Timing, pass_uid

MERGE_HZ = 5.0
GHOST_HZ = 20.0            # a weaker detection this close to a stronger one...
GHOST_T = 3.0              # ...with tau0 within this many symbols of it is a ghost
GHOST_Z_RATIO = 0.1        # a SUSPECT pass this much weaker than a sound one within GHOST_HZ
ALIAS_HZ = 0.5
ALIAS_RATIO = 3.0
MARGIN_S = 3.0
L_EXCLUDE_OFFSET_HZ = -9.5  # an L signal's band centre, from its pilot (module docstring)
L_OWN_BELOW_HZ = 22.1       # an L signal's own band: this far below its pilot (its data's edge)...
L_OWN_ABOVE_HZ = 3.1        # ...to this far above it (`_owned`; centre +- 12.6 Hz)
CE_RESID_MIN = 2.0          # joint receiver: a CE accept past this residual ratio is re-checked
L_HEADER_VERSIONS = (2,)    # what an L header may say: there is no L version 1


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


# --- which frames, and where a signal is ------------------------------------------------------

def frames(spec) -> tuple:
    """The frames a selection names, CE's first: one frame of either
    waveform, or a tuple (or list) of them, at most one per waveform."""
    m = acquire.frame_map(spec) if isinstance(spec, (tuple, list)) else {int(spec.waveform): spec}
    return tuple(m[w] for w in (WAVEFORM_CE, WAVEFORM_L) if w in m)


def is_joint(spec) -> bool:
    """Whether a selection takes the joint receiver: it holds an L frame. CE
    alone (a frame, or a tuple of one) is the CE receiver, call for call."""
    return any(s.waveform == WAVEFORM_L for s in frames(spec))


def _single(spec):
    """The one CE frame of a selection that is not joint, as given."""
    return frames(spec)[0] if isinstance(spec, (tuple, list)) else spec


def centre_hz(obj) -> float:
    """A Detection's or PassResult's band centre: f_hz for CE, the pilot
    plus L_EXCLUDE_OFFSET_HZ for L (module docstring)."""
    if getattr(obj, "waveform", WAVEFORM_CE) == WAVEFORM_L:
        return float(obj.f_hz) + L_EXCLUDE_OFFSET_HZ
    return float(obj.f_hz)


def _centres(skip) -> list[float]:
    """skip_hz for the joint receiver: centres, or Detections / PassResults."""
    return [float(f) if isinstance(f, (int, float, np.floating, np.integer)) else centre_hz(f)
            for f in skip]


def _ce_hints(hints):
    """Hints for the CE receiver: (f, waveform) pairs as f when the waveform
    is CE and left out when not; bare frequencies as they are."""
    if not any(isinstance(h, (tuple, list)) for h in hints):
        return hints
    return [float(h[0]) if isinstance(h, (tuple, list)) else h for h in hints
            if not isinstance(h, (tuple, list)) or int(h[1]) == WAVEFORM_CE]


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
    dur = spec.keyed_end_pos * spec.t_sym
    slip = 260e-6 * dur
    lo = -(LEAD_IN_MAX_S + SPAN * spec.t_sym) - MARGIN_S - slip
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
        # a confident read of another call disagrees, whatever the match; a
        # Python bool whatever z's type (cwid.match returns a float, but
        # z > Z_ACCEPT on a numpy scalar is an np.bool_, which json.dumps
        # refuses wherever a sidecar or the store writes the pass)
        agrees = bool((text == call) or (text is None and z > Z_ACCEPT))
        known = (keying_units(call), ook) if z > Z_ACCEPT else None
    else:
        z = cwid.match(soft, text) if text else 0.0
        agrees = None
        known = None
    return CwIdResult(text=text, z_match=float(z), keying=kt, agrees=agrees), known


def _check_waveform(spec, wf) -> None:
    """receive_pass's refusal of a detection of waveform wf on a frame of
    another waveform (ValueError)."""
    if wf != spec.waveform:
        raise ValueError(f"a waveform {WAVEFORM_NAMES.get(wf, wf)} detection cannot be "
                         f"received as frame {spec.name!r} (waveform "
                         f"{WAVEFORM_NAMES[spec.waveform]}): pass lframe.twin(spec, "
                         f"det.waveform)")


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

    An L frame takes `ltrack` and `ldemod` through the same steps
    (`_receive_pass_l`). `det` must be of the frame's waveform (the
    acquisition says which, `Detection.waveform`): a ValueError otherwise.
    """
    _check_waveform(spec, getattr(det, "waveform", WAVEFORM_CE))
    if spec.waveform == WAVEFORM_L:
        return _receive_pass_l(cap, spec, det, prior, estimator, exclude_hz=exclude_hz,
                               f_mix_hz=f_mix_hz, round_b=round_b,
                               return_tracks=return_tracks, erase_s=erase_s)
    prep = prepare(cap)
    chan = channel_for(prep, spec, det, exclude_hz, f_mix_hz)
    if not np.isfinite(det.z_ref) or det.timing is None:
        det = track.verify(chan, CH_FS, spec, det, erase_s=erase_s)
        if det is None:
            return None
    q = prep.q
    # round A
    tr = track.track(chan, spec, det, track.make_classes(spec, prior=prior),
                     timing=det.timing)
    tr.z_ref = det.z_ref
    track.erase(tr, erase_s)
    z, w, llr, diag = demod.extract(None, tr, spec, q, estimator)
    hdr, llr, spec = decode_header(llr, diag, spec)
    cw, cw_known = _cw_result(tr, spec, hdr)
    tracks = [tr]
    # round B
    if round_b and hdr is not None:
        tr2 = track.track(chan, spec, det, track.make_classes(spec, header.encode(hdr), prior),
                          cw_known=cw_known, timing=tr.timing, freq=tr.freq, outer=1)
        tr2.z_ref = det.z_ref
        track.erase(tr2, erase_s)
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


def _receive_pass_l(cap, spec, det: Detection, prior: EmPrior | None, estimator: str, *,
                    exclude_hz, f_mix_hz, round_b: bool, return_tracks: bool, erase_s):
    """`receive_pass` for an L frame: the channel centred on the pilot, gate V
    (`ltrack.verify_l`) for an unverified detection, round A, the header
    (version 2) and the callsign windows, then round B with the header and
    keying known. There is no spread removal (L's copy is linear, in
    `make_classes_l`'s data mean), and the keying is FSK: cw_known never
    says OOK. The report quotes SNR2500 on the frame's keyed power
    (`ltrack.report_l`).

    The refusals are receive_pass's: a detection of another waveform than
    the frame's, and (this receives an L pass) a frame that is not L, which
    gets receive_pass's refusal of an L detection on it."""
    _check_waveform(spec, getattr(det, "waveform", WAVEFORM_CE))
    _check_waveform(spec, WAVEFORM_L)
    prep = prepare(cap)
    chan = channel_for(prep, spec, det, exclude_hz, f_mix_hz)
    if not np.isfinite(det.z_ref) or det.timing is None:
        det = ltrack.verify_l(chan, CH_FS, spec, det, erase_s=erase_s)
        if det is None:
            return None
    q = prep.q
    # round A
    tr = ltrack.track_l(chan, spec, det, ltrack.make_classes_l(spec, prior=prior),
                        timing=det.timing)
    tr.z_ref = det.z_ref
    track.erase(tr, erase_s)
    z, w, llr, diag = ldemod.extract_l(None, tr, spec, q, estimator)
    hdr, llr, spec = decode_header(llr, diag, spec)
    cw, cw_known = _cw_result(tr, spec, hdr)
    tracks = [tr]
    # round B
    if round_b and hdr is not None:
        tr2 = ltrack.track_l(chan, spec, det,
                             ltrack.make_classes_l(spec, header.encode(hdr), prior),
                             cw_known=cw_known, timing=tr.timing, freq=tr.freq, outer=1)
        tr2.z_ref = det.z_ref
        track.erase(tr2, erase_s)
        z, w, _, diag = ldemod.extract_l(None, tr2, spec, q, estimator)
        tr = tr2
        tracks.append(tr2)
    rep = ltrack.report_l(tr, diag["kappa"], diag["suspect"])
    f_hz = float(tr.freq(0.0))
    pr = PassResult(
        uid=pass_uid(q, f_hz, z), q=int(q), frame=spec.name, waveform=WAVEFORM_L,
        f_hz=f_hz, timing=tr.timing, report=rep, z=z, w=w, hdr_llr=llr, header=hdr, cw=cw,
        ch=chan.ch.astype(np.complex64), ch_fs=CH_FS, ch_t0_index=float(chan.t0_index),
        f_mix_hz=Fraction(chan.f_mix), psi=_psi_frame(tr), estimator=estimator)
    return (pr, tracks) if return_tracks else pr


def decode_header(llr, diag, spec: FrameSpec):
    """(header or None, the LLRs to keep, the frame to go on with) after round A.

    Both copies together first (spec 5.1). Failing that, each alone: the
    spread copy, for a block lost to a fade whose mis-scaled LLRs spoiled
    the sum; and the block, which decodes either format version -- a
    version 1 recording carries no spread copy, so its spread LLRs are
    only noise. A version 1 header switches the pass to the frame without
    the spread copy (`frame.plain`), so round B removes nothing. A frame
    named as version 1 (`frame.LEGACY`, for an old recording) has only its
    block and accepts either version.

    An L frame (`lframe.LFrameSpec`) has no version 1: each try accepts
    version 2 alone, the frame never changes, and the LLRs kept are the
    sum's whichever copy decoded.
    """
    if not spec.has_header:
        return None, llr, spec
    if spec.waveform == WAVEFORM_L:
        return _decode_header_l(llr, diag, spec), llr, spec
    if spec.hdr_rho == 0:
        return header.decode(np.asarray(llr, np.float64), versions=FORMAT_VERSIONS), llr, spec
    hdr = header.decode(np.asarray(llr, np.float64))
    if hdr is not None or not spec.n_hdr:
        return hdr, llr, spec
    spread, block = diag.get("llr_spread"), diag.get("llr_block")
    if spread is not None:
        hdr = header.decode(np.asarray(spread, np.float64))
        if hdr is not None:
            return hdr, llr, spec
    if block is not None:
        hdr = header.decode(np.asarray(block, np.float64), versions=FORMAT_VERSIONS)
        if hdr is not None and hdr.version == 1:
            return hdr, np.asarray(block, llr.dtype), plain(spec)
    return hdr, llr, spec


def _decode_header_l(llr, diag, spec):
    """`decode_header`'s L branch: both copies, then the spread copy alone,
    then the block alone, version 2 only."""
    hdr = header.decode(np.asarray(llr, np.float64), versions=L_HEADER_VERSIONS)
    if hdr is not None or not spec.n_hdr or spec.hdr_rho == 0:
        return hdr
    for key in ("llr_spread", "llr_block"):
        part = diag.get(key)
        if part is not None:
            hdr = header.decode(np.asarray(part, np.float64), versions=L_HEADER_VERSIONS)
            if hdr is not None:
                return hdr
    return None


def detection_from_pass(p: PassResult, lead_in_s: float = 0.0) -> Detection:
    """A detection reproducing pass p's frequency and timing (for EM re-receives)."""
    t = np.array([-10.0, 0.0, 1800.0])
    r = p.report
    path = FreqPath(t_s=t, f_hz=r.offset_hz + r.drift_hz_per_min * t / 60.0,
                    weight=np.ones(3))
    return Detection(f_hz=p.f_hz, path=path, timing=p.timing, z_ref=p.report.z_ref,
                     method="template", lead_in_s=lead_in_s,
                     waveform=int(getattr(p, "waveform", WAVEFORM_CE)))


def capture_from_pass(p: PassResult) -> Prepared:
    """The CH-rate capture a pass keeps, as `receive_pass` input (pass
    f_mix_hz=p.f_mix_hz with it)."""
    return Prepared(fe=np.asarray(p.ch), keep=None, q=p.q, t0_index=p.ch_t0_index, fs=p.ch_fs)


def _aliases(dets: list[Detection], t_sym: float = T_SYM) -> list[Detection]:
    """Drop detections at a stronger one's f +- k/T (k = 1, 2) with its timing
    to within k + 1/2 symbols (a 1/T alias can sit a whole symbol off).
    T is t_sym, the detections' waveform's symbol (one waveform at a time)."""
    out: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.z_ref):
        if any(abs(abs(d.f_hz - g.f_hz) - k / t_sym) < ALIAS_HZ
               and abs(d.timing.tau0 - g.timing.tau0) < CH_FS * t_sym * (k + 0.5)
               and g.z_ref > ALIAS_RATIO * d.z_ref
               for g in out for k in (1, 2)):
            continue
        out.append(d)
    return out


def _ghost_of(f_hz: float, tau0: float, g_f_hz: float, g_tau0: float,
              t_sym: float = T_SYM) -> bool:
    """Whether a detection at (f_hz, tau0) is a ghost of a stronger one at (g_f_hz, g_tau0).

    For L (t_sym = T_L) the rule is CE's with its timing window in L
    symbols, unmeasured: L's references recur every 16 T_L, so its ghosts,
    if gate V passes them as it does CE's, would sit k 1.03 Hz off, inside
    GHOST_HZ as CE's 2.06 k Hz are; and GHOST_HZ also covers an L signal's
    own data band, 3-22 Hz below its pilot."""
    return abs(f_hz - g_f_hz) < GHOST_HZ and abs(tau0 - g_tau0) < GHOST_T * t_sym * CH_FS


def _suspect_neighbour(p: PassResult, out: list[PassResult]) -> bool:
    """Whether p is a SUSPECT pass at the gate beside a sound, much stronger one
    (module docstring, "Ghosts"): within GHOST_HZ, Z_ref under GHOST_Z_RATIO of it."""
    return bool(p.report.suspect) and any(
        abs(p.f_hz - g.f_hz) < GHOST_HZ and not g.report.suspect
        and p.report.z_ref < GHOST_Z_RATIO * g.report.z_ref for g in out)


def _suspect_beside(p: PassResult, out: list[PassResult]) -> bool:
    """`_suspect_neighbour` in the joint receiver: the same rule across
    waveforms, by band centre (a SUSPECT pass that weak beside a sound one
    has unusable data whichever waveform either is; within a waveform the
    centres' difference is the frequencies')."""
    c = centre_hz(p)
    return bool(p.report.suspect) and any(
        abs(c - centre_hz(g)) < GHOST_HZ and not g.report.suspect
        and p.report.z_ref < GHOST_Z_RATIO * g.report.z_ref for g in out)


def _a3_lines(cands) -> list[tuple[float, float, float]]:
    """(line, centre, Z) of each whole-slot (A3) candidate: what `_owned`
    weighs a line against."""
    return [(float(c.f_hz), centre_hz(c), float(c.a3_z)) for c in cands
            if c.method == "tbd" and math.isfinite(getattr(c, "a3_z", math.nan))]


def _owned(d: Detection, found: list[Detection], a3=()) -> bool:
    """Whether candidate d is not to be gated: a whole-slot (A3) line at a
    frequency inside the own band of a signal found already on its
    preamble (within GHOST_HZ of a CE carrier, or from L_OWN_BELOW_HZ
    below an L pilot to L_OWN_ABOVE_HZ above it) and no stronger in A3
    than that signal's own line there (`a3`, `_a3_lines`: the largest Z
    of an A3 line within MERGE_HZ of its line or its centre). Module
    docstring, "Ghosts". d.f_hz is the line: a CE carrier, or an L pilot.

    Only a preamble-found signal owns its band: those are gated first,
    so nothing weaker can have taken the band before them. Its own A3
    line is the yardstick because the preamble says nothing of the rest
    of the slot: a weaker station whose preamble was heard must not hide
    a stronger one whose preamble faded (2026-10-12 review). A ghost or
    the L copy of the owner's carrier is no stronger than that line; a
    candidate when the owner has none in A3, or a hint (not A3's), is
    gated."""
    if d.method != "tbd" or rules.old(rules.GATING):
        return False
    z = float(getattr(d, "a3_z", math.nan))
    if not math.isfinite(z):
        return False
    f = float(d.f_hz)
    for g in found:
        if g.method != "preamble":
            continue
        if getattr(g, "waveform", WAVEFORM_CE) == WAVEFORM_L:
            inside = g.f_hz - L_OWN_BELOW_HZ <= f <= g.f_hz + L_OWN_ABOVE_HZ
        else:
            inside = abs(f - g.f_hz) < GHOST_HZ
        if not inside:
            continue
        c = centre_hz(g)
        own = [zz for line, cc, zz in a3
               if abs(line - g.f_hz) < MERGE_HZ or abs(cc - c) < MERGE_HZ]
        if own and z <= max(own):
            return True
    return False


def _ghosts(dets: list[Detection], t_sym: float = T_SYM) -> list[Detection]:
    """Drop detections that are ghosts of a stronger one (module docstring)."""
    out: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.z_ref):
        if not any(_ghost_of(d.f_hz, d.timing.tau0, g.f_hz, g.timing.tau0, t_sym)
                   for g in out):
            out.append(d)
    return out


def candidates(prep: Prepared, spec, live_only: bool = False,
               hints=()) -> list[Detection]:
    """Acquisition over a prepared capture: the candidates for gate V,
    preamble hits first (A1 -> A2, and A3 unless live_only), with the
    `hints` (expected frequencies, `acquire.acquire`) A2 did not pass after them.

    `spec` is a frame or a tuple of frames (`frames`). Joint (`is_joint`):
    the waveform-aware acquisition, every Detection carrying the waveform
    it is to be gated as, in `acquire.Acquisition.detections`' order;
    hints are frequencies (every waveform) or (f, waveform) pairs. Not
    joint: CE's acquisition, a pair's frequency used only when its
    waveform is CE."""
    cap = frontend.Capture(prep.fe, prep.q, prep.t0_index, prep.fs)
    if is_joint(spec):
        specs = frames(spec)
        acq = acquire.acquire(cap, specs, live_only=live_only, preprocessed=True, hints=hints)
        return acq.detections(specs)
    spec = _single(spec)
    acq = acquire.acquire(cap, spec, live_only=live_only, preprocessed=True,
                          hints=_ce_hints(hints))
    return acq.detections(spec)


def detect(prep: Prepared, spec, live_only: bool = False,
           max_candidates: int | None = None, budget_s: float | None = None,
           erase_s=(), cands: list[Detection] | None = None,
           skip_hz=(), hints=()) -> list[Detection]:
    """Acquisition plus gate V over a prepared capture: verified detections.

    max_candidates bounds how many candidates go to the gate (preamble
    hits first, then the strongest of the rest; those `_owned` skips are
    not counted), and budget_s stops
    verifying new ones after that many seconds; a candidate that is not
    a signal costs most (about a minute over a whole FULL slot, measured
    on voice-like audio). erase_s, spans (a, b) in seconds after t0, were
    not heard: the gate counts only what was (`track.verify`). `cands`
    are `candidates` already found (the live view keeps a slot's), and a
    candidate within MERGE_HZ of `skip_hz` (signals known already) is not
    verified again, though it is still left out of the others' noise; nor
    is a whole-slot candidate in a preamble-found signal's own band and no
    stronger in A3 than its own line (`_owned`). `hints` go to `candidates`.

    `spec` is a frame or a tuple of frames: joint (`is_joint`) is
    `_detect_joint`, where skip_hz holds band centres (`centre_hz`; a
    Detection or PassResult is taken as its centre).
    """
    if is_joint(spec):
        return _detect_joint(prep, frames(spec), live_only, max_candidates, budget_s,
                             erase_s, cands, skip_hz, hints)
    spec = _single(spec)
    hints = _ce_hints(hints)
    t_end = None if budget_s is None else time.monotonic() + budget_s
    if cands is None:
        cands = candidates(prep, spec, live_only, hints)
    a3 = _a3_lines(cands)
    found: list[Detection] = []
    # a neighbour's sideband in the band the noise floor is measured in
    # (85-120 Hz off) reads as noise and sinks a weaker signal's Z_ref:
    # KC2G at -18.6 dB beside AG7EW 101 Hz away and 14 dB stronger read
    # Z_ref ~1 on the first minutes of the slot, and 8-11 without it. So
    # each candidate leaves out the preamble hits (signals, at A2's false
    # alarm rate) and whatever has been verified already.
    hits = [c.f_hz for c in cands if c.method == "preamble"] + [float(f) for f in skip_hz]
    taken = 0
    for d in cands:
        if t_end is not None and time.monotonic() > t_end:
            break
        if _owned(d, found, a3):
            continue                         # (not gated, so not counted)
        if max_candidates is not None and taken >= max_candidates:
            break
        taken += 1
        if any(abs(d.f_hz - f) < MERGE_HZ for f in [g.f_hz for g in found] + list(skip_hz)):
            continue
        others = {f for f in hits + [g.f_hz for g in found] if abs(f - d.f_hz) >= MERGE_HZ}
        chan = channel_for(prep, spec, d, tuple(sorted(others)))
        v = track.verify(chan, CH_FS, spec, d, erase_s=erase_s)
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


def _detect_joint(prep: Prepared, specs: tuple, live_only: bool, max_candidates, budget_s,
                  erase_s, cands, skip_hz, hints) -> list[Detection]:
    """`detect` with CE and L listened for: each candidate gated with its
    waveform's frame, everything compared by band centre.

    The candidates come in `acquire.Acquisition.detections`' order
    (preamble hits of either waveform, then hints, then A3's by spectral
    shape), and max_candidates and budget_s are one budget across both. A
    candidate whose centre is within MERGE_HZ of a signal found already
    (of either waveform) or of skip_hz is not gated, nor a whole-slot
    candidate in a preamble-found signal's own band and no stronger in A3
    than its own line (`_owned`, by the candidate's line and the found
    signal's waveform); the preamble hits'
    and the found signals' centres leave each gate's noise floor
    (`_gate`). Then one detection per centre, the larger Z_ref, and
    aliases and ghosts per waveform at its own symbol.
    """
    smap = acquire.frame_map(specs)
    t_end = None if budget_s is None else time.monotonic() + budget_s
    if cands is None:
        cands = candidates(prep, specs, live_only, hints)
    a3 = _a3_lines(cands)
    skip = _centres(skip_hz)
    hits = [centre_hz(c) for c in cands if c.method == "preamble"] + skip
    found: list[Detection] = []
    taken = 0
    for d in cands:
        if t_end is not None and time.monotonic() > t_end:
            break
        if _owned(d, found, a3):
            continue                         # (not gated, so not counted)
        if max_candidates is not None and taken >= max_candidates:
            break
        taken += 1
        sp = smap.get(getattr(d, "waveform", WAVEFORM_CE))
        if sp is None:
            continue                         # (a waveform not listened for)
        c0 = centre_hz(d)
        if any(abs(c0 - f) < MERGE_HZ for f in [centre_hz(g) for g in found] + skip):
            continue
        others = {f for f in hits + [centre_hz(g) for g in found] if abs(f - c0) >= MERGE_HZ}
        v = _gate(prep, sp, d, tuple(sorted(others)), erase_s)
        if v is not None:
            found.append(v)
    merged: list[Detection] = []
    for d in sorted(found, key=lambda d: -d.z_ref):
        if all(abs(centre_hz(d) - centre_hz(g)) >= MERGE_HZ for g in merged):
            merged.append(d)
    out: list[Detection] = []
    for w, sp in smap.items():
        out += _ghosts(_aliases([d for d in merged if d.waveform == w], sp.t_sym), sp.t_sym)
    return sorted(out, key=lambda d: -d.z_ref)


def _gate(prep: Prepared, spec, d: Detection, others: tuple, erase_s) -> Detection | None:
    """Gate V of one joint-receiver candidate with its waveform's frame, the
    signals at `others` (centres) left out of its noise floor; a CE accept
    is checked against an L signal (`_ce_resid_ratio`). A lead-in is
    measured for a detection that has none at the frame's own symbol."""
    chan = channel_for(prep, spec, d, others)
    if spec.waveform == WAVEFORM_L:
        v = ltrack.verify_l(chan, CH_FS, spec, d, erase_s=erase_s)
    else:
        v = track.verify(chan, CH_FS, spec, d, erase_s=erase_s)
        if v is not None:
            tr = track.verify_track(chan, spec, v, fit=False, erase_s=erase_s)
            R = _ce_resid_ratio(tr) if tr is not None else 1.0
            if R > CE_RESID_MIN and v.z_ref / math.sqrt(R) <= Z_ACCEPT:
                v = None                     # an L signal (or another) under CE's gate
    if v is not None and v.method != "preamble" and v.lead_in_s == 0.0:
        v.lead_in_s = acquire.lead_in_span(prep.fe, prep.t0_index, v.f_hz, 0.0,
                                           v.timing.tau0 / CH_FS, prep.fs, t_sym=spec.t_sym)
    return v


def _ce_resid_ratio(tr: track.TrackResult) -> float:
    """R of a CE gate-V track: the known symbols' residual |m - u c|^2 over
    N + |c|^2 P, weighted as Z_ref weighs them (CE's d = j Im c), as
    `ltrack.resid_ratio` is for L. 1.0 when no known symbol is usable.
    Measured (module docstring): CE 1.02-1.35, an L signal 3.6-15.4."""
    spec = tr.spec
    keyed_from = spec.keyed_start_pos - tr.lead_in_s / T_SYM
    c_known, _ = track.templates(spec, tr.pos, tr.classes.mu, tr.classes.nu, None, False,
                                 keyed_from)
    gk = tr.gi(np.asarray(frame_layout(spec).pos)[tr.classes.known])
    gk = gk[(gk >= 0) & (gk < len(tr.pos))]
    gk = gk[tr.psi[gk] >= track.PSI_MIN]
    if len(gk) == 0:
        return 1.0
    m, u, N, P = tr.m[gk], tr.u[gk], tr.N[gk], tr.P[gk]
    c = c_known[gk]
    u2, ci = np.abs(u) ** 2, c.imag
    V = ci * ci * (N * u2 + np.abs(c) ** 2 * u2 * P + N * P) / 2
    ok = V > 0
    if not ok.any():
        return 1.0
    w = ci[ok] ** 2 * u2[ok] / V[ok]
    om = w * w * ci[ok] ** 2 * u2[ok]
    e2 = np.abs(m[ok] - u[ok] * c[ok]) ** 2
    e0 = N[ok] + np.abs(c[ok]) ** 2 * P[ok]
    den = float(np.sum(om * e0))
    return float(np.sum(om * e2)) / den if den > 0 else 1.0


def receive_slot(cap, spec=FULL, live_only: bool = False,
                 estimator: str = "joint", *, erase_s=(), round_b: bool = True,
                 dets: list[Detection] | None = None,
                 max_candidates: int | None = None,
                 budget_s: float | None = None, hints=()) -> list[PassResult]:
    """A1 -> A2 (+ A3) -> V -> receive_pass for every signal in a stored slot.

    erase_s and round_b go to every `receive_pass`; `dets` skips the
    acquisition and receives those detections instead (the live view
    reuses the ones it found on the preamble). max_candidates, budget_s,
    erase_s and hints go to `detect`.

    `spec` is a frame or a tuple of frames: joint (`is_joint`) is
    `_receive_slot_joint`.
    """
    if is_joint(spec):
        return _receive_slot_joint(prepare(cap), frames(spec), live_only, estimator,
                                   erase_s=erase_s, round_b=round_b, dets=dets,
                                   max_candidates=max_candidates, budget_s=budget_s,
                                   hints=hints)
    spec = _single(spec)
    hints = _ce_hints(hints)
    prep = prepare(cap)
    if dets is None:
        dets = detect(prep, spec, live_only, max_candidates, budget_s, erase_s=erase_s,
                      hints=hints)
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
        # and two passes within MERGE_HZ are one signal whatever their
        # timings say (two signals that close cannot be received apart):
        # the stronger stands for it
        i = next((i for i, g in enumerate(out) if abs(p.f_hz - g.f_hz) < MERGE_HZ), None)
        if i is not None:
            if p.report.z_ref > out[i].report.z_ref:
                out[i] = p
            continue
        out.append(p)
    return out


def _receive_slot_joint(prep: Prepared, specs: tuple, live_only: bool, estimator: str, *,
                        erase_s, round_b: bool, dets, max_candidates, budget_s,
                        hints) -> list[PassResult]:
    """`receive_slot` with CE and L listened for: every detection received
    with its own waveform's frame (a waveform not in `specs` with its twin),
    the other signals' centres out of its noise floor. After tracking, the
    ghost rule applies within a waveform (at its own symbol), the
    suspect-neighbour rule across both by centre (`_suspect_beside`), and
    two passes within MERGE_HZ of one centre are one signal of either
    waveform, the larger Z_ref standing for it."""
    smap = acquire.frame_map(specs)
    if dets is None:
        dets = detect(prep, specs, live_only, max_candidates, budget_s, erase_s=erase_s,
                      hints=hints)
    out: list[PassResult] = []
    for d in dets:
        w = getattr(d, "waveform", WAVEFORM_CE)
        sp = smap[w] if w in smap else lframe.twin(specs[0], w)
        others = tuple(centre_hz(g) for g in dets if g is not d)
        p = receive_pass(prep, sp, d, estimator=estimator, exclude_hz=others,
                         round_b=round_b, erase_s=erase_s)
        if p is None:
            continue
        same = [g for g in out if g.waveform == p.waveform]
        if any(_ghost_of(p.f_hz, p.timing.tau0, g.f_hz, g.timing.tau0, sp.t_sym)
               for g in same):
            continue
        if _suspect_beside(p, out):
            continue
        c = centre_hz(p)
        i = next((i for i, g in enumerate(out) if abs(c - centre_hz(g)) < MERGE_HZ), None)
        if i is not None:
            if p.report.z_ref > out[i].report.z_ref:
                out[i] = p
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


def receive_wav(path, q: int, start_unix: float | None = None, spec=FULL,
                estimator: str = "joint") -> list[PassResult]:
    """`receive_slot` on a WAV file (`spec` a frame or a tuple of frames)."""
    return receive_slot(capture_from_wav(path, q, start_unix), spec, estimator=estimator)


def mean_w_db(p: PassResult) -> float:
    w = np.asarray(p.w, dtype=np.float64)
    return float(10 * math.log10(max(float(np.mean(w)), 1e-30)))


CONF_BINS = 120             # one per ~15 s of a FULL pass
CONF_FLOOR_DB = -30.0       # "no information": W of 0 (erased, or nothing there)


def confidence_db(p: PassResult, spec=None, elapsed_s: float | None = None,
                  n: int = CONF_BINS) -> list[float | None]:
    """A pass's confidence over time: per-latent SNR in dB, `n` bins from t0 to the frame's end.

    Data latents give their W (the per-latent SNR, demod's w = 1/v) at
    the time of their block. Header symbols give theirs from the LLRs,
    which are 2 y / s2 with E[y | x] = x = +-1, so E[llr^2] / 4 =
    (1 + s2) / s2^2, solved for 1/s2 over the bin. A bin's value is the
    mean over both, floored at CONF_FLOOR_DB. None where there is
    nothing to say: not yet arrived (past `elapsed_s` seconds after t0),
    not heard (psi 0 throughout), or no data or header symbols in it
    (the preamble).

    `spec` None, or of another waveform than the pass's, means the pass's
    own frame (`lframe.get_any(p.frame)`). In an L frame data symbol k
    carries latents 2k and 2k+1, and header symbol j bits 2j and 2j+1, at
    that symbol's time in L symbols.
    """
    pw = getattr(p, "waveform", None)
    if spec is None or (pw is not None and int(pw) != spec.waveform):
        spec = lframe.get_any(p.frame)
    if spec.waveform == WAVEFORM_L:
        lay = lframe.layout(spec)
        lat_at, bit_at = np.repeat(lay.data, 2), np.repeat(lay.hdr_sym, 2)
    else:
        lay = frame_layout(spec)
        lat_at, bit_at = lay.data, lay.hdr_bits
    t_sym = spec.t_sym
    end = float(spec.keyed_end_pos * t_sym)
    edges = np.linspace(0.0, end, n + 1)
    psi = getattr(p, "psi", None)
    psi = np.asarray(psi) if psi is not None and len(psi) == spec.n_pos else None

    def place(idx):
        pos = lay.pos[idx]
        t = pos * t_sym
        ok = np.ones(len(idx), dtype=bool) if psi is None else psi[pos] > 0
        if elapsed_s is not None:
            ok &= t <= elapsed_s
        return np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n - 1), ok

    w = np.asarray(p.w, dtype=np.float64)
    total = np.zeros(n)
    count = np.zeros(n)
    if len(w) == len(lat_at):
        b, ok = place(lat_at)
        total += np.bincount(b[ok], w[ok], minlength=n)
        count += np.bincount(b[ok], minlength=n)
    llr = getattr(p, "hdr_llr", None)
    if llr is not None and spec.n_hdr and len(llr) == len(bit_at):
        b, ok = place(bit_at)
        k = np.bincount(b[ok], minlength=n)
        m = np.bincount(b[ok], np.asarray(llr, dtype=np.float64)[ok] ** 2, minlength=n) / 4.0
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(k > 0, m / np.maximum(k, 1), 0.0)
            snr = np.where(m > 0, 2.0 * m / (1.0 + np.sqrt(1.0 + 4.0 * m)), 0.0)
        total += snr * k
        count += k
    out: list[float | None] = []
    for tot, c in zip(total, count):
        if c == 0:
            out.append(None)
        else:
            mean = tot / c
            db = 10.0 * math.log10(mean) if mean > 0 else CONF_FLOOR_DB
            out.append(round(max(db, CONF_FLOOR_DB), 1))
    return out


__all__ = ["Prepared", "prepare", "channel_for", "receive_pass", "receive_slot",
           "receive_wav", "capture_from_wav", "detect", "detection_from_pass",
           "capture_from_pass", "mean_w_db", "confidence_db", "Timing", "frames",
           "is_joint", "centre_hz", "candidates", "decode_header", "L_EXCLUDE_OFFSET_HZ"]
