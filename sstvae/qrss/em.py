"""Leave-one-out EM over a picture's passes, and the known-waveform
(template) search (design 8.3, spec 7 "Data-aided tracking with
leave-one-out references"; WP9).

Not a format module: nothing here goes on the air.

**EM.** Once a picture is partly received its own latents are a known
reference for tracking. For each member pass i of an accumulator, in
turn:

1. the leave-one-out accumulator (every member but i, plus what was
   merged from other receivers) gives the LMMSE latent
   a = (S'/W') W'/(1 + W') = S'/(1 + W') and its residual variance
   1/(1 + W'), in pass i's air order (`em_prior`);
2. that reference is re-modulated exactly as pass i was sent -- scrambled
   with pass i's q, precoded, phase-modulated with the e^{-v/2} factor of
   the mean waveform -- by giving the tracker x_hat = precode(a, q) and
   v = the block mean of 1/(1 + W') as soft data symbols (design 6.5);
3. pass i is re-tracked (timing refit included) and re-demodulated from
   its stored 250 Hz capture, and its (z, w) replace the old ones, which
   rebuilds the accumulator (`Store.update_pass`).

Step 3 reads each precoder block from a track whose templates leave that
block's parity of blocks out (`rereceive(split=True)`, the default; a
deviation from the design's single re-track): with the whole reference
in the templates, the reference's errors steer the gain and phase pass i
is read with at those same latents, the passes' errors correlate, and
the accumulator loses while every pass's own W rises.

What the reference buys, measured: little. Where tracking costs (a
disturbed SHORT pass at -24 dB beside a +8 dB reference) +0.1 to +0.3 dB
per pass on average against the same re-receive without it, -0.4 to
+0.7 dB seed to seed; on a quiet path, where the receiver tracks at the
genie, nothing measurable. The block split is
why: a precoder block spans about 2 s, and on a fast-fading path the
templates of the neighbouring blocks are already outside the channel's
coherence.

Rounds repeat until the accumulator's mean W gains less than `tol_db`.
The reference never contains pass i, so a pass of noise cannot confirm
itself: its template is independent of its own samples (P5).

What the re-receive knows beyond the pass's own first reception: the
accumulator's key fixes the header of every member (callsign, picture
ID, mode and codec ID from the key and the members' decoded headers, the
segment from membership), so the 2,474 header symbols are re-tracked as
known symbols even on a pass whose own header never decoded; and the
callsign windows' keying is known once any member's windows matched
(Z > 6). That is spec 8's "re-modulate exactly as pass i was sent".

**Template search** (spec 7 step 3, retroactive detection). A capture of
a slot that held no detectable pass is correlated with the expected
waveform E[s(t)] of the picture -- carrier, preamble, references, the
known header and the accumulator's soft data, with that slot's
scrambler -- over a grid of frequency, timing and clock hypotheses. The
statistic is A2's: 2 s chunks, coherent within a chunk and summed in
power over the frame,

    Lambda = sum_k |sum_n r[n + l] conj(e_k[n]) e^{-j 2 pi df n/fs}|^2 / (N0 E_k),

which is Gamma(K, 1) on noise for each cell (K chunks). The smallest
p-value is Bonferroni-corrected over every cell searched and turned into
a normal Z; a detection needs Z > 6, the receiver's single gate. The
carrier is the larger part of the template, so a pass of a *different*
picture from the same station is found too: the search finds passes,
and association (latent correlation) decides whose they are.
"""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import numpy as np
from scipy import stats

from . import ce, demod, frame, header, receiver, track
from .constants import CH_FS, LEAD_IN_MAX_S, N_HDR_BITS, SPAN, T_SYM, Z_ACCEPT
from .frame import FrameSpec
from .morse import check_callsign, keying_units
from .precoder import block_index, block_mean, precode
from .store import Store, canonical_index
from .types import Detection, EmPrior, FreqPath, PassResult, Timing

EM_ROUNDS = 3
EM_TOL_DB = 0.05

TS_CHUNK_S = 2.0                 # template search: coherent chunk length (66 symbols)
TS_NFFT = 2048                   # FFT per chunk at 250 Hz: 0.122 Hz bins
TS_DF_HZ = 1.0                   # frequency searched either side of each candidate
TS_TAU_S = 2.5                   # timing searched either side of nominal t0
TS_PPM_STEPS = 2                 # clock hypotheses either side of each candidate ppm
TS_SLIP_SAMPLES = 2.0            # clock step: at most this many CH samples of slip
TS_CH_REACH_HZ = 100.0           # a 250 Hz capture holds +-this about its mix (filter edge)


# --- references -------------------------------------------------------------------------------

def lmmse_latents(S, W) -> tuple[np.ndarray, np.ndarray]:
    """(a, r): LMMSE latents S/(1 + W) and their error variance 1/(1 + W).

    The latents are unit-RMS by contract, so their prior variance is 1;
    m = S/W is unbiased with variance 1/W, and a = m W/(1 + W). A latent
    nothing was heard of (W = 0) gives a = 0, r = 1: its prior.
    """
    S = np.asarray(S, dtype=np.float64)
    W = np.maximum(np.asarray(W, dtype=np.float64), 0.0)
    return S / (1.0 + W), 1.0 / (1.0 + W)


def prior_from(S, W, q: int) -> EmPrior:
    """The soft data symbols of a pass in slot q, from an air-order reference (S, W).

    x_hat = precode(a, q); v = the block mean of 1/(1 + W): the precoder is
    orthonormal within a block, so a precoded symbol's error variance is
    the mean of its block's latent error variances.
    """
    a, r = lmmse_latents(S, W)
    return EmPrior(x_hat=precode(a, q), v=block_mean(r))


def em_prior(acc, uid: str, store: Store, q: int, spec: FrameSpec) -> EmPrior:
    """Pass `uid`'s soft data symbols from the accumulator without it (design 8.3)."""
    S_, W_ = acc.loo(uid, store)
    n = spec.n_data
    if len(S_) != n:
        raise ValueError(f"pass {uid} holds {len(S_)} latents; frame {spec.name!r} sends {n}")
    return prior_from(S_, W_, q)


def acc_mean_w_db(acc) -> float:
    """10 log10 of the accumulator's mean W over the latents it has heard."""
    W = np.asarray(acc.W, dtype=np.float64)
    heard = W > 0
    if not heard.any():
        return float("-inf")
    return float(10 * np.log10(np.mean(W[heard])))


# --- what every member pass is known to have sent --------------------------------------------

def known_header(store: Store, acc, segment: int) -> header.HeaderFields | None:
    """The header the accumulator's passes of `segment` were sent with, or None.

    From a member's own decoded header if any matches the key, else from
    the members' header LLRs summed per segment (the soft header), else
    None. The segment field is set to `segment`.
    """
    key = (acc.key[0], int(acc.key[1]))
    found = None
    llr_by_seg: dict[int, np.ndarray] = {}
    for m in acc.members:
        info = store.pass_info(m["uid"])
        h = info["header"]
        if h is not None and (h["callsign"], int(h["picture_id"])) == key:
            found = header.HeaderFields(**h)
            break
        llr = np.asarray(info["hdr_llr"], dtype=np.float64)
        if llr.size == N_HDR_BITS and np.any(llr != 0):       # a frame with a header
            g = int(m["segment"])
            llr_by_seg[g] = llr_by_seg.get(g, 0.0) + llr
    if found is None:
        for llr in llr_by_seg.values():
            h = header.decode(np.asarray(llr, dtype=np.float64))
            if h is not None and (h.callsign, int(h.picture_id)) == key:
                found = h
                break
    if found is None:
        return None
    if not 0 <= int(segment) <= found.mode:
        return None
    return dataclasses.replace(found, segment=int(segment))


def known_keying(store: Store, acc, call: str | None):
    """(keying, ook) of the callsign windows if a member's windows matched the call, else None."""
    if call is None:
        return None
    try:
        check_callsign(call)
    except ValueError:
        return None
    votes = {"fsk": 0, "ook": 0}
    for m in acc.members:
        p = store.load_pass(m["uid"])
        if p.cw is not None and p.cw.agrees and p.cw.z_match > Z_ACCEPT:
            votes[p.cw.keying] = votes.get(p.cw.keying, 0) + 1
    if votes["fsk"] == 0 and votes["ook"] == 0:
        return None
    return keying_units(call), votes["ook"] > votes["fsk"]


# --- one re-receive -----------------------------------------------------------------------------

def _psi_frame(tr) -> np.ndarray:
    return tr.psi[tr.gi(np.arange(tr.spec.n_pos))].astype(np.float32)


def pass_path(p: PassResult, chan=None) -> FreqPath:
    """A frequency path for re-receiving pass p, measured on its own capture.

    A pass keeps its frequency and timing but not the path it was tracked
    on, and the straight line in its report can be far off (a path that
    ran away in the frame's last segment reads as drift). So the path is
    measured again: the best straight line over the whole frame
    (`track.line_path`) and that line refined segment by segment
    (`track.refine_path`), whichever gives the known symbols the larger
    Z_ref with the pass's own timing (known symbols withheld from the
    tracker, so the choice does not look at the data).
    """
    spec = frame.get(p.frame)
    det = receiver.detection_from_pass(p)
    if chan is None:
        chan = receiver.channel_for(receiver.capture_from_pass(p), spec, det,
                                    f_mix_hz=p.f_mix_hz)
    line = track.line_path(chan, spec, _flat(p.f_hz))
    best, best_z = line, -np.inf
    for pth in (line, track.refine_path(chan, spec, line)):
        tr = track.verify_track(chan, spec, dataclasses.replace(det, path=pth), fit=False)
        if tr is not None and tr.z_ref > best_z:
            best, best_z = pth, tr.z_ref
    return best


def _path_of(tr) -> FreqPath:
    t = np.arange(float(tr.t_pos_s(tr.pos[:1])[0]) - 10.0,
                  float(tr.t_pos_s(tr.pos[-1:])[0]) + 10.0, 1.0)
    return FreqPath(t_s=t, f_hz=np.asarray(tr.freq(t), dtype=np.float64), weight=np.ones(len(t)))


def rereceive(p: PassResult, prior: EmPrior | None, hdr: header.HeaderFields | None = None,
              cw_known=None, estimator: str | None = None, em_round: int | None = None,
              path: FreqPath | None = None, return_path: bool = False, split: bool = True):
    """Pass p re-tracked and re-demodulated from its stored 250 Hz capture.

    With `hdr` (the header p was sent with) the header symbols are known
    from the start and the tracker runs twice, as `receive_pass`'s rounds
    do (timing refit and re-centring each time); without, it is
    `receiver.receive_pass` itself on the capture. `prior` gives the data
    symbols as soft templates. `path` is the frequency path to start
    from (default `pass_path`). The result keeps p's uid and its
    callsign-window reading, and p's header unless p had none and the
    re-receive decoded one. With return_path, (PassResult, final path).
    """
    spec = frame.get(p.frame)
    est = estimator or p.estimator
    if p.ch is None or len(p.ch) == 0:
        return (None, None) if return_path else None
    prep = receiver.capture_from_pass(p)
    det = receiver.detection_from_pass(p)
    chan = receiver.channel_for(prep, spec, det, f_mix_hz=p.f_mix_hz)
    det = dataclasses.replace(det, path=pass_path(p, chan) if path is None else path)
    rnd = p.em_round if em_round is None else em_round
    if hdr is None or not spec.n_hdr:
        new, trs = receiver.receive_pass(prep, spec, det, prior=prior, estimator=est,
                                         f_mix_hz=p.f_mix_hz, return_tracks=True)
        hdr_out = p.header if p.header is not None else new.header
        out = dataclasses.replace(new, uid=p.uid, header=hdr_out, cw=p.cw, em_round=rnd)
        return (out, _path_of(trs[-1])) if return_path else out
    hdr_bits = header.encode(hdr)
    classes = track.make_classes(spec, hdr_bits, prior)
    # as receive_pass's two rounds: the second starts from the first's
    # timing and re-centred frequency path
    tr = track.track(chan, spec, det, classes, cw_known=cw_known, timing=p.timing)
    if prior is None or not split:
        tr = track.track(chan, spec, det, classes, cw_known=cw_known, timing=tr.timing,
                         freq=tr.freq, outer=1)
        tr.z_ref = p.report.z_ref
        z, w, llr, diag = demod.extract(None, tr, spec, p.q, est)
    else:
        # Block split: the symbols of a precoder block are demodulated by a
        # track whose templates hold no reference for that block (only the
        # other parity's blocks), so the reference's errors on a latent
        # cannot steer the gain and phase pass i is read with there.
        # Without it EM raises each pass's W while correlating the passes'
        # errors, and the accumulator loses (measured: 2.89 -> 2.13 dB
        # effective, error correlation 0.010 -> 0.058).
        par = block_index(spec.n_data) % 2
        z = np.zeros(spec.n_data, dtype=np.float32)
        w = np.zeros(spec.n_data, dtype=np.float32)
        first = tr
        for k in (0, 1):
            own = par == k
            pk = EmPrior(x_hat=np.where(own, 0.0, prior.x_hat), v=np.where(own, 1.0, prior.v))
            ck = track.make_classes(spec, hdr_bits, pk)
            trk = track.track(chan, spec, det, ck, cw_known=cw_known, timing=first.timing,
                              freq=first.freq, outer=1)
            trk.z_ref = p.report.z_ref
            zk, wk, llrk, dk = demod.extract(None, trk, spec, p.q, est)
            z[own], w[own] = zk[own], wk[own]
            if k == 0:
                tr, llr, diag = trk, llrk, dk
    rep = track.report(tr, diag["kappa"], diag["suspect"])
    out = dataclasses.replace(
        p, f_hz=float(tr.freq(0.0)), timing=tr.timing, report=rep, z=z, w=w, hdr_llr=llr,
        psi=_psi_frame(tr), estimator=est, em_round=rnd)
    return (out, _path_of(tr)) if return_path else out


# --- the EM loop ------------------------------------------------------------------------------

def em_refine(store: Store, key, rounds: int = EM_ROUNDS, tol_db: float = EM_TOL_DB, *,
              estimator: str | None = None, use_header: bool = True, log=None) -> list[float]:
    """Leave-one-out EM over accumulator `key`'s members (design 8.3).

    Returns the accumulator's mean W in dB before EM and after each round
    (so len = rounds run + 1). Members without a stored capture (merged
    or synthetic passes) take part in every reference but are not
    re-received. `log`, if given, is called with one line per pass.

    The history is W as the passes claim it. A re-receive raises a
    pass's claimed W by up to about twice its gain against the truth
    (measured on a FULL pass: W +0.53 dB, LMMSE SNR against the truth
    +0.19 dB): the known header joins the references the pass's kappa is
    calibrated on, and those are residuals of symbols the tracker fitted.
    """
    key = (key[0], int(key[1]))
    acc = store.accumulator(key)
    history = [acc_mean_w_db(acc)]
    hdrs: dict[int, header.HeaderFields | None] = {}
    cw_known = None
    if use_header:
        for g in range(acc.mode + 1):
            hdrs[g] = known_header(store, acc, g)
        h0 = next((h for h in hdrs.values() if h is not None), None)
        cw_known = known_keying(store, acc, h0.callsign.rstrip(" ") if h0 else None)
    paths: dict[str, FreqPath | None] = {}
    for r in range(1, int(rounds) + 1):
        for uid in list(acc.uids):
            p = store.load_pass(uid)
            if p.ch is None or len(p.ch) == 0:
                continue
            spec = frame.get(p.frame)
            m = acc.member(uid)
            prior = em_prior(acc, uid, store, p.q, spec)
            new, paths[uid] = rereceive(p, prior, hdrs.get(int(m["segment"])), cw_known,
                                        estimator, r, path=paths.get(uid), return_path=True)
            if new is None:
                continue
            store.update_pass(new)
            acc = store.accumulator(key)
            if log is not None:
                log(f"round {r} pass {uid}: mean W {receiver.mean_w_db(p):+.2f} -> "
                    f"{receiver.mean_w_db(new):+.2f} dB")
        history.append(acc_mean_w_db(acc))
        if history[-1] - history[-2] < tol_db:
            break
    return history


# --- template search ----------------------------------------------------------------------------

def _gauss_z(log_p: float) -> float:
    """The normal Z with upper tail exp(log_p)."""
    if log_p >= math.log(0.5):
        return 0.0
    if log_p > -700.0:
        return float(stats.norm.isf(math.exp(log_p)))
    # Mills ratio, to well under 0.01 in Z this far out
    z = math.sqrt(-2.0 * log_p)
    for _ in range(4):
        z = math.sqrt(max(-2.0 * log_p - 2.0 * math.log(z * math.sqrt(2 * math.pi)), 1.0))
    return z


def template_z(lam: float, n_chunks: int, n_cells: float) -> float:
    """Z of a template-search peak: Gamma(K, 1) tail, Bonferroni over n_cells."""
    log_p = float(stats.gamma.logsf(lam, n_chunks))
    if not np.isfinite(log_p):
        # far out the tail: log of the leading term x^(K-1) e^-x / Gamma(K)
        log_p = (n_chunks - 1) * math.log(lam) - lam - math.lgamma(n_chunks)
    log_p += math.log(max(n_cells, 1.0))
    return _gauss_z(min(log_p, 0.0))


def _candidates(store: Store, acc, extra_hz=()):
    """(f_hz, drift Hz/min, ppm) hypotheses from the members' own passes.

    A station's passes of one picture usually share a frequency and a
    clock; the search covers TS_DF_HZ about each frequency, and each
    member's drift and clock error.
    """
    out = []
    for m in acc.members:
        try:
            p = store.load_pass(m["uid"])
        except (OSError, KeyError):
            continue
        out.append((float(p.report.offset_hz), float(p.report.drift_hz_per_min),
                    float(p.timing.ppm)))
    for f in extra_hz:
        out.append((float(f), 0.0, 0.0))
    uniq = []
    for c in out:
        if not any(abs(c[0] - u[0]) < 0.25 and abs(c[1] - u[1]) < 0.05 and abs(c[2] - u[2]) < 1
                   for u in uniq):
            uniq.append(c)
    return uniq


@dataclasses.dataclass
class _Search:
    lam: float = -np.inf
    seg: int = 0
    f_hz: float = 0.0
    drift: float = 0.0
    ppm: float = 0.0
    tau0: float = 0.0
    cells: float = 0.0
    n_chunks: int = 0


def _chunk_template(spec: FrameSpec, mu, nu, keying, ook, t0_index: float, ppm: float,
                    n_ch: int, chunk: int):
    """(starts, chunks, energies): the mean waveform on the CH grid in chunks.

    Sample n of the channel is at tau = (n - t0_index)/(T fs (1 + ppm)) symbols
    (timing offset 0: the search's lags carry it). Chunks wholly outside
    the keyed span are dropped.
    """
    scale = T_SYM * CH_FS * (1.0 + ppm * 1e-6)
    lo = int(math.floor(t0_index + spec.keyed_start_pos * scale))
    hi = int(math.ceil(t0_index + spec.keyed_end_pos * scale))
    n = np.arange(max(lo, 0), min(hi, n_ch))
    tau = (n - t0_index) / scale
    e = ce.mean_template(tau, mu, nu, spec, keying, ook)
    n_k = len(n) // chunk
    starts = n[0] + chunk * np.arange(n_k)
    E = e[:n_k * chunk].reshape(n_k, chunk)
    en = np.sum(np.abs(E) ** 2, axis=1)
    ok = en > 1e-6 * max(float(en.max(initial=0.0)), 1e-300)
    return starts[ok], E[ok], en[ok]


def _search_chan(z, N0: float, starts, E, en, lag_max: int, m_max: int, drift_bins):
    """max over (lag, freq bin) of Lambda; returns (lam, lag, m)."""
    chunk = E.shape[1]
    nfft = TS_NFFT
    while nfft < chunk + 2 * lag_max:
        nfft *= 2
    K = len(starts)
    idx = starts[:, None] - lag_max + np.arange(chunk + 2 * lag_max)[None, :]
    ok = (idx >= 0) & (idx < len(z))
    R = np.where(ok, z[np.clip(idx, 0, len(z) - 1)], 0.0)
    FR = np.fft.fft(R, nfft, axis=1)
    FT = np.fft.fft(E, nfft, axis=1)
    norm = 1.0 / (N0 * en)
    best = (-np.inf, 0, 0)
    for m in range(-m_max, m_max + 1):
        lam = np.zeros(2 * lag_max + 1)
        mk = m + np.asarray(drift_bins, dtype=np.int64)
        # a template shifted up by mk bins: its spectrum rolled by mk
        FTm = FT[np.arange(K)[:, None], (np.arange(nfft)[None, :] - mk[:, None]) % nfft]
        C = np.fft.ifft(FR * np.conj(FTm), axis=1)[:, :2 * lag_max + 1]
        lam = np.sum(np.abs(C) ** 2 * norm[:, None], axis=0)
        j = int(np.argmax(lam))
        if lam[j] > best[0]:
            best = (float(lam[j]), j - lag_max, m)
    return best


def _search_input(cap, f_mix_hz):
    """(Prepared, f_mix or None) of a template-search input.

    A CH-rate capture is only meaningful with the frequency it was mixed
    at: a stored pass carries it (`PassResult.f_mix_hz`), any other
    250 Hz capture must be given it. Guessing it from a candidate's
    frequency puts the search off by the true mix's distance from that
    guess (a drifting pass is mixed at its mid-frame frequency).
    """
    if isinstance(cap, PassResult):
        if cap.ch is None or len(cap.ch) == 0:
            raise ValueError(f"pass {cap.uid} keeps no capture")
        return receiver.capture_from_pass(cap), Fraction(cap.f_mix_hz)
    prep = receiver.prepare(cap)
    if prep.fs == CH_FS:
        if f_mix_hz is None:
            raise ValueError("a 250 Hz capture needs the frequency it was mixed at (f_mix_hz)")
        return prep, Fraction(f_mix_hz)
    return prep, None


def _from_passband(cap, q, spec: FrameSpec):
    """A `frontend.PassbandStore` plus slot q -> that slot's Capture; anything else as is."""
    from .frontend import PassbandStore

    if not isinstance(cap, PassbandStore):
        return cap
    if q is None:
        raise ValueError("searching a PassbandStore needs the slot: pass q= "
                         "(or use em.retro_detect to search every stored slot)")
    c, _ = cap.capture(int(q), spec.keyed_end_pos * T_SYM)
    if c is None:
        raise ValueError(f"the passband store holds nothing for slot q={int(q)}")
    return c


def template_search(store: Store, key, cap, spec: FrameSpec, *, freqs=(), segments=None,
                    tau_s: float = TS_TAU_S, df_hz: float = TS_DF_HZ,
                    ppm_steps: int = TS_PPM_STEPS, z_accept: float = Z_ACCEPT,
                    f_mix_hz=None, q=None, return_stats: bool = False):
    """Search a stored slot capture for a pass of picture `key` (design 8.3).

    `cap` is a `frontend.Capture` (FE, raw: blanked and normalised here),
    a `frontend.PassbandStore` with the slot `q` to read from it, a
    `receiver.Prepared`, or a stored `PassResult` (its 250 Hz capture
    and mix). A 250 Hz Capture or Prepared needs `f_mix_hz`, the mix it
    was made with (Hz inside FE, as `PassResult.f_mix_hz`). The
    hypotheses are the members' frequencies, drifts and clocks (plus
    `freqs`, audio Hz) +- df_hz, timing within +- tau_s of t0, each
    segment with data (or `segments`). Returns a Detection (method
    "template", z_ref the Bonferroni-corrected Z) if Z > z_accept, else
    None; with return_stats, (Detection or None, Z, info).
    """
    key = (key[0], int(key[1]))
    acc = store.accumulator(key)
    prep, f_mix = _search_input(_from_passband(cap, q, spec), f_mix_hz)
    q = int(prep.q)
    segs = [g for g in range(acc.mode + 1) if acc.has_data(g)] if segments is None else list(segments)
    cands = _candidates(store, acc, freqs)
    if not segs or not cands:
        return (None, 0.0, {}) if return_stats else None
    h0 = known_header(store, acc, segs[0])
    cwk = known_keying(store, acc, h0.callsign.rstrip(" ") if h0 else None)
    keying, ook = (cwk if cwk is not None else
                   ((keying_units(h0.callsign.rstrip(" ")), False) if h0 is not None
                    and spec.n_win else (None, False)))
    chunk = int(round(TS_CHUNK_S * CH_FS))
    lag_max = int(math.ceil(tau_s * CH_FS))
    m_max = int(math.ceil(df_hz * TS_NFFT / CH_FS))
    dur = spec.keyed_end_pos * T_SYM
    ppm_step = TS_SLIP_SAMPLES / (dur * CH_FS) * 1e6
    best = _Search()
    n_cells = 0.0
    for g in segs:
        n = spec.n_data
        idx = canonical_index(g, n)
        prior = prior_from(acc.S[g, idx], acc.W[g, idx], q)
        h = known_header(store, acc, g)
        bits = header.encode(h) if (h is not None and spec.n_hdr) else None
        cl = track.make_classes(spec, bits, prior)
        for f0, drift, ppm0 in cands:
            if prep.fs == CH_FS:
                chan = receiver.channel_for(prep, spec, Detection(
                    f_hz=f0, path=_flat(f0), timing=None, z_ref=np.nan, method="template",
                    lead_in_s=0.0), f_mix_hz=f_mix)
                if abs(f0 - chan.mix_audio_hz) + df_hz > TS_CH_REACH_HZ:
                    continue                 # this candidate is not inside the capture
            else:
                lo = -(LEAD_IN_MAX_S + SPAN * T_SYM) - tau_s - 1.0
                hi = dur * (1 + 3e-4) + tau_s + 1.0
                chan = track.make_chan(prep.fe, prep.keep, prep.t0_index, prep.fs,
                                       f0, lo, hi)
            z = np.asarray(chan.ch, dtype=np.complex128)
            if drift:
                t = chan.t_s(np.arange(len(z)))
                turns = (drift / 60.0) * t * t / 2.0
                z = z * np.exp(-2j * np.pi * (turns - np.floor(turns)))
            N0 = max(chan.noise_floor() * CH_FS, 1e-300)
            f_ch = f0 - chan.mix_audio_hz
            mix = np.exp(-2j * np.pi * ((f_ch * chan.t_s(np.arange(len(z)))) % 1.0))
            z = z * mix
            for k in range(-int(ppm_steps), int(ppm_steps) + 1):
                ppm = ppm0 + k * ppm_step
                starts, E, en = _chunk_template(spec, cl.mu, cl.nu, keying, ook,
                                                chan.t0_index, ppm, len(z), chunk)
                if len(starts) == 0:
                    continue
                lam, lag, m = _search_chan(z, N0, starts, E, en, lag_max, m_max,
                                           np.zeros(len(starts), dtype=np.int64))
                n_cells += (2 * lag_max + 1) * (2 * m_max + 1)
                if lam > best.lam:
                    best = _Search(lam=lam, seg=g, f_hz=f0 + m * CH_FS / TS_NFFT, drift=drift,
                                   ppm=ppm, tau0=float(lag), cells=0.0, n_chunks=len(starts))
    if not np.isfinite(best.lam):
        return (None, 0.0, {}) if return_stats else None
    Z = template_z(best.lam, best.n_chunks, n_cells)
    info = dict(lam=best.lam, chunks=best.n_chunks, cells=n_cells, segment=best.seg,
                f_hz=best.f_hz, ppm=best.ppm, tau0=best.tau0, drift=best.drift)
    det = None
    if Z > z_accept:
        t = np.array([-LEAD_IN_MAX_S - 10.0, 0.0, dur + 10.0])
        path = FreqPath(t_s=t, f_hz=best.f_hz + best.drift * t / 60.0, weight=np.ones(3))
        tm = Timing(tau0=best.tau0, ppm=best.ppm, gamma=0.0, z=Z,
                    cov=np.diag([1.0, 1.0, 0.0]))
        det = Detection(f_hz=best.f_hz, path=path, timing=tm, z_ref=Z, method="template",
                        lead_in_s=0.0)
    return (det, Z, info) if return_stats else det


def _flat(f_hz: float) -> FreqPath:
    t = np.array([-20.0, 0.0, 1800.0])
    return FreqPath(t_s=t, f_hz=np.full(3, float(f_hz)), weight=np.ones(3))


def receive_template(store: Store, key, cap, spec: FrameSpec, det: Detection | None = None,
                     estimator: str = "joint", segment: int | None = None,
                     **search) -> PassResult | None:
    """Retroactive detection: template-search a capture and receive what it finds.

    `cap` as `template_search` (a 250 Hz capture with `f_mix_hz` among
    the search arguments, a PassbandStore with `q`, or a stored pass).
    With `det` given, `segment` is the segment it was found as (default
    0; the search's own when it runs here).

    The pass is tracked with the accumulator's soft data as its template
    (the accumulator does not hold this pass, so this is its leave-one-out
    reference) and the picture's header known. It is not associated:
    pass the result to `associate.associate`.
    """
    key = (key[0], int(key[1]))
    cap = _from_passband(cap, search.pop("q", None), spec)
    prep, f_mix = _search_input(cap, search.pop("f_mix_hz", None))
    if det is None:
        det, _, info = template_search(store, key, prep, spec, return_stats=True,
                                       f_mix_hz=f_mix, **search)
        if det is None:
            return None
        seg = info["segment"]
    else:
        seg = 0 if segment is None else int(segment)
    acc = store.accumulator(key)
    idx = canonical_index(seg, spec.n_data)
    prior = prior_from(acc.S[seg, idx], acc.W[seg, idx], prep.q)
    h = known_header(store, acc, seg)
    chan = receiver.channel_for(prep, spec, det, f_mix_hz=f_mix)
    if h is not None and spec.n_hdr:
        classes = track.make_classes(spec, header.encode(h), prior)
        cwk = known_keying(store, acc, h.callsign.rstrip(" "))
        tr = track.track(chan, spec, det, classes, cw_known=cwk, timing=det.timing)
    else:
        tr = track.track(chan, spec, det, track.make_classes(spec, prior=prior),
                         timing=det.timing)
    tr.z_ref = det.z_ref
    z, w, llr, diag = demod.extract(None, tr, spec, prep.q, estimator)
    rep = track.report(tr, diag["kappa"], diag["suspect"])
    f_hz = float(tr.freq(0.0))
    from .types import pass_uid
    return PassResult(
        uid=pass_uid(prep.q, f_hz, z), q=int(prep.q), frame=spec.name, waveform=0,
        f_hz=f_hz, timing=tr.timing, report=rep, z=z, w=w, hdr_llr=llr,
        header=header.decode(llr.astype(np.float64)) if spec.n_hdr else None, cw=None,
        ch=chan.ch.astype(np.complex64), ch_fs=CH_FS, ch_t0_index=float(chan.t0_index),
        f_mix_hz=Fraction(chan.f_mix), psi=_psi_frame(tr), estimator=estimator)


RETRO_MIN_COVERAGE = 0.9         # a stored slot is searched once this much of its frame is on disk
RETRO_SAME_HZ = 1.0              # a find this close to a stored pass of its slot is that pass


def retro_detect(store: Store, key, pb, spec: FrameSpec, *, slots=None,
                 estimator: str = "joint", log=None, **search) -> list[tuple]:
    """Retroactive detection (spec 7 and 8): search the 48 h passband store.

    Every slot of `pb` (a `frontend.PassbandStore`; or just `slots`)
    whose frame is at least RETRO_MIN_COVERAGE on disk and that holds
    no member of picture `key` is template-searched with the
    accumulator's current soft data. What is found is received
    (`receive_template`, the accumulator as its reference and the header
    known) and associated (`associate.associate`: the search finds
    passes, association decides whose they are). A find within
    RETRO_SAME_HZ of a pass already stored for that slot is that pass and
    is skipped. Returns [(PassResult, key or None, rule), ...].
    """
    from .associate import associate

    key = (key[0], int(key[1]))
    say = log or (lambda *_: None)
    acc = store.accumulator(key)
    dur = spec.keyed_end_pos * T_SYM
    held = {store.pass_info(u)["q"] for u in acc.uids}
    stored = [(i["q"], i["f_hz"]) for i in map(store.pass_info, store.pass_uids())]
    out = []
    for q in (pb.slots(dur) if slots is None else slots):
        q = int(q)
        if q in held:
            continue
        cap, cov = pb.capture(q, dur)
        if cap is None or cov < RETRO_MIN_COVERAGE:
            continue
        det, Z, info = template_search(store, key, cap, spec, return_stats=True, **search)
        if det is None:
            say(f"slot q={q}: nothing (Z {Z:.1f})")
            continue
        if any(qq == q and abs(f - det.f_hz) < RETRO_SAME_HZ for qq, f in stored):
            say(f"slot q={q}: Z {Z:.1f} at {det.f_hz:.3f} Hz, already stored")
            continue
        p = receive_template(store, key, cap, spec, det=det, estimator=estimator,
                             segment=info["segment"])
        if p is None:
            continue
        k, rule = associate(p, store)
        stored.append((q, p.f_hz))
        say(f"slot q={q}: Z {Z:.1f} at {p.f_hz:.3f} Hz, segment {info['segment']}, "
            + (f"attached to {k[0]} {k[1]:08x} by {rule or 'existing membership'}"
               if k is not None else "stored provisional"))
        out.append((p, k, rule))
    return out


__all__ = ["em_prior", "em_refine", "prior_from", "lmmse_latents", "acc_mean_w_db",
           "known_header", "known_keying", "rereceive", "template_search", "template_z",
           "receive_template", "retro_detect"]
