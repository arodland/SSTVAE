"""Acquisition: preamble detection, timing, and carrier frequency offset.

The preamble is periodic with M samples, so an autocorrelation at lag M
gives detection plus a fractional CFO estimate that is unambiguous over
+/- FS/(2M) = +/-25 Hz. The remaining offset is a multiple of the 50 Hz
carrier spacing, resolved by trying integer-bin candidates against the
known preamble template. Net tolerance comfortably exceeds +/-50 Hz.

`max_bins` is that tolerance and is the *only* thing setting it out to
about +/-700 Hz, where the sync lowpass finally takes over: detection
itself is CFO-blind (an offset multiplies every lag-M product by one
constant phasor, which |.| removes), so a mis-tuned signal fails here
with "header decode failed" rather than "no preamble found". Raising it
costs ~0.14 ms per extra candidate and, measured, returns bit-identical
answers for every signal the narrow search already acquires -- see
docs/todo.md, "Wider acquisition search, for a mis-tuned counterpart".
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.fft import next_fast_len, fft, ifft

from ..config import (
    FS,
    M,
    FRAME_SAMPLES,
    PREAMBLE_CORR_WINDOW,
    PREAMBLE_CP,
    PREAMBLE_REPEATS,
    PREAMBLE_SAMPLES,
    PREAMBLE_THRESHOLD,
)
from .dsp import freq_correct, sync_lowpass
from .ofdm import preamble_template, pilot_template


class SyncError(Exception):
    pass


@dataclass
class Acquisition:
    preamble_start: int  # index of first preamble sample (CP start)
    freq_offset: float  # Hz
    metric: float  # detection confidence, ~0..1


def _autocorr_metric(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sliding lag-M autocorrelation over a PREAMBLE_CORR_WINDOW window.

    The window, not the preamble's length, is what sets this metric's
    noise floor: it must be widened along with the preamble or the
    extra repeats buy nothing at all. See config.PREAMBLE_REPEATS.
    """
    W = PREAMBLE_CORR_WINDOW
    prod = z[M:] * np.conj(z[:-M])
    power = np.abs(z) ** 2
    kernel = np.ones(W)
    a = signal.fftconvolve(prod, kernel, mode="valid")  # A[n] over window n..n+W
    e1 = signal.fftconvolve(power[:-M], kernel, mode="valid")
    e2 = signal.fftconvolve(power[M:], kernel, mode="valid")
    # Floor the energies at a fraction of the typical window energy so
    # near-silent regions (filter ringing) can't produce inflated metrics.
    floor = 1e-3 * W * np.mean(power)
    energy = np.sqrt(np.maximum(e1, floor) * np.maximum(e2, floor)) + 1e-12
    return np.abs(a) / energy, a


def acquire(
    z: np.ndarray,
    threshold: float = PREAMBLE_THRESHOLD,
    max_bins: int = 2,
    search: tuple[int, int] | None = None,
) -> Acquisition:
    """Find the preamble in baseband signal z.

    `search` optionally restricts the preamble hunt to a [start, end)
    sample range (the rest of the signal is still used for frames).
    """
    if len(z) < PREAMBLE_SAMPLES + 2 * M:
        raise SyncError("signal too short")

    z = sync_lowpass(z)
    metric, a = _autocorr_metric(z)
    if search is not None:
        s0 = max(0, int(search[0]))
        s1 = min(len(metric), int(search[1]))
        if s1 - s0 < 1:
            raise SyncError(f"empty search window {search}")
        masked = np.full_like(metric, -1.0)
        masked[s0:s1] = metric[s0:s1]
        metric = masked
    n_star = int(np.argmax(metric))
    if metric[n_star] < threshold:
        raise SyncError(f"no preamble found (peak metric {metric[n_star]:.2f})")

    f_frac = np.angle(a[n_star]) / (2 * np.pi * M / FS)

    # Integer-bin CFO search + fine timing via template correlation.
    template = preamble_template()
    t_norm = np.sqrt(np.sum(np.abs(template) ** 2))
    lo = max(0, n_star - PREAMBLE_CP - 200)
    hi = min(len(z) - PREAMBLE_SAMPLES, n_star + 200)
    if hi <= lo:
        raise SyncError("preamble at signal edge")
    seg = z[lo : hi + PREAMBLE_SAMPLES]

    best = None
    for m_bin in range(-max_bins, max_bins + 1):
        f_cand = f_frac + m_bin * FS / M
        seg_c = freq_correct(seg, f_cand)
        corr = signal.fftconvolve(seg_c, np.conj(template[::-1]), mode="valid")
        peak = int(np.argmax(np.abs(corr)))
        seg_energy = np.sqrt(
            np.sum(np.abs(seg_c[peak : peak + PREAMBLE_SAMPLES]) ** 2)
        )
        score = np.abs(corr[peak]) / (t_norm * seg_energy + 1e-12)
        if best is None or score > best[0]:
            best = (score, lo + peak, f_cand)

    _, p0, f_hat = best

    # Refine CFO from the phase between successive preamble periods at
    # the now-known timing (same lag-M estimate, but noise-averaged at
    # the exact alignment, over every repeat rather than one pair).
    n_pre = PREAMBLE_REPEATS * M
    u0 = p0 + PREAMBLE_CP
    zc = freq_correct(z[u0 : u0 + n_pre], f_hat)
    if len(zc) == n_pre:
        d = np.sum(zc[M:] * np.conj(zc[:-M]))
        if np.abs(d) > 0:
            f_hat += np.angle(d) / (2 * np.pi * M / FS)

    return Acquisition(preamble_start=p0, freq_offset=f_hat, metric=float(metric[n_star]))


@dataclass
class BlindAcquisition:
    frame_start: int  # sample index of some pilot symbol's useful-window start
    freq_offset: float  # Hz
    metric: float  # relative confidence, not directly comparable to Acquisition.metric


def acquire_blind(
    z: np.ndarray,
    max_offset_hz: float = 55.0,
    bin_step_hz: float = 1.7,
    min_periods: int = 8,
    threshold: float = 4.0,
    search: tuple[int, int] | None = None,
) -> BlindAcquisition:
    """Recover frame-boundary timing and carrier frequency purely from the
    frame pilot's own periodicity (repeats every FRAME_SAMPLES), with NO
    dependence on the transmission-start preamble — the preamble is sent
    once and doesn't recur, so it's useless for a recording that starts
    mid-transmission. This is the mechanism that lets a receiver recover
    position (and, combined with beacon.decode, the absolute frame index
    and callsign) from any long-enough stretch of audio, including audio
    recorded before the receiver "noticed" the signal.

    For each candidate CFO bin, this matched-filters the whole window
    against one bare pilot symbol and folds the matched-filter energy
    into FRAME_SAMPLES-periodic phase bins, integrating across every
    period available — the periodic-pilot analogue of the preamble's
    single-shot correlation, needed because unlike the preamble the
    pilot symbol is only ~1/6 of each frame, not the whole thing.
    `score` is the winning phase's prominence over the other 1151 phase
    bins (peak / median), not an absolute SNR-like quantity — scale
    invariant, so `threshold` doesn't need retuning per signal level.

    Searching many CFO candidates against the same segment is a
    Doppler-search matched filter, so instead of re-modulating the
    (long) segment and re-running a fresh FFT convolution for every
    candidate bin (each one recomputing the *template's* FFT too, even
    though it's fixed), this takes a single FFT of the unmodulated
    segment and, per candidate, applies a circular shift to its
    spectrum instead — time-domain modulation by f is exactly a shift
    of the DFT by f/bin_hz bins, so this produces the same matched-
    filter magnitude (up to <0.1 Hz quantization to the nearest FFT
    bin, negligible next to bin_step_hz) for a small fraction of the
    FFT work.
    """
    template = pilot_template()
    kernel = np.conj(template[::-1])

    seg = z if search is None else z[search[0] : search[1]]
    if len(seg) < FRAME_SAMPLES * min_periods:
        raise SyncError("window too short for blind acquisition")

    n_bins = int(np.ceil(max_offset_hz / bin_step_hz))
    n_fft = next_fast_len(len(seg) + len(kernel) - 1)
    bin_hz = FS / n_fft
    lo = len(kernel) - 1
    valid_len = len(seg) - len(kernel) + 1

    Sf = fft(seg, n_fft)
    Tf = fft(kernel, n_fft)

    best = None
    for k in range(-n_bins, n_bins + 1):
        shift_bins = int(round(k * bin_step_hz / bin_hz))
        f_cand = shift_bins * bin_hz
        mf = ifft(np.roll(Sf, -shift_bins) * Tf)[lo : lo + valid_len]
        p2 = np.abs(mf) ** 2
        n_periods = len(p2) // FRAME_SAMPLES
        if n_periods < min_periods:
            continue
        folded = p2[: n_periods * FRAME_SAMPLES].reshape(n_periods, FRAME_SAMPLES).sum(axis=0)
        phase = int(np.argmax(folded))
        score = folded[phase] / (np.median(folded) + 1e-12)
        if best is None or score > best[0]:
            best = (score, phase, f_cand)

    if best is None:
        raise SyncError("signal too short for blind acquisition at any CFO bin")
    score, phase, f_hat = best
    if score < threshold:
        raise SyncError(f"no periodic pilot found (peak prominence {score:.3g})")

    off = (search[0] if search is not None else 0) + phase
    return BlindAcquisition(frame_start=off, freq_offset=f_hat, metric=float(score))


class BlindAccumulator:
    """Incremental counterpart to `acquire_blind()`.

    `rx/engine.py` used to bound the one-shot function's search window
    purely for CPU reasons: it recomputes the whole window's per-CFO-bin
    FFT from scratch on every poll, even though most of a sliding window
    is audio it already folded last poll. This class instead keeps a
    running per-CFO-bin, per-phase-bin energy accumulator and folds in
    only the *new* samples each call, via block-wise overlap-save -- so
    the cost of `push()` is O(new samples), not O(window length).

    That does **not** mean longer integration lets this pull a weaker
    signal out of the noise, and it is worth being precise about why,
    because the intuitive "more integration = deeper into the noise"
    story is wrong for this detector specifically. `result()`'s score is
    a ratio: the winning phase bin's summed matched-filter *power* over
    the other 1151 bins' median. Both numerator and denominator grow
    roughly proportionally with the number of periods folded in, so the
    ratio converges to a value set by the signal's *per-period* SNR, not
    by how many periods you integrate -- confirmed by measurement
    (`docs/todo.md`): at a fading SNR comfortably above the algorithm's
    floor, score barely moves between a 10 s and a 95 s one-shot window;
    well below the floor, no window length up to and including the
    entire transmission ever crosses threshold. What longer integration
    *does* buy is reliability near that floor: a signal whose true ratio
    sits just above threshold can read below it on a short, noisy sample
    and get missed, and a longer window converges to the true ratio and
    catches it -- measured, one seed in eight went from a miss at a
    10 s window to a catch at 95 s, at an SNR where every other seed
    already passed at 10 s. That benefit runs out once the window covers
    the transmission's own duration; audio older than the transmission
    is pure dilution, not more signal to integrate (see `window_s`
    below).

    The block size is chosen purely from the frequency resolution the
    search needs (`FS / bin_step_hz`), not from how long the caller
    intends to integrate for -- decoupling those two is the entire
    point, and it also means the block's own FFT and the per-bin
    circular-shift table are computed once for the accumulator's whole
    lifetime rather than once per poll.

    Correctness rests on one fact about the circular-shift CFO trick:
    applied per block instead of over one huge segment, it is no longer
    equivalent to continuous-phase demodulation from sample 0 of the
    whole recording -- each block's phase resets to 0 at the block's own
    start, introducing a constant (per block, per CFO bin) phase error
    relative to the "true" correction. But the accumulator only ever
    uses `|matched filter|**2`: multiplying an entire block by a
    constant unit-magnitude phasor before a linear matched filter
    multiplies that block's whole correlation output by the same
    constant, which a squared magnitude removes exactly. So block-local
    and whole-recording CFO correction fold to the same energy. Checked
    against `acquire_blind`'s one-shot result in
    `tests/test_blind_acquisition.py`.

    `window_s` bounds how long old audio keeps influencing the result --
    without it, an ever-growing accumulator is wrong independent of the
    integration-time argument above: each phase bin's fold sums matched-
    filter power over every period pushed, real signal or not, so a
    real (but short) transmission surrounded by a long stretch of
    silence or noise gets its peak diluted by all that irrelevant
    history, and the score drifts toward 1 as the irrelevant fraction
    grows. Implemented as exponential decay (`window_s` is the ~1/e time
    constant) rather than exact eviction of aged-out blocks: eviction
    needs to retain every bin's per-block contribution for the whole
    window to undo it later (67 bins x ~8k samples x 25 blocks, tens of
    MB), while decay needs none -- just scale the existing `folded`
    array down before adding each new block, which is also this
    codebase's established idiom for "recent-weighted, bounded-memory"
    state (see the pilot clock-drift tracker in modem.py).

    Since the two things `window_s` needs to balance -- enough of it to
    get a marginal, long (mode C, ~95 s) transmission's full reliability
    benefit, not so much that a short (mode A, ~32 s) one sits diluted by
    an unrelated 60+ s of history behind it -- depend on which mode is
    transmitting, which is exactly what blind acquisition does not know
    yet, `window_s` accepts **several** timescales run in parallel
    instead of a single one. The expensive part (block FFT + per-bin
    matched filter) is shared and computed once per block regardless of
    how many timescales are given; each just gets its own cheap decay-
    and-fold pass over the same per-block result, so running e.g. three
    timescales costs almost nothing beyond one. `result()` reports
    whichever timescale's peak score is highest. `rx/engine.py` passes
    one timescale per mode, each capped at that mode's own duration
    (`config.MODES`), so no timescale ever integrates past the point
    where there is any more real signal to gain from it. A single float
    (or `None`, meaning no decay at all) still means one timescale, for
    equivalence testing against `acquire_blind`'s unweighted one-shot
    integration.
    """

    def __init__(
        self,
        max_offset_hz: float = 55.0,
        bin_step_hz: float = 1.7,
        min_periods: int = 8,
        threshold: float = 4.0,
        block_samples: int | None = None,
        window_s: float | None | Sequence[float | None] = 25.0,
    ) -> None:
        template = pilot_template()
        kernel = np.conj(template[::-1])
        m = len(kernel)
        self._m = m
        self._min_periods = min_periods
        self._threshold = threshold

        # Large enough that a block's own FFT still resolves
        # bin_step_hz, and well above the kernel so overlap-save stays
        # efficient (a 160-sample kernel against an ~8k block, not
        # against the whole window as acquire_blind's single big FFT
        # does).
        min_block = int(np.ceil(FS / bin_step_hz))
        self._block = next_fast_len(max(min_block, 4 * m))
        self._step = self._block - (m - 1)
        if self._step <= 0:
            raise ValueError("block_samples too small for the pilot kernel")
        if block_samples is not None:
            self._block = block_samples
            self._step = self._block - (m - 1)
        self._bin_hz = FS / self._block

        n_bins = int(np.ceil(max_offset_hz / bin_step_hz))
        shift_bins = np.array(
            [int(round(k * bin_step_hz / self._bin_hz)) for k in range(-n_bins, n_bins + 1)]
        )
        self._shift_bins = shift_bins
        self._freqs = shift_bins * self._bin_hz
        self._kernel_f = fft(np.concatenate([kernel, np.zeros(self._block - m, dtype=complex)]))

        # One or several decay timescales, run in parallel off the same
        # per-block matched-filter result -- see the class docstring for
        # why a single one can't serve every mode well. A bare float or
        # None is still one timescale, for callers (and the equivalence
        # tests) that only want that.
        window_s_list = (
            list(window_s) if isinstance(window_s, Sequence) and not isinstance(window_s, str)
            else [window_s]
        )
        # Applied once per processed block, uniformly across every phase
        # and CFO bin *within one timescale*, so it never changes the
        # *shape* of a fresh block's contribution to that timescale --
        # only how much the past is still worth relative to it. Because
        # it scales a timescale's bins equally, that timescale's peak/
        # median score is unaffected by decay alone; only the mix of
        # old-vs-new evidence behind it changes.
        self._decay_per_block = np.array(
            [1.0 if w is None else float(np.exp(-self._step / (w * FS))) for w in window_s_list]
        )

        self._folded = np.zeros((len(window_s_list), len(shift_bins), FRAME_SAMPLES))
        self._n_valid = 0
        self._buf = np.zeros(0, dtype=complex)
        self._buf_start: int | None = None

    def push(self, z: np.ndarray, start_sample: int) -> None:
        """Fold new complex-baseband samples in. `z[0]` must sit at
        absolute sample index `start_sample`, and pushes must be
        contiguous (no gaps, no re-sent samples) -- the overlap-save
        history this needs is carried internally between calls."""
        if self._buf_start is None:
            self._buf_start = start_sample
        elif start_sample != self._buf_start + self._buf.size:
            raise ValueError(
                "BlindAccumulator.push: expected a contiguous continuation "
                f"at sample {self._buf_start + self._buf.size}, got {start_sample}"
            )
        buf = np.concatenate([self._buf, z]) if self._buf.size else np.asarray(z, dtype=complex)

        B, m, step = self._block, self._m, self._step
        pos = 0
        while pos + B <= buf.size:
            block_f = fft(buf[pos : pos + B])
            # mf[i] (0-indexed within the valid slice) is the matched
            # filter's response for a pilot window starting at block-
            # local index i -- the same convention acquire_blind's p2[j]
            # uses (its "lo = m - 1" offset into the FFT's own index
            # space exactly cancels against the "-(m-1)" in a matched
            # filter's window-start-from-output-index formula, so no
            # (m - 1) belongs here).
            abs0 = self._buf_start + pos
            phase0 = abs0 % FRAME_SAMPLES
            idx = (phase0 + np.arange(step)) % FRAME_SAMPLES
            # Decay every timescale's whole array first -- cheap
            # (n_timescales x n_bins x FRAME_SAMPLES scalars) next to the
            # per-bin FFT below, which is why adding timescales barely
            # moves push()'s cost.
            self._folded *= self._decay_per_block[:, None, None]
            for i, shift in enumerate(self._shift_bins):
                mf = ifft(np.roll(block_f, -shift) * self._kernel_f)[m - 1 : B]
                p2 = np.abs(mf) ** 2
                # Every timescale folds in the *same* per-block matched-
                # filter power -- only the decay each one already applied
                # to its own history differs. Looped rather than
                # vectorized across timescales: np.add.at must run once
                # per target array to handle idx's repeats (step exceeds
                # FRAME_SAMPLES, so each block wraps the phase several
                # times) correctly, and n_timescales is small enough
                # (2-3) that the loop costs nothing next to the FFT.
                for t in range(len(self._decay_per_block)):
                    np.add.at(self._folded[t, i], idx, p2)
            self._n_valid += step
            pos += step

        self._buf = buf[pos:]
        self._buf_start += pos

    def result(self) -> BlindAcquisition:
        """The best (timescale, bin, phase) so far, in the same shape as
        `acquire_blind`'s return value -- reports whichever timescale's
        peak score is highest, not a fixed one, since which timescale
        that is depends on which mode (if any) is actually transmitting.
        Raises `SyncError` exactly as the one-shot function does: too
        little data pushed yet, or no timescale's peak clears
        `threshold`."""
        if self._n_valid < FRAME_SAMPLES * self._min_periods:
            raise SyncError("window too short for blind acquisition")

        # self._folded is (n_timescales, n_bins, FRAME_SAMPLES); reduce
        # the phase axis to a score per (timescale, bin), then pick the
        # best cell over both.
        scores = self._folded.max(axis=2) / (np.median(self._folded, axis=2) + 1e-12)
        t, i = np.unravel_index(np.argmax(scores), scores.shape)
        t, i = int(t), int(i)
        score = float(scores[t, i])
        if score < self._threshold:
            raise SyncError(f"no periodic pilot found (peak prominence {score:.3g})")

        phase = int(np.argmax(self._folded[t, i]))
        return BlindAcquisition(frame_start=phase, freq_offset=float(self._freqs[i]), metric=score)
