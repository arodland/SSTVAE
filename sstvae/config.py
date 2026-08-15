"""Central waveform and latent-layout constants.

Every number that must agree between the modem, the channel simulator,
and (later) the neural network training loop lives here.
"""

from dataclasses import dataclass, field

FS = 8000  # audio sample rate, Hz

# --- OFDM waveform ---------------------------------------------------------
RS = 50  # carrier spacing == symbol rate of one carrier, Hz
NC = 24  # data carriers
M = FS // RS  # useful symbol length, samples (160)
NCP = 32  # cyclic prefix, samples (4 ms)
NSYM = M + NCP  # full symbol, samples (192)

# Carriers sit on integer multiples of RS so the cyclic prefix is truly
# cyclic: passband carrier k is at CARRIER0 + k*RS.
CARRIER0 = 950  # Hz; occupied band ~925..2125 Hz, centered ~1525 Hz
FCENTER = 1500  # Hz; baseband conversion frequency (carriers land on
#                 integer multiples of RS: bin k-11 for carrier k)

# --- framing ---------------------------------------------------------------
# Short frames keep pilots 144 ms apart so interpolated equalization can
# follow ~1 Hz Doppler fading (coherence time ~400 ms). Costs 16.7%
# pilot overhead vs 8.3% for 12-symbol frames.
SYMS_PER_FRAME = 6  # 1 pilot + 5 data
DATA_SYMS_PER_FRAME = SYMS_PER_FRAME - 1

# One data carrier is permanently reserved for the beacon (resync +
# callsign) side-channel on every frame of every mode; the other 23
# carry latents. This is a capacity trade, not a time trade: frame
# duration and FRAMES_PER_GROUP are unchanged (see below), so mode
# durations are identical to before the beacon existed.
BEACON_CARRIER = NC - 1  # carrier index 23
NC_LATENT = NC - 1  # 23 carriers carrying latents
CHIPS_PER_FRAME = DATA_SYMS_PER_FRAME  # 5 beacon BPSK chips/frame

FRAME_SAMPLES = SYMS_PER_FRAME * NSYM  # 1152 samples = 144 ms
LATENTS_PER_FRAME = NC_LATENT * DATA_SYMS_PER_FRAME * 2  # 230 real values

# --- beacon / resync side-channel -------------------------------------
# Barker-13 gives a clean, unambiguous chip-level autocorrelation peak
# to find superframe (and thus absolute frame-counter) phase from any
# contiguous run of frames, without needing the transmission-start
# preamble. BEACON_COUNTER_BITS=10 covers frame indices 0..1023, comfortably
# above mode C's 660 frames.
BEACON_SYNC = (1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1)  # Barker-13
BEACON_COUNTER_BITS = 10
BEACON_CALLSIGN_CHARS = 8
BEACON_CALLSIGN_CHAR_BITS = 6  # 64-symbol alphabet, see beacon.py
BEACON_CALLSIGN_BITS = BEACON_CALLSIGN_CHARS * BEACON_CALLSIGN_CHAR_BITS  # 48
BEACON_CRC_BITS = 16

# Preamble: one OFDM symbol repeated PREAMBLE_REPEATS times with a
# double-length cyclic prefix, so the waveform is periodic with M over
# the whole block.
#
# Four repeats rather than two, because the detector's noise floor is
# set by how many sample pairs it can integrate, and at two repeats it
# sat *inside* the threshold: on AWGN the median 5-second maximum of
# the lag-M metric measured 0.461 against a threshold of 0.50, i.e.
# 0.47 threshold crossings per second of noise. The only thing behind
# it is the header, which admits 3 of 4096 Golay codewords (7.2e-4,
# measured) -- and that gate cannot be tightened without more header
# bits, since 3 valid messages in a 12-bit payload is already its
# floor. Hence a live receiver on a quiet band false-locks and starts
# decoding every few hours. See PREAMBLE_CORR_WINDOW: the length is
# only half the change, and on its own buys nothing.
PREAMBLE_REPEATS = 4
PREAMBLE_CP = 2 * NCP
PREAMBLE_SAMPLES = PREAMBLE_CP + PREAMBLE_REPEATS * M  # 704 (88 ms)

# Samples the lag-M autocorrelation integrates over. With R repeats
# there are (R-1)*M sample pairs M apart that both land inside the
# preamble, so this is what the length actually buys -- a longer
# preamble read through the old one-symbol window has the same noise
# floor and the same false-alarm rate as before.
PREAMBLE_CORR_WINDOW = (PREAMBLE_REPEATS - 1) * M  # 480

# Detection threshold on that metric. A *receiver* parameter, not part
# of the on-air format -- it lives here so the two implementations
# cannot disagree about it, not because a transmitter cares.
#
# Measured at 0.42 with the 480-sample window: 3000 s of AWGN (24M
# metric positions) produced no crossing at all and peaked at 0.358,
# against the two-repeat preamble's 0.47 crossings/s; mode A
# acquisition at -2 dB went from 0.40 to 0.93, at -4 dB from 0.17 to
# 0.35. Both directions at once, which the old preamble could not do:
# raising *its* threshold to 0.64 also cleared the false alarms, and
# cost 0 dB acquisition (0.82 -> 0.53) to do it.
PREAMBLE_THRESHOLD = 0.42

# Half-width of the integer-bin CFO search, in 50 Hz carrier spacings,
# so acquisition covers +-(25 + 50*ACQUIRE_MAX_BINS) Hz. A *receiver*
# parameter like the threshold above, here so the two implementations
# and the parity shims cannot disagree about it.
#
# 12 (+-625 Hz) rather than the original 2 (+-125 Hz), because measured
# it is free in every direction that matters *at the true preamble's own
# location* (2026-08-11, see docs/todo.md). Detection there is CFO-blind
# -- an offset multiplies every lag-M product by one constant phasor,
# which |.| removes -- so the false-alarm rate at that location cannot
# move, and the extra candidates never beat the true one there (max_bins
# 2 vs 12 returned the identical preamble start and CFO in 160/160
# trials from 0 to -4 dB); and a candidate is one FFT convolution over
# an ~1100-sample segment, about 0.14 ms, against a 63 ms detection
# stage. +-625 Hz covers the whole range an SSB filter passes, and
# dsp.sync_lowpass takes over as the limit at about +-700 Hz anyway.
#
# That measurement did *not* cover a different question -- whether a
# wrong LOCATION, elsewhere in a real transmission's own data, can win
# more often with 25 candidates than with 5 -- and the answer turned out
# to be yes (found 2026-08-13 against a real, deliberately mis-tuned
# recording): a genuinely off-frequency signal has real spectral energy
# spread near its own true offset even where there's no preamble, so
# widening the bin search widens the chance one candidate resonates with
# it enough to out-score the others and Golay-decode a plausible (wrong)
# header. See TEMPLATE_SCORE_THRESHOLD, which is the fix -- max_bins
# stays here as the tolerance and is not itself the problem.
ACQUIRE_MAX_BINS = 12

# Minimum accepted quality of `acquire()`'s winning integer-CFO
# candidate: the fraction of the matched-filter template's energy the
# best-scoring candidate actually explains, ~0..1. A second gate,
# independent of PREAMBLE_THRESHOLD above -- that one only checks the
# lag-M autocorrelation *before* any candidate is even tried, and real
# transmission data can clear it too (unlike pure noise, which is what
# PREAMBLE_THRESHOLD was calibrated against). Nothing previously checked
# whether the winning candidate was actually a *good* match, only that
# it was the *best available* one.
#
# Measured (2026-08-13): the winning score at the false lock this fixes
# -- a real, deliberately mis-tuned mode C recording, found ~25 s into
# its own data, nowhere near its actual preamble, decoding a plausible
# but wrong mode C header -- was 0.338. Across ~1400 synthetic trials
# at the sensitivity floor (modes A and C, -4 to -6 dB, no/mpp/mpd
# fading, 0/300/600 Hz offset) the lowest winning score for a candidate
# whose header *did* go on to decode correctly was 0.430. 0.40 sits
# between the two, biased toward the false-lock side rather than the
# midpoint: a rejected weak-but-genuine preamble only costs a fallback
# to blind sync (see docs/todo.md's "safety net" paragraph), while an
# accepted false lock reports a wrong, misleading picture as a
# completed reception -- the worse failure of the two, so the gate
# leans toward rejecting rather than accepting when unsure. One
# concrete false-lock measurement and a synthetic floor are what this
# rests on; revisit if a second real false lock is ever found scoring
# above it.
TEMPLATE_SCORE_THRESHOLD = 0.40

HEADER_SYMS = 2  # identical Golay-coded BPSK symbols
HEADER_SAMPLES = HEADER_SYMS * NSYM  # 384

LEADIN_SAMPLES = 800  # 100 ms of silence before the preamble
LEADOUT_SAMPLES = 800

# --- blind (preamble-free) acquisition -------------------------------------
# Receiver parameters, like the two above.
#
# BLIND_BIN_STEP_HZ is the CFO *search grid*, and it is deliberately far
# coarser than it looks like it should be. For a fixed lag the pilot
# matched filter as a function of CFO is a DTFT of a 160-sample (M)
# sequence, so |matched filter|**2 is exactly band-limited in CFO with
# dual support 2M-1, and is fully determined by samples every
# FS/(2*M) = 25 Hz. The old 1.7 Hz grid was ~15x finer than the signal
# can support -- interpolation done the expensive way. 12.5 Hz is half
# the sampling limit, which measured costs no detection at all down to
# -10 dB (50 Hz, past the limit, collapses); the sub-bin peak is
# recovered by sync.refine_cfo, which is *more* accurate than the old
# grid's raw argmax (0.14-0.62 Hz against 0.56 Hz).
BLIND_BIN_STEP_HZ = 12.5

# What the accumulator's *block* must resolve, which is a different
# question from the search grid and must not be tied to it: the block is
# sized for overlap-save efficiency (a 160-sample kernel against a
# ~4700-sample block), and the shift quantization it buys, FS/block, is
# the finest the grid could ever be. Deriving the block from
# BLIND_BIN_STEP_HZ instead would shrink it to 640 samples and give back
# most of the coarse grid's saving -- measured, 106 ms against 33 ms.
# Kept at the value the pre-2026-08-11 receiver used, so the block, and
# with it the accumulator's overlap-save structure, is unchanged.
BLIND_BLOCK_RES_HZ = 1.7

# Search range. The narrow one is the default and covers ordinary
# mis-tuning; the wide one covers a counterpart whose dial is off by
# hundreds of Hz and is opt-in because, unlike the preamble path, this
# one really does cost CPU -- it searches CFO directly, so the cost is
# linear in the number of bins.
BLIND_MAX_OFFSET_HZ = 55.0
BLIND_WIDE_MAX_OFFSET_HZ = 625.0

# --- drift tracking --------------------------------------------------------
# Gains for the optional second-order loop that tracks residual carrier
# frequency across a transmission (modem.demodulate's drift_track).
# Off by default: on HF with modern radios the receiver's ~+-2 Hz
# budget is not usually threatened, and a loop that is not needed can
# only cost. See docs/todo.md for why there are two settings rather than
# one -- the loop bandwidth has to sit above the drift's own spectrum
# and below the channel's Doppler spread, and measured, those two can
# overlap: "fast" tracks rapid wander that "slow" cannot follow, and
# "slow" leaves fading alone that "fast" chases for 2.3 dB under mpd.
DRIFT_SLOW_ALPHA = 0.1
DRIFT_SLOW_BETA = 0.01
DRIFT_FAST_ALPHA = 0.3
DRIFT_FAST_BETA = 0.05
DRIFT_TRACK_MODES = ("off", "slow", "fast")


def drift_gains(mode: str) -> tuple[float, float]:
    """(alpha, beta) for a drift_track setting; (0, 0) means no loop."""
    if mode == "off":
        return 0.0, 0.0
    if mode == "slow":
        return DRIFT_SLOW_ALPHA, DRIFT_SLOW_BETA
    if mode == "fast":
        return DRIFT_FAST_ALPHA, DRIFT_FAST_BETA
    raise ValueError(f"drift_track must be one of {DRIFT_TRACK_MODES}, got {mode!r}")


# --- latent layout ---------------------------------------------------------
LATENT_H = 30  # spatial grid for a 240-high image, x8 downsample
LATENT_W = 40
LATENT_GROUPS = 3
CHANNELS_PER_GROUP = 44
LATENT_CHANNELS = LATENT_GROUPS * CHANNELS_PER_GROUP  # 132
GROUP_LATENTS = CHANNELS_PER_GROUP * LATENT_H * LATENT_W  # 52800
# FRAMES_PER_GROUP (and hence mode durations) is pinned to the pre-beacon
# capacity so reserving the beacon carrier costs capacity, not time: the
# 23-carrier LATENTS_PER_FRAME transmits fewer than GROUP_LATENTS per
# group, and the remainder is simply never given an on-air slot (treated
# as a permanent erasure, weight 0 — the same erasure-robustness path
# stage-1 training already exercises via random truncation/erasure).
_FULL_LATENTS_PER_FRAME = NC * DATA_SYMS_PER_FRAME * 2  # 240
FRAMES_PER_GROUP = GROUP_LATENTS // _FULL_LATENTS_PER_FRAME  # exactly 220
assert FRAMES_PER_GROUP * _FULL_LATENTS_PER_FRAME == GROUP_LATENTS
TRANSMIT_LATENTS_PER_GROUP = FRAMES_PER_GROUP * LATENTS_PER_FRAME  # 50600
DROPPED_LATENTS_PER_GROUP = GROUP_LATENTS - TRANSMIT_LATENTS_PER_GROUP  # 2200 (~4.2%)
assert 0 <= DROPPED_LATENTS_PER_GROUP < GROUP_LATENTS

# --- TX conditioning -------------------------------------------------------
CLIP_HEADROOM_DB = 0.0  # envelope clip threshold above mean envelope power;
#                         gives ~4.0 dB envelope PAPR
# Lowered from 0.5 with the Zadoff-Chu pilot (PROTOCOL_VERSION 3), and
# the two are one change rather than two. Most of what a wider headroom
# used to buy was relief for a pilot that was itself being clipped to
# pieces; with a pilot that no longer clips, the optimum moves down --
# measured PEP-fair on latent SNR, the AWGN optimum went from ~3.0 dB to
# ~1.0 and the multipath optimum to ~0.0, and 0.0 sits within 0.08 dB of
# both. Do not lower this without the ZC pilot: at the old pilot 0.0 is
# 0.2 dB *worse* than 0.5, i.e. a step away from that arm's optimum.
TX_BANDPASS = (850.0, 2200.0)  # Hz, post-clip filter
DEMOD_BACKOFF = 6  # samples: demod window starts this early inside the CP

# --- SNR reporting convention ---------------------------------------------
# Every SNR the project quotes -- the channel simulator's --snr, the
# receiver's pilot-based estimate, the figures in README and docs -- is
# signal power over the noise power falling in this bandwidth. Shared so
# the simulator and the estimator cannot drift apart, since a mismatch
# between them is invisible (both keep working, they just disagree about
# what a number means).
SNR_REF_BW_HZ = 2500.0

# --- frozen format constants ----------------------------------------------
# The seeds below are *provenance*, not a runtime input: they record how
# these constants were originally produced, and nothing reads them to
# rebuild anything.
#
# The distinction is load-bearing. These values are part of the on-air
# format. Deriving them at import time from
# `np.random.default_rng(seed)` would make numpy's PCG64, its bounded
# integer draw and its shuffle loop part of that format -- so a future
# numpy that changed any of them would change what this program
# transmits, silently, and two stations on different numpy versions
# would fail to talk to each other. The correct response to numpy
# changing is to keep transmitting the old sequence, which is only
# possible if the old sequence is written down.
#
# So the interleaver permutations are frozen as a committed array next
# to `framing.py`. `tools/freeze_format_constants.py --verify`
# re-derives them from the seed and reports whether numpy still agrees,
# which is information rather than a gate -- see that script.
#
# The pilot below is no longer in that category at all: it is a closed
# form over exact integers, so there is no generator whose behaviour it
# could depend on.
INTERLEAVER_SEED = 1000  # + group index; provenance only

# --- pilot sequence --------------------------------------------------------
# A numerically minimized crest-factor phase set: 0.99 dB envelope PAPR.
#
# Replaces a frozen random QPSK draw as of PROTOCOL_VERSION 3. That draw
# had 7.9 dB envelope PAPR against a clip threshold ~1 dB above the mean,
# which made the pilot the most heavily clipped symbol in the waveform --
# while also being the frame channel-estimate reference, the preamble
# (four repeats of it) and the blind-acquisition template. It lost power
# and gained distortion exactly where the receiver could least afford
# either. This set is worth ~+2.5 dB of latent SNR end to end (+3.3 at
# 4 dB AWGN, +2.5 at multipath 8 dB).
#
# **Zadoff-Chu was tried here first and must not be tried again.** At
# 2.70 dB it looked like the better engineering choice -- a one-line
# closed form with ideal periodic autocorrelation, far nicer to freeze
# into a waveform than an opaque vector. But ZC's defining property is an
# exact delay-Doppler equivalence: shifting it in frequency is the same
# as shifting it in time. This pilot *is* the acquisition template, so
# that makes CFO and timing confusable, and it is not a small effect --
# blind acquisition locked 55.7 Hz off inside its own +-55 Hz search
# range, slipped up to 7 frames of timing, and the metric gap between a
# true and a false lock collapsed from 14.4 (old pilot) to 2.07. No
# threshold recalibration survives that. The property that makes ZC
# elegant is the one that breaks it here.
#
# Note also that pilot PAPR is a proxy rather than the mechanism: the
# candidate ordering by PAPR is not the ordering by latent SNR (ZC at
# 2.70 dB beat a 2.85 dB QPSK set on multipath and lost to it on a
# noiseless clipping measurement). Anything that pushes this further
# should optimize against latent SNR and acquisition directly.

# The phase as an exact rational turn: phi_k = 2*pi * NUM[k] / DEN.
#
# Integers rather than radians so C++ evaluates the same expression on
# the same values -- see ofdm._phasor for why a phase that is a property
# of the libm rather than of the format is a real hazard here. DEN=1024
# quantizes the continuous optimum at a cost of 0.009 dB of PAPR (0.977
# -> 0.986), which buys exact cross-platform reproducibility.
PILOT_PHASE_DEN = 1024
PILOT_PHASE_NUM = (
    725, 497, 359, 322, 193, 849,
    710, 345, 960, 628, 347, 570,
    551, 678, 448, 713, 839, 90,
    236, 545, 1020, 403, 985, 304,
)

# --- blind acquisition gate ------------------------------------------------
# `sync.acquire_blind` / `BlindAccumulator` accept a lock when the winning
# phase bin's prominence (peak / median over the other 1151 bins) clears
# this. Raised from 4.0 with the new pilot, and the two are one change:
# the score is scale invariant in signal level but *not* in how much of
# the pilot survives the clipper, so a pilot that no longer clips lifts
# true and false scores together and the old gate admits locks it used
# to refuse.
#
# Calibrated like TEMPLATE_SCORE_THRESHOLD -- between the highest
# measured false score and the lowest real one worth keeping. The binding
# constraint is not noise (25 trials of pure noise peaked at 1.43) but a
# real transmission *outside* the search range, which reached 7.33; the
# lowest true score kept is 10.19 (mode A/C, mpp, -2 dB). Measured
# against the old pilot at 4.0, 25 trials per cell: AWGN unchanged at
# 1.00 down to -6 dB, mpp *better* (+2 dB 0.64 -> 0.72, -2 dB 0.56 ->
# 0.68), and the out-of-range false-lock rate 0.60 -> 0.00.
#
# What it gives up is mpp at -6 dB (0.52 -> 0.00), and that is not a
# tuning choice: those true locks score 6.4 while out-of-range signals
# reach 7.3, so no threshold separates them. The old setting bought that
# cell by accepting garbage 60% of the time.
BLIND_SCORE_THRESHOLD = 9.0

# Bumped to 2 with the 4-repeat preamble, and to 3 with the new pilot. The formats cannot sync to each other anyway (v1's acquisition
# template is a different length; v2's is a different sequence, so its
# correlation peak does not survive), so this is belt and braces: it
# stops an older header that happens to survive a correlation peak from
# decoding to a plausible mode.
PROTOCOL_VERSION = 3


@dataclass(frozen=True)
class ModeSpec:
    name: str
    index: int
    groups: int  # latent channel groups transmitted

    @property
    def n_frames(self) -> int:
        return self.groups * FRAMES_PER_GROUP

    @property
    def n_latents(self) -> int:
        """Full canonical latent count the model produces/expects for this
        mode (groups * GROUP_LATENTS) — the model-facing contract, not the
        smaller on-air transmit budget (see n_tx_latents)."""
        return self.groups * GROUP_LATENTS

    @property
    def n_tx_latents(self) -> int:
        """Number of latent values actually carried on-air (23-carrier
        capacity x frames): smaller than n_latents by DROPPED_LATENTS_PER_GROUP
        per group, which are permanently erased rather than transmitted."""
        return self.n_frames * LATENTS_PER_FRAME

    @property
    def duration_s(self) -> float:
        n = (
            LEADIN_SAMPLES
            + PREAMBLE_SAMPLES
            + HEADER_SAMPLES
            + self.n_frames * FRAME_SAMPLES
            + LEADOUT_SAMPLES
        )
        return n / FS


MODES = {
    "A": ModeSpec("A", 0, 1),  # ~32 s
    "B": ModeSpec("B", 1, 2),  # ~64 s
    "C": ModeSpec("C", 2, 3),  # ~95 s
}
MODES_BY_INDEX = {m.index: m for m in MODES.values()}
