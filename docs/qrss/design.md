# QRSSTVAE build design: waveform CE, single-pass receiver, multi-pass combining

**QRSSTVAE in one paragraph.** [SSTVAE](https://github.com/arodland/SSTVAE) sends a picture over HF radio as about 50,000 analog numbers (latents) made by a neural encoder; a neural decoder rebuilds the picture and degrades gently with noise. QRSSTVAE is a slow, narrow variant in the spirit of QRSS beacons. One group of 50,600 latents is sent as a single carrier about 50 Hz wide over a ~30-minute slot that starts on the quarter hour. The same picture is repeated over hours or days, and the receiver averages every pass it hears into one accumulator per (callsign, picture ID). That lets a picture build up from signals far below the noise, around −26 dB SNR in 2500 Hz after 16 passes. Waveform CE, built here, is a constant-amplitude carrier whose phase carries the latents. It is meant for Si5351 clock-chip beacons that can only set frequency and phase. Waveform L does the same with a linear signal for SSB rigs: two latents per complex symbol below a pilot tone, about 22 Hz wide ([waveform-l.md](waveform-l.md)). Its receiver, and the receiver that hears both at once, are §6.10.

This document is the build design for spec **rev 9** (`/mnt/project-files/qrss-spec/qrsstvae-options.md`), covering build steps 2 and 4 (spec §14) plus a portable C reference of the beacon's phase generator. It merges two earlier designs: A, "faithful and minimal", and B, "robust receiver first". Appendix A records what came from each and the errors found in both.

Notation used throughout:
- T = 485/16000 s is the CE symbol period and Rs = 1/T = 32.990 Bd.
- β = 0.8 rad rms is the phase deviation.
- A0 = e^(−β²/2) = 0.726149, and K = β·A0 = 0.580919 is the gain of the linear part.
- FS = 8000 Hz.
- q = floor(unix(QH)/900) is the slot's quarter-hour count.
- **t0 = 900·q + 1 s**, the centre of the first preamble symbol.
- "Stream index" s counts the frame's symbols without the callsign windows. "Position" p counts them with the windows, so it is proportional to time.

---

## 0. Scope, sources and ground rules

**In scope:**
- spec §14 step 2: CE transmitter with scrambler, precoder, header, picture ID and callsign windows; a beacon file for Si5351 boards; a channel simulator with the consumer impairments; the single-pass coherent receiver; the comparison of per-block and joint estimation;
- step 4: accumulator, association, leave-one-out EM, template search;
- a C reference of the beacon's phase and Si5351-step generator;
- waveform L in the receiver and the live listener (2026-10-12, §6.10): L's acquisition, tracker, gate V, demodulator and callsign windows, and one receiver that listens for both waveforms by default. L's on-air definition is `lframe.py` and `lwave.py`, with the spec's rev 13 and [waveform-l.md](waveform-l.md).

**Out of scope:** EM, the template search and retroactive detection for waveform L (§8.3, §13), the GUI, model retraining, and anything that needs hardware.

**Slot length.** Rev 9 §2.7 adds four callsign windows per pass, so a CE frame is 58,810 positions = **1782.678125 s**, not the 1736 s of the earlier brief. This design follows rev 9. Each window delays what follows it and changes nothing else.

**Ground rules.**
1. The code lives in a new package `sstvae/qrss/`, which the existing setuptools finder (`include = ["sstvae*"]`) picks up with no reinstall. CLIs are top-level `qrss_*.py` scripts in the `sstvae_*.py` pattern: usage docstring, `argparse.ArgumentParser(description=__doc__)`, `main()`, `print` summaries. Tests are `tests/test_qrss_*.py`.
2. No torch. Import the codec from `sstvae.codec` and `sstvae.checkpoint`, images from `sstvae.images`, audio from `sstvae.wavio`, and the latent layout from `sstvae.latents`. Never import `sstvae.data`, `sstvae.models` or `sstvae.waveform_channel`.
3. Never edit `sstvae/config.py` (including `MODES`, `ModeSpec`, `PROTOCOL_VERSION`), `sstvae/modem/*`, `native/`, golden vectors, `interleaver_perms.npy` or `NATIVE_SUBSTITUTIONS`. SSTVAE names are imported read-only. Private SSTVAE names (`framing._TX_PERMS`, `beacon._crc16`, `_int_to_bits`, `_bits_to_int`, `_ALPHABET`) are each wrapped once, in `picture.py` or `header.py`.
4. **Format modules** (marked F in §1) define what goes on air. They may not call `default_rng`, `RandomState` or anything named `seed`. `tests/test_qrss_frozen.py` checks this by walking their AST, as `test_frozen_format.py:118` does for the modem. Only `channel.py`, the CLIs and tests may use `default_rng`. Every on-air sequence is closed-form, built from SHA-256 over integers or from committed literal data.
5. **Phase rule.** Carrier mixing uses exact integer turn arithmetic, as `(n·f_num) mod D`. Accumulated phases are carried in float64 turns and wrapped with `dsp.wrap_cycles` before `exp`. The modulation phase φ is bounded (|φ| < 5 rad) and may stay in radians. The callsign-window phase is in turns.
6. Codec tests are marked `@pytest.mark.codec`. They resolve models from `$QRSSTVAE_MODEL_DIR`, then `/home/user/qrss-data/models`, then the HF cache, and **never download**. They skip through the `_codec_skip` / `(Exception, SystemExit)` pattern of `test_native_parity.py:961-993`, and fail if `SSTVAE_REQUIRE_CODEC` is set. Multi-minute tests are marked `@pytest.mark.slow`.

---

## 1. Files and ownership

F marks a format module. WP is the owning work package (§12). No file has two owners.

| File | WP | Contents |
|---|---|---|
| `sstvae/qrss/__init__.py` | 1 | docstring, `__all__`, nothing heavy |
| `sstvae/qrss/constants.py` F | 1 | every on-air constant (§2.1) |
| `sstvae/qrss/sequences.py` F | 1 | SHA-256 bit streams: preamble, references, scrambler |
| `sstvae/qrss/precoder.py` F | 1 | block split, fast blockwise WHT, precode and unprecode |
| `sstvae/qrss/frame.py` F | 1 | `FrameSpec`, presets, `Layout`, positions, callsign windows, `assemble` |
| `sstvae/qrss/morse.py` F | 1 | Morse table and keying units |
| `sstvae/qrss/picture.py` F | 1 | segment preparation, fp16 store, picture ID, codec ID, int8, air↔canonical maps, `.qrsp` files |
| `sstvae/qrss/types.py` | 1 | shared dataclasses (§6.9): `Timing`, `FreqPath`, `CarrierCandidate`, `Detection`, `TrackReport`, `CwIdResult`, `PassResult`, `EmPrior` |
| `tests/qrss_helpers.py` | 1 | synthetic unit-RMS latents, latent SNR, model-dir resolver, codec skip, `Q_TEST` |
| `sstvae/qrss/polar.py` F | 2 | polar (2048, 142) encoder, rate matching, CA-SCL decoder, `INFO_SET` literal |
| `sstvae/qrss/header.py` F | 2 | `HeaderFields`, pack and unpack, CRC wrapper, grid codec, `encode`, `decode` |
| `tools/gen_qrss_polar.py` | 2 | regenerates and `--check`s `INFO_SET` by Gaussian approximation |
| `sstvae/qrss/ce.py` F | 3 | pulse, phase synthesis at any rate, callsign-window phase, baseband and audio, mean-waveform template |
| `sstvae/qrss/tx.py` F | 3 | slot symbols from a stored picture or a beacon file, transmit audio |
| `sstvae/qrss/beaconfile.py` F | 3 | beacon file read and write |
| `sstvae/qrss/si5351.py` F | 3 | frequency-step synthesis model with error feedback |
| `qrss_encode.py`, `qrss_beacon.py`, `qrss_transmit.py` | 3 | CLIs |
| `sstvae/qrss/channel.py` | 4 | channel simulator (uses `default_rng`) |
| `qrss_simulate.py` | 4 | CLI |
| `sstvae/qrss/frontend.py` | 5 | audio↔FE, blanker, inverse AGC, channeliser, noise floor, `PassbandStore` (48 h on-disk FE store, §6.9) |
| `sstvae/qrss/acquire.py` | 5 | A1 carrier lines, A2 preamble search, A3 track-before-detect, CFAR constants |
| `sstvae/qrss/track.py` | 6 | frequency-path spline, timing fit, complex-gain Kalman filter and RTS smoother, process noise by maximum likelihood, verification gate |
| `sstvae/qrss/demod.py` | 6 | symbol matched filter, extraction, weights, plain and joint estimators, header LLRs |
| `sstvae/qrss/cwid.py` | 6 | callsign-window reading and matching |
| `sstvae/qrss/receiver.py` | 6 | `receive_pass`, `receive_slot`, `receive_wav` |
| `qrss_receive.py` | 6 | CLI |
| `sstvae/qrss/store.py` | 7 | pass store, accumulators, expiry, merge |
| `sstvae/qrss/associate.py` | 7 | association rules 1 to 4 |
| `sstvae/qrss/render.py` | 7 | decoder planes, codec call |
| `tests/qrss_fakes.py` | 7 | `make_pass_result(...)`, synthetic passes for multi-pass tests |
| `qrss_beacon_c/` (`qrss_ce.h`, `qrss_ce.c`, `sha256.h`, `sha256.c`, `qrss_tables.h`, `test_main.c`, `Makefile`), `tools/gen_qrss_tables.py` | 8 | portable C reference |
| `sstvae/qrss/em.py` | 9 | leave-one-out EM, template search |
| `qrss_decode.py` | 9 | CLI |

Tests are owned by the work package they belong to:

| WP | Test files |
|---|---|
| 1 | `tests/test_qrss_{frozen,format}.py` |
| 2 | `tests/test_qrss_header.py` |
| 3 | `tests/test_qrss_{ce,beacon}.py` |
| 4 | `tests/test_qrss_channel.py` |
| 5 | `tests/test_qrss_{frontend,acquire}.py` |
| 6 | `tests/test_qrss_{track,receiver,cwid}.py` |
| 7 | `tests/test_qrss_multipass.py` |
| 8 | `tests/test_qrss_c_ref.py` |
| 9 | `tests/test_qrss_{em,cli,acceptance}.py` |

**Waveform L** came after the work packages and has no WP: `lframe.py` F and `lwave.py` F (its frame and signal, both in F6's frozen list), `ltrack.py` and `ldemod.py` (its tracker and demodulator, §6.10), and L branches in `acquire`, `cwid`, `receiver`, `em`, `store`, `associate`, `live` and the CLIs. Its tests are `tests/test_qrss_{lunits,lchannel,lacquire,ltrack,ldemod,lreceiver,live_l}.py` (§10.9).

`pyproject.toml` is edited only if WP7 ends up shipping `qrss/latent_means.npy` (§8.2), in which case WP7 adds `"sstvae.qrss" = ["*.npy"]` to package data.

**Calibrated receiver constants** live in the module that uses them, so no work package edits another's file:
- `acquire.TBD_GUMBEL_MU` and `TBD_GUMBEL_BETA` belong to WP5.
- `demod.D_PASS` and `track.KAPPA_SELF` belong to WP6, measured with WP3's `ce.loopback_stats`.

---

## 2. On-air definition (normative)

### 2.1 Constants (`constants.py`)

```python
from fractions import Fraction
from sstvae import config as _c
FS = _c.FS                                   # 8000
T_NUM, T_DEN = 485, 16000                    # T = 485/16000 s exactly
T_SYM = T_NUM / T_DEN                        # 0.0303125
ALPHA = Fraction(3, 20)                      # RRC roll-off
SPAN = 8                                     # pulse cut at |tau| <= 8 symbols
BETA = Fraction(4, 5)                        # rad rms
A0 = math.exp(-0.32)                         # e^(-beta^2/2) = 0.7261490370736908
K_LIN = 0.8 * A0                             # 0.5809192296589527
CARRIER_FRAC = math.exp(-0.64)               # 0.5272924240430485
USEFUL_FRAC = 0.64 * CARRIER_FRAC            # 0.3374671513875510
N_PRE, N_HDR, REF_PERIOD = 660, 2640, 16
N_HDR_BITS, N_INFO_BITS, POLAR_N = 2474, 142, 2048
FORMAT_VERSION = 2                           # 1 had no spread header (tests only); 2.6.1
FORMAT_VERSIONS = (1, 2)                     # what a header may say
SPREAD_RHO = 0.054                           # spread-header power, latent units (2.6.1)
WAVEFORM_CE, WAVEFORM_L = 0, 1               # CE: frame.py, ce.py; L: lframe.py, lwave.py
LEAD_IN_MAX_S = 10
CW_WINDOW = 384                              # CE symbols per callsign window (11.64 s)
CW_UNIT = 2                                  # CE symbols per Morse unit (60.625 ms)
CW_LEAD_UNITS, CW_ROOM_UNITS = 8, 176        # plain units before the call; room for the call
CW_AFTER_FULL = (13888, 28350, 42812, 57274) # windows follow these stream symbols (2x L's)
INT8_SCALE = 20                              # beacon latent = q/20, q in [-127, 127]
FE_FS, FE_CENTER_HZ = 4000, 1500             # receiver front end: complex, 300-2700 Hz
CH_FS, CH_DECIM = 250, 16                    # per-signal channel: complex, +-125 Hz
Z_ACCEPT = 6.0                               # single detection/association gate
Q_EPOCH_NOTE = "q = floor(unix_utc(QH)/900); unix time ignores leap seconds"
```

`config.TRANSMIT_LATENTS_PER_GROUP` (50,600), `config.GROUP_LATENTS` (52,800) and `config.SNR_REF_BW_HZ` (2500) are imported from SSTVAE and never redefined.

### 2.2 Frame, positions and timing (`frame.py`)

```python
@dataclass(frozen=True)
class FrameSpec:
    name: str = "full"
    n_pre: int = 660
    n_hdr: int = 2640                       # 0 (no header) or 2640
    n_data: int = 50600                     # data latents (a prefix of the group's air order)
    cw_after: tuple[int, ...] = CW_AFTER_FULL   # stream symbols before each window
    hdr_rho: float | None = None            # spread header (2.6.1): None = SPREAD_RHO with a block, else 0
    # derived properties:
    #   n_data_sym = smallest L with L - ceil(L/16) == n_data   (53,974 for 50,600)
    #   n_sym  = n_pre + n_hdr + n_data_sym                      (57,274)
    #   n_pos  = n_sym + CW_WINDOW * len(cw_after)               (58,810)
    #   duration_s = n_pos * T_SYM                               (1782.678125 s)
FULL   = FrameSpec()
MEDIUM = FrameSpec("medium", n_data=16384, cw_after=(10000, 20777))   # 20,777 sym, 21,545 pos, 653.1 s
SHORT  = FrameSpec("short",  n_data=4096,  cw_after=(5000, 7670))     #  7,670 sym,  8,438 pos, 255.8 s
TINY   = FrameSpec("tiny", n_hdr=0, n_data=1024, cw_after=())         #  1,753 sym,  1,753 pos,  53.1 s
PRESETS = {s.name: s for s in (FULL, MEDIUM, SHORT, TINY)}
FULL_V1, MEDIUM_V1, SHORT_V1 = ...       # the same with hdr_rho = 0: format version 1, in LEGACY
def get(name) -> FrameSpec               # PRESETS, then LEGACY, then PROTOTYPES
def plain(spec) -> FrameSpec             # spec without the spread copy (its version 1 frame)
```

FULL, MEDIUM and SHORT carry the spread copy of the header; TINY has no header at all. The `-v1` frames are for receiving recordings made before format version 2 and for measuring the spread copy against.

`FrameSpec.__post_init__` validates the window list. Its entries must be strictly increasing, each must lie in (n_pre + n_hdr, n_sym], and the last may equal n_sym, in which case that window ends the frame.

**Stream layout**
- Stream symbols [0, n_pre) are the preamble, [n_pre, n_pre + n_hdr) the header, and the rest data.
- Let j = s − n_pre. Symbol s is a **reference** iff j ≥ 0 and j mod 16 = 0. Since 2640 ≡ 0 mod 16, data references also fall at data-local index ≡ 0 mod 16.
- Header non-reference symbol i carries coded bit i, for i < 2474. Header non-reference symbol 2474 is the **spare** and sends a known +1.
- Data non-reference symbol i carries precoded value x_i (§2.5).
- FULL has 165 header references, 2,474 coded symbols, 1 spare, 3,374 data references and 50,600 data symbols.

**Positions and time**
- pos(s) = s + 384·|{w : cw_after[w] ≤ s}|.
- Window w occupies positions P_w … P_w + 383, where P_w = cw_after[w] + 384·w. For FULL, P = 13,888 / 28,734 / 43,580 / 58,426.
- Stream symbol s is **centred at t0 + pos(s)·T** on the sender's clock. The 16 kHz grid index of sample n at rate f_s (f_s dividing 16000) is m = (16000/f_s)·n − 485·pos. This is an integer, so a pulse table on the 1/485-symbol grid is exact at 16000, 8000, 4000, 2000, 1000 and 500 Hz. A 64-phase polyphase table covers 250 Hz.
- **Keying.** The carrier is keyed from t0 − 8T at the latest. A sender with a lead-in keys earlier (§2.8). The carrier stays keyed until the end of the frame, t0 + (n_pos − ½)·T, which is the end of the last window. For a frame whose last window is not at the end, such as TINY, it stays keyed until t0 + (pos(n_sym − 1) + 8)·T.
- Window w spans [t0 + (P_w − ½)T, t0 + (P_w + 383.5)T). For FULL these start 420.96, 870.98, 1321.00 and 1771.02 s after t0, and the frame ends 1782.68 s after t0, which is 29:43.7 past the quarter hour.

```python
@dataclass(frozen=True)
class Layout:                                  # int64 arrays; stream indices unless named pos
    pre: ndarray; hdr_ref: ndarray; hdr_bits: ndarray; hdr_spare: ndarray
    data_ref: ndarray; data: ndarray
    ref: ndarray                               # hdr_ref ++ data_ref, in time order (reference counter m)
    known_idx: ndarray; known_val: ndarray     # preamble + all refs + spare, int8 +-1
    pos: ndarray                               # (n_sym,) position of each stream symbol
    win_pos: ndarray                           # (n_win, 2) first and last position of each window
def layout(spec: FrameSpec) -> Layout          # lru_cached
def assemble(spec, hdr_bits: uint8[2474] | None, x: float64[n_data]) -> float64[n_sym]
def disassemble(y_sym, spec) -> tuple[ndarray, ndarray]   # (header-symbol values (2474,), data values (n_data,))
```

### 2.3 SHA-256 sequences (`sequences.py`)

```python
def sha_bits(domain: bytes, n: int, prefix: bytes = b"") -> np.ndarray:   # uint8[n] of 0/1
    # D_b = SHA256(domain || prefix || uint32_be(b)), b = 0, 1, 2, ...
    # bit i = (D_{i>>8}[(i & 255) >> 3] >> (7 - (i & 7))) & 1          (MSB first)
def pm1(bits) -> np.ndarray                    # int8: 0 -> +1, 1 -> -1
def preamble_ce() -> int8[660]                 # pm1(sha_bits(b"QRSSTVAE CE preamble", 660)), lru_cached
def references_ce(n_ref: int) -> int8[n_ref]   # pm1(sha_bits(b"QRSSTVAE CE reference", n_ref)); ref m takes bit m
def scrambler(q: int, n: int) -> int8[n]       # pm1(sha_bits(b"QRSSTVAE scramble", n, struct.pack(">Q", q)))
def quarter_hour_count(utc: datetime) -> int   # aware datetime exactly on a quarter hour, else ValueError
```

- Domain strings are ASCII with no terminator.
- The scrambler index is the data-latent index in air order. FULL uses 198 hash blocks.
- The preamble, the references and the header are never scrambled or precoded.
- `b"QRSSTVAE L preamble"` and `b"QRSSTVAE L reference"` are waveform L's (bit pairs as QPSK: 330 preamble symbols, and reference m takes bits 2m and 2m+1).

### 2.4 Pulse and signal (`ce.py`)

τ is time in symbols. The pulse is the textbook unit-energy RRC (the `pmcore.rrc` formula), cut at ±8 symbols and renormalised on the 16 kHz grid:

```
h(τ)      = [sin(πτ(1−α)) + 4ατ·cos(πτ(1+α))] / [πτ(1 − (4ατ)²)]
h(0)      = 1 − α + 4α/π
h(±1/4α)  = (α/√2)·[(1 + 2/π)·sin(π/4α) + (1 − 2/π)·cos(π/4α)]
p(τ)      = h(τ)/c  for |τ| ≤ 8, else 0,   c² = (1/485)·Σ_{m=−3880}^{3880} h(m/485)²
```

Checked values: c = 0.9999137, p(0) = 1.041076, and G0 = (1/485)·Σ p(m/485) = 1.001665, which is the matched-filter gain of a constant.

**Phase.** The signal is s(t) = a(t)·exp(j·[φ(t) + 2π·θ_cw(t)]), with

φ(t) = β·Σ_s x_s·p((t − t0)/T − pos(s)).

- x_s is the assembled symbol vector (§2.2).
- θ_cw is the callsign-window phase in turns (§2.7). It is 0 outside windows.
- a(t) is 1 while keyed and 0 otherwise. Inside an on-off-keyed window it follows the keying (§2.7).
- φ has time-averaged variance β², measured at 0.645 on Gaussian symbols.
- The lead-in is simply φ = 0 with a = 1.

```python
def pulse(tau) -> ndarray                                     # continuous, float64
PULSE_TABLE: ndarray                                          # (7761,) p(m/485), m = -3880..3880
def phase_at(tau_pos, sym, spec, var=None) -> tuple[ndarray, ndarray | None]
    # tau_pos: time in symbols from t0, i.e. (t - t0)/T (any spacing, float64).
    # Returns phi_mean = beta*sum mu_s p(tau - pos_s), and with var given also
    # v = beta^2 * sum var_s p(tau - pos_s)^2.  17 vectorized offsets.
def phase_grid(fs: int, n0: int, n: int, sym, spec) -> ndarray
    # exact polyphase path: sample i at t = t0 + (n0 + i)/fs, fs in {16000, 8000, 4000, 2000, 1000, 500, 250}
def cw_phase_turns(tau_pos, spec, keying: list[uint8[192]], ook=False) -> tuple[ndarray, ndarray]   # (theta_cw, a)
def baseband(sym, spec, fs, start_s, n, *, keying=None, ook=False, lead_in_s=0.0,
             ramp_s=0.05, time_scale=1.0) -> complex128[n]
    # a*exp(j(phi + 2 pi theta_cw)) at t_i = t0 + (start_s + i/fs)*time_scale; zero where unkeyed;
    # a raised-cosine amplitude ramp of ramp_s outside the keyed span (where phi = 0) against key clicks.
    # time_scale != 1 (clock offset) uses phase_at; == 1 uses phase_grid.
def to_audio(z, fs, carrier_hz_num: int, carrier_hz_den: int = 1, amplitude=0.5) -> float64
    # resample_poly up to 8 kHz, mix with phase turns (n*num mod (den*8000))/(den*8000) exactly; sqrt(2)*Re
def mean_template(tau_pos, mu, nu, spec, keying=None, ook=False, keyed_from_pos=None) -> complex128
    # E[s(t)] = a * exp(j(phi_mu + 2 pi theta_cw)) * exp(-v/2), mu/nu per stream symbol
    # (known: nu=0; unknown data: mu=0, nu=1; EM: mu=x_hat, nu=v). Used by the receiver (§6.5).
def loopback_stats(sym, spec, fs=CH_FS) -> dict
    # noiseless genie loopback: linear gain, per-class residual variance of MF output vs mean_template MF
```

**Audio level.** `to_audio` writes √2·Re{s·e^{jωt}}·amplitude/√2, so complex baseband power equals real audio power. A WAV made by `qrss_transmit` starts at t0 − 12 s by default, which is QH − 11 s.

### 2.5 Latents, scrambling, precoding (`picture.py`, `precoder.py`)

```python
def block_sizes(n: int) -> list[int]       # [64]*(n//64) + binary decomposition of n%64, descending
                                           # 50600 -> 790x64, 32, 8;  16384 -> 256x64;  4096 -> 64x64;  1024 -> 16x64
def wht(v: float64[..., n]) -> ndarray     # blockwise Sylvester WHT, H[i,j] = (-1)^popcount(i&j)/sqrt(M);
                                           # in-place butterflies on (nblocks, M); self-inverse
def precode(a_air, q) -> ndarray           # x = WHT(c_q * a)
def unprecode(y, q) -> ndarray             # a = c_q * WHT(y)
def block_mean(v) -> ndarray               # per-block mean, broadcast back to (n,)
```

```python
def air_to_canonical(group: int) -> intp[50600]     # framing._TX_PERMS[group] (offset within the group); intp
def prepare_segment(full: float64[158400], group: int) -> float16[52800]
    # canonical group slice; zero the 2,200 never-sent; scale so the 50,600 sent values have RMS 1
    # (computed in float64); then round to float16. THE explicit unit-RMS step (spec §3).
def air_values(seg: float16[52800], group: int) -> float64[50600]       # seg[_TX_PERMS[group]]
def canonical_from_air(air: ndarray, group: int) -> ndarray[52800]      # inverse; never-sent = 0
def picture_id(codec_id: int, mode: int, segs: list[float16[52800]]) -> int
    # sha256(uint16_be(codec_id) || uint8(mode) || b"".join(s.astype("<f2").tobytes()))[:4], big-endian
def codec_id_from_onnx(path_or_dir: str, precision: str = "fp16") -> int | None
    # onnxruntime.InferenceSession(..., providers=["CPUExecutionProvider"]).get_modelmeta()
    # .custom_metadata_map["sstvae.source_sha256"][:4] as hex; v5 -> 0xD1D8; None if absent
def to_int8(air) -> int8 ; def from_int8(q) -> float64     # clip(rint(20*a), -127, 127) (round half even); q/20
@dataclass class StoredPicture: mode: int; codec_id: int; segs: list[float16[52800]]; picture_id: int
def save_qrsp(path, sp) ; def load_qrsp(path) -> StoredPicture   # .npz: mode, codec_id, segs (f2), picture_id
```

- `segs` holds the mode's groups 0..mode, in order.
- The transmitter encodes once, stores the fp16 tensor, and never re-encodes for a repeat.
- A beacon sends the int8 version of the same air values. That rounding is identical in every pass and 37 dB below the latents.

### 2.6 Header and FEC (`header.py`, `polar.py`)

**Fields.** 142 bits, in table order, MSB first:

| Bits | Field | Encoding |
|---|---|---|
| 0–47 | callsign | 8 × 6 bits via `beacon.callsign_to_codes`. QRSS accepts only `[A-Z0-9/]`, 1 to 8 characters, and raises `ValueError` otherwise (§2.7 needs Morse for every character) |
| 48–62 | grid | ((F1·18 + F2)·10 + D1)·10 + D2, where F is A..R as 0..17; 32767 means none |
| 63–94 | picture ID | 32 bits |
| 95–96 | mode | A = 0, B = 1, C = 2 |
| 97–98 | segment | 0 to 2, the latent group |
| 99–114 | codec ID | 16 bits |
| 115–118 | version | 2 (1: format version 1, without the spread copy of 2.6.1; sent only in tests) |
| 119–125 | reserved | 0 |
| 126–141 | CRC | `beacon._crc16(bits[0:126])`, wrapped as `header.crc16` |

The CRC check value is **0xA69D over ASCII "123456789"**, verified on the repo. It is **not** CRC-16/CCITT-FALSE, which gives 0x29B1 (decision D8).

**Polar code: N = 2048, K = 142 (CRC inside), E = 2474, CA-SCL with L = 8.**

- Mother code x = u·F^{⊗11} (mod 2) with F = [[1,0],[1,1]], in natural order with no bit reversal: x_j = ⊕_{i ⊇ j} u_i.
- Rate matching is circular repetition: coded bit e = x_{e mod 2048}, e = 0..2473. Mother bits 0..425 are sent twice.
- Header bit t, in table order, goes to u[INFO_SET[t]]. Frozen bits are 0.
- **`INFO_SET` is committed literal data** (normative):

```
INFO_SET = (1015, 1019, 1021, 1022, 1023, 1503, 1519, 1527, 1530, 1531, 1532, 1533, 1534, 1535, 1663,
 1727, 1759, 1771, 1773, 1774, 1775, 1779, 1781, 1782, 1783, 1785, 1786, 1787, 1788, 1789, 1790, 1791,
 1853, 1854, 1855, 1879, 1883, 1885, 1886, 1887, 1895, 1899, 1901, 1902, 1903, 1907, 1909, 1910, 1911,
 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1935, 1943, 1947, 1949, 1950, 1951, 1959, 1961, 1962,
 1963, 1964, 1965, 1966, 1967, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980,
 1981, 1982, 1983, 1989, 1990, 1991, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2001, 2002, 2003, 2004,
 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2017, 2018, 2019, 2020, 2021, 2022,
 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039,
 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047)
sha256(np.array(INFO_SET, ">u2").tobytes()) = 4e064eee9b1df68ee4d044970b51f05a16700d51599f857f86e7d15d782b97bb
```

  It comes from Gaussian-approximation density evolution at the design point Es/N0 = −9.4 dB per coded bit, which is SNR₂₅₀₀ = −23.5 dB, the spec's header threshold. The evolution follows the exact repetition pattern and the natural-order butterfly: the top split's f/g operation is chosen by the MSB of the index.
  - `tools/gen_qrss_polar.py` regenerates the set, and its `--check` compares against the literal.
  - The literal is the definition. A platform difference in the GA is a tool bug, not a format change.
  - Moving the design point to −10 or −8.8 dB changes one index, so the choice is robust.

- **Decoding.**
  1. Fold the coded LLRs to mother LLRs: L_i = Σ_{e ≡ i mod 2048} L_e.
  2. Run SCL with L = 8, f = min-sum, exact g and the LLR path metric, vectorized over the list. Target: under 0.5 s per decode in numpy.
  3. Accept the most likely path that passes the CRC **and** has version = 2, reserved = 0, mode ≤ 2, segment ≤ mode, and a callsign made only of `[A-Z0-9/ ]`. Version 1 is accepted only by the block-alone fallback of 2.6.1 and on a `-v1` frame.
  4. Otherwise return None and keep the LLRs.
- **LLRs.** llr = log P(b=0)/P(b=1), so a positive value means symbol +1. Coded bit e rides header non-reference symbol e as +1 for 0 and −1 for 1. For waveform L, bit e goes to L symbol ⌊e/2⌋, on I for even e and Q for odd, so LLR indices are shared across waveforms (§6.10).

```python
@dataclass(frozen=True)
class HeaderFields: callsign: str; grid: str | None; picture_id: int; mode: int; segment: int
                    codec_id: int; version: int = 2; reserved: int = 0
def pack(h) -> uint8[142] ; def unpack(bits) -> HeaderFields | None      # None if CRC or field checks fail
def crc16(bits) -> uint8[16] ; def grid_encode(g) -> int ; def grid_decode(v) -> str | None
INFO_SET: tuple[int, ...]
def polar_encode(info: uint8[142]) -> uint8[2474]
def polar_decode(llr: float[2474], list_size=8) -> uint8[142] | None
def encode(h) -> uint8[2474] ; def decode(llr, list_size=8) -> HeaderFields | None
```

**Performance check** (GA density evolution, successive-cancellation upper bound, ideal channel; done for this design, Appendix A):

| Construction | SC BLER at Es/N0 −10 dB | SC BLER at −9.4 dB |
|---|---|---|
| (2048) GA, the chosen set | 2.6e-3 | 2.3e-4 |
| (1024) GA | 4.1e-3 | 4.2e-4 |
| (2048) polarization weight (design B) | 4.0e-3 | 6.8e-4 |
| (1024) Bhattacharyya as written in design A | 1.0 | 1.0 |

CA-SCL with L = 8 only improves on these.

### 2.6.1 Spread copy of the header (format version 2, spec rev 11 §5.1)

Every CE frame with a header block also carries the header on its data symbols, as a small extra phase. Data symbol i (stream order, after the precoder) is sent as x_i + √ρ·h_i, with h_i = w_i·(1 − 2c[i mod 2474]), c the 2,474 coded header bits above, w = `pm1(sha_bits(b"QRSSTVAE CE spread header", n_data))` (`sequences.spread_whitener`, not keyed by q) and ρ = `SPREAD_RHO` = 0.054. `frame.assemble` adds it; `frame.spread_symbols` is the term on its own.

- **Round A** decodes the header from the block's LLRs plus the spread copy's (`demod.spread_llr`, folded over the repeats), then from the spread copy alone, then from the block alone accepting either version. A version 1 header switches the pass to `frame.plain(spec)` (`receiver.decode_header`).
- **Round B** removes the known phase exactly (`track.remove_spread`), so the latents lose nothing once the header is known.
- Design, costs and Andrew's measurements: [spread-header.md](spread-header.md).
- Waveform L carries the same copy linearly: its data latents are sent as (x + √ρ·h)/√(1+ρ), two per symbol (`lframe.data_latents`). A linear signal cannot be de-spread on the channel, so L's receiver subtracts the copy in the demodulator once the header is known (§6.10).

### 2.7 Callsign windows (`morse.py`, `ce.py`)

These implement spec §2.7.
- Each window is 384 CE symbols, which is 192 Morse units of 2T = 60.625 ms each.
- Units 0–7 are plain carrier. The call starts at unit 8. The units after the call, through unit 191, are plain carrier.
- The call takes at most 176 units.

**Text.** The header callsign with trailing spaces stripped, in international Morse: dot = 1 unit, dash = 3, 1 unit between elements, 3 between characters, no trailing gap. The table holds A–Z, 0–9 and '/', exactly as `sims/cwid/cwid.py`. The longest 8-character call, "00000000", takes 173 units.

```python
MORSE: dict[str, str]
def keying_units(callsign: str) -> uint8[192]     # k[u] = 1 for key-down, units 8 .. 8+len-1; ValueError if > 176
```

**FSK (default).** Key-down is one cycle per unit below the carrier, −16.495 Hz, with each change smoothed by a centred Hann pulse one unit long. With x = (t − t_wstart)/(2T) in units, where t_wstart = t0 + (P_w − ½)T, define

R(y) = 0 for y < −½;  (y + ½)²/2 − (cos 2πy + 1)/(4π²) for |y| ≤ ½;  y for y > ½,

which is the integral of the Hann CDF. Then

θ_cw(x) = −Σ_u k[u]·[R(x − u) − R(x − u − 1)]  turns,

which is exact, closed form, and an integer number of turns once the window has ended. a(t) = 1 throughout. The data pulse tails (±8T = ±4 units) die inside the 8 plain units at each end.

**On-off keying (allowed alternative).** θ_cw = 0. In units 5–183, a = k[u] with hard switching: key-down means on; units 5–7 are therefore off (key-up), so the call's first element is readable. Units 0–4 and 184–191 stay on, which covers the data pulse tails (they reach at most 4 units into the window) and keeps the phase reference at the window edges (decision D14, as adopted in the spec 2026-10-09). The beacon file's flags select the keying, and receivers accept both.

### 2.8 Lead-in carrier

This implements spec §2.8.
- The lead-in is optional, 0 to 10 s long, and ends where keying would start anyway (t0 − 8T).
- It is the same s(t) with φ = 0 and a = 1, so it runs phase-continuously into the preamble by construction.
- It is not flagged in the header, carries no symbols and adds nothing to accumulators.
- `baseband(..., lead_in_s=L)` sets a = 1 on [t0 − L, t0 − 8T).

### 2.9 Beacon file (`beaconfile.py`), little-endian

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `b"QRSB"` |
| 4 | 1 | file version = 1 |
| 5 | 1 | waveform (0 = CE) |
| 6 | 1 | mode |
| 7 | 1 | segment |
| 8 | 4 | picture ID (u32) |
| 12 | 2 | codec ID (u16) |
| 14 | 8 | callsign, ASCII, space padded |
| 22 | 2 | grid code (u16, 32767 = none) |
| 24 | 4 | n_latents (u32) = 50600 |
| 28 | 2 | int8 scale (u16) = 20 |
| 30 | 1 | flags: bit 0 = callsign keying (0 FSK, 1 OOK); other bits 0 |
| 31 | 1 | reserved = 0 |
| 32 | 24 | callsign keying, 192 units, 1 bit each MSB first (1 = key-down), from `keying_units` |
| 56 | 310 | header coded bits, 2474 bits MSB first, last 6 pad bits 0 |
| 366 | n | int8 latents in **air order**, neither scrambled nor precoded |
| 366 + n | 4 | CRC-32 (zlib) of every preceding byte |

- A full segment is **50,970 bytes**, which fits a 24LC512 (65,536 bytes).
- The keying mask is stored so that a microcontroller needs no Morse table. `read` checks that the mask equals `keying_units(callsign)`, and that the CRC-32 matches.
- The beacon itself computes the preamble, references, scrambler, precoder, the spread copy of the header (2.6.1, from the stored header bits), phase and window phase.

```python
@dataclass class BeaconFile: waveform: int; mode: int; segment: int; picture_id: int; codec_id: int
                             callsign: str; grid: str | None; ook: bool; keying: uint8[192]
                             hdr_bits: uint8[2474]; latents_i8: int8[n]
def write(path, bf) ; def read(path) -> BeaconFile          # ValueError on CRC, magic or keying mismatch
def from_picture(sp: StoredPicture, segment, header: HeaderFields, ook=False) -> BeaconFile
```

### 2.10 Si5351 synthesis model (`si5351.py`)

```python
def si5351_steps(sym, spec, q_unused, f_update: Fraction, step_hz: float, n_updates: int,
                 keying=None, ook=False, u0_pos: float = -8.0) -> int32[n_updates]
    # update u at t_u = t0 + (u0_pos*T + u/f_update); target phase Phi(t_{u+1}) (rad, from phase_at + cw);
    # f_u = round((Phi_tgt(u+1) - Phi_reached)/(2 pi Tu)/step_hz); Phi_reached += 2 pi f_u step_hz Tu  (error feedback)
def synth_phase(steps, f_update, step_hz, fs, n, u0_pos=-8.0) -> float64[n]   # piecewise-linear phase at fs
```

This is exactly the `pm_spectrum_synth.synth` recipe. On-off-keyed windows switch only the output, so the model leaves the phase running.

---

## 3. Transmit pipeline (`tx.py`) and CLIs (WP3)

```python
def slot_symbols(src: StoredPicture | BeaconFile, q: int, spec=FULL, *, segment=0,
                 header: HeaderFields | None = None, use_int8: bool | None = None) -> float64[spec.n_sym]
    # data = precode(air_values[:n_data] (or from_int8), q); header bits from header.encode or the file
def transmit_audio(src, q, carrier_hz: int = 1500, *, spec=FULL, lead_in_s=0.0, ook=False,
                   pre_s=12.0, post_s=3.0, amplitude=0.5) -> float64     # WAV sample 0 = t0 - pre_s
```

| CLI | Arguments |
|---|---|
| `qrss_encode.py` | `image out.qrsp [--model DIR] [--precision fp16\|fp32] [--mode A\|B\|C] [--codec-id HEX]`. Prints the picture ID. Refuses if the codec ID is unknown and `--codec-id` is not given |
| `qrss_beacon.py` | `in.qrsp out.bin --callsign K1ABC [--grid FN42] [--segment 0] [--ook]` |
| `qrss_transmit.py` | `in.qrsp\|in.bin out.wav --slot 2026-10-09T06:00Z [--callsign --grid --segment, when given a .qrsp] [--freq 1500] [--lead-in 0..10] [--ook] [--frame full\|medium\|short\|tiny] [--float]` |

`--float` writes with `wavio.write_wav_float`, giving exact round trips. Otherwise output goes through `write_wav`, which peak-normalises.

---

## 4. Channel simulator (`channel.py`, WP4)

The simulator works in the receiver's front-end format: complex at FE_FS = 4000 Hz, with 0 Hz at 1500 Hz audio. A full slot is about 7.2 M samples. `frontend.fe_to_audio` produces 8 kHz real audio only when a WAV is wanted.

```python
@dataclass(frozen=True)
class Neighbour: df_hz: float; rel_db: float; lead_in_s: float = 0.0; drift_hz_per_min: float = 0.0
                 seed: int = 1                                   # CE with its own random latents, picture, q
@dataclass(frozen=True)
class Impulses: rate_hz: float = 0.0; amp_db: float = 30.0; amp_sd_db: float = 6.0; tau_ms: float = 1.0
                crash_per_min: float = 0.0; crash_db: tuple = (30.0, 60.0); crash_strokes: tuple = (5, 20)
                crash_len_s: float = 0.5; crash_tau_ms: tuple = (20.0, 100.0)
@dataclass(frozen=True)
class Agc: attack_ms: float = 2.0; decay_ms: float = 200.0; smooth_ms: float = 5.0
@dataclass(frozen=True)
class ChannelConfig:
    snr_db: float | None = None                      # in SNR_REF_BW_HZ (2500), on keyed samples
    preset: str | None = None                        # key of PRESETS, or None with doppler/delay below
    doppler_hz: float = 0.0; delay_ms: float = 0.0   # Watterson 2-sigma spread; two equal paths if delay > 0
    freq_offset_hz: float = 0.0
    drift_hz_per_min: float = 0.0; drift_tau_s: float | None = None   # None: linear; else warm-up settling
    wander_rms_hz: float = 0.0; wander_tau_s: float = 10.0            # Ornstein-Uhlenbeck
    tx_ppm: float = 0.0; rx_ppm: float = 0.0
    impulses: Impulses | None = None; agc: Agc | None = None
    neighbours: tuple[Neighbour, ...] = ()
    seed: int = 0
PRESETS = {"steady": (0.0, 0.0), "quiet": (0.1, 0.5), "moderate": (0.5, 1.0),
           "disturbed": (1.0, 2.0), "mps": (0.15, 2.0)}             # (doppler_hz, delay_ms)
@dataclass class Truth: u: complex128[n_pos]; f_hz: float64[n_pos]; t_pos_s: float64[n_pos]
                        noise_var_fe: float; impulse_mask: bool[n_fe]; agc_gain: float32[n_fe] | None
@dataclass class Sim: fe: complex64[n]; fs: int; t0_index: float; q: int; spec: FrameSpec; truth: Truth
def simulate(sym, spec, q, cfg, *, carrier_hz=1500.0, lead_in_s=0.0, keying=None, ook=False,
             pre_s=12.0, post_s=3.0) -> Sim
def gaussian_taps(n, spread_hz, fs, rng) -> complex128[n]   # hfchannel._gaussian_taps recipe at a general rate
```

The steps are applied in the order a real link sees them.

1. **Clocks.** Evaluate `ce.baseband` with time_scale = (1 + rx_ppm·1e-6)/(1 + tx_ppm·1e-6). This is exact, and it avoids the circular FFT resampling of `hfchannel.sample_clock_offset`. The RX clock also scales the audio carrier, so carrier_hz·(1 − rx_ppm·1e-6) − 1500 is added to the offset.
2. **Frequency trajectory.** f(t) = offset + drift + wander.
   - Drift is D·t/60 when linear. For warm-up it is (D/60)·τ·(1 − e^{−t/τ}), with t counted from key-up, lead-in included.
   - Wander is an OU process at 20 Hz: f[n+1] = a·f[n] + √(1 − a²)·σ·ξ, with a = e^{−Δ/τ}. It is linearly interpolated.
   - Integrate in float64 turns with a cumulative sum in 2^16-sample blocks, wrapping each block's running total. Then multiply by exp(2πj·wrapped).
3. **Watterson fading.** Two equal-power paths, each Gaussian-Doppler Rayleigh from `gaussian_taps`. The second path is delayed by round(delay_ms·4) samples, and the sum is scaled by 1/√2.
   - `gaussian_taps` re-implements `hfchannel._gaussian_taps`: shape at `max(64·spread, 8)` Hz, interpolate, normalise to unit power. The private function is not imported.
   - The `"steady"` preset is a unit gain.
   - Truth u at each position is the composite gain times the trajectory phasor.
4. **Neighbours.** Each is a CE signal with its own random latents, a synthetic header, keying and drift, added at rel_db relative to the wanted signal's power.
5. **AWGN.** P is the wanted signal's power over keyed samples, the `hfchannel.awgn` rule. Noise is complex white with E|n|² = P·FE_FS/(2500·10^(snr/10)), which matches `hfchannel.awgn` on audio (test C-1).
6. **Impulses and crashes.**
   - Clicks: Poisson, each a complex Gaussian burst with exponential decay τ, at a log-normal level amp_db ± amp_sd_db above the signal rms.
   - Crashes: Poisson; 5 to 20 strokes over crash_len_s, each a decaying noise burst with τ ∈ crash_tau_ms, at 30 to 60 dB.
7. **AGC.** env is a peak detector with the given attack and decay, run on |fe| smoothed over smooth_ms. The output is fe/max(env, floor), which models a 2.4 kHz receive AGC acting on the whole front end.

A full slot runs in under 8 s. `qrss_simulate.py` takes `in.(wav|qrsp|bin) out.(wav|npz) --slot ... [--snr] [--preset] [--doppler] [--delay] [--offset] [--drift] [--drift-tau] [--wander] [--wander-tau] [--tx-ppm] [--rx-ppm] [--impulse-rate] [--crashes] [--agc] [--neighbour DF:DB] [--lead-in] [--ook] [--frame] [--seed]`.
- Given a `.qrsp` or `.bin` input, it synthesises directly at baseband.
- Given a WAV, it converts to FE, applies the channel and converts back.
- An `.npz` output holds `fe` (c8), `fs`, `t0_index`, `q`, `frame`, and the truth arrays.

---

## 5. Codec boundary (`render.py`, WP7)

```python
def g_shrink(W) -> ndarray                 # 0.785*sqrt(W/(W+0.39))
def decoder_planes(S: float32[3,52800], W: float32[3,52800], mode: int) -> tuple[float32[158400], float32[158400]]
    # lat = g(W)*S/W (0 where W==0); W~ = max over groups with data of median(W[g, sent]);
    # w = min(sqrt(W/W~), 1); never-sent latents and groups without data: w = 0;
    # groups placed at g*52800 (latents.flat_to_latents layout, canonical (channel,h,w) C-order)
def render(codec, S, W, mode) -> PIL.Image  # codec.decode(lat, w)
def model_dir() -> str | None               # $QRSSTVAE_MODEL_DIR, else None (HF cache)
def load(precision="fp32")                  # codec.load_codec(model_dir(), precision=precision)
```

Only `render.py`, `picture.py` (for the codec ID) and `qrss_encode.py` touch the codec.

---

## 6. Receiver (WP5 and WP6)

### 6.1 Signal chain and rates

```
8 kHz audio --audio_to_fe--> FE 4000 Hz complex (300-2700 Hz) --blank, normalise--> FE'
FE' --A1 carrier lines, A2 preamble, A3 track-before-detect--> candidates
candidate --channelise(f_mix)--> CH 250 Hz complex (+-125 Hz about the signal)
CH --frequency path, timing fit, symbol MF, complex-gain KF/RTS, gate V--> Detection
CH --extract (plain or joint), header decode, round B, callsign windows--> PassResult
```

### 6.2 Front end (`frontend.py`, WP5)

```python
def audio_to_fe(x8k: float64) -> complex64      # x*sqrt(2)*exp(-j2pi 1500 n/8000) via dsp.to_baseband (exact 3/16
                                                # table), FIR LPF 1250 Hz (firwin 255, kaiser 8), decimate by 2.
                                                # Power preserved.
def fe_to_audio(fe) -> float64                  # inverse: upsample by 2, mix up, sqrt(2)*Re
@dataclass class Capture: fe: complex64; q: int; t0_index: float; fs: int = FE_FS
    # t0_index: FE sample index of nominal t0 on the receiver's clock (fractional allowed)
def blank(fe, k_sigma=5.0, guard_ms=2.0, win_s=1.0, hop_s=0.05) -> tuple[complex64, float32 keep]
def normalise(fe, keep, win_ms=100) -> tuple[complex64, float32 level]      # inverse AGC
def channelise(fe, fs_in, f_mix_hz, fs_out=CH_FS) -> tuple[complex64, Fraction f_mix]
    # f_mix quantised to 1/64 Hz; exact integer-turn mixing (n*f_num mod (64*fs_in))/(64*fs_in);
    # resample_poly(., 1, 16, window=("kaiser", 10))
def noise_psd(ch, fs, exclude_hz: list[float], win_s=10.0, *, audio_hz=None) -> float32[n_win]
    # bins 85-120 Hz off carrier; with audio_hz (the channel's 0 Hz in audio) only where the FE is flat
```

**Blanker** (spec §7: "blanked sample by sample before anything else"):
1. ν(t) is the running median of |fe|² over 1 s windows at 50 ms hop, interpolated.
2. Blank where |fe|² > k²·ν, with k = 5 (14 dB). Add a 2 ms guard and a 1 ms raised-cosine taper.
3. keep ∈ [0, 1] per sample.
4. A constant-envelope signal raises the median along with everything else, so CE is never blanked. Test R9 checks this up to +20 dB.

**Inverse AGC.** Divide FE by √(level), where level is the median |fe|² on kept samples over 100 ms windows, interpolated.
- A fast receive AGC multiplies signal and noise by the same g(t). At weak SNR the wideband power is noise, so normalising removes g(t) and leaves the noise flat.
- The residual signal-gain change is common to all symbols in a window, and the gain tracker follows it.
- A 100 ms window keeps the level estimate's jitter at about −28 dB, below anything a per-pass latent SNR cares about. R2 (no AGC, no loss) and R10 (fast AGC) pin this trade-off.

**Blanked share per position.** ψ_p = MF{keep}(t_p)/G0, the matched filter applied to the keep indicator after channelising. A position with ψ_p < 0.5 is erased.

**Noise floor** (`noise_psd`, D43). The floor under a signal is the median of the CH periodogram 85 to 120 Hz either side of it, less a 45 Hz guard about each neighbour, divided by the decimation filter's aliased response (`_noise_response`); with neighbours on both sides leaving no bin, the quieter side. Given the channel's audio frequency, it takes only bins the FE low-pass leaves within 0.1 dB of white (`FE_FLAT_DB`), counting the decimation's folds from 250 Hz away (`_fe_flat`). The low-pass is −0.10 dB at 310 Hz, −0.29 at 300 and −6.0 at 250, mirrored about 1500 Hz, so before 2026-10-12 a signal near 300 or 2700 Hz took half its bins off that skirt: on white audio the floor read 9.1 dB low at 305 Hz, 7.0 at 326, 2.5 at 360 and 9.6 at 2695, and now within about 0.5 dB. With no neighbour the flat side alone is the floor. When neighbours leave fewer than 3 Hz of flat bins clear (`FE_MIN_HZ`; a neighbour about 95 Hz inward of a signal at the edge covers its only flat side), each kind of bin errs one way: the flat bins, neighbour and all, can only read high, and the skirt's clear bins down to −20 dB (`FE_SKIRT_DB`), each divided by the low-pass's response there (`_fe_response`), only low, since a radio filter narrower than the FE takes them down further. The floor is then the geometric mean of the two medians, its error at most half their spread in dB. From the flat bins alone a weak pass beside a stronger neighbour was lost, and from the skirt alone, beside an equal one through a filter as narrow as the FE (the simulator's `fe_to_audio`), it read about 16 dB strong (figures in §6.10's known limits). A channel whose bins are all flat (audio from about 430 to 2570 Hz) is not changed at all. `--old-edge-noise` (`rules.EDGE_NOISE`) restores the old rule.

### 6.3 Acquisition (`acquire.py`, WP5; gate V in `track.py`, WP6)

**A1. Carrier lines.**
- Window [t0 − 2.5 s, t0 + 22.5 s]. The lead-in is deliberately excluded, so a lead-in can never move the preamble timing.
- STFT of FE′ with 4 s Hann frames (0.25 Hz bins) at 2 s hop. Sum the frames noncoherently along linear drift hypotheses of ±3 Hz/min in 0.5 Hz/min steps.
- CFAR against the median over bins within 300–2700 Hz. Under H0 the sum of M frames is Gamma(M). The threshold is `gammainccinv(M, 1e-4)` per bin.
- Return up to 16 `CarrierCandidate(f_hz, drift_hz_per_min, metric)`, local maxima at least 10 Hz apart.
- Searching the whole FE band covers ±400 Hz offsets of an uncalibrated Si5351 about any carrier frequency the sender chose.

**A2. Preamble search at each candidate.**
1. Channelise at f_c.
2. **Carrier notch:** y_hp = y − LP(y), with LP a zero-phase ±0.6 Hz low-pass. This removes the carrier and any lead-in.
3. Matched-filter y_hp with p, sampled at every CH sample.
4. Hypotheses:
   - δf ∈ [−0.5, +0.5] Hz in 0.05 Hz steps;
   - τ0 ∈ ±2.5 s in CH samples (0.13 T), with preamble symbol s sampled at round(τ0 + pos(s)·T·250);
   - ten 66-symbol (2 s) coherent chunks, each template made zero-mean.
5. Statistic: Λ = Σ_chunks |Σ_s (a_s − ā_c)·y(τ0 + sT)·e^{−j2πδf t_s}|² / (σ̂²·Σ(a − ā_c)²), with σ̂² = median|y_hp|²/ln 2.
6. Under H0, Λ ~ Gamma(10, 1). Threshold: `gammainccinv(10, 1e-7)` per cell.
7. Fine estimates: τ0 and δf by parabolas, phase from the chunk sums.

**A3. Whole-slot track-before-detect** (retrospective, run on the stored slot).
- Channels at each A1 candidate, plus a grid every 50 Hz across the band, each covering ±40 Hz. Overlaps are deduplicated by frequency.
- STFT with 4 s frames at 2 s hop and 0.25 Hz bins, starting at t0 − 10 s so a lead-in contributes.
- Viterbi over bins:
  - |Δbin| ≤ 2 per hop, which allows up to 15 Hz/min;
  - transition cost 0.5·|Δbin|·σ_score;
  - emission = CFAR-normalised power − 1.
- Path statistic Z = Σ emissions / √n_frames.
- The H0 distribution of Z is measured on noise-only CH by WP5, and the fitted Gumbel `TBD_GUMBEL_MU` and `TBD_GUMBEL_BETA` are committed in `acquire.py`.
- Output: `FreqPath(t_s, f_hz, weight)` plus the Gumbel p-value. Candidates pass to V at p < 1e-3.

**V. One verification gate** (`track.verify`, WP6). Every candidate from A2 or A3 goes through it.
1. Run the timing fit (§6.4).
2. Run one tracker pass (§6.5) **with the known-symbol positions withheld**: the carrier and unknown data only.
3. Compute Z_ref = Σ_known Im(a_k·m_k·ĉ_k*·û_k*) / √(Σ_known |ĉ_k·û_k|²·N_k/2), where ĉ_k is the template's imaginary part for a known symbol.
4. Because û never saw the known positions, Z_ref ~ N(0, 1) under H0. **Accept at Z_ref > 6.**

Timing search over about 3e4 effective cells gives Pfa ≈ 3e-5 per candidate. This is the same 6σ rule as spec §9's 6/√M. `receive_slot` merges detections within 5 Hz of each other, and a whole-slot candidate inside the own band of a signal found on its preamble, no stronger in A3 than that signal's own line, is not gated at all (§6.8, D44).

**Lead-in handling.**
- A1 and A2 exclude it, by window and by the notch plus zero-mean templates.
- A3 uses it as extra carrier.
- The tracker uses it as virtual known positions (§6.5) over the span an energy detector finds: 1 s windows of the notch-free MF output, Z > 4 per window.
- Measurements earlier than the last 8 s of detected lead-in are dropped, so the starting state comes from the last few seconds, as spec §2.8 asks.

### 6.4 Timing (`track.fit_timing`, WP6)

Position p is received at t_p = τ0 + p·T·(1 + δ) + γ·(p/n_pos)², on the receiver's clock and in CH samples.
- δ combines both sound cards and the beacon crystal. The search covers ±250 ppm, more than the ±200 ppm worst case, which slips 11.9 symbols over FULL.
- γ is fitted only when n_pos > 10,000 and only if it raises Z_ref by more than 1. Otherwise γ = 0.
- Timing is not a Kalman state (decision D17). Sound-card clocks do not wander on a symbol scale, and delay spread is at most 0.13 T.

```python
def fit_timing(ch, ch_fs, path: FreqPath, spec, tmpl_fn, tau0_init, ppm_range=250.0) -> Timing
```

1. Derotate CH by the frequency path. Form r(n) = y(n)·û_c(n)*, where û_c is the carrier-only estimate, a 0.3 Hz low-pass of y.
2. Grid δ in steps of 0.2·T/duration (3.4 ppm for FULL). For each δ, cross-correlate Im r with the sparse known-symbol template by FFT.
3. Refine (τ0, δ[, γ]) by Gauss–Newton on the coherent statistic, using the exact closed-form matched filter (§6.5).

The covariance comes from the curvature. It is reported, and R7 checks it against truth.

### 6.5 Carrier and complex-gain tracker (`track.py`, WP6)

Frequency, phase and gain are factorised so that every smoother is linear-Gaussian (decision D16). The spec's joint state is still fully estimated and reported.

**1. Frequency path.** The A3 Viterbi path, or the A2 estimate held constant, is fitted by a smoothing spline with knots every 60 s, weighted by emission. That gives f_A(t). Integrate θ_A(t) in float64 turns at CH rate, wrap it, and derotate: z_d = CH·exp(−2πj·θ_A).

**2. Symbol matched filter at every position** p of the tracker grid: lead-in virtual positions, frame symbols, and window positions.

m_p = (1/(250·T))·Σ_n z_d(n)·p((n/250 − t_p)/T)

- Evaluated directly with closed-form p over the 122 samples per position, chunked; about 7 M MACs per pass.
- z_d is band-limited to ±125 Hz and p to ±19 Hz, so the Riemann sum is exact.
- This is also where a neighbour 50 Hz away is rejected.

**3. Measurement model (A's template inside B's symbol-domain filter).** For every position:

m_p = u_p·c_p + e_p,  c_p = MF{ mean_template(·; μ, ν) }(t_p),

where `mean_template` is §2.4's E[s(t)] = a·e^{j(φ_μ + 2πθ_cw)}·e^{−v/2} (spec §8: re-modulate exactly as sent). Per-symbol (μ, ν) by class:

| Class | μ | ν | Notes |
|---|---|---|---|
| preamble, references, spare | known ±1 | 0 | always |
| header coded symbols | 0, then decoded ±1 | 1, then 0 | ±1 after a CRC-valid decode (round B) |
| unknown data | 0 | 1 | |
| EM soft data | x̂ | v | §8.3 |
| lead-in, detected span | (φ = 0) | 0 | c_p ≈ G0 |
| callsign window, plain units | (tails of neighbours) | from neighbours | |
| callsign window, Morse units | unknown: no measurement | | known after the callsign is known (FSK or OOK, picked by likelihood) |

- Var(e_p) = N_p + R_self,p, where R_self,p = KAPPA_SELF·Pu_p·(1 − MF{e^{−v}}(t_p)/G0) is the self-noise of unknown modulation.
- KAPPA_SELF is calibrated once from `ce.loopback_stats` and pinned in `track.py`.
- N_p is the noise variance of m_p. It is measured as the median CH power 85–120 Hz from the carrier (excluding detected neighbours) times the MF noise bandwidth, in 10 s windows, then cross-checked against the known-symbol residuals (§6.6).
- Pu_p is the smoothed local |u|² from the previous iteration. On the first iteration it is the global mean.
- Normalising gives μ_p = m_p/c_p with R_p = Var(e_p)/|c_p|². That is a direct circular observation of u_p, so one real Kalman gain sequence serves the complex state.

**4. Dynamics.** Constant velocity per position: x = [u, u′], F = [[1, 1], [0, 1]], Q = q·Pu·[[1/3, 1/2], [1/2, 1]].
- q is chosen by **maximum innovation likelihood** over a 13-point log grid, equivalent to effective spreads of 0.01 to 3 Hz.
- That q absorbs the Doppler spread plus the wander the path does not follow (spec §7), and is reported as `doppler_hz`.

**5. RTS smoother** over the whole grid, lead-in included. It gives û_p and P_p. The forward pass alone is the causal "live" filter (spec §7) and is exposed as `kalman_rts(..., causal=True)`.

**6. Re-centring.**
- Δf(t) = arg(Σ û_{p+1}·û_p*·|û|²)/(2πT) over 30 s windows.
- Add the spline-smoothed Δf to f_A and repeat steps 2–5.
- Run two outer iterations, and a third only if |Δf| > 0.02 Hz anywhere.
- The spline continues straight through the callsign windows.

**7. Report** (`TrackReport`):
- offset (f_A at t0, as absolute audio Hz);
- drift (linear fit, Hz/min);
- wander rms (the 3–60 s band-passed residual of f_A + Δf);
- doppler_hz, Z_ref, N, the timing and ppm;
- `slip_free`: no |arg(û_{p+1}/û_p)| > π/2 where |û|² > Pu/4.

The model has no PLL and no phase state. A deep fade lets û pass near zero, so there is nothing to slip.

```python
def freq_path_spline(path: FreqPath, knots_s=60.0) -> Callable
def symbol_mf(ch, ch_fs, timing, pos: ndarray) -> complex128[len(pos)]
def kalman_rts(mu, R, q, Pu, causal=False) -> tuple[u_hat, P, loglik]
def track(ch, ch_fs, spec, det: Detection, classes, cw_known=None, prior: EmPrior | None = None) -> TrackResult
def verify(ch, ch_fs, spec, cand) -> Detection | None
```

### 6.6 Extraction and weights (`demod.py`, WP6)

For each non-reference stream symbol k at position p:

```
y_k  = Im(m_p·û_p*) / (K·|û_p|²)                                        # estimate of x_k (data) or ±1 (header)
σ²_k = [N_p/2 + (A0² + K²)·P_p/2] / (K²·|û_p|²·ψ_p²)  +  D_PASS          # erased (∞) if ψ_p < 0.5 or |û_p|² < 1e-4·median
```

- The ½ factors are there because only the imaginary dimension counts. Design B left them out.
- D_PASS is the per-pass phase-modulation distortion in latent units, initially 10^(−13.2/10) = 0.0479, re-measured by WP6 on Gaussian latents with `ce.loopback_stats` and pinned.

**Calibration.**
- On the reference symbols, κ = mean((y − a)²)/mean(σ²) over the pass.
- If κ ∈ [0.5, 2], every σ² is scaled by κ. Otherwise the pass is flagged `suspect` and κ is not applied.
- This keeps W honest: R3 tests the predicted variance against the measured MSE.

**Header LLRs:** llr_e = 2·y_k/σ²_k on the header coded symbols.

**Data estimators**, per precoder block B of M ∈ {64, 32, 8} symbols:
- **Plain** (spec §7): â = c ⊙ WHT(y_B), with a single variance v = mean(σ²_B) for the block.
- **Joint** (spec §2.4's "64×64 solve" in closed form; **the default**):
  - The precoded block has covariance I and H is orthogonal and symmetric, so the LMMSE estimate is (HΣ⁻¹H + I)⁻¹HΣ⁻¹y = H·diag(d)·y, with d_k = 1/(1 + σ²_k).
  - Its diagonal gain is γ = mean(d), so the unbiased estimate is â = c ⊙ WHT(d ⊙ y)/γ.
  - The variance is v = [mean(d²σ²) + mean((d − γ)²)]/γ². The second term is crosstalk.
  - With flat σ², joint and plain are identical. With one symbol at σ² = 100 among 63 at 0.1, plain gives v = 1.66 and joint v = 0.117 (checked by hand).
  - The flag `estimator="plain"` keeps the spec's method, and R15 is the comparison spec §14 asks for.

The weight is w = 1/v for every latent of the block, as float32. W then equals the per-latent SNR. A noiseless pass reports about 13.2 dB, not infinity.

```python
def extract(m, track, spec, q, estimator="joint") -> tuple[float32 z, float32 w, float32 llr, dict diag]
```

### 6.7 Callsign windows (`cwid.py`, WP6)

```python
def window_soft(ch, ch_fs, track, spec) -> float32[n_win, 176]
    # per Morse unit: coherent decision between key-up (0 Hz) and key-down (-1 cycle/unit) using u_hat
    # interpolated through the window (the plain units anchor it); LLR-like metric, + = key-down
def read(soft: float32[176] (windows and passes summed), ook=False) -> str | None   # run-length Morse decode
def match(soft, callsign) -> float                                                 # Z of the expected +-1 keying
def keying_type(ch, track, spec, callsign) -> str                                  # "fsk" | "ook" by likelihood
```

- Windows from all four positions and from many passes sum before `read`, because the windows are identical in every pass.
- Once the header gives the callsign, the receiver forms the expected keying pattern. `PassResult.cw` reports `CwIdResult(text, z_match, keying, agrees)`, where agrees = (text == header callsign) or z_match > 6.
- Once the keying is confirmed with z_match > 6, the Morse units become known positions for a final tracker pass.

### 6.8 Orchestration (`receiver.py`, WP6)

```python
def receive_pass(cap: Capture, spec, det: Detection, prior: EmPrior | None = None,
                 estimator="joint") -> PassResult
    # round A: track with known symbols; extract; header decode
    # round B (if header CRC-valid): header symbols known; callsign keying known; re-track; re-extract
def receive_slot(cap: Capture, spec=FULL, live_only=False) -> list[PassResult]   # A1 -> A2 (+A3) -> V -> receive_pass
def receive_wav(path, q: int, start_unix: float | None = None, spec=FULL) -> list[PassResult]
```

If `start_unix` is None, the WAV is assumed to start at t0 − 12 s, as `qrss_transmit` writes it.

**A found signal's own band is not gated again** (`_owned`, D44). `detect` gates candidates in order, preamble hits first. A whole-slot (A3) candidate inside the own band of a signal already found on its preamble is skipped when A3 found it no stronger than that signal's own line (the largest A3 Z within 5 Hz of its line or its centre; `Detection.a3_z`). The band is within GHOST_HZ (20 Hz) of a CE carrier, or from 22.1 Hz below an L pilot to 3.1 Hz above it, judged by the candidate's line (a CE carrier or an L pilot). Those are a strong CE's sidebands about 12.4 Hz either side, the L copy of its carrier, and lines in an L signal's data band, which were gated one by one only to be refused or dropped as ghosts: 1.1 to 8.9 CPU s each on SHORT at −12 and −6 dB, and about a minute each of the end-of-slot search's 180 s on FULL. A skipped candidate is not counted against `max_candidates`. Over SHORT at −12 and −6 dB, a CE alone and beside an L 60 Hz up, CE-only and joint, the detections are identical and acquisition plus gating took 2.8 to 8.3 CPU s against 5.3 to 24.9. Only a preamble-found signal owns a band, and only against lines no stronger than its own: a whole-slot or hint detection can be the weaker of two lines, so it owns nothing, and a station whose preamble faded can be stronger than one whose preamble was heard. In the review's three cases (a CE 9 dB stronger 15 Hz up and one 4 dB stronger 17 Hz up beside a weak CE found on its preamble, and a CE 9 dB stronger inside an L's band, each with its preamble zeroed) the stronger one is gated and received with its header, as under the old rule. A hint has no A3 Z and is always gated. The price is the ghost rule's without its timing condition: a station heard only by the whole-slot search inside a preamble-found one's band, and weaker there in A3, is not received. `--old-gating` (`rules.GATING`) gates and counts every candidate as before. `spec` may also be a tuple of frames, at most one per waveform. One that holds an L frame is the joint receiver of §6.10; CE alone, as a frame or a tuple of one, is the receiver described here, call for call.

`qrss_receive.py WAV|NPZ --slot 2026-10-09T06:00Z [--start ISO] [--frame ...] [--just-ce | --just-l] [--store DIR] [--no-store] [--passband DIR] [--no-passband] [--estimator joint|plain] [--image OUT.png] [--model DIR] [--precision] [--old-edge-noise] [--old-gating]` keeps the slot's FE stream in the passband store (default STORE/passband) and prints, per pass:
- frequency at t0 (the report's `offset_hz`, absolute audio Hz), drift, wander, Doppler and ppm;
- Z_ref and SNR₂₅₀₀;
- the header, and the callsign-window text with its match;
- mean W in dB;
- the association result.

With neither `--just-ce` nor `--just-l` it listens for both waveforms, and each pass line names its waveform and frame (an L pass's frequency is its pilot's). `--frame` takes either waveform's name ("short" or "l-short") and the other waveform listens for the same shape (`lframe.twin`). An .npz's own frame fixes its waveform, so a flag that excludes it is refused.

### 6.9 Shared types (`types.py`, WP1)

```python
@dataclass class Timing: tau0: float; ppm: float; gamma: float; z: float; cov: ndarray
@dataclass class FreqPath: t_s: ndarray; f_hz: ndarray; weight: ndarray
@dataclass class CarrierCandidate: f_hz: float; drift_hz_per_min: float; metric: float
                                   waveform: int | None = None    # a hint's waveform; None: an A1 line (any)
@dataclass class Detection: f_hz: float; path: FreqPath; timing: Timing | None; z_ref: float
                            method: str; lead_in_s: float          # method: "preamble" | "hint" | "tbd" | "template"
                            waveform: int = WAVEFORM_CE            # the frame this candidate is gated with
@dataclass class TrackReport: offset_hz: float; drift_hz_per_min: float; wander_hz_rms: float
                              doppler_hz: float; ppm: float; snr2500_db: float; z_ref: float
                              kappa: float; slip_free: bool; suspect: bool
@dataclass class CwIdResult: text: str | None; z_match: float; keying: str; agrees: bool | None
@dataclass class EmPrior: x_hat: float64[n_data]; v: float64[n_data]   # precoded-symbol domain of that pass
@dataclass
class PassResult:
    uid: str                      # f"{q}_{round(f_hz*1000)}_{sha8 of z bytes}"
    q: int; frame: str; waveform: int                 # 0 = CE; 1 = L, frame "l-...", f_hz its pilot
    f_hz: float; timing: Timing; report: TrackReport
    z: float32[n_data]; w: float32[n_data]            # air order, unscrambled, unprecoded
    hdr_llr: float32[2474]; header: HeaderFields | None; cw: CwIdResult | None
    ch: complex64[...]; ch_fs: int; ch_t0_index: float; f_mix_hz: Fraction   # narrow capture kept for EM
    psi: float32[n_pos]; estimator: str; em_round: int = 0
    def save(self, path) ; @classmethod def load(cls, path)   # .npz plus a JSON metadata entry
```

An L pass fills the same fields: waveform 1, its frame's name ("l-full" and so on), f_hz and the report's offset at the pilot, z and w in air order as for CE (two latents per L symbol), hdr_llr in coded-bit order, and psi over the L frame's n_pos. The store, association and `render` read either waveform's passes unchanged, so CE and L passes of one segment add.

A FULL pass at 250 Hz keeps about 3.6 MB of capture for EM. **This does not replace spec §7's 48 h passband store, which is also built** (spec owner, 2026-10-09): `frontend.PassbandStore` (WP5) keeps the whole 4 kHz complex FE stream as int16 I/Q in hourly files on disk (about 1.4 GB/day), with `write(t0, x)`, `read(t_start, t_end) -> complex64`, expiry after 48 h, and survives restarts. `em.template_search` (WP9) reads past slots from it to find passes too weak to detect when they arrived (retroactive detection). Wiring (integration review, 2026-10-09): `qrss_receive.py` writes every slot's raw FE stream with `PassbandStore.write_capture`, `PassbandStore.capture(q, dur)` reads a slot back as a `Capture`, `template_search`/`receive_template` take the store with `q=`, and `em.retro_detect(store, key, pb, spec)` searches every stored slot without a member of the picture, receives and associates what it finds; `qrss_decode.py --retro` runs it.

### 6.10 Waveform L (`acquire`, `ltrack.py`, `ldemod.py`, `cwid`, `receiver`, `live`; 2026-10-12)

L's frame and signal are `lframe.py` and `lwave.py` (format modules). What the receiver takes from them:
- T_L = 2T = 0.060625 s (16.495 Bd). Positions count the frame's own symbols (`spec.t_sym`), and every duration comes from the frame: L_FULL keys up to 1782.648 s after t0 (`keyed_end_pos·t_sym`), FULL to 1782.663 s.
- One complex symbol carries two latents, √2·Re on latent 2k and √2·Im on 2k+1 (`lframe.unpair`). The preamble is 330 QPSK symbols. The header block is 1,320 symbols, 1,237 of which carry the 2,474 coded bits two at a time (there is no spare). L_FULL has 1,770 references against FULL's 3,539.
- The data occupy 3.10 to 22.07 Hz below a pilot tone. The pilot sits f_p = 12207/970 = 12.5845 Hz above the data centre (the band edge, 920/97 = 9.4845 Hz, plus 3.1 Hz), and f_p·T_L = 12207/16000 turn exactly. An L signal is named by its pilot's frequency f.
- A callsign window is 192 L symbols, one Morse unit each (60.625 ms, as in CE). L keys by FSK only.

**Selection.** `receiver.frames(spec)` lists the frames a selection names, CE's first, at most one per waveform, and `is_joint(spec)` says whether one is L. A selection of CE alone, a frame or a tuple of one, is the CE receiver of §6.1 to §6.9 call for call (D37). One holding an L frame is the joint receiver: every candidate carries its waveform and is gated, tracked and demodulated with that waveform's frame. `receive_pass` dispatches on the frame's waveform and raises `ValueError` for a detection of the other one.

**Joint acquisition** (`acquire.acquire` with a tuple of frames; one frame keeps exactly the path of §6.3).
1. A1 once, unchanged: up to 16 lines at least 10 Hz apart.
2. At each line one channelisation and notch (`_a2_input`) feeds the A2 of every enabled waveform. A2 takes a `PreambleKind`:
   - `CE_PRE`: the real ±1 preamble at T in 66-symbol chunks, phase offset −π/2;
   - `L_PRE`: L's QPSK preamble at T_L in 33-symbol chunks, so both have ten 2 s chunks. The notched channel is mixed up by f_p from nominal t0 before L's matched filter, which puts the data centre at 0 Hz and what the notch leaves of the pilot at +12.58 Hz, outside the filter's ±9.48 Hz.
   - The chunk templates are complex and zero-mean, W = b*·e^{−j2πδf·t} normalised by Σ|b|², and Λ is Gamma(10, 1) on noise as before. Zero-mean costs L about 0.15 dB.
   - L's phase is the pilot's at t0: arg ΣS·exp(−j2π[(δf + f_p)·τ0 − ½·rate·τ0²]).
3. **Pilot hypotheses** (`pilot_lines`). During the preamble the pilot carries 10.5% of the power and the data band's spectral lumps sit 10 to 20 Hz below it, so a lump can take A1's line and, by the 10 Hz separation, hide the pilot. For each line f, A1's best (bin, drift) cell in (f + 3.1, f + 22.1] Hz becomes one more line for A2-L, whether or not it clears A1's threshold. One within 0.5 Hz of a line or an earlier hypothesis is dropped.
4. **Hints** are frequencies or (frequency, waveform) pairs (D40). A bare frequency is searched in every enabled waveform. An L hint whose own A1 cell does not clear A1's threshold gets A2-L at f and f ± 1 Hz (`L_HINT_OFFSETS_HZ`), the best of the three kept, since A2's δf reaches only ±0.5 Hz.
5. **Arbitration**, of the passed hits:
   - per waveform, `drop_symbol_rate_aliases` at that waveform's own T: a hit at f ± k/T (k = 1, 2, within 0.5 Hz), within T/2 of a stronger hit's timing and with a 2 s Λ 3× smaller, is that hit;
   - for L, `drop_band_owned`: an L hit whose pilot lies 3.1 to 22.1 Hz below a stronger L hit's, within 3 T_L of its timing, is in that signal's data band and is that signal. It is there for the −1/T_L alias 16.5 Hz below the pilot, which passes A2-L in 3 of 12 SHORT preambles at −12 dB and in 12 of 12 at −6 and 0 dB. Its timing peak is broad (measured −1.9 to +0.8 T_L off), hence 3 T_L;
   - across waveforms, `merge_hits`: hits within 5 Hz and T_L/2 of each other are one signal, kept in the waveform with the smaller p;
   - each survivor's lead-in is measured at its own T.
6. **A3** runs once, over the longest enabled frame, and each hit carries its spectral shape. With E the CFAR-normalised power − 1 (§6.3) and b the path's bin in each frame, over the frames that overlap no callsign window of any enabled frame and start after keying:
   - S_below = Σ E over the bins 3.10 to 22.07 Hz below b (76 bins of 0.25 Hz), S_above over the mirror band above;
   - z1 = (S_below − S_above)/(sd·√2) and z_above = S_above/sd, with sd one band sum's noise sd, allowing for the frames' half overlap. Both are N(0, 1) on noise;
   - **L** when z1 > 4 and z_above ≤ 3 (`SHAPE_L_Z`); **CE** when z_above > 3 and |z1| < 3; otherwise ambiguous, gated as both.
   
   A weak L signal's Viterbi path can run in its data band rather than on the pilot (L_FULL at −28 dB: 6 Hz below it). With L enabled each hit is also re-anchored (`l_pilot_bins`): a Viterbi from −6 to +26 Hz about the path's line, on an emission of the bin plus the one-sided band below it, which peaks at the pilot. The verdict takes either shape's waveform where the other says nothing, and is ambiguous where they disagree (`_shape_verdict`). The anchored path only places the pilot; A3's statistic and its Gumbel are the pilot-only Z of §6.3.
7. **Order** (`Acquisition.detections`): preamble hits of every enabled waveform by Λ; hints A2 did not pass, in their waveform; A3 hits in the waveform their shape says, an ambiguous one as CE; then an L copy of each ambiguous hit, and last the hits whose shape names a waveform that is not enabled, gated as the one that is. An A3 detection within 10 Hz of an earlier A3 one of the same waveform is left out.

**Tracker** (`ltrack.track_l`). The channel is centred on the pilot's path, so derotation puts the pilot at 0 Hz. The model is one complex gain u, the data band's reference gain (a reference symbol a arrives as u·a), fed by two independent streams:
- **Data band.** z_dd = z_d·exp(+j2π·f_p·T_L·τ(n)), the data mixed to 0 Hz in sender time (τ from the timing model), then L's matched filter at every position. A receiver-time mix would drift 2.2 turns over L_FULL at 100 ppm. The template is c = g·μ, g the class gain, with Var = N + Pu·g²(ν + D) and N = PSD/T_L.
- **Pilot.** An RRC at 2T_L on z_d (`PILOT_K` = 2), read at even positions, where its noise is white. It lets in 0.9% of the data's power, against 23.7% for an RRC at T_L. Its template is `lwave.pilot` through the same filter times the clipper's pilot gain per section, over A_ID·G_REF_ABS, with N_pl = PSD/(2T_L) and a residual of K_PL_DATA = 0.016 of the signal power. With the keying unknown, units 8 to 183 of every window, plus the filter's reach, give no pilot measurement; with it known the FSK is in the template.
- **δ**, the pilot's gain over the data band's (the rig's passband phase, up to 0.15 rad, or a delayed path), so the pilot measures u·δ·c_pl. It is fitted once per pass by least squares at the known symbols with a prior sd of 0.25 about 1, profiled out in the timing refinement, and held at 1 inside gate V.

The two streams are fused per position by inverse variance and fed to §6.5's Kalman filter and RTS smoother unchanged, with q chosen on `track.q_grid(L_T_SYM)`: the 13 spreads of §6.5 at L's symbol, 8× CE's q per position. In round A the data band measures only the preamble and the references, 1 in 16 of the frame, and the pilot carries the track between them as CE's carrier does. Round B adds the header as known. L's spread copy is linear, so it enters the data template's mean (`make_classes_l`) and nothing is removed from the channel (contrast `track.remove_spread`).

**Timing.** `fit_timing_l` correlates the known symbols with MF{z_d·e^{+j2πf_p·t}}·conj(LP z_d) over lags and clock errors, the pilot's low-pass (0.03, 0.3, 1 or 2 Hz) being the phase reference. `refine_timing_l` maximises the known symbols' correlation with the pilot-only gain track, re-mixing z_dd at each trial timing and profiling δ's phase out against its prior. τ0 then comes mostly from the pulse envelope, and a rig's passband phase cannot bias it: 0.13 CH sample rms on l-short at −12 dB (6 seeds, from a blind gate V). `report_l` quotes SNR₂₅₀₀ on the frame's keyed power (`L_KEYED_POWER`, per frame) rather than on the references' |u|².

**Gate V for L** (`ltrack.verify_l`). The known symbols are withheld and δ = 1, so û is the pilot's alone. With d the known symbols' complex template:
- t_k = Re(m_k·conj(û_k·d_k)), V_k = |d_k|²(N_k|û_k|² + |c_k|²|û_k|²P_k + N_kP_k)/2, w_k = |d_k|²|û_k|²/V_k;
- Z = Σ w_k·t_k / √(Σ w_k²·V_k). This is `track.z_ref_stat` in its general form; CE's is d = j·Im c (§6.3).

On l-tiny at −12 dB it reads 0.97 of √(Σ 2|uc|²/N), where a CE-style Im-only form would read 0.68. A CE signal also correlates under the L hypothesis: with the pilot taken on a CE carrier or its sideband, L's references see CE's sideband in L's data band, at about 1.5σ of the true noise (Z 6 to 12 on CE SHORT at −12 and −6 dB). So the gate checks the known symbols' residual against the noise model,

R = Σ ω_k·|m_k − û_k·c_k|² / Σ ω_k·(N_k + |c_k|²P_k + |û_k|²g_k²D_k),  ω_k = w_k²|c_k|²|û_k|²,

and uses Z_ref = Z/√max(R, 1), accepted at Z_ref > 6. Measured R on gate-V tracks: an L signal 0.77 to 1.54 (l-short, steady to disturbed, −6 to −25 dB); noise 1.05 ± 0.20 at a fixed timing (l-tiny, 40 seeds); a CE signal under the L hypothesis 2.8 to 30. Z_ref only shrinks, so its tail on noise stays N(0, 1)'s or narrower: at a fixed timing, 60 noise seeds gave mean 0.02, sd 0.78, max 1.86; blind, 20 gave mean 0.5, sd 1.2, max 3.2. The tail out at 6 is not measured.

**CE under the joint gate** (`receiver._gate`, D39). CE's gate V passes an L signal too (Z_ref 6 to 11 on l-short at −12 and −6 dB: L's references correlate with CE's in the data band). In the joint receiver a CE accept is re-checked with the same ratio over CE's known symbols, weighted as CE's Z_ref weighs them (`_ce_resid_ratio`), and refused when R > 2 (`CE_RESID_MIN`) and Z_ref/√R ≤ 6. Measured R:
- CE SHORT 1.02 to 1.07 at −12 dB and 1.29 to 1.35 at −6 (CE's own distortion is not in the model), CE TINY 0.93 to 1.48;
- fading CE 0.55 to 1.31 on TINY, and up to 2.16 on SHORT at Z_ref 48, which is kept;
- an L signal under CE's gate 3.6 to 4.3 at −12 dB and 12.7 to 15.4 at −6;
- CE's own ±11.5 Hz sideband ghosts 5.2 to 6.2, refused.

It costs about 0.09 s per CE accept on TINY. The CE-only receiver does not run it (D37).

**Extraction** (`ldemod.extract_l`), from the tracker's data-band m, u, P, N and ψ and the class gain g (`G_*_L`):

```
r   = m·u* / (|u|²·ψ·g)                  E[r | a] = a
y   = unpair(r)                          √2·Re, √2·Im on latents 2k, 2k+1
s2n = (N + g²·P) / (g²·|u|²·ψ²)          per latent, the same on I and Q
```

- There is no ½ (unpair's √2 puts each half of the complex noise on its own latent) and no carrier term (the pilot does not reach the data matched filter). A symbol is erased where ψ < 0.5 or |u|² < `U_MIN_REL` of its median, as for CE.
- Header unknown: y_x = √(1+ρ)·y and s2n_x = (1+ρ)·s2n, and the unknown copy adds ρ of variance. Header known: y_x = √(1+ρ)·y − √ρ·h, with nothing extra.
- σ² = κ·s2n_x + D_PASS_L·(1+ρ) + extra, and CE's `block_estimate` (§6.6) does the rest.
- **Pinned constants**, from 24 noiseless loopbacks through the clipper (L_MEDIUM and L_FULL, Gaussian and tanh latents, precoded as sent, 2026-10-12):
  - class gains against the references': G_REF_L = 1, G_HDR_L = 0.996, G_DATA_L = 0.881, G_PRE_L = 0.995;
  - the pilot's gain through the clipper per section: 0.656 (preamble), 0.664 (header), 0.703 (data), 1.0 where the pilot runs alone (lead-in, window units 8 to 183); the references' own gain G_REF_ABS = 1.006;
  - distortion per latent: D_PASS_L = 0.058 (measured 0.0583, 12.34 dB), D_REF_L = 0.0236, D_HDR_L = 0.0201, D_PRE_L = 0.0200.
- **κ** is CE's calibration (§6.6) on L's residuals: the references' complex residual against D_REF_L, and the data's unit-RMS contract E[y_x²] = 1 + extra + D_x + κ·s2n_x. L has about half CE's references and they carry much more of u (the pilot is 14% of the power, CE's carrier 53%), so the reference branch allows for each measured reference's pull on u: κ = (mean|e|² − D_REF_L + mean P_m)/mean(N + P_u), P_m the measured references' P and P_u the others'. On `track_l` passes of l-short in moderate fading CE's form read 0.2 to 0.7 where the honest κ was 0.9 to 1.2, and overstated W by up to 1.7 dB.

**Header.** llr = 2·y_h/(κ·s2n_h + D_HDR_L), scaled by G_HDR_L, with coded bit 2j on I and 2j+1 on Q of header symbol j, so the LLRs are in coded-bit order and mean what CE's do. While the header is unknown the spread copy adds `demod.spread_llr(y_x, κ·s2n_x + D_x)`. `decode_header` tries the sum, then the spread copy alone, then the block alone, each accepting version 2 only (`L_HEADER_VERSIONS`): L has no version 1, so an L pass never switches to `frame.plain`, and the LLRs kept are the sum's.

**Callsign windows** (`cwid`). A Morse unit is `spec.cw_window // CW_UNITS` symbols: 2 for CE, 1 for L. With z_d derotated on the pilot, every function reads an L window as it reads CE's, except that `keying_type` returns "fsk" without computing anything, OOK raises, and `window_llr` takes the FSK tone amplitude from units 8 to 183 alone, since units 0 to 7 and 184 to 191 of an L window hold the pilot's ramp and the data pulses' tails. Round B takes the keying as known (FSK) when the header's callsign matched at Z > 6.

**Orchestration** (`receiver.py`).
- `_receive_pass_l`: gate V (`verify_l`) for an unverified detection; round A (`track_l`, `extract_l`, `decode_header`, the callsign windows); round B when the header decodes, with `make_classes_l(spec, header.encode(hdr), prior)`, the keying if it matched, round A's timing and frequency, and one outer iteration; then `report_l`.
- **One signal, one identity** (D38). Signals are compared at their band centre, `centre_hz`: f for CE, and f − 9.5 Hz for L (`L_EXCLUDE_OFFSET_HZ`). An L neighbour's 45 Hz guard in another signal's noise floor, symmetric about that centre, then covers [f − 54.5, f + 35.5] Hz, all of its spectrum. Merging, "already found", the noise exclusion and the merge after tracking compare centres within 5 Hz across waveforms; where a CE and an L result share a centre, the larger Z_ref stands for the signal, and its waveform with it.
- `detect` (`_detect_joint`): candidates in `detections` order, with `max_candidates` and `budget_s` one budget across both waveforms. A candidate within 5 Hz of a found signal or a `skip_hz` centre is not gated, nor a whole-slot candidate in a preamble-found signal's own band and no stronger in A3 than its own line (§6.8: by the candidate's line, and the found signal's waveform); each gate's noise floor leaves out the preamble hits' and the found signals' centres; then one detection per centre, and aliases and ghosts per waveform at its own symbol.
- `receive_slot` (`_receive_slot_joint`): each detection received with its own waveform's frame (the selection's twin, `lframe.twin`, for a waveform not in it). After tracking, the ghost rule applies within a waveform at its own symbol (GHOST_T = 3 symbols), the suspect-neighbour rule across both by centre (`_suspect_beside`), and one pass per centre.
- `confidence_db` uses the pass's own frame when none is given, and places L's latents 2k, 2k+1 and coded bits 2j, 2j+1 at their symbol's time.

**Live listener** (`live.py`, `qrss_listen.py`).
- `LiveConfig.waveforms`: None is CE alone, the listener as it was before L, and so is (WAVEFORM_CE,). `specs` is `spec`'s twin in each waveform listened for (a shape with no twin, such as full-v1, is a `ValueError`). `selection` hands the receiver the tuple when it holds an L frame and the CE frame itself otherwise, so CE alone makes exactly the calls, with the same object, that it made before L.
- The slot lasts as long as the longest enabled frame (`frame_seconds` of the tuple), and a tile's confidence time is measured against that.
- The end-of-slot search verifies at most 12 candidates in at most 180 s (`FINISH_MAX_CANDIDATES`, `FINISH_BUDGET_S`): one budget across both waveforms, in `detections` order.
- Tiles are matched by band centre on the same dial (`centre_of`). A pass of the tile's waveform updates it; a pass of the other waveform only with a strictly larger Z_ref, and the tile then becomes that waveform. `_keep` holds one detection per centre by the same rule.
- `Tile.waveform` is "CE" or "L" (a tile from before L reads as CE) and `Tile.frame` the pass's frame; an L tile's f_hz is its pilot. state.json gains "waveforms", the names listened for.
- `in_band`: an L signal is in band when its data's low edge, f − 22.07 Hz, is at or above 300 Hz and its pilot at or below 2700 Hz.
- Hints carry the waveform of the stored pass that names them: its frame's ("l-..." is L), else its "waveform" field, else CE (D41).
- `qrss_listen.py` listens for both unless given `--just-ce` or `--just-l`; the app starts it with only `--state` and so hears both. `--frame` names the shape in either waveform, and a CE-only frame such as full-v1 with L enabled is a `SystemExit` that names `--just-ce`.
- `acquire.live_end_s()`, how far past t0 a live acquisition reads, is computed from CE's preamble. L's lasts as long in time (330 T_L = 660 T), so it holds for both; it is not checked for frames whose twins differ.

**Measured** (simulated; SNR in 2500 Hz; where latent SNR is given, the genie's is in brackets):
- A2-L at the true pilot on l-short: 8/8 at −25, −28 and −31 dB, Λ medians 105 / 67 / 43, timing rms 0.040 / 0.057 / 0.083 T_L. The joint live chain on l-short, limited by A1: 6/8 at −24 dB, 2/8 at −28, 0/8 at −31.
- Shape on SHORT at −15 / −20 / −25 dB: L's z1 +155 / +50 / +16; CE's |z1| ≤ 2.3 with z_above 29 to 34 / 6 to 10 / −0.2 to 3.2. On noise, z1 has mean +0.01 and sd 1.025. L_FULL along its true path: z1 21.5 / 13.3 / 5.0 at −28 / −30 / −34 dB. A3 finds L_FULL at −28 and −30 dB, not at −34. A3's emission for L is about 6 dB behind CE's, the pilot holding 13.8% of the power.
- Joint acquisition costs about twice a single one: TINY 0.8 s against 0.42 s, SHORT 2 to 3 s against 1.1 to 1.3 s.
- Round A against the genie in data SINR, l-short at −12 dB, two seeds: steady −0.07 / −0.17 dB, quiet −0.29 / −0.46, moderate −1.19 / −1.81. Round B adds +0.02 to +0.18. The moderate loss is noise in the pilot track, not δ: using the true pilot gain as the data gain costs only 0.05 to 0.10 dB there. Timing after `track_l` is 0.005 to 0.010 T_L rms (l-short, −12 dB).
- The joint receiver, blind: l-tiny at −12 dB 6.46 (6.73), 6.70 (6.84), 7.73 (7.31), 6.67 (6.78) dB; l-short at −12 dB 6.97 (6.97), 6.71 (6.91), 7.04 (6.97); at −6 dB 10.16 (10.06), 9.78 (9.92), header and callsign decoded; at −24 dB −3.63 (−3.62), −4.10 (−3.72), header decoded; at −28 dB −7.50 (−7.57), header decoded (one seed, steady path). Every case is within about 0.4 dB of the genie, either side. Noise alone gives nothing, and an L and a CE signal 60 Hz apart at −12 dB are both received, either way round.
- A CE recording through the joint path gives the pass the CE-only receiver gives. Listening for L costs CE recordings time: a CE SHORT .npz at −12 or −6 dB took 7.9 to 8.1 s through `qrss_receive.py` by default against 6.8 s with `--just-ce` (wall clock, shared CPUs; 12.6 s against 7.7 before D44).
- Live, fed as audio: a TINY slot with CE at 1500 Hz and an L pilot 80 Hz above, both at −12 dB. Both waveforms: CE 4.71 dB and L 6.80 dB, 5.7 s; `--just-l`: L 6.79 dB, 3.6 s; `--just-ce`: CE 4.71 dB plus a junk tile, 3.9 s. The same on SHORT: CE K1ABC 4.68 dB and L W1AW 6.57 dB with headers and callsigns, 30.3 s; `--just-ce` 21.3 s, `--just-l` 22.2 s.

**Known limits.**
- A1 finds an L pilot only to about −26 dB (the pilot is 10.5% of the preamble's power), so live L acquisition stops there, against −31 dB for A2-L itself. An L-shaped A1 statistic is the fix (§13).
- Weaker L is found at the end of the slot by A3 and gate V, to about −30 dB on L_FULL (found at −28 and −30, not at −34).
- The inverse AGC (§6.2, 100 ms median) follows L's envelope above about −6 dB SNR₂₅₀₀: the genie on L_SHORT at +10 dB drops from 12.13 to 9.89 dB of latent SNR through it, and a strong L signal modulates the others in the band.
- Near the band edges the noise floor read low for both waveforms until D43 (2026-10-12): its region 85 to 120 Hz from the signal reached past the front end's passband. Through audio at −12 dB (TINY, both waveforms, one seed), old rule against new: CE alone at 305 to 340 or 2680 to 2695 Hz reported −2.4 to −4.4 dB, now −11.9 to −12.6, its latent SNR 3.8 to 4.3 dB, now 4.1 to 4.9; an L pilot at 324 to 370 or 2650 to 2690 Hz reported −3.9 to −9.5 dB with κ at its clamp of 2, now −12.1 to −12.6; an L pilot at 326 Hz beside CE at 400 reported +8.0 dB, suspect, latent SNR −0.24 dB, now −9.0 dB and 5.88 dB. An L pilot at 2695 Hz beside CE at 2600 has the neighbour over its only flat side: it reported +23.8 dB, suspect, latent SNR 2.98 dB, and now −6.6 dB, κ 1.65 and 6.87 dB. That is halfway between the flat bins, neighbour and all (−17.8 dB, κ at its lower clamp of 0.5, 6.92 dB), and the compensated skirt, which the simulator's `fe_to_audio` filter takes down as a narrow radio filter would (+4.2 dB, suspect, 5.46 dB: as a headerless TINY pass the live listener would drop it as too strong). Beside a stronger neighbour (the review's cases, CE at 2600 Hz): CE SHORT at 2695 Hz and −8 dB beside one 18 dB stronger was lost from the flat bins alone and is received with its header, Z_ref 53, −12.6 dB and latent SNR 7.80 dB (old rule: Z_ref 31, +27.1 dB, 6.13); CE TINY at 2695 Hz and −15 dB beside one 15 dB stronger, also lost from the flat bins, reports −15.2 dB and 1.82 dB (old: +21.0 dB, which the live listener drops as too strong for a headerless pass, and 0.97). Mid-band signals (CE at 1100, L at 1500) are unchanged.
- Moderate fading costs L 1.2 to 1.8 dB of data SINR against the genie in round A (l-short at −12 dB, two seeds), from the pilot track's noise.
- Under `--just-ce`, which is the CE receiver from before L (with D43 and D44, without D39's re-check), an L signal can show as a weak headerless CE pass or a junk CE tile: Z_ref 6.41 (R 4.26) on l-short at −12 dB, Z 9.2 to 12.6 on l-tiny at −6 dB, a live tile at Z 7.3 and −8.0 dB latent SNR (TINY) or Z 7.2 (SHORT), and a SUSPECT pass at Z_ref 6.5 from `qrss_receive.py` on an l-short recording. With both waveforms enabled the CE re-check refuses these (D37, D39).

---

## 7. Interface contracts

- **Signs.** Bit 0 maps to +1 everywhere, and a positive LLR means bit 0.
- **Scaling.**
  - y is in latent units: E[y | x] = x for every symbol class.
  - z is unbiased, w = 1/var, and W is the per-latent SNR in linear units.
- **Orders.**
  - z and w are in air order: data-latent index k, the k-th entry of `_TX_PERMS[segment]`.
  - Accumulators are canonical, shape [3, 52800].
  - Symbol arrays are indexed by stream index. Tracker arrays are indexed by position.
- **Time.** Times are relative to t0 in seconds, or in symbols τ = (t − t0)/T.
- **Rates.** FE 4000 Hz complex at 1500 Hz audio, and CH 250 Hz complex.
- **Slot identity.** q is the integer quarter-hour count. A pass uid is as in §6.9.

---

## 8. Multi-pass (`store.py`, `associate.py` in WP7; `em.py` in WP9)

### 8.1 Store

`$QRSSTVAE_HOME` (default `~/.local/share/qrsstvae`) holds:

```
passes/<uid>.npz                 PassResult (z, w, hdr_llr, ch, psi, metadata JSON)
acc/<CALL>_<pid:08x>.json        {callsign, picture_id, mode, codec_id, created, updated,
                                  members: [{uid, segment, method, z_ref, mean_w_db, em_round}], foreign: [...]}
acc/<CALL>_<pid:08x>.npz         S, W float32[3,52800] canonical; S_ext, W_ext (merged from other receivers)
prov/<cluster>.json              provisional clusters of unassociated pass uids
```

- Writes are atomic (temporary file, then `os.replace`).
- `/` in a callsign becomes `_` in file names.
- The invariant is **S = S_ext + Σ_members w·z** and **W = W_ext + Σ_members w**, canonical.
- `rebuild` recomputes S and W from the members, and every attach, detach or EM update calls it. It takes microseconds at this size.

```python
class Store:
    def __init__(self, root) ; def add_pass(self, p: PassResult) -> None ; def load_pass(self, uid) -> PassResult
    def accumulator(self, key) -> Accumulator            # key = (callsign, picture_id)
    def attach(self, key, uid, segment, method) ; def detach(self, key, uid)
    def open_keys(self) -> list ; def expire(self, now, days=7.0) ; def merge(self, key, S, W) -> bool
@dataclass class Accumulator: key; mode: int; codec_id: int; S: float32[3,52800]; W: float32[3,52800]
                              S_ext; W_ext; members: list[dict]
    def rebuild(self, store) ; def loo(self, uid, store) -> tuple[float64[n], float64[n]]  # air order of that segment
```

`merge` adds a foreign (S, W) only after its correlation with the local members passes the §8.2 test (spec §8: "verify merged data").

**Waveform L.** A stored pass's metadata carries its "waveform" (0 when absent, a pass stored before L) and its "frame", and `pass_info` and `due` report both. The frame name decides a stored pass's spec (`lframe.get_any`: "l-..." is L), ahead of the waveform field (D41). `twin(acc, q, f_hz, segment, uid=None, waveform=None)` finds the member that is this transmission received again; with `waveform` given, a member of the other waveform is compared by band centre (§6.10), so one transmission is one member whichever waveform read it.

### 8.2 Association (`associate.py`)

The first rule that succeeds wins.

1. **Header decoded on this pass.** Look up (callsign, picture ID). If the accumulator's segment has median W > −10 dB, validate by correlation: reject the header match if t < 3 while the expected t exceeds 8. A rejected pass goes to provisional and a collision is logged.
2. **Soft-combined header.** Sum this pass's LLRs with each provisional cluster's summed LLRs, then pairwise with every other undecoded pass from the last 7 days within ±10 Hz, and try `header.decode` on each sum. Success attaches every pass involved, after the correlation check of rule 1. Only passes of a frame with a header take part: a headerless frame (TINY, l-tiny) keeps no LLRs, and decoding their empty sum used to make storing a slot's second TINY pass fail, in either waveform (fixed 2026-10-12).
3. **Latent correlation** against every open accumulator, over latents where w > 0 and W > 0, with m = S/W:
   - v = w·W/(1 + w + W);
   - t = Σ v·z·m / √(Σ v²·z²·m²);
   - **accept at t > 6.** This is the spec's 6/√M in self-normalised form, and N(0, 1) under H0 when the latents are independent. The highest t wins.
4. **Provisional.** Compute pairwise t against provisional passes and clusters, and join at t > 6, else start a new cluster. When a cluster's header later decodes, it is renamed to its key.

**Risk.** Unrelated pictures' latents may correlate through shared statistics.
- P7 measures the H0 distribution of t on 40 COCO pictures.
- If max |t| > 5, subtract the per-channel latent mean before correlating. That mean is measured from COCO and committed as `sstvae/qrss/latent_means.npy`, with the pyproject package-data line.

```python
def corr_stat(z, w, m, W) -> tuple[float, int]          # (t, M)
def associate(p: PassResult, store: Store) -> tuple[tuple | None, str]   # rule in {"header","soft-header","corr","prov",""}
```

### 8.3 Leave-one-out EM (`em.py`, WP9)

```python
def em_prior(acc, uid, store, q, spec) -> EmPrior
    # S_, W_ = acc.loo(uid); a = (S_/W_)*W_/(1+W_) (0 where W_==0)       LMMSE latent
    # x_hat = precode(a, q); v = block_mean(1/(1+W_))                     residual variance per precoded symbol
def em_refine(store, key, rounds=3, tol_db=0.05) -> list[float]           # mean W (dB) per round
def template_search(store, key, cap: Capture, spec) -> Detection | None
```

Each round, for each member pass i in turn:
1. Compute the prior.
2. Call `receive_pass(cap_from(p_i.ch), spec, det_i, prior=...)`. The data enter the measurement model as EM soft symbols (§6.5), including the timing refit. This re-modulates exactly as pass i was sent, with the same scrambler, WHT and β, and the e^{−v/2} factor.
3. Replace pass i's (z, w) and rebuild.

Stop after `rounds`, or when mean W gains less than `tol_db`. The leave-one-out reference never contains pass i, so a noise-only pass cannot confirm itself (P5). CE's reference fraction rises from about 55% to 55% + 32%·s/(1 + s) (spec §8), and this falls out of the template with no special casing.

**Template search** (spec §7, step 3).
- Correlate a stored capture with the expected waveform mean_template(μ = x̂, ν = v) of that slot's q, after A3/A1 frequency candidates, using the same Z statistic and gate as V.
- Accept at Z > 6 after a Bonferroni correction over the search grid.
- It works only on captures that are kept: `Capture`s in tests, or the per-candidate CH captures of earlier passes.

**Waveform L.** EM, the template search and retroactive detection are CE-only. An accumulator may hold members of both waveforms, and they decode alike; an L member is never refined or searched for, and nothing crashes on it:
- `pass_spec` resolves a pass's frame through `lframe.get_any`, and `is_l(p)` says whether it is L.
- `em_refine` skips L members and logs each once, in the first round: "pass <uid>: waveform L, not re-received (EM is CE-only)". `rereceive` returns None for an L pass.
- `known_keying(store, acc, call, waveform=WAVEFORM_CE)` counts only that waveform's members (OOK is False for L), and the template search's frequency hypotheses come from CE members only.
- `pass_path`, `template_search`, `receive_template` and `retro_detect` raise `ValueError` ("... is CE-only (design 8.3)") for an L frame.
- `retro_detect`'s "already stored" check compares by band centre (§6.10): a find within `RETRO_SAME_HZ` of a stored pass of that slot, of either waveform, is that pass and is skipped.
- `qrss_decode.py --retro --frame l-...` is refused; `qrss_decode.py` lists each member's waveform and frame.

`qrss_decode.py OUT.png (--key CALL:PID | --pass FILE.npz) [--store DIR] [--retro [--passband DIR] [--frame F]] [--em ROUNDS] [--model DIR] [--precision] [--any-codec]` writes the image, after retroactive detection and EM if asked. The picture's codec ID (accumulator or header) must equal the decoder's (`render.check_codec_id`), or nothing is decoded unless `--any-codec`; `qrss_receive.py --image` checks the same.

---

## 9. C reference (`qrss_beacon_c/`, WP8)

C99, no allocation, integer arithmetic only (a float self-test is optional). About 500 lines.

```c
typedef struct { const uint8_t *bf; uint32_t n_lat; uint64_t q; uint8_t flags; uint8_t keying[24];
                 int16_t blk[64]; uint32_t blk_idx; /* block cache, window state ... */ } qrss_ce_t;
int      qrss_ce_init(qrss_ce_t *s, const uint8_t *file, uint32_t len, uint64_t q); /* 0 ok; <0 bad magic/CRC */
int32_t  qrss_ce_symbol(qrss_ce_t *s, uint32_t k);       /* stream symbol k, in units of 1/(20*sqrt(M)) or +-1 class-scaled */
int32_t  qrss_ce_spread(qrss_ce_t *s, uint32_t k);       /* its spread-header part h (2.6.1): +-1 on a data symbol, else 0 */
uint32_t qrss_ce_phase(qrss_ce_t *s, uint32_t u, uint32_t fu_num, uint32_t fu_den);
         /* total phase (phi + 2 pi theta_cw) in 2^-32 turn at t = t0 - 8T + u*fu_den/fu_num */
int32_t  qrss_si5351_next(qrss_ce_t *s, uint32_t u, uint32_t fu_num, uint32_t fu_den, uint32_t step_mhz,
                          int64_t *reached);             /* error-feedback step for the next interval */
```

- `sha256.[ch]` is a minimal public-domain-style implementation, checked against the FIPS 180-4 vectors. The preamble (3 hashes), references (14), scrambler (198 per slot) and spread-header whitener (198, the same every slot) are all hashed on the device.
- `qrss_tables.h` is generated by `tools/gen_qrss_tables.py`, which has a `--check` mode. It holds:
  - the pulse half-table, 513 entries at T/64 in **Q14** (Q15 would overflow, since p(0) = 1.041), used with linear interpolation;
  - the Hann-integral R(y), 65 entries in Q30;
  - per-block-size scale constants β/(2π·20·√M) in Q-format, and β·√ρ/(2π) for the spread header;
  - each preset's `spread` flag (1 for every preset with a header block).
- Measured for this design: T/64 linear interpolation with Q14 or Q15 gives 8e-5 rad rms and 5e-4 rad max phase error, against −13.2 dB of the signal's own distortion. That is negligible.
- `test_main.c` reads a beacon file and prints phases and steps.
- `tests/test_qrss_c_ref.py` builds with `cc` if present and skips otherwise.
- AVR notes:
  - The half-table fits in 1 KB of flash.
  - A 64-sample WHT runs every ~2 s.
  - Phase per update is 16 multiply-adds plus a window term.

---

## 10. Verification plan

**Conventions**
- Fast tests are the default selection. The QRSS fast suite **must stay under 90 s** on this 4-CPU box. Each fast test is ≤ 5 s, except receiver tests at ≤ 15 s.
- Slow tests are `@pytest.mark.slow` and codec tests `@pytest.mark.codec`. Every random test uses fixed seeds, and statistical tests state their tolerance.
- **Losses** are measured against the genie receiver (true u, true timing, same noise realisation) in delivered per-latent SNR, i.e. dB of 1/MSE against the true latents.
- **Reduced-length test mode.** Frames come from `frame.PRESETS`: FULL, MEDIUM, SHORT and TINY. Per-latent SNR does not depend on frame length, so thresholds measured on MEDIUM or SHORT apply to FULL. SHORT keeps the header and two callsign windows, MEDIUM two windows, and TINY has no header and no windows.
- `Q_TEST = 1954752`, the count for 2025-10-01T00:00Z.

### 10.1 Format and bit-exactness (WP1, WP2): all fast

| ID | Check |
|---|---|
| F1 | FULL: 57,274 stream symbols, 58,810 positions; 165 / 2,474 / 1 / 3,374 / 50,600; windows at P = 13,888 / 28,734 / 43,580 / 58,426; duration 1782.678125 s = 14,261,425 samples at 8 kHz. MEDIUM, SHORT and TINY counts as in §2.2. Invalid `cw_after` raises |
| F2 | `block_sizes(50600)` = 790×64 + 32 + 8; the fast WHT equals `scipy.linalg.hadamard(M)/√M` for M = 8, 32, 64; self-inverse to 1e-12; `unprecode(precode(a))` = a |
| F3 | Pinned SHA-256 of the int8 bytes of the preamble, the 3,539 references, `scrambler(Q_TEST, 50600)` and `keying_units("K1ABC/P")`. The digests are pinned after the first implementation and cross-checked by C1. Each ±1 sequence is balanced within ±4σ |
| F4 | `quarter_hour_count`: 2025-10-01T00:00Z gives 1954752, and a non-quarter-hour time raises |
| F5 | Picture ID: a hand-computed digest on a synthetic fp16 tensor; unit RMS over the sent latents only; the never-sent values are 0. `codec_id_from_onnx` gives 0xD1D8 (codec). The v5 encoder on `wonder_wheel.jpg` gives the same ID twice from the stored `.qrsp` (codec) |
| F6 | Frozen AST rule over `constants`, `sequences`, `precoder`, `frame`, `morse`, `picture`, `header`, `polar`, `ce`, `tx`, `beaconfile` and `si5351`. A module added to the list fails until it complies |
| F7 | Morse: 173 units for "00000000"; "K1ABC" pattern equals the `cwid.py` table; a 9-character call or a lowercase/space-internal/'?' character raises |
| H1 | CRC check value 0xA69D; equal to `beacon._crc16`; `pack`/`unpack` round trip including grid none; any single bit flip gives None |
| H2 | `INFO_SET` hash; `tools/gen_qrss_polar.py --check` passes; noiseless encode and decode; the rate-matching fold is correct |
| H3 | BPSK AWGN, CA-SCL L = 8, at Es/N0 = −8.5 dB per coded bit (Eb/N0 = 3.9 dB): 30/30 decode (fast). At −9.4 dB (Eb/N0 = 3.0 dB): BLER ≤ 1e-2 over 500 trials (slow) |
| H4 | 300 noise-only LLR vectors: 0 accepted (fast, 30; slow, 300) |

### 10.2 Modulator, beacon file and Si5351 model (WP3)

| ID | Check | Tolerance |
|---|---|---|
| M1 | `phase_grid` at 16000, 8000, 4000 and 250 Hz equals `phase_at` | < 1e-12 rad |
| M2 | Power split on 50,600 Gaussian latents: carrier \|E e^{jφ}\|², linear part by regression | 0.527 ± 0.005; 0.337 ± 0.005 |
| M3 | Noiseless genie loopback: gain y·x/x·x; per-pass distortion on Gaussian latents (pin it); real v5 latents of `wonder_wheel.jpg` group 0 (codec) | 0.581 ± 0.003; 13.2 ± 0.4 dB |
| M4 (slow, codec) | 32 passes, independent q, genie, averaged, 5 COCO pictures: fixed part; `precode=False` variant | 23.1 ± 1 dB; 13.4 ± 0.6 dB |
| M5 | Spectrum (Welch on a 200 s FE capture): 99% and 99.9% widths; sideband density at ±19/25/30/40/50/60/80 Hz; power into a CE neighbour 50 Hz away | 51.5 ± 1.5 Hz, 72.5 ± 3 Hz; −8/−11/−15/−24/−34/−44/−65 dB ± 2 (± 4 at −65); −28 ± 1.5 dB |
| M6 | Si5351: error after the matched filter at 990, 250 and 125 Hz updates with a 0.4 Hz step; frequency rms and maximum | 43 / 41 / 33 dB ± 2; 7.7 ± 0.5 Hz, max ≤ 40 Hz |
| M7 | int8 storage error relative to the latents | 37 ± 0.5 dB |
| M8 | Lead-in: phase continuous; φ ≡ 0 before t0 − 8T; \|s\| = 1 throughout a FSK frame | exact |
| M9 | Callsign window: θ_cw at the window end is an integer to 1e-12 turn; phase continuous across the window; data tails at the window edges below 1e-3 rad inside units 4–187; FSK 99% power within −20.7 to +4.6 Hz of the carrier | ± 1 Hz |
| M10 | Beacon file: round trip; 50,970 bytes for a full segment; CRC-32 detects a flipped byte; a keying mismatch raises; `slot_symbols` from `.bin` equals that from `.qrsp` within the int8 error | exact / M7 |

### 10.3 Channel (WP4)

| ID | Check |
|---|---|
| C-1 | `fe_to_audio`, then `hfchannel.awgn`, then `audio_to_fe` gives the same SNR, measured on a pure carrier, as `simulate(snr)`, within ±0.1 dB |
| C-2 | Simulated tap Doppler matches a Gaussian with 2σ = spread within 10% in rms bandwidth; tap power 1 ± 5% |
| C-3 | `truth.f_hz` matches the configured offset, drift and OU trajectory: OU rms and correlation time within 10%; integrated-phase residual under 1e-9 turns. The time scale gives ppm·duration of slip within 1 µs. Slow: one 600 s run against `hfchannel.sample_clock_offset` on padded audio |
| C-4 | AGC step response matches its attack and decay; impulse and crash rates and levels are as configured; a full slot simulates in under 8 s (slow) |

### 10.4 Front end and acquisition (WP5)

| ID | Check |
|---|---|
| A-1 | `audio_to_fe` then `fe_to_audio` round trip, with in-band error < −60 dB; power preserved within 0.05 dB; `channelise` exact for f_mix on its 1/64 Hz grid |
| A-2 | The blanker never touches a CE signal at up to +20 dB with no impulses; it catches ≥ 99% of the sample energy of 30 dB clicks |
| A-3 | `normalise` removes a configured fast-AGC gain to within 1 dB rms at SNR₂₅₀₀ ≤ −10 dB |
| A-4 | A1 finds the carrier at offsets −400, −150, 0, +150 and +400 Hz about 1500 Hz, and at band edges 340 and 2660 Hz, at −25 dB, f error < 0.25 Hz |
| A-5 | A2 on SHORT at −25 dB, offsets as A-4, timing ∈ {−1.9, 0, +1.7} s: detection 100%; τ0 error < T/20; δf < 0.05 Hz |
| A-6 | A2 and A1 tails on noise: empirical Λ (1e6 cells) and A1 bins match Gamma within a factor 2 at 1e-3 and 1e-4. A lead-in-only carrier at +10 dB gives no A2 peak above threshold |
| A-7 | `TBD_GUMBEL_*` measured on 200 noise-only CH captures (slow) and committed; a fast test checks a 20-capture subsample against them |
| A-8 | `PassbandStore`: write a few minutes of FE samples across an hour boundary, read back an arbitrary span (int16 quantisation within 1 LSB), expiry deletes files older than 48 h, a restart re-opens existing files, and gaps read as zeros with a mask |

### 10.5 Single-pass receiver (WP6)

These run on SHORT frames at SNR₂₅₀₀ = −12 dB on a quiet path unless stated. † marks adversarial tests.

| ID | Scenario | Pass criterion |
|---|---|---|
| R1 | Steady AWGN, genie: SNR₂₅₀₀ = −25, −20, −15 dB | per-latent SNR = SNR₂₅₀₀ + 17.09 dB combined in parallel with M3's distortion, ± 0.2 dB (at −20: −2.97 dB) |
| R2 | Steady AWGN, full receiver, no AGC | loss ≤ 0.3 dB at −15, ≤ 1 dB at −25 |
| R3 | **Weight honesty.** Blocks binned by predicted v over all scenarios of this table | empirical MSE within ±0.5 dB per bin with ≥ 200 blocks; NEES of û in [0.8, 1.25] |
| R4 | Offsets as A-4 through the full chain | loss ≤ 0.2 dB |
| R5 | Drift: 1 Hz/min linear (MEDIUM) and warm-up (τ = 300 s, 1 Hz/min initial) | loss ≤ 0.2 dB; reported drift ± 0.1 Hz/min |
| R6 | Wander 0.1 and 0.5 Hz rms with τ = 3, 10, 60 s (MEDIUM) | loss ≤ 0.1 and ≤ 0.5 dB (spec §2.6); reported wander ± 30% |
| R7 | Clocks: tx/rx ppm ∈ {+100/−100, −100/+100, +100/+100} (MEDIUM; FULL slow) | timing error ≤ 0.02 T rms; loss ≤ 0.2 dB |
| R8 | Fading quiet, moderate, disturbed, 10 seeds each | effective SNR within ±1 dB of s(1 − ε)/(1 + sε) (`qrss_helpers` ports `tracking.mmse_gauss` and `s_eff`) |
| R9 † | Clicks 10/s at +30 dB; crashes 2/min at +50 dB | loss ≤ 1 dB and ≤ 1.5 dB; blanked blocks' w falls; no NaN; R3 holds |
| R10 † | Fast AGC (2 ms / 200 ms) plus R9 crashes | loss ≤ 0.7 dB against R9 alone |
| R11 † | CE neighbour at ±50 Hz, +10 dB, independent drift | the right signal tracked (header or truth correlation); loss ≤ 1 dB; neighbour decoded separately by `receive_slot` |
| R12 † | Lead-in: none, 3 s, 10 s; 10 s with 2 Hz/min drift during the lead-in only; lead-in at +20 dB | no τ0 error > 0.1 T; loss difference with and without ≤ 0.1 dB; with it at −25 dB, ≤ 0 dB |
| R13 † | **False alarms** through `receive_slot`: TINY noise-only (fast 30, slow 300); lead-in-style carrier only (+10 dB, 50); neighbour only (50); noise plus crashes plus AGC (50) | 0 PassResults in each category; Z_ref on noise has KS p > 0.01 against N(0,1) |
| R14 | Detection probability on a quiet path (slow): A2 at −28 and −31 dB; A3+V at −34 and −38 dB | ≥ 90% at −28; report −31 (spec ±3); ≥ 90% at −34; report −38 |
| R15 | Plain vs joint: disturbed, and 2 Hz (beyond spec), 10 seeds | joint ≥ plain − 0.05 dB in every case; the gain is printed (the spec §14 deliverable) |
| R16 † | Timing at the ±2 s edges; a preamble deep-faded over 0–20 s | found by A3+V; header decodes at −12 dB |
| R17 | Header in the full chain, quiet, single pass; four passes' LLRs summed | P(decode) ≥ 50% at −23.5 ± 1.5 dB; four passes gain ≥ 5.5 dB |
| R18 | Callsign windows: `read` correct in ≥ 8/10 seeds at −10 dB from one window; `match` Z > 6 at −16 dB over one pass's four windows (MEDIUM, FULL slow); OOK frames read and matched the same way; window-free tracking loss ≤ 0.1 dB vs genie through the window | spec §2.7: −12 / −18 dB ± 3 |
| R19 | Round B: after a header decode, re-tracking with header and keying known does not lose (≥ −0.05 dB) | |

### 10.6 Multi-pass (WP7, WP9)

| ID | Check | Criterion |
|---|---|---|
| P1 | Store algebra: rebuild from members; LOO exact; merge = sum; expiry; atomic writes; canonical mapping round trip | exact (fast) |
| P2 | N passes, steady, genie | +10·log10 N ± 0.2 dB up to N = 32 (fast, synthetic `qrss_fakes`) |
| P3 (slow) | Good-picture thresholds on a quiet path: the SNR₂₅₀₀ where mean W reaches +2.2 dB, MEDIUM frames, independent q | N = 2/4/8/16: −16.1/−19.9/−23.3/−26.4 ± 1.5 dB; N = 16 moderate/disturbed −25.7/−25.1 ± 1.5 |
| P4 | EM: 8 passes at −23 dB quiet (MEDIUM, slow; SHORT ×4 fast smoke) | mean W rises ≥ 0.5 dB; converges in ≤ 5 rounds |
| P5 † | EM self-confirmation: 4 signal passes plus 4 noise-only passes forced into one accumulator | noise passes' W ≤ pre-EM + 0.1 dB, and their t to the LOO reference stays < 3 |
| P6 | Association: a −16 dB-per-latent pass against a +2.2 dB accumulator; two passes at s = 0.075 | accepted by M ≈ 2,500 ± 30%; by M ≈ 7,300 ± 30% |
| P7 † (codec) | H0 of t on 40 COCO pictures pairwise, real v5 | max \|t\| < 5, else the §8.2 fallback, then retest |
| P8 (codec) | Decoder planes: g(W), W̃ rule, never-sent weight 0, shapes; noiseless round trip PSNR ≥ 27 dB on `wonder_wheel.jpg` | |
| P9 (slow) | Template search finds a −40 dB pass against a +5 dB accumulator | Z > 6 |
| P10 | Association end to end: 4 passes at −28 dB whose headers cannot decode alone, found by soft header or correlation, land in one accumulator (slow) | |

### 10.7 C reference and end to end (WP8, WP9)

- **C1:** the C sequences equal Python's bit for bit (preamble, references, scrambler for 3 values of q), and the C SHA-256 passes FIPS 180-4.
- **C2:** the C phase is within 2e-4 rad rms and 1e-3 rad max of Python's phase over [−8T, 600 s] of a SHORT beacon file. It covers both callsign keyings and update rates of 990, 250 and 8000 Hz.
- **C3:** C Si5351 steps are identical to `si5351_steps` (which uses the same Q-format rounding mode as C) for 3 slots at 990 Hz.
- **E1 (slow, codec):** FULL slot through the CLIs:
  1. `qrss_encode` → `qrss_beacon` → `qrss_transmit`, with a 10 s lead-in, +250 Hz offset and FSK ID;
  2. `qrss_simulate`: quiet, −8 dB, 1 Hz/min, 0.1 Hz wander, ±100 ppm, clicks, fast AGC;
  3. `qrss_receive` → `qrss_decode`.

  The header decodes, the callsign window agrees, and PSNR is within 2 dB of the genie latents' decode.
- **E2 (slow):** the same signal through `hfchannel.apply_channel` on 8 kHz audio (`mpg`, ppm, awgn) also decodes.
- **E3 (fast, CLI):** SHORT frame, steps `encode` (codec, else a synthetic `.qrsp`) → `beacon` → `transmit` → `simulate --snr -10` → `receive`. The store holds one pass with the decoded header.

### 10.8 Single-pass threshold checks (WP9, slow, codec, MEDIUM frames, real v5 latents)

Each check asserts that measured per-latent SNR ≥ +2.2 − 0.5 dB at the spec's threshold, and ≤ +2.2 + 1.5 dB there, so the receiver is not secretly better than the model:
- quiet −10.9, moderate −10.7, disturbed −10.5 dB (5 seeds, median);
- steady −14.9 dB;
- very good: quiet −2.5 dB gives ≥ +5.2 dB;
- all consumer impairments together (±400 Hz, 1 Hz/min, 0.5 Hz rms wander at τ = 10 s, ±100 ppm each end, clicks, fast AGC) at −10.4 dB on quiet gives ≥ +1.7 dB;
- decoded PSNR on 5 COCO pictures at the −10.9 dB case is within 0.5 dB of a direct-latent-noise reference at the same measured SNR.

### 10.9 Waveform L (2026-10-12)

The L receiver's tests, one file per part of §6.10. Fast-tier seconds are the file's test time, and slow counts and times are as measured once each.

| File | Checks | Tier |
|---|---|---|
| `test_qrss_lunits.py` | The shared pieces in either waveform's units, each a check a CE constant would fail silently: durations from the frame (L_FULL's 1782.65 s, where CE's T would give 891), `rx_index` and its inverse at T_L, the L matched-filter table (unit energy, agreeing with `lwave.matched_filter`), `q_grid(L_T_SYM)` = 8× CE's q, window units 8 to 183 FSK only, `z_ref_stat` with CE's d = j·Im c equal to CE's own form, line search, erase margin and report in L's period | fast, 0.5 s |
| `test_qrss_lchannel.py` | The channel simulator on L: noiseless pass sample for sample, SNR₂₅₀₀ on a pilot-only frame, keying span and clock ramp in L symbols, u the pilot's gain and u_data the data band's, neighbours of the other waveform with their own spectra, npz round trip, `qrss_simulate.py --waveform L`. Slow: a FULL L slot with every impairment in under 8 s, the CLI from a .qrsp | fast, 3.7 s; slow, 2 in 5.7 s |
| `test_qrss_lacquire.py` | A2-L at −20 dB (δf, timing, phase) and its Gamma tail; the −1/T_L alias dropped by band ownership; neither waveform's A2 passing on the other's signal at 0 and +6 dB; `merge_hits`; the shape test on noise and at −15 dB, the pilot re-anchoring; a mixed slot end to end and the detection order. Slow: A2-L at −28 and −31 dB, the joint live rate at −24 dB, the tail over 1e6 cells, a lead-in pilot with no preamble after it, the shape test along A3's paths on noise, re-anchoring on L_FULL at −28 dB, hints at −29 dB | fast, 6.2 s; slow, 7 in 48.2 s |
| `test_qrss_ltrack.py` | q grid at T_L (NEES near 1); the classes in rounds A and B and with an EM prior; the pilot filter against its closed form, white at even positions; noiseless genie (m = u·c, class gains, δ ≈ 1, K_PL_DATA, FSK pilot in the template with the keying known); `track_l` at −12 dB against the genie; gate V's Z_ref as the complex matched statistic and N(0, 1) or narrower on noise; a 0.15 rad rig phase recovered in δ; lead-in claims; `report_l`. Slow: Z_ref on noise, ±100 ppm, rounds A and B against the genie (steady, quiet, moderate), a CE signal taken for L refused by R, refinement at −24 dB and from a slip on the shoulder at −26 dB, the pilot's path through an oscillator's warm-up | fast, 5.1 s; slow, 9 in 33.3 s |
| `test_qrss_ldemod.py` | On an L_MEDIUM genie through the real clipper: the symbol-estimate formula with P and ψ, latent SINR to genie theory within 0.3 dB with the header unknown and known, W honest, the header from either copy, κ near 1 and following a 1.5× noise, the references' pull on u, an L and a CE pass too weak alone decoding their header together, the pinned constants against this frame's own loopback | fast, 0.8 s |
| `test_qrss_lreceiver.py` | Selection and band centres, the L header (version 2 only, sum or either copy), a wrong-waveform detection refused, the joint detect and receive (merging by centre, aliases and ghosts per waveform, suspect rule across), a CE selection taking CE's path, the CE re-check of D39 (stubbed), store and association by centre, EM and retro leaving L alone, both CLIs in process, one blind l-tiny pass at −12 dB. Slow: l-short at −6 and −24 dB, noise, L beside CE either way round, the CE re-check on real gate-V tracks, `qrss_receive.py` on an L recording | fast, 2.3 s; slow, 8 in 41.6 s |
| `test_qrss_live_l.py` | What each command line listens for, the frames every receiver call is handed (CE alone exactly as before L), the slot length per selection, one tile per signal whichever waveform reads it, waveform in tiles, state.json and hints, a stored member read as the other waveform counted once, L's band edge. Slow: real slots holding a CE and an L signal, written to the ring and heard as audio across the slot's end | fast, 2.7 s; slow, 3 in 21.9 s |

- The seven files' fast tier is 145 tests in 22.8 s wall (21.4 s of test time). With them the default QRSS tier measured 591 passed in 157.4 s once, before six tests moved to slow (against 440 in 123.9 s before L), so §10's 90 s budget is not met; the tier was already over it before L.
- **CE is unchanged**: with a CE-only selection the receiver's 34 output arrays on a fixed set of CE cases are bit-identical to the tree before L, checked with a scratchpad script that is not in the repo, and live runs at −12 dB with no selection and with CE alone traced every receiver call identical to the old `live.py` (9 of 9 calls on TINY, 19 of 19 on SHORT).

### 10.10 Receiver rules of 2026-10-12 (D43, D44)

| File | Checks | Tier |
|---|---|---|
| `test_qrss_rules.py` | The switches (environment, an unknown rule refused, a spawned process following its parent, the command lines' flags, `qrss_decode.py` taking `--old-edge-noise` only), with the environment restored after each test; on white audio through the real front end, the floor at 305, 326 and 2695 Hz within 0.75 dB of the truth (old rule: 6 dB or more low), the same with a tone excluded on the only flat side (400 Hz beside 326, 2600 beside 2695), and with 50 Hz of noise 20 dB up there, halfway in dB between the flat bins' reading and the truth; mid-band floors equal with and without `audio_hz`, under either rule; `_fe_flat` all-flat from 430 to 2570 Hz with the decimation's folds counted; `ChanCapture` and `lead_in_span` handing their audio frequency over; an A3 hit's Z on its detection; `_owned`'s bands and boundaries for CE and L, a stronger line, a hint, a preamble hit and a line beside a signal with no A3 line of its own all gated, a whole-slot or hint detection owning nothing; and which candidates `detect` gates, CE alone and joint, with and without `max_candidates`, under either rule (stub gates) | fast, 19 tests |

`test_qrss_lreceiver.py`'s merge-by-centre test runs with D44 set back, since under it the CE line inside the L's band is no longer gated. The review's three D44 cases (§6.8) and four edge-floor cases (§6.10's known limits) were run end to end, not in the suite (10 to 26 s each). CE in mid-band: the 34 arrays of §10.9's CE check, regenerated on the tree before D43 and D44, are bit-identical with the new rules and with both set back.

---

## 11. CLIs (summary)

| Script | WP | Purpose |
|---|---|---|
| `qrss_encode.py` | 3 | image → `.qrsp` (picture ID printed) |
| `qrss_beacon.py` | 3 | `.qrsp` → beacon `.bin` |
| `qrss_transmit.py` | 3 | `.qrsp` or `.bin` → slot WAV |
| `qrss_simulate.py` | 4 | WAV, `.qrsp` or `.bin` → impaired WAV or `.npz` |
| `qrss_receive.py` | 6 | WAV or `.npz` → passes in the store (+ optional single-pass image) |
| `qrss_decode.py` | 9 | store → image, with optional EM |

---

## 12. Work packages

At most two packages run at once on this machine. Each wave starts when its dependencies are met. A package works against the signatures and dataclasses in this document and ships fakes for its consumers.

| Wave | Packages in parallel |
|---|---|
| 1 | WP1 ∥ WP2 |
| 2 | WP3 ∥ WP4 |
| 3 | WP5 ∥ WP7 |
| 4 | WP6 ∥ WP8 |
| 5 | WP9 |

| WP | Name | Owns | Depends on | Acceptance tests |
|---|---|---|---|---|
| **1** | Core format | `__init__`, `constants`, `sequences`, `precoder`, `frame`, `morse`, `picture`, `types`; `tests/qrss_helpers.py`, `test_qrss_{frozen,format}.py` | none. Writes `constants.py` first, because WP2 imports it | F1–F7; F5 codec part |
| **2** | Header and FEC | `polar`, `header`, `tools/gen_qrss_polar.py`, `test_qrss_header.py` | `constants` (from WP1's first commit); `sstvae.modem.beacon` | H1–H4 |
| **3** | CE transmitter | `ce`, `tx`, `beaconfile`, `si5351`, `qrss_{encode,beacon,transmit}.py`, `test_qrss_{ce,beacon}.py` | WP1; WP2 for `header.encode` (stub with zeros until present) | M1–M10; `ce.loopback_stats` delivered |
| **4** | Channel | `channel`, `qrss_simulate.py`, `test_qrss_channel.py` | WP1 (`FrameSpec`); `ce.baseband` (an internal tone stub until WP3 lands, in the same wave) | C-1 to C-4 |
| **5** | Front end and acquisition | `frontend` (incl. `PassbandStore`), `acquire`, `test_qrss_{frontend,acquire}.py` | WP1, WP3, WP4 | A-1 to A-7; commits `TBD_GUMBEL_*` |
| **7** | Multi-pass core | `store`, `associate`, `render`, `tests/qrss_fakes.py`, `test_qrss_multipass.py`; optional `latent_means.npy` and its pyproject line | WP1 (`types.PassResult`, `picture`), WP2 (`header.decode` for soft headers) | P1, P2, P6, P8, P7 (codec); P10 deferred to WP9 |
| **6** | Tracker, demodulator, receiver | `track`, `demod`, `cwid`, `receiver`, `qrss_receive.py`, `test_qrss_{track,receiver,cwid}.py` | WP1–WP5 | R1–R19; pins `D_PASS` and `KAPPA_SELF` |
| **8** | C reference | `qrss_beacon_c/*`, `tools/gen_qrss_tables.py`, `test_qrss_c_ref.py` | WP3 | C1–C3 |
| **9** | EM, template search, end to end | `em`, `qrss_decode.py`, `test_qrss_{em,cli,acceptance}.py` | WP6, WP7 | P3, P4, P5, P9, P10; E1–E3; §10.8 |

Every package must leave `.venv/bin/python -m pytest` green, including the existing 323 tests, and keep the QRSS fast tests within budget.

---

## 13. Seams for later steps (not built now)

**Waveform L, built** (§6.10, 2026-10-12). What the earlier seams became:
- Frames carry `waveform`; L's are `lframe.LFrameSpec`, each the twin of a CE frame of the same shape (`lframe.twin`).
- `sequences` holds L's preamble and references.
- Header LLR indices are shared (§2.6), so CE and L passes of one segment add.
- The callsign windows are 192 L symbols, one per Morse unit, FSK only.
- `ldemod` is L's demodulator: √2·Re and √2·Im over the class gain, the data's G_DATA_L = 0.881.
- The beacon file's waveform byte stays CE: `beaconfile` refuses any other, and L is sent from a sound card (`qrss_transmit.py --waveform L`, which has no OOK).

**Waveform L, still seams.**
- **An L-shaped A1 statistic.** A1 looks for a line, and an L pilot carries 10.5% of the preamble's power, so live L acquisition stops at about −26 dB (§6.10). A statistic matched to the pilot and its one-sided data band is estimated to gain 2 to 3 dB; that is not measured. An A3 statistic on an L-matched emission would need its own Gumbel; today A3's Z is the pilot-only one, about 6 dB behind CE's, and `l_pilot_bins` only places the path.
- **EM, the template search and retroactive detection for L** (§8.3): an L mean template given (x̂, v), L's Z for the search, and `retro_detect` over L frames.
- **The L ghost rule.** It is CE's rule with its timing window in L symbols (GHOST_T = 3 T_L within GHOST_HZ = 20 Hz). L's references recur every 16 T_L, so its ghosts would sit near multiples of 1/(16 T_L) = 1.03 Hz; none appeared in the runs of §6.10, and the rule is pinned by stub tests only. Unmeasured.
- **Recovery bookkeeping per waveform.** The listener's `finished` set and `recover`'s "passes already in the store" test are per slot: a slot with any stored pass is not received again, so a slot an earlier listener heard with one waveform enabled is not recovered for the other.
- **`live_end_s` per frame.** It is computed from CE's preamble. L's lasts as long in time, so it holds today; a frame whose twin's preamble differed would need its own.
- **The app's L tile.** state.json carries each tile's "waveform", and an L tile's f_hz is its pilot, with its signal reaching 22.07 Hz below. How the app draws that is the app's.

**Live view and GUI (step 6).** `kalman_rts(causal=True)` and `TrackReport` are what the live view and GUI use.


**AVR beacon.** The C reference is the parity oracle for any port.

---

## 14. Decisions not in the spec (for the spec owner)

| # | Decision | Reason |
|---|---|---|
| D1 | **Timing origin.** Stream symbol s is centred at QH + 1 s + pos(s)·T. The keyed carrier starts by t0 − 8T and ends with the last window at t0 + (n_pos − ½)T, so the frame is 1782.678 s centre to end. At 8 kHz a symbol is 242.5 samples, and the modulator is exact through a 1/485-symbol table on the 16 kHz grid | the spec does not place symbol 0 or say where the pulse tails go |
| D2 | **Pulse.** RRC with α = 0.15, cut at ±8T on both TX and RX, renormalised to unit energy on the 16 kHz grid. β = 4/5 rad | spec §2.5 cuts at ±8; the sims used ±16 |
| D3 | **Reference grid.** References sit where (s − 660) mod 16 = 0, in stream indices | the only placement giving 165 header and 3,374 data references |
| D4 | **Spare header symbol.** The 2,475th non-reference header symbol sends +1 and is used as a reference | 2,475 slots for 2,474 bits |
| D5 | **Sequences.** ASCII domain strings with no terminator; `uint64_be(q)` with q = floor(unix(QH)/900); `uint32_be` block counter; bits MSB first; bit 0 → +1. Reference m takes bit m of one stream. Labels: "QRSSTVAE CE preamble", "QRSSTVAE CE reference", "QRSSTVAE scramble" (L: "QRSSTVAE L preamble" and "QRSSTVAE L reference", both taking bit pairs as QPSK) | spec gives only the scrambler's ingredients |
| D6 | **Precoder tail.** The 32 and 8 blocks are orthonormal Sylvester WHTs of their own size. Shorter test frames use a binary decomposition of n mod 64 | not named in the spec |
| D7 | **Picture ID bytes.** `uint16_be(codec ID) ‖ uint8(mode) ‖` the mode's groups as fp16 LE, canonical 52,800 each, never-sent values zeroed, each scaled to unit RMS over its 50,600 sent values in float64 and then rounded. The fp16 values are what is sent | spec gives the idea, not the bytes |
| D8 | **CRC.** `beacon._crc16` bit for bit, check value 0xA69D. **This is not CRC-16/CCITT-FALSE (0x29B1).** The spec should say "SSTVAE beacon CRC" or switch to true CCITT-FALSE; either is one line. SSTVAE's docstring is also wrong | spec's two phrases contradict each other |
| D9 | **Header FEC.** Polar N = 2048, K = 142 including the CRC, natural-order F^{⊗11}, circular repetition to 2,474, `INFO_SET` committed as literal data (GA design at −23.5 dB), CA-SCL L = 8. Acceptance also requires version = 2 (1 only in the fallbacks of 2.6.1), reserved = 0, mode ≤ 2, segment ≤ mode and a valid callsign, which cuts false accepts from about 1.2e-4 (8 list paths x 2^-16) to below 1e-8 per attempt | spec allows polar or LDPC without specifying either; best of the constructions checked |
| D10 | **Header fields.** Table order, MSB first. Grid = ((F1·18 + F2)·10 + D1)·10 + D2, with 32767 for none. Version 1. CRC over bits 0–125. Callsigns limited to `[A-Z0-9/]`, 1 to 8 characters, rejected otherwise instead of silently mapped to space | unspecified, and the Morse ID needs a Morse code for every character |
| D11 | **Codec ID.** The first 2 bytes of `sstvae.source_sha256` (v5: 0xD1D8). The encoder refuses a model without it unless `--codec-id` is given | third-party exports may lack the metadata |
| D12 | **Beacon file.** Format of §2.9: 50,970 bytes for mode A; int8 at a scale of 20, clipped symmetrically to ±127 (±6.35, not −6.4); stores the callsign keying mask and the keying flag | spec gives only the contents |
| D13 | **Callsign-window details.** Unit u of window w starts at t0 + (P_w − ½ + 2u)·T. The call starts at unit 8. The Hann smoothing is centred on each unit boundary, giving the closed-form θ_cw of §2.7. The Morse alphabet is A–Z, 0–9 and '/', with no trailing gap | spec gives timing to 0.1 s and the keying rules, not sample-exact phase |
| D14 | **On-off-keyed ID.** The carrier stays on, unmodulated, for units 0–4 and 184–191 and keys on and off in units 5–183 (5–7 off). Adopted in the spec with this change, since on through unit 7 would merge with the call's first element | spec says only "key-down = on" |
| D15 | **Lead-in.** a = 1 and φ = 0 from the lead-in start, up to 10 s, to t0 − 8T. Receivers exclude it from A1 and A2 and use it in A3 and as virtual known positions, keeping only its last 8 s | spec §2.8 wording pending |
| D16 | **Receiver state factorisation.** The spec's joint Kalman state is split into a frequency path (Viterbi plus a 60 s spline, re-centred from û), a complex-gain constant-velocity Kalman filter and RTS smoother on u (which holds the phase), and an affine timing model. The measurement for every position uses the matched-filtered mean template E[s(t)], covering carrier, references, header, lead-in, windows and EM soft data in one form. Process noise is chosen by maximum innovation likelihood | linear-Gaussian, cannot cycle-slip, and all spec quantities are still estimated and reported |
| D17 | **Timing** is a fitted affine model, plus an optional quadratic, over the whole pass, not a Kalman state | sound-card clocks do not wander on a symbol scale |
| D18 | **Weights.** σ² = [N/2 + (A0² + K²)P/2]/(K²\|û\|²ψ²) + D_PASS, then one scalar calibration κ ∈ [0.5, 2] per pass from the reference residuals. N comes from out-of-band CH power in 10 s windows | 10 s holds only about 20 references, too few for a stable 1/var on their own; W must match 1/MSE |
| D19 | **Joint block estimator by default.** The closed-form LMMSE H·diag(1/(1 + σ²))·y, debiased, with crosstalk counted in its variance. It equals the plain estimator on a flat channel. Plain is kept behind a flag | spec §2.4 asks for the joint solve on fast fading |
| D20 | **Impulse blanker and inverse AGC.** k = 5 (14 dB) over a 1 s running median, with a 2 ms guard and 1 ms taper, on the 4 kHz FE. Level normalisation is over 100 ms. Symbols with blanked share ψ < 0.5 are erased; otherwise the gain is ψ and the variance 1/ψ² | spec gives no algorithm |
| D21 | **Front end.** 4 kHz complex at 1500 Hz (300–2700 Hz), and a 250 Hz channel per signal. Each pass keeps its 250 Hz capture (3.6 MB) for EM, in addition to the 48 h passband store, which the spec requires for retroactive detection | efficiency; all of the signal lies within ±75 Hz |
| D22 | **Acquisition statistics and the single verification gate.** Gamma CFAR in A1 and A2, a Gumbel-calibrated A3, and acceptance only at Z_ref > 6 with known symbols withheld from the tracker | a measured false-alarm rate in one place |
| D23 | **Association statistic.** t = Σvzm/√Σv²z²m² with v = wW/(1 + w + W), accepted at t > 6 (the spec's 6/√M in self-normalised form). Header matches are validated by correlation, and foreign merges are checked by correlation | unequal weights |
| D24 | **Channel presets.** quiet = (0.1 Hz, 0.5 ms), moderate = (0.5 Hz, 1.0 ms), disturbed = (1.0 Hz, 2.0 ms), mps = (0.15 Hz, 2.0 ms) | spec names the Doppler only |
| D25 | **Store layout.** One accumulator per (callsign, picture ID), with S and W [3, 52800] canonical plus S_ext and W_ext, per-pass files, atomic writes and 7-day expiry | spec §8 sets the contents, not the format |
| D26 | **Spec arithmetic corrections.** The frame is 1782.678 s (stream 1736.118 s, not "1736 s"). CE timing slip at ±100 ppm per end is up to 12 symbols, not "~6". CE's reference share is 54.8%, not 54.6% | arithmetic |
| D27 | **The receiver uses the closed-form K = 0.58092.** The sims measured 0.4% higher, which is within M3's tolerance | closed form over calibration |
| D28 | **One A2, a preamble kind per waveform.** A2 takes a `PreambleKind` (sequence, symbol, chunk, data offset, phase offset, matched filter). L's is its QPSK preamble at T_L in 33-symbol chunks, mixed up by f_p so the notch's pilot remnant falls outside the filter, with complex zero-mean templates; Λ stays Gamma(10, 1) on noise | spec §2 and §2.3 give L's preamble, not how to search for it; one search with two kinds leaves CE's numbers untouched |
| D29 | **A1 unchanged, plus pilot hypotheses.** For L, A1's best cell 3.1 to 22.1 Hz above each line is searched as a pilot, and an L hint whose A1 cell is under threshold gets A2-L at f and f ± 1 Hz | during the preamble L's data lumps take A1's line and hide the pilot; an L-shaped A1, estimated at 2 to 3 dB better, is not built (§13) |
| D30 | **A3's spectral-shape test.** z1 compares the one-sided band 3.10 to 22.07 Hz below the path with its mirror, and z_above measures the mirror alone: L at z1 > 4 with z_above ≤ 3, CE at z_above > 3 with \|z1\| < 3, otherwise ambiguous and gated as both. A pilot-anchored path fills in where the path's own shape is silent | the spec has no blind waveform test. z1 alone gated a CE signal whose preamble A2 missed as L only (z1 4.2 to 5.1 on 2 of 8 SHORT passes at −10 dB); power above the line is what L never has |
| D31 | **Arbitration.** Symbol-rate aliases per waveform at its own T; L band ownership (a hit 3.1 to 22.1 Hz below a stronger L hit, within 3 T_L, is that signal); across waveforms one hit per signal within 5 Hz and T_L/2, the smaller p kept | the −1/T_L alias passes A2-L from about −12 dB (12 of 12 at −6 dB, SHORT); a pilot can be found by a line and a hypothesis at once |
| D32 | **L tracker.** One complex gain, the data band's reference gain, from two independent streams: the data band mixed to 0 Hz in sender time, and the pilot through an RRC at 2T_L read at even positions. δ, the pilot's gain over the data's, is fitted by least squares with a 0.25 prior and never inside gate V. §6.5's Kalman filter and smoother run unchanged on `q_grid(L_T_SYM)` | linear-Gaussian, as D16. A receiver-time mix would drift 2.2 turns over L_FULL at 100 ppm; an RRC at T_L lets in 23.7% of the data's power against 0.9% |
| D33 | **L gate V.** Z_ref is the matched statistic over the complex known symbols, divided by √max(R, 1), R the known symbols' residual over the noise model's | an Im-only form reads 0.68 of the matched statistic; a CE signal taken for L reads Z 6 to 12 and R 2.8 to 30, against 0.77 to 1.54 for L |
| D34 | **L extraction and weights.** r = m·u*/(\|u\|²ψg) split into two latents with no ½; class gains, pilot gains and D_PASS_L = 0.058 pinned from 24 noiseless loopbacks; κ's reference branch allows for each measured reference's pull on u | spec gives the 0.881 data gain and the √2 split, not the variance. CE's κ form read 0.2 to 0.7 on moderately fading L passes and overstated W by up to 1.7 dB |
| D35 | **L header.** Coded bits 2j and 2j+1 on I and Q of header symbol j; version 2 only, tried as the sum, the spread copy alone, then the block alone, never `frame.plain`. The spread copy is subtracted in the demodulator once the header is known, and sits in the data template's mean in round B | L has no version 1; a linear signal cannot be de-spread on the channel as CE's constant envelope is |
| D36 | **L callsign windows.** One L symbol per Morse unit, FSK only: `keying_type` says "fsk" without computing, OOK raises, and the tone amplitude comes from units 8 to 183 | spec §2.7 keys L as CE; units 0 to 7 and 184 to 191 of an L window hold the pilot's ramp and the data pulses' tails |
| D37 | **A CE-only selection is the old receiver bit for bit,** with D43 and D44 set back (`--old-edge-noise --old-gating`; without them it is the 2026-10-12 rules' CE receiver). `--just-ce`, `LiveConfig.waveforms` None or (CE,) make exactly the calls they made before L, without D39's re-check, so an L signal there can come out as a weak headerless junk CE tile or pass (Z_ref 6.41, R 4.26 on l-short at −12 dB) | keeps the CE receiver on the air since 2026-10-09 exactly as tested (34 output arrays bit-identical); running the re-check there would break that, and has not been decided |
| D38 | **One signal, one identity.** Signals are compared at their band centre: f for CE, f − 9.5 Hz for L, f being L's pilot. Within 5 Hz the larger Z_ref stands for the signal, and its waveform with it | both waveforms can lock on one signal, and the offset puts the symmetric 45 Hz noise-floor guard over L's one-sided spectrum, [f − 54.5, f + 35.5] Hz |
| D39 | **CE accepts are re-checked in the joint path.** One is refused when R > 2 and Z_ref/√R ≤ 6 | CE's gate V passes an L signal at Z_ref 6 to 11. CE reads R 0.93 to 1.48 (TINY and SHORT at −12 and −6 dB; fading up to 2.16, at Z_ref 48, kept), L under CE's gate 3.6 to 15.4; it costs about 0.09 s per CE accept on TINY |
| D40 | **Hints are frequencies or (frequency, waveform) pairs.** A bare frequency is searched in every enabled waveform, a pair in its own; the CE-only receiver keeps CE pairs as frequencies and drops the rest | the store knows each due pass's waveform, so its hint need not cost an A2 search in both; a bare frequency keeps the old argument working |
| D41 | **A stored pass's frame name decides its spec.** "l-..." is L, ahead of the "waveform" field, and a pass stored before L is CE | the frame name fixes the layout as well as the waveform, so the two cannot disagree; the field is only a fallback |
| D42 | **Both waveforms by default** (Jeff, 2026-10-12), with `--just-ce` and `--just-l` on `qrss_receive.py` and `qrss_listen.py`. `--frame` names the shape in either waveform and the other listens for its twin; the app starts the listener with only `--state` and hears both | a receiver cannot know which waveform a station sends. The cost: a CE SHORT .npz took 7.9 to 8.1 s by default against 6.8 s with `--just-ce` (12.6 s against 7.7 before D44), and joint acquisition about twice a single one |
| D43 | **The noise floor only where the front end is flat** (§6.2, 2026-10-12; Jeff chose it with D44). `noise_psd` takes bins within 0.1 dB of the FE low-pass's passband, counting the decimation's folds; where neighbours leave fewer than 3 Hz of those clear, the geometric mean of the flat bins (high, if anything) and the skirt compensated for the low-pass (low, if anything). `--old-edge-noise` on `qrss_listen.py`, `qrss_receive.py` and `qrss_decode.py` restores the old rule | near 300 and 2700 Hz half the bins sat on the low-pass's skirt, so a signal's SNR read up to 9.6 dB high, κ clamped at 2 and a headerless pass there was dropped as too strong; in the middle of the band nothing changes, bit for bit |
| D44 | **A found signal's own band is not gated again** (§6.8, 2026-10-12). A whole-slot candidate within GHOST_HZ of a preamble-found CE carrier, or 22.1 Hz below to 3.1 Hz above a preamble-found L pilot, and no stronger in A3 than that signal's own line, is skipped and not counted against `max_candidates`; preamble hits and hints are always gated. `--old-gating` on `qrss_listen.py` and `qrss_receive.py` restores gating every candidate | those lines were gated only to be refused or dropped as ghosts, at 1.1 to 8.9 CPU s each on SHORT and about a minute each on FULL. Only a preamble-found signal owns a band, and only against lines no stronger than its own, since a whole-slot detection can be the weaker of two lines and a station whose preamble faded the stronger |

---

## Appendix A: how the two designs were merged

**Taken from design B (robust receiver first):**
- the 4 kHz front end and 250 Hz channels;
- the blanker with inverse AGC;
- the A1/A2/A3 detectors and the single Z_ref gate with measured tails;
- the factorised frequency-path plus complex-gain Kalman/RTS tracker, with re-centring;
- the affine timing fit with an optional quadratic;
- the closed-form joint block estimator, as the default;
- the 250 Hz capture stored with each pass for EM;
- the [3, 52800] per-picture accumulators;
- the self-normalised association statistic, plus the COCO H0 test;
- the `FrameSpec` presets;
- the integer C reference and generated tables;
- the adversarial tests (false alarms, weight honesty, EM self-confirmation, neighbours, the lead-in at +20 dB).

**Taken from design A (faithful and minimal):**
- the Bussgang mean-template measurement E[s(t)] = e^{jφ_μ}e^{−v/2}. It replaces B's linear c = A0 + jKx̂, so known stretches such as the preamble are exact and EM re-modulates exactly as sent;
- the κ residual calibration on references;
- the `S_ext`/`W_ext` invariant and `rebuild`;
- the canonical `Q_TEST` with its date;
- SHA-256 for all sequences in C, with no stored sequence tables;
- the explicit residual-check flag;
- a single shared `phase_at` used by TX, templates and EM.

**Errors found and fixed:**
1. **Both designs missed rev 9's callsign windows** (spec §2.7). Without them the frame is 46.6 s short and does not identify the station. They are added throughout: layout, phase, beacon file, C, receiver and tests.
2. **Design A's polar frozen set is wrong for natural-order encoding.** It applies the Bhattacharyya operations LSB-first. For x = u·F^{⊗n} in natural order, the channel-side split is chosen by the MSB, which a 4-bit hand check confirms: u1 has z = (2z − z²)², not 2z² − z⁴. GA density evolution gives SC BLER = 1.0 for A's set at every SNR checked (table in §2.6). Design B's PW set was correct but 0.2–0.3 dB worse than a GA design at the threshold.
3. **Design B's C table in Q15 overflows,** since p(0) = 1.041. It is Q14 here. T/64 linear interpolation was measured at 8e-5 rad rms.
4. **Design B's demodulator variance omitted the ½ of the imaginary projection** for both noise and estimate error. That made W 3 dB pessimistic before calibration.
5. **Design A's 1 kHz single-rate receiver re-read source audio for EM, and gated the preamble on an ad hoc threshold.** Both are replaced as above.

**Numbers checked for this document** (scripts kept in the session scratchpad, not in the repo):
- the frame and window counts, and 14,261,425 samples;
- the pulse constants c, p(0) and G0, and Var φ = 0.645 at β = 0.8;
- C-table interpolation error;
- the GA polar comparison, design-point robustness, and the `INFO_SET` literal and hash;
- CRC check value 0xA69D from the repo's own `_crc16`;
- B's joint-estimator closed form and its one-fade example;
- the SHORT/MEDIUM/TINY counts.
