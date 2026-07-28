# The cyclic prefix, and how it differs from the pilots

Explainer, not a design doc — this describes what the modem already
does. The CP and the pilot symbols are often conflated because both are
"overhead that isn't payload", but they solve orthogonal problems:

> **The CP makes the channel look like one complex multiply per carrier;
> the pilots measure what that multiply is.**

## Where it lives in the signal

`config.py`:

```
RS   = 50              # carrier spacing, Hz
M    = FS // RS = 160  # "useful" symbol, 20 ms
NCP  = 32              # cyclic prefix, 4 ms
NSYM = M + NCP = 192   # what actually goes on the air
```

These are RADE's numbers (arXiv:2505.06671 gives `Tcp = 0.004 s`,
`Ts' = 0.020 s`, `Rs = 50 Hz`; RADE uses `Nc = 30` carriers where we use
24 in a narrower slice of the passband).

Per symbol, on air:

```
|<-- 32 samples CP -->|<--------- 160 samples useful --------->|
|  == tail of useful  |          what the DFT integrates       |
```

Note that **no code copies the tail to the front.** `ofdm.py`:

```python
_n_sym = np.arange(NSYM) - NCP
MOD_MATRIX = np.exp(2j * np.pi * np.outer(_n_sym, CARRIER_FREQS) / FS)
```

instead evaluates all 192 samples of the continuous sinusoids, with the
phase reference at `n = 0` meaning "start of the useful part" (hence the
`- NCP`). That is identical to a copy-paste CP **only because every
carrier is an integer multiple of RS = 50 Hz**: a 50·k Hz tone has period
`FS/(50k) = 160/k` samples, so it repeats exactly every 160 samples, so
sample `n` and sample `n + 160` are equal. This is what the comments at
`config.py` ("carriers sit on integer multiples of RS so the cyclic
prefix is truly cyclic") and `ofdm.py` are protecting. Move `CARRIER0`
off a 50 Hz multiple and the waveform silently stops being cyclic —
nothing errors, the modem just degrades in multipath.

## Why it works

Two effects, both from the same periodicity.

**1. Inter-symbol interference.** A delayed echo of the previous symbol
spills into the current one. If the spill is confined to the CP region
and the receiver integrates only *after* the CP, the previous symbol's
energy never enters the integral.

So the CP length is a **delay-spread budget**: 4 ms. Against the presets
in `hfchannel.py`, `mpg` is 0.5 ms (comfortable), `mpp` (CCIR poor) is
2.0 ms (comfortable), and `mpd` (disturbed) is 4.0 ms — *exactly* NCP.
mpd sits right on the edge of the guard by design; it is the worst case
the waveform is dimensioned for, not a case with margin.

**2. Inter-carrier interference.** The subtler and arguably more
important one. Over the useful window, a delayed copy of the *current*
symbol is not a truncated sinusoid: by periodicity, a copy delayed by
`d < NCP` looks within the window exactly like the same sinusoid with a
phase shift. Direct + echo is therefore still a pure tone at the same
frequency, with a different complex amplitude.

In DSP terms the CP turns linear convolution with the channel into
**circular** convolution, which the DFT diagonalizes: carriers stay
mutually orthogonal, and the multipath channel collapses to one complex
scalar `H[k]` per carrier. Without a CP a 2 ms echo would splash each
carrier's energy into its neighbours, and no per-carrier equalizer could
fix it.

## Where the receiver uses it

`ofdm.demod_window`:

```python
def demod_window(z, start, backoff=0):
    s = start - backoff
    win = z[s : s + M]
    return (2.0 / M) * (DEMOD_MATRIX @ win)
```

The window is `M = 160`, not 192 — **the CP is deliberately discarded at
the receiver.** It carries no information; it is pure redundancy paid for
in airtime (32/192 = 16.7% of transmit time, which is why CP length is a
real design tension and not free insurance).

Callers pass `start = frame_start + s*NSYM + NCP` (see `modem.py`), i.e.
skip the CP and integrate the useful part. But `DEMOD_BACKOFF = 6` pulls
the window 6 samples *earlier*, into the CP — the CP's third job,
**timing slack**. Residual timing error doesn't corrupt anything as long
as the window stays inside the CP-protected region. The cost is a linear
phase ramp across carriers, which as the docstring notes "is absorbed by
pilot equalization as long as it is applied consistently": the pilots see
the same offset, so their measured `H[k]` includes it.

The preamble is a special case (`PREAMBLE_CP = 2 * NCP`, two useful
periods), making the whole 384-sample block periodic with 160. That is
what the lag-160 autocorrelation in `sync.py` correlates against — same
periodicity property, used for acquisition rather than multipath.

## CP vs. pilots

|                    | Cyclic prefix                         | Pilot symbols                                  |
|--------------------|---------------------------------------|------------------------------------------------|
| Lives              | first 32 samples of *every* symbol    | 1 whole symbol in every 6 (`SYMS_PER_FRAME`)   |
| Costs              | 16.7% of time, always                 | 16.7% of symbols                                |
| Fixes              | delay spread, ISI/ICI                 | unknown, time-varying `H[k]`                    |
| Timescale          | ~ms (multipath delay)                 | ~hundreds of ms (Doppler)                       |
| Needs prior knowledge? | No — blind, structural            | Yes — receiver knows `pilot_sequence()`         |
| Received as        | discarded                             | divided out: `h_pilot[f] = raw[f,0] / self.pilot` |

Sequence of events for one data symbol:

1. The CP guarantees the channel acts as `Y[k] = H[k]·X[k]` — one
   multiply, no cross-talk. **Without this, step 3 is impossible**,
   because there is no single `H[k]` to divide by.
2. Pilots measure `H[k]` at 6.9 Hz (once per 144 ms frame) and
   Catmull-Rom-interpolate between frames to track fading.
3. Data symbols are divided by the interpolated `H[k]`.

The CP does not reduce the number of pilots needed, and pilots cannot
substitute for a CP. The pilot *rate* is set by an entirely different
constraint: pilots land 144 ms apart so interpolation can follow ~1 Hz
Doppler (RADE's `Bds ≈ 1 Hz`). Doppler spread and delay spread are the
two independent axes of an HF channel; the CP handles one and the pilots
handle the other.

One overlap worth knowing: the pilots also do a job the CP cannot touch —
sample-clock drift, tracked from the pilot phase slope across carriers.
The CP gives 32 samples of *static* tolerance; it does nothing about
tolerance that slowly walks away over a 95-second transmission.
