# TODO

Known work items that aren't bugs with a clear fix, kept here so the
reasoning doesn't have to be rediscovered.

## Wider acquisition search, for a mis-tuned counterpart

**Goal.** Recover a transmission from a station whose dial is off by
more than the current tolerance — hundreds of Hz, not tens. Not a
user-facing "pick your centre frequency" knob; an opt-in wider search
(or an automatic second pass) so a mis-tuned partner still decodes.

**This is achievable, and it is purely an acquisition-side change.**
The demod path has no dependence on absolute centre frequency at all.
Measured: signal placed at each centre, receiver's heterodyne retuned to
match, recovered latent SNR (mode B, no noise, so deterministic):

| centre (Hz) | 900 | 950 | 1000 | 1012.5 | 1250 | 1500 | 1750 | 2000 | 2100 |
|---|---|---|---|---|---|---|---|---|---|
| latent SNR (dB) | 8.73 | 8.73 | 8.73 | **8.63** | 8.73 | 8.73 | 8.73 | 8.73 | 8.73 |

Identical to the last digit across the whole usable passband. Once the
signal is found and its centre known, where it sits is irrelevant.

**What would have to change** (all small, all in acquisition):

- `sync.acquire()`'s `max_bins=2` caps the integer-bin search at
  ±100 Hz. This is the hard limit today.
- `dsp.sync_lowpass()` is `firwin(129, 850, fs=FS)` around the *nominal*
  centre. At a 1000 Hz centre the carriers land at −1075..−475 Hz in
  that filter's frame, so half of them are cut — this is why a
  mis-tuned signal currently shows `NO LOCK` well before `max_bins`
  becomes the binding constraint. Widen or recentre it. Sync-only, so
  the ISI argument that keeps `to_baseband()` unfiltered doesn't apply.
- `dsp.to_baseband()` takes `FCENTER` from config. Note
  `ofdm.BASEBAND_FREQS` and the DFT matrix need *no* change: the wanted
  carriers land at `g - FCENTER` regardless of where the signal is, so
  only the mixing frequency moves.
- `sync.acquire_blind()` already searches CFO directly over
  `max_offset_hz` (default 55.0) at `bin_step_hz` resolution and has no
  ±100 Hz structural limit — widening its range may be the cheaper path
  to a "wide search" mode than reworking `acquire()`.

**Cost.** Acquisition CPU grows with the search range, which is why this
should be opt-in rather than always-on — it's already the dominant cost
of the blind path (see `--blind-search-seconds`).

**Non-issue: the 25 Hz grid.** A mis-tuned radio won't land on any
convenient grid, but it doesn't need to. Working through the mixing
algebra, after the receiver corrects to put carriers on their bins the
heterodyne image sits at −(g + 2ε + FCENTER), where ε is the *true*
mistune — it does not depend on the receiver's choice of mixing
frequency, so no amount of cleverness at the receiver moves it. The
penalty peaks where `2ε mod 50 = 25` and is bounded at about **0.1 dB**
(measured: 8.63 vs 8.73 at a deliberately off-grid centre; same effect
and same size as the ripple documented under the item below). Accept it;
don't engineer around it, and specifically don't try to snap the
correction to a 25 Hz grid — that would leave up to 12.5 Hz of genuine
residual CFO, which is far more damaging than 0.1 dB.

**Practical bound.** The 1150 Hz carrier span has to fit inside a
typical 300–2700 Hz SSB filter, which limits the centre to roughly
900–2100 Hz — exactly the range measured above. Beyond that the
transmitter's own filtering, not the modem, is the limit.

## Improve acquisition at large frequency offsets

**Confirmed:** offsets approaching the advertised ±50 Hz limit cost
real acquisition sensitivity — but only near the SNR threshold, which
is why it's easy to miss.

Mode C, AWGN (no fading, so deep-fade luck can't mask the effect),
6 random noise seeds per point, counting successful preamble sync:

| SNR | 0 Hz | 25 Hz | 45 Hz | 50 Hz | 55 Hz |
|---|---|---|---|---|---|
| 0 dB | 5/6 | 6/6 | 4/6 | 5/6 | 5/6 |
| −1 dB | 4/6 | **6/6** | **1/6** | **1/6** | 3/6 |

At −1 dB, offsets ≥45 Hz acquire 2/12 versus 10/12 for offsets ≤25 Hz.
One dB higher and the effect nearly vanishes; at 6 dB with `mpp` fading
it is invisible (7–8/8 flat across 0–50 Hz, 8 seeds per point). So the
cost is roughly *a decibel of acquisition threshold* at large offset,
not an outright failure — which matches the impression that "most
failures happen near the edges" without contradicting the modes working
fine off-frequency in good conditions.

**Mechanism: not yet identified.** The obvious theory is wrong, so
don't start there. `sync.acquire()` estimates CFO in two stages
(`sstvae/modem/sync.py`):

1. A lag-`M` autocorrelation over the periodic preamble gives a
   fractional CFO, unambiguous only over ±FS/(2M) = **±25 Hz**.
2. The remainder is an integer multiple of the 50 Hz carrier spacing,
   resolved by scoring candidate bins (`f_cand = f_frac + m_bin*FS/M`)
   against the known preamble template and taking the best.

That structure predicts the *worst* performance at the half-bin
boundary, i.e. 25 Hz, where noise most easily pushes the fractional
estimate into the neighbouring bin's basin. **25 Hz measured best
(6/6 at both SNRs)** — the cleanest point in the sweep. So plain
wrong-bin ambiguity is not the story.

The response is also non-monotonic (45 and 50 Hz worst, 55 Hz partly
recovering), which argues against a simple rolloff explanation.

**Already ruled out: the sync lowpass.** `sync_lowpass()` is
`firwin(129, 850, fs=FS)` on the complex baseband, and the carriers sit
at −550..+600 Hz around `FCENTER` before any offset, so it looked like a
large offset might push them into the skirt. Measured attenuation at the
shifted carrier positions is **flat to within 0.02 dB out to 55 Hz** at
both band edges — the filter is not costing anything here. Don't
re-check this.

Candidates to investigate, cheapest first:

- Instrument `acquire()` on the failing cases: log `f_frac`, the chosen
  `m_bin`, the per-candidate scores, and the final `f_hat` versus truth.
  This distinguishes "picked the wrong bin" from "detected nothing" in
  one run, and the sweep above gives ready-made failing seeds. Start
  here — with the filter excluded and half-bin ambiguity contradicted,
  there is no strong prior left to test against.
- Check whether residual CFO after bin selection degrades the *template
  correlation* used for fine timing: a phase ramp across the correlation
  window costs coherent gain, and the residual grows with total offset
  even when the bin is chosen correctly.
- Score bin candidates over more than the preamble — fold in the first
  few frames' pilots before committing. The information is already
  demodulated; only the decision is premature.
- Keep the top-2 candidates and let the Golay header arbitrate, turning
  a hard argmax into a cheap 2-way retry.

**Scope check: this is an acquisition problem, essentially entirely.**
Once a signal is acquired and its centre frequency known, decode quality
is nearly offset-independent. Measured modem-only (random latents,
mode B, *no noise at all*, so the numbers are deterministic), recovered
latent SNR vs offset:

| offset | 0 | 6.25 | 12.5 | 18.75 | 25 | 31.25 | 37.5 | 43.75 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| latent SNR (dB) | 8.73 | 8.69 | 8.68 | 8.70 | 8.73 | 8.67 | 8.66 | 8.70 | 8.73 |

Total spread **0.07 dB** — negligible beside the ~1 dB of acquisition
threshold above. The small structure that is there is real and
explainable: `to_baseband()` is deliberately unfiltered and relies on
the heterodyne image landing exactly on the 50 Hz bin grid, where the
160-sample demod correlation nulls it. After acquisition corrects by δ,
the image sits **2δ** off that grid, so leakage peaks where `2δ mod 50`
is 25 Hz (δ = 12.5, 37.5 → 8.66–8.68 dB) and vanishes where it is 0
(δ = 0, 25, 50 → 8.73 dB). The ordering follows `2δ mod 50` perfectly,
and the CFO estimate error is constant (−0.021 Hz) across the sweep, so
it isn't residual CFO doing this.

Conclusion: don't go looking for offset sensitivity in the demod/EQ
path. Fix acquisition.

**Safety net.** Less catastrophic than it looks: the beacon gives a
fresh sync opportunity roughly every 10 s for the whole transmission
(`sync.acquire_blind` / `Modem.demodulate_blind`), and
`sstvae_listen.py` falls back to it automatically. A missed preamble
costs latency and the header's mode information, not the image. But the
preamble path sets the floor for `sstvae_decode.py` on a recording, and
`acquire_blind` searches CFO bins directly at `bin_step_hz` resolution
without a preamble — so it may already be more offset-robust, which
would itself be a useful comparison.

**Reproducing.** Sync success vs offset, everything else fixed:

```sh
uv run sstvae_encode.py photo.jpg tx.wav --mode C --model checkpoint.pt
for off in 0 25 45 50 55; do
  for seed in 0 1 2 3 4 5; do
    uv run sstvae_simulate.py tx.wav rx.wav --snr -1 --freq-offset $off --seed $seed
    uv run sstvae_decode.py rx.wav out.png --model checkpoint.pt
  done
done
```

Count `SyncError: header decode failed` against successes. Two traps:
use AWGN rather than fading (with fading, whether the preamble lands in
a deep fade swamps the effect), and stay within about a dB of the
threshold — at 6 dB there is nothing to see.
