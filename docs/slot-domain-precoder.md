# Design: slot-domain precoder (not implemented)

Written 2026-07-25 alongside [latent-mixer-results.md](latent-mixer-results.md),
which established that no mixer on the latent grid's axes can affect
PAPR: the interleaver scatters the 46 latents that share an OFDM symbol
across the whole latent group, so only 0.083% of within-symbol latent
pairs are ones a per-position channel mixer can correlate.

A precoder that acts **after interleaving, on the 46 real values that
share one OFDM symbol**, is the smallest change that gives the model any
influence over within-symbol structure — which is what sets the crest
factor.

## Where it goes

Per OFDM symbol, a 46→46 real transform (equivalently 23→23 complex),
inserted between the interleaver and the carrier mapping:

```
encoder -> latents -> framing.interleave -> [PRECODER] -> slots_to_symbols -> OFDM
                                                             (23 carriers)
RX: demod -> pilot EQ -> symbols_to_slots -> [PRECODER^-1] -> deinterleave -> decoder
```

Three places must apply the identical transform, and they are the whole
cost of this feature:

1. `sstvae/modem/framing.py` — NumPy TX (`slots_to_symbols` caller in
   `modem.modulate`) and the RX inverse in `modem.demodulate` /
   `demodulate_blind`.
2. `sstvae/waveform_channel.py` — the torch stage-2 replica, or stage-2
   training optimizes a waveform the real modem does not transmit.
3. Nothing in `sstvae/models/` — the autoencoder stays unchanged. This
   is deliberately *not* an autoencoder feature.

The beacon carrier is excluded (only the 23 latent-carrying carriers are
spread). Pilot symbols and the Golay header are untouched.

## Variants, in the order worth trying

**1. Fixed DFT spreading (DFT-s-OFDM / SC-FDMA).** Precode each symbol's
23 complex values with a 23-point DFT before carrier mapping. Zero
learned parameters, exactly invertible, unitary, and a known-good
reference point from LTE uplink. Implement first: it exercises all the
plumbing above and gives a number to beat before any training is
involved.

**Caveat worth stating up front:** SC-FDMA's 3-4 dB crest-factor win
comes from its inputs being *constant-modulus* (QPSK/QAM), which makes
the time-domain signal look genuinely single-carrier. Our inputs are
Gaussian-ish analog latents (measured kurtosis 3.40), so the "single
carrier" signal is itself Gaussian and the naive expectation is a
**much smaller gain, possibly none**. Do not budget for 3-4 dB.

**2. Learned unitary precoder.** A 46×46 real (or 23×23 complex)
orthogonal matrix, shared across all symbols, trained through the
stage-2 channel under the existing PAPR penalty. Keep it exactly
unitary — parametrize as `expm(A - A^T)` or via a Cayley transform — so
it preserves the unit-RMS contract, is exactly invertible for RX, and
does not colour the noise. Ship the learned matrix in the checkpoint and
export it to the NumPy modem.

This is the variant with the actual upside, and the reason is *joint*
training rather than the precoder itself: a fixed precoder can only
rotate whatever the encoder happens to emit, but with the precoder in
the graph the encoder finally has a gradient path connecting latents
that share a symbol. Today that path does not exist at all. The
hypothesis to test is that encoder+precoder together can find low-crest
within-symbol structure, not that the precoder alone can.

**3. Learned general (non-unitary) linear map.** Rejected: needs a
pseudo-inverse at RX, breaks the power contract, and conditioning
becomes a training failure mode. No reason to accept those costs before
variant 2 is shown to be insufficient.

## Consequences that must be designed for

**Erasure and confidence accounting changes, and this is the big one.**
Today a lost slot is one erased latent with weight 0, and
`framing.deinterleave` returns that weight mask directly. After
spreading, every latent in a symbol is a combination of all 23 carriers,
so a single faded or erased carrier damages all 46 latents in that
symbol instead of erasing one. Consequences:

- `deinterleave`'s per-latent weight semantics no longer follow from
  slot presence; per-latent confidence has to be *derived* through the
  inverse precoder from per-carrier confidence.
- Equalization matters much more. With per-carrier fading, a deep fade
  on one carrier is spread across all 23 latents, so zero-forcing will
  amplify noise badly — MMSE weighting off the pilot-EQ channel estimate
  becomes necessary rather than optional.
- This trades erasure-locality for frequency diversity. That is the
  known OFDMA-vs-SC-FDMA tradeoff and it may well be a *net loss* on
  Watterson fading channels even if it wins on PAPR. Evaluate on mpp/mpd,
  not just AWGN.

**On-air compatibility.** A precoded transmission is unintelligible to a
non-precoding receiver and vice versa. It needs a waveform version bit —
either a new mode letter or a spare field in the Golay-coded header.
Blind decode (`demodulate_blind`) never sees the header, so it needs the
setting out of band or a fixed default.

**Interaction with the interleaver.** Spreading happens after
interleaving, per symbol. The interleaver scatter is *helpful* here: it
means the 46 values entering each precoder are near-independent, which
is the regime the transform is designed for. No change needed.

## Alternative worth recording

Rather than adding a precoder, **change the interleaver** so each symbol
draws its 46 latents from a structured set (one spatial neighbourhood,
or one channel) instead of scattering them. The existing latent-grid
mixer would then have real leverage over within-symbol structure, at
zero waveform-contract cost. The reason not to lead with this: the
scatter exists for erasure robustness — burst erasures currently damage
latents spread across the image rather than destroying one region — so
this trades a property that is known to work for one that is
speculative. Cheap to test in stage-2 simulation before committing.

## Suggested order of work

1. Fixed DFT spreading end-to-end (framing + waveform_channel + a
   round-trip test), measure PAPR and PSNR-vs-SNR on AWGN and mpp. This
   is where the caveat above gets settled empirically and cheaply.
2. Only if the plumbing holds up: learned unitary, trained jointly.
3. MMSE per-latent confidence derivation, needed for either to perform
   on fading channels.
