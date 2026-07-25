# Latent MLP-mixer experiment (2026-07-25)

**Hypothesis.** A simple MLP mixer (no dimension change) at the end of
the encoder and the start of the decoder lets the model adapt to the
channel and improves the PSNR/PAPR tradeoff.

**Result.** Negative, and closed. No PAPR effect at either tanh
placement; the encoder mixer is inert (deleting it costs 0.009 dB); the
apparent +0.29 dB PSNR gain was reproduced exactly by a `--mixer-depth 0`
control, so it belongs to the resumed run, not the architecture. The
negative result has a structural explanation (below) that rules out this
*class* of mixer for PAPR, not just this instance.

## What was built

`--mixer-depth N` (`sstvae/models/autoencoder.py`): residual per-position
MLP blocks over the channel axis, `GroupNorm -> 1x1 conv -> GELU ->
1x1 conv -> +x`, hidden width = channel count (`--mixer-hidden-mult`).
Encoder side is block-diagonal over the three latent groups so
progressive truncation stays valid; decoder side is full-width over
`[z*w, w]`. Output projections are zero-initialized, so blocks are exact
identities at init and drop into a pre-mixer checkpoint unchanged.
`--mixer-place {post_tanh,pre_tanh}` selects which side of the bounding
tanh the encoder mixer sits on. `--mixer-depth 0` is the original
architecture, bit-for-bit.

## Runs

Baseline is the pre-mixer stage-2 run at epochs 165-170, which had been
stable within noise for several epochs.

| | baseline | pre_tanh, +50 ep | post_tanh, +6 ep |
|---|---|---|---|
| train_loss | 0.11446 | 0.11058 | 0.11397 |
| papr_db | 4.2678 | 4.2680 | 4.2677 |
| papr_pre_db | 9.656 | 9.661 | 9.654 |
| val clean | 25.68 | 25.97 | 25.72 |
| val 8dB_e20 | 24.86 | 25.09 | — |
| val text | 24.96 | 25.36 | — |

PAPR is unchanged to four decimal places in both runs. Epoch cost rose
~2.9%.

## Diagnostics

Residual-branch magnitude relative to its own input, measured on the
trained checkpoints:

| | encoder mixer | decoder mixer |
|---|---|---|
| pre_tanh @ epoch 227 (+50) | 0.035 | 0.59 |
| post_tanh @ epoch 176 (+6) | 0.026 | — |

The encoder mixer barely leaves its identity init either way, so the
tanh was *not* what was suppressing it — the leading hypothesis going
into the second run, now ruled out.

The decoder mixer does move, but splitting its action by half: 0.052 on
the `z` half, **0.84 on the weight planes**. On a clean forward pass the
weight planes are constant ones, so most of what it learned is a learned
constant bias into the decoder's first conv — added capacity, not
channel adaptation.

## Why no mixer on the latent grid can move PAPR

PAPR is a per-OFDM-symbol property: peaks come from summing the 23 data
carriers of one symbol, i.e. `NC_LATENT * 2 = 46` real latents
(`framing.slots_to_symbols` reshapes a frame's slots to
`(DATA_SYMS_PER_FRAME, NC_LATENT, 2)`). Counting which latents actually
share a symbol across a full mode-C transmission:

| within-symbol latent pairs | count | share |
|---|---|---|
| total | 3,415,500 | 100% |
| sharing a spatial position — all a **channel** mixer can correlate | 2,831 | 0.083% |
| sharing a channel — all a **token/spatial** mixer could correlate | 57,140 | 1.67% |

The interleaver's job is to scatter adjacent latents across time, so the
46 latents in a symbol are effectively a random draw from the group. Any
mixer defined on the latent grid's own axes is nearly orthogonal to the
symbol grouping and cannot shape the sum that creates peaks. Adding the
token-mixing half would not have rescued it — 1.67% is still nothing.

Second, independent reason: measured latent kurtosis is 3.40, already
essentially Gaussian, and a sum of 46 near-Gaussian values is Gaussian
by CLT regardless of the marginals. There was no headroom in the
marginal distribution to exploit.

This also undercuts the original rationale for choosing channel-mixing
over token-mixing (that erasure bursts hit a run of positions within one
channel) — the interleaver scatters those too.

## Control run: the +0.29 dB was not the mixer

A `--mixer-depth 0` resume from the same epoch-170 checkpoint, same
`--lr`/`--epochs`/seed (`arodland/sstvae-s2-640-mixer0`), reproduces the
climb:

| window | mixer | control |
|---|---|---|
| 171-180 | 25.711 | 25.706 |
| 181-190 | 25.748 | 25.749 |
| 191-200 | 25.816 | 25.809 |
| 201-210 | 25.950 | 25.871 |

So the gain belongs to the resumed run, not the architecture. Two
candidate causes, not separated and not worth separating: the cosine
`T_max` restart at the resume boundary, and image augmentation having
been enabled shortly before epoch 170 with its generalization benefit
still being realized. Both runs get both, which is why they agree.

Other ablations on the epoch-227 mixer checkpoint, forcing each half
back to exact identity:

| | clean | e20 | text |
|---|---|---|---|
| as trained | 25.573 | 24.709 | 25.071 |
| encoder mixer -> identity | 25.564 | 24.711 | 25.064 |
| decoder mixer -> identity | 22.145 | 22.029 | 21.795 |

Deleting the encoder mixer costs **0.009 dB** — it is inert, so the
transmitted latents are effectively unchanged and channel adaptation
cannot be the mechanism for anything observed. (The decoder row is
co-adaptation damage from removing a block the decoder trained against
for 50 epochs, not a measure of its value.) Consistently, a matched-epoch
A/B of `pre_tanh` vs `post_tanh` over epochs 171-176 differs by 0.003 dB
on clean — placement is irrelevant when the block is inert.

Untested: whether the decoder-side mixer helps under erasures, where the
weight planes carry real information rather than constant ones.

## Status

Code is kept, defaulted off (`--mixer-depth 0`). Checkpoints record
`mixer_depth`/`mixer_hidden_mult`/`mixer_place`;
`SSTVAE.mixer_kwargs_from_checkpoint` reads a placement-less mixer
checkpoint back as `pre_tanh` so the first experiment's checkpoints keep
their trained semantics.

The follow-up — a slot-domain precoder, the one mechanism that *can*
reach within-symbol structure — was then measured and also came up
empty, for a related reason (unitary transforms preserve Gaussianity).
See [slot-domain-precoder.md](slot-domain-precoder.md).
