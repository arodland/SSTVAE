# ONNX export and quantisation

**Status: exported and published; the runtime path is not implemented.**
`scripts/export_onnx.py` exists and the six `v1` artifacts are on the
Hub, but no code *loads* them yet — `codec.py` is still torch. See "What
implementing it would involve" for what remains.

Numbers were measured on 2026-07-26 against the published `v1.pt`
checkpoint, on an x86-64 Linux box (24 cores, onnxruntime 1.28 CPU
provider, 4 intra-op threads), and the accuracy figures were **re-measured
2026-07-27** through the real publish script — which corrected the int8
row upward. Where the two disagree, the later measurement is the one
taken from the artifacts actually being distributed.

## Why bother

Packaging. `torch` is by far the largest thing a receiving station
installs, and it exists solely to run two convolutional passes:

| | torch path | ONNX path |
|---|---|---|
| runtime installed | 336 MB | **27 MB** |
| model artifacts | `v1.pt`, ~40 MB | 41.3 MB fp32 / 20.7 fp16 / 12.7 int8 |
| `import` cost | 0.48 s | 0.04 s |

`onnx` (50 MB), `onnxscript` (108 KB) and `onnxconverter-common` are
**export-time only** — `onnxruntime` alone loads and runs `.onnx` files.

That takes a `pip install sstvae[gui]` from roughly 400 MB to about
90 MB before any bundling, and puts a single-file distribution in the
100–120 MB class instead of 300+. Note that the model artifacts row is
not part of a download at all under the first-run-fetch decision below —
it is a cache cost, paid once.

## Exporting

The model is a plain convolutional autoencoder with fixed shapes, which
is the easy case: export takes seconds, needs no rewrites, and hits no
unsupported operators at opset 17.

Use the **default `dynamo=True` exporter**. The legacy TorchScript path
(`dynamo=False`) is in maintenance mode. They were measured to produce
numerically identical graphs — same errors to every digit, same
inference speed — so there is no reason to opt out of the default.

Two practical wrinkles with the dynamo exporter:

- It needs `onnxscript` installed, which is not a `torch` dependency.
  Without it the export fails with `ModuleNotFoundError`.
- It writes weights as **external data** by default: a ~30 KB graph plus
  a `.onnx.data` sidecar. Ask for the weights inline when publishing —
  `checkpoint.py` pins named files, and four artifacts that must arrive
  together are worse than two.

## Faithfulness at fp32

The codec *is* the on-air format, so the question is not just "does it
run" but "would an ONNX station and a torch station still understand
each other".

| | max abs difference vs torch | reference scale |
|---|---|---|
| encoder | 7.03e-06 | latents are unit RMS → ~103 dB down |
| decoder | 2.21e-06 | pixels in 0..1; one 8-bit step is 3.9e-03 |

The modem delivers about 8.7 dB latent SNR, so the fp32 ONNX/torch
divergence sits roughly **112 dB below the channel noise**. End to end,
an ONNX-encoded mode A transmission decoded to 27.31 dB PSNR on a torch
receiver and 27.31 dB on an ONNX receiver — identical.

fp32 ONNX and torch are the same codec.

## Quantisation

The instinct to treat quantisation as dangerous for an on-air format is
wrong here, and it is worth understanding why before reading the table:
**latents are analog values.** Quantisation is just another small
additive noise source on a channel that already carries a much larger
one. There is no bitstream to corrupt and no compatibility cliff — a
quantised station transmits a slightly noisier picture and every
receiver decodes it without knowing or caring.

So the yardstick is not fp32, it is the channel. At the modem's
operating point the channel puts **0.367 RMS of noise on unit-RMS
latents**:

Re-measured 2026-07-27 by `scripts/export_onnx.py` (the real publish
path) over 10 COCO validation images. **The int8 row is worse than this
document originally recorded** — see the correction below.

| | model size | encoder latent error | vs channel noise |
|---|---|---|---|
| fp32 | 41.5 MB | 1.95e-06 RMS | 105 dB below |
| fp16 | 20.9 MB | 4.58e-04 RMS | 58 dB below |
| int8 | 12.8 MB | **1.88e-01 RMS** | **5.8 dB below** |

| | local decode, whole pipeline at this precision |
|---|---|
| torch | 26.71 dB |
| fp32 | 26.71 dB (−0.000) |
| fp16 | 26.71 dB (−0.000) |
| int8 | 26.43 dB (−0.282) |

**fp16 is free.** Half the size, no measurable cost anywhere. This
survived re-measurement unchanged and is the basis for shipping fp16.

**int8 costs about 0.28 dB PSNR** for a 3.2× smaller model — roughly
twice what was first recorded here (0.13 dB), because the encoder latent
error is 1.88e-01 RMS rather than 8.0e-02.

> **Correction, 2026-07-27.** The original 8.0e-02 figure does not
> reproduce with the current toolchain (torch 2.13, onnxruntime 1.28)
> and the export path we actually publish from. Investigated:
> `quantize_dynamic` turns all 13 Convs into `ConvInteger`, which
> supports only a **per-tensor** weight scale, so `per_channel=True` is
> silently a no-op — verified, byte-identical output and identical
> error. The earlier number was probably taken from a differently
> optimized graph. The lever, if int8 accuracy ever matters, is **static
> (calibrated) quantisation**, which yields `QLinearConv` and is both
> more accurate and faster; it is not implemented.

The arithmetic still explains the pictures, but it now lands somewhere
less comfortable. Quantisation noise at 0.188 RMS adding in power to
channel noise at 0.367 gives 0.412 — a **1.0 dB effective SNR penalty**,
not the 0.2 dB originally derived here.

> **Not re-measured:** the on-air column (quantised TX → fp32 RX)
> originally read 24.79 / 24.79 / 24.63 dB. Only the local pipeline was
> re-run. Given the revised arithmetic, assume the int8 on-air cost is
> nearer 1 dB than the 0.15 dB recorded before, and re-measure it before
> relying on the number.

> A first single-image run had int8 *beating* fp32 by 0.17 dB — the
> effect is the same size as the per-image spread, so anything quoted
> from one picture is noise. Use ≥10 images. Same trap as the acquisition
> sweeps in `docs/todo.md`, which manufactured a whole phantom finding
> out of 6 samples per point.

### The asymmetry that matters

Decoder and encoder quantisation are not the same decision:

- **Decoder** quantisation only changes the picture *you* see. It cannot
  affect anyone else and needs no coordination. Quantise freely.
- **Encoder** quantisation changes what goes on the air, so its cost is
  paid by every receiver. The re-measurement above sharpens this from a
  caution into a recommendation: at 1.88e-01 RMS the int8 encoder sits
  only 5.8 dB under the channel noise and costs ~1 dB of effective SNR.
  **Use fp16 for the encoder.** It is free, and the picture you are
  spending is someone else's.

### Do not chase precision below int8

Not measured, and not worth measuring for on-air use. Note that the
revised numbers put **int8 itself** close to this line rather than
safely inside it: 5.8 dB under the channel noise is no longer
"comfortably under", which is the whole premise of the
graceful-degradation argument. int8 remains reasonable for the decoder
and for small devices; below it there is nothing to recommend.

## Speed

| | encoder | decoder |
|---|---|---|
| torch fp32 | 22.4 ms | 57.7 ms |
| onnx fp32 | 36.4 ms | 86.6 ms |
| onnx fp16 | 36.5 ms | 86.1 ms |
| onnx int8 | 188.7 ms | 228.3 ms |

Two honest caveats:

**ONNX is ~1.5× slower than torch here**, and no exporter choice or
thread setting recovered it (`intra_op_num_threads` swept 1/4/24; 4 was
best, 1 was 190 ms). This does not matter in context: the receive loop
reconstructs once per 5-second poll, `demodulate` alone costs 173 ms,
and transmissions run 32–95 s. But it is a real regression, not a wash.

**fp16 is not faster on the CPU provider** — identical timings to fp32,
consistent with the convolutions running in fp32 after an up-cast.
Treat fp16 as a download-size win, not a compute one.

**The int8 timings are an x86 artifact.** Dynamic quantisation without
well-matched kernels for these convolution shapes is often slower on
desktop CPUs and faster on ARM — which is exactly the small-device case
int8 would be for. Benchmark on the target before concluding anything;
this table would talk you out of int8 for the wrong reason.

## Intended direction (Andrew, 2026-07-27)

- **Publish all three precisions with every model revision.** They cost
  75 MB of Hub storage between them and they save every other
  implementer from rolling their own export — which is the case that
  actually risks divergence, since a third party choosing different
  quantisation settings could land well outside the 0.28 dB measured
  here — as this document's own superseded 0.13 dB figure demonstrates.
  Canonical artifacts make "compatible app" a checkable claim rather than
  a hope.
- **Use fp16 in the packaged distributions.** Free by every measure
  above, and halves the largest single artifact.
- **Fetch it on first run rather than baking it into the installer**
  (revised 2026-07-27; the original wording here said "ship fp16 *in*
  the packaged distributions", which read as bundling). Keeps
  `checkpoint.py`'s cache-first, immutable-filename model — one
  mechanism for CLI and app alike — and keeps ~20 MB out of every
  download. The cost is a network dependency at first launch, so an
  offline machine needs a clear message and a manual import path.
  See `docs/native-app.md` decision 5.

All three follow from the fp32 equivalence result: because every
precision decodes on every other precision's receiver, the choice is a
local packaging matter and not a format decision. That is worth stating
explicitly for anyone building against these artifacts — **there is one
on-air format, and the precisions are not variants of it.**

The same equivalence is what makes first-run fetch safe rather than
risky. A station that fetches fp16 and a station that was shipped fp32
interoperate exactly, so delivery and precision are independent choices
and either can be revisited without touching the other.

## What implementing it would involve

1. ~~`scripts/export_onnx.py`, run at publish time from the `.pt` so the
   artifacts cannot drift from the checkpoint.~~ **Done 2026-07-27.** It
   verifies every artifact against the torch model before writing, and
   refuses to push if a tolerance is breached; provenance (source
   checkpoint name and sha256) is stamped into each `.onnx`'s
   `metadata_props` and into a published manifest. It is in
   `launch_job.sh`'s code snapshot, so training jobs carry it.
2. An ONNX path in `sstvae/codec.py`; torch stays for training only.
3. Port `SSTVAE.latents_to_flat` / `flat_to_latents` to numpy — they are
   pure reshape and concatenate, about ten lines. Without this the
   runtime still imports torch and the whole exercise is pointless.
4. `checkpoint.py` publishes and pins the `.onnx` artifacts; its
   immutability model already fits.
5. A test asserting each exported artifact matches the `.pt`. Exact
   comparison (~1e-5) works for fp32; quantised variants need a
   PSNR-based bound instead, since exact match is only meaningful
   without quantisation.

## Reproducing

The accuracy figures are reproduced by the publish script itself, which
prints the whole table on every run:

```
scripts/export_onnx.py --images path/to/coco-val --n-probe 10
```

Nothing is uploaded without `--push`, so this is safe to run for
measurement alone. Two things it does that a hand-rolled probe will get
wrong: `external_data=False` (**not** the default — otherwise the
weights land in a `.onnx.data` sidecar and each artifact becomes two
files that must travel together), and verification against real
photographs. Synthetic low-frequency probes make int8 look ~3× worse
than it is, because the model is far off its training distribution and
reconstructs them at 21.5 dB rather than 26.7 dB.

The **speed** table above and the **on-air** figures are not reproduced
by the script — they were one-off measurements, the latter run through
`Modem.modulate` / `demodulate`.
