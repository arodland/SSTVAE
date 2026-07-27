# ONNX export and quantisation

**Status: measured, not implemented.** Nothing in the codebase uses ONNX
yet. This records what an ONNX runtime path would cost and buy, so the
decision doesn't have to be re-derived. All numbers below were measured
on 2026-07-26 against the published `v1.pt` checkpoint, on an x86-64
Linux box (24 cores, onnxruntime 1.28 CPU provider, 4 intra-op threads).

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
100–120 MB class instead of 300+.

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

| | model size | encoder latent error | vs channel noise |
|---|---|---|---|
| fp32 | 41.3 MB | — | — |
| fp16 | 20.7 MB | 3.7e-04 RMS | 60 dB below |
| int8 | 12.7 MB | 8.0e-02 RMS | 13.3 dB below |

Measured over 10 COCO val2017 images, mode A:

| | local decode (fp32 latents) | on air, quantised TX → fp32 RX |
|---|---|---|
| fp32 | 26.44 dB | 24.79 dB |
| fp16 | 26.44 dB (−0.00) | 24.79 dB (+0.00) |
| int8 | 26.31 dB (−0.13) | 24.63 dB (−0.15) |

**fp16 is free.** Half the size, no measurable cost anywhere.

**int8 costs about 0.15 dB PSNR** for a 3.2× smaller model, and even
that is arithmetic rather than surprise: quantisation noise at 0.08 RMS
adding in power to channel noise at 0.367 gives 0.376, a 0.2 dB
effective SNR penalty, which is what the pictures show.

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
  paid by every receiver. Still small (0.15 dB for int8), but it is
  someone else's picture, so be more conservative here than you would be
  locally.

### Do not chase precision below int8

Not measured, and not worth measuring for on-air use: below int8 the
quantisation noise stops being comfortably under the channel noise, and
the graceful-degradation argument above stops applying.

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
  quantisation settings could land well outside the 0.15 dB measured
  here. Canonical artifacts make "compatible app" a checkable claim
  rather than a hope.
- **Ship fp16 in the packaged distributions.** Free by every measure
  above, and halves the largest single file in the bundle.

Both follow from the fp32 equivalence result: because every precision
decodes on every other precision's receiver, the choice is a local
packaging matter and not a format decision. That is worth stating
explicitly for anyone building against these artifacts — **there is one
on-air format, and the precisions are not variants of it.**

## What implementing it would involve

1. `scripts/export_onnx.py`, run at publish time from the `.pt` so the
   artifacts cannot drift from the checkpoint.
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

The probes are not committed — they were one-off measurements. The
shape of them:

```python
torch.onnx.export(model.encoder, (image,), "encoder.onnx",
                  input_names=["image"], output_names=["latents"],
                  opset_version=17)          # dynamo=True is the default
```

then `onnxruntime.quantization.quantize_dynamic(..., weight_type=QInt8)`
for int8 and `onnxconverter_common.float16.convert_float_to_float16(...,
keep_io_types=True)` for fp16, comparing against torch on the same
inputs and through `Modem.modulate` / `demodulate` for the on-air
figures.
