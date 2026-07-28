# ONNX export and quantisation

**Status: implemented.** `sstvae/codec.py` runs on onnxruntime, the six
`v1` artifacts are published, and the `cli`/`listen`/`gui` extras no
longer depend on torch at all. Torch is training-only (plus `dev`, where
it is the reference implementation the tests check against).

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
| runtime installed | 345 MB | **53 MB** |
| model artifacts | `v1.pt`, ~40 MB | 41.5 MB fp32 / 20.9 fp16 / 12.8 int8 |
| `import` cost | 0.48 s | 0.04 s |

`onnx` (50 MB), `onnxscript` (108 KB) and `onnxconverter-common` are
**export-time only** — `onnxruntime` alone loads and runs `.onnx` files.

Measured after implementation, `uv pip install '.[cli]'` into a clean
venv: **263 MB, down from ~555 MB.** The model artifacts row is not part
of that at all — they are fetched on first run (see below), so they are
a cache cost paid once, not a download.

> **Two figures in the first draft of this table were wrong**, both
> flattering. onnxruntime is 53 MB *installed*; 27 MB was its wheel
> size. And the claim that this took an install "from roughly 400 MB to
> about 90 MB" ignored the base dependencies, which dominate now that
> torch is gone: scipy is 110 MB, numpy 57 MB, pillow 20 MB. The real
> saving is 2.1×, not 4.4×. It is still the single biggest thing that
> can be removed from a receiving station — but scipy, not torch, is now
> the largest dependency, and `sstvae/modem/` uses only `firwin`,
> `hilbert`, `fftconvolve` and `resample_poly` from it.

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
| int8 | 15.1 MB | 7.31e-02 RMS | 14.0 dB below |

Whole pipeline at each precision, against the torch reference — and
measured on **two kinds of picture**, because the difference between
them turned out to be the whole story:

| | COCO photographs | off-distribution |
|---|---|---|
| fp32 | −0.000 dB | −0.000 dB |
| fp16 | −0.000 dB | −0.000 dB |
| int8 | −0.002 dB | −0.112 dB |
| *int8, untuned export* | *−0.187 dB* | *−1.573 dB* |

**fp16 is free.** Half the size, no measurable cost anywhere. This
survived every re-measurement and is the basis for shipping fp16.

**int8 is now very nearly free too**, at 2.7× smaller than fp32 — but
only because the export leaves one layer per part at fp32 (see
"Sensitive layers stay at fp32"). Untuned it costs 0.19 dB on
photographs and **1.57 dB** on everything else. As an *on-air* cost the
tuned int8 encoder is 0.17 dB of effective SNR, against ~1 dB untuned.

> The off-distribution column uses smooth synthetic probes at a seed the
> tuner never saw, so these are held-out numbers, not the search
> scoring itself.

> **Correction, 2026-07-27.** The original 8.0e-02 figure does not
> reproduce with the current toolchain (torch 2.13, onnxruntime 1.28)
> and the export path we actually publish from. Investigated:
> `quantize_dynamic` turns all 13 Convs into `ConvInteger`, which
> supports only a **per-tensor** weight scale, so `per_channel=True` is
> silently a no-op — verified, byte-identical output and identical
> error. The earlier number was probably taken from a differently
> optimized graph.
>
> **Superseded 2026-07-28**, before anyone could have these artifacts
> cached. Per-tensor scaling turned out to be the whole story, and the
> fix is in "Sensitive layers stay at fp32" below: int8 now measures
> 5.83e-02 RMS, **16.0 dB** under the channel noise. The numbers in the
> table above are the tuned ones; the untuned 1.88e-01 is what you get
> with `--no-int8-tuning`.

The arithmetic explains the pictures. Quantisation noise at 0.0731 RMS
adding in power to channel noise at 0.367 gives 0.374 — a **0.17 dB
effective SNR penalty**. Untuned, at 0.188 RMS, it was 1.0 dB.

### Sensitive layers stay at fp32

`ConvInteger` — what `quantize_dynamic` emits — carries **one scale per
weight tensor**. So quantisation error is not spread evenly across a
network: one tensor with outlier weights forces a coarse scale on
everything in it, and dominates the total. On `v1`'s encoder a single
0.59 MB layer accounted for nearly all of it. Leaving just that one at
fp32 moved the encoder from 1.88e-01 to 7.31e-02 RMS — **5.8 dB to
14.0 dB under the channel noise** — for 8% more file.

`scripts/export_onnx.py` therefore *measures* which layers to leave
alone rather than hardcoding them, since the sensitive tensor is a
property of the trained weights and will move between revisions. It
greedily excludes the worst until the error is under target.

Three decisions worth keeping, each of which was got wrong first:

- **Tune against off-distribution pictures, or the search measures
  nothing.** This is the big one. Sensitivity barely shows on the
  photographs the model was trained on: the fully-quantised decoder
  costs 0.10 dB on COCO and **1.54 dB** on smooth synthetic probes.
  Scored on COCO alone the search sees 0.10 dB, concludes there is
  nothing worth fixing, and ships the 1.54 dB. The tuning set therefore
  always mixes in synthetic probes
  (`INT8_TUNE_SYNTHETIC_PROBES`). This is not a corner case — operators
  send test cards, charts, callsign graphics and screenshots.
- **The stopping rule is an absolute target, not a relative
  improvement.** 13 dB under the channel noise for the encoder,
  0.10 dB of picture for the decoder. A relative rule ("keep going while
  the next exclusion cuts error by 15%") looks reasonable and is not: at
  the tail a 25% error reduction is worth 0.00 dB, and that rule talked
  itself into three decoder exclusions that together bought 0.005 dB.
- **Both parts get tuned, but to different targets, and the asymmetry
  sets them.** Encoder error goes on the air and every receiver pays it,
  so it is held to the channel-noise yardstick. Decoder error only
  changes its own operator's picture, so a tenth of a dB is plenty —
  a tighter 0.05 dB target buys a second decoder exclusion worth 0.02 dB
  for +1.8 MB, which is the wrong trade for the precision that exists
  for constrained devices.

Measured decoder options, kept so the choice does not get re-litigated.
Note how different the two columns are, and that the COCO column alone
would justify shipping nothing at all:

| decoder int8 | size | fp32 compute | COCO | off-distribution |
|---|---|---|---|---|
| 0 excluded | 6.78 MB | 0% | −0.100 dB | **−1.537 dB** |
| **1 excluded (shipped)** | **8.60 MB** | **2%** | **−0.011 dB** | **+0.139 dB** |
| 2 excluded | 9.04 MB | 4% | −0.017 dB | — |
| 3 excluded | 9.15 MB | 6% | −0.010 dB | — |

The excluded layer is the decoder's input convolution, which is large in
weights (2.4 MB) but runs at 30×40, so it is only 2% of the decoder's
compute — a good layer to spend on. Past the first exclusion the second
measures *worse* than the first on COCO, which is noise, and is the tell
that there is no signal left to chase.

**Mixed int8/fp16 does not work**, and would have been the obvious way
to make an excluded layer cheaper. `convert_float_to_float16` on an
already-quantised graph produces a model onnxruntime refuses to load
(`Type (tensor(float16)) ... does not match expected type
(tensor(float))` at the int8 boundary) and takes pathologically long
doing it, even with the quantisation ops in `op_block_list`. It would
have saved ~1.7 MB across the pair. Not worth graph surgery.

> **Not re-measured:** the on-air column (quantised TX → fp32 RX)
> originally read 24.79 / 24.79 / 24.63 dB. Only the local pipeline was
> re-run. The arithmetic now puts the tuned int8 encoder at 0.17 dB of
> effective SNR, so those figures are plausible again — but they were
> recorded against a different export and have not been checked.

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
  paid by every receiver. **fp16 remains the default** — it is free by
  every measure, so there is no reason to spend anything here. But the
  int8 encoder is now defensible where size really binds: 14.0 dB under
  the channel noise, 0.17 dB of effective SNR. Before the export was
  taught to leave its worst layer at fp32 it was 5.8 dB and ~1 dB, which
  was not defensible.

  This asymmetry is also what decides how the export tunes int8: the
  encoder gets the sensitivity search, the decoder does not.

#### int8 degrades faster off-distribution

This is fixed in the shipped artifacts, but the mechanism is worth
keeping because it nearly went unnoticed and it generalises.

The untuned int8 export cost **1.57 dB** on smooth synthetic probes
against **0.19 dB** on COCO photographs — eight times worse on pictures
the model is only mildly unfamiliar with.

The cause is not where the intuition points. Two wrong guesses,
both measured:

- **Not the encoder.** Its *latent* error was actually smaller on the
  off-distribution probes. And halving that error by tuning the encoder
  changed the off-distribution picture by −0.05 dB — nothing. Encoder
  quantisation was not what those pictures were paying for.
- **Not diffuse.** It was one layer. Excluding the decoder's input
  convolution took off-distribution from −1.54 dB to +0.14 dB, while
  moving COCO by only 0.09 dB.

So the sensitivity is concentrated in the **decoder**, and it is nearly
invisible on the training distribution. Two consequences:

1. **Tune against off-distribution content.** A search scored on
   photographs alone rates that layer as worth 0.09 dB and leaves it
   quantised, shipping a 1.5 dB regression to anyone sending a test
   card. This is now built into the export.
2. **Evaluate on both.** An earlier draft of this document concluded the
   opposite — "quantisation must be evaluated on real photographs" —
   from the same data. Photographs alone tell you what quantisation
   costs a *typical* picture and nothing about what it costs a *hard*
   one, and for this codec the second number was 8× the first.

### Making future revisions more quantisation-tolerant

**Nothing to do now** — a soft constraint to weigh when the training
recipe is next touched, not a reason to retrain. The export-side fix
below was tried first and largely solved the problem, so what remains
here is genuinely optional.

1. ~~**Static (calibrated) quantisation**, for the per-channel weight
   scales `QLinearConv` supports and `ConvInteger` does not.~~
   **Measured 2026-07-28, and it is worse:** 3.53e-01 latent RMS against
   dynamic's 1.88e-01, i.e. only 0.3 dB under the channel noise. Static
   quantisation also quantises *activations*, and this encoder ends in
   `tanh` followed by unit-RMS normalisation, which does not survive
   uint8 activations gracefully. Several configurations would not build
   at all (`Quantization parameter shared mode is not supported for
   weight yet`). The per-channel weights are real but they do not pay
   for the activations. **Do not retry this without a reason to think
   the activation problem has changed.**
2. **Inject weight noise during training.** Add small uniform noise to
   conv weights in the forward pass, sized like int8 quantisation error.
   This is the same trick the project already runs on: channel noise in
   the loop *is* the regulariser (`latent_channel.py`), and this is that
   idea one level down, on the weights. Cheap, no graph or export
   changes, and it does not commit anyone to a quantised deployment.
3. **Regularise the per-tensor dynamic range.** Penalise max-to-RMS (or
   kurtosis) of each conv weight tensor. This attacks the outlier
   mechanism from (1) directly, so it is the right partner for
   *dynamic* quantisation specifically.
4. **Broaden the training data.** Probably the most valuable item here,
   and it is not really about quantisation at all: the model is trained
   on COCO photographs, while operators send test cards, charts,
   screenshots and text. The quantisation work above only removed the
   *extra* penalty those pictures were paying; they are still
   reconstructed by a model that has never seen anything like them, at
   every precision including fp32. Non-photographic content in the
   training mix would improve them across the board.
5. **Full QAT** — fake-quant nodes, QDQ export. The heavyweight option.
   Probably not worth it for a codec whose quantisation noise is already
   supposed to sit under a channel that carries far more.

The measuring stick stays the one this document uses throughout: not
fp32, but `CHANNEL_NOISE_RMS`. The export fix already put int8's encoder
error at 14.0 dB under channel noise, so **there is no deficit left to
close** — anything here would be buying margin, not repairing a
regression.

### Do not chase precision below int8

Not measured, and not worth measuring for on-air use. With the tuned
export int8 is back to a comfortable 14.0 dB under the channel noise,
which is where the graceful-degradation argument holds; below int8 the
quantisation noise stops being comfortably under the channel noise and
that argument stops applying.

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
  quantisation settings could land well outside the 0.002 dB measured
  here. That is not hypothetical: a plain `quantize_dynamic` of this
  same model — the obvious thing for a third party to do — costs 1.57 dB
  on non-photographic pictures, and this document reached three
  different wrong int8 figures before measuring properly.
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

## How it was implemented

All done 2026-07-27. Recorded because the *shape* of the change is worth
keeping, not just the fact of it.

1. **`scripts/export_onnx.py`**, run at publish time from the `.pt` so
   the artifacts cannot drift from the checkpoint. Verifies every
   artifact against the torch model before writing and refuses to push
   if a tolerance is breached; stamps provenance (source checkpoint name
   and sha256) into each `.onnx`'s `metadata_props` and into a published
   manifest. Carried in `launch_job.sh`'s code snapshot.
2. **`sstvae/codec.py`** grew `OnnxCodec` (shipping) and `TorchCodec`
   (reference), behind `load_codec(path, precision=, backend="auto")`.
   `backend="auto"` sends a `.pt` to torch and everything else to ONNX,
   so `--model something.pt` keeps working untouched.
   - `reconstruct(codec, latents, weights)` **kept its exact signature**
     so `sstvae/rx/engine.py` — load-bearing, per CLAUDE.md — needed no
     edit at all. `pytest -m slow` passes unchanged.
   - **Parts load lazily and independently.** Encoding touches only the
     encoder; decoding and listening only the decoder. A receive-only
     station therefore fetches 9 MB, not the 21 MB pair, and never pays
     memory for a graph it will not run.
3. **`sstvae/latents.py`** — the numpy `latents_to_flat` /
   `flat_to_latents`. `tests/test_latents.py` asserts they equal the
   torch statics *exactly*; both are pure reshape and concatenate, so
   any tolerance would be hiding something. `sstvae/images.py` needed
   the same treatment — it returned torch tensors, which would have
   dragged torch back in through the send path regardless of the codec.
4. **`checkpoint.py`** gained `onnx_filename` / `default_onnx` /
   `resolve_onnx`, sharing one cache-first `_fetch` with the `.pt` path
   because the immutability argument is identical. `DEFAULT_REVISION` is
   derived from `DEFAULT_FILE`, so the artifact names cannot be bumped
   independently of the checkpoint.
   - `--model` had to grow up: the `.pt` is one file and the ONNX codec
     is two. It now accepts a `.pt`, a directory of exported artifacts,
     or a single `.onnx` (deriving the sibling part by name, and only
     when that part is actually needed).
5. **`tests/test_onnx_artifacts.py`** checks each artifact two ways:
   provenance (the stamped sha256 against the real `v1.pt` — exact, and
   catches a stale artifact even when the numbers look fine) and
   numerics (tight for fp32, channel-referenced for the quantised
   variants). Marked slow, and skips rather than downloading, so the
   fast suite stays ~20 s and network-free.
6. **`pyproject.toml`** — `cli`/`listen`/`gui` take onnxruntime instead
   of torch, which deleted three CPU-index pins and three `conflicts`
   entries. `dev` keeps CPU torch, because several tests `importorskip`
   it as the reference implementation and there is no CI to notice them
   silently vanishing.

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
