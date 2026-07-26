# SSTVAE

**Send pictures over HF radio using a neural image codec instead of scanlines or file bytes.**

SSTVAE is an experimental amateur-radio image mode. A convolutional
autoencoder compresses your image into a few hundred thousand
*continuous-valued* numbers ("latents"), and those numbers are sent
directly as OFDM carrier amplitudes — no bits, no packets, no JPEG. The
decoder network is trained with a simulated HF channel in the loop, so
noise and fading push the picture quality gently downhill instead of
breaking it.

It is directly inspired by [FreeDV RADE](https://freedv.org/radio-autoencoder/)
(the "radio autoencoder"), which does the same trick for speech.

![Sent and received images over a 3150 km path to Utah](docs/images/ota-20m-irises.png)

![Sent and received images over a 1700 km path to Minnesota](docs/images/ota-20m-airplane.png)

**Actually over the air.** Mode B (64 s) on 20 m, transmitted from New
Jersey. Top row received at the
[Northern Utah WebSDR](http://www.sdrutah.org/) in Corinne, Utah —
**3150 km**. Bottom row received in Minnesota — **1700 km**. In both,
the middle panel is 100 W and the right panel is **10 W**, sent back to
back over the same path. One tenth the power costs about 1 dB of
picture: it gets slightly softer rather than breaking up, which is the
whole idea.

> ### ⚠️ Status: working beta
>
> It works on real HF, as above. But:
>
> - **The on-air format is not frozen.** Expect incompatible changes.
>   Two stations must run the same commit *and* the same model
>   checkpoint to talk to each other.
> - **It is not yet user-friendly.** Command-line tools, no rig control,
>   no packaged installer, no integration with existing SSTV software.
> - Not registered with, or coordinated with, any band-plan authority.
>   Use it thoughtfully and identify per your licence.
>
> If you want something that just works and interoperates with the rest
> of the world today, use MMSSTV or an existing digital SSTV mode. If
> you want to help shake out a new one, read on.

## What makes it different

**Analog SSTV** fails softly — noise looks like noise — but QRM tears a
band across the image, lost sync slants the frame, and tuning in late
means the top of the picture is simply gone.

**HamDRM** and **SSDV** send a compressed file. While the FEC holds, the
picture is pixel-perfect; when it doesn't, you lose whole blocks or the
image outright. That's the digital cliff: excellent, excellent,
excellent, nothing.

SSTVAE aims at the middle. Latents are real numbers, so channel noise
adds to them instead of corrupting a bitstream — there's no threshold to
fall off. And a fixed interleaver scatters every frame's latents across
the whole picture, so losing part of a transmission dims detail
*everywhere* slightly rather than removing a region.

![SSTVAE mode B and analog Scottie 2 received over the same path](docs/images/ota-vs-analog.png)

The same picture between the same two stations, on the same frequency,
100 W both, a few minutes apart — and comparable airtime, 64 s against
71 s. The analog copy shows what the channel did to it: impulse noise as
coloured streaks, speckle through the sky, softened detail. The SSTVAE
copy absorbed the same channel into a slightly softer picture.

One comparison on one path at one moment isn't a controlled study —
propagation shifts minute to minute. And Scottie 2 is 320×256 *by
design*; it isn't failing, it's doing its job at its own resolution.

| | Analog SSTV | HamDRM | SSDV | SSTVAE |
|---|---|---|---|---|
| Payload | Scanlines as tones | Compressed file + FEC | JPEG in FEC'd packets | Autoencoder latents |
| Degradation | Gradual, but localized damage | Digital cliff | Lost packets = missing blocks | Gradual, spread over whole image |
| Missed the start? | Top of image lost | Usually unrecoverable | Depends on transport | **Full image**, via beacon resync |
| Bandwidth | ~1200 Hz, most modes | 2200 or 2500 Hz | Depends on transport | ~1200 Hz |
| Time on air | 36–120 s, common modes | ~30–180 s, varies with size and FEC | Depends on transport and size | 32 / 64 / 95 s |
| Needs a trained model | No | No | No | **Yes — both ends** |
| Exactness | Faithful-but-noisy | Exact or nothing | Exact, or blocks missing | Reconstruction, never exact |

SSDV is a packet format, not a mode — its bandwidth and timing depend
entirely on how you carry it.

SSTVAE targets the same airtime and bandwidth as the common analog
modes. The bandwidth saving is only against HamDRM, and frame size isn't
a differentiator either: PD120 already sends 640×496 in 126 s. The claim
is picture quality for comparable time on air.

Two caveats on the last two rows:

**You need the model.** The network *is* the codec, so both stations
need the same ~40 MB checkpoint — an interoperability cost analog SSTV
doesn't have.

**It's a reconstruction, not a photograph.** The decoder produces the
most plausible image consistent with the latents that arrived. Fine
detail, small text especially, can come back subtly wrong rather than
merely blurry — and it looks confident either way. Judge it by how the
pictures look, not the pixel count, and don't use it where exactness
matters.

## Performance

Measured end-to-end (encode → modem → simulated channel → modem →
decode) on 6 validation images never seen in training, at 640×480.
PSNR in dB; higher is better.

**AWGN** (SNR in a 3 kHz noise bandwidth):

| Mode | Time | clean | 20 dB | 10 dB | 6 dB | 3 dB | 0 dB | −2 dB |
|---|---|---|---|---|---|---|---|---|
| A | 32 s | 24.7 | 24.7 | 24.4 | 24.0 | 23.4 | 22.3 | 21.4 |
| B | 64 s | 25.7 | 25.6 | 25.4 | 25.0 | 24.4 | 23.5 | — |
| C | 95 s | 26.0 | 26.0 | 25.8 | 25.5 | 25.0 | 24.1 | 22.8 |

**Watterson "poor" multipath** (`mpp`: 2 ms second-path delay, 1 Hz Doppler spread):

| Mode | 20 dB | 10 dB | 6 dB |
|---|---|---|---|
| A | 23.7 | 23.3 | 22.8 |
| B | 25.2 | 24.7 | 24.2 |
| C | 25.6 | 25.3 | 24.8 |

Every point above acquired sync and received **100% of frames**. Mode C
gives up only 1.9 dB of PSNR across a 20 dB drop in channel SNR — that
gentle slope is the entire point of the design.

The on-air results at the top of this page land where the table says
they should. For the irises, the 100 W pass scored 25.5 dB — between the
mode B simulation's 6 and 10 dB SNR rows — and the 10 W pass 24.6 dB,
near its 3 dB row. Dropping 10 dB of power moved the picture by 0.9 dB
on that path, and by 1.0 dB on the shorter one.

Compare like with like, though: absolute PSNR depends heavily on the
picture. The airplane shot scores 28.2 / 27.2 dB, above every number in
the table, because a smooth sky and distant fields are simply easier to
code than iris petals — not because that path was better than the
simulation. Only the *differences* between passes of the same image are
meaningful across rows.

**Where it actually breaks:** the limit is acquisition, not image
quality. Below roughly −2 dB (AWGN) the preamble stops being detectable
and you get no image at all rather than a poor one. Fading pulls that
threshold up: mode C under `mpp` failed to acquire at −2 dB. So the
cliff hasn't been abolished — it has been moved off the picture and onto
the sync problem, which is a much better place for it.

Acquisition is also probabilistic once fading is involved, because it
depends on the preamble not landing in a deep fade. At 6 dB SNR with a
43 Hz frequency offset, preamble sync succeeded on 6–8 of 8 random
channel realizations across all three fading presets — an individual
over can miss even when conditions are nominally fine. The beacon's
mid-stream resync is the safety net here: it gets a second chance every
~10 s for the whole length of the transmission, rather than one chance
at the start.

Other measured properties:

- **Envelope PAPR ≈ 4.5 dB** for a real transmission. This matters
  because amateur transmitters are peak-power-limited: every dB of PAPR
  is a dB of average power you don't get to transmit. The network is
  explicitly trained to keep its latents' crest factor low.
- **Frequency offset tolerance > ±50 Hz** — no need to zero-beat.
- **Mid-stream join.** Start listening 60 seconds into a transmission
  and you still get the *whole* picture, including the part that was
  sent before you tuned in (see below).

## How it works

One of the 24 carriers is permanently reserved for a **beacon
side-channel**: a continuously repeating, Golay-coded packet carrying an
absolute frame counter and an optional 8-character callsign. Because
that counter is absolute rather than relative, a receiver that never
heard the transmission's start can recover its exact position in the
stream from any ~10 s window — and then decode frames it recorded
*before* it locked on. This costs about 4.2% of latent capacity but no
airtime.

The three modes are **nested**: the latents are ordered into 3 groups,
and modes A/B/C send 1/2/3 of them. Mode A's transmission is literally a
prefix of mode C's. A mode C reception can therefore be decoded
progressively as it arrives, and a truncated or partly-faded reception
simply decodes at lower fidelity.

Slower modes are also more robust in a way that isn't obvious: with more
total frames on air, there are more chances for the beacon's ~10 s
resync window to land somewhere clean enough to lock, even if the
dedicated preamble was lost. Mode C actually has a minimal quality
increase over mode B, but it gives a better chance of a picture getting
through at all on a poor channel.

### Waveform

| property | value |
|---|---|
| audio | 8 kHz mono, SSB-compatible |
| carriers | 24 × 50 Hz spacing, 950–2100 Hz |
| occupied bandwidth | ~1200 Hz (99% power) |
| symbol | 20 ms + 4 ms cyclic prefix |
| frame | 1 pilot + 5 data symbols (144 ms), 230 latents + 5 beacon chips |
| sync | preamble + Golay(24,12) BPSK header, plus continuous beacon |
| freq offset tolerance | > ±50 Hz |
| envelope PAPR | ~4.5 dB |
| image size | 640×480 (inputs ≥320×240 accepted, upscaled) |

## Install

You need **Python 3.10+**. [uv](https://docs.astral.sh/uv/) is
recommended but not required.

Get the code and a model checkpoint:

```sh
git clone https://github.com/arodland/SSTVAE     # or your fork/source
cd SSTVAE
```

The tools fetch the published model on first use and cache it, so
there's nothing else to download.

### Command-line tools (encode / decode / simulate)

This is the smaller install: PyTorch runs the codec, but CPU-only is
fine — decoding a picture takes a second or two.

```sh
uv sync --extra cli                      # with uv
pip install -e '.[cli]'                  # or plain pip, ideally in a venv
```

With `uv`, prefix commands with `uv run`; with pip, activate your venv
and call `python` directly. Examples below use `uv run`.

<details>
<summary>Platform notes</summary>

- **Linux** — nothing extra. `pip install -e '.[cli]'` pulls a CPU or
  CUDA build of torch depending on your platform; either works.
- **macOS** (Intel or Apple Silicon) — nothing extra. Torch uses the CPU
  or MPS automatically.
- **Windows** — works in PowerShell or WSL. In PowerShell, quote the
  extras differently: `pip install -e ".[cli]"`.
- **AMD GPU / ROCm** — the default wheels are CPU/CUDA. You do not need
  a GPU for encode/decode; only training benefits.

</details>

### Live receiver (listen to a soundcard)

Adds PortAudio bindings and a preview window on top of the above.

```sh
uv sync --extra listen
pip install -e '.[listen]'
```

<details>
<summary>PortAudio is a system dependency — install it first</summary>

- **Debian/Ubuntu** — `sudo apt install libportaudio2`
- **Fedora** — `sudo dnf install portaudio`
- **Arch** — `sudo pacman -S portaudio`
- **macOS** — `brew install portaudio`
- **Windows** — nothing to do; the `sounddevice` wheel bundles PortAudio.

If `sstvae_listen.py --list-devices` raises an OSError about PortAudio,
that library is missing or not on the loader path.

</details>

Route receiver audio to the computer however you normally would for
digital modes — a rig interface, a VAC/VB-Cable loopback, `pulse`/
PipeWire monitor sources, or just a microphone near the speaker.

## Usage

### Transmit

```sh
# image -> transmit audio (any PIL-readable format; --callsign optional)
uv run sstvae_encode.py photo.jpg tx.wav --mode B --callsign N0CALL
```

Play `tx.wav` into your transmitter at a level that doesn't trip ALC.
Modes: **A** (32 s, coarse), **B** (64 s), **C** (95 s, best).

### Receive from a recording

```sh
uv run sstvae_decode.py rx.wav out.png
uv run sstvae_decode.py rx.wav out.png --snapshots 4   # progressive
```

It prints the recovered beacon frame position and callsign alongside the
image. It does not need the recording to start at the beginning of the
transmission.

### Receive live

```sh
uv run sstvae_listen.py --list-devices   # find your input
uv run sstvae_listen.py --device 3       # GUI preview
uv run sstvae_listen.py --no-gui         # headless
uv run sstvae_listen.py --low-cpu        # small machines
```

Received images land in `received/` (`--out-dir`). The listener keeps a
rolling buffer and decodes continuously, so it picks up transmissions
already in progress and reconstructs them in full. `--low-cpu` trades
that mid-stream capability for much lower idle CPU — useful on a Pi.

### Test it without a radio

The channel simulator lets you check the whole path end to end:

```sh
uv run sstvae_encode.py photo.jpg tx.wav --mode C
uv run sstvae_simulate.py tx.wav rx.wav --snr 10 --fading mpp --freq-offset 43
uv run sstvae_decode.py rx.wav out.png
```

`--fading` takes `mpg` (good), `mpp` (poor), or `mpd` (disturbed) —
the standard Watterson presets. Also `--ppm` for sample-clock error,
`--zero-span` to blank intervals, and `--seed` to pick a different
random channel realization.

Push it harder and sync will eventually fail rather than the picture
getting worse — that's the acquisition threshold described above, and
it's the expected behaviour, not a bug. If a particular combination
won't decode, try another `--seed` before concluding anything: with
fading enabled, whether the preamble lands in a fade is luck.

### Using a different model

All three tools take `--model /path/to/checkpoint.pt` to override the
published default — for testing a checkpoint you trained yourself, or
one someone else published. Both stations have to be running the same
one: latents from a different network are meaningless to this decoder,
and there's no handshake to detect the mismatch, so a wrong checkpoint
decodes to noise rather than an error.

## Training

Most people won't need this; a checkpoint is published and the tools
fetch it automatically. If you do:

```sh
uv sync --extra train

# pipeline smoke test (tiny model, synthetic data, any torch device)
uv run scripts/train.py --smoke --out runs/smoke

# stage 1 — autoencoder with a simple latent-domain channel
uv run scripts/train.py --hf-dataset arodland/coco640-sstvae \
    --epochs 60 --out runs/s1

# stage 2 — fine-tune through a differentiable replica of the real modem
uv run scripts/train.py --hf-dataset arodland/coco640-sstvae \
    --stage2 --resume runs/s1/checkpoint.pt --lr 1e-4 \
    --papr-weight 0.002 --out runs/s2
```

Stage 2 (`sstvae/waveform_channel.py`) puts OFDM synthesis, envelope
clip-and-filter, Watterson fading, noisy-pilot equalization and burst
erasures in the training loop, with a RADE-style PAPR penalty in the
loss. That's where the low crest factor and the graceful SNR curve come
from.

`scripts/train_job.py` is a self-contained uv script for remote GPUs; it
pulls code and data from the Hub and pushes checkpoints back every epoch
(interruption-safe, `SSTVAE_RESUME=1` to resume).

## Development

```sh
uv sync --extra dev --extra cli
uv run pytest                 # fast suite, ~10 s
uv run pytest -m slow         # + multi-minute listener/end-to-end tests
```

`sstvae/config.py` holds every constant shared between the modem,
channel simulator and training code — all waveform and latent numbers
must agree through that module. `CLAUDE.md` has an architecture tour and
a list of non-obvious gotchas.

## Credits

The radio-autoencoder concept, the two-stage training recipe and the
PAPR-penalty approach all come from **FreeDV RADE** by David Rowe and
the FreeDV team. SSTVAE applies those ideas to still images; any
mistakes in the translation are mine.

The autoencoder is trained on **[COCO](https://cocodataset.org)**
(Common Objects in Context), Lin *et al.*, 2014 — resized to 640×480 as
`arodland/coco640-sstvae` on the Hugging Face Hub.

The photographs used on this page are my own.

## License

[Artistic License 2.0](LICENSE).
