# SSTVAE

A hybrid-digital SSTV mode for HF radio, inspired by [FreeDV RADE](https://freedv.org/radio-autoencoder/):
instead of analog scanlines or digital bits, images are sent as the
continuous-valued latents of a convolutional autoencoder, modulated
directly onto OFDM carrier amplitudes. The decoder network is trained
with the channel in the loop, so quality degrades gracefully with SNR
instead of failing hard.

## Waveform

| property | value |
|---|---|
| audio | 8 kHz mono, SSB-compatible |
| carriers | 24 × 50 Hz spacing, 950–2100 Hz |
| occupied bandwidth | ~1200 Hz (99% power) |
| symbol | 20 ms + 4 ms cyclic prefix |
| frame | 1 pilot + 5 data symbols (144 ms), 240 latents |
| sync | periodic preamble + Golay(24,12) BPSK header |
| freq offset tolerance | > ±50 Hz |
| envelope PAPR | ~6.7 dB (clip-and-filter; NN-shaped in stage 2) |

Images are transmitted at **640×480** (any input ≥320×240 is accepted
and upscaled; decode with `--size 320x240` for a classic-SSTV-sized
output). Modes: **A** ≈ 32 s (1 latent group, coarse), **B** ≈ 64 s
(2 groups), **C** ≈ 95 s (3 groups, best fidelity). Faster modes are literal
prefixes of mode C's stream, so a mode C reception can be decoded
progressively as it arrives, and truncated or faded receptions decode
at reduced fidelity.

## Usage

```sh
# encode an image to transmit audio
python sstvae_encode.py photo.jpg tx.wav --mode B --model runs/s1/checkpoint.pt

# simulate an HF channel (AWGN / Watterson fading / offsets)
python sstvae_simulate.py tx.wav rx.wav --snr 3 --fading mpp --freq-offset 43

# decode received audio (with progressive snapshots)
python sstvae_decode.py rx.wav out.png --model runs/s1/checkpoint.pt --snapshots 4
```

## Training

```sh
# pipeline smoke test (tiny model, synthetic data, any torch device)
python scripts/train.py --smoke --out runs/smoke

# stage 1, local folder:
python scripts/train.py --data /path/to/images --epochs 60 --out runs/s1

# stage 1, dataset from the Hub (with validation split + PSNR eval):
python scripts/train.py --hf-dataset arodland/coco320-sstvae \
    --push-to-hub arodland/sstvae-s1 --epochs 60 --out runs/s1
```

The training dataset (COCO 2017 at 320x240, proper train/val splits) is
`arodland/coco320-sstvae` on the Hub; rebuild it with
`scripts/build_hf_dataset.py`.

### Cloud / remote training

`scripts/train_job.py` is a self-contained uv script: it pulls the code
snapshot and dataset from the Hub, trains, and pushes checkpoints back
every epoch (interruption-safe; resume with `SSTVAE_RESUME=1`).

```sh
scripts/launch_job.sh l4x1 --epochs 60 --batch 48   # HF Jobs
HF_TOKEN=... uv run scripts/train_job.py            # any GPU box
```

Stage 2 fine-tunes through a differentiable replica of the real modem
chain (`sstvae/waveform_channel.py`): OFDM synthesis, envelope
clip-and-filter, Watterson fading, noisy-pilot equalization, burst
erasures — with a RADE-style continuous linear-ratio PAPR penalty in
the loss (see scripts/train.py's --papr-weight help for the rationale):

```sh
python scripts/train.py --hf-dataset arodland/coco320-sstvae \
    --stage2 --resume runs/s1/checkpoint.pt --lr 1e-4 \
    --papr-weight 0.002 --out runs/s2
```

## Development

```sh
pip install -e .[dev]        # numpy/scipy/pillow + pytest
pip install -e .[train]      # + torch, lpips, datasets
pytest                       # modem unit + end-to-end tests
```

Project status: modem and stage-1 training pipeline complete and tested
over simulated channels; on-air calibration not yet started.
