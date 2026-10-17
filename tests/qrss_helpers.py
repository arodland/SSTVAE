"""Shared helpers for the QRSSTVAE tests (`tests/test_qrss_*.py`).

Not a test module (no `test_` prefix), so pytest only imports it. Kept
free of anything heavy at import time: the codec helpers import
onnxruntime lazily and skip cleanly when the models are absent.

- `Q_TEST`: the canonical slot, 2025-10-01T00:00Z.
- `unit_rms_latents`: synthetic air-order latents for tests that do not
  need the codec.
- `latent_snr_db`: delivered per-latent SNR, the unit every loss in the
  verification plan is measured in.
- `model_dir`, `codec_skip`, `require_codec_models`: the codec-test
  resolution and skip rules of design 0 rule 6 -- $QRSSTVAE_MODEL_DIR,
  then /home/user/qrss-data/models, then the HF cache; never download;
  skip, unless SSTVAE_REQUIRE_CODEC is set, in which case fail.
- `mmse_gauss`, `s_eff`: the fading-tracking model of the spec's
  `sims/tracking.py`, ported for R8.
"""

import os
from pathlib import Path

import numpy as np
import pytest

Q_TEST = 1954752                     # floor(unix(2025-10-01T00:00Z) / 900)

DEFAULT_MODEL_DIR = "/home/user/qrss-data/models"
COCO_VAL_PARQUET = ("/home/user/qrss-data/data/data/"
                    "validation-00000-of-00002.parquet")
REPO_ROOT = Path(__file__).resolve().parent.parent


# --- synthetic latents --------------------------------------------------------

def unit_rms_latents(n: int = 50600, seed: int = 0, dist: str = "gauss") -> np.ndarray:
    """float64[n] with RMS exactly 1.

    "gauss" is plain Gaussian; "tanh" mimics the encoder's bounded
    latents (a tanh of a wider Gaussian), which has the heavier
    shoulders that matter for phase-modulation distortion.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    if dist == "tanh":
        a = np.tanh(1.5 * a)
    elif dist != "gauss":
        raise ValueError(f"unknown dist {dist!r}")
    return a / np.sqrt(np.mean(a ** 2))


def synthetic_full_latents(seed: int = 0) -> np.ndarray:
    """float64[158400]: a stand-in for the encoder's flat canonical output."""
    from sstvae import config
    n = config.LATENT_GROUPS * config.GROUP_LATENTS
    rng = np.random.default_rng(seed)
    return np.tanh(1.2 * rng.standard_normal(n)) * 1.1


def latent_snr_db(est, truth, w=None) -> float:
    """10 log10( mean(truth^2) / mean((est - truth)^2) ), optionally where w > 0."""
    est = np.asarray(est, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if w is not None:
        keep = np.asarray(w) > 0
        est, truth = est[keep], truth[keep]
    err = np.mean((est - truth) ** 2)
    return float(10 * np.log10(np.mean(truth ** 2) / err)) if err > 0 else float("inf")


# --- codec resolution and skips -------------------------------------------------

def codec_skip(reason: str):
    """Skip, unless SSTVAE_REQUIRE_CODEC says these tests must run."""
    if os.environ.get("SSTVAE_REQUIRE_CODEC"):
        pytest.fail(f"SSTVAE_REQUIRE_CODEC is set but codec tests cannot run: {reason}")
    pytest.skip(reason)


def model_dir() -> str | None:
    """Directory holding the v5 ONNX models, or None for the HF cache.

    $QRSSTVAE_MODEL_DIR, then DEFAULT_MODEL_DIR if it holds .onnx files,
    else None (meaning: let `checkpoint.resolve_onnx` look in the cache).
    """
    env = os.environ.get("QRSSTVAE_MODEL_DIR")
    if env:
        return env
    p = Path(DEFAULT_MODEL_DIR)
    if p.is_dir() and any(p.glob("*.onnx")):
        return str(p)
    return None


def resolve_model(part: str = "encoder", precision: str = "fp16") -> str:
    """Path to one ONNX part, or a codec skip. Never downloads.

    With no model directory, only an *already cached* published artifact
    is accepted (`checkpoint._any_cached_revision`), mirroring
    `test_native_parity._codec_artifacts`.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        codec_skip("onnxruntime is not installed")
    from sstvae import checkpoint

    d = model_dir()
    try:
        if d is not None:
            return checkpoint.resolve_onnx(part, d, precision)
        found = checkpoint._any_cached_revision(checkpoint.onnx_filename(part, precision))
        if found is None:
            codec_skip(f"no cached {part}-{precision} model and no model directory")
        return found
    except (Exception, SystemExit) as e:
        # SystemExit: resolve_onnx raises it for a missing artifact.
        codec_skip(f"codec model unavailable: {e}")


def load_codec(precision: str = "fp16"):
    """An `sstvae.codec` codec from the resolved models, or a codec skip."""
    path = resolve_model("encoder", precision)
    from sstvae import codec
    try:
        return codec.load_codec(str(Path(path).parent), precision=precision)
    except (Exception, SystemExit) as e:
        codec_skip(f"cannot load codec: {e}")


def require_coco():
    """Path to the COCO640 validation parquet, or a skip."""
    if not Path(COCO_VAL_PARQUET).exists():
        pytest.skip(f"COCO validation parquet not present at {COCO_VAL_PARQUET}")
    return COCO_VAL_PARQUET


# --- tracking model (spec sims/tracking.py) ----------------------------------------

def mmse_gauss(c: float, B: float) -> float:
    """Smoother MMSE for a unit-power Rayleigh tap with a Gaussian Doppler
    PSD of 2-sigma spread B (Hz), at reference C/N0 = c (Hz)."""
    sig = B / 2
    f = np.linspace(-8 * sig, 8 * sig, 4001)
    S = np.exp(-f ** 2 / (2 * sig ** 2)) / (np.sqrt(2 * np.pi) * sig)
    return float(np.trapezoid(S / (1 + c * S), f))


def s_eff(S: float, B: float, f_r: float, f_d: float, G: float, rho: float,
          mm=mmse_gauss) -> float:
    """Effective per-latent SNR after tracking error (sims/tracking.py).

    S: SNR in 2500 Hz (linear); B: Doppler spread; f_r, f_d: reference
    and data symbol rates feeding the tracker; G: per-latent SNR gain
    over S; rho: data-aided share of the data symbols.
    """
    c = 2500 * S * (f_r + f_d * rho)
    e = mm(c, B)
    s1 = G * S
    return s1 * (1 - e) / (1 + s1 * e)
