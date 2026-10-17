"""Stored pictures: segment preparation, the fp16 store, picture ID,
codec ID, int8 beacon values, air <-> canonical maps and `.qrsp` files
(design 2.5, spec sections 3-4).

A **format module**. The transmitter encodes a picture *once*, stores the
fp16 tensor (`StoredPicture`, a `.qrsp` file) and never re-encodes for a
repeat: SSTVAE's fp16 and fp32 encoders differ by ~4.6e-4 RMS and latent
optimization is not repeatable, so a re-encode would send a different
picture under the same ID.

Orders:

- **canonical**: SSTVAE's flat latent order (`sstvae.latents`), group g
  at [g*52800, (g+1)*52800), each group flattened (channel, h, w) C-order.
- **air**: one group's 50,600 sent latents in the order of SSTVAE's
  frozen interleaver, `framing._TX_PERMS[g]` (offsets within the group).
  The other 2,200 canonical latents per group are never sent.

`framing._TX_PERMS` is private to SSTVAE; it is wrapped once, here.
"""

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sstvae import config as _c
from sstvae.modem import framing as _framing

from .constants import INT8_SCALE

GROUP_LATENTS = _c.GROUP_LATENTS                      # 52,800
SENT = _c.TRANSMIT_LATENTS_PER_GROUP                  # 50,600
N_GROUPS = _c.LATENT_GROUPS                           # 3
MODE_NAMES = ("A", "B", "C")                          # mode m sends groups 0..m
QRSP_VERSION = 1


def _check_group(group: int) -> int:
    g = int(group)
    if not 0 <= g < N_GROUPS:
        raise ValueError(f"group must be 0..{N_GROUPS - 1}, not {group}")
    return g


def _check_mode(mode: int) -> int:
    m = int(mode)
    if not 0 <= m < N_GROUPS:
        raise ValueError(f"mode must be 0..{N_GROUPS - 1} (A..C), not {mode}")
    return m


def air_to_canonical(group: int) -> np.ndarray:
    """intp[50600]: canonical offset within the group of each air-order latent."""
    out = _framing._TX_PERMS[_check_group(group)].astype(np.intp, copy=True)
    out.setflags(write=False)
    return out


def never_sent(group: int) -> np.ndarray:
    """bool[52800]: True for the group's 2,200 canonical latents never sent."""
    mask = np.ones(GROUP_LATENTS, dtype=bool)
    mask[air_to_canonical(group)] = False
    return mask


def prepare_segment(full, group: int) -> np.ndarray:
    """float16[52800]: one group's canonical latents, ready to store and send.

    `full` is the encoder's flat canonical vector (158,400 values, any
    shape of that size). The never-sent 2,200 are zeroed, the 50,600 sent
    values are scaled to RMS 1 (in float64), and the result is rounded to
    float16 -- **the** explicit unit-RMS step of spec 3. The fp16 values
    are what is hashed and what is sent.
    """
    g = _check_group(group)
    flat = np.asarray(full, dtype=np.float64).reshape(-1)
    if flat.size != N_GROUPS * GROUP_LATENTS:
        raise ValueError(
            f"expected {N_GROUPS * GROUP_LATENTS} latents, got {flat.size}")
    seg = flat[g * GROUP_LATENTS:(g + 1) * GROUP_LATENTS].copy()
    seg[never_sent(g)] = 0.0
    rms = np.sqrt(np.mean(seg[air_to_canonical(g)] ** 2))
    if not np.isfinite(rms) or rms == 0.0:
        raise ValueError(f"group {g} has no energy to normalise (rms {rms})")
    return (seg / rms).astype(np.float16)


def air_values(seg, group: int) -> np.ndarray:
    """float64[50600]: a stored segment's sent latents in air order."""
    seg = np.asarray(seg)
    if seg.shape[-1] != GROUP_LATENTS:
        raise ValueError(f"segment has {seg.shape[-1]} latents, expected {GROUP_LATENTS}")
    return seg[..., air_to_canonical(group)].astype(np.float64)


def canonical_from_air(air, group: int) -> np.ndarray:
    """(..., 52800) canonical from (..., 50600) air order; never-sent = 0.

    Keeps a floating input's dtype (float32 z and w stay float32).
    """
    air = np.asarray(air)
    if air.shape[-1] != SENT:
        raise ValueError(f"air vector has {air.shape[-1]} values, expected {SENT}")
    dtype = air.dtype if np.issubdtype(air.dtype, np.floating) else np.float64
    out = np.zeros(air.shape[:-1] + (GROUP_LATENTS,), dtype=dtype)
    out[..., air_to_canonical(group)] = air
    return out


def picture_id(codec_id: int, mode: int, segs) -> int:
    """First 32 bits (big-endian) of SHA-256 over the stored tensor (D7).

    Bytes: uint16_be(codec_id) || uint8(mode) || each of the mode's
    groups 0..mode as float16 little-endian, canonical 52,800 each.
    """
    m = _check_mode(mode)
    if not 0 <= int(codec_id) < 1 << 16:
        raise ValueError(f"codec_id must fit in 16 bits, not {codec_id}")
    segs = list(segs)
    if len(segs) != m + 1:
        raise ValueError(f"mode {MODE_NAMES[m]} stores {m + 1} segments, got {len(segs)}")
    h = hashlib.sha256(struct.pack(">HB", int(codec_id), m))
    for s in segs:
        s = np.asarray(s)
        if s.shape != (GROUP_LATENTS,):
            raise ValueError(f"segment shape {s.shape}, expected ({GROUP_LATENTS},)")
        h.update(s.astype("<f2").tobytes())
    return int.from_bytes(h.digest()[:4], "big")


def codec_id_from_onnx(path_or_dir: str, precision: str = "fp16") -> int | None:
    """The codec ID: first 2 bytes of the artifact's `sstvae.source_sha256` (D11).

    `path_or_dir` is an `.onnx` file or a directory of them (the encoder
    is preferred, then the decoder: both carry the same checkpoint hash).
    Returns None when the metadata is absent, so the caller can insist
    on an explicit `--codec-id`. v5 gives 0xD1D8.
    """
    import onnxruntime

    p = Path(path_or_dir)
    if p.is_dir():
        from sstvae import checkpoint
        for part in ("encoder", "decoder"):
            try:
                p = Path(checkpoint.resolve_onnx(part, str(p), precision))
                break
            except SystemExit:
                continue
        else:
            raise FileNotFoundError(f"no *-encoder-{precision}.onnx or "
                                    f"*-decoder-{precision}.onnx in {path_or_dir}")
    sess = onnxruntime.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    sha = meta.get("sstvae.source_sha256")
    if not sha or len(sha) < 4:
        return None
    try:
        return int(sha[:4], 16)
    except ValueError:
        return None


def to_int8(air) -> np.ndarray:
    """Beacon storage: clip(rint(20*a), -127, 127) as int8 (round half even)."""
    q = np.rint(np.asarray(air, dtype=np.float64) * INT8_SCALE)
    return np.clip(q, -127, 127).astype(np.int8)


def from_int8(q) -> np.ndarray:
    """q / 20 as float64."""
    return np.asarray(q, dtype=np.float64) / INT8_SCALE


@dataclass
class StoredPicture:
    """A picture encoded once: what every pass of it sends."""
    mode: int
    codec_id: int
    segs: list = field(repr=False)           # float16[52800] per group 0..mode
    picture_id: int

    @classmethod
    def from_latents(cls, full, codec_id: int, mode: int) -> "StoredPicture":
        """Prepare the mode's segments from the encoder's flat latents."""
        m = _check_mode(mode)
        segs = [prepare_segment(full, g) for g in range(m + 1)]
        return cls(m, int(codec_id), segs, picture_id(codec_id, m, segs))

    def check(self) -> None:
        """ValueError unless the stored ID matches the stored tensor."""
        pid = picture_id(self.codec_id, self.mode, self.segs)
        if pid != self.picture_id:
            raise ValueError(
                f"picture ID {self.picture_id:08x} does not match the stored "
                f"latents ({pid:08x})")

    @property
    def id_hex(self) -> str:
        return f"{self.picture_id:08x}"


def save_qrsp(path, sp: StoredPicture) -> None:
    """Write a `.qrsp` file (an .npz: version, mode, codec_id, segs f2, picture_id)."""
    sp.check()
    segs = np.stack([np.asarray(s, dtype=np.float16) for s in sp.segs])
    # A file handle, so numpy does not append ".npz" to the name.
    with open(path, "wb") as f:
        np.savez(f, version=np.int64(QRSP_VERSION), mode=np.int64(sp.mode),
                 codec_id=np.int64(sp.codec_id), segs=segs,
                 picture_id=np.int64(sp.picture_id))


def load_qrsp(path) -> StoredPicture:
    """Read a `.qrsp` file; ValueError if it is malformed or its ID is wrong."""
    with np.load(path, allow_pickle=False) as z:
        try:
            version = int(z["version"])
            mode, codec_id = int(z["mode"]), int(z["codec_id"])
            segs = z["segs"]
            pid = int(z["picture_id"])
        except KeyError as e:
            raise ValueError(f"{path}: not a .qrsp file (missing {e})") from None
    if version != QRSP_VERSION:
        raise ValueError(f"{path}: .qrsp version {version}, expected {QRSP_VERSION}")
    if segs.dtype != np.float16 or segs.shape != (mode + 1, GROUP_LATENTS):
        raise ValueError(f"{path}: segs {segs.dtype}{segs.shape} do not fit mode {mode}")
    sp = StoredPicture(_check_mode(mode), codec_id, [s.copy() for s in segs], pid)
    sp.check()
    return sp
