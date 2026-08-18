"""One reception, on disk and on the wire.

A `.npz` holding the latents and weights a receiver demodulated, plus
the metadata needed to say *which transmission* they belong to. It is
deliberately one format for two jobs -- a local archive and the body of
an upload to an aggregation server (docs/reception-aggregation.md) --
because they carry identical information and a second format would be a
second thing to keep in step.

**Why latents rather than a picture.** The decoder consumes
`latents x weights` and returns pixels, which discards the per-latent
confidence that says how much each value should be trusted. Two
stations that each heard a transmission badly can be combined into one
good picture (`sstvae.modem.diversity`), but only in the latent domain
and only while those weights survive. A PNG is the one thing that
cannot be combined.

**Why numpy and stdlib only.** Both the receiving station and the
server import this, and the server has no audio, no codec and no torch.
Keeping the format's dependencies to numpy also keeps it trivially
writable from the C++ implementation later, which is a live intention
rather than a courtesy.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np

from .. import __version__
from ..checkpoint import DEFAULT_REVISION
from ..config import MODES, PROTOCOL_VERSION
from ..modem.modem import BlindDemodResult, DemodResult

__all__ = ["to_bytes", "write", "read", "FORMAT", "FORMAT_VERSION", "ReceptionFileError"]

FORMAT = "sstvae-reception"
FORMAT_VERSION = 1


class ReceptionFileError(ValueError):
    """A payload that cannot be read as a reception."""


def _iso(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z"
    )


def _parse_iso(text: str) -> float:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError) as exc:
        raise ReceptionFileError(f"bad utc_start {text!r}") from exc


def to_bytes(
    result: DemodResult | BlindDemodResult,
    *,
    utc_start: float,
    station_callsign: str,
    dial_freq_hz: float | None = None,
) -> bytes:
    """Serialize one demodulator output plus its identifying metadata.

    `result` is what `Reception.result` carries. Arrays are stored as
    float32: latents are clipped to +-10 and weights live in [0, 1], so
    float32 is far finer than the channel, and the whole payload is
    ~1.3 MB before compression for a transmission lasting 32-95 s. The
    npz is self-describing, so a future writer may narrow this without
    breaking any reader -- that would be a `format_version` bump, not a
    format break.
    """
    if isinstance(result, DemodResult):
        meta = {
            "path": "header",
            "mode_name": result.mode.name,
            "frames_received": int(result.frames_received),
        }
    elif isinstance(result, BlindDemodResult):
        meta = {
            "path": "blind",
            "frame_offset": (
                None if result.frame_offset is None else int(result.frame_offset)
            ),
            "n_frames": int(result.n_frames),
        }
    else:
        raise ReceptionFileError(
            f"expected a DemodResult or BlindDemodResult, got {type(result).__name__}"
        )

    meta.update(
        {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "codec_revision": DEFAULT_REVISION,
            "software": f"sstvae {__version__}",
            "snr_db": float(result.snr_db),
            "freq_offset": float(result.freq_offset),
            "callsign": str(result.callsign or ""),
            "station_callsign": str(station_callsign or "").strip().upper(),
            # Recorded and displayed, never used to decide which
            # transmission this is -- see the matching rule in
            # docs/reception-aggregation.md.
            "dial_freq_hz": None if dial_freq_hz is None else float(dial_freq_hz),
            "utc_start": _iso(utc_start),
        }
    )

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        latents=np.asarray(result.latents, dtype=np.float32),
        weights=np.asarray(result.weights, dtype=np.float32),
        meta=np.array(json.dumps(meta, sort_keys=True)),
    )
    return buf.getvalue()


def write(path, result, **kwargs) -> None:
    """`to_bytes` straight to a file."""
    with open(path, "wb") as fh:
        fh.write(to_bytes(result, **kwargs))


def read(data) -> tuple[DemodResult | BlindDemodResult, dict]:
    """Inverse of `to_bytes`: the demodulator output and its metadata.

    The returned object is a *real* `DemodResult`/`BlindDemodResult`,
    not a look-alike, because `diversity.combine_diversity_results`
    dispatches on the type and rejects anything else.

    Fields that only mean something to the receiver that produced them
    -- `sync_metric`, `preamble_start`, `frame0_start`, `beacon` -- are
    filled with neutral values rather than carried. They are positions
    in *that station's* audio buffer, meaningless anywhere else, and
    combining reads none of them.
    """
    if isinstance(data, (bytes, bytearray)):
        data = io.BytesIO(data)
    try:
        with np.load(data, allow_pickle=False) as npz:
            latents = np.asarray(npz["latents"], dtype=np.float64)
            weights = np.asarray(npz["weights"], dtype=np.float64)
            meta = json.loads(str(npz["meta"]))
    except ReceptionFileError:
        raise
    except Exception as exc:  # malformed zip, missing key, bad JSON
        raise ReceptionFileError(f"unreadable reception file: {exc}") from exc

    if meta.get("format") != FORMAT:
        raise ReceptionFileError(f"not a {FORMAT} file: {meta.get('format')!r}")
    if meta.get("format_version") != FORMAT_VERSION:
        raise ReceptionFileError(
            f"format_version {meta.get('format_version')!r}, expected {FORMAT_VERSION}"
        )
    if meta.get("protocol_version") != PROTOCOL_VERSION:
        # A different waveform: the latents would be combined with ones
        # that do not describe the same thing.
        raise ReceptionFileError(
            f"protocol_version {meta.get('protocol_version')!r}, "
            f"this build speaks {PROTOCOL_VERSION}"
        )
    if len(latents) != len(weights):
        raise ReceptionFileError(
            f"latents ({len(latents)}) and weights ({len(weights)}) differ in length"
        )

    meta["utc_start_epoch"] = _parse_iso(meta.get("utc_start"))
    path = meta.get("path")

    if path == "header":
        spec = MODES.get(meta.get("mode_name"))
        if spec is None:
            raise ReceptionFileError(f"unknown mode {meta.get('mode_name')!r}")
        if len(latents) != spec.n_latents:
            raise ReceptionFileError(
                f"mode {spec.name} wants {spec.n_latents} latents, got {len(latents)}"
            )
        return (
            DemodResult(
                latents=latents,
                weights=weights,
                mode=spec,
                freq_offset=float(meta.get("freq_offset", 0.0)),
                sync_metric=0.0,
                frames_received=int(meta.get("frames_received", 0)),
                beacon=None,
                callsign=str(meta.get("callsign", "")),
                preamble_start=0,
                snr_db=float(meta.get("snr_db", float("nan"))),
            ),
            meta,
        )

    if path == "blind":
        full = MODES["C"].n_latents
        if len(latents) != full:
            raise ReceptionFileError(
                f"a blind reception is sized for mode C ({full} latents), "
                f"got {len(latents)}"
            )
        frame_offset = meta.get("frame_offset")
        return (
            BlindDemodResult(
                latents=latents,
                weights=weights,
                freq_offset=float(meta.get("freq_offset", 0.0)),
                beacon=None,
                callsign=str(meta.get("callsign", "")),
                frame_offset=None if frame_offset is None else int(frame_offset),
                n_frames=int(meta.get("n_frames", 0)),
                frame0_start=None,
                snr_db=float(meta.get("snr_db", float("nan"))),
            ),
            meta,
        )

    raise ReceptionFileError(f"unknown path {path!r}")
