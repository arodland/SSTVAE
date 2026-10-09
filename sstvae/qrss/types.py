"""Shared dataclasses passed between the receiver, the store and EM
(design 6.9).

Not a format module: nothing here goes on the air. `PassResult` is the
one record a single-pass receive produces and everything after it
(association, accumulators, leave-one-out EM) consumes, so its fields
and its on-disk form are the contract between work packages.

Orders and scales follow design section 7: z and w are in air order,
unscrambled and unprecoded; z is unbiased and w = 1/var; tracker arrays
are indexed by position.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # header.py belongs to another work package
    from .header import HeaderFields


@dataclass
class Timing:
    """Affine (+ optional quadratic) timing of one pass (design 6.4).

    Position p is received at tau0 + p*T*(1 + ppm*1e-6) + gamma*(p/n_pos)^2,
    in CH samples on the receiver's clock.
    """
    tau0: float
    ppm: float
    gamma: float
    z: float
    cov: np.ndarray = field(repr=False)


@dataclass
class FreqPath:
    """A frequency trajectory: f_hz at times t_s (relative to t0), with weights."""
    t_s: np.ndarray
    f_hz: np.ndarray
    weight: np.ndarray


@dataclass
class CarrierCandidate:
    f_hz: float
    drift_hz_per_min: float
    metric: float


@dataclass
class Detection:
    f_hz: float
    path: FreqPath
    timing: Timing | None
    z_ref: float
    method: str                              # "preamble" | "tbd" | "template"
    lead_in_s: float


@dataclass
class TrackReport:
    offset_hz: float
    drift_hz_per_min: float
    wander_hz_rms: float
    doppler_hz: float
    ppm: float
    snr2500_db: float
    z_ref: float
    kappa: float
    slip_free: bool
    suspect: bool


@dataclass
class CwIdResult:
    text: str | None
    z_match: float
    keying: str                              # "fsk" | "ook"
    agrees: bool | None


@dataclass
class EmPrior:
    """Soft data symbols for one pass, in that pass's precoded-symbol domain."""
    x_hat: np.ndarray                        # float64[n_data]
    v: np.ndarray                            # float64[n_data]


def pass_uid(q: int, f_hz: float, z) -> str:
    """f"{q}_{round(f_hz*1000)}_{sha8 of z bytes}" -- a pass's stable name."""
    z = np.ascontiguousarray(np.asarray(z, dtype=np.float32))
    sha8 = hashlib.sha256(z.tobytes()).hexdigest()[:8]
    return f"{int(q)}_{int(round(f_hz * 1000))}_{sha8}"


def _jsonable(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


@dataclass
class PassResult:
    """Everything one pass of one signal delivered."""
    uid: str
    q: int
    frame: str
    waveform: int                            # 0 = CE
    f_hz: float
    timing: Timing
    report: TrackReport
    z: np.ndarray = field(repr=False)        # float32[n_data], air order
    w: np.ndarray = field(repr=False)        # float32[n_data]
    hdr_llr: np.ndarray = field(repr=False)  # float32[2474]
    header: HeaderFields | None
    cw: CwIdResult | None
    ch: np.ndarray = field(repr=False)       # complex64 narrow capture, kept for EM
    ch_fs: int
    ch_t0_index: float
    f_mix_hz: Fraction
    psi: np.ndarray = field(repr=False)      # float32[n_pos]
    estimator: str
    em_round: int = 0

    _ARRAYS = ("z", "w", "hdr_llr", "ch", "psi")

    def save(self, path) -> None:
        """Write an .npz holding the arrays plus a JSON `meta` entry."""
        meta = {
            "uid": self.uid, "q": int(self.q), "frame": self.frame,
            "waveform": int(self.waveform), "f_hz": float(self.f_hz),
            "timing": {k: _jsonable(getattr(self.timing, k))
                       for k in ("tau0", "ppm", "gamma", "z")},
            "report": {k: _jsonable(v)
                       for k, v in dataclasses.asdict(self.report).items()},
            "header": (None if self.header is None else
                       {k: _jsonable(v) for k, v in dataclasses.asdict(self.header).items()}),
            "cw": (None if self.cw is None else
                   {k: _jsonable(v) for k, v in dataclasses.asdict(self.cw).items()}),
            "ch_fs": int(self.ch_fs), "ch_t0_index": float(self.ch_t0_index),
            "f_mix_hz": str(Fraction(self.f_mix_hz)),
            "estimator": self.estimator, "em_round": int(self.em_round),
        }
        arrays = {k: np.asarray(getattr(self, k)) for k in self._ARRAYS}
        arrays["timing_cov"] = np.asarray(self.timing.cov, dtype=np.float64)
        buf = io.BytesIO()
        np.savez(buf, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
                 **arrays)
        with open(path, "wb") as f:
            f.write(buf.getvalue())

    @classmethod
    def load(cls, path) -> PassResult:
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(d["meta"].tobytes().decode())
            arrays = {k: d[k] for k in cls._ARRAYS}
            cov = d["timing_cov"]
        header = None
        if meta["header"] is not None:
            from .header import HeaderFields
            header = HeaderFields(**meta["header"])
        return cls(
            uid=meta["uid"], q=meta["q"], frame=meta["frame"],
            waveform=meta["waveform"], f_hz=meta["f_hz"],
            timing=Timing(cov=cov, **meta["timing"]),
            report=TrackReport(**meta["report"]),
            header=header,
            cw=None if meta["cw"] is None else CwIdResult(**meta["cw"]),
            ch_fs=meta["ch_fs"], ch_t0_index=meta["ch_t0_index"],
            f_mix_hz=Fraction(meta["f_mix_hz"]),
            estimator=meta["estimator"], em_round=meta["em_round"],
            **arrays,
        )
