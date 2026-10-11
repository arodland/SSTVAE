"""The L frame: stream layout, positions, callsign windows and timing
(spec section 2; docs/qrss/waveform-l.md).

A **format module**, L's counterpart of `frame.py`, kept apart from it so
that nothing CE sends or receives moves. The two frames are the same
frame in time: L sends two latents per complex symbol at half CE's
symbol rate, so its preamble (330 symbols), header (1,320), callsign
windows (192 symbols each, after 6,944, 14,175, 21,406 and 28,637
stream symbols) and end all fall exactly where CE's do, and every
L stream count is half of CE's.

Index spaces are `frame.py`'s: stream index s (no windows), on which the
references, the latent order, the scrambler and the precoder are
defined, and position p = pos(s) (windows included), with stream symbol
s centred at t0 + pos(s) * T_L.

- **Preamble:** 330 QPSK symbols, `sequences.preamble_l`.
- **References:** the stream symbols with j = s - n_pre >= 0 and
  j % 16 == 0 (83 in the header, 1,687 in the data), QPSK from
  `sequences.references_l` in time order, header references first.
- **Header:** the 1,237 non-reference header symbols carry the 2,474
  coded bits as QPSK, bits 2j and 2j+1 to I and Q of symbol j (spec 5).
  There is no spare: 1,237 x 2 = 2,474.
- **Data:** non-reference data symbol k carries precoded latents 2k (I)
  and 2k+1 (Q), scaled by 1/sqrt(2) (spec 2, SSTVAE's `slots_to_symbols`).

**Spread header** (spec 5.1 for CE; for L, docs/qrss/waveform-l.md). With
`hdr_rho` > 0 latent i is sent as (x_i + sqrt(rho) h_i) / sqrt(1 + rho),
h being CE's: the same whitener and the same coded bits, indexed by
latent, so the spread copy's soft bits from L and CE passes add. Unlike
CE, a linear signal has no room on top, so the copy is made by scaling
the latents down: a permanent 10 log10(1 + rho) = 0.23 dB.
"""

import functools
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    L_CW_AFTER_FULL,
    L_CW_WINDOW,
    L_N_HDR,
    L_N_PRE,
    L_T_SYM,
    N_HDR_BITS,
    REF_PERIOD,
    SPAN,
    SPREAD_RHO,
    WAVEFORM_L,
)
from .frame import _ro, spread_signs
from .sequences import preamble_l, qpsk, references_l

N_HDR_SLOTS = (L_N_HDR - (-(-L_N_HDR // REF_PERIOD)))     # 1237 header symbols with bits
assert 2 * N_HDR_SLOTS == N_HDR_BITS


def _refs_below(j: int) -> int:
    """References among j' in [0, j): j' % 16 == 0."""
    return -(-j // REF_PERIOD)


def _data_symbols(n_payload: int, j0: int) -> int:
    """Smallest D with n_payload non-references in j = j0 .. j0 + D - 1.

    L's header (1,320 symbols) is not a whole number of reference periods,
    so the data section does not start on a reference, unlike CE's
    (`frame._data_symbols`): the first data reference is at j = 1,328.
    """
    D = n_payload + n_payload // REF_PERIOD
    while D - (_refs_below(j0 + D) - _refs_below(j0)) < n_payload:
        D += 1
    while D > 0 and (D - 1) - (_refs_below(j0 + D - 1) - _refs_below(j0)) >= n_payload:
        D -= 1
    return D


@dataclass(frozen=True)
class LFrameSpec:
    """One L frame shape. L_FULL is the on-air frame; the rest are test frames.

    `n_data` counts latents, as in CE (a prefix of air order), and must be
    even: data symbol k carries latents 2k and 2k+1. `cw_after` lists the
    stream symbols sent before each callsign window.
    """
    name: str = "l-full"
    n_pre: int = L_N_PRE
    n_hdr: int = L_N_HDR                     # 0 (no header) or 1320
    n_data: int = 50600
    cw_after: tuple[int, ...] = L_CW_AFTER_FULL
    # Spread-header share, latent units. None: SPREAD_RHO with a header block,
    # 0 without (as CE; pending Jeff's call, docs/qrss/waveform-l.md).
    hdr_rho: float | None = None

    waveform = WAVEFORM_L                    # class attribute, not a field
    t_sym = L_T_SYM
    cw_window = L_CW_WINDOW

    def __post_init__(self):
        object.__setattr__(self, "cw_after", tuple(int(c) for c in self.cw_after))
        rho = (SPREAD_RHO if self.n_hdr else 0.0) if self.hdr_rho is None else self.hdr_rho
        object.__setattr__(self, "hdr_rho", float(rho))
        if not 0.0 <= self.hdr_rho < 1.0:
            raise ValueError(f"hdr_rho must be in [0, 1), not {self.hdr_rho}")
        if self.n_pre != L_N_PRE:
            raise ValueError(f"n_pre must be {L_N_PRE}, not {self.n_pre}")
        if self.n_hdr not in (0, L_N_HDR):
            raise ValueError(f"n_hdr must be 0 or {L_N_HDR}, not {self.n_hdr}")
        if self.n_data <= 0 or self.n_data % 2:
            raise ValueError(f"n_data must be positive and even, not {self.n_data}")
        lo, hi = self.n_pre + self.n_hdr, self.n_sym
        prev = None
        for c in self.cw_after:
            if not lo < c <= hi:
                raise ValueError(f"{self.name}: window after stream symbol {c} is "
                                 f"outside ({lo}, {hi}]")
            if prev is not None and c <= prev:
                raise ValueError(f"{self.name}: cw_after must be strictly increasing")
            prev = c

    @property
    def has_header(self) -> bool:
        return bool(self.n_hdr) or self.hdr_rho > 0

    @property
    def n_data_payload(self) -> int:
        """Data symbols that carry latents (25,300 for L_FULL)."""
        return self.n_data // 2

    @property
    def n_data_sym(self) -> int:
        """Data-section symbols including its references (26,987 for L_FULL)."""
        return _data_symbols(self.n_data_payload, self.n_hdr)

    @property
    def n_sym(self) -> int:
        """Stream symbols (28,637 for L_FULL)."""
        return self.n_pre + self.n_hdr + self.n_data_sym

    @property
    def n_win(self) -> int:
        return len(self.cw_after)

    @property
    def n_pos(self) -> int:
        """Positions, windows included (29,405 for L_FULL)."""
        return self.n_sym + L_CW_WINDOW * self.n_win

    @property
    def n_ref(self) -> int:
        """Reference symbols, header and data (1,770 for L_FULL)."""
        return -(-(self.n_sym - self.n_pre) // REF_PERIOD)

    @property
    def duration_s(self) -> float:
        return self.n_pos * L_T_SYM

    @property
    def win_start_pos(self) -> tuple[int, ...]:
        return tuple(c + L_CW_WINDOW * w for w, c in enumerate(self.cw_after))

    @property
    def keyed_end_pos(self) -> float:
        """End of keying in L symbols after t0 (as `FrameSpec.keyed_end_pos`)."""
        if self.cw_after and self.cw_after[-1] == self.n_sym:
            return self.n_pos - 0.5
        return self.pos_of(self.n_sym - 1) + SPAN

    @property
    def keyed_start_pos(self) -> float:
        return -float(SPAN)

    def pos_of(self, s):
        s_arr = np.asarray(s, dtype=np.int64)
        n_before = np.searchsorted(np.asarray(self.cw_after, dtype=np.int64),
                                   s_arr, side="right")
        out = s_arr + L_CW_WINDOW * n_before
        return int(out) if out.ndim == 0 else out

    def time_s(self, pos):
        return np.asarray(pos, dtype=np.float64) * L_T_SYM

    def window_spans_s(self) -> list[tuple[float, float]]:
        return [((p - 0.5) * L_T_SYM, (p + L_CW_WINDOW - 0.5) * L_T_SYM)
                for p in self.win_start_pos]


L_FULL = LFrameSpec()
L_MEDIUM = LFrameSpec("l-medium", n_data=16384, cw_after=(5000, 10388))
L_SHORT = LFrameSpec("l-short", n_data=4096, cw_after=(2500, 3834))
L_TINY = LFrameSpec("l-tiny", n_hdr=0, n_data=1024, cw_after=())
PRESETS = {s.name: s for s in (L_FULL, L_MEDIUM, L_SHORT, L_TINY)}


def get(name_or_spec) -> LFrameSpec:
    if isinstance(name_or_spec, LFrameSpec):
        return name_or_spec
    if name_or_spec in PRESETS:
        return PRESETS[name_or_spec]
    raise ValueError(f"unknown L frame {name_or_spec!r}; one of {', '.join(PRESETS)}")


@dataclass(frozen=True, eq=False)
class Layout:
    """Where every symbol class sits (int64 stream indices), as `frame.Layout`."""
    pre: np.ndarray
    hdr_ref: np.ndarray
    hdr_sym: np.ndarray                      # (1237,) or (0,): header symbol j carries bits 2j, 2j+1
    data_ref: np.ndarray
    data: np.ndarray                         # (n_data/2,): data symbol k carries latents 2k, 2k+1
    ref: np.ndarray
    known_idx: np.ndarray                    # preamble + all refs, ascending
    known_val: np.ndarray                    # complex QPSK at known_idx
    pos: np.ndarray
    win_pos: np.ndarray
    is_ref: np.ndarray = field(repr=False)


@functools.lru_cache(maxsize=16)
def layout(spec: LFrameSpec) -> Layout:
    n_sym, n_pre, n_hdr = spec.n_sym, spec.n_pre, spec.n_hdr
    s = np.arange(n_sym, dtype=np.int64)
    j = s - n_pre
    is_ref = (j >= 0) & (j % REF_PERIOD == 0)
    in_hdr = (s >= n_pre) & (s < n_pre + n_hdr)
    in_data = s >= n_pre + n_hdr
    hdr_sym = s[in_hdr & ~is_ref]
    if n_hdr and len(hdr_sym) != N_HDR_SLOTS:
        raise AssertionError(f"{len(hdr_sym)} header symbols for {N_HDR_BITS} bits")
    data = s[in_data & ~is_ref]
    if len(data) != spec.n_data_payload:
        raise AssertionError(f"{len(data)} data symbols for {spec.n_data} latents")
    ref = s[is_ref]
    pre = s[:n_pre]
    known_idx = np.concatenate([pre, ref])
    known_val = np.concatenate([preamble_l(), references_l(len(ref))])
    order = np.argsort(known_idx, kind="stable")
    win = np.array(spec.win_start_pos, dtype=np.int64).reshape(-1, 1)
    win_pos = np.hstack([win, win + L_CW_WINDOW - 1]) if len(win) else \
        np.zeros((0, 2), dtype=np.int64)
    return Layout(pre=_ro(pre), hdr_ref=_ro(s[in_hdr & is_ref]), hdr_sym=_ro(hdr_sym),
                  data_ref=_ro(s[in_data & is_ref]), data=_ro(data), ref=_ro(ref),
                  known_idx=_ro(known_idx[order]), known_val=_ro(known_val[order]),
                  pos=_ro(spec.pos_of(s)), win_pos=_ro(win_pos), is_ref=_ro(is_ref))


def pair(v) -> np.ndarray:
    """complex128[n/2]: values 2k and 2k+1 to I and Q of symbol k, scaled by 1/sqrt(2)."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or len(v) % 2:
        raise ValueError(f"pair takes an even number of values, not shape {v.shape}")
    return (v[0::2] + 1j * v[1::2]) / np.sqrt(2.0)


def unpair(z) -> np.ndarray:
    """float64[..., 2n]: the inverse of `pair` (sqrt(2) Re, sqrt(2) Im, interleaved)."""
    z = np.asarray(z)
    out = np.empty(z.shape[:-1] + (2 * z.shape[-1],), dtype=np.float64)
    out[..., 0::2] = np.sqrt(2.0) * z.real
    out[..., 1::2] = np.sqrt(2.0) * z.imag
    return out


def data_latents(spec: LFrameSpec, x, hdr_bits=None) -> np.ndarray:
    """float64[n_data]: what the data symbols carry, latent by latent: x, or
    (x + sqrt(rho) h) / sqrt(1 + rho) with the spread header."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (spec.n_data,):
        raise ValueError(f"x has shape {x.shape}, expected ({spec.n_data},)")
    if spec.hdr_rho == 0:
        return x
    rho = spec.hdr_rho
    return (x + np.sqrt(rho) * spread_signs(spec, hdr_bits)) / np.sqrt(1.0 + rho)


def assemble(spec: LFrameSpec, hdr_bits, x) -> np.ndarray:
    """complex128[n_sym]: the frame's symbols in stream order.

    `hdr_bits` (uint8 0/1, the 2,474 coded bits) is required when the frame
    has a header and must be None when it does not; `x` is the n_data
    precoded latents.
    """
    lay = layout(spec)
    out = np.zeros(spec.n_sym, dtype=np.complex128)
    out[lay.known_idx] = lay.known_val
    if spec.has_header:
        if hdr_bits is None:
            raise ValueError(f"frame {spec.name!r} has a header: hdr_bits is required")
        b = np.asarray(hdr_bits)
        if b.shape != (N_HDR_BITS,) or np.any((b != 0) & (b != 1)):
            raise ValueError(f"hdr_bits must be {N_HDR_BITS} values of 0/1")
        if spec.n_hdr:
            out[lay.hdr_sym] = qpsk(b)
    elif hdr_bits is not None and len(hdr_bits):
        raise ValueError(f"frame {spec.name!r} has no header but hdr_bits were given")
    out[lay.data] = pair(data_latents(spec, x, hdr_bits))
    return out


def disassemble(y_sym, spec: LFrameSpec) -> tuple[np.ndarray, np.ndarray]:
    """(header bit values (..., 2474), data latent values (..., n_data)) from
    stream-order symbol estimates, through `unpair`."""
    lay = layout(spec)
    y = np.asarray(y_sym)
    if y.shape[-1] != spec.n_sym:
        raise ValueError(f"y_sym has {y.shape[-1]} symbols, expected {spec.n_sym}")
    return unpair(y[..., lay.hdr_sym]), unpair(y[..., lay.data])
