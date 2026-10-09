"""The CE frame: stream layout, positions, callsign windows and timing
(design section 2.2).

A **format module**, and the one place frame lengths live. Everything
time-like elsewhere (slot length, window times, keyed span) must be
derived from a `FrameSpec` -- the slot timing is still an open choice
(the 1782.7 s frame with callsign windows, or a faster picture keeping
1736 s), and keeping it here makes that choice one edit.

Two index spaces:

- **Stream index** s counts the frame's symbols without the callsign
  windows: preamble [0, n_pre), header [n_pre, n_pre + n_hdr), data.
  The reference pattern, latent order, scrambler and precoder are all
  defined on it, so a window only delays what follows it.
- **Position** p counts symbols *with* the windows and is proportional
  to time: stream symbol s is centred at t0 + pos(s)*T, where
  t0 = QH + 1 s.

Reference symbols are the stream symbols with j = s - n_pre >= 0 and
j % 16 == 0 (decision D3). Header non-reference symbol i carries coded
bit i for i < 2474; the 2475th is the spare and sends a known +1
(decision D4). Data non-reference symbol i carries precoded value x_i.
"""

import functools
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    CW_AFTER_FULL,
    CW_WINDOW,
    N_HDR,
    N_HDR_BITS,
    N_PRE,
    REF_PERIOD,
    SPAN,
    T_SYM,
)
from .sequences import preamble_ce, references_ce


def _data_symbols(n_data: int) -> int:
    """Smallest L with L - ceil(L/16) == n_data (data refs at local j%16==0).

    L - ceil(L/16) never exceeds 15L/16 and grows by 0 or 1 per step, so
    counting up from n_data + n_data//16 finds the smallest L.
    """
    L = n_data + n_data // REF_PERIOD
    while L - (-(-L // REF_PERIOD)) < n_data:
        L += 1
    return L


@dataclass(frozen=True)
class FrameSpec:
    """One frame shape. FULL is the on-air frame; the rest are test frames.

    `cw_after` lists, for each callsign window, the number of stream
    symbols sent before it; a window equal to n_sym ends the frame.
    """
    name: str = "full"
    n_pre: int = N_PRE
    n_hdr: int = N_HDR                       # 0 (no header) or 2640
    n_data: int = 50600                      # data latents (a prefix of air order)
    cw_after: tuple[int, ...] = CW_AFTER_FULL

    def __post_init__(self):
        object.__setattr__(self, "cw_after", tuple(int(c) for c in self.cw_after))
        if self.n_pre != N_PRE:
            # The preamble is a fixed sequence; a different length would
            # be a different waveform, not a different frame.
            raise ValueError(f"n_pre must be {N_PRE}, not {self.n_pre}")
        if self.n_hdr not in (0, N_HDR):
            raise ValueError(f"n_hdr must be 0 or {N_HDR}, not {self.n_hdr}")
        if self.n_data <= 0:
            raise ValueError(f"n_data must be positive, not {self.n_data}")
        lo, hi = self.n_pre + self.n_hdr, self.n_sym
        prev = None
        for c in self.cw_after:
            if not lo < c <= hi:
                raise ValueError(
                    f"{self.name}: window after stream symbol {c} is outside "
                    f"({lo}, {hi}]")
            if prev is not None and c <= prev:
                raise ValueError(f"{self.name}: cw_after must be strictly increasing")
            prev = c

    # --- derived lengths ---------------------------------------------------
    @property
    def n_data_sym(self) -> int:
        """Data-section symbols including its references (53,974 for FULL)."""
        return _data_symbols(self.n_data)

    @property
    def n_sym(self) -> int:
        """Stream symbols (57,274 for FULL)."""
        return self.n_pre + self.n_hdr + self.n_data_sym

    @property
    def n_win(self) -> int:
        return len(self.cw_after)

    @property
    def n_pos(self) -> int:
        """Positions, i.e. symbols including the windows (58,810 for FULL)."""
        return self.n_sym + CW_WINDOW * self.n_win

    @property
    def n_ref(self) -> int:
        """Reference symbols, header and data (3,539 for FULL)."""
        return -(-(self.n_sym - self.n_pre) // REF_PERIOD)

    @property
    def duration_s(self) -> float:
        """n_pos * T: from t0 - T/2 to the end of the last position."""
        return self.n_pos * T_SYM

    @property
    def win_start_pos(self) -> tuple[int, ...]:
        """P_w = cw_after[w] + 384*w, first position of each window."""
        return tuple(c + CW_WINDOW * w for w, c in enumerate(self.cw_after))

    @property
    def keyed_end_pos(self) -> float:
        """End of keying in symbols after t0 (sender's clock).

        n_pos - 1/2 when the last window ends the frame, else
        pos(n_sym - 1) + SPAN so the last pulse tail is sent whole.
        """
        if self.cw_after and self.cw_after[-1] == self.n_sym:
            return self.n_pos - 0.5
        return self.pos_of(self.n_sym - 1) + SPAN

    @property
    def keyed_start_pos(self) -> float:
        """Latest start of keying, in symbols after t0 (-SPAN = -8)."""
        return -float(SPAN)

    def pos_of(self, s):
        """pos(s) = s + 384 * |{w : cw_after[w] <= s}| (scalar or array)."""
        s_arr = np.asarray(s, dtype=np.int64)
        n_before = np.searchsorted(np.asarray(self.cw_after, dtype=np.int64),
                                   s_arr, side="right")
        out = s_arr + CW_WINDOW * n_before
        return int(out) if out.ndim == 0 else out

    def time_s(self, pos) -> np.ndarray | float:
        """Seconds after t0 at which position `pos` is centred."""
        return np.asarray(pos, dtype=np.float64) * T_SYM

    def window_spans_s(self) -> list[tuple[float, float]]:
        """[start, end) of each window in seconds after t0."""
        return [((p - 0.5) * T_SYM, (p + CW_WINDOW - 0.5) * T_SYM)
                for p in self.win_start_pos]


FULL = FrameSpec()
MEDIUM = FrameSpec("medium", n_data=16384, cw_after=(10000, 20777))
SHORT = FrameSpec("short", n_data=4096, cw_after=(5000, 7670))
TINY = FrameSpec("tiny", n_hdr=0, n_data=1024, cw_after=())
PRESETS = {s.name: s for s in (FULL, MEDIUM, SHORT, TINY)}


def get(name_or_spec) -> FrameSpec:
    """A preset by name, or a FrameSpec passed through."""
    if isinstance(name_or_spec, FrameSpec):
        return name_or_spec
    try:
        return PRESETS[name_or_spec]
    except KeyError:
        raise ValueError(
            f"unknown frame {name_or_spec!r}; one of {', '.join(PRESETS)}") from None


def _ro(a) -> np.ndarray:
    a = np.asarray(a)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class Layout:
    """Where every symbol class sits. int64 stream indices unless named pos."""
    pre: np.ndarray
    hdr_ref: np.ndarray
    hdr_bits: np.ndarray                     # (2474,) or (0,) without a header
    hdr_spare: np.ndarray                    # (1,) or (0,)
    data_ref: np.ndarray
    data: np.ndarray                         # (n_data,)
    ref: np.ndarray                          # hdr_ref ++ data_ref (reference counter m)
    known_idx: np.ndarray                    # preamble + all refs + spare, ascending
    known_val: np.ndarray                    # int8 +-1 at known_idx
    pos: np.ndarray                          # (n_sym,) position of each stream symbol
    win_pos: np.ndarray                      # (n_win, 2) first and last position of each window
    is_ref: np.ndarray = field(repr=False)   # (n_sym,) bool


@functools.lru_cache(maxsize=16)
def layout(spec: FrameSpec) -> Layout:
    """The symbol layout of `spec`. Cached; arrays are read-only."""
    n_sym, n_pre, n_hdr = spec.n_sym, spec.n_pre, spec.n_hdr
    s = np.arange(n_sym, dtype=np.int64)
    j = s - n_pre
    is_ref = (j >= 0) & (j % REF_PERIOD == 0)
    in_hdr = (s >= n_pre) & (s < n_pre + n_hdr)
    in_data = s >= n_pre + n_hdr

    pre = s[:n_pre]
    hdr_ref = s[in_hdr & is_ref]
    hdr_nonref = s[in_hdr & ~is_ref]
    if n_hdr:
        if len(hdr_nonref) != N_HDR_BITS + 1:
            raise AssertionError(f"{len(hdr_nonref)} header slots for {N_HDR_BITS} bits")
        hdr_bits, hdr_spare = hdr_nonref[:N_HDR_BITS], hdr_nonref[N_HDR_BITS:]
    else:
        hdr_bits = hdr_spare = np.zeros(0, dtype=np.int64)
    data_ref = s[in_data & is_ref]
    data = s[in_data & ~is_ref]
    if len(data) != spec.n_data:
        raise AssertionError(f"{len(data)} data slots for {spec.n_data} latents")
    ref = s[is_ref]                              # time order: header refs first

    known_idx = np.concatenate([pre, ref, hdr_spare])
    known_val = np.concatenate([
        preamble_ce(), references_ce(len(ref)), np.ones(len(hdr_spare), np.int8)
    ]).astype(np.int8)
    order = np.argsort(known_idx, kind="stable")

    win = np.array(spec.win_start_pos, dtype=np.int64).reshape(-1, 1)
    win_pos = np.hstack([win, win + CW_WINDOW - 1]) if len(win) else \
        np.zeros((0, 2), dtype=np.int64)

    return Layout(
        pre=_ro(pre), hdr_ref=_ro(hdr_ref), hdr_bits=_ro(hdr_bits),
        hdr_spare=_ro(hdr_spare), data_ref=_ro(data_ref), data=_ro(data),
        ref=_ro(ref), known_idx=_ro(known_idx[order]),
        known_val=_ro(known_val[order]), pos=_ro(spec.pos_of(s)),
        win_pos=_ro(win_pos), is_ref=_ro(is_ref),
    )


def assemble(spec: FrameSpec, hdr_bits, x) -> np.ndarray:
    """float64[n_sym]: the frame's symbol values in stream order.

    `hdr_bits` (uint8 0/1, 2474 coded bits, bit 0 -> +1) is required when
    the frame has a header and must be None when it does not; `x` is the
    n_data precoded data values.
    """
    lay = layout(spec)
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (spec.n_data,):
        raise ValueError(f"x has shape {x.shape}, expected ({spec.n_data},)")
    out = np.zeros(spec.n_sym, dtype=np.float64)
    out[lay.known_idx] = lay.known_val
    if spec.n_hdr:
        if hdr_bits is None:
            raise ValueError(f"frame {spec.name!r} has a header: hdr_bits is required")
        b = np.asarray(hdr_bits)
        if b.shape != (N_HDR_BITS,) or np.any((b != 0) & (b != 1)):
            raise ValueError(f"hdr_bits must be {N_HDR_BITS} values of 0/1")
        out[lay.hdr_bits] = 1.0 - 2.0 * b
    elif hdr_bits is not None and len(hdr_bits):
        raise ValueError(f"frame {spec.name!r} has no header but hdr_bits were given")
    out[lay.data] = x
    return out


def disassemble(y_sym, spec: FrameSpec) -> tuple[np.ndarray, np.ndarray]:
    """(header-symbol values (..., 2474), data values (..., n_data)) from stream order."""
    lay = layout(spec)
    y = np.asarray(y_sym)
    if y.shape[-1] != spec.n_sym:
        raise ValueError(f"y_sym has {y.shape[-1]} symbols, expected {spec.n_sym}")
    return y[..., lay.hdr_bits], y[..., lay.data]
