"""Reading and matching the callsign windows (spec 2.7, design 6.7, WP6).

Not a format module. Each window is 192 Morse units of two CE symbols;
units 0..7 and everything after the call are plain carrier, the call
sits in units 8..183. With FSK keying (the default) key-down is the
carrier one cycle per unit low (-16.495 Hz), with the same envelope;
with on-off keying it is the carrier switched on (units 5..183 follow
the keying, the rest stay on).

`window_soft` turns each unit of each window into one soft value, + for
key-down. For FSK it is the noncoherent tone comparison
(|X1|^2 - |X0|^2)/sigma^2, X0 the unit's samples summed at the carrier
and X1 at one cycle per unit below it (orthogonal over a unit); for OOK
(|X0|^2 - sigma^2 - E/2)/sigma^2 with E the carrier energy per unit
measured on the window's plain units. Noncoherent, because inside a
window the tracker has no measurement until the keying is known, and an
11.6 s gap is long next to a fading path's coherence time; the Morse
units are long enough (Eu/N0 = SNR2500 + 21.8 dB) that this costs little.
The frequency comes from the tracker's path, which runs straight through
the windows. `match` scores the expected keying of a known callsign
(from the header) on these: Z = sum s_u soft_u / sqrt(176 n) with
s = +-1, which is about N(0, 1) when the windows hold no signal.

`window_llr` gives each unit's exact log-likelihood ratio, key-down to
key-up, under the Rician model of the same sums: with a the unit's tone
amplitude,

    FSK:  ln I0(2 a |X1| / sigma^2) - ln I0(2 a |X0| / sigma^2)
    OOK:  ln I0(2 a |X0| / sigma^2) - a^2 / sigma^2

a^2 measured as the local mean of |X0|^2 + |X1|^2 - 2 sigma^2 over 15
units for FSK (every unit holds the tone in one of the two sums, so the
estimate does not depend on the keying) and on the window's plain units
for OOK. This is what `read` takes. The quadratic soft values above are
heavy-tailed on key-down units (|X|^2 grows with the signal), and a
reader fed them misreads a strong window and distrusts a correct one;
the LLR grows only like |X|, and sums over windows and passes are again
LLRs, so the reader's margin against the nearest rival call is a
log-likelihood ratio with no fitting.

Windows are identical in every window position and every pass, so soft
values (or LLRs) from all four positions and from many passes add before
`match` (or `read`).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .constants import CW_LEAD_UNITS, CW_ROOM_UNITS, CW_UNIT, CW_UNITS
from .frame import FrameSpec
from .morse import MORSE, check_callsign, keying_units
from .track import TrackResult, rx_index

_INV = {v: k for k, v in MORSE.items()}
CALL_UNITS = slice(CW_LEAD_UNITS, CW_LEAD_UNITS + CW_ROOM_UNITS)     # units 8..183
PLAIN_UNITS = np.r_[0:5, 184:CW_UNITS]                               # on in both keyings


def _unit_sums(tr: TrackResult, spec: FrameSpec, z=None):
    """(X0, X1, sigma2): (n_win, 192) sums of each unit at 0 Hz and at -1 cycle/unit.

    sigma2 is the noise variance of each sum (noise PSD x fs x samples).
    """
    z = tr.z_d if z is None else np.asarray(z)
    chan = tr.chan
    n_win = spec.n_win
    X0 = np.zeros((n_win, CW_UNITS), dtype=np.complex128)
    X1 = np.zeros((n_win, CW_UNITS), dtype=np.complex128)
    s2 = np.ones((n_win, CW_UNITS))
    for w, pw in enumerate(spec.win_start_pos):
        u = np.arange(CW_UNITS)
        ta = pw - 0.5 + CW_UNIT * u                       # sender time of unit start (symbols)
        na = chan.t0_index + rx_index(tr.timing, ta, spec.n_pos)
        nb = chan.t0_index + rx_index(tr.timing, ta + CW_UNIT, spec.n_pos)
        first = np.ceil(na).astype(np.int64)
        width = int(np.max(np.ceil(nb) - first)) + 1
        nn = first[:, None] + np.arange(width)[None, :]
        inside = (nn < nb[:, None]) & (nn >= 0) & (nn < len(z))
        x = (nn - na[:, None]) / (nb - na)[:, None]        # fraction of the unit
        v = np.where(inside, z[np.clip(nn, 0, len(z) - 1)], 0)
        X0[w] = np.sum(v, axis=1)
        X1[w] = np.sum(v * np.exp(2j * np.pi * x), axis=1)
        cnt = inside.sum(axis=1)
        t_mid = chan.t_s(0.5 * (na + nb))
        s2[w] = np.maximum(chan.noise_psd_at(t_mid) * chan.fs * cnt, 1e-300)
    return X0, X1, s2


def window_soft(ch, ch_fs, track: TrackResult, spec: FrameSpec, ook: bool = False
                ) -> np.ndarray:
    """float32[n_win, 176]: per Morse unit of the call region, + = key-down.

    `ch` is the derotated channel (None: the track's own z_d); `ch_fs` is
    the CH rate. Noise-only units give values of about unit variance.
    """
    if spec.n_win == 0:
        return np.zeros((0, CW_ROOM_UNITS), dtype=np.float32)
    X0, X1, s2 = _unit_sums(track, spec, ch)
    p0, p1 = np.abs(X0) ** 2, np.abs(X1) ** 2
    if not ook:
        soft = (p1 - p0) / (math.sqrt(2.0) * s2)
    else:
        E = np.maximum(np.mean(p0[:, PLAIN_UNITS] - s2[:, PLAIN_UNITS], axis=1), 0.0)
        soft = (p0 - s2 - E[:, None] / 2) / s2
    return soft[:, CALL_UNITS].astype(np.float32)


AMP_SMOOTH_UNITS = 15      # FSK: units over which the tone amplitude is averaged


def _ln_i0(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log(special.i0e(x)) + x


def window_llr(ch, ch_fs, track: TrackResult, spec: FrameSpec, ook: bool = False
               ) -> np.ndarray:
    """float32[n_win, 176]: each unit's log-likelihood ratio, key-down : key-up.

    The Rician model of the module docstring; noise-only units give
    values near 0 (their measured amplitude is near 0), not unit variance.
    """
    if spec.n_win == 0:
        return np.zeros((0, CW_ROOM_UNITS), dtype=np.float32)
    X0, X1, s2 = _unit_sums(track, spec, ch)
    p0, p1 = np.abs(X0) ** 2, np.abs(X1) ** 2
    if not ook:
        e = p0 + p1 - 2 * s2
        k = np.ones(AMP_SMOOTH_UNITS)
        num = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, e)
        cnt = np.convolve(np.ones(CW_UNITS), k, "same")
        a = np.sqrt(np.maximum(num / cnt[None, :], 0.0))
        llr = (_ln_i0(2 * a * np.sqrt(p1) / s2) - _ln_i0(2 * a * np.sqrt(p0) / s2))
    else:
        E = np.maximum(np.mean(p0[:, PLAIN_UNITS] - s2[:, PLAIN_UNITS], axis=1), 0.0)
        a = np.sqrt(E)[:, None]
        llr = _ln_i0(2 * a * np.sqrt(p0) / s2) - a * a / s2
    return llr[:, CALL_UNITS].astype(np.float32)


def _combine(soft) -> tuple[np.ndarray, int]:
    s = np.asarray(soft, dtype=np.float64)
    if s.ndim == 1:
        return s, 1
    return s.sum(axis=0), s.shape[0]


def decode_units(hard) -> str | None:
    """Run-length Morse decode of key-down flags for units 8..183, or None."""
    k = np.asarray(hard, dtype=bool)
    on = np.nonzero(k)[0]
    if len(on) == 0:
        return None
    k = k[:on[-1] + 1]
    runs = []
    i = 0
    while i < len(k):
        j = i
        while j < len(k) and k[j] == k[i]:
            j += 1
        runs.append((bool(k[i]), j - i))
        i = j
    if not runs[0][0]:
        runs = runs[1:]                                  # tolerate a late first element
    chars, cur = [], ""
    for val, n in runs:
        if val:
            if n <= 2:
                cur += "."
            elif n <= 5:
                cur += "-"
            else:
                return None
        else:
            if n <= 2:
                continue
            if n <= 6:
                chars.append(cur)
                cur = ""
            else:
                return None
    chars.append(cur)
    try:
        text = "".join(_INV[c] for c in chars)
    except KeyError:
        return None
    try:
        return check_callsign(text)
    except ValueError:
        return None


# A read is accepted when its log-likelihood ratio, against the nearest
# rival call and against no call at all, exceeds this (natural log: a wrong
# read beats the right one at odds of about e^-READ_LLR_MIN per rival).
READ_LLR_MIN = 6.0


def _patterns():
    chars = sorted(MORSE)
    pats = []
    for ch in chars:
        k = []
        for j, el in enumerate(MORSE[ch]):
            if j:
                k.append(0)
            k += [1] * (1 if el == "." else 3)
        pats.append(np.array(k, dtype=np.int64))
    return chars, pats


_CHARS, _PATS = _patterns()


def _dp(gains, lens, n, max_chars, forbid=None):
    """best[k, e], back: k characters, the last ending at unit e (see read_ml)."""
    G = np.stack(gains)                              # (n_chars, n + 1)
    if forbid is not None:
        G = G.copy()
        c, st = forbid
        G[c, st] = -np.inf
    # characters of one length end together: keep the best of each length
    ulen = np.unique(lens)
    Gm = np.full((len(ulen), n + 1), -np.inf)
    Ga = np.zeros((len(ulen), n + 1), dtype=np.int64)
    for i, L in enumerate(ulen):
        cs = np.nonzero(lens == L)[0]
        j = np.argmax(G[cs], axis=0)
        Ga[i] = cs[j]
        Gm[i] = G[cs, :][j, np.arange(n + 1)]
    best = np.full((max_chars + 1, n + 1), -np.inf)
    back = np.full((max_chars + 1, n + 1, 2), -1, dtype=np.int64)
    ok = ulen <= n
    best[1, ulen[ok]] = Gm[ok, 0]
    back[1, ulen[ok], 0] = Ga[ok, 0]
    back[1, ulen[ok], 1] = 0
    for k in range(1, max_chars):
        for e in np.nonzero(np.isfinite(best[k]))[0]:
            st = e + 3
            if st >= n:
                continue
            ends = st + ulen
            ok = ends <= n
            v = best[k, e] + Gm[ok, st]
            en = ends[ok]
            better = v > best[k + 1, en]
            best[k + 1, en[better]] = v[better]
            back[k + 1, en[better], 0] = Ga[ok, st][better]
            back[k + 1, en[better], 1] = e
    return best, back


def _trace(best, back):
    k, e = np.unravel_index(int(np.argmax(best)), best.shape)
    if not np.isfinite(best[k, e]):
        return None, -np.inf, []
    score = float(best[k, e])
    items = []
    while k > 0:
        c, pe = back[k, e]
        items.append((int(c), int(e - len(_PATS[c]))))
        k, e = k - 1, pe
    items.reverse()
    return "".join(_CHARS[c] for c, _ in items), score, items


def read_ml(llr, max_chars: int = 8, margin: bool = False):
    """(callsign, Z[, margin]): the call whose keying best fits per-unit LLRs.

    `llr` holds per-unit log-likelihood ratios, key-down : key-up
    (`window_llr`, summed or stacked over windows and passes). A dynamic
    programme over Morse characters (1 to max_chars, 3-unit gaps,
    starting at unit 8, key-up to the end) maximises S = sum s_u llr_u
    with s = +-1, which is twice the log-likelihood of the keying up to a
    constant: the maximum-likelihood call. Z = S / sqrt(176 n).

    With margin=True also the read's log-likelihood ratio against its
    nearest rival (the best-scoring call differing from it in some
    character), (S - S_rival)/2, capped by its ratio against no call at
    all (every unit key-up).
    """
    s, nrow = _combine(llr)
    n = len(s)
    total = float(s.sum())
    norm = math.sqrt(n * nrow)
    lens = np.array([len(p) for p in _PATS], dtype=np.int64)
    gains = []
    for pat in _PATS:
        L = len(pat)
        on = np.nonzero(pat)[0]
        g = np.full(n + 1, -np.inf)
        if L <= n:
            starts = np.arange(n - L + 1)
            g[:n - L + 1] = 2 * s[starts[:, None] + on[None, :]].sum(axis=1)
        gains.append(g)
    best, back = _dp(gains, lens, n, max_chars)
    text, score, items = _trace(best, back)
    if text is None:
        return (None, 0.0, 0.0) if margin else (None, 0.0)
    S = score - total                    # sum over on units minus sum over off units
    z = S / norm
    if not margin:
        return text, z
    m = score / 2                        # against every unit key-up: sum of llr over on units
    for it in items:
        b2, bk2 = _dp(gains, lens, n, max_chars, forbid=it)
        t2, score2, _ = _trace(b2, bk2)
        if t2 is None:
            continue
        m = min(m, (score - score2) / 2)
    return text, z, float(m)


def read(llr, ook: bool = False) -> str | None:
    """The callsign in the windows, or None.

    `llr`: per-unit LLRs from `window_llr`, summed (or stacked) over
    windows and passes. Maximum-likelihood over every sendable call
    (`read_ml`), accepted when its log-likelihood ratio against the
    nearest rival call and against no call exceeds READ_LLR_MIN. `ook` is
    accepted for symmetry with `window_llr`; the LLRs already carry the
    keying.
    """
    text, _, m = read_ml(llr, margin=True)
    if text is None or m < READ_LLR_MIN:
        return None
    return text


def match(soft, callsign: str) -> float:
    """Z of the expected +-1 keying of `callsign` against the soft values."""
    s, n = _combine(soft)
    k = keying_units(callsign)[CALL_UNITS].astype(np.float64) * 2 - 1
    return float(np.dot(k, s) / math.sqrt(len(k) * n))


def keying_type(ch, track: TrackResult, spec: FrameSpec, callsign: str | None = None) -> str:
    """"fsk" or "ook", whichever explains the windows better.

    With a callsign, the larger `match` Z; without, whether the call units
    hold energy one cycle per unit below the carrier (only FSK puts any there).
    """
    if spec.n_win == 0:
        return "fsk"
    if callsign is not None:
        zf = match(window_soft(ch, None, track, spec, ook=False), callsign)
        zo = match(window_soft(ch, None, track, spec, ook=True), callsign)
        return "fsk" if zf >= zo else "ook"
    X0, X1, s2 = _unit_sums(track, spec, ch)
    e = (np.abs(X1[:, CALL_UNITS]) ** 2 - s2[:, CALL_UNITS]) / s2[:, CALL_UNITS]
    z = float(np.sum(e) / math.sqrt(e.size))
    return "fsk" if z > 3.0 else "ook"
