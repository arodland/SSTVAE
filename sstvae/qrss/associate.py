"""Associating a pass with a picture: association rules 1 to 4
(design 8.2, spec section 9).

A pass is attached to an accumulator by the first rule that succeeds:

1. **header** -- the pass's own header decoded: (callsign, picture ID).
   Validated by correlation when the accumulator's segment is already
   good (median W > -10 dB): a pass whose latents do not correlate
   (t < 3 while t > 8 was expected) is a picture-ID collision or a
   re-encoded picture under an old ID; it is logged and held
   provisionally instead.
2. **soft-header** -- the pass's header LLRs summed with each
   provisional cluster's, then pairwise with every other provisional
   pass within +-10 Hz and 7 days, and decoded. The header is identical
   in every pass, so N passes' LLRs decode like one pass at N times the
   SNR. Every pass involved is attached, after rule 1's check.
3. **corr** -- latent correlation against every open accumulator (each
   segment with data); the highest accepted t wins.
4. **prov** -- otherwise the pass joins the provisional cluster it
   correlates with best, or starts a new one. When a cluster later
   correlates with an accumulator, or its summed header decodes, its
   passes are attached (`resolve_provisional`).

**The statistic** (D23). With z, w the pass's latents and weights and
m = S/W, W the reference's, over the M latents where both weights are
positive,

    v = w W / (1 + w + W),   t = sum v z m / sqrt(sum v^2 z^2 m^2).

Under H0 (independent latents) t ~ N(0, 1); it is the spec's 6/sqrt(M)
test in self-normalised form. For the same picture (unit-RMS latents,
honest weights) its mean is `expected_t` = sum v / sqrt(sum (v + k v^2))
with k = E[a^4] ~ 3, which is what P6's "accepted by M ~ 2,500"
follows from.

**Acceptance: t > 6 and t >= RHO_MIN * expected_t** (`corr_accept`).
The second half is not in the spec and is the fallback P7 called for
(see the deviation note in `tests/test_qrss_multipass.py`): v5 latents
of *different* COCO pictures are not independent. Noiseless pairs of 40
pictures reach |t| = 17 (std 3.2 instead of 1), from genuine content
similarity -- the correlation coefficient between two pictures' latents
reaches ~0.13 -- and subtracting the per-channel latent mean (the
design's fallback) leaves it at 17. Because t under H0 scales like
r * expected_t while the same picture gives expected_t itself, a pass
must also reach half its expected value. That costs nothing at the low
SNRs association is for (there 6 is the binding half) and removes the
high-SNR false matches entirely.
"""

from __future__ import annotations

import numpy as np

from . import header as _header
from .constants import Z_ACCEPT
from .store import DAY_S, SENT, SLOT_S, Store, canonical_index
from .types import PassResult

T_ACCEPT = Z_ACCEPT              # 6: the single detection/association gate
RHO_MIN = 0.5                    # t must also reach half its same-picture expectation
KURTOSIS = 3.0                   # E[a^4] of unit-RMS latents (v5 measures 3.09)
VALIDATE_MEDIAN_W = 0.1          # rule 1 validates only above -10 dB median W
HDR_REJECT_T = 3.0               # rule 1: reject at t < 3 ...
HDR_REJECT_EXPECT = 8.0          # ... while more than 8 was expected
PAIR_HZ = 10.0                   # rule 2 pairs: within +-10 Hz ...
PAIR_DAYS = 7.0                  # ... and 7 days


def _pair(z, w, m, W):
    z = np.asarray(z, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n = min(z.size, m.size)
    z, w, m, W = z[:n], w[:n], m[:n], W[:n]
    keep = (w > 0) & (W > 0) & np.isfinite(z) & np.isfinite(m) \
        & np.isfinite(w) & np.isfinite(W)
    w, W = w[keep], W[keep]
    return z[keep], w, m[keep], W, w * W / (1 + w + W)


def corr_stat(z, w, m, W) -> tuple[float, int]:
    """(t, M): the self-normalised correlation of a pass with a reference.

    z, w: the pass's latents and weights; m, W: the reference's latents
    (S/W) and weights, all in the same order. Only latents with w > 0
    and W > 0 count (M of them). Arrays of different lengths are
    compared over the common prefix (shorter frames carry a prefix of
    the air order).
    """
    z, _, m, _, v = _pair(z, w, m, W)
    if v.size == 0:
        return 0.0, 0
    num = float(np.sum(v * z * m))
    den = float(np.sum((v * z * m) ** 2))
    return (num / np.sqrt(den) if den > 0 else 0.0), int(v.size)


def expected_t(w, W, kurtosis: float = KURTOSIS) -> float:
    """Mean of t for the same picture: sum v / sqrt(sum (v + k v^2)).

    With z = a + noise of variance 1/w and m = a + noise of variance
    1/W, E[v z m] = v and E[v^2 z^2 m^2] = k v^2 + v exactly (that
    identity is why v is the weight), so this is the value t
    concentrates on when the weights are honest.
    """
    w = np.asarray(w, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n = min(w.size, W.size)
    w, W = w[:n], W[:n]
    keep = (w > 0) & (W > 0) & np.isfinite(w) & np.isfinite(W)
    v = w[keep] * W[keep] / (1 + w[keep] + W[keep])
    if v.size == 0:
        return 0.0
    return float(np.sum(v) / np.sqrt(np.sum(v + kurtosis * v * v)))


def corr_accept(t: float, t_exp: float) -> bool:
    """t > 6 and t >= RHO_MIN * expected t (see the module docstring)."""
    return t > T_ACCEPT and t >= RHO_MIN * t_exp


# --- references ---------------------------------------------------------------------


def _acc_reference(acc, g: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Accumulator segment g as (m, W) in air order, first n latents."""
    idx = canonical_index(g, n)
    S = acc.S[g, idx].astype(np.float64)
    W = acc.W[g, idx].astype(np.float64)
    m = np.divide(S, W, out=np.zeros_like(W), where=W > 0)
    return m, W


def _cluster_sums(store: Store, uids) -> tuple[np.ndarray, np.ndarray]:
    """A provisional cluster's air-order (m, W): its passes summed as one."""
    zws = [store.pass_zw(u) for u in uids]
    n = max(z.size for z, _ in zws)
    S = np.zeros(n)
    W = np.zeros(n)
    for z, w in zws:
        S[:z.size] += np.where(w > 0, w * z, 0.0)
        W[:w.size] += w
    m = np.divide(S, W, out=np.zeros_like(W), where=W > 0)
    return m, W


def _best_accumulator(store: Store, z, w, exclude=()):
    """Rule 3: the (t, key, segment) of the best accepted accumulator, or None."""
    best = None
    for key in store.open_keys():
        if key in exclude:
            continue
        acc = store.accumulator(key)
        for g in range(acc.mode + 1):
            if not acc.has_data(g):
                continue
            m, W = _acc_reference(acc, g, min(z.size, SENT))
            t, _ = corr_stat(z, w, m, W)
            if corr_accept(t, expected_t(w, W)) and (best is None or t > best[0]):
                best = (t, key, g)
    return best


# --- rule 1 validation ----------------------------------------------------------------


def validate_header_match(store: Store, key, mode: int, segment: int, z, w) -> tuple[bool, dict]:
    """Rule 1's correlation check of a header match. Returns (ok, details).

    No check (ok) while there is no accumulator or its segment's median
    W is at most -10 dB. A mode that disagrees with the accumulator's is
    a collision as well.
    """
    if not store.has_accumulator(key):
        return True, {}
    acc = store.accumulator(key)
    if int(mode) != acc.mode:
        return False, {"reason": "mode", "acc_mode": acc.mode, "mode": int(mode)}
    if not acc.has_data(segment) or acc.median_w(segment) <= VALIDATE_MEDIAN_W:
        return True, {}
    m, W = _acc_reference(acc, segment, np.asarray(z).size)
    t, M = corr_stat(z, w, m, W)
    t_exp = expected_t(w, W)
    ok = not (t < HDR_REJECT_T and t_exp > HDR_REJECT_EXPECT)
    return ok, {"t": t, "M": M, "t_expected": t_exp}


def _attach_by_header(store: Store, h, uids, method: str) -> list[str]:
    """Attach each pass in `uids` under header h after rule 1's check.

    Returns the uids attached; the others are logged as collisions and
    left (or put) in the provisional clusters.
    """
    key = (h.callsign, int(h.picture_id))
    done = []
    for u in uids:
        z, w = store.pass_zw(u)
        ok, info = validate_header_match(store, key, h.mode, h.segment, z, w)
        if ok:
            store.attach(key, u, h.segment, method, mode=h.mode, codec_id=h.codec_id)
            done.append(u)
        else:
            store.log({"event": "collision", "uid": u, "key": [key[0], key[1]],
                       "rule": method, **info})
            if store.prov_cluster_of(u) is None:
                store.prov_add(u)
    return done


# --- rule 2 --------------------------------------------------------------------------


def _soft_header(store: Store, uid: str, llr) -> tuple[object, list[str]] | None:
    """Rule 2: (HeaderFields, uids involved) from summed LLRs, or None."""
    me = store.pass_info(uid)
    llr = np.asarray(llr, dtype=np.float64)
    near = lambda o: (abs(o["q"] - me["q"]) * SLOT_S <= PAIR_DAYS * DAY_S)  # noqa: E731
    clusters = [c for c in store.prov_clusters() if uid not in c["members"]]
    for c in clusters:
        infos = [store.pass_info(u) for u in c["members"]]
        if not any(near(o) for o in infos):
            continue
        h = _header.decode(llr + sum(o["hdr_llr"] for o in infos))
        if h is not None:
            return h, [uid] + list(c["members"])
    for c in clusters:
        if len(c["members"]) < 2:
            continue                          # already tried as a cluster sum
        for u in c["members"]:
            o = store.pass_info(u)
            if not near(o) or abs(o["f_hz"] - me["f_hz"]) > PAIR_HZ:
                continue
            h = _header.decode(llr + o["hdr_llr"])
            if h is not None:
                return h, [uid, u]
    return None


# --- the rules ------------------------------------------------------------------------


def associate(p: PassResult, store: Store) -> tuple[tuple | None, str]:
    """Attach pass p by the first rule that succeeds (stores the pass first).

    Returns (key, rule) with rule in {"header", "soft-header", "corr",
    "prov", ""}: key is the accumulator's (callsign, picture_id), or None
    for a provisional pass. "" means the pass was already a member of
    `key` and nothing changed.
    """
    if not store.has_pass(p.uid):
        store.add_pass(p)
    held = store.find_member(p.uid)
    if held is not None:
        return held, ""
    z = np.asarray(p.z, dtype=np.float64)
    w = np.asarray(p.w, dtype=np.float64)

    # 1. Header decoded on this pass.
    if p.header is not None:
        h = p.header
        if _attach_by_header(store, h, [p.uid], "header"):
            resolve_provisional(store)
            return (h.callsign, int(h.picture_id)), "header"
        return None, "prov"

    # 2. Soft-combined header.
    found = _soft_header(store, p.uid, p.hdr_llr)
    if found is not None:
        h, uids = found
        done = _attach_by_header(store, h, uids, "soft-header")
        if done:
            resolve_provisional(store)
        if p.uid in done:
            return (h.callsign, int(h.picture_id)), "soft-header"
        return None, "prov"

    # 3. Latent correlation against the open accumulators.
    best = _best_accumulator(store, z, w)
    if best is not None:
        t, key, g = best
        store.attach(key, p.uid, g, "corr")
        return key, "corr"

    # 4. Provisional: join the best-correlated cluster, or start one. A
    # pass that is already provisional (re-associated after a restart) is
    # scored against each cluster *without* itself, and stays where it is
    # unless another cluster now accepts it.
    best = None
    for c in store.prov_clusters():
        others = [u for u in c["members"] if u != p.uid]
        if not others:
            continue
        m, W = _cluster_sums(store, others)
        t, _ = corr_stat(z, w, m, W)
        if corr_accept(t, expected_t(w, W)) and (best is None or t > best[0]):
            best = (t, c["cluster"])
    current = store.prov_cluster_of(p.uid)
    if best is None:
        if current is None:
            store.prov_add(p.uid)
    elif current is None or current["cluster"] != best[1]:
        store.prov_add(p.uid, best[1])
    return None, "prov"


def resolve_provisional(store: Store) -> list[tuple[str, tuple]]:
    """Attach whole provisional clusters that now correlate with an accumulator.

    Each cluster's passes, summed, are tested by rule 3; an accepted
    cluster's passes all join that accumulator with the matched segment
    (method "corr"). Returns [(cluster, key), ...].
    """
    out = []
    for c in store.prov_clusters():
        m, W = _cluster_sums(store, c["members"])
        best = _best_accumulator(store, m, W)
        if best is None:
            continue
        _, key, g = best
        for u in list(c["members"]):
            store.attach(key, u, g, "corr")
        out.append((c["cluster"], key))
    return out
