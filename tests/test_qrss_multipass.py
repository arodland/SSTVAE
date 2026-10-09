"""Multi-pass core: the store, association and the decoder planes
(design 10.6: P1, P2, P6, P7, P8).

Fast except P7 and the P8 round trip, which are codec tests. Passes are
the genie fakes of `qrss_fakes` (z unbiased, var 1/w), so these check
the store and association arithmetic alone.

Deviations from the design, both measured here:

- **P7**: on 40 COCO pictures, noiseless pairwise |t| reaches ~17 (std
  ~3, not 1). The design's fallback, subtracting the per-channel latent
  mean, leaves it at ~17: the cause is genuine content similarity
  between pictures (latent correlation up to ~0.13), not a common
  mean. So `latent_means.npy` is not shipped, and acceptance adds
  `t >= RHO_MIN * expected_t` to `t > 6` (`associate.corr_accept`).
  P7 then checks the *decision*: no pair of different pictures is
  accepted at any SNR, and at the association operating point (P6) the
  H0 max |t| is under 5.
- **P8**: `wonder_wheel.jpg` decodes at ~24.4 dB through v5 even with
  every latent clean (27.5 dB is the COCO set's *mean*), so ">= 27 dB"
  cannot hold for it. The round trip is checked against the direct
  decode of the full clean latents instead: within 0.6 dB (the 2,200
  never-sent latents per group are worth ~0.4 dB) and >= 23.5 dB.
"""

import io
import json
import os

import numpy as np
import pytest

from qrss_fakes import make_header, make_pass_result, synthetic_picture
from qrss_helpers import (
    Q_TEST, REPO_ROOT, latent_snr_db, load_codec, require_coco, unit_rms_latents,
)
from sstvae.qrss import associate as asc
from sstvae.qrss import header, picture, render
from sstvae.qrss.store import Store, canonical_index, key_name

KEY_CALL = "K1ABC/P"


class Clock:
    def __init__(self, t=1.76e9):
        self.t = float(t)

    def __call__(self):
        return self.t


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "qrss", clock=Clock())


def _air(sp, g):
    return picture.air_values(sp.segs[g], g)


def _expected_sums(store, members):
    """float64 (S, W) from (uid, segment) pairs, the long way round."""
    S = np.zeros((3, 52800))
    W = np.zeros((3, 52800))
    for uid, g in members:
        p = store.load_pass(uid)
        idx = picture.air_to_canonical(g)[:p.z.size]
        w = p.w.astype(np.float64)
        S[g, idx] += w * p.z.astype(np.float64)
        W[g, idx] += w
    return S, W


# --- P1: store algebra -----------------------------------------------------------


def test_p1_rebuild_from_members_and_canonical_mapping(store):
    """S = sum w z and W = sum w at each segment's canonical offsets; never-sent 0."""
    sp = synthetic_picture(1, mode=2)
    key = (KEY_CALL, sp.picture_id)
    passes = [(make_pass_result(_air(sp, 0), 0.5, seed=1), 0),
              (make_pass_result(_air(sp, 2), 2.0, seed=2), 2),
              (make_pass_result(_air(sp, 0)[:16384], 1.0, seed=3), 0),   # MEDIUM prefix
              (make_pass_result(_air(sp, 1), 0.1, seed=4, erase_after=30000), 1)]
    for p, g in passes:
        store.add_pass(p)
        store.attach(key, p.uid, g, "header", mode=2, codec_id=sp.codec_id)
    acc = store.accumulator(key)
    S, W = _expected_sums(store, [(p.uid, g) for p, g in passes])
    assert acc.S.dtype == np.float32 and acc.S.shape == (3, 52800)
    np.testing.assert_array_equal(acc.S, S.astype(np.float32))
    np.testing.assert_array_equal(acc.W, W.astype(np.float32))
    for g in range(3):
        assert not np.any(acc.W[g, picture.never_sent(g)])
        assert not np.any(acc.S[g, picture.never_sent(g)])
    # Air -> canonical -> air round trip of one pass's contribution.
    p0 = passes[1][0]
    idx = canonical_index(2, p0.z.size)
    np.testing.assert_array_equal(acc.W[2, idx], p0.w)
    np.testing.assert_allclose(acc.S[2, idx] / acc.W[2, idx], p0.z, rtol=1e-6)
    np.testing.assert_array_equal(
        picture.canonical_from_air(p0.w, 2)[picture.air_to_canonical(2)], p0.w)
    # A reload sees the same thing, and rebuild() is idempotent.
    again = Store(store.root).accumulator(key)
    again.rebuild(store)
    np.testing.assert_array_equal(again.S, acc.S)
    np.testing.assert_array_equal(again.W, acc.W)
    assert [m["segment"] for m in again.members] == [0, 2, 0, 1]
    assert again.mode == 2 and again.codec_id == sp.codec_id


def test_p1_loo_is_exact(store):
    """loo(i) equals the accumulator without pass i, bit for bit."""
    sp = synthetic_picture(2)
    key = ("K1ABC", sp.picture_id)
    ps = [make_pass_result(_air(sp, 0), s, seed=i) for i, s in enumerate((0.3, 1.0, 3.0))]
    for p in ps:
        store.add_pass(p)
        store.attach(key, p.uid, 0, "header", mode=0)
    acc = store.accumulator(key)
    S_, W_ = acc.loo(ps[1].uid, store)
    assert S_.dtype == np.float64 and S_.shape == (50600,)
    S, W = _expected_sums(store, [(ps[0].uid, 0), (ps[2].uid, 0)])
    idx = picture.air_to_canonical(0)
    np.testing.assert_array_equal(S_, S[0, idx])
    np.testing.assert_array_equal(W_, W[0, idx])
    # ... and equals what the accumulator holds once the pass is detached.
    acc2 = store.detach(key, ps[1].uid)
    np.testing.assert_array_equal(acc2.S[0, idx], S_.astype(np.float32))
    np.testing.assert_array_equal(acc2.W[0, idx], W_.astype(np.float32))
    with pytest.raises(KeyError):
        acc2.loo(ps[1].uid, store)


def test_p1_merge_is_sum_and_verified(store):
    """A correlated foreign (S, W) adds exactly; unrelated or repeated data is refused."""
    sp = synthetic_picture(3)
    key = ("K1ABC", sp.picture_id)
    a = _air(sp, 0)
    p = make_pass_result(a, 1.0, seed=5)
    store.add_pass(p)
    store.attach(key, p.uid, 0, "header", mode=0)
    before = store.accumulator(key)
    # A foreign receiver's accumulator of the same picture.
    q = make_pass_result(a, 0.5, seed=6)
    Sf = np.zeros((3, 52800), np.float32)
    Wf = np.zeros((3, 52800), np.float32)
    idx = picture.air_to_canonical(0)
    Wf[0, idx] = q.w
    Sf[0, idx] = q.w * q.z
    assert store.merge(key, Sf, Wf, source="rx2")
    acc = store.accumulator(key)
    np.testing.assert_array_equal(acc.S_ext, Sf)
    np.testing.assert_array_equal(acc.W_ext, Wf)
    np.testing.assert_array_equal(
        acc.S, (before.S.astype(np.float64) + Sf).astype(np.float32))
    np.testing.assert_array_equal(
        acc.W, (before.W.astype(np.float64) + Wf).astype(np.float32))
    assert acc.foreign[0]["source"] == "rx2"
    assert not store.merge(key, Sf, Wf, source="rx2 again")          # no double count
    # Rebuilding keeps the invariant S = S_ext + sum of members.
    acc.rebuild(store)
    np.testing.assert_array_equal(acc.W, (before.W.astype(np.float64) + Wf).astype(np.float32))
    # Another picture's accumulator does not verify.
    other = _air(synthetic_picture(4), 0)
    Wo = np.zeros((3, 52800), np.float32)
    So = np.zeros((3, 52800), np.float32)
    Wo[0, idx] = 1.0
    So[0, idx] = other
    assert not store.merge(key, So, Wo, source="stranger")
    np.testing.assert_array_equal(store.accumulator(key).W_ext, Wf)


def test_p1_merge_from_the_same_source_replaces(store):
    """A source's later, grown accumulator replaces its earlier one rather than adding."""
    sp = synthetic_picture(3)
    key = ("K1ABC", sp.picture_id)
    a = _air(sp, 0)
    idx = picture.air_to_canonical(0)
    p = make_pass_result(a, 1.0, seed=5)
    store.add_pass(p)
    store.attach(key, p.uid, 0, "header", mode=0)
    q1, q2, r = (make_pass_result(a, 0.5, seed=s) for s in (6, 7, 8))

    def accum(ps):
        S = np.zeros((3, 52800), np.float32)
        W = np.zeros((3, 52800), np.float32)
        for q in ps:
            W[0, idx] += q.w
            S[0, idx] += q.w * q.z
        return S, W
    assert store.merge(key, *accum([q1]), source="rx2")
    assert store.merge(key, *accum([r]), source="rx3")
    S12, W12 = accum([q1, q2])
    assert store.merge(key, S12, W12, source="rx2")
    acc = store.accumulator(key)
    Sr, Wr = accum([r])
    np.testing.assert_array_equal(acc.W_ext, (W12.astype(np.float64) + Wr).astype(np.float32))
    np.testing.assert_array_equal(acc.S_ext, (S12.astype(np.float64) + Sr).astype(np.float32))
    assert np.median(acc.W_ext[0, idx]) == pytest.approx(1.5)
    assert [f["source"] for f in acc.foreign] == ["rx3", "rx2"]
    assert sorted(acc.parts) == sorted(f["sha"] for f in acc.foreign)
    acc.rebuild(store)
    np.testing.assert_array_equal(acc.W[0, idx], (acc.W_ext[0, idx].astype(np.float64)
                                                  + p.w).astype(np.float32))


def test_pass_cache_is_bounded(store, monkeypatch):
    """pass_zw keeps only the most recently used ZW_CACHE passes in memory."""
    from sstvae.qrss import store as store_mod
    monkeypatch.setattr(store_mod, "ZW_CACHE", 3)
    a = unit_rms_latents(800, seed=1)
    ps = [make_pass_result(a, 1.0, seed=i, q=Q_TEST + i) for i in range(5)]
    for p in ps:
        store.add_pass(p)
        store.pass_zw(p.uid)
    assert list(store._zw) == [p.uid for p in ps[2:]]
    z, w = store.pass_zw(ps[0].uid)                    # evicted: read back from disk
    np.testing.assert_array_equal(z, ps[0].z.astype(np.float64))
    assert len(store._zw) == 3

def test_p1_expiry(store):
    sp_old, sp_new = synthetic_picture(5), synthetic_picture(6)
    old = make_pass_result(_air(sp_old, 0), 1.0, seed=1, q=Q_TEST)
    new = make_pass_result(_air(sp_new, 0), 1.0, seed=2, q=Q_TEST + 4 * 96 * 8)
    stray = make_pass_result(_air(sp_old, 0), 0.01, seed=3, q=Q_TEST)
    k_old, k_new = ("K1ABC", sp_old.picture_id), ("K1ABC", sp_new.picture_id)
    t0 = store.clock.t
    for p, k in ((old, k_old), (new, k_new)):
        store.add_pass(p)
    store.attach(k_old, old.uid, 0, "header", mode=0)
    store.add_pass(stray)
    store.prov_add(stray.uid)
    store.clock.t = t0 + 8 * 86400
    store.attach(k_new, new.uid, 0, "header", mode=0)
    removed = store.expire(days=7.0)
    assert removed["acc"] == [k_old]
    assert len(removed["prov"]) == 1
    assert sorted(removed["passes"]) == sorted([old.uid, stray.uid])
    assert store.open_keys() == [k_new]
    assert store.pass_uids() == [new.uid]
    assert store.expire(days=7.0) == {"acc": [], "prov": [], "passes": []}


def test_p1_atomic_writes(store, monkeypatch):
    """No temporary files are left, and a failed write leaves the old file whole."""
    sp = synthetic_picture(7)
    key = ("K1ABC", sp.picture_id)
    p = make_pass_result(_air(sp, 0), 1.0, seed=1)
    store.add_pass(p)
    store.attach(key, p.uid, 0, "header", mode=0)
    names = [f.name for f in store.root.rglob("*")]
    assert not [n for n in names if n.endswith(".tmp")]
    jpath = store.root / "acc" / f"{key_name(key)}.json"
    good = jpath.read_bytes()

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    p2 = make_pass_result(_air(sp, 0), 1.0, seed=2)
    with pytest.raises(OSError):
        store.add_pass(p2)
    with pytest.raises(OSError):
        store.attach(key, p.uid, 0, "renamed")
    monkeypatch.undo()
    assert jpath.read_bytes() == good
    assert not store.has_pass(p2.uid)
    assert not [f for f in store.root.rglob("*") if f.name.endswith(".tmp")]
    assert key_name(("K1ABC/P", 0x1234)) == "K1ABC_P_00001234"


def test_p1_update_pass_rebuilds(store):
    """EM's replacement of a pass's (z, w) reaches every accumulator holding it."""
    sp = synthetic_picture(8)
    key = ("K1ABC", sp.picture_id)
    p = make_pass_result(_air(sp, 0), 0.5, seed=1)
    store.add_pass(p)
    store.attach(key, p.uid, 0, "header", mode=0)
    better = make_pass_result(_air(sp, 0), 2.0, seed=9, em_round=1)
    p.z, p.w, p.em_round = better.z, better.w, 1
    assert store.update_pass(p) == [key]
    acc = store.accumulator(key)
    np.testing.assert_array_equal(acc.W[0, picture.air_to_canonical(0)], better.w)
    assert acc.members[0]["em_round"] == 1
    assert acc.members[0]["mean_w_db"] == pytest.approx(10 * np.log10(2.0), abs=1e-5)


def test_p1_attach_moves_a_pass(store):
    sp1, sp2 = synthetic_picture(9), synthetic_picture(10)
    k1, k2 = ("K1ABC", sp1.picture_id), ("K1ABC", sp2.picture_id)
    p = make_pass_result(_air(sp1, 0), 1.0, seed=1)
    store.add_pass(p)
    store.prov_add(p.uid)
    store.attach(k1, p.uid, 0, "corr", mode=0)
    assert store.prov_clusters() == []
    store.attach(k2, p.uid, 0, "header", mode=0)
    assert store.accumulator(k1).members == []
    assert not np.any(store.accumulator(k1).W)
    assert store.find_member(p.uid) == k2
    with pytest.raises(ValueError):
        store.attach(k2, p.uid, 1, "header")           # mode A has no segment 1


# --- P2: N passes add like 10 log10 N ----------------------------------------------


def test_p2_combining_gain(store):
    """N genie passes at s = -10 dB: S/W delivers 10 log10(N s) +- 0.2 dB up to N = 32."""
    s = 0.1
    a = unit_rms_latents(50600, seed=3)
    key = ("K1ABC", 0x0BADCAFE)
    n_done = 0
    for n in (1, 2, 4, 8, 16, 32):
        while n_done < n:
            p = make_pass_result(a, s, seed=100 + n_done)
            store.add_pass(p)
            store.attach(key, p.uid, 0, "header", mode=0)
            n_done += 1
        acc = store.accumulator(key)
        idx = picture.air_to_canonical(0)
        est = acc.S[0, idx] / acc.W[0, idx]
        assert latent_snr_db(est, a) == pytest.approx(10 * np.log10(n * s), abs=0.2)
        assert 10 * np.log10(acc.median_w(0)) == pytest.approx(10 * np.log10(n * s), abs=1e-4)
    # Leave-one-out of 32 is 31 passes' worth.
    S_, W_ = acc.loo(acc.members[0]["uid"], store)
    assert latent_snr_db(S_ / W_, a) == pytest.approx(10 * np.log10(31 * s), abs=0.2)


# --- P6: association sensitivity ---------------------------------------------------------


def _crossing_m(w, W, n_seeds=24, n=50600):
    """M at which the seed-mean of t reaches 6 (pass z vs reference m, same latents)."""
    ms = np.unique(np.geomspace(200, n, 60).astype(int))
    ts = np.zeros(ms.size)
    v = w * W / (1 + w + W)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal(n)
        z = a + rng.standard_normal(n) / np.sqrt(w)
        m = a + rng.standard_normal(n) / np.sqrt(W)
        num = np.cumsum(v * z * m)
        den = np.sqrt(np.cumsum((v * z * m) ** 2))
        ts += (num / den)[ms - 1]
    ts /= n_seeds
    return float(np.interp(6.0, ts, ms))


@pytest.mark.parametrize("w_db, W_db, m_design", [(-16.0, 2.2, 2500), (-11.25, -11.25, 7300)])
def test_p6_acceptance_point(w_db, W_db, m_design):
    """-16 dB pass vs +2.2 dB accumulator: accepted by M ~ 2,500; two s = 0.075 passes: ~7,300.

    Measured as the M where the mean t over 24 seeds reaches 6 (+-30%),
    and checked against `expected_t`, the closed form behind it.
    """
    w, W = 10 ** (w_db / 10), 10 ** (W_db / 10)
    m_meas = _crossing_m(w, W)
    assert m_meas == pytest.approx(m_design, rel=0.3)
    m_pred = 6.0 ** 2 * (1 + 3 * (w * W / (1 + w + W))) / (w * W / (1 + w + W))
    assert m_meas == pytest.approx(m_pred, rel=0.1)
    assert asc.expected_t(np.full(int(m_pred), w), np.full(int(m_pred), W)) \
        == pytest.approx(6.0, abs=0.01)


def test_p6_null_is_standard_normal():
    """Independent latents: t ~ N(0, 1) (mean within 0.15, std within 10%, 400 draws)."""
    rng = np.random.default_rng(7)
    ts = []
    for _ in range(400):
        z = rng.standard_normal(5000) + rng.standard_normal(5000) * 2
        m = rng.standard_normal(5000) + rng.standard_normal(5000) * 0.5
        ts.append(asc.corr_stat(z, np.full(5000, 0.25), m, np.full(5000, 4.0))[0])
    ts = np.array(ts)
    assert abs(ts.mean()) < 0.15
    assert ts.std() == pytest.approx(1.0, rel=0.1)
    t, M = asc.corr_stat(np.ones(10), np.zeros(10), np.ones(10), np.ones(10))
    assert (t, M) == (0.0, 0)


def test_p6_associate_by_correlation(store):
    """A full -16 dB pass with no header joins a +2.2 dB accumulator; 800 latents do not."""
    sp = synthetic_picture(11, mode=1)
    key = (KEY_CALL, sp.picture_id)
    for g in (0, 1):
        h = make_header(sp, segment=g, callsign=KEY_CALL)
        p = make_pass_result(_air(sp, g), 10 ** 0.22, seed=g, hdr=h, decoded=True)
        assert asc.associate(p, store) == (key, "header")
    weak = make_pass_result(_air(sp, 1), 10 ** -1.6, seed=21, q=Q_TEST + 1)
    assert asc.associate(weak, store) == (key, "corr")
    assert store.accumulator(key).member(weak.uid)["segment"] == 1
    short = make_pass_result(_air(sp, 0), 10 ** -1.6, seed=22, erase_after=800, q=Q_TEST + 2)
    assert asc.associate(short, store) == (None, "prov")
    assert asc.associate(weak, store) == (key, "")         # already a member


# --- association rules -----------------------------------------------------------------


def test_rule1_collision_goes_provisional(store):
    """A header match whose latents do not correlate is rejected and logged."""
    sp = synthetic_picture(12)
    key = ("K1ABC", sp.picture_id)
    h = make_header(sp, callsign="K1ABC")
    p = make_pass_result(_air(sp, 0), 10.0, seed=1, hdr=h, decoded=True)
    assert asc.associate(p, store) == (key, "header")
    impostor = make_pass_result(_air(synthetic_picture(13), 0), 1.0, seed=2,
                                hdr=h, decoded=True, q=Q_TEST + 1)
    assert asc.associate(impostor, store) == (None, "prov")
    assert store.prov_cluster_of(impostor.uid) is not None
    log = [json.loads(x) for x in (store.root / "log.jsonl").read_text().splitlines()]
    assert log[-1]["event"] == "collision" and log[-1]["t_expected"] > 8
    # A genuine weak repeat still passes the check.
    rep = make_pass_result(_air(sp, 0), 0.05, seed=3, hdr=h, decoded=True, q=Q_TEST + 2)
    assert asc.associate(rep, store) == (key, "header")


def test_rule2_soft_combined_header(store):
    """Passes whose headers fail alone cluster by correlation, then decode summed.

    Four passes at Es/N0 = -15 dB per coded bit (-9 dB summed): each
    joins the provisional cluster (rule 4) until the summed LLRs decode
    (rule 2), and then every pass involved is attached.
    """
    sp = synthetic_picture(14)
    h = make_header(sp, callsign="G4XYZ", grid=None)
    key = ("G4XYZ", sp.picture_id)
    ps = [make_pass_result(_air(sp, 0), 0.05, seed=i, hdr=h, hdr_es_n0_db=-15.0,
                           q=Q_TEST + 4 * i, f_hz=1500.0 + i) for i in range(4)]
    assert all(header.decode(p.hdr_llr) is None for p in ps)
    results = [asc.associate(p, store) for p in ps]
    first = next(i for i, r in enumerate(results) if r[1] == "soft-header")
    assert first >= 1
    assert all(r == (None, "prov") for r in results[:first])
    assert results[first] == (key, "soft-header")
    acc = store.accumulator(key)
    assert set(acc.uids) >= {p.uid for p in ps[:first + 1]}
    assert {m["method"] for m in acc.members} <= {"soft-header", "corr"}
    assert store.prov_clusters() == []


def test_rule4_provisional_clusters_and_resolution(store):
    """Weak repeats cluster; a later header pass names the cluster and absorbs it."""
    sp, other = synthetic_picture(15), synthetic_picture(16)
    a, b = _air(sp, 0), _air(other, 0)
    p1 = make_pass_result(a, 0.075, seed=1)
    p2 = make_pass_result(a, 0.075, seed=2, q=Q_TEST + 1)
    x = make_pass_result(b, 0.075, seed=3, q=Q_TEST + 2)
    assert asc.associate(p1, store) == (None, "prov")
    assert asc.associate(p2, store) == (None, "prov")
    assert asc.associate(x, store) == (None, "prov")
    clusters = {c["cluster"]: sorted(c["members"]) for c in store.prov_clusters()}
    assert sorted(clusters.values()) == sorted([sorted([p1.uid, p2.uid]), [x.uid]])
    h = make_header(sp, callsign="K1ABC")
    named = make_pass_result(a, 1.0, seed=4, hdr=h, decoded=True, q=Q_TEST + 3)
    key = ("K1ABC", sp.picture_id)
    assert asc.associate(named, store) == (key, "header")
    assert sorted(store.accumulator(key).uids) == sorted([p1.uid, p2.uid, named.uid])
    assert [c["members"] for c in store.prov_clusters()] == [[x.uid]]


def test_similar_picture_rejected_at_high_snr(store):
    """The RHO_MIN gate: a strong pass of a 30%-correlated picture is not a match.

    t alone would accept it (t ~ 0.3 * expected); the same picture at the
    same SNR is accepted.
    """
    rng = np.random.default_rng(0)
    a = unit_rms_latents(50600, seed=1)
    b = 0.3 * a + np.sqrt(1 - 0.09) * rng.standard_normal(a.size)
    acc_pass = make_pass_result(a, 10.0, seed=1)
    key = ("K1ABC", 0x01020304)
    store.add_pass(acc_pass)
    store.attach(key, acc_pass.uid, 0, "header", mode=0)
    acc = store.accumulator(key)
    m, W = asc._acc_reference(acc, 0, 50600)
    sim = make_pass_result(b, 10.0, seed=2)
    t, _ = asc.corr_stat(sim.z, sim.w, m, W)
    t_exp = asc.expected_t(sim.w, W)
    assert t > 6 and t < asc.RHO_MIN * t_exp
    assert not asc.corr_accept(t, t_exp)
    assert asc.associate(sim, store) == (None, "prov")
    same = make_pass_result(a, 10.0, seed=3)
    assert asc.associate(same, store) == (key, "corr")


def test_corr_stat_weights_each_latent_by_v():
    """t weights latent i by v_i = w_i W_i / (1 + w_i + W_i), not uniformly.

    A pass that fades half way (w = 0.05, then 1e-3) against W = 1.66:
    the weighted t sits on `expected_t` (~25), where an unweighted sum
    would give ~7 -- and fail the RHO_MIN half of the gate.
    """
    rng = np.random.default_rng(0)
    n = 50600
    a = rng.standard_normal(n)
    w = np.where(np.arange(n) < n // 2, 0.05, 1e-3)
    W = np.full(n, 1.66)
    W[::7] = 0.4                                       # and a non-uniform reference
    z = a + rng.standard_normal(n) / np.sqrt(w)
    m = a + rng.standard_normal(n) / np.sqrt(W)
    t, M = asc.corr_stat(z, w, m, W)
    v = w * W / (1 + w + W)
    assert M == n
    assert t == pytest.approx(np.sum(v * z * m) / np.sqrt(np.sum((v * z * m) ** 2)), rel=1e-12)
    t_exp = asc.expected_t(w, W)
    assert t_exp > 20
    assert t == pytest.approx(t_exp, abs=3.0)
    t_flat = np.sum(z * m) / np.sqrt(np.sum((z * m) ** 2))
    assert t_flat < asc.RHO_MIN * t_exp                # what v = 1 would have rejected


def test_fading_pass_associates_by_correlation(store):
    """The same fading pass, end to end: rule 3 attaches it."""
    sp = synthetic_picture(17)
    key = ("K1ABC", sp.picture_id)
    a = _air(sp, 0)
    ref = make_pass_result(a, 1.66, seed=1, hdr=make_header(sp, callsign="K1ABC"), decoded=True)
    assert asc.associate(ref, store) == (key, "header")
    w = np.where(np.arange(a.size) < a.size // 2, 0.05, 1e-3)
    fade = make_pass_result(a, w=w, seed=2, q=Q_TEST + 1)
    assert asc.associate(fade, store) == (key, "corr")


def test_reassociating_a_provisional_pass_is_idempotent(store):
    """associate() on a pass already in a cluster leaves the clusters as they were.

    It is scored against each cluster without itself, so a lone pass
    does not match (and delete) its own cluster.
    """
    sp, other = synthetic_picture(18), synthetic_picture(19)
    a, b = _air(sp, 0), _air(other, 0)
    x = make_pass_result(b, 0.075, seed=3)
    assert asc.associate(x, store) == (None, "prov")
    before = store.prov_clusters()
    assert asc.associate(x, store) == (None, "prov")
    assert store.prov_clusters() == before
    p1 = make_pass_result(a, 0.075, seed=1, q=Q_TEST + 1)
    p2 = make_pass_result(a, 0.075, seed=2, q=Q_TEST + 2)
    for p in (p1, p2):
        assert asc.associate(p, store) == (None, "prov")
    before = {c["cluster"]: sorted(c["members"]) for c in store.prov_clusters()}
    for p in (x, p1, p2):
        assert asc.associate(p, store) == (None, "prov")
    after = {c["cluster"]: sorted(c["members"]) for c in store.prov_clusters()}
    assert after == before
    assert sorted(after.values()) == sorted([sorted([p1.uid, p2.uid]), [x.uid]])
    cid = store.prov_cluster_of(x.uid)["cluster"]
    assert store.prov_add(x.uid, cid) == cid                # already there: a no-op
    assert {c["cluster"]: sorted(c["members"]) for c in store.prov_clusters()} == before


def test_reassociation_scores_a_pass_without_itself(store):
    """A pass held with a stranger moves to its own picture's cluster when re-associated.

    Scored with itself included, its own cluster would always win.
    """
    sp, other = synthetic_picture(22), synthetic_picture(23)
    a, b = _air(sp, 0), _air(other, 0)
    stranger = make_pass_result(b, 0.075, seed=1)
    p = make_pass_result(a, 0.075, seed=2, q=Q_TEST + 1)
    mate = make_pass_result(a, 0.075, seed=3, q=Q_TEST + 2)
    for u in (stranger, p, mate):
        store.add_pass(u)
    c1 = store.prov_add(stranger.uid)
    store.prov_add(p.uid, c1)
    c2 = store.prov_add(mate.uid)
    assert asc.associate(p, store) == (None, "prov")
    assert store.prov_cluster_of(p.uid)["cluster"] == c2
    assert store.prov_cluster_of(stranger.uid)["cluster"] == c1


def _rule2_setup(store, sp, h, *, u1_q, u1_f, u2_q, u2_f, two=True):
    """A cluster [u1, u2] (or [u1]) and a new pass `new`, none decodable alone.

    new + u1 decodes; u2's LLRs cancel new + u1 exactly, so the
    cluster *sum* is zero and only rule 2's pairwise step can find h.
    """
    a = _air(sp, 0)
    new = make_pass_result(a, 0.05, seed=10, hdr=h, hdr_es_n0_db=-15.0,
                           q=Q_TEST + 10, f_hz=1500.0)
    u1 = make_pass_result(a, 0.05, seed=11, hdr=h, hdr_es_n0_db=-15.0, q=u1_q, f_hz=u1_f)
    for p in (new, u1):
        assert header.decode(p.hdr_llr) is None
    assert header.decode(new.hdr_llr.astype(np.float64) + u1.hdr_llr) is not None
    store.add_pass(u1)
    cid = store.prov_add(u1.uid)
    if two:
        u2 = make_pass_result(a, 0.05, seed=12, q=u2_q, f_hz=u2_f)
        u2.hdr_llr = -(new.hdr_llr + u1.hdr_llr)
        store.add_pass(u2)
        store.prov_add(u2.uid, cid)
    return new, u1


def test_rule2_pairwise_sums(store):
    """Rule 2's pairwise step decodes a pair whose cluster sum does not."""
    sp = synthetic_picture(20)
    h = make_header(sp, callsign="G4XYZ", grid=None)
    new, u1 = _rule2_setup(store, sp, h, u1_q=Q_TEST + 11, u1_f=1505.0,
                           u2_q=Q_TEST + 12, u2_f=1500.0)
    key = ("G4XYZ", sp.picture_id)
    assert asc.associate(new, store) == (key, "soft-header")
    acc = store.accumulator(key)
    assert {new.uid, u1.uid} <= set(acc.uids)


@pytest.mark.parametrize("case", ["far_hz", "far_days", "cluster_far_days"])
def test_rule2_restrictions(store, case):
    """Pairs need +-10 Hz and 7 days; a cluster sum needs a member within 7 days."""
    sp = synthetic_picture(21)
    h = make_header(sp, callsign="G4XYZ", grid=None)
    week = int(7 * 86400 / 900) + 4                    # just over 7 days, in slots
    near_q, far_q = Q_TEST + 11, Q_TEST + 10 + week
    if case == "far_hz":
        new, _ = _rule2_setup(store, sp, h, u1_q=near_q, u1_f=1520.0,
                              u2_q=near_q, u2_f=1520.0)
    elif case == "far_days":
        new, _ = _rule2_setup(store, sp, h, u1_q=far_q, u1_f=1500.0,
                              u2_q=near_q, u2_f=1520.0)
    else:                                              # a lone, decodable but old cluster
        new, _ = _rule2_setup(store, sp, h, u1_q=far_q, u1_f=1500.0,
                              u2_q=0, u2_f=0.0, two=False)
    assert asc.associate(new, store) == (None, "prov")
    assert not store.has_accumulator(("G4XYZ", sp.picture_id))


# --- P8: decoder planes --------------------------------------------------------------------


def test_p8_decoder_planes_rules():
    """g(W), the one-reference W~ rule, never-sent and absent groups at weight 0, shapes."""
    assert render.g_shrink(0.0) == 0.0
    assert float(render.g_shrink(1e12)) == pytest.approx(0.785, rel=1e-9)
    assert float(render.g_shrink(0.39)) == pytest.approx(0.785 / np.sqrt(2))
    rng = np.random.default_rng(0)
    S = np.zeros((3, 52800), np.float32)
    W = np.zeros((3, 52800), np.float32)
    a = rng.standard_normal((3, 52800))
    for g, level in ((0, 4.0), (1, 0.5)):            # group 2 has no data
        idx = picture.air_to_canonical(g)
        W[g, idx] = level * rng.uniform(0.5, 1.5, idx.size)
        S[g, idx] = W[g, idx] * a[g, idx]
    W[1, picture.air_to_canonical(1)[:1000]] = 0.0   # a few erasures
    S[1, picture.air_to_canonical(1)[:1000]] = 0.0
    lat, wt = render.decoder_planes(S, W, mode=2)
    assert lat.shape == wt.shape == (158400,) and lat.dtype == wt.dtype == np.float32
    lat, wt = lat.reshape(3, 52800), wt.reshape(3, 52800)
    w_ref = np.median(W[0, picture.air_to_canonical(0)])
    assert render.reference_w(W, 2) == pytest.approx(w_ref)
    for g in (0, 1):
        idx = picture.air_to_canonical(g)
        heard = idx[W[g, idx] > 0]
        np.testing.assert_allclose(
            wt[g, heard], np.minimum(np.sqrt(W[g, heard] / w_ref), 1), rtol=1e-6)
        np.testing.assert_allclose(
            lat[g, heard], render.g_shrink(W[g, heard]) * a[g, heard], rtol=1e-5, atol=1e-6)
        assert not np.any(wt[g, picture.never_sent(g)])
        assert not np.any(lat[g, picture.never_sent(g)])
    assert np.sum(wt[0] == 1.0) > 0.4 * 50600         # full-strength latents keep weight 1
    assert wt[1].max() <= np.sqrt(1.5 * 0.5 / (0.5 * 4.0)) + 1e-6
    assert not np.any(wt[1, picture.air_to_canonical(1)[:1000]])
    assert not np.any(wt[2]) and not np.any(lat[2])
    # Groups above the mode are ignored even if W holds something there.
    lat0, wt0 = render.decoder_planes(S, W, mode=0)
    assert not np.any(wt0.reshape(3, -1)[1:])
    # One W~ for the picture: the strongest group's median sets it.
    W2 = W.copy()
    W2[0] *= 0.01                                    # now group 1 is the strongest
    assert render.reference_w(W2, 2) == pytest.approx(
        np.median(W[1, picture.air_to_canonical(1)]))
    with pytest.raises(ValueError):
        render.decoder_planes(S, -W, 2)


@pytest.mark.codec
def test_p8_noiseless_round_trip():
    """wonder_wheel.jpg through store-shaped planes decodes like the clean latents."""
    from sstvae.images import load_image

    img = REPO_ROOT / "wonder_wheel.jpg"
    if not img.exists():
        pytest.skip("wonder_wheel.jpg not in the repo")
    codec = load_codec("fp16")
    x = load_image(img)
    full = codec.encode(x)
    sp = picture.StoredPicture.from_latents(full, 0xD1D8, 0)
    S = np.zeros((3, 52800), np.float32)
    W = np.zeros((3, 52800), np.float32)
    W[0, picture.air_to_canonical(0)] = 1e4
    S[0] = W[0] * sp.segs[0].astype(np.float32)

    def psnr(pic):
        y = np.asarray(pic, np.float32).transpose(2, 0, 1) / 255
        return float(10 * np.log10(1 / np.mean((y - x) ** 2)))

    got = psnr(render.render(codec, S, W, 0))
    ref_w = np.zeros_like(full)
    ref_w[:52800] = 1
    ref = psnr(codec.decode(0.785 * full, ref_w))
    print(f"P8: wonder_wheel mode A round trip {got:.2f} dB, all-latent decode {ref:.2f} dB")
    assert got >= 23.5
    assert got >= ref - 0.6


# --- P7: H0 of t on real latents ------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.codec
def test_p7_null_on_coco_pictures():
    """40 COCO pictures pairwise, v5: no false association at any SNR; H0 max|t| < 5 at P6."""
    import pyarrow.parquet as pq
    from PIL import Image

    from sstvae.images import fit_image, image_to_array

    path = require_coco()
    codec = load_codec("fp16")
    rows = pq.ParquetFile(path).read_row_group(0).slice(0, 40).to_pylist()
    segs = []
    for r in rows:
        x = image_to_array(fit_image(Image.open(io.BytesIO(r["image"]["bytes"]))))
        sp = picture.StoredPicture.from_latents(codec.encode(x), 0xD1D8, 2)
        segs.append([_air(sp, g) for g in range(3)])
    n = len(segs)
    off = ~np.eye(n, dtype=bool)
    report = []
    for name, w, W in (("noiseless", 1e4, 1e4), ("header-grade", 0.23, 10.0),
                       ("P6", 10 ** -1.6, 10 ** 0.22)):
        rng = np.random.default_rng(1)
        t_exp = asc.expected_t(np.full(50600, w), np.full(50600, W))
        for g in range(3):
            A = np.stack([s[g] for s in segs])
            Z = A + rng.standard_normal(A.shape) / np.sqrt(w)
            M = A + rng.standard_normal(A.shape) / np.sqrt(W)
            # Uniform weights: v cancels, t_ij = sum z_i m_j / sqrt(sum z_i^2 m_j^2).
            T = (Z @ M.T) / np.sqrt((Z ** 2) @ (M ** 2).T)
            h0 = T[off]
            accepted = (h0 > asc.T_ACCEPT) & (h0 >= asc.RHO_MIN * t_exp)
            assert not accepted.any(), f"{name} group {g}: a different picture accepted"
            assert np.all(np.diag(T) > asc.T_ACCEPT)                   # the same picture is
            assert np.all(np.diag(T) >= asc.RHO_MIN * t_exp)
            report.append(f"{name} g{g}: H0 max|t| {np.abs(h0).max():.1f} "
                          f"std {h0.std():.2f}, t_expected {t_exp:.0f}")
            if name == "P6":
                assert np.abs(h0).max() < 5.0
    print("P7: " + "; ".join(report))
