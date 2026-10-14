"""Parity of native/core/qrss/associate (sstvae_native.qrss.associate) with
sstvae/qrss/associate.py: association rules 1 to 4, run by the C++ against
the C++ store. tests/qrss_native/associate.py substitutes it under
--native-qrss=rx (with tests/qrss_native/store.py's Store).

The contract (the bar), and in brackets what was measured (2026-10-14):

- **the statistics**: corr_stat's t and M, expected_t, corr_accept, the
  references (_acc_reference, _cluster_sums) and _best_accumulator, bit
  for bit and of the reference's types -- t an np.float64, or the Python
  float 0.0 where the reference returns it; corr_accept an np.bool_ where
  a numpy operand decides and a bool where a Python one does -- on random
  and degenerate inputs (no overlap, zero numerator and denominator, NaN
  and infinite values, unequal lengths, lists, float32, a custom
  kurtosis) [all equal];
- **rule 1 and 2's checks**: validate_header_match's (ok, details),
  header_agrees and _soft_header (HeaderFields and uids) the same, and
  _rule's strings. header_agrees sums with np.dot (BLAS) in the reference
  and pairwise here, so the two could part only where its statistic lands
  exactly on the threshold [never, over 3000 random LLR sets];
- **association decisions**: every (key, rule) associate returns on
  test_qrss_multipass's scenarios (P1, P2, P6, rules 1 to 4, twins, due),
  run on the reference's store and functions and then on the C++'s, is
  the same, as are the files they leave (`_store.compare_trees`, member
  mean_w_db to the last place as test_qrss_native_store.py says) -- and
  likewise on a fuzz of 300 synthetic passes in six sessions (three
  pictures each, modes A to C, SHORT, TINY, MEDIUM and FULL frames, CE and
  L, decoded headers, header LLRs from -17 to -12 dB, an impostor header
  now and then, same-slot copies, erasures), resolve_provisional's
  result every tenth pass included [all equal; rules reached over the six
  sessions: header, soft-header, corr, prov and ""; 2 collisions logged,
  2 clusters resolved, 148 twins left out];
- **the whole caller**: LiveListener._finish_apply over four slots of
  real fake passes (provisional, then a header that names the cluster,
  then a correlating pass) with every row of this lane in place and
  counting spies installed before the rows: the tiles, the notes, the
  hints (store.due), the log and the files are the reference's, with the
  Store and associate rows reached.

The same rules on pinned inputs, without Python, are
native/tests/qrss/test_associate.cpp's (pins from gen_store_pins.py).

Fast tier ~35 s; slow: three of the six fuzz sessions (~6 s each).
"""

import dataclasses
import functools
import math
import random

import numpy as np
import pytest
from qrss_fakes import make_header, make_pass_result, noisy_llr, synthetic_picture
from qrss_helpers import Q_TEST
from qrss_native import _store as H
from qrss_native import require

import test_qrss_multipass as tm
from sstvae.qrss import associate as asc
from sstvae.qrss import picture
from sstvae.qrss import store as S
from sstvae.qrss.constants import WAVEFORM_L


@pytest.fixture(scope="module")
def na(native):
    require(native, "store")
    return require(native, "associate")


@pytest.fixture(scope="module")
def NStore(native, na):
    return H.store_class(native.qrss)


@pytest.fixture(scope="module")
def row(native, na):
    return {name: fn for (_t, name), fn in H.associate_rows(native.qrss).items()}


def _typed(x):
    """A value with its type: the reference's np.float64 / float / np.bool_
    / bool are part of the contract."""
    if isinstance(x, tuple):
        return tuple(_typed(v) for v in x)
    if isinstance(x, list):
        return [_typed(v) for v in x]
    if isinstance(x, dict):
        return {k: _typed(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return ("nd", x.dtype.str, x.shape, x.tobytes())
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return (type(x).__name__, _typed(dataclasses.asdict(x)))
    return (type(x).__name__, repr(x))


def _outcome(fn, *a, **k):
    try:
        return "ok", _typed(fn(*a, **k))
    except Exception as e:  # noqa: BLE001 - the reference's own type and text
        return type(e).__name__, str(e)


# --- the statistics -------------------------------------------------------------------------------

def _stat_inputs(rng):
    n = rng.choice([0, 1, 5, 64, 1000, 4096])
    m_n = rng.choice([n, n, max(n - 3, 0), n + 7])
    z = rng.standard_normal(n)
    w = rng.uniform(0, 2, n) * (rng.random(n) > 0.2)
    m = rng.standard_normal(m_n)
    W = rng.uniform(0, 3, m_n) * (rng.random(m_n) > 0.1)
    for arr in (z, w, m, W):
        if arr.size and rng.random() < 0.15:
            arr[rng.integers(arr.size)] = rng.choice([np.nan, np.inf, -np.inf])
    kind = rng.integers(6)
    if kind == 1:
        z[:] = 0.0                                   # num = den = 0
    elif kind == 2:
        w[:] = 0.0                                   # no overlap
    elif kind == 3:
        z, w = z.astype(np.float32), w.astype(np.float32)
    elif kind == 4:
        z, w, m, W = z.tolist(), w.tolist(), m.tolist(), W.tolist()
    elif kind == 5:
        m = z[:m_n] if len(z) >= m_n else m          # a correlated reference
    return z, w, m, W


def test_statistics(row):
    """corr_stat, expected_t and corr_accept on 1500 random and degenerate
    inputs: bit for bit, of the reference's types."""
    rng = np.random.default_rng(0)
    for i in range(1500):
        z, w, m, W = _stat_inputs(rng)
        ref = _outcome(asc.corr_stat, z, w, m, W)
        assert _outcome(row["corr_stat"], z, w, m, W) == ref, i
        for args, kw in (((w, W), {}), ((w, W, 2.5), {}), ((W, w), {"kurtosis": 3.09})):
            assert _outcome(row["expected_t"], *args, **kw) == _outcome(asc.expected_t, *args, **kw), i
    vals = [0.0, 6.0, 6.0000001, 7.5, 12.0, 30.0, math.nan, math.inf, -1.0]
    kinds = [float, np.float64, np.float32, int, np.int64, bool, np.bool_]
    for a in vals:
        for b in vals:
            for ka in kinds:
                for kb in kinds:
                    try:
                        x, y = ka(a), kb(b)
                    except (ValueError, OverflowError):
                        continue
                    assert _outcome(row["corr_accept"], x, y) == _outcome(asc.corr_accept, x, y), \
                        (x, y)


def _pair_of_stores(NStore, tmp_path, name, passes, attach):
    out = []
    for cls, sub in ((S.Store, "py"), (NStore, "cpp")):
        st = cls(tmp_path / name / sub, clock=tm.Clock())
        for p in passes:
            st.add_pass(p)
        for key, uid, g, mode in attach:
            st.attach(key, uid, g, "header", mode=mode)
        out.append(st)
    return out


def test_references_and_best(row, NStore, tmp_path):
    """_acc_reference, _cluster_sums, _best_accumulator (exclude included)
    and validate_header_match on the same stores, one per implementation."""
    sps = [synthetic_picture(31, mode=2), synthetic_picture(32), synthetic_picture(33, mode=1)]
    keys = [("K1ABC", sp.picture_id) for sp in sps]
    ps, att = [], []
    for k, sp in enumerate(sps):
        for g in range(sp.mode + 1):
            n = (50600, 4096, 16384)[(k + g) % 3]
            p = make_pass_result(picture.air_values(sp.segs[g], g)[:n], 0.3 + k + g, seed=10 * k + g,
                                 q=Q_TEST + 10 * k + g)
            ps.append(p)
            att.append((keys[k], p.uid, g, sp.mode))
    loose = [make_pass_result(picture.air_values(sps[1].segs[0], 0)[:n], 0.4, seed=90 + i,
                              q=Q_TEST + 90 + i, erase_after=e)
             for i, (n, e) in enumerate([(4096, None), (50600, 3000), (1024, None)])]
    py, cpp = _pair_of_stores(NStore, tmp_path, "refs", ps + loose, att)
    for k, key in enumerate(keys):
        a_py, a_c = py.accumulator(key), cpp.accumulator(key)
        for g in range(3):
            for n in (1, 1024, 4096, 50600):
                ref = _outcome(asc._acc_reference, a_py, g, n)
                assert _outcome(row["_acc_reference"], a_c, g, n) == ref, (key, g, n)
    uids = [p.uid for p in loose]
    for sel in (uids, uids[:1], uids[::-1], [uids[2], uids[0]], []):
        assert _outcome(row["_cluster_sums"], cpp, sel) == _outcome(asc._cluster_sums, py, sel), sel
    for p in ps + loose:
        z, w = p.z.astype(np.float64), p.w.astype(np.float64)
        for ex in ((), (keys[1],), tuple(keys), [keys[0]]):
            assert _outcome(row["_best_accumulator"], cpp, z, w, ex) == \
                _outcome(asc._best_accumulator, py, z, w, ex), (p.uid, ex)
        for key, mode, g in ((keys[0], 2, 0), (keys[0], 1, 0), (keys[1], 0, 0), (keys[2], 1, 1),
                             (("NONE", 1), 0, 0)):
            assert _outcome(row["validate_header_match"], cpp, key, mode, g, z, w) == \
                _outcome(asc.validate_header_match, py, key, mode, g, z, w), (p.uid, key)
    for uid in uids + [ps[0].uid]:
        for key in keys:
            assert _outcome(row["_rule"], cpp, key, uid, "corr") == _outcome(asc._rule, py, key, uid, "corr")
    assert _outcome(row["_rule"], cpp, ("NONE", 1), uids[0], "x") == \
        _outcome(asc._rule, py, ("NONE", 1), uids[0], "x")


def test_header_agrees(row):
    """header_agrees on 3000 LLR sets: its own header, another segment,
    mode or picture, weak to strong, partly heard; None, empty, all-zero
    and wrong-size LLRs."""
    sps = [synthetic_picture(41, mode=2), synthetic_picture(42)]
    hs = [make_header(sps[0], segment=g) for g in range(3)] + [make_header(sps[1])]
    rng = np.random.default_rng(3)
    agree = 0
    for i in range(3000):
        h_sent, h_test = hs[rng.integers(4)], hs[rng.integers(4)]
        llr = noisy_llr(h_sent, float(rng.uniform(-22, -6)), rng).astype(np.float64)
        if rng.random() < 0.3:
            llr[rng.random(llr.size) < rng.uniform(0.1, 0.9)] = 0.0
        if rng.random() < 0.1:
            llr = llr.astype(np.float32)
        ref = _outcome(asc.header_agrees, h_test, llr)
        assert _outcome(row["header_agrees"], h_test, llr) == ref, i
        agree += ref == ("ok", ("bool", "True"))
    assert 300 < agree < 2700
    for llr in (None, np.zeros(2474), np.zeros(10), [], np.ones(2474), -np.ones(2474), [1.0] * 2474):
        assert _outcome(row["header_agrees"], hs[0], llr) == _outcome(asc.header_agrees, hs[0], llr)


def test_soft_header(row, NStore, tmp_path):
    """Rule 2's search on the same provisional clusters: the same header
    (the reference's HeaderFields) and uids, or None."""
    sp = synthetic_picture(14)
    h = make_header(sp, callsign="G4XYZ", grid=None)
    a = picture.air_values(sp.segs[0], 0)
    ps = [make_pass_result(a, 0.05, seed=i, hdr=h, hdr_es_n0_db=-15.0, q=Q_TEST + 4 * i,
                           f_hz=1500.0 + 3 * i) for i in range(6)]
    far = make_pass_result(a, 0.05, seed=9, hdr=h, hdr_es_n0_db=-15.0, q=Q_TEST + 800)
    tiny = make_pass_result(a[:1024], 0.05, seed=10, q=Q_TEST + 1)
    tiny.hdr_llr = np.zeros(0, np.float32)
    for st in _pair_of_stores(NStore, tmp_path, "soft", ps + [far, tiny], []):
        c = st.prov_add(ps[0].uid)
        st.prov_add(ps[1].uid, c)
        st.prov_add(ps[2].uid)
        st.prov_add(far.uid)
        st.prov_add(tiny.uid, c)
    py, cpp = (S.Store(tmp_path / "soft" / "py", clock=tm.Clock()),
               NStore(tmp_path / "soft" / "cpp", clock=tm.Clock()))
    for p in ps + [far, tiny]:
        for llr in (p.hdr_llr, np.zeros(2474), p.hdr_llr[:100]):
            assert _outcome(row["_soft_header"], cpp, p.uid, llr) == \
                _outcome(asc._soft_header, py, p.uid, llr), p.uid


# --- association decisions: test_qrss_multipass's scenarios ------------------------------------------

_SCENARIOS = [
    "test_p1_rebuild_from_members_and_canonical_mapping", "test_p1_loo_is_exact",
    "test_p1_merge_is_sum_and_verified", "test_p1_merge_from_the_same_source_replaces",
    "test_pass_cache_is_bounded", "test_p1_expiry", "test_p1_atomic_writes",
    "test_p1_update_pass_rebuilds", "test_p1_attach_moves_a_pass", "test_p2_combining_gain",
    "test_p6_associate_by_correlation", "test_rule1_collision_goes_provisional",
    "test_rule2_soft_combined_header", "test_rule2_files_only_the_passes_that_carry_the_header",
    "test_rule4_provisional_clusters_and_resolution", "test_similar_picture_rejected_at_high_snr",
    "test_fading_pass_associates_by_correlation",
    "test_reassociating_a_provisional_pass_is_idempotent",
    "test_reassociation_scores_a_pass_without_itself", "test_rule2_pairwise_sums",
    "test_rule2_restrictions[far_hz]", "test_rule2_restrictions[far_days]",
    "test_rule2_restrictions[cluster_far_days]", "test_one_transmission_is_one_member",
    "test_due_names_the_later_passes_of_a_picture",
]


def _run_scenario(name, root):
    """test_qrss_multipass's scenario on a fresh store (whatever Store and
    associate the modules hold), with every associate decision recorded."""
    fn_name, _, param = name.partition("[")
    fn = getattr(tm, fn_name)
    decisions = []
    real = asc.associate
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(asc, "associate", lambda p, store: decisions.append((p.uid, real(p, store))) or
                   decisions[-1][1])
        st = S.Store(root, clock=tm.Clock())
        kw = {}
        params = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        if "monkeypatch" in params:
            kw["monkeypatch"] = mp
        if "case" in params:
            kw["case"] = param.rstrip("]")
        fn(st, **kw)
    finally:
        mp.undo()
    return decisions


@pytest.mark.parametrize("name", _SCENARIOS)
def test_multipass_scenarios(native, NStore, name, tmp_path, monkeypatch):
    """Each scenario passes on the reference and on the C++ (store and
    associate), with the same decisions and the same files."""
    ref = _run_scenario(name, tmp_path / "py" / "qrss")
    with monkeypatch.context() as mp:
        H.apply_store_rows(native, mp)
        for k in ("Store", "canonical_index", "key_name"):          # the test module's own imports
            mp.setattr(tm, k, getattr(S, k))
        assert S.Store is NStore
        got = _run_scenario(name, tmp_path / "cpp" / "qrss")
    assert got == ref
    H.compare_trees(tmp_path / "py" / "qrss", tmp_path / "cpp" / "qrss")


# --- the fuzz ---------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _fuzz_passes(seed, n):
    rng = random.Random(seed)
    pics = [synthetic_picture(200 + 5 * seed + i, mode=rng.randrange(3)) for i in range(3)]
    calls = ["K1ABC", "G4XYZ/P", "K1ABC"]
    out = []
    for i in range(n):
        k = rng.randrange(3)
        sp = pics[k]
        g = rng.randrange(sp.mode + 1)
        size = rng.choice([4096, 4096, 4096, 1024, 16384, 50600])
        a = picture.air_values(sp.segs[g], g)[:size]
        s = 10 ** rng.uniform(-1.6, 0.6)
        if rng.random() < 0.06:                    # an impostor: another picture's header
            o = pics[(k + 1) % 3]
            h = make_header(o, segment=0, callsign=calls[(k + 1) % 3])
        else:
            h = make_header(sp, segment=g, callsign=calls[k])
        u = rng.random()
        kw = (dict(hdr=h, decoded=True) if u < 0.25 else
              dict(hdr=h, hdr_es_n0_db=rng.uniform(-17.0, -12.0)) if u < 0.7 else {})
        q = Q_TEST + 2 * g + rng.choice([0, 0, 12, 24, 36])
        f = 1500.0 + 40.0 * k + rng.choice([0.0, 0.0, 1.5, -3.0, 7.0])
        waveform = WAVEFORM_L if size == 4096 and rng.random() < 0.15 else 0
        if waveform == WAVEFORM_L:
            f += 9.5
        erase = rng.choice([None, None, None, size // 3])
        p = make_pass_result(a, s, seed=1000 * seed + i, q=q, f_hz=f, erase_after=erase,
                             waveform=waveform, **kw)
        if size == 1024 and rng.random() < 0.5:
            p.hdr_llr = np.zeros(0, np.float32)    # a frame without a header keeps no LLRs
        out.append(p)
    return out


def _fuzz_run(store, assoc, resolve, passes, clock):
    out = []
    for i, p in enumerate(passes):
        clock.t += 7.0
        out.append(_outcome(assoc, p, store))
        if i % 10 == 9:
            out.append(_outcome(resolve, store))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2] + [pytest.param(s, marks=pytest.mark.slow) for s in (3, 4, 5)])
def test_associate_fuzz(row, NStore, seed, tmp_path):
    """50 synthetic passes per session (module docstring): the same
    decisions, resolutions and files."""
    passes = _fuzz_passes(seed, 50)
    c1, c2 = tm.Clock(), tm.Clock()
    ref = _fuzz_run(S.Store(tmp_path / "py", clock=c1), asc.associate, asc.resolve_provisional,
                    passes, c1)
    got = _fuzz_run(NStore(tmp_path / "cpp", clock=c2), row["associate"], row["resolve_provisional"],
                    passes, c2)
    assert got == ref
    H.compare_trees(tmp_path / "py", tmp_path / "cpp")
    rules = {r[1][1][1] for r in ref if r[0] == "ok" and isinstance(r[1], tuple) and len(r[1]) == 2
             and isinstance(r[1][1], tuple)}
    print(f"seed {seed}: rules {sorted(rules)}")


# --- the whole caller: LiveListener -------------------------------------------------------------------

def _quiet(p):
    """A headerless pass weak enough for a tile (live.HEADERLESS_MAX_SNR_DB)."""
    p.report = dataclasses.replace(p.report, snr2500_db=-12.0)
    return p


def _live_run(tmp):
    from sstvae.qrss import frame, live
    from sstvae.qrss.frontend import slot_t0

    TINY = frame.TINY
    sp, other = synthetic_picture(51, mode=1), synthetic_picture(52)
    h = make_header(sp, callsign="AG7EW")
    a, b = (picture.air_values(x.segs[0], 0)[:TINY.n_data] for x in (sp, other))
    slots = [
        (Q_TEST, [_quiet(make_pass_result(a, 0.6, seed=1, q=Q_TEST, f_hz=1500.0)),
                  _quiet(make_pass_result(b, 0.6, seed=2, q=Q_TEST, f_hz=1700.0))]),
        (Q_TEST + 2, [_quiet(make_pass_result(a, 0.6, seed=3, q=Q_TEST + 2, f_hz=1500.4))]),
        (Q_TEST + 4, [make_pass_result(a, 2.0, seed=4, q=Q_TEST + 4, f_hz=1500.2, hdr=h, decoded=True),
                      _quiet(make_pass_result(b, 0.6, seed=5, q=Q_TEST + 4, f_hz=1700.3))]),
        (Q_TEST + 6, [_quiet(make_pass_result(a, 0.6, seed=6, q=Q_TEST + 6, f_hz=1499.8)),
                      make_pass_result(a, 2.0, seed=7, q=Q_TEST + 6, f_hz=1500.1, hdr=h,
                                       decoded=True)]),
    ]
    logs = []
    st = S.Store(tmp / "store", clock=tm.Clock())
    L = live.LiveListener(tmp / "live", live.LiveConfig(spec=TINY, render=False), store=st,
                          log=logs.append)
    out = []
    for q, passes in slots:
        L.now = float(slot_t0(q + 1))
        s = live.SlotState(q)
        L._finish_apply(s, [(None, passes, 1.0)])
        L.done_tiles.extend(s.tiles.values())
        out.append([(t.f_hz, t.status, t.note, t.uid, t.callsign, t.picture_id, t.mode, t.segment,
                     t.passes, t.mean_w_db) for t in L.done_tiles])
        out.append(L._hints(q + 2, None))
    return out, logs


def test_live_listener_whole_caller(native, NStore, tmp_path, monkeypatch):
    """LiveListener._finish_apply on the C++ store and associate, every row
    of this lane in place with counting spies before them: the reference's
    tiles, hints, log and files."""
    ref, ref_log = _live_run(tmp_path / "py")
    used = {}
    with monkeypatch.context() as mp:
        H.apply_store_rows(native, mp, wrap=H.counting(used))
        got, got_log = _live_run(tmp_path / "cpp")
    assert got == ref and got_log == ref_log
    notes = {t[2] for t in ref[-2]}
    assert {"stored (provisional)", "stored (header)", "stored (corr)"} <= notes, notes
    assert used.get("Store", 0) >= 1 and used.get("associate", 0) >= 7, used
    H.compare_trees(tmp_path / "py" / "store", tmp_path / "cpp" / "store")
