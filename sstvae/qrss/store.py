"""The multi-pass store: per-pass files, per-picture accumulators,
provisional clusters, expiry and merging (design 8.1, spec section 8).

Not a format module: nothing here goes on the air, and two receivers
may lay out their stores differently. What *is* a contract is the
accumulator's meaning, because accumulators from different receivers
add (spec 8, "Mergeable accumulators"):

- one accumulator per (callsign, picture ID), canonical [3, 52800];
- **S = S_ext + sum over members of w*z** and **W = W_ext + sum of w**,
  where z, w are each member pass's latents and weights placed at their
  segment's canonical offsets (`picture.air_to_canonical`) and S_ext,
  W_ext are what was merged in from other receivers;
- W is therefore the effective per-latent SNR, linear, and S/W the
  accumulated unbiased latent.

`Accumulator.rebuild` recomputes S and W from the members, in float64
and in member order, and every attach, detach or pass update calls it,
so the invariant holds by construction rather than by careful
incremental bookkeeping (it takes milliseconds at this size).

Layout under the root (default `$QRSSTVAE_HOME`, else
`~/.local/share/qrsstvae`):

    passes/<uid>.npz           PassResult.save()
    acc/<CALL>_<pid:08x>.json  key, mode, codec_id, created, updated, members, foreign
    acc/<CALL>_<pid:08x>.npz   S, W, S_ext, W_ext  float32[3, 52800], plus each
                               foreign merge's own (Sx_<sha>, Wx_<sha>)
    prov/<cluster>.json        a provisional cluster of unassociated pass uids
    log.jsonl                  association events (collisions, merges)

Every write goes to a temporary file in the same directory and is moved
into place with `os.replace`, so a crash leaves either the old file or
the new one, never half of either. `/` in a callsign becomes `_` in file
names (QRSS callsigns are `[A-Z0-9/]`, so the mapping cannot collide);
the JSON carries the real callsign.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import picture
from .types import PassResult

N_GROUPS = picture.N_GROUPS                 # 3
GROUP_LATENTS = picture.GROUP_LATENTS       # 52,800
SENT = picture.SENT                         # 50,600
DAY_S = 86400.0
SLOT_S = 900                                # a slot's q counts quarter hours
ZW_CACHE = 128                              # passes' (z, w) kept in memory, ~0.8 MB each


def default_root() -> Path:
    """$QRSSTVAE_HOME, else ~/.local/share/qrsstvae."""
    env = os.environ.get("QRSSTVAE_HOME")
    return Path(env) if env else Path.home() / ".local" / "share" / "qrsstvae"


def key_name(key) -> str:
    """File stem of an accumulator: "<CALL>_<pid:08x>", '/' -> '_'."""
    call, pid = key
    return f"{call.replace('/', '_')}_{int(pid):08x}"


def slot_time(q: int) -> float:
    """Unix time of slot q's quarter hour."""
    return float(int(q) * SLOT_S)


def mean_w_db(w) -> float:
    """10 log10 of a pass's mean weight (its mean per-latent SNR); -99 if none."""
    m = float(np.mean(np.asarray(w, dtype=np.float64))) if np.size(w) else 0.0
    return float(10 * np.log10(m)) if m > 0 else -99.0


def canonical_index(segment: int, n: int) -> np.ndarray:
    """Canonical offsets (within the group) of a pass's n air-order latents.

    A full pass carries all 50,600; the shorter test frames carry a
    prefix of the same air order.
    """
    if not 0 < n <= SENT:
        raise ValueError(f"a pass carries 1..{SENT} latents, not {n}")
    return picture.air_to_canonical(segment)[:n]


# --- atomic files ----------------------------------------------------------------


def _atomic_write(path: Path, write) -> None:
    """Call write(file) on a temporary sibling, then os.replace it into place."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_save(path: Path, save) -> None:
    """Like _atomic_write, for a saver that takes a path (PassResult.save)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, obj) -> None:
    data = json.dumps(obj, indent=1, sort_keys=True).encode()
    _atomic_write(path, lambda f: f.write(data))


def _read_json(path: Path):
    with open(path, "rb") as f:
        return json.loads(f.read().decode())


# --- the accumulator -------------------------------------------------------------


@dataclass
class Accumulator:
    """One picture's accumulated latents, canonical [3, 52800] float32."""
    key: tuple                                  # (callsign, picture_id)
    mode: int
    codec_id: int
    S: np.ndarray = field(repr=False)
    W: np.ndarray = field(repr=False)
    S_ext: np.ndarray = field(repr=False)
    W_ext: np.ndarray = field(repr=False)
    members: list = field(default_factory=list)  # dicts, see Store.attach
    foreign: list = field(default_factory=list)  # dicts, see Store.merge
    parts: dict = field(default_factory=dict, repr=False)  # sha -> foreign (S, W)
    created: float = 0.0
    updated: float = 0.0

    @classmethod
    def empty(cls, key, mode: int, codec_id: int, now: float) -> Accumulator:
        z = lambda: np.zeros((N_GROUPS, GROUP_LATENTS), dtype=np.float32)  # noqa: E731
        return cls((str(key[0]), int(key[1])), int(mode), int(codec_id),
                   z(), z(), z(), z(), [], [], {}, now, now)

    @property
    def uids(self) -> list[str]:
        return [m["uid"] for m in self.members]

    def member(self, uid: str) -> dict | None:
        return next((m for m in self.members if m["uid"] == uid), None)

    def _sums(self, store: Store, skip: str | None = None, ext: bool = True):
        """float64 (S, W): S_ext/W_ext (if `ext`) plus every member but `skip`, in order."""
        S = self.S_ext.astype(np.float64) if ext else np.zeros((N_GROUPS, GROUP_LATENTS))
        W = self.W_ext.astype(np.float64) if ext else np.zeros((N_GROUPS, GROUP_LATENTS))
        for m in self.members:
            if m["uid"] == skip:
                continue
            z, w = store.pass_zw(m["uid"])
            idx = canonical_index(m["segment"], z.size)
            g = int(m["segment"])
            w = w.astype(np.float64)
            S[g, idx] += np.where(w > 0, w * z, 0.0)
            W[g, idx] += w
        return S, W

    def rebuild(self, store: Store) -> None:
        """S = S_ext + sum w*z, W = W_ext + sum w over the members (float64, then float32)."""
        S, W = self._sums(store)
        self.S = S.astype(np.float32)
        self.W = W.astype(np.float32)

    def loo(self, uid: str, store: Store) -> tuple[np.ndarray, np.ndarray]:
        """Leave-one-out (S, W) for member `uid`, float64, in that pass's air order.

        Recomputed from the other members rather than subtracted, so a
        pass's own contribution cannot survive in it through rounding:
        the leave-one-out reference never contains pass i (spec 8, EM).
        """
        m = self.member(uid)
        if m is None:
            raise KeyError(f"{uid} is not a member of {key_name(self.key)}")
        S, W = self._sums(store, skip=uid)
        z, _ = store.pass_zw(uid)
        idx = canonical_index(m["segment"], z.size)
        g = int(m["segment"])
        return S[g, idx], W[g, idx]

    def has_data(self, g: int) -> bool:
        return bool(np.any(self.W[g] > 0))

    def median_w(self, g: int) -> float:
        """Median W over group g's 50,600 sent latents (linear)."""
        return float(np.median(self.W[g, picture.air_to_canonical(g)]))

    def seg_snr_db(self) -> list[float | None]:
        """Effective SNR per segment, 10 log10 median W (spec 11); None without data."""
        out = []
        for g in range(self.mode + 1):
            med = self.median_w(g) if self.has_data(g) else 0.0
            out.append(float(10 * np.log10(med)) if med > 0 else None)
        return out

    def meta(self) -> dict:
        return {"callsign": self.key[0], "picture_id": int(self.key[1]),
                "mode": self.mode, "codec_id": self.codec_id,
                "created": self.created, "updated": self.updated,
                "members": self.members, "foreign": self.foreign}


# --- the store -------------------------------------------------------------------


class Store:
    """The on-disk multi-pass store (see the module docstring).

    `clock` is the time source (unix seconds) for created/updated stamps
    and for `expire`; tests pass a fake one.
    """

    def __init__(self, root=None, clock=time.time):
        self.root = Path(root) if root is not None else default_root()
        self.clock = clock
        for d in ("passes", "acc", "prov"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self._zw: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._light: dict[str, dict] = {}

    # --- passes ---

    def _pass_path(self, uid: str) -> Path:
        if "/" in uid or uid.startswith("."):
            raise ValueError(f"bad pass uid {uid!r}")
        return self.root / "passes" / f"{uid}.npz"

    def add_pass(self, p: PassResult) -> None:
        """Write (or overwrite) a pass file."""
        _atomic_save(self._pass_path(p.uid), p.save)
        self._zw.pop(p.uid, None)
        self._light.pop(p.uid, None)

    def has_pass(self, uid: str) -> bool:
        return self._pass_path(uid).exists()

    def load_pass(self, uid: str) -> PassResult:
        return PassResult.load(self._pass_path(uid))

    def pass_uids(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "passes").glob("*.npz"))

    def pass_zw(self, uid: str) -> tuple[np.ndarray, np.ndarray]:
        """A pass's (z, w), float64: what accumulators are built from.

        The ZW_CACHE most recently used passes are kept in memory; older
        ones are read from disk again when next needed.
        """
        if uid in self._zw:
            self._zw.move_to_end(uid)
        else:
            with np.load(self._pass_path(uid), allow_pickle=False) as d:
                z = d["z"].astype(np.float64)
                w = d["w"].astype(np.float64)
            if z.shape != w.shape or z.ndim != 1:
                raise ValueError(f"pass {uid}: z{z.shape} and w{w.shape} do not match")
            self._zw[uid] = (z, w)
            while len(self._zw) > ZW_CACHE:
                self._zw.popitem(last=False)
        return self._zw[uid]

    def pass_info(self, uid: str) -> dict:
        """A pass's light metadata: q, f_hz, header (dict or None), hdr_llr, z_ref."""
        if uid not in self._light:
            with np.load(self._pass_path(uid), allow_pickle=False) as d:
                meta = json.loads(d["meta"].tobytes().decode())
                llr = d["hdr_llr"].astype(np.float64)
            self._light[uid] = {"uid": uid, "q": int(meta["q"]),
                                "f_hz": float(meta["f_hz"]),
                                "header": meta["header"], "hdr_llr": llr,
                                "z_ref": float(meta["report"]["z_ref"]),
                                "em_round": int(meta["em_round"])}
        return self._light[uid]

    def update_pass(self, p: PassResult) -> list[tuple]:
        """Replace a stored pass (EM's new z, w) and rebuild every accumulator holding it.

        Returns the keys rebuilt.
        """
        self.add_pass(p)
        keys = []
        for key in self.open_keys():
            acc = self.accumulator(key)
            m = acc.member(p.uid)
            if m is None:
                continue
            m["em_round"] = int(p.em_round)
            m["mean_w_db"] = mean_w_db(p.w)
            self._commit(acc)
            keys.append(key)
        return keys

    # --- accumulators ---

    def _acc_paths(self, key) -> tuple[Path, Path]:
        stem = self.root / "acc" / key_name(key)
        return stem.with_suffix(".json"), stem.with_suffix(".npz")

    def has_accumulator(self, key) -> bool:
        return self._acc_paths(key)[0].exists()

    def accumulator(self, key) -> Accumulator:
        """Load an accumulator; KeyError if there is none for `key`."""
        jpath, npath = self._acc_paths(key)
        if not jpath.exists():
            raise KeyError(f"no accumulator {key_name(key)}")
        meta = _read_json(jpath)
        with np.load(npath, allow_pickle=False) as d:
            arrays = {k: d[k].astype(np.float32) for k in ("S", "W", "S_ext", "W_ext")}
            parts = {f["sha"]: (d[f"Sx_{f['sha']}"], d[f"Wx_{f['sha']}"])
                     for f in meta["foreign"] if f"Sx_{f['sha']}" in d.files}
        return Accumulator(key=(meta["callsign"], int(meta["picture_id"])),
                           mode=int(meta["mode"]), codec_id=int(meta["codec_id"]),
                           members=meta["members"], foreign=meta["foreign"], parts=parts,
                           created=float(meta["created"]), updated=float(meta["updated"]),
                           **arrays)

    def create(self, key, mode: int, codec_id: int) -> Accumulator:
        """A new, empty accumulator (saved), or the existing one for `key`."""
        if self.has_accumulator(key):
            return self.accumulator(key)
        acc = Accumulator.empty(key, mode, codec_id, self.clock())
        self.save_accumulator(acc)
        return acc

    def save_accumulator(self, acc: Accumulator) -> None:
        """Write the arrays first, then the JSON index (each atomically)."""
        jpath, npath = self._acc_paths(acc.key)

        def write_npz(f):
            buf = io.BytesIO()
            extra = {}
            for sha, (Sx, Wx) in acc.parts.items():
                extra[f"Sx_{sha}"], extra[f"Wx_{sha}"] = Sx, Wx
            np.savez(buf, S=acc.S, W=acc.W, S_ext=acc.S_ext, W_ext=acc.W_ext, **extra)
            f.write(buf.getvalue())
        _atomic_write(npath, write_npz)
        _write_json(jpath, acc.meta())

    def _commit(self, acc: Accumulator) -> None:
        acc.rebuild(self)
        acc.updated = self.clock()
        self.save_accumulator(acc)

    def open_keys(self) -> list[tuple]:
        """Every accumulator's key, sorted by file name."""
        keys = []
        for j in sorted((self.root / "acc").glob("*.json")):
            meta = _read_json(j)
            keys.append((meta["callsign"], int(meta["picture_id"])))
        return keys

    def find_member(self, uid: str) -> tuple | None:
        """The key of the accumulator holding pass `uid`, or None."""
        for key in self.open_keys():
            if any(m["uid"] == uid for m in _read_json(self._acc_paths(key)[0])["members"]):
                return key
        return None

    def attach(self, key, uid: str, segment: int, method: str, *,
               mode: int | None = None, codec_id: int | None = None) -> Accumulator:
        """Make pass `uid` a member of `key` with the given segment, and rebuild.

        Creates the accumulator if needed: mode and codec ID come from the
        arguments, else from the pass's own header, else (mode = segment,
        codec 0). A pass belongs to one accumulator: attaching moves it
        out of any other one and out of the provisional clusters.
        """
        if not self.has_pass(uid):
            raise KeyError(f"no pass {uid}")
        segment = int(segment)
        if not self.has_accumulator(key):
            hdr = self.pass_info(uid)["header"]
            if mode is None:
                mode = hdr["mode"] if hdr else segment
            if codec_id is None:
                codec_id = hdr["codec_id"] if hdr else 0
            self.create(key, mode, codec_id)
        acc = self.accumulator(key)
        if not 0 <= segment <= acc.mode:
            raise ValueError(f"segment {segment} is not in mode {acc.mode}")
        other = self.find_member(uid)
        if other is not None and tuple(other) != tuple(acc.key):
            self.detach(other, uid)
        self.prov_remove(uid)
        info = self.pass_info(uid)
        _, w = self.pass_zw(uid)
        rec = {"uid": uid, "segment": segment, "method": str(method),
               "z_ref": info["z_ref"], "mean_w_db": mean_w_db(w),
               "em_round": info["em_round"]}
        old = acc.member(uid)
        if old is None:
            acc.members.append(rec)
        else:
            old.update(rec)
        self._commit(acc)
        return acc

    def detach(self, key, uid: str) -> Accumulator:
        """Remove pass `uid` from `key` and rebuild (the pass file is kept)."""
        acc = self.accumulator(key)
        acc.members = [m for m in acc.members if m["uid"] != uid]
        self._commit(acc)
        return acc

    def delete_accumulator(self, key) -> None:
        for p in self._acc_paths(key):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def merge(self, key, S, W, source: str = "") -> bool:
        """Add a foreign (S, W) from another receiver, if it verifies (spec 8).

        The foreign data must correlate with this receiver's own members
        by the association test (`associate.corr_accept`); then it goes
        into S_ext/W_ext and the accumulator is rebuilt, so S and W become
        exactly the sums. The same foreign arrays are never merged twice,
        and a later merge from the same (non-empty) `source` *replaces*
        that source's earlier one: a receiver sends its whole accumulator
        each time, so adding the new one would count its old passes
        twice. S_ext/W_ext are recomputed from the parts in float64.

        Not detectable here: a foreign accumulator that already contains
        this receiver's own passes (a server composite sent back). A
        server must leave out the receiving station's own contribution.
        """
        from .associate import corr_accept, corr_stat, expected_t

        S = np.asarray(S, dtype=np.float32).reshape(N_GROUPS, GROUP_LATENTS)
        W = np.asarray(W, dtype=np.float32).reshape(N_GROUPS, GROUP_LATENTS)
        if not np.all(np.isfinite(S)) or not np.all(np.isfinite(W)) or np.any(W < 0):
            return False
        acc = self.accumulator(key)
        digest = hashlib.sha256(S.tobytes() + W.tobytes()).hexdigest()[:16]
        if any(f.get("sha") == digest for f in acc.foreign):
            return False
        stale = [f for f in acc.foreign if source and f.get("source") == source]
        Sm, Wm = acc._sums(self, ext=False)
        Wf = W.astype(np.float64)
        mf = np.divide(S, Wf, out=np.zeros_like(Wf), where=Wf > 0)
        mm = np.divide(Sm, Wm, out=np.zeros_like(Wm), where=Wm > 0)
        t, M = corr_stat(mf.ravel(), Wf.ravel(), mm.ravel(), Wm.ravel())
        t_exp = expected_t(Wf.ravel(), Wm.ravel())
        ok = corr_accept(t, t_exp)
        self.log({"event": "merge", "key": list(acc.key), "source": source,
                  "t": t, "M": M, "t_expected": t_exp, "accepted": ok})
        if not ok:
            return False
        for f in stale:
            acc.foreign.remove(f)
            acc.parts.pop(f["sha"], None)
        acc.foreign.append({"source": source, "sha": digest, "t": t, "M": M,
                            "merged": self.clock(),
                            "replaced": [f["sha"] for f in stale]})
        acc.parts[digest] = (S, W)
        S_ext = np.zeros((N_GROUPS, GROUP_LATENTS))
        W_ext = np.zeros((N_GROUPS, GROUP_LATENTS))
        for f in acc.foreign:
            Sx, Wx = acc.parts[f["sha"]]
            S_ext += Sx
            W_ext += Wx
        acc.S_ext = S_ext.astype(np.float32)
        acc.W_ext = W_ext.astype(np.float32)
        self._commit(acc)
        return True

    # --- provisional clusters ---

    def _prov_path(self, cid: str) -> Path:
        return self.root / "prov" / f"{cid}.json"

    def prov_clusters(self) -> list[dict]:
        return [_read_json(p) for p in sorted((self.root / "prov").glob("*.json"))]

    def prov_cluster_of(self, uid: str) -> dict | None:
        return next((c for c in self.prov_clusters() if uid in c["members"]), None)

    def prov_add(self, uid: str, cluster: str | None = None) -> str:
        """Put pass `uid` in provisional cluster `cluster` (a new one if None).

        A no-op when `uid` is already in `cluster`.
        """
        if cluster is not None:
            c = _read_json(self._prov_path(cluster))
            if uid in c["members"]:
                return cluster
        self.prov_remove(uid)
        now = self.clock()
        if cluster is None:
            cluster = f"c_{uid}"
            c = {"cluster": cluster, "created": now, "members": []}
        else:
            c = _read_json(self._prov_path(cluster))
        c["members"].append(uid)
        c["updated"] = now
        _write_json(self._prov_path(cluster), c)
        return cluster

    def prov_remove(self, uid: str) -> None:
        """Take pass `uid` out of whatever cluster holds it; drop emptied clusters."""
        c = self.prov_cluster_of(uid)
        if c is None:
            return
        c["members"] = [u for u in c["members"] if u != uid]
        if c["members"]:
            c["updated"] = self.clock()
            _write_json(self._prov_path(c["cluster"]), c)
        else:
            self._prov_path(c["cluster"]).unlink()

    # --- housekeeping ---

    def log(self, event: dict) -> None:
        event = {"time": self.clock(), **event}
        with open(self.root / "log.jsonl", "a") as f:
            f.write(json.dumps(event, default=float) + "\n")

    def expire(self, now: float | None = None, days: float = 7.0) -> dict:
        """Drop accumulators and clusters not updated for `days`, and orphan passes.

        A pass file is removed once no surviving accumulator or cluster
        holds it and its slot is older than the cutoff. Returns what was
        removed: {"acc": [...keys], "prov": [...], "passes": [...]}.
        """
        now = self.clock() if now is None else float(now)
        cutoff = now - days * DAY_S
        removed = {"acc": [], "prov": [], "passes": []}
        held = set()
        for key in self.open_keys():
            acc = self.accumulator(key)
            if acc.updated < cutoff:
                self.delete_accumulator(key)
                removed["acc"].append(key)
            else:
                held.update(acc.uids)
        for c in self.prov_clusters():
            if c["updated"] < cutoff:
                self._prov_path(c["cluster"]).unlink()
                removed["prov"].append(c["cluster"])
            else:
                held.update(c["members"])
        for uid in self.pass_uids():
            if uid in held:
                continue
            if slot_time(self.pass_info(uid)["q"]) < cutoff:
                self._pass_path(uid).unlink()
                self._zw.pop(uid, None)
                self._light.pop(uid, None)
                removed["passes"].append(uid)
        return removed
