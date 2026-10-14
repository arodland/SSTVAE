"""The passband store's FLAC encoder (native/core/qrss/flac, as
sstvae_native.qrss.flac) against the decoders other programs use.

Nothing in the reference has these names: soundfile packs the Python
listener's hours, and the two encoders are not byte-identical (nor meant
to be). The bar is a FLAC file every decoder reads, about as small as
libFLAC's:

- every stream decodes sample for sample through libFLAC (soundfile),
  dr_flac (the C++ reader) and, where installed, ffmpeg's FLAC decoder and
  `flac -t` (the reference decoder's own test mode; not installed on the
  machine this was written on, so that test skips there);
- STREAMINFO holds the MD5 hashlib computes of the samples, the stream's
  real minimum and maximum frame sizes (each frame found through a seek
  point per frame, starting with a frame sync), the frame count and the
  block size; every seek point names its frame's first sample, offset and
  length;
- at the defaults the stream is a FLAC subset one (LPC order <= 12, by
  the encoder's own count), LPC is what codes passband audio, and the
  file is no larger than soundfile's (libFLAC level 5) on it;
- the fuzz: lengths (nearly always a partial last block), mono and
  stereo, silence, full-scale squares and impulses, white noise at every
  level, band-limited noise, sines, bursts in silence and passband audio,
  every encoder option drawn at random;
- refusals: ValueError for every option out of range.

native/tests/qrss/test_flac_lpc.cpp holds what needs the encoder's
insides: a strict decoder of its own (every field, 32-bit prediction),
every LPC order and precision, the residual kernel against a 64-bit
reference.

Measured (2026-10-14, this machine; not asserted): on eight simulated
passband hours (channel.py transmissions over band noise through
live.StreamFE, levels 0.01-0.3, two through an SSB filter, one heard for
12% of the hour) the C++ files are 0.898-0.968x the size of soundfile's
(the FIXED-only encoder before LPC: 1.22-1.55x), for 0.41-1.51 s of one
core per hour (RelWithDebInfo) against soundfile's 0.36-0.76 s of wall
time. The synthetic stretches below measure alike: 0.90-0.97x, FIXED
alone 1.18-1.32x.
"""

import hashlib
import io
import shutil
import subprocess

import numpy as np
import pytest
from qrss_native import require
from scipy import signal

from sstvae.qrss import live
from sstvae.qrss.constants import FE_FS, FS


@pytest.fixture(scope="module")
def F(native):
    return require(native, "flac")


def _sf():
    return pytest.importorskip("soundfile")


def _md5(q):
    return hashlib.md5(np.ascontiguousarray(q).astype("<i2").tobytes()).digest()


def _meta(b: bytes):
    """STREAMINFO's fields, the seek points and where the frames start."""
    assert b[:4] == b"fLaC"
    pos, last, info, seek = 4, False, None, []
    while not last:
        last = bool(b[pos] & 0x80)
        typ = b[pos] & 0x7F
        n = int.from_bytes(b[pos + 1:pos + 4], "big")
        body = b[pos + 4:pos + 4 + n]
        assert len(body) == n
        if typ == 0:
            packed = int.from_bytes(body[10:18], "big")
            info = dict(min_block=int.from_bytes(body[0:2], "big"), max_block=int.from_bytes(body[2:4], "big"),
                        min_frame=int.from_bytes(body[4:7], "big"), max_frame=int.from_bytes(body[7:10], "big"),
                        fs=packed >> 44, channels=((packed >> 41) & 7) + 1, bps=((packed >> 36) & 31) + 1,
                        total=packed & ((1 << 36) - 1), md5=body[18:34])
        elif typ == 3:
            seek = [(int.from_bytes(body[k:k + 8], "big"), int.from_bytes(body[k + 8:k + 16], "big"),
                     int.from_bytes(body[k + 16:k + 18], "big")) for k in range(0, n, 18)]
        else:
            raise AssertionError(f"metadata block type {typ}")
        pos += 4 + n
    return info, seek, pos


def _ffmpeg(data: bytes, channels: int):
    out = subprocess.run(["ffmpeg", "-v", "error", "-f", "flac", "-i", "pipe:0", "-f", "s16le", "-acodec",
                          "pcm_s16le", "pipe:1"], input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i2").reshape(-1, channels)


def _check(F, q, fs=FE_FS, ffmpeg=False, **opts):
    """q through the encoder and back through libFLAC and dr_flac (and
    ffmpeg); STREAMINFO, seek points and stats checked. Returns (bytes,
    stats)."""
    sf = _sf()
    data, st = F.encode(q, fs, **opts)
    seek_every = opts.get("seek_every", F.DEFAULTS["seek_every"])
    bs = opts.get("blocksize", F.DEFAULTS["blocksize"])
    what = (q.shape, opts)
    y, f = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
    assert f == fs and np.array_equal(y, q), ("libFLAC", what)
    z, info = F.decode(data)
    assert np.array_equal(z, q), ("dr_flac", what)
    if ffmpeg and shutil.which("ffmpeg"):
        assert np.array_equal(_ffmpeg(data, q.shape[1]), q), ("ffmpeg", what)
    meta, seek, first = _meta(data)
    assert meta["md5"] == _md5(q) == st["md5"] == info["md5"], what
    assert (meta["fs"], meta["channels"], meta["bps"], meta["total"]) == (fs, q.shape[1], 16, len(q)), what
    assert meta["min_block"] == meta["max_block"] == bs, what
    nframes = -(-len(q) // bs)
    assert st["frames"] == nframes and st["bytes"] == len(data), what
    assert len(seek) == (-(-nframes // seek_every) if seek_every else 0), what
    for k, (sample, offset, n) in enumerate(seek):
        f0 = k * seek_every
        assert sample == f0 * bs and n == min(bs, len(q) - f0 * bs), what
        assert data[first + offset:first + offset + 2] == b"\xff\xf8", what      # a frame sync
    if seek_every == 1 and nframes:
        ends = [s[1] for s in seek] + [len(data) - first]
        sizes = np.diff(ends)
        assert (meta["min_frame"], meta["max_frame"]) == (sizes.min(), sizes.max()), what
    assert sum(st["subframes"]) == nframes * q.shape[1], what
    assert sum(st["lpc_orders"]) == st["subframes"][7], what
    assert not any(st["lpc_orders"][opts.get("max_lpc_order", F.DEFAULTS["max_lpc_order"]) + 1:]), what
    return data, st


# --- signals ----------------------------------------------------------------------------------------

def _passband(seconds, level, ssb=False, carrier_hz=0.0, seed=0):
    """Receiver hiss at an audio level (rms, full scale 1.0), through an SSB
    filter or not, with a steady carrier or not, through live.StreamFE at
    the store's scale (8192 counts per unit): what a packed hour holds."""
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(int(seconds * FS))
    if ssb:
        audio = signal.sosfilt(signal.butter(6, (300.0, 2700.0), btype="bandpass", fs=FS, output="sos"), audio)
    if carrier_hz:
        audio += 0.3 * np.sin(2 * np.pi * carrier_hz * np.arange(len(audio)) / FS)
    audio *= level / np.sqrt(np.mean(audio ** 2))
    sfe = live.StreamFE()
    fe = np.concatenate([y for a in range(0, len(audio), FS) for _, y in sfe.push(a, audio[a:a + FS])])
    return np.stack([np.clip(np.rint(fe.real * 8192.0), -32767, 32767),
                     np.clip(np.rint(fe.imag * 8192.0), -32767, 32767)], 1).astype(np.int16)


def _resonant(rng, n, f, r, rms):
    """White noise through a two-pole resonator at f cycles per sample."""
    x = signal.lfilter([1.0], [1.0, -2 * r * np.cos(2 * np.pi * f), r * r], rng.standard_normal(n + 1000))[1000:]
    return x * (rms / max(np.sqrt(np.mean(x ** 2)), 1e-300)) if n else x


def _content(rng, n, kind):
    if kind == "band":
        x = _resonant(rng, n, rng.uniform(0, 0.5), rng.uniform(0.5, 0.9999), rng.uniform(1, 40000))
    elif kind == "sine":
        x = rng.uniform(0, 33000) * np.sin(rng.uniform(0, 3) * np.arange(n)) + rng.normal(0, rng.uniform(0, 100), n)
    elif kind == "white":
        x = rng.uniform(-rng.uniform(0, 33000), rng.uniform(0, 33000), n)
    elif kind == "square":
        per = int(rng.integers(1, 300))
        x = np.where((np.arange(n) // per) % 2, 32767.0, -32768.0)
        spikes = rng.random(n) < 0.02
        x[spikes] = rng.choice([32767.0, -32768.0], spikes.sum())
    elif kind == "impulses":
        x = np.zeros(n)
        k = rng.random(n) < 0.003
        x[k] = rng.choice([32767.0, -32768.0, 1.0], k.sum())
    elif kind == "silence":
        x = np.zeros(n)
    elif kind == "burst":
        x = _resonant(rng, n, rng.uniform(0, 0.5), rng.uniform(0.8, 0.999), rng.uniform(10, 5000))
        a, b = sorted(rng.integers(0, n + 1, 2))
        x[:a] = 0
        x[b:] = 0
    else:
        raise ValueError(kind)
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


KINDS = ("band", "sine", "white", "square", "impulses", "silence", "burst", "passband")


def _case(rng, F, defaults):
    opts = {}
    if not defaults:
        opts = dict(blocksize=int(rng.choice([16, 192, 576, 1152, 2048, 4096, 4608, 1000])),
                    max_fixed_order=int(rng.integers(0, 5)), max_lpc_order=int(rng.integers(0, 33)),
                    qlp_precision=int(rng.choice([0, int(rng.integers(5, 16))])),
                    lpc_orders_tried=int(rng.integers(1, 34)), partial_tukey=bool(rng.random() < 0.5),
                    max_partition_order=int(rng.integers(0, 9)), stereo=bool(rng.random() < 0.75),
                    seek_every=int(rng.choice([0, 1, 1, 3])))
    bs = opts.get("blocksize", F.DEFAULTS["blocksize"])
    n = int(rng.integers(1, 3 * bs + 1000))
    ch = int(rng.integers(1, 3))
    kind = KINDS[int(rng.integers(0, len(KINDS)))]
    if kind == "passband":
        q = _passband(n / FE_FS + 1, rng.uniform(0.003, 0.5), ssb=bool(rng.random() < 0.5),
                      seed=int(rng.integers(1 << 30)))[:n, :ch]
        q = np.ascontiguousarray(q)
    else:
        q = np.stack([_content(rng, n, kind) for _ in range(ch)], 1)
        if ch == 2 and rng.random() < 0.3:      # correlated channels: the stereo modes get used
            q[:, 1] = np.clip(q[:, 0].astype(np.int32) + _content(rng, n, "band") // 64, -32768, 32767)
    return q, opts, kind


# --- tests ------------------------------------------------------------------------------------------

def test_defaults(F):
    assert F.DEFAULTS == dict(blocksize=4096, max_fixed_order=4, max_lpc_order=12, qlp_precision=0,
                              lpc_orders_tried=3, partial_tukey=True, max_partition_order=6, stereo=True,
                              seek_every=16)
    q = np.zeros((100, 2), np.int16)
    assert F.encode(q, 4000, **F.DEFAULTS)[0] == F.encode(q, 4000)[0]


@pytest.mark.parametrize("level,ssb,carrier_hz", [(0.3, False, 0.0), (0.03, False, 1500.0), (0.1, True, 0.0),
                                                  (0.01, False, 0.0)])
def test_passband_audio(F, level, ssb, carrier_hz):
    """What the store packs: decoded exactly by everything, coded by LPC, no
    larger than libFLAC's file and well under FIXED's."""
    sf = _sf()
    q = _passband(60.0, level, ssb, carrier_hz, seed=int(level * 1000))
    data, st = _check(F, q, ffmpeg=True, seek_every=1)
    assert st["subframes"][7] >= 0.9 * 2 * st["frames"]
    ref = io.BytesIO()
    sf.write(ref, q, FE_FS, format="FLAC", subtype="PCM_16")
    assert len(data) <= len(ref.getvalue())             # (measured 0.90-0.97x)
    fixed, _ = F.encode(q, FE_FS, max_lpc_order=0)
    assert len(data) < 0.9 * len(fixed)                  # (measured 0.73-0.77x)
    # orders above libFLAC level 5's 8 pay on it
    assert len(_check(F, q, max_lpc_order=8)[0]) > len(data)
    # the windows, the orders tried and the stereo modes move the size a little
    for kw in (dict(partial_tukey=False), dict(lpc_orders_tried=1), dict(lpc_orders_tried=32), dict(stereo=False)):
        other, _ = _check(F, q, **kw)
        assert 0.97 < len(other) / len(data) < 1.03, kw


def test_encode_file_and_partial_reads(F, tmp_path):
    q = _passband(10.0, 0.1, seed=3)
    p = tmp_path / "x.flac"
    st = F.encode_file(p, q, FE_FS)
    data, st2 = F.encode(q, FE_FS)
    assert p.read_bytes() == data and st == st2
    rng = np.random.default_rng(4)
    assert np.array_equal(F.read(p), q)
    for _ in range(20):
        a = int(rng.integers(0, len(q) + 2))
        b = int(rng.integers(0, len(q) + 3))
        assert np.array_equal(F.read(p, a, b), q[a:b]), (a, b)


def _fuzz(F, seed, count):
    rng = np.random.default_rng(seed)
    kinds = set()
    for i in range(count):
        q, opts, kind = _case(rng, F, defaults=i % 3 == 0)
        kinds.add(kind)
        _check(F, q, ffmpeg=i % 7 == 0, **opts)
    return kinds


@pytest.mark.parametrize("seed", range(6))
def test_fuzz(F, seed):
    _fuzz(F, seed, 30)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(100, 110))
def test_fuzz_long(F, seed):
    assert len(_fuzz(F, seed, 100)) == len(KINDS)


def test_extremes(F):
    """Full scale, impulses, white noise and silence, mono and stereo, at
    the defaults and at the options' far ends (order 32, precision 15 and
    5, every order tried)."""
    rng = np.random.default_rng(5)
    n = 2 * 4096 + 777
    t = np.arange(n)
    sq = np.where((t // 29) % 2, 32767, -32768).astype(np.int16)
    sigs = {
        "square opposite": np.stack([sq, ~sq], 1),       # (~x = -1 - x: the side channel's widest)
        "square same": np.stack([sq, sq], 1),
        "alternation": np.stack([np.where(t % 2, 32767, -32768), np.where(t % 3, -32768, 32767)], 1),
        "impulses": np.stack([np.where(t % 997 == 13, -32768, 0), np.where(t % 1500 == 700, 32767, 0)], 1),
        "white": rng.integers(-32768, 32768, (n, 2)),
        "silence": np.zeros((n, 2)),
        "clipped band": np.stack([_content(rng, n, "band") for _ in range(2)], 1),
    }
    far = [{}, dict(max_lpc_order=32, qlp_precision=15, lpc_orders_tried=32, max_partition_order=8),
           dict(max_lpc_order=32, qlp_precision=5, blocksize=4608, seek_every=1),
           dict(blocksize=33, max_lpc_order=32, stereo=False)]
    for name, x in sigs.items():
        q = np.ascontiguousarray(np.asarray(x).astype(np.int16))
        for opts in far:
            data, st = _check(F, q, **opts)
            if name == "silence":
                assert st["subframes"][0] == 2 * st["frames"]
            if name == "white":
                assert len(data) <= q.nbytes + 64 * st["frames"] + 4096
            _check(F, np.ascontiguousarray(q[:, :1]), **opts)


@pytest.mark.skipif(shutil.which("flac") is None, reason="the flac command-line tool is not installed")
def test_flac_cli(F, tmp_path):
    """`flac -t`: the reference decoder's own check of every frame's CRC
    and the STREAMINFO MD5."""
    rng = np.random.default_rng(6)
    for i in range(12):
        q, opts, _ = _case(rng, F, defaults=i % 2 == 0)
        p = tmp_path / f"{i}.flac"
        p.write_bytes(F.encode(q, FE_FS, **opts)[0])
        subprocess.run(["flac", "-t", "--totally-silent", str(p)], check=True)


def test_refusals(F):
    q = np.zeros((100, 2), np.int16)
    for kw in (dict(max_lpc_order=33), dict(max_lpc_order=-1), dict(qlp_precision=4), dict(qlp_precision=16),
               dict(qlp_precision=-1), dict(lpc_orders_tried=0), dict(max_partition_order=9),
               dict(max_partition_order=-1), dict(max_fixed_order=5), dict(blocksize=15), dict(blocksize=65536)):
        with pytest.raises(ValueError):
            F.encode(q, 4000, **kw)
    with pytest.raises(ValueError):
        F.encode(np.zeros((10, 3), np.int16), 4000)
    with pytest.raises(ValueError):
        F.encode(np.zeros(10, np.int16), 4000)
    for kw in (dict(max_lpc_order=0), dict(lpc_orders_tried=1000), dict(qlp_precision=5),
               dict(max_partition_order=8)):
        F.encode(q, 4000, **kw)
