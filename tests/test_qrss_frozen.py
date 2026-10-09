"""QRSSTVAE on-air sequences are frozen, and the format modules cannot
reseed them (design 10.1: F3, F6).

The digests below were pinned from the first implementation of
`sstvae/qrss/sequences.py` and `morse.py`. They are the on-air format:
if one of these tests fails, the waveform changed, and every receiver in
the field would stop understanding every transmitter. The fix is never
"update the digest" unless the format change was deliberate and the
spec owner signed it off. The C reference (`qrss_beacon_c/`, test C1)
cross-checks the same sequences bit for bit.
"""

import ast
import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from qrss_helpers import Q_TEST
from sstvae.qrss import frame, morse, sequences

QRSS_DIR = Path(sequences.__file__).parent

# --- F3: pinned sequences ----------------------------------------------------------

PINNED = {
    "preamble": "41deb0c5a75cfd48a759f005837125a4f062fa857846553f323262758d680bdd",
    "references_3539": "715b6ad703783d0c23678a3400362778834b0c56fd82c53fe4083c9b58ad7f02",
    "scrambler_qtest_50600": "a6f366775e63b7acff30c0aa479daa1e0b3b05c024331fad7f8334286a58c557",
    "keying_K1ABC/P": "46c4fb4542fce939dffea24afa991d130d650e7bfca58a7e1735023750a83089",
}


def _sequences():
    return {
        "preamble": sequences.preamble_ce(),
        "references_3539": sequences.references_ce(frame.FULL.n_ref),
        "scrambler_qtest_50600": sequences.scrambler(Q_TEST, 50600),
        "keying_K1ABC/P": morse.keying_units("K1ABC/P"),
    }


@pytest.mark.parametrize("name", sorted(PINNED))
def test_sequence_digest_is_pinned(name):
    a = _sequences()[name]
    assert hashlib.sha256(a.tobytes()).hexdigest() == PINNED[name], (
        f"{name} changed: that is an on-air format change")


def test_sequence_lengths_and_dtypes():
    seqs = _sequences()
    assert frame.FULL.n_ref == 3539
    assert seqs["preamble"].shape == (660,) and seqs["preamble"].dtype == np.int8
    assert seqs["references_3539"].shape == (3539,)
    assert seqs["scrambler_qtest_50600"].shape == (50600,)
    assert seqs["keying_K1ABC/P"].shape == (192,) and seqs["keying_K1ABC/P"].dtype == np.uint8


@pytest.mark.parametrize("name", ["preamble", "references_3539", "scrambler_qtest_50600"])
def test_pm1_sequences_are_balanced(name):
    """Each +-1 sequence is +-1 only and balanced within 4 sigma."""
    a = _sequences()[name].astype(np.int64)
    assert set(np.unique(a)) <= {-1, 1}
    assert abs(a.sum()) <= 4 * np.sqrt(len(a))


def _sha_bits_by_hand(domain: bytes, n: int, prefix: bytes = b"") -> list[int]:
    """Design 2.3's definition, bit by bit, with no numpy."""
    out = []
    for i in range(n):
        d = hashlib.sha256(domain + prefix + struct.pack(">I", i >> 8)).digest()
        out.append((d[(i & 255) >> 3] >> (7 - (i & 7))) & 1)
    return out


@pytest.mark.parametrize("n", [0, 1, 7, 255, 256, 257, 700])
def test_sha_bits_matches_the_written_definition(n):
    got = sequences.sha_bits(b"QRSSTVAE CE preamble", n, b"\x01\x02")
    assert got.dtype == np.uint8
    assert got.tolist() == _sha_bits_by_hand(b"QRSSTVAE CE preamble", n, b"\x01\x02")


def test_streams_follow_their_definitions():
    """Domain strings, prefixes and the bit-0 -> +1 map (decision D5)."""
    pre = 1 - 2 * np.array(_sha_bits_by_hand(b"QRSSTVAE CE preamble", 660))
    assert np.array_equal(sequences.preamble_ce(), pre)
    ref = 1 - 2 * np.array(_sha_bits_by_hand(b"QRSSTVAE CE reference", 300))
    assert np.array_equal(sequences.references_ce(300), ref)
    scr = 1 - 2 * np.array(_sha_bits_by_hand(
        b"QRSSTVAE scramble", 600, struct.pack(">Q", Q_TEST)))
    assert np.array_equal(sequences.scrambler(Q_TEST, 600), scr)


def test_shorter_streams_are_prefixes():
    assert np.array_equal(sequences.references_ce(1258), sequences.references_ce(3539)[:1258])
    assert np.array_equal(sequences.scrambler(Q_TEST, 4096),
                          sequences.scrambler(Q_TEST, 50600)[:4096])


def test_scrambler_differs_between_slots():
    a, b = sequences.scrambler(Q_TEST, 50600), sequences.scrambler(Q_TEST + 1, 50600)
    agree = np.mean(a == b)
    assert abs(agree - 0.5) < 4 * 0.5 / np.sqrt(50600)


def test_cached_sequences_are_read_only():
    with pytest.raises(ValueError):
        sequences.preamble_ce()[0] = 0
    with pytest.raises(ValueError):
        sequences.references_ce(10)[0] = 0


# --- F6: format modules may not reseed the format ------------------------------

# Design section 1: modules marked F define what goes on air.
FORMAT_MODULES = ("constants", "sequences", "precoder", "frame", "morse", "picture",
                  "header", "polar", "ce", "tx", "beaconfile", "si5351")
# Everything else the design places in sstvae/qrss. A module in neither
# list fails `test_every_qrss_module_is_classified`, so a new file has to
# be declared format or not before it can land.
NON_FORMAT_MODULES = ("__init__", "types", "channel", "frontend", "acquire", "track",
                      "demod", "cwid", "receiver", "store", "associate", "render", "em")

_FORBIDDEN_CALLS = {"default_rng", "RandomState", "seed", "Generator", "PCG64",
                    "MT19937", "SeedSequence"}


def _offences(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name in _FORBIDDEN_CALLS:
                out.append(f"{path.name}:{node.lineno}: calls {name}()")
        elif isinstance(node, ast.Attribute) and node.attr == "random" \
                and isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy"):
            out.append(f"{path.name}:{node.lineno}: uses {node.value.id}.random")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in ("random", "numpy.random"):
                    out.append(f"{path.name}:{node.lineno}: imports {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in ("random", "numpy.random") or (
                    mod == "numpy" and any(a.name == "random" for a in node.names)):
                out.append(f"{path.name}:{node.lineno}: imports from {mod}")
    return out


@pytest.mark.parametrize("module", FORMAT_MODULES)
def test_format_module_does_not_reseed_the_format(module):
    path = QRSS_DIR / f"{module}.py"
    if not path.exists():
        pytest.skip(f"sstvae/qrss/{module}.py not written yet")
    offences = _offences(path)
    assert not offences, (
        "a QRSSTVAE format module draws from a random generator:\n  "
        + "\n  ".join(offences)
        + "\nOn-air sequences must be closed form (SHA-256 over integers) or "
          "committed literal data; see design ground rule 4.")


def test_the_rule_catches_a_seeded_draw(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import numpy as np\n"
                   "x = np.random.default_rng(1).standard_normal(4)\n")
    assert len(_offences(bad)) >= 2


def test_every_qrss_module_is_classified():
    known = set(FORMAT_MODULES) | set(NON_FORMAT_MODULES)
    present = {p.stem for p in QRSS_DIR.glob("*.py")}
    assert not present - known, (
        f"unclassified modules in sstvae/qrss: {sorted(present - known)}; "
        "add each to FORMAT_MODULES (and make it comply) or NON_FORMAT_MODULES")


_FORBIDDEN_IMPORTS = ("torch", "sstvae.data", "sstvae.models", "sstvae.waveform_channel",
                      "torchvision")


def _forbidden_imports(path: Path) -> list[str]:
    """Imports of torch or SSTVAE's training code, for a module in sstvae/qrss."""
    bad = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:                  # relative: from .. import data
                pkg = ["sstvae", "qrss"][:max(0, 3 - node.level)]
                base = ".".join(pkg + ([base] if base else []))
            names = [base] + [f"{base}.{a.name}" for a in node.names]
        for n in names:
            if any(n == f or n.startswith(f + ".") for f in _FORBIDDEN_IMPORTS):
                bad.append(f"{path.name}:{node.lineno}: {n}")
    return bad


def test_nothing_in_qrss_imports_torch_or_training_code():
    """Ground rule 2: the QRSS package is torch-free, at any import depth."""
    bad = [b for path in sorted(QRSS_DIR.glob("*.py")) for b in _forbidden_imports(path)]
    assert not bad, "forbidden imports in sstvae/qrss:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("src", ["import torch\n", "from torch import nn\n",
                                 "from .. import data\n", "from ..models import x\n",
                                 "from sstvae import waveform_channel\n"])
def test_the_import_rule_catches(src, tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(src)
    assert _forbidden_imports(bad)


def test_the_import_rule_allows_the_codec(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text("from sstvae import codec, checkpoint, latents\nfrom . import frame\n"
                  "from ..modem import framing\n")
    assert not _forbidden_imports(ok)
