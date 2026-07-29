"""The RNG-derived parts of the on-air format are frozen data.

Two constants were originally produced by seeding numpy: the pilot
symbol's QPSK quadrants and the per-group interleaver permutations. Both
are part of the waveform — two stations must agree on them exactly or
the picture is noise, with no error to say why.

**The direction of authority is what these tests protect.** If a future
numpy changed its generator stream, the correct behaviour is to keep
transmitting the same pilots and the same interleave, *not* to follow
numpy. That is only possible if the values are written down rather than
re-derived at import, so these tests assert they are written down and
that nothing reaches for the RNG.

`tools/freeze_format_constants.py --verify` re-derives both from the
seeds and reports whether numpy still agrees. It is deliberately a
script rather than a test: wiring it into CI would make numpy's current
behaviour authoritative over the format, and the obvious way to fix a
red build would be to regenerate the frozen data — silently changing
what the radio transmits.
"""

import numpy as np

from sstvae import config
from sstvae.modem import framing, ofdm


def test_pilot_quadrants_are_a_literal():
    """Not an array, not computed — something a person can read in a diff."""
    assert isinstance(config.PILOT_QUADRANTS, tuple)
    assert len(config.PILOT_QUADRANTS) == config.NC
    assert all(isinstance(k, int) and 0 <= k < 4 for k in config.PILOT_QUADRANTS)


def test_pilot_sequence_comes_from_the_frozen_quadrants():
    expected = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.asarray(config.PILOT_QUADRANTS)))
    assert np.array_equal(ofdm.pilot_sequence(), expected)
    # Unit magnitude is the on-air contract, independent of the values.
    assert np.max(np.abs(np.abs(ofdm.pilot_sequence()) - 1.0)) < 1e-15


def test_interleaver_perms_are_loaded_not_derived():
    """The file must exist and be what framing uses.

    A regression here would most likely look like someone "simplifying"
    the load back into a `default_rng(...).permutation(...)` call, which
    would work perfectly until the day numpy changed.
    """
    path = framing._PERMS_PATH
    assert path.exists(), f"{path} is missing; it is shipped package data"
    stored = np.load(path)
    assert np.array_equal(stored.astype(np.intp), framing._TX_PERMS)


def test_interleaver_perms_are_widened_on_load():
    """uint16 on disk, but not in use.

    Indices are < GROUP_LATENTS = 52,800 so uint16 stores them, but
    callers add a group offset of up to 2*GROUP_LATENTS = 105,600. Under
    NEP 50 a Python int plus a uint16 array stays uint16: numpy 2 raises,
    an older numpy wraps silently and scrambles groups 1 and 2.
    """
    assert framing._TX_PERMS.dtype == np.intp
    offset = 2 * config.GROUP_LATENTS + framing._TX_PERMS[2]
    assert offset.max() >= config.GROUP_LATENTS * 2


def test_each_group_permutation_is_a_valid_prefix():
    """Structural properties that hold whatever the values are: each
    group's slots are distinct canonical latents inside that group."""
    assert framing._TX_PERMS.shape == (config.LATENT_GROUPS,
                                       config.TRANSMIT_LATENTS_PER_GROUP)
    for g in range(config.LATENT_GROUPS):
        perm = framing._TX_PERMS[g]
        assert len(np.unique(perm)) == len(perm), f"group {g} repeats an index"
        assert perm.min() >= 0 and perm.max() < config.GROUP_LATENTS

        # The dropped remainder is exactly the documented accounting.
        dropped = config.GROUP_LATENTS - len(perm)
        assert dropped == config.DROPPED_LATENTS_PER_GROUP


def test_nothing_in_the_modem_reseeds_the_format():
    """No module in `sstvae/modem/` may *call* a seeded RNG.

    Parsed rather than grepped: the modules explain at length why they
    do not call `default_rng`, so a text search finds its own
    documentation. The AST finds calls, which is the thing that would
    actually put numpy's generator back into the on-air format.

    `default_rng` is entirely fine in tests, simulation and training —
    the rule is specific to the modem package, where a seeded draw means
    a format constant being rebuilt at runtime.
    """
    import ast
    from pathlib import Path

    modem_dir = Path(framing.__file__).parent
    offenders = []
    for path in sorted(modem_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name in ("default_rng", "RandomState", "seed"):
                offenders.append(f"{path.name}:{node.lineno}: calls {name}()")
    assert not offenders, (
        "the modem package derives something from a seeded RNG:\n  "
        + "\n  ".join(offenders)
        + "\nFormat constants must be frozen data (see config.py); if this is "
          "not a format constant, move it out of sstvae/modem/."
    )
