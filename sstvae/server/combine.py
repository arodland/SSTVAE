"""Combine every station's copy of one transmission into one picture.

This is the reason the server exists. The arithmetic is
`sstvae.modem.diversity`, unchanged and unwrapped -- it is pure numpy
over (latents, weights, snr_db), so it does not care that the branches
arrived over HTTP hours apart rather than from two soundcards at once.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ..codec import pad_to_full, reconstruct
from ..modem.diversity import branch_contribution, combine_diversity_results
from ..modem.modem import DemodResult
from ..rx.receptionfile import ReceptionFileError, read

__all__ = ["CombineResult", "combine_transmission", "load_branches"]


class CombineResult:
    def __init__(self, image, snr_db, mode_name, contributions, used, skipped):
        self.image = image
        self.snr_db = snr_db
        self.mode_name = mode_name
        self.contributions = contributions  # station_id -> mean share 0..1
        self.used = used  # station ids that went into the picture
        self.skipped = skipped  # (station_id, why) for the rest


def load_branches(rows) -> tuple[list, list, list]:
    """Read each station's payload back into a demodulator result.

    A payload that will not read is skipped rather than fatal: one
    station uploading something corrupt must not cost every other
    station their picture.
    """
    branches, station_ids, skipped = [], [], []
    for row in rows:
        try:
            result, _meta = read(Path(row["file_path"]).read_bytes())
        except (ReceptionFileError, OSError) as exc:
            skipped.append((row["station_id"], str(exc)))
            continue
        branches.append(result)
        station_ids.append(row["station_id"])
    return branches, station_ids, skipped


def _agree_on_mode(branches, station_ids):
    """Drop header branches that disagree with the majority mode.

    The combiner treats a mode mismatch as a hard error, and rightly:
    two different modes are two different transmissions. But the server
    has no way to ask, and one station reporting the wrong mode must not
    take the whole picture down with it, so the majority wins and the
    odd one out is skipped with a reason. Blind branches have no mode at
    all and are always kept -- their arrays are full-size and aligned by
    the beacon.
    """
    modes = [
        b.mode.name for b in branches if isinstance(b, DemodResult)
    ]
    if not modes:
        return branches, station_ids, []
    counts = Counter(modes)
    best = max(
        counts,
        key=lambda name: (
            counts[name],
            max(
                (b.snr_db for b in branches
                 if isinstance(b, DemodResult) and b.mode.name == name
                 and np.isfinite(b.snr_db)),
                default=float("-inf"),
            ),
        ),
    )
    keep, keep_ids, skipped = [], [], []
    for branch, sid in zip(branches, station_ids):
        if isinstance(branch, DemodResult) and branch.mode.name != best:
            skipped.append((sid, f"reported mode {branch.mode.name}, others say {best}"))
            continue
        keep.append(branch)
        keep_ids.append(sid)
    return keep, keep_ids, skipped


def combine_transmission(rows, codec) -> CombineResult | None:
    """Every reception of one transmission, as one picture."""
    branches, station_ids, skipped = load_branches(rows)
    branches, station_ids, mode_skips = _agree_on_mode(branches, station_ids)
    skipped = skipped + mode_skips
    if not branches:
        return None

    combined = combine_diversity_results(branches)

    # Mean share of the combined estimate, per station. Latents nobody
    # received contribute a zero column to every branch, so they are
    # left out rather than diluting everyone equally.
    contributions = {}
    if len(branches) > 1:
        shares = branch_contribution(branches)
        received = shares.sum(axis=0) > 0
        if received.any():
            for sid, row in zip(station_ids, shares):
                contributions[sid] = float(row[received].mean())
    else:
        contributions[station_ids[0]] = 1.0

    latents = combined.latents
    weights = combined.weights
    mode_name = None
    if isinstance(combined, DemodResult):
        mode_name = combined.mode.name
        latents = pad_to_full(latents)
        weights = pad_to_full(weights)

    image = reconstruct(codec, latents, weights)
    return CombineResult(
        image=image,
        snr_db=float(combined.snr_db),
        mode_name=mode_name,
        contributions=contributions,
        used=station_ids,
        skipped=skipped,
    )
