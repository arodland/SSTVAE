"""Diversity reception: combine two or more independent receivers of the
*same* transmission (different antennas/audio devices, independent noise
and fading, no assumption of phase lock -- only that each branch is
within a frame time of the others, which is all a preamble-based
re-acquisition on each branch needs).

Deliberately implemented as a post-processing step over `Modem.demodulate`
rather than a change to the demod internals: `DemodResult.latents` and
`.weights` are already in canonical (post-deinterleave) order, so two
branches' arrays are directly comparable index-for-index without sharing
a sample timebase at all -- each branch resyncs on its own copy of the
preamble and does its own clock-drift tracking. See
`docs/diversity-reception.md` for the derivation and measured gain.
"""

from dataclasses import replace

import numpy as np
from PIL import Image

from ..config import LATENT_CHANNELS, LATENT_H, LATENT_W
from . import framing
from .modem import DemodResult
from .sync import SyncError

__all__ = [
    "combine_demod_results",
    "demodulate_diversity",
    "branch_contribution",
    "contribution_image",
]


def _validate_branches(results: list[DemodResult]):
    """Common precondition for every function below: every branch must
    have locked onto the same mode -- a header mismatch means the
    branches aren't looking at the same transmission, or one badly
    misdecoded its header, and guessing which one to believe would be
    worse than failing loud."""
    if not results:
        raise ValueError("needs at least one branch")
    spec = results[0].mode
    for r in results[1:]:
        if r.mode.name != spec.name:
            raise ValueError(f"branch mode mismatch: {spec.name!r} vs {r.mode.name!r}")


def _mrc_weights(results: list[DemodResult]) -> tuple[np.ndarray, np.ndarray]:
    """`(snr_lin, mrc_w)`: each branch's linear SNR, and the
    `(n_branches, n_latents)` inverse-variance (MRC) combining weight
    per branch/latent -- `snr_lin * w**2`, see `combine_demod_results`
    for the derivation. Shared by `combine_demod_results` (which sums
    these into a combined estimate) and `branch_contribution` (which
    normalizes them into each branch's fractional share)."""
    snr_lin = np.array(
        [10 ** (r.snr_db / 10) if np.isfinite(r.snr_db) else 0.0 for r in results]
    )
    if not np.any(snr_lin > 0):
        # No branch has a usable SNR estimate (e.g. too few received
        # frames to form one) -- fall back to unweighted averaging
        # rather than producing an all-zero combine.
        snr_lin = np.ones(len(results))
    weights = np.stack([r.weights for r in results])
    return snr_lin, snr_lin[:, None] * weights**2


def combine_demod_results(results: list[DemodResult]) -> DemodResult:
    """Maximal-ratio combine already-demodulated branches.

    Every branch's per-latent weight `w` in [0, 1] is a *relative*
    confidence -- fading depth against that branch's own median channel
    gain (see `Modem.demodulate`) -- not comparable in absolute terms
    across branches with different noise floors (different antennas,
    preamps, band noise). `snr_db` puts branches on a common footing: a
    branch/latent's noise variance scales as `1 / (snr_lin * w**2)`, so
    the MRC (inverse-variance) weight is `snr_lin * w**2`, and the
    combined weight is capped at 1 so a multi-branch combine never
    reports more confidence to the decoder than a single clean branch
    did during training (the decoder never saw weight > 1). This is also
    why the combine is worth doing even when it can't move the reported
    weight: the combined *latent* value has genuinely lower noise for
    the same nominal weight, which is where the gain actually comes
    from -- see docs/diversity-reception.md.

    Requires every branch to have locked onto the same mode -- a header
    mismatch means the branches aren't looking at the same transmission,
    or one badly misdecoded its header, and guessing which one to
    believe would be worse than failing loud.

    The MRC weight derivation assumes independent noise between
    branches -- true for separate antennas/receivers, the case this is
    for. Feeding it two branches derived from the same recording (fully
    correlated "noise") isn't a diversity scenario and will overstate
    the combined confidence; the combined latent values are still
    correct in that degenerate case (there's nothing to average away),
    only the reported weight would be optimistic.
    """
    _validate_branches(results)
    if len(results) == 1:
        return results[0]

    snr_lin, mrc_w = _mrc_weights(results)
    ref = float(np.max(snr_lin))

    latents = np.stack([r.latents for r in results])
    denom = mrc_w.sum(axis=0)
    combined_latents = np.divide(
        (mrc_w * latents).sum(axis=0), denom,
        out=np.zeros_like(denom), where=denom > 0,
    )
    combined_weights = np.minimum(np.sqrt(denom / ref), 1.0)

    best = max(
        results, key=lambda r: r.snr_db if np.isfinite(r.snr_db) else -np.inf
    )
    combined_snr_db = (
        10 * np.log10(float(np.sum(snr_lin))) if np.any(snr_lin > 0) else float("nan")
    )
    return replace(
        best,
        latents=combined_latents,
        weights=combined_weights,
        frames_received=max(r.frames_received for r in results),
        snr_db=combined_snr_db,
    )


def demodulate_diversity(modem, streams: list[np.ndarray], search_s=None) -> DemodResult:
    """Demodulate every branch independently and MRC-combine the result.

    A branch that fails to acquire (faded out entirely, or too short) is
    dropped rather than failing the whole combine -- that is the point
    of diversity reception, one antenna losing lock while another
    doesn't. If every branch fails, the first branch's `SyncError`
    propagates.
    """
    results = []
    first_error: SyncError | None = None
    for x in streams:
        try:
            results.append(modem.demodulate(x, search_s=search_s))
        except SyncError as e:
            if first_error is None:
                first_error = e
    if not results:
        raise first_error if first_error is not None else SyncError(
            "demodulate_diversity: no branches provided"
        )
    return combine_demod_results(results)


def branch_contribution(results: list[DemodResult]) -> np.ndarray:
    """`(n_branches, n_latents)`: each branch's fractional share of the
    MRC combine at every latent -- `mrc_w / sum(mrc_w)`, so the columns
    sum to 1 wherever at least one branch has nonzero weight there, and
    to 0 where every branch erased that latent (nothing to attribute).
    This is what `contribution_image` visualizes; exposed separately so
    a caller that only wants the numbers doesn't have to render an
    image to get them.
    """
    _validate_branches(results)
    if len(results) == 1:
        return (results[0].weights > 0).astype(np.float64)[None, :]
    _, mrc_w = _mrc_weights(results)
    denom = mrc_w.sum(axis=0)
    return np.divide(mrc_w, denom, out=np.zeros_like(mrc_w), where=denom > 0)


def contribution_image(results: list[DemodResult], scale: int = 6) -> Image.Image:
    """Debug visualization of which branch supplied each transmitted
    latent: rows are latent channel (0..`LATENT_CHANNELS`-1), columns
    are absolute frame index (time). Red is branch 0's fractional share
    of the combined MRC estimate at that channel/frame, blue is branch
    1's -- a cell that's pure red or pure blue means one branch carried
    that latent essentially alone (the other faded), magenta means both
    contributed roughly equally, and black means either no transmitted
    latent that frame touched that channel (the interleaver scatters
    each frame's latents across channels, not one-per-frame) or the
    latent was erased on both branches.

    Requires exactly two branches (red/blue is a two-way encoding) of
    the same mode. Frame `f`'s transmitted latents are wherever
    `framing.slot_range_for_frame(f)` says, which is mode-independent
    (every mode's frames are a prefix of the same canonical layout), so
    this reads directly off `results[0].mode.n_frames` without needing
    the interleaver detail beyond that lookup.
    """
    if len(results) != 2:
        raise ValueError("contribution_image needs exactly two branches")
    _validate_branches(results)
    frac = branch_contribution(results)  # (2, n_latents)
    n_frames = results[0].mode.n_frames
    per_channel = LATENT_H * LATENT_W

    grid = np.zeros((LATENT_CHANNELS, n_frames, 2))
    counts = np.zeros((LATENT_CHANNELS, n_frames))
    for f in range(n_frames):
        _, idx = framing.slot_range_for_frame(f)
        channels = idx // per_channel
        np.add.at(grid[:, f, 0], channels, frac[0, idx])
        np.add.at(grid[:, f, 1], channels, frac[1, idx])
        np.add.at(counts[:, f], channels, 1)

    mask = counts > 0
    grid = np.divide(grid, counts[:, :, None], out=grid, where=mask[:, :, None])

    rgb = np.zeros((LATENT_CHANNELS, n_frames, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(grid[:, :, 0] * 255, 0, 255).astype(np.uint8)
    rgb[:, :, 2] = np.clip(grid[:, :, 1] * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    if scale != 1:
        img = img.resize((n_frames * scale, LATENT_CHANNELS * scale), Image.NEAREST)
    return img
