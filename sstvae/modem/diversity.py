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

A branch that never gets a header lock can still contribute via
`Modem.demodulate_blind` (the preamble-free path -- see
`sstvae/modem/beacon.py`), which is not only usable here but slightly
*more* directly comparable than the header path: `BlindDemodResult.
latents`/`.weights` are always sized to mode C's full canonical range
and placed by the beacon's absolute frame counter, so two independent
blind locks of the same transmission land in the same array positions
automatically, with no epsilon-based "is this the same reception"
matching needed the way two header locks' sample positions require.
`combine_diversity_results` below handles any mix of header-locked and
blind-locked branches for that reason.
"""

from dataclasses import replace

import numpy as np
from PIL import Image

from ..config import FRAMES_PER_GROUP, LATENT_GROUPS, MODES, NC_LATENT
from . import framing
from .modem import BlindDemodResult, DemodResult
from .sync import SyncError

__all__ = [
    "combine_demod_results",
    "combine_blind_results",
    "combine_diversity_results",
    "demodulate_diversity",
    "branch_contribution",
    "contribution_image",
]

_FULL_C_LATENTS = MODES["C"].n_latents
_FULL_C_FRAMES = LATENT_GROUPS * FRAMES_PER_GROUP


def _pad_to_full(vec: np.ndarray) -> np.ndarray:
    """Zero-pad a mode A/B/C latent or weight vector to mode C's full
    length. Reimplements `codec.pad_to_full`'s two lines locally rather
    than importing `sstvae.codec` from `sstvae.modem`, which sits above
    modem in the dependency graph."""
    full = np.zeros(_FULL_C_LATENTS, dtype=vec.dtype)
    full[: len(vec)] = vec
    return full


def _validate_branches(results: list[DemodResult]):
    """Common precondition for combining two or more header-locked
    branches: they must have locked onto the same mode -- a header
    mismatch means the branches aren't looking at the same transmission,
    or one badly misdecoded its header, and guessing which one to
    believe would be worse than failing loud. Only meaningful for 2+
    branches; callers skip it for 0 or 1."""
    spec = results[0].mode
    for r in results[1:]:
        if r.mode.name != spec.name:
            raise ValueError(f"branch mode mismatch: {spec.name!r} vs {r.mode.name!r}")


def _mrc_weights_raw(
    weights_list: list[np.ndarray], snr_dbs: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """`(snr_lin, mrc_w)`: each branch's linear SNR, and the
    `(n_branches, n_latents)` inverse-variance (MRC) combining weight
    per branch/latent -- `snr_lin * w**2`, see `combine_demod_results`
    for the derivation. Raw-array form, so it works identically whether
    the branches are `DemodResult`, `BlindDemodResult`, or padded copies
    of either."""
    snr_lin = np.array(
        [10 ** (s / 10) if np.isfinite(s) else 0.0 for s in snr_dbs]
    )
    if not np.any(snr_lin > 0):
        # No branch has a usable SNR estimate (e.g. too few received
        # frames to form one) -- fall back to unweighted averaging
        # rather than producing an all-zero combine.
        snr_lin = np.ones(len(snr_dbs))
    weights = np.stack(weights_list)
    return snr_lin, snr_lin[:, None] * weights**2


def _mrc_weights(results) -> tuple[np.ndarray, np.ndarray]:
    return _mrc_weights_raw([r.weights for r in results], [r.snr_db for r in results])


def _combined_confidence(mrc_w: np.ndarray, snr_lin: np.ndarray) -> np.ndarray:
    """`(n_latents,)` overall combined-branch confidence, on the same
    [0, 1] scale as `DemodResult.weights`: `min(sqrt(sum(mrc_w)/ref), 1)`
    where `ref` is the best single branch's own linear SNR. Shared by
    `_mrc_combine_arrays` (which reports this as the combine's weights)
    and `contribution_image`'s brightness channel (which instead
    normalizes it to the reception's own peak rather than capping it at
    1 -- see `_combined_weight`)."""
    ref = float(np.max(snr_lin))
    denom = mrc_w.sum(axis=0)
    return np.minimum(np.sqrt(denom / ref), 1.0)


def _mrc_combine_arrays(
    latents_list: list[np.ndarray], weights_list: list[np.ndarray], snr_dbs: list[float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """The core MRC arithmetic shared by every combine_* function below,
    over raw arrays rather than any particular result dataclass. Needs
    at least 2 branches; callers handle the single-branch identity case
    themselves (each dataclass has its own "return unchanged" semantics).
    """
    snr_lin, mrc_w = _mrc_weights_raw(weights_list, snr_dbs)
    denom = mrc_w.sum(axis=0)
    latents = np.divide(
        (mrc_w * np.stack(latents_list)).sum(axis=0), denom,
        out=np.zeros_like(denom), where=denom > 0,
    )
    weights = _combined_confidence(mrc_w, snr_lin)
    # snr_lin always has at least one positive entry after the fallback
    # above, so this sum is always > 0.
    snr_db = 10 * np.log10(float(np.sum(snr_lin)))
    return latents, weights, snr_db


def combine_demod_results(results: list[DemodResult]) -> DemodResult:
    """Maximal-ratio combine already header-locked branches.

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
    if not results:
        raise ValueError("combine_demod_results needs at least one branch")
    if len(results) == 1:
        return results[0]
    _validate_branches(results)

    latents, weights, snr_db = _mrc_combine_arrays(
        [r.latents for r in results], [r.weights for r in results],
        [r.snr_db for r in results],
    )
    best = max(results, key=lambda r: r.snr_db if np.isfinite(r.snr_db) else -np.inf)
    return replace(
        best,
        latents=latents,
        weights=weights,
        frames_received=max(r.frames_received for r in results),
        snr_db=snr_db,
    )


def combine_blind_results(results: list[BlindDemodResult]) -> BlindDemodResult:
    """Maximal-ratio combine independently blind-acquired branches.

    Simpler than the header case in one respect: `BlindDemodResult.
    latents`/`.weights` are already full mode-C-sized and placed by the
    beacon's *absolute* frame counter, so two branches of the same
    transmission are automatically aligned index-for-index -- there is
    no mode to mismatch and no sample-position matching needed here
    (the caller, `combine_diversity_results`/`decode_loop_diversity`,
    still checks the branches' *positions* agree before calling this,
    as a "these are really the same transmission" sanity check, not to
    align the data).
    """
    if not results:
        raise ValueError("combine_blind_results needs at least one branch")
    if len(results) == 1:
        return results[0]

    latents, weights, snr_db = _mrc_combine_arrays(
        [r.latents for r in results], [r.weights for r in results],
        [r.snr_db for r in results],
    )
    best = max(results, key=lambda r: r.snr_db if np.isfinite(r.snr_db) else -np.inf)
    return replace(
        best,
        latents=latents,
        weights=weights,
        n_frames=max(r.n_frames for r in results),
        snr_db=snr_db,
    )


def combine_diversity_results(results):
    """Combine any mix of `DemodResult` (header-locked) and
    `BlindDemodResult` (blind-locked) branches with one MRC pass,
    regardless of which acquisition path found each one -- both already
    report latents/weights in canonical order, `DemodResult` sized to
    its mode's prefix and `BlindDemodResult` always the full mode-C
    range, so the only extra step is padding a header branch's arrays up
    to the common size (`codec.pad_to_full`'s own trick, reimplemented
    locally -- see `_pad_to_full`).

    If any branch got a header lock, its mode is authoritative for what
    was actually sent, so the combine happens at full size and is then
    truncated back down to that mode's range, and the result is a
    `DemodResult` -- keeping the caller's exact-frame-count completion
    check available, the same preference `Modem.demodulate`'s own
    header-first, blind-fallback order embodies for a single receiver.
    Only when every branch is blind-locked does this return a
    `BlindDemodResult`, matching `demodulate_blind`'s own shape.

    All header-locked branches present (2 or more) must share one mode,
    same as `combine_demod_results`.
    """
    if not results:
        raise ValueError("combine_diversity_results needs at least one branch")
    headered = [r for r in results if isinstance(r, DemodResult)]
    blind = [r for r in results if isinstance(r, BlindDemodResult)]
    if len(headered) + len(blind) != len(results):
        raise TypeError("branches must be DemodResult or BlindDemodResult")

    if not blind:
        return combine_demod_results(headered)
    if not headered:
        return combine_blind_results(blind)

    if len(headered) > 1:
        _validate_branches(headered)
    spec = headered[0].mode
    n = spec.n_latents

    latents_list = [_pad_to_full(r.latents) for r in headered] + [r.latents for r in blind]
    weights_list = [_pad_to_full(r.weights) for r in headered] + [r.weights for r in blind]
    snr_dbs = [r.snr_db for r in headered] + [r.snr_db for r in blind]
    latents, weights, snr_db = _mrc_combine_arrays(latents_list, weights_list, snr_dbs)

    best = max(headered, key=lambda r: r.snr_db if np.isfinite(r.snr_db) else -np.inf)
    return replace(
        best,
        latents=latents[:n],
        weights=weights[:n],
        frames_received=max(r.frames_received for r in headered),
        snr_db=snr_db,
    )


def demodulate_diversity(modem, streams: list[np.ndarray], search_s=None) -> DemodResult:
    """Demodulate every branch independently and MRC-combine the result.

    A branch that fails to acquire (faded out entirely, or too short) is
    dropped rather than failing the whole combine -- that is the point
    of diversity reception, one antenna losing lock while another
    doesn't. If every branch fails, the first branch's `SyncError`
    propagates. Header path only (`Modem.demodulate`); a caller that
    wants blind-fallback diversity combines `Modem.demodulate_blind`
    results itself with `combine_diversity_results` -- see
    `sstvae/rx/engine.py`'s `decode_loop_diversity` for that shape.
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


def _weights_list(results) -> list[np.ndarray]:
    """Each branch's weight array, padded to mode C's full length when
    the mix includes a blind branch (see `combine_diversity_results`)
    -- the validation/padding shared by `branch_contribution` and
    `_combined_weight` below."""
    headered = [r for r in results if isinstance(r, DemodResult)]
    blind = [r for r in results if isinstance(r, BlindDemodResult)]
    if len(headered) + len(blind) != len(results):
        raise TypeError("branches must be DemodResult or BlindDemodResult")

    if not blind:
        _validate_branches(headered)
        return [r.weights for r in results]
    if len(headered) > 1:
        _validate_branches(headered)
    return [
        _pad_to_full(r.weights) if isinstance(r, DemodResult) else r.weights
        for r in results
    ]


def branch_contribution(results) -> np.ndarray:
    """`(n_branches, n_latents)`: each branch's fractional share of the
    MRC combine at every latent -- `mrc_w / sum(mrc_w)`, so the columns
    sum to 1 wherever at least one branch has nonzero weight there, and
    to 0 where every branch erased that latent (nothing to attribute).
    This is what `contribution_image` visualizes as hue; exposed
    separately so a caller that only wants the numbers doesn't have to
    render an image to get them. See `_combined_weight` for the
    complementary *overall* strength `contribution_image` visualizes as
    brightness.

    Accepts the same mix of `DemodResult`/`BlindDemodResult` branches as
    `combine_diversity_results`; `n_latents` is the header-locked mode's
    range if every branch agrees on one, else the full mode-C range.
    """
    if not results:
        raise ValueError("branch_contribution needs at least one branch")
    if len(results) == 1:
        return (results[0].weights > 0).astype(np.float64)[None, :]

    weights_list = _weights_list(results)
    _, mrc_w = _mrc_weights_raw(weights_list, [r.snr_db for r in results])
    denom = mrc_w.sum(axis=0)
    return np.divide(mrc_w, denom, out=np.zeros_like(mrc_w), where=denom > 0)


def _combined_weight(results) -> np.ndarray:
    """`(n_latents,)`: the branches' *overall* combined confidence at
    every latent -- what `contribution_image` scales brightness by, on
    top of `branch_contribution`'s hue, so a latent both branches faded
    on goes dark rather than staying a saturated color just because the
    two branches happened to split it evenly. A single branch is its
    own weight (nothing to combine); two or more use the same MRC
    confidence `_mrc_combine_arrays` reports as `DemodResult.weights`.
    Internal: `contribution_image` normalizes this to its own
    reception's peak rather than exposing the raw [0, 1] scale, which
    would make two different receptions' images incomparable at a
    glance for no benefit."""
    if len(results) == 1:
        return results[0].weights
    weights_list = _weights_list(results)
    snr_lin, mrc_w = _mrc_weights_raw(weights_list, [r.snr_db for r in results])
    return _combined_confidence(mrc_w, snr_lin)


def contribution_image(results, scale: int = 6) -> Image.Image:
    """Debug visualization of which branch supplied each transmitted
    latent, and how much either of them had to offer: rows are the data
    carrier index (0..`NC_LATENT`-1, row 0 the lowest frequency),
    columns are absolute frame index (time). Hue is `branch_contribution`
    -- red is branch 0's fractional share of the combined MRC estimate
    on that carrier/frame, blue is branch 1's, magenta means both
    contributed roughly equally -- and brightness is `_combined_weight`,
    normalized to the *brightest* cell this reception ever reached. A
    carrier that fades on one branch but stays strong on the other still
    reads as a saturated, bright color (that's the case this feature
    exists to keep transmitting through); a carrier that fades on
    *both* branches goes dark regardless of how evenly they split what
    little they had, down to black where every branch erased it. Without
    the brightness term, two branches equally weak would draw exactly
    like two branches equally strong -- a pure hue -- and give no visual
    signal that combining them didn't actually help there.

    Rows are carrier index, not the decoder's latent-channel index
    `branch_contribution`'s columns are ordered by: that index is where
    the interleaver's PAPR-motivated permutation *sends* a canonical
    latent to be transmitted, which scrambles frequency order and, for
    modes B/C -- whose groups are sent as sequential blocks, each
    confined to its own slice of decoder channels -- would render as a
    staircase instead of one continuous band. Carrier index is instead
    read off each latent's on-air *position*, `slot_range_for_frame`'s
    index into its own returned array (not the array's value, which is
    the scrambled canonical index) -- `framing.slots_to_symbols` reshapes
    a frame's `LATENTS_PER_FRAME` slots as `(DATA_SYMS_PER_FRAME,
    NC_LATENT, 2)`, so position `k`'s carrier is `(k // 2) % NC_LATENT`,
    real/imag-independent and identical across every group and mode.
    Every carrier carries data in every symbol of every frame, so unlike
    the old channel-indexed image there are no structurally-black
    cells from unused positions -- black here always means the combine
    was weak (at the limit, both branches erased that carrier this
    frame), never "the interleaver didn't touch it."

    Requires exactly two branches (red/blue is a two-way encoding), any
    mix of `DemodResult`/`BlindDemodResult`. `n_frames` is the
    header-locked mode's frame count if both branches agree on one,
    else mode C's full frame range (every mode's frames are a prefix of
    it) -- `framing.slot_range_for_frame(f)` is mode-independent either
    way, so this reads directly off whichever range applies.
    """
    if len(results) != 2:
        raise ValueError("contribution_image needs exactly two branches")
    frac = branch_contribution(results)  # (2, n_latents)
    overall = _combined_weight(results)  # (n_latents,)

    if all(isinstance(r, DemodResult) for r in results):
        _validate_branches(results)
        n_frames = results[0].mode.n_frames
    else:
        n_frames = _FULL_C_FRAMES

    grid = np.zeros((NC_LATENT, n_frames, 2))
    strength = np.zeros((NC_LATENT, n_frames))
    counts = np.zeros((NC_LATENT, n_frames))
    for f in range(n_frames):
        _, idx = framing.slot_range_for_frame(f)
        carriers = (np.arange(len(idx)) // 2) % NC_LATENT
        np.add.at(grid[:, f, 0], carriers, frac[0, idx])
        np.add.at(grid[:, f, 1], carriers, frac[1, idx])
        np.add.at(strength[:, f], carriers, overall[idx])
        np.add.at(counts[:, f], carriers, 1)

    mask = counts > 0
    grid = np.divide(grid, counts[:, :, None], out=grid, where=mask[:, :, None])
    strength = np.divide(strength, counts, out=strength, where=mask)

    # Relative to this reception's own peak, not the [0, 1] confidence
    # scale: a reception that never got much above 0.3 anywhere should
    # still show its strongest carriers at full brightness, the same
    # way the fractional-share hue is relative to what the two branches
    # had between them rather than to some absolute unit.
    peak = float(strength.max())
    norm = strength / peak if peak > 0 else strength  # all zero -> stays black

    rgb = np.zeros((NC_LATENT, n_frames, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(grid[:, :, 0] * norm * 255, 0, 255).astype(np.uint8)
    rgb[:, :, 2] = np.clip(grid[:, :, 1] * norm * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    if scale != 1:
        img = img.resize((n_frames * scale, NC_LATENT * scale), Image.NEAREST)
    return img
