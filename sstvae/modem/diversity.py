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

from .modem import DemodResult
from .sync import SyncError

__all__ = ["combine_demod_results", "demodulate_diversity"]


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
    if not results:
        raise ValueError("combine_demod_results needs at least one branch")
    spec = results[0].mode
    for r in results[1:]:
        if r.mode.name != spec.name:
            raise ValueError(f"branch mode mismatch: {spec.name!r} vs {r.mode.name!r}")
    if len(results) == 1:
        return results[0]

    snr_lin = np.array(
        [10 ** (r.snr_db / 10) if np.isfinite(r.snr_db) else 0.0 for r in results]
    )
    if not np.any(snr_lin > 0):
        # No branch has a usable SNR estimate (e.g. too few received
        # frames to form one) -- fall back to unweighted averaging
        # rather than producing an all-zero combine.
        snr_lin = np.ones(len(results))
    ref = float(np.max(snr_lin))

    latents = np.stack([r.latents for r in results])
    weights = np.stack([r.weights for r in results])
    mrc_w = snr_lin[:, None] * weights**2  # (branch, latent) MRC weight

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
