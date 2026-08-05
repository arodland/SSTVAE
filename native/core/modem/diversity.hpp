// Diversity reception: combine two or more independent receivers of the
// same transmission (different antennas/audio devices, independent
// noise and fading). Port of sstvae/modem/diversity.py -- see
// docs/diversity-reception.md for the derivation and measured gain.
//
// Deliberately a post-processing step over Modem::demodulate rather than
// a change to the demod internals, for the same reason as the Python
// side: DemodResult::latents/weights are already in canonical order, so
// two branches' arrays are directly comparable index-for-index with no
// shared sample timebase needed.

#ifndef SSTVAE_MODEM_DIVERSITY_HPP
#define SSTVAE_MODEM_DIVERSITY_HPP

#include <vector>

#include "images/types.hpp"
#include "modem/modem.hpp"

namespace sstvae::modem::diversity {

// Maximal-ratio combine already-demodulated branches. A single branch
// is returned unchanged. Throws std::invalid_argument if `results` is
// empty or the branches locked different modes -- see the .cpp for the
// weight derivation (identical to combine_demod_results.py's docstring).
DemodResult combine_demod_results(const std::vector<DemodResult>& results);

// `results.size()` vectors of `n_latents` each: branch i's fractional
// share of the MRC combine at every latent (columns sum to 1 wherever
// any branch has nonzero weight, 0 where every branch erased that
// latent). Same preconditions as combine_demod_results.
std::vector<std::vector<double>> branch_contribution(
    const std::vector<DemodResult>& results);

// Debug visualization of which branch supplied each transmitted latent:
// rows are latent channel (0..LATENT_CHANNELS-1), columns are absolute
// frame index (time). Red is branch 0's fractional share, blue is
// branch 1's; a cell with no transmitted latent that frame or erased on
// both branches is black. `scale` replicates each cell into a
// scale x scale block (nearest-neighbor, not a smoothing resize -- the
// data is categorical per cell). Requires exactly two branches of the
// same mode.
images::Picture contribution_image(const std::vector<DemodResult>& results,
                                   int scale = 6);

}  // namespace sstvae::modem::diversity

#endif
