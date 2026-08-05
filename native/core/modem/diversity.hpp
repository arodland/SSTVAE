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
//
// A branch that never gets a header lock can still contribute via
// Modem::demodulate_blind. BlindDemodResult::latents/weights are always
// sized to mode C's full canonical range and placed by the beacon's
// absolute frame counter, so two independent blind locks of the same
// transmission land in the same array positions automatically -- more
// directly comparable than two header locks, whose sample positions
// need epsilon-based matching (done by the caller, decode_loop_diversity,
// not here). combine_diversity_results handles any mix of header-locked
// and blind-locked branches for that reason.

#ifndef SSTVAE_MODEM_DIVERSITY_HPP
#define SSTVAE_MODEM_DIVERSITY_HPP

#include <variant>
#include <vector>

#include "images/types.hpp"
#include "modem/modem.hpp"

namespace sstvae::modem::diversity {

// Maximal-ratio combine already header-locked branches. A single branch
// is returned unchanged. Throws std::invalid_argument if `results` is
// empty or the branches locked different modes -- see the .cpp for the
// weight derivation (identical to combine_demod_results.py's docstring).
DemodResult combine_demod_results(const std::vector<DemodResult>& results);

// Maximal-ratio combine independently blind-acquired branches. Simpler
// than the header case: BlindDemodResult's arrays are already full
// mode-C-sized and aligned by the beacon's absolute frame counter, so
// there is no mode to mismatch and no size to reconcile.
BlindDemodResult combine_blind_results(const std::vector<BlindDemodResult>& results);

// One diversity branch, however it was acquired.
using Branch = std::variant<DemodResult, BlindDemodResult>;

// Combine any mix of header-locked and blind-locked branches with one
// MRC pass. If any branch is header-locked, its mode is authoritative
// (the combine happens at full mode-C size, header branches padded up
// to it the same way latents::pad_to_full does, then truncated back
// down to that mode's range) and the result is a DemodResult -- keeping
// the caller's exact-frame-count completion check available. Only when
// every branch is blind-locked does this return a BlindDemodResult. All
// header-locked branches present (2 or more) must share one mode.
Branch combine_diversity_results(const std::vector<Branch>& results);

// `results.size()` vectors of `n_latents` each: branch i's fractional
// share of the MRC combine at every latent (columns sum to 1 wherever
// any branch has nonzero weight, 0 where every branch erased that
// latent). Same preconditions as combine_demod_results.
//
// Both this and contribution_image below are overloaded on
// vector<DemodResult> vs. vector<Branch> rather than only taking the
// latter, so a caller that already has a vector<DemodResult> does not
// need to wrap every element in a Branch. The cost: DemodResult
// converts implicitly to Branch, so a bare brace-init-list call site
// like `branch_contribution({a, b})` is ambiguous between the two
// overloads and fails to compile -- spell the type,
// `branch_contribution(std::vector<DemodResult>{a, b})` or
// `std::vector<Branch>{a, b}`. A call site already holding a typed
// vector<DemodResult> or vector<Branch> variable is never ambiguous;
// this only bites inline literals, which is why it shows up in tests
// and nowhere in decode_loop_diversity.
std::vector<std::vector<double>> branch_contribution(
    const std::vector<DemodResult>& results);
std::vector<std::vector<double>> branch_contribution(const std::vector<Branch>& results);

// Debug visualization of which branch supplied each transmitted latent,
// and how much either of them had to offer: rows are the data carrier
// index (0..NC_LATENT-1, row 0 the lowest frequency -- carriers are
// contiguous on-air positions, in frequency order by construction,
// unlike the decoder's latent-channel index, which the interleaver's
// PAPR-motivated permutation scatters and which -- for modes B/C, whose
// groups transmit as sequential blocks each confined to its own slice of
// decoder channels -- would draw as a staircase instead of one
// continuous band). Columns are absolute frame index (time). Hue is
// branch_contribution -- red is branch 0's fractional share, blue is
// branch 1's, magenta means both contributed roughly equally -- and
// brightness is the branches' combined confidence at that latent,
// normalized to the brightest cell this particular reception ever
// reached (not the raw [0, 1] weight scale, which would make two
// receptions' images incomparable at a glance for no benefit). A carrier
// that fades on one branch but stays strong on the other still reads as
// a saturated, bright color; a carrier that fades on *both* branches
// goes dark regardless of how evenly they split what little they had,
// down to black where every branch erased it -- without the brightness
// term, two branches equally weak would draw identically to two
// branches equally strong, giving no visual signal that combining them
// didn't actually help there. `scale` replicates each cell into a scale
// x scale block (nearest-neighbor, not a smoothing resize -- the data is
// categorical per cell). Requires exactly two branches; the DemodResult
// overload requires the same mode, the Branch overload allows any mix
// (using mode C's full frame range unless both branches happen to be
// header-locked to the same mode).
images::Picture contribution_image(const std::vector<DemodResult>& results,
                                   int scale = 6);
images::Picture contribution_image(const std::vector<Branch>& results, int scale = 6);

}  // namespace sstvae::modem::diversity

#endif
