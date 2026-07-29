// The frozen interleaver permutations.
//
// The table itself is in interleaver_table.cpp, which is GENERATED from
// sstvae/modem/interleaver_perms.npy by tools/gen_interleaver_table.py.
// That .npy is the on-air format: it was originally drawn from
// np.random.default_rng(INTERLEAVER_SEED + g), but nothing re-derives it,
// because doing so would make numpy's generator part of the waveform.
// See sstvae/modem/framing.py.

#pragma once

#include <cstdint>
#include <span>

#include "config.hpp"

namespace sstvae::framing {

inline constexpr int N_GROUPS = config::LATENT_GROUPS;
inline constexpr int TX_PERM_LEN = config::TRANSMIT_LATENTS_PER_GROUP;

extern const std::uint16_t TX_PERMS_DATA[N_GROUPS * TX_PERM_LEN];

// One group's permutation. Indices are canonical latent positions
// *within* the group, so callers add `g * GROUP_LATENTS` themselves --
// as the reference does.
inline std::span<const std::uint16_t> tx_perm(int group) {
    return {TX_PERMS_DATA + static_cast<std::size_t>(group) * TX_PERM_LEN,
            static_cast<std::size_t>(TX_PERM_LEN)};
}

}  // namespace sstvae::framing
