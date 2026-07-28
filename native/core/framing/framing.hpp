// Frame layout: latent interleaving and the Golay-coded header.
//
// Port of sstvae/modem/framing.py. Latents are ordered by channel group
// (G0 first) so that early-stopping a transmission keeps whole coarse
// groups. Within each group, a fixed pseudo-random permutation spreads
// latents across that group's frames, so a lost frame becomes diffuse
// noise over the whole image rather than a missing region.
//
// Each group has more canonical latents (GROUP_LATENTS) than on-air
// slots (TRANSMIT_LATENTS_PER_GROUP), because one carrier per frame is
// reserved for the beacon side-channel. Only the first
// TRANSMIT_LATENTS_PER_GROUP entries of each group's permutation get a
// slot; the rest are permanently erased (weight 0), never transmitted.

#pragma once

#include <complex>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "config.hpp"
#include "framing/interleaver_table.hpp"

namespace sstvae::framing {

using cdouble = std::complex<double>;
using config::ModeSpec;

// Absolute frame index (0..3*FRAMES_PER_GROUP-1, i.e. mode C's full
// frame range -- every mode is a prefix of it) -> the canonical latent
// indices for that one frame's LATENTS_PER_FRAME slots, in slot order.
//
// Lets a single frame be placed into canonical latent space without
// knowing which mode produced the transmission -- the situation a
// blind, no-header resync lands in.
struct FrameSlots {
    int group;
    std::vector<std::int64_t> indices;  // LATENTS_PER_FRAME canonical indices
};
FrameSlots slot_range_for_frame(int abs_frame);

// Canonical latent vector -> on-air slot order (mode.n_tx_latents).
std::vector<double> interleave(std::span<const double> latents,
                               const ModeSpec& mode);

// On-air slot order -> (canonical latent vector, weight mask).
//
// Weight is 1 everywhere a slot maps to; the latents each group
// permanently drops come back as 0 with weight 0, the same contract as
// a channel erasure.
struct Deinterleaved {
    std::vector<double> latents;
    std::vector<double> weight;
};
Deinterleaved deinterleave(std::span<const double> slots, const ModeSpec& mode);

// One frame's real slot values -> (DATA_SYMS_PER_FRAME, NC_LATENT)
// complex symbols, row-major, covering only the 23 latent-carrying
// carriers (the beacon carrier is handled separately by the caller).
//
// Pairs of consecutive slots map to I/Q of one carrier; 1/sqrt(2) keeps
// unit-RMS latents at unit symbol power.
std::vector<cdouble> slots_to_symbols(std::span<const double> frame_slots);

// Inverse of slots_to_symbols; accepts any (n_sym, NC_LATENT) shape.
std::vector<double> symbols_to_slots(std::span<const cdouble> symbols);

// --- header ---------------------------------------------------------

// 24 codeword bits (0/1) for the header BPSK symbol.
std::vector<int> header_bits(const ModeSpec& mode);

// Header as (NC,) complex BPSK carrier amplitudes.
std::vector<cdouble> header_symbol(const ModeSpec& mode);

// Soft values (one per carrier, summed over header repeats) -> mode.
// Empty when the version or check nibble rejects it.
std::optional<ModeSpec> decode_header(std::span<const double> soft);

}  // namespace sstvae::framing
