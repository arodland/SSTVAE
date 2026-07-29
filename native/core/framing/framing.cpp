#include "framing/framing.hpp"

#include <cmath>
#include <stdexcept>

#include "golay/golay.hpp"

namespace sstvae::framing {
namespace {

using config::DATA_SYMS_PER_FRAME;
using config::FRAMES_PER_GROUP;
using config::GROUP_LATENTS;
using config::LATENTS_PER_FRAME;
using config::NC_LATENT;
using config::PROTOCOL_VERSION;
using config::TRANSMIT_LATENTS_PER_GROUP;

int check_nibble(int mode_idx, int version) {
    return (mode_idx ^ version ^ 0xA) & 0xF;
}

}  // namespace

FrameSlots slot_range_for_frame(int abs_frame) {
    if (abs_frame < 0 || abs_frame >= config::LATENT_GROUPS * FRAMES_PER_GROUP)
        throw std::out_of_range("slot_range_for_frame: frame outside mode C's range");
    const int g = abs_frame / FRAMES_PER_GROUP;
    const int fg = abs_frame % FRAMES_PER_GROUP;
    const std::size_t slo = static_cast<std::size_t>(fg) * LATENTS_PER_FRAME;

    const std::span<const std::uint16_t> perm = tx_perm(g);
    FrameSlots out;
    out.group = g;
    out.indices.resize(LATENTS_PER_FRAME);
    // The group offset is added in 64-bit. The table is uint16 because
    // the indices fit; the *offsets* reach 2*GROUP_LATENTS = 105,600 and
    // do not. Python has the identical hazard -- see framing.py, where
    // the frozen array is widened on load.
    const std::int64_t base = static_cast<std::int64_t>(g) * GROUP_LATENTS;
    for (int i = 0; i < LATENTS_PER_FRAME; ++i)
        out.indices[static_cast<std::size_t>(i)] =
            base + static_cast<std::int64_t>(perm[slo + static_cast<std::size_t>(i)]);
    return out;
}

std::vector<double> interleave(std::span<const double> latents,
                               const ModeSpec& mode) {
    if (latents.size() != static_cast<std::size_t>(mode.n_latents))
        throw std::invalid_argument("interleave: wrong latent count for this mode");
    std::vector<double> out(static_cast<std::size_t>(mode.n_tx_latents));
    for (int g = 0; g < mode.groups; ++g) {
        const std::span<const std::uint16_t> perm = tx_perm(g);
        const std::size_t lo = static_cast<std::size_t>(g) * GROUP_LATENTS;
        const std::size_t slo =
            static_cast<std::size_t>(g) * TRANSMIT_LATENTS_PER_GROUP;
        for (int i = 0; i < TRANSMIT_LATENTS_PER_GROUP; ++i)
            out[slo + static_cast<std::size_t>(i)] =
                latents[lo + perm[static_cast<std::size_t>(i)]];
    }
    return out;
}

Deinterleaved deinterleave(std::span<const double> slots, const ModeSpec& mode) {
    if (slots.size() != static_cast<std::size_t>(mode.n_tx_latents))
        throw std::invalid_argument("deinterleave: wrong slot count for this mode");
    Deinterleaved out;
    out.latents.assign(static_cast<std::size_t>(mode.n_latents), 0.0);
    out.weight.assign(static_cast<std::size_t>(mode.n_latents), 0.0);
    for (int g = 0; g < mode.groups; ++g) {
        const std::span<const std::uint16_t> perm = tx_perm(g);
        const std::size_t lo = static_cast<std::size_t>(g) * GROUP_LATENTS;
        const std::size_t slo =
            static_cast<std::size_t>(g) * TRANSMIT_LATENTS_PER_GROUP;
        for (int i = 0; i < TRANSMIT_LATENTS_PER_GROUP; ++i) {
            const std::size_t idx = lo + perm[static_cast<std::size_t>(i)];
            out.latents[idx] = slots[slo + static_cast<std::size_t>(i)];
            out.weight[idx] = 1.0;
        }
    }
    return out;
}

std::vector<cdouble> slots_to_symbols(std::span<const double> frame_slots) {
    if (frame_slots.size() != LATENTS_PER_FRAME)
        throw std::invalid_argument("slots_to_symbols: expected LATENTS_PER_FRAME");
    // (DATA_SYMS_PER_FRAME, NC_LATENT, 2) -> (DATA_SYMS_PER_FRAME, NC_LATENT)
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    std::vector<cdouble> out(static_cast<std::size_t>(DATA_SYMS_PER_FRAME) *
                             NC_LATENT);
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = cdouble(frame_slots[2 * i], frame_slots[2 * i + 1]) * inv_sqrt2;
    return out;
}

std::vector<double> symbols_to_slots(std::span<const cdouble> symbols) {
    const double sqrt2 = std::sqrt(2.0);
    std::vector<double> out(symbols.size() * 2);
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        out[2 * i] = symbols[i].real() * sqrt2;
        out[2 * i + 1] = symbols[i].imag() * sqrt2;
    }
    return out;
}

std::vector<int> header_bits(const ModeSpec& mode) {
    int data = (mode.index & 0xF) | ((PROTOCOL_VERSION & 0xF) << 4);
    data |= check_nibble(mode.index, PROTOCOL_VERSION) << 8;
    const auto bits = golay::codeword_bits(data);
    return std::vector<int>(bits.begin(), bits.end());
}

std::vector<cdouble> header_symbol(const ModeSpec& mode) {
    const std::vector<int> bits = header_bits(mode);
    std::vector<cdouble> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i)
        out[i] = cdouble(1.0 - 2.0 * static_cast<double>(bits[i]), 0.0);
    return out;
}

std::optional<ModeSpec> decode_header(std::span<const double> soft) {
    const int data = golay::decode_soft(soft);
    const int mode_idx = data & 0xF;
    const int version = (data >> 4) & 0xF;
    const int check = (data >> 8) & 0xF;
    if (version != PROTOCOL_VERSION || check != check_nibble(mode_idx, version))
        return std::nullopt;
    for (const ModeSpec& m : config::MODES)
        if (m.index == mode_idx) return m;
    return std::nullopt;
}

}  // namespace sstvae::framing
