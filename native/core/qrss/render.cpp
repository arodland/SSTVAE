// Port of render.py's g_shrink, reference_w and decoder_planes; see render.hpp.

#include "qrss/render.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "math/npstats.hpp"
#include "qrss/picture.hpp"

namespace sstvae::qrss::render {

namespace {

constexpr std::size_t G = static_cast<std::size_t>(GROUP_LATENTS);
constexpr std::size_t TOTAL = static_cast<std::size_t>(N_GROUPS) * G;

static_assert(N_GROUPS == picture::N_GROUPS && GROUP_LATENTS == picture::GROUP_LATENTS,
              "render's group geometry is picture's");

// np.maximum(w, 0.0): NaN stays NaN; -0.0 and anything not above 0 give 0.0.
inline double max0(double w) { return std::isnan(w) ? w : (w > 0.0 ? w : 0.0); }

inline double shrink(double w) {
    const double W = max0(w);
    return G_MAX * std::sqrt(W / (W + G_KNEE));
}

void check_size(std::size_t n) {
    if (n != TOTAL)
        throw std::invalid_argument("cannot reshape array of size " + std::to_string(n) + " into shape (" +
                                    std::to_string(N_GROUPS) + "," + std::to_string(GROUP_LATENTS) + ")");
}

bool any_positive(std::span<const double> W, std::size_t g) {
    for (std::size_t i = 0; i < G; ++i)
        if (W[g * G + i] > 0) return true;
    return false;
}

// reference_w after the reshape check.
double reference_w_checked(std::span<const double> W, std::int64_t mode) {
    bool have = false;
    double best = 0.0;
    for (std::int64_t g = 0; g <= mode; ++g) {
        if (g >= N_GROUPS)
            throw std::out_of_range("index " + std::to_string(g) + " is out of bounds for axis 0 with size " +
                                    std::to_string(N_GROUPS));
        const auto gg = static_cast<std::size_t>(g);
        if (!any_positive(W, gg)) continue;
        const std::span<const std::uint16_t> a2c = picture::air_to_canonical(g);
        std::vector<double> sel(a2c.size());
        for (std::size_t i = 0; i < a2c.size(); ++i) sel[i] = W[gg * G + a2c[i]];
        const double med = math::median(sel);
        // Python's max() over the list: a later value replaces only if greater.
        if (!have || med > best) best = med;
        have = true;
    }
    return have ? best : 0.0;
}

}  // namespace

std::vector<double> g_shrink(std::span<const double> W) {
    std::vector<double> out(W.size());
    for (std::size_t i = 0; i < W.size(); ++i) out[i] = shrink(W[i]);
    return out;
}

double reference_w(std::span<const double> W, std::int64_t mode) {
    check_size(W.size());
    return reference_w_checked(W, mode);
}

std::pair<std::vector<float>, std::vector<float>> decoder_planes(std::span<const double> S,
                                                                 std::span<const double> W, std::int64_t mode) {
    if (mode < 0 || mode >= N_GROUPS)
        throw std::invalid_argument("mode must be 0.." + std::to_string(N_GROUPS - 1) + ", not " +
                                    std::to_string(mode));
    check_size(S.size());
    check_size(W.size());
    bool bad = false;
    for (const double w : W)
        if (w < 0 || !std::isfinite(w)) bad = true;
    for (const double s : S)
        if (!std::isfinite(s)) bad = true;
    if (bad) throw std::invalid_argument("W must be finite and non-negative, S finite");
    std::vector<std::uint8_t> sent(TOTAL, 0);
    for (std::int64_t g = 0; g <= mode; ++g) {
        const auto gg = static_cast<std::size_t>(g);
        if (!any_positive(W, gg)) continue;
        for (const std::uint16_t c : picture::air_to_canonical(g)) sent[gg * G + c] = 1;
    }
    const double w_ref = reference_w_checked(W, mode);
    std::pair<std::vector<float>, std::vector<float>> out;
    out.first.assign(TOTAL, 0.0f);
    out.second.assign(TOTAL, 0.0f);
    for (std::size_t i = 0; i < TOTAL; ++i) {
        if (!sent[i] || !(W[i] > 0)) continue;
        // g_shrink(W) * S / W, then min(sqrt(W / w_ref), 1) (or 1)
        const double lat = (shrink(W[i]) * S[i]) / W[i];
        double wt = 1.0;
        if (w_ref > 0) {
            const double r = std::sqrt(W[i] / w_ref);
            wt = (r < 1.0 || std::isnan(r)) ? r : 1.0;
        }
        out.first[i] = static_cast<float>(lat);
        out.second[i] = static_cast<float>(wt);
    }
    return out;
}

std::pair<std::vector<float>, std::vector<float>> decoder_planes(std::span<const float> S, std::span<const float> W,
                                                                 std::int64_t mode) {
    const std::vector<double> s(S.begin(), S.end()), w(W.begin(), W.end());
    return decoder_planes(std::span<const double>(s), std::span<const double>(w), mode);
}

}  // namespace sstvae::qrss::render
