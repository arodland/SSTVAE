#include "latents/latents.hpp"

#include <algorithm>

namespace sstvae::latents {

std::vector<double> pad_to_full(const std::vector<double>& vec, double fill) {
    std::vector<double> out(static_cast<std::size_t>(N_LATENTS), fill);
    std::copy(vec.begin(), vec.begin() + std::min(vec.size(), out.size()), out.begin());
    return out;
}

}  // namespace sstvae::latents
