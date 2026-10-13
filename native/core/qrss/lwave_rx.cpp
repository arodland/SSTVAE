// Port of lwave.py's matched_filter and PULSE_NORM; see lwave_rx.hpp.

#include "qrss/lwave_rx.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/ce_rx.hpp"
#include "qrss/config.hpp"
#include "qrss/lwave.hpp"
#include "qrss/npsum.hpp"

namespace sstvae::qrss::lwave_rx {

double pulse_norm() {
    // float(np.sqrt(np.sum(ce._rrc(_M / L_T_NUM) ** 2) / L_T_NUM)), _M = -HALF..HALF
    static const double v = [] {
        std::vector<double> sq(static_cast<std::size_t>(2 * lwave::HALF + 1));
        for (std::int64_t m = -lwave::HALF; m <= lwave::HALF; ++m) {
            const double h = ce::rrc(static_cast<double>(m) / static_cast<double>(config::L_T_NUM));
            sq[static_cast<std::size_t>(m + lwave::HALF)] = h * h;
        }
        return std::sqrt(npsum::sum(std::span<const double>(sq)) / static_cast<double>(config::L_T_NUM));
    }();
    return v;
}

std::vector<double> matched_filter(std::span<const double> z, std::int64_t fs, std::int64_t n0,
                                   std::span<const std::int64_t> pos) {
    return ce_rx::matched_filter_table(z, lwave::grid_step(fs), n0, pos, config::L_T_NUM, lwave::HALF,
                                       lwave::pulse_table());
}

std::vector<cplx> matched_filter(std::span<const cplx> z, std::int64_t fs, std::int64_t n0,
                                 std::span<const std::int64_t> pos) {
    return ce_rx::matched_filter_table(z, lwave::grid_step(fs), n0, pos, config::L_T_NUM, lwave::HALF,
                                       lwave::pulse_table());
}

}  // namespace sstvae::qrss::lwave_rx
