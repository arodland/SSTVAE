// Port of ce.py's mean_template, _mean_template_grid and matched_filter;
// see ce_rx.hpp.

#include "qrss/ce_rx.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "qrss/qrss_config.hpp"
#include "qrss/lwave.hpp"
#include "qrss/npsum.hpp"

namespace sstvae::qrss::ce_rx {

namespace {

// 2 * np.pi, as the reference's `phi / (2 * np.pi)` has it.
constexpr double TWO_PI = 2 * 3.141592653589793;

// Python's (and numpy's) floor division of int64s.
inline std::int64_t floordiv(std::int64_t a, std::int64_t b) {
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

inline std::int64_t clip(std::int64_t x, std::int64_t lo, std::int64_t hi) {
    // np.clip(x, lo, hi) = minimum(maximum(x, lo), hi): hi wins when lo > hi
    return std::min(std::max(x, lo), hi);
}

void check_shape(const FrameSpec& spec, std::size_t n, const char* name) {
    const std::int64_t shape[1] = {static_cast<std::int64_t>(n)};
    ce::check_sym(spec, shape, name);
}

// keyed * a * _turns_to_phasor(phi / (2 pi) + theta) * np.exp(-v / 2), each
// product numpy's: a float64 array times a complex128 one goes through the
// complex multiply with the real operand cast to (x, 0).
std::vector<cplx> compose(std::span<const double> tau, const std::vector<double>& phi, const std::vector<double>& v,
                          double lo, double hi, const ce::CwPhase& cw) {
    std::vector<cplx> out(tau.size());
    for (std::size_t i = 0; i < tau.size(); ++i) {
        const double keyed = (tau[i] >= lo && tau[i] < hi) ? 1.0 : 0.0;
        const double ka = keyed * cw.a[i];
        const cplx ph = ce::turns_to_phasor(phi[i] / TWO_PI + cw.theta[i]);
        const double e = std::exp(-v[i] / 2);
        out[i] = lwave::np_mul(lwave::np_mul(cplx(ka, 0.0), ph), cplx(e, 0.0));
    }
    return out;
}

template <typename T>
std::vector<T> mf_impl(std::span<const T> z, std::int64_t step, std::int64_t n0, std::span<const std::int64_t> pos,
                       std::int64_t t_num, std::int64_t half, std::span<const double> table) {
    const std::int64_t width = 2 * half / step + 2;
    const auto nz = static_cast<std::int64_t>(z.size());
    std::vector<T> out(pos.size());
    std::vector<T> row(static_cast<std::size_t>(width));
    if (!pos.empty() && nz == 0)
        throw std::out_of_range("index -1 is out of bounds for axis 0 with size 0");
    for (std::size_t k = 0; k < pos.size(); ++k) {
        const std::int64_t p = pos[k];
        const std::int64_t first = -floordiv(half - t_num * p, step);   // ceil((t_num p - half) / step)
        for (std::int64_t j = 0; j < width; ++j) {
            const std::int64_t nn = first + j;
            const std::int64_t mm = step * nn - t_num * p;
            const std::int64_t idx = nn - n0;
            const bool ok = (mm <= half && mm >= -half) && idx >= 0 && idx < nz;
            const double tap = ok ? table[static_cast<std::size_t>(clip(mm + half, 0, 2 * half))] : 0.0;
            const T zv = z[static_cast<std::size_t>(clip(idx, 0, nz - 1))];
            if constexpr (std::is_same_v<T, cplx>)
                row[static_cast<std::size_t>(j)] = lwave::np_mul(zv, cplx(tap, 0.0));
            else
                row[static_cast<std::size_t>(j)] = zv * tap;
        }
        out[k] = npsum::sum(std::span<const T>(row));
    }
    const double scale = static_cast<double>(step) / static_cast<double>(t_num);   // L / T_NUM
    for (T& o : out) {
        if constexpr (std::is_same_v<T, cplx>)
            o = lwave::np_mul(o, cplx(scale, 0.0));
        else
            o = o * scale;
    }
    return out;
}

}  // namespace

// CE's layout, whose slot counts an L frame never fits: its header has
// 1,237 non-reference symbols for 2,474 bits and its data half as many
// symbols as latents. (An L frame with no data and no header would pass
// the reference's assertions and be laid out as CE; none exists, and that
// one is refused too.)
void require_ce_layout(const FrameSpec& spec) {
    if (spec.waveform() == frame::Waveform::CE) return;
    const std::int64_t n_sym = spec.n_sym(), n_pre = spec.n_pre(), n_hdr = spec.n_hdr();
    std::int64_t hdr_nonref = 0, data = 0;
    for (std::int64_t s = 0; s < n_sym; ++s) {
        const std::int64_t j = s - n_pre;
        const bool is_ref = j >= 0 && j % config::REF_PERIOD == 0;
        if (is_ref) continue;
        if (s >= n_pre && s < n_pre + n_hdr) ++hdr_nonref;
        if (s >= n_pre + n_hdr) ++data;
    }
    if (n_hdr && hdr_nonref != config::N_HDR_BITS + 1)
        throw frame::WrongWaveform(std::to_string(hdr_nonref) + " header slots for " +
                                   std::to_string(config::N_HDR_BITS) + " bits");
    if (data != spec.n_data())
        throw frame::WrongWaveform(std::to_string(data) + " data slots for " + std::to_string(spec.n_data()) +
                                   " latents");
    throw frame::WrongWaveform("ce.mean_template is CE's; " + frame::py_repr(spec.name()) + " is waveform 1");
}

std::vector<cplx> mean_template(std::span<const double> tau, std::span<const double> mu, std::span<const double> nu,
                                const FrameSpec& spec, const ce::Keying* keying, bool ook,
                                std::optional<double> keyed_from_pos) {
    check_shape(spec, mu.size(), "sym");
    require_ce_layout(spec);
    check_shape(spec, nu.size(), "var");
    const auto [phi, v] = ce::phase_at(tau, mu, nu, spec);
    const double lo = keyed_from_pos ? *keyed_from_pos : spec.keyed_start_pos();
    const ce::CwPhase cw = ce::cw_phase_turns(tau, spec, keying, ook);
    return compose(tau, phi, v, lo, spec.keyed_end_pos(), cw);
}

std::vector<cplx> mean_template_grid(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const double> mu,
                                     std::span<const double> nu, const FrameSpec& spec, const ce::Keying* keying,
                                     bool ook) {
    // tau = (np.arange(n0, n0 + n, dtype=np.int64) * _grid_step(fs)) / T_NUM
    const std::int64_t step = ce::grid_step(fs);
    const std::size_t count = n > 0 ? static_cast<std::size_t>(n) : 0;
    std::vector<double> tau(count);
    for (std::size_t i = 0; i < count; ++i)
        tau[i] = static_cast<double>((n0 + static_cast<std::int64_t>(i)) * step) / static_cast<double>(config::T_NUM);
    check_shape(spec, mu.size(), "sym");
    require_ce_layout(spec);
    std::vector<double> phi = ce::phase_grid(fs, n0, n, mu, spec);
    check_shape(spec, nu.size(), "var");
    std::vector<double> v = ce::var_grid(fs, n0, n, nu, spec);
    phi.resize(count, 0.0);
    v.resize(count, 0.0);
    const ce::CwPhase cw = ce::cw_phase_turns(tau, spec, keying, ook);
    return compose(tau, phi, v, spec.keyed_start_pos(), spec.keyed_end_pos(), cw);
}

std::vector<double> matched_filter_table(std::span<const double> z, std::int64_t step, std::int64_t n0,
                                         std::span<const std::int64_t> pos, std::int64_t t_num, std::int64_t half,
                                         std::span<const double> table) {
    return mf_impl<double>(z, step, n0, pos, t_num, half, table);
}

std::vector<cplx> matched_filter_table(std::span<const cplx> z, std::int64_t step, std::int64_t n0,
                                       std::span<const std::int64_t> pos, std::int64_t t_num, std::int64_t half,
                                       std::span<const double> table) {
    return mf_impl<cplx>(z, step, n0, pos, t_num, half, table);
}

std::vector<double> matched_filter(std::span<const double> z, std::int64_t fs, std::int64_t n0,
                                   std::span<const std::int64_t> pos) {
    return matched_filter_table(z, ce::grid_step(fs), n0, pos, config::T_NUM, ce::HALF, ce::pulse_table());
}

std::vector<cplx> matched_filter(std::span<const cplx> z, std::int64_t fs, std::int64_t n0,
                                 std::span<const std::int64_t> pos) {
    return matched_filter_table(z, ce::grid_step(fs), n0, pos, config::T_NUM, ce::HALF, ce::pulse_table());
}

}  // namespace sstvae::qrss::ce_rx
