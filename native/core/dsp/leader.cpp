#include "dsp/leader.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

#include "config.hpp"

namespace sstvae::dsp {

namespace {

constexpr double PI = std::numbers::pi;

// The occupied band, from config rather than spelled out: the leader
// should sweep exactly what the transmission itself occupies, so any
// filtering that passes the signal passes the leader too, and moving a
// carrier constant cannot leave this behind.
constexpr double kLow = config::CARRIER0;
constexpr double kHigh = config::CARRIER0 + (config::NC - 1) * config::RS;

}  // namespace

std::vector<double> vox_leader(double seconds, int sample_rate, double amplitude) {
    if (seconds <= 0.0 || sample_rate <= 0) return {};

    const auto n = static_cast<std::size_t>(std::lround(seconds * sample_rate));
    if (n == 0) return {};

    const auto sweep_n = static_cast<std::size_t>(
        std::max<std::int64_t>(1, std::llround(VOX_SWEEP_S * sample_rate)));
    const double rate = (kHigh - kLow) / static_cast<double>(sweep_n);  // Hz/sample

    // Ramp over 5 ms, or half the leader if it is shorter than 10 ms --
    // the same shape and the same fallback as the CW keying, for the
    // same reason.
    const std::size_t ramp_n =
        std::min(n / 2, static_cast<std::size_t>(std::lround(0.005 * sample_rate)));

    std::vector<double> out(n);
    // Phase in *cycles*, accumulated and wrapped every sample rather than
    // computed from a running time. Same rule as everywhere else in this
    // project's DSP: reduce the argument exactly before the
    // transcendental, so the result is a property of the signal and not
    // of the machine's argument reduction.
    double phase = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        // Position within the current sweep. Phase carries across the
        // wrap, so the jump back to the low edge is a discontinuity in
        // frequency and not in the waveform -- no click.
        const double freq = kLow + rate * static_cast<double>(i % sweep_n);

        double env = 1.0;
        if (ramp_n > 0) {
            if (i < ramp_n) {
                env = 0.5 - 0.5 * std::cos(PI * static_cast<double>(i) /
                                           static_cast<double>(ramp_n));
            } else if (i >= n - ramp_n) {
                env = 0.5 - 0.5 * std::cos(PI * static_cast<double>(n - 1 - i) /
                                           static_cast<double>(ramp_n));
            }
        }

        out[i] = amplitude * env * std::sin(2.0 * PI * phase);
        phase = std::fmod(phase + freq / sample_rate, 1.0);
    }
    return out;
}

}  // namespace sstvae::dsp
