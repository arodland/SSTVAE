#include "ofdm/ofdm.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace sstvae::ofdm {
namespace {

using config::CARRIER0;
using config::FCENTER;
using config::FS;
using config::NCP;
using config::PREAMBLE_CP;
using config::PREAMBLE_SAMPLES;
using config::RS;

constexpr double PI = std::numbers::pi;
constexpr double TWO_PI = 2.0 * PI;

// exp(2i*pi*n*f/FS), with the argument reduced to one turn *exactly*,
// in integer arithmetic, before any floating-point work happens.
//
// Every frequency here is an integer number of Hz and every sample
// index is an integer, so n*f/FS has an exact integer remainder and the
// phasor depends only on (n*f mod FS)/FS. Two things follow, and both
// matter more than they look:
//
// * **Accuracy.** Computing the unreduced argument gives |theta| up to
//   262 rad, where one ulp is 5.7e-14 -- so the phasor inherits ~1e-13
//   of error from the argument alone. The reference has exactly this
//   problem: sstvae/modem/ofdm.py's MOD_MATRIX is correctly rounded for
//   some entries and a ulp out for others, and the resulting phasors
//   differ from the true value by up to ~3e-14. Reduced, |theta| < 2*pi
//   and one ulp is 8.9e-16.
//
// * **Cross-platform determinism**, which is the real reason. sin/cos
//   of 262 rad disagree between glibc, musl and MSVC by far more than
//   they do near zero, because they differ in how far they carry
//   argument reduction. Under 2*pi they agree to well under an ulp. The
//   parity tolerance therefore has to hold on three platforms, and this
//   is what makes that a safe promise rather than a hope.
//
// The residual disagreement with Python is bounded by Python's own
// large-argument error, not by ours -- see PHASOR_TOL in the tests.
cdouble carrier_phasor(std::int64_t n, std::int64_t f) {
    const std::int64_t num = n * f;
    // Euclidean remainder: n is negative for the cyclic prefix.
    std::int64_t q = num % FS;
    if (q < 0) q += FS;
    return std::polar(1.0, TWO_PI * (static_cast<double>(q) / static_cast<double>(FS)));
}

struct Matrices {
    // Frequencies are held as integers as well as doubles: the integer
    // copies are what carrier_phasor needs for exact argument
    // reduction, and the doubles are the public API's shape.
    std::array<std::int64_t, NC> carrier_hz{};
    std::array<std::int64_t, NC> baseband_hz{};
    std::array<double, NC> carrier{};
    std::array<double, NC> baseband{};
    std::vector<cdouble> mod;    // (NSYM, NC)
    std::vector<cdouble> demod;  // (NC, M)
    std::array<cdouble, NC> pilot{};

    Matrices()
        : mod(static_cast<std::size_t>(NSYM) * NC),
          demod(static_cast<std::size_t>(NC) * M) {
        for (int k = 0; k < NC; ++k) {
            const std::size_t u = static_cast<std::size_t>(k);
            carrier_hz[u] = CARRIER0 + static_cast<std::int64_t>(RS) * k;
            baseband_hz[u] = carrier_hz[u] - FCENTER;
            carrier[u] = static_cast<double>(carrier_hz[u]);
            baseband[u] = static_cast<double>(baseband_hz[u]);
        }

        for (int n = 0; n < NSYM; ++n) {
            for (int k = 0; k < NC; ++k)
                mod[static_cast<std::size_t>(n) * NC + static_cast<std::size_t>(k)] =
                    carrier_phasor(n - NCP, carrier_hz[static_cast<std::size_t>(k)]);
        }

        for (int k = 0; k < NC; ++k) {
            for (int n = 0; n < M; ++n)
                demod[static_cast<std::size_t>(k) * M + static_cast<std::size_t>(n)] =
                    carrier_phasor(-n, baseband_hz[static_cast<std::size_t>(k)]);
        }

        // Zadoff-Chu as an exact rational turn: phase = 2*pi * NUM[k] / DEN.
        // Not the closed form -pi*u*k^2/NC -- that argument reaches -69 rad,
        // where sin/cos differ between libms, and the reference reduces for
        // exactly that reason. Same expression, same values, both sides.
        for (int k = 0; k < NC; ++k) {
            const double phase =
                2.0 * PI *
                static_cast<double>(config::PILOT_PHASE_NUM[static_cast<std::size_t>(k)]) /
                static_cast<double>(config::PILOT_PHASE_DEN);
            pilot[static_cast<std::size_t>(k)] = std::polar(1.0, phase);
        }
    }
};

const Matrices& tables() {
    static const Matrices m;
    return m;
}

// The preamble and pilot replicas differ only in their frequency set,
// their sample count and an overall scale, so they share one builder
// rather than three near-copies that could drift apart.
std::vector<cdouble> replica(const std::array<std::int64_t, NC>& freqs, int n_samples,
                             int origin, double scale) {
    const auto& p = tables().pilot;
    std::vector<cdouble> out(static_cast<std::size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        cdouble acc{0.0, 0.0};
        for (int k = 0; k < NC; ++k)
            acc += carrier_phasor(i - origin, freqs[static_cast<std::size_t>(k)]) *
                   p[static_cast<std::size_t>(k)];
        out[static_cast<std::size_t>(i)] = scale * acc;
    }
    return out;
}

}  // namespace

const std::array<double, NC>& carrier_freqs() { return tables().carrier; }
const std::array<double, NC>& baseband_freqs() { return tables().baseband; }
std::span<const cdouble> mod_matrix() { return tables().mod; }
std::span<const cdouble> demod_matrix() { return tables().demod; }
const std::array<cdouble, NC>& pilot_sequence() { return tables().pilot; }

std::vector<double> modulate_symbols(std::span<const cdouble> symbols,
                                     std::size_t n_sym) {
    if (symbols.size() != n_sym * NC)
        throw std::invalid_argument("modulate_symbols: expected n_sym * NC symbols");
    const auto& mod = tables().mod;
    std::vector<double> out(n_sym * NSYM);
    for (std::size_t i = 0; i < n_sym; ++i) {
        const cdouble* s = symbols.data() + i * NC;
        for (int n = 0; n < NSYM; ++n) {
            const cdouble* row = mod.data() + static_cast<std::size_t>(n) * NC;
            // Accumulated as complex and only then reduced to its real
            // part, as `np.real(MOD_MATRIX @ symbols.T)` does. Summing
            // the real parts alone would be algebraically identical and
            // would round differently.
            cdouble acc{0.0, 0.0};
            for (int k = 0; k < NC; ++k) acc += row[k] * s[k];
            out[i * NSYM + static_cast<std::size_t>(n)] = acc.real();
        }
    }
    return out;
}

std::array<cdouble, NC> demod_window(std::span<const cdouble> z, std::int64_t start,
                                     std::int64_t backoff) {
    const std::int64_t s = start - backoff;
    if (s < 0)
        throw std::invalid_argument(
            "demod_window: window starts before the signal; the Python "
            "reference wraps a negative slice here and returns garbage");

    // Zero-padded copy of the useful window, so the matrix product below
    // is a plain fixed-length loop no matter where the window lands.
    std::array<cdouble, M> win{};
    const std::size_t us = static_cast<std::size_t>(s);
    const std::size_t avail = us < z.size() ? std::min<std::size_t>(M, z.size() - us) : 0;
    for (std::size_t i = 0; i < avail; ++i) win[i] = z[us + i];

    const auto& dm = tables().demod;
    std::array<cdouble, NC> out{};
    const double gain = 2.0 / static_cast<double>(M);
    for (int k = 0; k < NC; ++k) {
        const cdouble* row = dm.data() + static_cast<std::size_t>(k) * M;
        cdouble acc{0.0, 0.0};
        for (int n = 0; n < M; ++n) acc += row[n] * win[static_cast<std::size_t>(n)];
        out[static_cast<std::size_t>(k)] = gain * acc;
    }
    return out;
}

std::vector<double> preamble_waveform() {
    const std::vector<cdouble> e =
        replica(tables().carrier_hz, PREAMBLE_SAMPLES, PREAMBLE_CP, 1.0);
    std::vector<double> out(e.size());
    for (std::size_t i = 0; i < e.size(); ++i) out[i] = e[i].real();
    return out;
}

std::vector<cdouble> preamble_template() {
    return replica(tables().baseband_hz, PREAMBLE_SAMPLES, PREAMBLE_CP, 0.5);
}

std::vector<cdouble> pilot_template() {
    return replica(tables().baseband_hz, M, 0, 0.5);
}

}  // namespace sstvae::ofdm
