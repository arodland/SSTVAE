// A3's false-alarm calibration (TBD_GUMBEL_*) re-measured on the C++
// statistic with no Python: the procedure of test_qrss_acquire.py's
// `_noise_tbd_z` -- complex white noise at 500 Hz (0.25 Hz bins, +-250 Hz),
// spectrogram frames every 2 s over the whole 2000-bin spectrum,
// fftshifted, cfar_normalise - 1, six disjoint 321-bin channels from bin
// 37, Z of each channel's best Viterbi path -- through
// native/core/qrss/acquire_stats and acquire_tbd, and the Gumbel by
// moments (beta = sd sqrt 6 / pi, mu = mean - gamma beta) held to the
// committed constants.
//
// The noise is this file's own (splitmix64 through Box-Muller), so this is
// a sample independent of the numpy seeds the constants were fitted to.
// tests/test_qrss_native_calibration.py is the other half: there the C++ Z
// is the reference's bit for bit on numpy's noise, on the 2,400 values per
// length the constants came from. A normalisation slip (a median over
// other bins, a CFAR in another precision, a path summed in another order)
// moves every false-alarm rate built on these constants with no other
// ctest failing, since a calibration is only ever checked on noise; a 1%
// scale error in the CFAR alone moves mean Z by ~0.8 at 896 frames.
//
// Bars (measured 2026-10-14 in brackets; n = 31 / 132 / 331 / 896 frames,
// 200 / 150 / 80 / 40 captures, 1200 / 900 / 480 / 240 Z): mean Z within 4
// standard errors of the committed Gumbel's mean [|d| / se 0.38 / 0.17 /
// 0.32 / 1.06]; sd within 4 standard errors of the committed Gumbel's,
// relative (Gumbel kurtosis 5.4) [0.57 / 0.46 / 0.50 / 1.24]; and past the
// committed 10% and 1% points no more than 1.25 and 1.0 of nominal plus 4
// binomial sd [1.04 / 1.10 / 1.04 / 1.04 and 0.25 / 0.11 / 0.00 / 0.83 of
// nominal]. By moments: mu 9.842 / 18.627 / 28.927 / 47.225, beta 0.690 /
// 0.637 / 0.639 / 0.651. Mutation-tested: E scaled by 1.01 before the - 1
// fails every length (mean Z 6 / 12 / 14 / 16 se out). ~16 s at
// RelWithDebInfo.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include "check.hpp"
#include "infra_testgen.hpp"
#include "qrss/acquire_stats.hpp"
#include "qrss/acquire_tbd.hpp"
#include "qrss/rx_config.hpp"

using namespace sstvae;
namespace acq = sstvae::qrss::acquire;
namespace rxa = sstvae::qrss::rxconfig::acquire;
using sstvae::qrss_testgen::splitmix;
using sstvae::qrss_testgen::unit;

namespace {

constexpr double CAL_FS = 500.0;  // test_qrss_acquire.CAL_FS
constexpr int N_CH = 6;
constexpr std::size_t COL0 = 37;

// Z of N_CH channels of one noise capture of n_frames frames.
std::vector<double> capture_z(int n_frames, std::uint64_t seed) {
    const auto L = static_cast<std::size_t>(acq::frame_len(CAL_FS));
    const auto hop = static_cast<std::size_t>(rxa::STFT_HOP_S * CAL_FS);
    const std::size_t n = static_cast<std::size_t>(n_frames - 1) * hop + L;
    const auto z = splitmix(seed, 2 * n);
    std::vector<std::complex<double>> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double r = std::sqrt(-2.0 * std::log(1.0 - unit(z[2 * i])));
        const double th = 2.0 * std::numbers::pi * unit(z[2 * i + 1]);
        x[i] = {r * std::cos(th) / std::numbers::sqrt2, r * std::sin(th) / std::numbers::sqrt2};
    }
    std::vector<double> starts(static_cast<std::size_t>(n_frames));
    for (std::size_t t = 0; t < starts.size(); ++t) starts[t] = static_cast<double>(t * hop);
    const acq::Grid32 P = acq::spectrogram(std::span<const std::complex<double>>(x), CAL_FS, starts);
    check::equal(P.cols, L, "all bins");
    acq::Grid32 S(P.rows, P.cols);  // np.fft.fftshift(P, axes=1), L even
    for (std::size_t i = 0; i < P.rows; ++i)
        for (std::size_t j = 0; j < P.cols; ++j) S(i, j) = P(i, (j + L / 2) % L);
    acq::Grid32 E = acq::cfar_normalise(S);
    for (float& v : E.data) v -= 1.0f;  // float32 - 1, as numpy's
    const auto w = static_cast<std::size_t>(2 * static_cast<int>(std::lround(rxa::TBD_HALF_HZ / 0.25)) + 1);
    std::vector<double> out;
    for (int k = 0; k < N_CH; ++k) {
        acq::Grid32 ch(E.rows, w);
        for (std::size_t i = 0; i < E.rows; ++i)
            for (std::size_t j = 0; j < w; ++j) ch(i, j) = E(i, COL0 + static_cast<std::size_t>(k) * w + j);
        out.push_back(acq::tbd_viterbi(ch).z);
    }
    return out;
}

double gumbel_point(double mu, double be, double p) { return mu - be * std::log(-std::log1p(-p)); }

void measure(int n_frames, int n_caps) {
    std::vector<double> Z;
    for (int c = 0; c < n_caps; ++c) {
        const auto zc = capture_z(n_frames, 0x5EED0000ULL + static_cast<std::uint64_t>(n_frames) * 1000 + c);
        Z.insert(Z.end(), zc.begin(), zc.end());
    }
    const auto N = static_cast<double>(Z.size());
    double mean = 0.0;
    for (double v : Z) mean += v;
    mean /= N;
    double var = 0.0;
    for (double v : Z) var += (v - mean) * (v - mean);
    const double sd = std::sqrt(var / N);  // np.std, ddof 0
    const auto [mu_c, be_c] = acq::tbd_gumbel(n_frames);
    const double mean_c = mu_c + std::numbers::egamma * be_c;
    const double sd_c = be_c * std::numbers::pi / std::sqrt(6.0);
    const double se_mean = sd_c / std::sqrt(N);
    const double se_sd = std::sqrt((5.4 - 1.0) / (4.0 * N));  // relative, Gumbel excess kurtosis 2.4
    const double d_mean = (mean - mean_c) / se_mean;
    const double d_sd = (sd / sd_c - 1.0) / se_sd;
    const std::string tag = "n_frames " + std::to_string(n_frames);
    check::is_true(std::abs(d_mean) <= 4.0, tag + ": mean Z " + std::to_string(mean) + " against " +
                                                std::to_string(mean_c));
    check::is_true(std::abs(d_sd) <= 4.0, tag + ": sd Z " + std::to_string(sd) + " against " +
                                              std::to_string(sd_c));
    double tail[2];
    const double ps[2] = {0.1, 0.01}, caps[2] = {1.25, 1.0};
    for (int k = 0; k < 2; ++k) {
        const double zp = gumbel_point(mu_c, be_c, ps[k]);
        const auto past = static_cast<double>(std::count_if(Z.begin(), Z.end(), [&](double v) { return v > zp; }));
        tail[k] = past / N / ps[k];
        const double bound = caps[k] * ps[k] * N + 4.0 * std::sqrt(ps[k] * (1 - ps[k]) * N);
        check::is_true(past <= bound, tag + ": past the " + std::to_string(ps[k]) + " point " +
                                          std::to_string(past) + " > " + std::to_string(bound));
    }
    const double be = sd * std::sqrt(6.0) / std::numbers::pi;
    std::printf("n_frames %d: %d Z, mu %.4f beta %.4f (committed %.3f %.3f); mean |d|/se %.2f, sd |d|/se %.2f; "
                "exceedance/nominal at 10%%, 1%%: %.2f, %.2f\n",
                n_frames, static_cast<int>(N), mean - std::numbers::egamma * be, be, mu_c, be_c,
                std::abs(d_mean), std::abs(d_sd), tail[0], tail[1]);
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    try {
        check::equal(rxa::TBD_GUMBEL_N.size(), std::size_t{4}, "four lengths");
        const int caps[4] = {200, 150, 80, 40};
        for (std::size_t i = 0; i < rxa::TBD_GUMBEL_N.size(); ++i) measure(rxa::TBD_GUMBEL_N[i], caps[i]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_calibration");
}
