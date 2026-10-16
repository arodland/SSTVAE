// Fixtures shared by test_em_api.cpp and test_em_chain.cpp: a picture, a
// genie member pass of it (tests/qrss_fakes.py's make_pass_result, with this
// file's own noise), Gaussian noise from a fixed seed, and a pass's audio
// added into a recording at a given SNR in the QRSS convention (power over
// the keyed samples, noise in 2500 Hz: for real audio at FS, sigma^2 =
// P (FS/2) / (2500 10^(snr/10)), the real-audio form of channel.simulate's
// noise_var = P fs / (2500 10^(snr/10)) on the complex front end).
//
// Nothing here is pinned against Python: the noise is this file's
// (splitmix64 and Box-Muller), the same on every platform to the last bit
// of the doubles it adds (libm's log, sqrt, cos and sin at most an ulp
// apart), which is all the tests need.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include "math/linalg.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/picture.hpp"
#include "qrss/rational.hpp"
#include "qrss/tx.hpp"
#include "qrss/types.hpp"

namespace sstvae::em_fixture {

inline std::uint64_t splitmix64(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// n draws of N(0, 1) from `seed` (Box-Muller on 53-bit uniforms in (0, 1]).
inline std::vector<double> gauss(std::uint64_t seed, std::size_t n) {
    std::vector<double> out(n);
    std::uint64_t s = seed * 0xD1B54A32D192ED03ull + 1;
    for (std::size_t i = 0; i < n; i += 2) {
        const double u1 = (static_cast<double>(splitmix64(s) >> 11) + 1.0) / 9007199254740992.0;
        const double u2 = static_cast<double>(splitmix64(s) >> 11) / 9007199254740992.0;
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i] = r * std::cos(2.0 * std::numbers::pi * u2);
        if (i + 1 < n) out[i + 1] = r * std::sin(2.0 * std::numbers::pi * u2);
    }
    return out;
}

// A mode-0 picture of closed-form latents (test_api.cpp's formula).
inline qrss::picture::StoredPicture picture0() {
    std::vector<double> v(3 * 52800);
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double g = static_cast<double>(i);
        v[i] = std::sin(std::fmod(0.001 * g * g, 6.283)) + 0.3 * std::cos(g);
    }
    return qrss::picture::StoredPicture::from_latents(v, 0xD1D8, 0);
}

inline qrss::header::HeaderFields header_of(const qrss::picture::StoredPicture& sp, std::int64_t segment = 0) {
    qrss::header::HeaderFields h;
    h.callsign = "K1ABC";
    h.grid = "FN42";
    h.picture_id = sp.picture_id;
    h.mode = sp.mode;
    h.segment = segment;
    h.codec_id = sp.codec_id;
    return h;
}

// make_pass_result(slot_air_latents(sp, 0)[:n], s, hdr=h, decoded=True):
// z = a + N(0, 1/s), w = s, its header decoded and its LLRs certain, no
// capture (so EM leaves it as it is), at (q, f_hz).
inline qrss::types::PassResult genie_pass(const qrss::picture::StoredPicture& sp, const qrss::frame::FrameSpec& spec,
                                          double s, std::uint64_t seed, std::int64_t q, double f_hz = 1500.0) {
    namespace types = qrss::types;
    const std::vector<double> a = qrss::tx::slot_air_latents(qrss::tx::Source(sp), 0);
    const auto n = static_cast<std::size_t>(spec.n_data());
    const std::vector<double> g = gauss(seed, n);
    types::PassResult p;
    p.z.resize(n);
    p.w.assign(n, static_cast<float>(s));
    for (std::size_t i = 0; i < n; ++i) p.z[i] = static_cast<float>(a[i] + g[i] / std::sqrt(s));
    const qrss::header::HeaderFields h = header_of(sp);
    const std::vector<std::uint8_t> bits = qrss::header::encode(h);
    p.hdr_llr.resize(bits.size());
    for (std::size_t j = 0; j < bits.size(); ++j) p.hdr_llr[j] = static_cast<float>(8.0 * (1.0 - 2.0 * bits[j]));
    p.header = h;
    p.q = q;
    p.f_hz = f_hz;
    p.uid = types::pass_uid(q, f_hz, p.z);
    p.frame = spec.name();
    p.waveform = qrss::config::WAVEFORM_CE;
    p.timing = {0.0, 0.0, 0.0, 10.0, math::Matrix(2, 2, {1.0, 0.0, 0.0, 1.0})};
    p.report = {f_hz, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 1.0, true, false};
    p.estimator = "joint";
    return p;
}

// Mean power of the keyed (non-zero) samples.
inline double keyed_power(std::span<const float> x) {
    double s = 0.0;
    std::size_t n = 0;
    for (const float v : x)
        if (v != 0.0f) {
            s += static_cast<double>(v) * static_cast<double>(v);
            ++n;
        }
    return n ? s / static_cast<double>(n) : 0.0;
}

// The real-audio noise sigma that puts a pass of keyed power P at snr_db.
inline double noise_sigma(double P, double snr_db) {
    return std::sqrt(P * (qrss::config::FS / 2.0) / (2500.0 * std::pow(10.0, snr_db / 10.0)));
}

// Add `pass` into `rec` from sample `at`, scaled so that against noise of
// `sigma` it sits at snr_db.
inline void add_pass(std::vector<double>& rec, std::span<const float> pass, std::size_t at, double sigma,
                     double snr_db) {
    const double P = keyed_power(pass);
    const double want = sigma * sigma * 2500.0 * std::pow(10.0, snr_db / 10.0) / (qrss::config::FS / 2.0);
    const double k = std::sqrt(want / P);
    for (std::size_t i = 0; i < pass.size() && at + i < rec.size(); ++i)
        rec[at + i] += k * static_cast<double>(pass[i]);
}

}  // namespace sstvae::em_fixture
