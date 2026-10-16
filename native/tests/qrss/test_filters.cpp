// native/core/qrss/filters against digests of scipy 1.18's output.
//
// The port is bit-exact against scipy (tests/test_qrss_native_filters.py
// compares the two side by side), so each check here is the SHA-256 of
// the float bytes of the same call in Python, made with
//
//   hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
//
// on inputs that are exact in both languages: k/16 grids and
// ((i*7919) % 2001 - 1000)/1000 sequences. No Python at test time.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/filters.hpp"
#include "qrss/tables.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace flt = sstvae::qrss::filters;
using qrss_check::digest_is;
using cd = std::complex<double>;
using cf = std::complex<float>;

namespace {

// xr[i] = ((i * 7919) % 2001 - 1000) / 1000, xi[i] = ((i * 104729) % 1999 - 999) / 999.
std::vector<double> seq_r(std::size_t n) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<double>(static_cast<long long>((i * 7919) % 2001) - 1000) / 1000.0;
    return v;
}

std::vector<double> seq_i(std::size_t n) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<double>(static_cast<long long>((i * 104729) % 1999) - 999) / 999.0;
    return v;
}

std::vector<cd> seq_c(std::size_t n) {
    const auto r = seq_r(n), im = seq_i(n);
    std::vector<cd> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = cd(r[i], im[i]);
    return v;
}

void refuses(const std::function<void()>& fn, const std::string& want, const std::string& what) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        check::equal(std::string(e.what()), want, what);
        return;
    }
    check::fail(what, "did not throw");
}

}  // namespace

static void test_i0() {
    // x = np.arange(-12000, 12001) / 16.0: special.i0(x), special.i0e(x)
    std::vector<double> a, b;
    for (int k = -12000; k <= 12000; ++k) {
        a.push_back(flt::i0(k / 16.0));
        b.push_back(flt::i0e(k / 16.0));
    }
    qrss_check::libm_digest_is(a, "f75edae26014b4742243e3749590bf955f2c7d9e4f052f9ebf34d7edfd02c753", "i0 on k/16");
    digest_is(b, "7ef3897c7dc26287622883f4a6379f74b39bff7b8459b86f3b0e9784291b70a6", "i0e on k/16");
    check::equal(flt::i0(0.0), 1.0, "i0(0)");
    check::is_true(std::isinf(flt::i0(710.0)), "i0 overflows past log(DBL_MAX), as Cephes does");
    check::is_true(std::isnan(flt::i0(INFINITY)), "i0(inf) is NaN, as scipy's is");
    check::equal(flt::i0e(INFINITY), 0.0, "i0e(inf)");
}

static void test_firwin() {
    const auto l = flt::firwin(481, 125.0, flt::Window::kaiser(12.0), flt::PassZero::Yes, true, 8000.0);
    qrss_check::libm_pin(std::equal(l.begin(), l.end(), qrss::tables::L_INTERP_TAPS.begin()) &&
                       l.size() == qrss::tables::L_INTERP_TAPS.size(),
                   "firwin(481, 125, fs=8000, ('kaiser', 12)) is the frozen L interpolator");
    qrss_check::libm_digest_is(l, std::string(qrss::tables::L_INTERP_TAPS_SHA256), "L interpolator digest");
    qrss_check::libm_digest_is(flt::firwin(255, 1250.0, flt::Window::kaiser(8.0), flt::PassZero::Yes, true, 8000.0),
              "89877403527d899cd53cefc55b66eb525da275af69c6fb3aa2a668de1b44c1c8", "frontend._FE_TAPS");
    qrss_check::libm_digest_is(flt::firwin(321, 1.0 / 16, flt::Window::kaiser(10.0)),
              "9ed77c54109fe70fdb5b8c53650121a171fd2d76dc4a53ad24917eb179250e5a", "channeliser decimator");
    qrss_check::libm_digest_is(flt::firwin(2001, 0.6, flt::Window::hann(), flt::PassZero::Yes, true, 250.0),
              "b58532339cb2b35e2a7d170a2c4344f19b18d1bd83af8cfa5d00af3df133075e", "acquire notch");
    const double bp[2] = {0.2, 0.5};
    qrss_check::libm_digest_is(flt::firwin(31, bp, flt::Window::hamming(), flt::PassZero::No),
              "f076a2cb2447618ffdebd14c4dff23d1892403d8bcc496769932fe1826d5901d", "bandpass hamming");
    const double bs[3] = {0.1, 0.3, 0.6};
    digest_is(flt::firwin(51, bs, flt::Window::hann(), flt::PassZero::Yes, false),
              "a069d08623812b22c31678b5743408dc8c78b0edc35e71f130899be8ba14c426", "3 edges, hann, unscaled");
    qrss_check::libm_digest_is(flt::firwin(61, 0.25, flt::Window::hamming(), flt::PassZero::Highpass, true, std::nullopt, 0.05),
              "6ce7b0a313d55c4752f553bd35621e3bcce94c83a6730b42497bd872c0e3c911", "highpass by width");

    const double two[2] = {0.1, 0.4};
    refuses([] { flt::firwin(31, 0.0); },
            "Invalid cutoff frequency: frequencies must be greater than 0 and less than fs/2.", "cutoff 0");
    refuses([] { flt::firwin(31, 4000.0, flt::Window::hamming(), flt::PassZero::Yes, true, 8000.0); },
            "Invalid cutoff frequency: frequencies must be greater than 0 and less than fs/2.", "cutoff at Nyquist");
    refuses([] { flt::firwin(31, std::span<const double>()); }, "At least one cutoff frequency must be given.",
            "no cutoff");
    refuses([&] { flt::firwin(31, std::span<const double>(two, 2), flt::Window::hamming(), flt::PassZero::Lowpass); },
            "cutoff must have one element if pass_zero==\"lowpass\", got (2,)", "lowpass with two edges");
    refuses([] { flt::firwin(31, 0.1, flt::Window::hamming(), flt::PassZero::Bandstop); },
            "cutoff must have at least two elements if pass_zero==\"bandstop\", got (1,)", "bandstop with one");
    refuses([] { flt::firwin(30, 0.1, flt::Window::hamming(), flt::PassZero::No); },
            "A filter with an even number of coefficients must have zero response at the Nyquist frequency.",
            "even highpass");
    refuses([] { flt::firwin(0, 0.1); }, "Parameter Nx=0 is not a positive integer", "no taps");
}

static void test_windows() {
    qrss_check::libm_digest_is(flt::hann(16000, false), "1f280d5a244ca42ec6fa3acb5e16e2f141c24e5a1c1ab1c8f186412a38c0c6c0",
              "hann(16000, sym=False)");
    qrss_check::libm_digest_is(flt::hamming(64), "cec21907a722277d2de18e93dd9d0baa5368b5967148c61643920cd4d0fcc8ea", "hamming(64)");
    digest_is(flt::kaiser(101, 8.6), "d3e7664f48f7ce7bbe1c0bee7b093e7db57510860f916c75040e4640fae6a960",
              "kaiser(101, 8.6)");
    check::equal(flt::hann(1).size(), std::size_t(1), "hann(1)");
    check::equal(flt::hann(0).size(), std::size_t(0), "hann(0)");
    refuses([] { flt::kaiser(-1, 2.0); }, "Window length M must be a non-negative integer", "negative M");
}

static void test_upfirdn() {
    const auto& ct = qrss::tables::CE_PULSE_TABLE;
    const auto& lt = qrss::tables::L_PULSE_TABLE;
    const std::vector<double> ceh(ct.begin(), ct.end()), lh(lt.begin(), lt.end());
    const auto xr = seq_r(300);
    const auto xc = seq_c(300);

    const auto a = flt::upfirdn<double>(ceh, xr, 485, 2);
    check::equal(a.size(), std::size_t(76388), "CE grid length");
    digest_is(a, "2c37510a446c87e93abef8c4c3ca58255958de271397a6e4bba194e198beca46", "upfirdn(CE table, x, 485, 2)");

    std::vector<double> z1(1, 0.0);
    z1.insert(z1.end(), ceh.begin(), ceh.end());
    digest_is(flt::upfirdn(z1, xc, 485, 2), "e4455604d6ff42c67cd9011ee267c8a12cf82b5a9e70c9019288aeb6a37647bb",
              "upfirdn(0 + CE table, complex x, 485, 2)");

    const auto l = flt::upfirdn(lh, xc, 970, 64);
    check::equal(l.size(), std::size_t(4775), "L grid length");
    digest_is(l, "432fb01af44edfd9015aa6ff4d7a266b129294b267907dd9f183697d23eea46f", "upfirdn(L table, x, 970, 64)");

    std::vector<double> interp = flt::firwin(481, 125.0, flt::Window::kaiser(12.0), flt::PassZero::Yes, true, 8000.0);
    for (auto& v : interp) v *= 32;
    qrss_check::libm_digest_is(flt::upfirdn(interp, std::span<const cd>(xc.data(), 50), 32),
              "e11c886cedf927f5ad27647c40425b99dcc4615f796d9e865594b3b24cadeea0", "lwave.resample's x32");

    const std::vector<float> hf = {0.5f, -1.25f, 2.0f};
    std::vector<float> xf(20);
    for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = static_cast<float>(xr[i]);
    digest_is(flt::upfirdn<float>(hf, xf, 3, 2, 0.75f),
              "bc56fd7d280fde95a1d5376ce8d86dd794d43aa90bbe90cd63ca45ccbe70e055", "float32 with cval");

    digest_is(flt::upfirdn<double>(std::span<const double>(ceh.data(), 100), std::span<const double>(xr.data(), 3), 7, 3),
              "6e1eb5dfff91768b7c16dc9bb79182cd5bfa6bc86f871179b65cb9d6f9070c2b", "signal shorter than a phase");

    check::equal(flt::upfirdn_output_len(1, 0, 3, 1), std::int64_t(-2), "_output_len floors");
    check::equal(flt::upfirdn_output_len(-5, 1, 1, -3), std::int64_t(3), "_output_len floors a negative divisor");
    // Cython wraps the int64 arithmetic and raises on the two divisions C traps on
    check::equal(flt::upfirdn_output_len(INT64_MIN + 2, 1, 1, -1), INT64_MIN, "_output_len wraps as Cython's does");
    bool zero = false, overflow = false;
    try {
        flt::upfirdn_output_len(1, 1, 1, 0);
    } catch (const std::domain_error&) {
        zero = true;
    }
    try {
        flt::upfirdn_output_len(INT64_MIN + 1, 1, 1, -1);
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    check::is_true(zero, "_output_len refuses down == 0 rather than trapping");
    check::is_true(overflow, "_output_len refuses INT64_MIN // -1 rather than trapping");
    refuses([] { flt::upfirdn<double>(std::vector<double>{1.0}, std::vector<double>{}, 3, 1); },
            "negative dimensions are not allowed", "empty x, up 3");
    refuses([] { flt::upfirdn<double>(std::vector<double>{}, std::vector<double>{1.0}); },
            "h must be 1-D with non-zero length", "empty h");
    refuses([] { flt::upfirdn<double>(std::vector<double>{1.0}, std::vector<double>{1.0}, 0, 1); },
            "Both up and down must be >= 1", "up 0");
}

static void test_resample_poly() {
    const auto yr = seq_r(1000);
    const auto yc = seq_c(1000);
    qrss_check::libm_digest_is(flt::resample_poly<cd>(yc, 1, 16, flt::Window::kaiser(10.0)),
              "60f01a0d73951cfb1715348b9c111521c48c7e2826b396f7c55d4727b037c7a7", "channelise's decimation");
    qrss_check::libm_digest_is(flt::resample_poly<double>(yr, 1, 16, flt::Window::kaiser(10.0)),
              "0e5a7f53fc33a90253ae88c595355e1dbbc9bee109d5315825f24bc2bfcd645e", "channelise_keep's decimation");
    const auto up = flt::resample_poly<cd>(std::span<const cd>(yc.data(), 250), 32, 1);
    check::equal(up.size(), std::size_t(8000), "250 Hz to 8 kHz length");
    qrss_check::libm_digest_is(up, "45826a6a260ae7d08a40cbc3572892112a6e5e67c6b1ac7264256ae81c4ae0da", "ce.to_audio from 250 Hz");
    std::vector<float> yf(yr.size());
    for (std::size_t i = 0; i < yf.size(); ++i) yf[i] = static_cast<float>(yr[i]);
    digest_is(flt::resample_poly<float>(yf, 3, 2), "bdf03d3cf872259ef4e414a6cff7a19a586b65d5869a32739d23ba9ceaa6e818",
              "float32 3/2");
    std::vector<cf> yc64(yc.size());
    for (std::size_t i = 0; i < yc.size(); ++i)
        yc64[i] = cf(static_cast<float>(yc[i].real()), static_cast<float>(yc[i].imag()));
    digest_is(flt::resample_poly<cf>(yc64, 2, 7, flt::Window::hann()),
              "c8e9c8990b30fb0e3ecfa9ef6ea7f50d19f4a6811e29c31019b9616f97757547", "complex64 2/7 hann");
    const auto same = flt::resample_poly<double>(std::span<const double>(yr.data(), 5), 4, 4);
    check::is_true(same == std::vector<double>(yr.begin(), yr.begin() + 5), "equal rates copy");
    refuses([&] { flt::resample_poly<double>(yr, 0, 1); }, "up and down must be >= 1", "up 0");
}

int main() {
    try {
        test_i0();
        test_firwin();
        test_windows();
        test_upfirdn();
        test_resample_poly();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_filters");
}
