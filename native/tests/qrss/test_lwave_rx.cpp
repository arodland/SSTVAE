// native/core/qrss/lwave_rx against sstvae/qrss/lwave.py's receive side:
// PULSE_NORM recomputed equal to the generated value, and the matched
// filter (real and complex, 250 Hz and 16 kHz) pinned from Python by
// SHA-256 of its bytes. tests/test_qrss_native_lwave_rx.py compares against
// Python directly. The pins (inputs as test_ce_rx.cpp's):
//
//   lwave.matched_filter(z, fs, -1000, np.arange(-20, 300)) for fs in (250, 16000)

#include <complex>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/lwave.hpp"
#include "qrss/lwave_rx.hpp"
#include "qrss/rx_config.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace lr = sstvae::qrss::lwave_rx;
using cd = std::complex<double>;
using qrss_check::digest_is;

static void test_norm() {
    check::equal(lr::pulse_norm(), lr::PULSE_NORM, "PULSE_NORM recomputed");
    check::equal(lr::PULSE_NORM, 0x1.fff4afe37430dp-1, "PULSE_NORM pinned");
}

static void test_matched_filter() {
    std::vector<double> zr;
    std::vector<cd> zc;
    std::vector<std::int64_t> pos;
    for (std::int64_t n = 0; n < 4000; ++n) {
        const double r = static_cast<double>((n * 7) % 19 - 9) / 16.0;
        zr.push_back(r);
        zc.emplace_back(r, static_cast<double>((n * 3) % 17 - 8) / 32.0);
    }
    for (std::int64_t p = -20; p < 300; ++p) pos.push_back(p);
    digest_is(lr::matched_filter(zr, 250, -1000, pos),
              "de7bc6046163cfdd24d8e8e3b26289b46db534c4ef3981c559e328cec818752d", "real, 250 Hz");
    digest_is(lr::matched_filter(zc, 250, -1000, pos),
              "31f37ef8ca8f491a2f217ff1b8eae1e38457fd6c367fdab037358e6dc01fc5b5", "complex, 250 Hz");
    digest_is(lr::matched_filter(zr, 16000, -1000, pos),
              "477c789c40c3e91fa9ec10b31b6bb1d0b87f8f2131f7654b70de18bab357b55f", "real, 16 kHz");
    digest_is(lr::matched_filter(zc, 16000, -1000, pos),
              "145d7872e4084d835a9472a6d360f8563ad07ee24fecde87db58813e1dd9a09c", "complex, 16 kHz");
    const std::vector<double> ones(64000, 1.0);
    const auto g = lr::matched_filter(ones, 16000, -32000, std::vector<std::int64_t>{0});
    check::close(g, std::vector<double>{qrss::lwave::g0()}, 1e-12, "a constant gives G0");
}

int main() {
    try {
        test_norm();
        test_matched_filter();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_lwave_rx");
}
