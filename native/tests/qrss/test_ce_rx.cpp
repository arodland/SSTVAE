// native/core/qrss/ce_rx against sstvae/qrss/ce.py's receive side: the
// matched filter (real and complex, at 250 Hz and on the 16 kHz grid)
// pinned from Python by SHA-256 of its bytes, its gain on a constant (G0),
// and the mean template at a few times (1e-12: np.exp is numpy's own, a
// libm's ulp away on some arguments). tests/test_qrss_native_ce_rx.py
// compares against Python directly. The pins:
//
//   n = np.arange(4000); zr = ((n * 7) % 19 - 9) / 16.0
//   zc = zr + 1j * (((n * 3) % 17 - 8) / 32.0); pos = np.arange(-20, 300)
//   ce.matched_filter(z, fs, -1000, pos) for fs in (250, 16000)
//   ce.matched_filter(np.ones(32000), 16000, -16000, np.arange(-5, 5))[0]
//   k = np.arange(frame.TINY.n_sym); mu = ((k * 5) % 9 - 4) / 4.0; nu = ((k * 3) % 7) / 8.0
//   ce.mean_template([-9.5, -8.0, -3.25, 0.0, 1.5, 100.125, 1000.75, 1752.5, 1760.0], mu, nu, frame.TINY)
//   ce._mean_template_grid(250, 2000, 5, mu, nu, frame.TINY)

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/ce_rx.hpp"
#include "qrss/frame.hpp"
#include "qrss/rx_config.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace cr = sstvae::qrss::ce_rx;
namespace fr = sstvae::qrss::frame;
using cd = std::complex<double>;
using qrss_check::digest_is;

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("threw another type: ") + e.what());
        return;
    }
    check::fail(what, "did not throw");
}

static void inputs(std::vector<double>& zr, std::vector<cd>& zc, std::vector<std::int64_t>& pos) {
    for (std::int64_t n = 0; n < 4000; ++n) {
        const double r = static_cast<double>((n * 7) % 19 - 9) / 16.0;
        zr.push_back(r);
        zc.emplace_back(r, static_cast<double>((n * 3) % 17 - 8) / 32.0);
    }
    for (std::int64_t p = -20; p < 300; ++p) pos.push_back(p);
}

static void test_matched_filter() {
    std::vector<double> zr;
    std::vector<cd> zc;
    std::vector<std::int64_t> pos;
    inputs(zr, zc, pos);
    digest_is(cr::matched_filter(zr, 250, -1000, pos),
              "9d33a60eebf8db582961c5af6fe840a11e998f7b976737f69be3e548fe92f6ed", "real, 250 Hz");
    digest_is(cr::matched_filter(zc, 250, -1000, pos),
              "8c43a6e12d0364b8012d813c91c93194c9d658326f724176a5f3c372b10f1de2", "complex, 250 Hz");
    digest_is(cr::matched_filter(zr, 16000, -1000, pos),
              "676db3aa1b0e11622674047e84eaf1e8dd4d30cd49ea9bfcd077d697b51ff1f7", "real, 16 kHz");
    digest_is(cr::matched_filter(zc, 16000, -1000, pos),
              "993ea09a599af6a75a937c2ca9cfa961c76970fa8c65e6b6c97beb867a748c1a", "complex, 16 kHz");
    const std::vector<double> ones(32000, 1.0);
    const std::vector<std::int64_t> p5{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    const auto g = cr::matched_filter(ones, 16000, -16000, p5);
    check::equal(g[0], 0x1.006d1ed38b121p+0, "a constant gives G0");
    check::equal(g[0], qrss::rxconfig::ce::G0, "G0 as generated");
    throws<std::invalid_argument>([&] { (void)cr::matched_filter(zr, 3000, 0, pos); },
                                  "fs must divide 16000, not 3000", "fs");
    throws<std::out_of_range>([&] { (void)cr::matched_filter(std::vector<double>{}, 250, 0, pos); },
                              "index -1 is out of bounds for axis 0 with size 0", "an empty z");
    check::equal(cr::matched_filter(std::vector<double>{}, 250, 0, std::vector<std::int64_t>{}).size(),
                 std::size_t{0}, "no positions");
}

static void test_mean_template() {
    const auto& spec = fr::get("tiny");
    std::vector<double> mu, nu;
    for (std::int64_t k = 0; k < spec.n_sym(); ++k) {
        mu.push_back(static_cast<double>((k * 5) % 9 - 4) / 4.0);
        nu.push_back(static_cast<double>((k * 3) % 7) / 8.0);
    }
    const std::vector<double> tau{-9.5, -8.0, -3.25, 0.0, 1.5, 100.125, 1000.75, 1752.5, 1760.0};
    const auto mt = cr::mean_template(tau, mu, nu, spec);
    const std::vector<cd> want{{0.0, 0.0},
                               {0x1.fffe83e30fd0ap-1, -0x1.37f163629dcb7p-8},
                               {0x1.fdaa4f8e4aa6ap-1, 0x1.725a6d69c4be9p-4},
                               {0x1.3880fed99cedap-1, -0x1.9504fd76adfc8p-1},
                               {0x1.a93b5c571beb2p-1, -0x1.d5a2f25deb317p-3},
                               {0x1.889f72c43e0acp-1, 0x1.17acc8fb6a72ap-3},
                               {0x1.95ef44ae576e7p-1, -0x1.26b86d7154534p-1},
                               {0x1.b54c0c179ce6fp-1, -0x1.2f5a9e17bdca4p-2},
                               {0.0, 0.0}};
    check::close(mt, want, 1e-12, "mean_template");
    const auto mg = cr::mean_template_grid(250, 2000, 5, mu, nu, spec);
    const std::vector<cd> wg{{0x1.b26d5a68e04bdp-1, 0x1.c1bd26f20b37ap-2},
                             {0x1.af921199a2c7cp-1, 0x1.cc9958a4b6abfp-2},
                             {0x1.bc94af28a8aebp-1, 0x1.8aeee09b9a46cp-2},
                             {0x1.cee9eb8d98853p-1, 0x1.04d2bcc444569p-2},
                             {0x1.d64a9b928c7e0p-1, 0x1.3fafc1ba76e79p-4}};
    check::close(mg, wg, 1e-12, "_mean_template_grid");
    const auto before = cr::mean_template_grid(250, -1000, 7, mu, nu, spec);
    check::close(before, std::vector<cd>(7), 0.0, "unkeyed before the span");
    // the grid path agrees with the closed form at the grid's own times
    std::vector<double> t;
    for (int i = 0; i < 5; ++i) t.push_back(static_cast<double>((2000 + i) * 64) / 485.0);
    check::close(cr::mean_template(t, mu, nu, spec), mg, 1e-9, "grid path against the closed form");
    // the keyed span can start earlier (a lead-in)
    const auto early = cr::mean_template(std::vector<double>{-9.5}, mu, nu, spec, nullptr, false, -12.0);
    check::is_true(std::abs(early[0]) > 0.5, "keyed_from_pos moves the start");
    throws<std::invalid_argument>([&] { (void)cr::mean_template(tau, std::vector<double>(3), nu, spec); },
                                  "sym has shape (3,), expected (1753,)", "mu's length");
    throws<std::invalid_argument>([&] { (void)cr::mean_template(tau, mu, std::vector<double>(3), spec); },
                                  "var has shape (3,), expected (1753,)", "nu's length");
    // An L frame fails the reference's CE layout assertion (frame.layout,
    // reached from at_positions after mu's shape check and before nu's).
    const auto& lshort = fr::get_l("l-short");
    const auto& ltiny = fr::get_l("l-tiny");
    const std::vector<double> mul(static_cast<std::size_t>(lshort.n_sym())), mut(static_cast<std::size_t>(ltiny.n_sym()));
    throws<fr::WrongWaveform>([&] { (void)cr::mean_template(tau, mul, std::vector<double>(3), lshort); },
                              "1237 header slots for 2474 bits", "an L frame, before nu's shape");
    throws<fr::WrongWaveform>([&] { (void)cr::mean_template_grid(250, 0, 4, mut, mut, ltiny); },
                              "512 data slots for 1024 latents", "an L frame with no header");
    throws<std::invalid_argument>([&] { (void)cr::mean_template(tau, std::vector<double>(3), nu, lshort); },
                                  "sym has shape (3,), expected (3834,)", "mu's shape is checked first");
}

int main() {
    try {
        test_matched_filter();
        test_mean_template();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_ce_rx");
}
