// native/core/qrss/render against sstvae/qrss/render.py's pure part: the
// decoder planes of a deterministic accumulator pinned from Python by
// SHA-256 of their float32 bytes, W~, g(W) at its edges (-0.0, a negative
// W), and the refusals. tests/test_qrss_native_render.py compares against
// Python directly. The pins:
//
//   i = np.arange(3 * 52800); W = ((i * 37) % 101) / 16.0; W[52800:105600] = 0.0
//   S = ((i * 13) % 29 - 14) / 8.0
//   render.decoder_planes(S, W, mode), render.reference_w(W, mode), mode 0..2
//   W2 = W.copy(); W2[52800:105600] = ((i[:52800] * 11) % 7) / 4.0; render.decoder_planes(S, W2, 2)
//   render.g_shrink([-0.0, 0.0, 0.5, 1.0, 7.25, -3.0])

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/render.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace rd = sstvae::qrss::render;
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

static std::uint64_t bits(double x) {
    std::uint64_t b;
    std::memcpy(&b, &x, sizeof b);
    return b;
}

static void test_planes() {
    const std::size_t n = 3 * 52800;
    std::vector<double> W(n), S(n);
    for (std::size_t i = 0; i < n; ++i) {
        W[i] = static_cast<double>((i * 37) % 101) / 16.0;
        S[i] = static_cast<double>(static_cast<std::int64_t>((i * 13) % 29) - 14) / 8.0;
    }
    for (std::size_t i = 52800; i < 2 * 52800; ++i) W[i] = 0.0;
    const char* lat[3] = {"ea36864d5e0432b1832e997759790409bb49f71668a4ba79d678546bdf498991",
                          "ea36864d5e0432b1832e997759790409bb49f71668a4ba79d678546bdf498991",
                          "baeb8133102ddb33d09815fc96cfbd1f5783f2276eb823302d448a21f3a349bf"};
    const char* wt[3] = {"699aac680e77961621b97d8578c26821e0c29ea50687b405b2bfe3fd91f923ac",
                         "699aac680e77961621b97d8578c26821e0c29ea50687b405b2bfe3fd91f923ac",
                         "f010b877882a431363e52065531ee90e47c8f23da4b3f7923493d18f77b0bbd3"};
    for (int mode = 0; mode < 3; ++mode) {
        const auto [l, w] = rd::decoder_planes(std::span<const double>(S), std::span<const double>(W), mode);
        digest_is(l, lat[mode], "latents, mode " + std::to_string(mode));
        digest_is(w, wt[mode], "weights, mode " + std::to_string(mode));
        check::equal(rd::reference_w(W, mode), 3.125, "W~, mode " + std::to_string(mode));
    }
    std::vector<double> W2 = W;
    for (std::size_t i = 0; i < 52800; ++i) W2[52800 + i] = static_cast<double>((i * 11) % 7) / 4.0;
    const auto [l2, w2] = rd::decoder_planes(std::span<const double>(S), std::span<const double>(W2), 2);
    digest_is(l2, "f94496b0565df35d15f4738fa796ae1762d848921db6ba3cbdbfa20ad65d996d", "latents, every group");
    digest_is(w2, "3a95190e2d2c1c4f4d3040a72ae95294233f8f5f59da27d77efef1f1c8eac2be", "weights, every group");
    // float32 planes in, as an accumulator holds them: the same
    const std::vector<float> Sf(S.begin(), S.end()), Wf(W2.begin(), W2.end());
    const auto [l3, w3] = rd::decoder_planes(std::span<const float>(Sf), std::span<const float>(Wf), 2);
    digest_is(l3, "f94496b0565df35d15f4738fa796ae1762d848921db6ba3cbdbfa20ad65d996d", "float32 planes in");
    // nothing heard: W~ is 0 and the planes are zero
    const std::vector<double> zero(n, 0.0);
    check::equal(rd::reference_w(zero, 2), 0.0, "W~ with no data");
    check::equal(rd::reference_w(W, -1), 0.0, "W~ of no groups");
}

static void test_g_shrink() {
    const std::vector<double> w{-0.0, 0.0, 0.5, 1.0, 7.25, -3.0};
    const auto g = rd::g_shrink(w);
    const std::vector<double> want{0.0, 0.0, 0x1.2d40746bbcedcp-1, 0x1.54e772beac649p-1, 0x1.8786f6c42f156p-1, 0.0};
    for (std::size_t i = 0; i < w.size(); ++i)
        check::equal(bits(g[i]), bits(want[i]), "g_shrink bits " + std::to_string(i));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check::is_true(std::isnan(rd::g_shrink(std::vector<double>{nan})[0]), "NaN stays NaN");
}

static void test_refusals() {
    const std::vector<double> ok(3 * 52800, 1.0), small(5, 1.0);
    throws<std::invalid_argument>([&] { (void)rd::decoder_planes(std::span<const double>(ok), std::span<const double>(ok), 3); },
                                  "mode must be 0..2, not 3", "mode 3");
    throws<std::invalid_argument>([&] { (void)rd::decoder_planes(std::span<const double>(ok), std::span<const double>(ok), -1); },
                                  "mode must be 0..2, not -1", "mode -1");
    throws<std::invalid_argument>([&] { (void)rd::decoder_planes(std::span<const double>(small), std::span<const double>(ok), 0); },
                                  "cannot reshape array of size 5 into shape (3,52800)", "S of the wrong size");
    std::vector<double> bad = ok;
    bad[7] = -1.0;
    throws<std::invalid_argument>([&] { (void)rd::decoder_planes(std::span<const double>(ok), std::span<const double>(bad), 0); },
                                  "W must be finite and non-negative, S finite", "negative W");
    bad[7] = std::numeric_limits<double>::infinity();
    throws<std::invalid_argument>([&] { (void)rd::decoder_planes(std::span<const double>(bad), std::span<const double>(ok), 0); },
                                  "W must be finite and non-negative, S finite", "infinite S");
    throws<std::out_of_range>([&] { (void)rd::reference_w(ok, 3); }, "index 3 is out of bounds for axis 0 with size 3",
                              "reference_w past the groups");
    throws<std::invalid_argument>([&] { (void)rd::reference_w(small, 0); },
                                  "cannot reshape array of size 5 into shape (3,52800)", "reference_w size");
}

int main() {
    try {
        test_planes();
        test_g_shrink();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_render");
}
