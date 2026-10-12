// native/core/qrss/precoder against digests of the Python reference.
//
// Inputs are closed form (no generator), on the three grids that matter:
//   i = np.arange(50600, dtype=np.int64)
//   fp16 = ((i * 2654435761) % 4095 - 2047) / 1024.0    (every value an fp16)
//   int8 = ((i * 40503) % 255 - 127).astype(np.float64) / 20   (picture.from_int8)
//   rat  = ((i * 7919) % 10007 - 5003) / 997.0
// and each digest is hashlib.sha256(np.ascontiguousarray(out).tobytes())
// of the same call in sstvae/qrss/precoder.py, Q = 1954752 unless named.
// tests/test_qrss_native_precoder.py compares the two implementations
// directly where Python is available.

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "qrss/precoder.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace pc = sstvae::qrss::precoder;
using qrss_check::digest_is;

namespace {

constexpr std::uint64_t Q_TEST = 1954752;
constexpr std::size_t N = 50600;

std::vector<double> fp16_input(std::size_t n) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<double>(static_cast<std::int64_t>((i * 2654435761ULL) % 4095) - 2047) / 1024.0;
    return v;
}

std::vector<double> int8_input(std::size_t n) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<double>(static_cast<std::int64_t>((i * 40503ULL) % 255) - 127) / 20.0;
    return v;
}

std::vector<double> rat_input(std::size_t n) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<double>(static_cast<std::int64_t>((i * 7919ULL) % 10007) - 5003) / 997.0;
    return v;
}

bool throws(void (*f)()) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

static void test_block_sizes() {
    const auto s = pc::block_sizes(N);
    check::equal(s.size(), std::size_t(792), "block_sizes(50600) count");
    bool ok = s.size() == 792 && s[790] == 32 && s[791] == 8;
    for (std::size_t i = 0; ok && i < 790; ++i) ok = s[i] == 64;
    check::is_true(ok, "block_sizes(50600) = 790 x 64, 32, 8");
    check::is_true(pc::block_sizes(63) == std::vector<std::size_t>{32, 16, 8, 4, 2, 1}, "block_sizes(63)");
    check::is_true(pc::block_sizes(16384) == std::vector<std::size_t>(256, 64), "block_sizes(16384)");
    check::is_true(pc::block_sizes(4096) == std::vector<std::size_t>(64, 64), "block_sizes(4096)");
    check::is_true(pc::block_sizes(0).empty(), "block_sizes(0)");
    auto s777 = std::vector<std::size_t>(12, 64);
    s777.push_back(8);
    s777.push_back(1);
    check::is_true(pc::block_sizes(777) == s777, "block_sizes(777)");

    const auto r = pc::runs(N);
    check::is_true(r.size() == 3 && r[0].start == 0 && r[0].m == 64 && r[0].count == 790 && r[1].start == 50560 &&
                       r[1].m == 32 && r[1].count == 1 && r[2].start == 50592 && r[2].m == 8 && r[2].count == 1,
                   "runs(50600)");
}

static void test_digests() {
    struct Case {
        const char* name;
        std::vector<double> x;
        const char *in, *wht, *pre, *unp, *bm;
    };
    const Case cases[] = {
        {"fp16", fp16_input(N), "319df8a75da1a1586778c7ee6a1061edd96ab932d8d43425544221995f5c86b9",
         "24a6605f8ddb686e0c844512d164e70d5f35e9758fba4c7522085b9c8f64cbe4",
         "1d859dcda5dbb83728c97cd76813f8a0618daa4aed8c2640b64890256b298a24",
         "cedced3e2e5b5291b0feed441f81a06d4b2775bcd67c3d2df238389c5efd4ae4",
         "4b2a9ce2f54d4344de24bf3898a8326138fa53ac06b2dcfe448eeb0ff46c8759"},
        {"int8", int8_input(N), "5e069bec8db55fba38be53e17d254fabb6a135f5e63bd80982b12cb3eb25bb33",
         "42c8c20cec768a52695d344c053e497e1f4fe466acbfc6f0ce34499e86694251",
         "d0f8a73e5dd6f115696aa9c61b5cb10a1fc8223c4e1774d58c37d7b73eea9199",
         "44d105d7f494751071ede4013b4da6e045686cc26d10af69a5536ac1fbb6db96",
         "e2c3f1032edf028e3004d30ddd47fbc381464c80d026d043869c8e2d7adc517d"},
        {"rat", rat_input(N), "b3628ab4c16395ed03c0020109d60a703c6895394e31e4d0a4895a40bcf90fc0",
         "831180de2445da999edb5cde998f54c77bc95be3a1ccf5b5f67754efadb8b5b1",
         "f1599937598dbff5b1c805823b9543e6eae74569b2912195dc6ba941edc9a729",
         "71299703fd5a79d1d77f001fad4ae50dbd0bdbff577961bef2103b9717048ef9",
         "833ac20941ae0d680c7d3718c7cdcea4e6c306ac77a8f20f2657c466b8a254ef"},
    };
    for (const Case& c : cases) {
        const std::string n = c.name;
        digest_is(c.x, c.in, n + " input");
        digest_is(pc::wht(c.x), c.wht, n + " wht");
        digest_is(pc::precode(c.x, Q_TEST), c.pre, n + " precode");
        digest_is(pc::unprecode(c.x, Q_TEST), c.unp, n + " unprecode");
        digest_is(pc::block_mean(c.x), c.bm, n + " block_mean");
    }
    digest_is(pc::block_index(N), "626b42cc91efd600cd804add565edae001e6fe855589bbe1545678bfc48eae5e",
              "block_index(50600)");
    const auto r = rat_input(777);
    digest_is(pc::wht(r), "dc09474369e52995e34c533eca94904468b77a670742f4d6fda7c774a1187c47", "wht, 777");
}

static void test_batch_rows() {
    // (3, 4096): rat[:12288].reshape(3, 4096).
    const auto b = rat_input(3 * 4096);
    digest_is(pc::wht(b, 4096), "5e5bce3a7790a5588228577989ecb3f62fffcdf0a1839b84d3f8bad23c550006", "wht rows");
    digest_is(pc::precode(b, (std::uint64_t(1) << 40) + 12345, 4096),
              "237eb16e9d152e4c9eceda0e0304ce4e6ce44520bbaf9d37a2f3cb98c5ac8a99", "precode rows");
    // Rows are independent: each equals the single-vector transform.
    const auto w = pc::wht(b, 4096);
    for (std::size_t r = 0; r < 3; ++r) {
        const std::vector<double> row(b.begin() + r * 4096, b.begin() + (r + 1) * 4096);
        check::is_true(pc::wht(row) == std::vector<double>(w.begin() + r * 4096, w.begin() + (r + 1) * 4096),
                       "wht row " + std::to_string(r));
    }
    // block_mean of (3, 777) = rat[:2331].reshape(3, 777): C order sums each
    // block pairwise; np.asfortranarray of it makes numpy's reduction run
    // the batch axis innermost and sum one element at a time from 0.0.
    const auto b777 = rat_input(3 * 777);
    digest_is(pc::block_mean(b777, 777), "57c158d5eb824f91fb8cdfc546af1fc377506410517097f55cbf9d8b4d139fc5",
              "block_mean rows, pairwise");
    digest_is(pc::block_mean(b777, 777, pc::Summation::sequential),
              "15c5b30915513a45ccaf5c6390dec23f073d3c3415d3233db76b0b0a08ce34c9",
              "block_mean rows, sequential (Fortran order)");
    check::is_true(throws([] { pc::wht(std::vector<double>(10), 4); }), "rows must be whole");
    check::is_true(throws([] { pc::wht(std::vector<double>(3), 0); }), "rows of 0 hold nothing");
    check::is_true(pc::wht(std::vector<double>{}, 0).empty(), "empty");
}

static void test_properties() {
    // fp16-grid blocks of 64 transform exactly (sums of k/1024, then / 8),
    // so the WHT is its own inverse bit for bit on the 64-blocks.
    const auto x = fp16_input(4096);
    check::is_true(pc::wht(pc::wht(x)) == x, "wht self-inverse on fp16 64-blocks");
    // precode then unprecode is the identity within rounding, everywhere.
    const auto r = rat_input(N);
    const auto back = pc::unprecode(pc::precode(r, Q_TEST), Q_TEST);
    double worst = 0;
    for (std::size_t i = 0; i < N; ++i) worst = std::fmax(worst, std::fabs(back[i] - r[i]));
    check::is_true(worst < 1e-12, "unprecode(precode(a)) == a");
    // Sylvester order: the transform of a unit vector e_j in a block of 8 is
    // column j of H / sqrt(8), H[i, j] = (-1)^popcount(i & j).
    for (std::size_t j = 0; j < 8; ++j) {
        std::vector<double> e(8, 0.0);
        e[j] = 1.0;
        const auto y = pc::wht(e);
        bool ok = true;
        for (std::size_t i = 0; i < 8; ++i) {
            int parity = 0;
            for (std::size_t b = i & j; b; b >>= 1) parity ^= static_cast<int>(b & 1);
            const double want = (parity ? -1.0 : 1.0) / std::sqrt(8.0);
            ok = ok && y[i] == want;
        }
        check::is_true(ok, "Sylvester column " + std::to_string(j));
    }
}

int main() {
    try {
        test_block_sizes();
        test_digests();
        test_batch_rows();
        test_properties();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_precoder");
}
