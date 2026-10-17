// native/core/qrss/polar against the Python reference (sstvae/qrss/polar.py).
//
// Digests are hashlib.sha256(np.ascontiguousarray(a).tobytes()) of the same
// call in Python; path metrics are float.hex() of the reference's. The
// decoder's inputs are closed form over integers, so both sides build the
// identical LLR vector:
//
//     e = np.arange(2474); noise = ((e * mult) % mod).astype(float) / mod - 0.5
//     llr = (1.0 - 2.0 * header.encode(H)) * scale + amp * noise
//
// tests/test_qrss_native_polar.py compares the two implementations directly
// (thousands of noisy vectors) where Python is available.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/header.hpp"
#include "qrss/polar.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace pol = sstvae::qrss::polar;
namespace hdr = sstvae::qrss::header;
using qrss_check::digest_is;

namespace {

hdr::HeaderFields test_header() {
    hdr::HeaderFields h;
    h.callsign = "K1ABC/P";
    h.grid = "FN42";
    h.picture_id = 0xDEADBEEF;
    h.mode = 2;
    h.segment = 1;
    h.codec_id = 0xD1D8;
    return h;
}

std::vector<double> closed_form_llr(const std::vector<std::uint8_t>& coded, double scale, std::int64_t mult,
                                    std::int64_t mod, double amp) {
    std::vector<double> llr(coded.size());
    for (std::size_t e = 0; e < coded.size(); ++e) {
        const double noise =
            static_cast<double>((static_cast<std::int64_t>(e) * mult) % mod) / static_cast<double>(mod) - 0.5;
        llr[e] = (1.0 - 2.0 * static_cast<double>(coded[e])) * scale + amp * noise;
    }
    return llr;
}

template <typename F>
std::string refusal(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "(no refusal)";
}

void pm_close(const std::vector<double>& got, const std::vector<double>& want, const std::string& what) {
    check::equal(got.size(), want.size(), what + " path count");
    for (std::size_t i = 0; i < got.size() && i < want.size(); ++i)
        check::is_true(std::fabs(got[i] - want[i]) <= 1e-9 * std::fabs(want[i]),
                       what + " pm[" + std::to_string(i) + "]");
}

}  // namespace

static void test_info_set() {
    check::is_true(pol::info_set_ok(), "INFO_SET hashes to 4e064eee... and is ascending in [0, N)");
    check::equal(pol::INFO_SET.front(), 1015, "first information index");
    check::equal(pol::INFO_SET.back(), 2047, "last information index");
}

static void test_transform() {
    // x_j = XOR over i containing j of u_i: the natural-order Kronecker power.
    std::vector<std::uint8_t> u(pol::N);
    for (int i = 0; i < pol::N; ++i) u[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((i * 37 + i / 5) % 3 == 0);
    const auto x = pol::polar_transform(u);
    bool ok = true;
    for (int j = 0; j < pol::N && ok; ++j) {
        std::uint8_t v = 0;
        for (int i = 0; i < pol::N; ++i)
            if ((i & j) == j) v ^= u[static_cast<std::size_t>(i)];
        ok = v == x[static_cast<std::size_t>(j)];
    }
    check::is_true(ok, "polar_transform is u F^(x)11 in natural order");
    check::is_true(pol::polar_transform(x) == u, "polar_transform is its own inverse");
    // numpy's reshape rule: n = 6 splits (3, then 1); n = 5 does not.
    std::vector<std::uint8_t> six = {1, 0, 1, 1, 0, 1};
    check::is_true(pol::polar_transform(six) == std::vector<std::uint8_t>{0, 0, 1, 1, 1, 1}, "length 6");
    check::is_true(refusal([] { pol::polar_transform(std::vector<std::uint8_t>(5)); }) != "(no refusal)",
                   "length 5 is refused");
    // No rows at all (Python shape (0, 5)): numpy reshapes a size-0 array
    // into any size-0 shape, so the reference refuses nothing.
    check::equal(refusal([] {
                     std::vector<std::uint8_t> none;
                     pol::polar_transform_rows(none, 5);
                 }),
                 std::string("(no refusal)"), "no rows of 5 are not refused");
}

static void test_rate_match_and_fold() {
    std::vector<std::uint8_t> x(pol::N);
    for (int i = 0; i < pol::N; ++i) x[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i % 2);
    const auto coded = pol::rate_match(x);
    check::equal(coded.size(), std::size_t(2474), "rate_match length");
    bool ok = true;
    for (int e = 0; e < pol::E; ++e) ok = ok && coded[static_cast<std::size_t>(e)] == x[static_cast<std::size_t>(e % pol::N)];
    check::is_true(ok, "coded bit e is mother bit e mod 2048");
    std::vector<double> llr(pol::E);
    for (int e = 0; e < pol::E; ++e) llr[static_cast<std::size_t>(e)] = e;
    const auto f = pol::fold(llr);
    check::is_true(f[0] == 2048.0 && f[425] == 425.0 + 2473.0 && f[426] == 426.0 && f[2047] == 2047.0,
                   "fold adds the repeated copies");
    check::equal(refusal([] { pol::fold(std::vector<double>(2048)); }),
                 std::string("expected 2474 coded LLRs, got shape (2048,)"), "fold refusal");
}

static void test_encode() {
    std::vector<std::uint8_t> info(pol::K);
    for (int t = 0; t < pol::K; ++t) info[static_cast<std::size_t>(t)] = static_cast<std::uint8_t>((t * 7 + t / 3) % 2);
    const auto coded = pol::polar_encode(info);
    digest_is(coded, "162455852ad6390df216b64a792fb7de23b93e47b2e0e2e37fe803d83a6aad41", "polar_encode closed form");
    // Information on INFO_SET, every other u frozen at 0, repeats equal.
    const auto u = pol::polar_transform(std::span<const std::uint8_t>(coded).first(pol::N));
    bool ok = true;
    std::size_t t = 0;
    for (int i = 0; i < pol::N; ++i) {
        const bool is_info = t < info.size() && pol::INFO_SET[t] == i;
        ok = ok && u[static_cast<std::size_t>(i)] == (is_info ? info[t] : 0);
        if (is_info) ++t;
    }
    for (int e = pol::N; e < pol::E; ++e) ok = ok && coded[static_cast<std::size_t>(e)] == coded[static_cast<std::size_t>(e - pol::N)];
    check::is_true(ok, "encoder puts info on INFO_SET and freezes the rest");
    check::equal(refusal([] { pol::polar_encode(std::vector<std::uint8_t>(141)); }),
                 std::string("expected 142 information bits, got shape (141,)"), "encode length refusal");
    std::vector<std::uint8_t> two(pol::K, 0);
    two[3] = 2;
    check::equal(refusal([&] { pol::polar_encode(two); }), std::string("information bits must be 0 or 1"),
                 "encode value refusal");
}

struct DecodeCase {
    const char* name;
    double scale;
    std::int64_t mult, mod;
    double amp;
    const char* info1;                  // decode_list(llr, 1)
    std::vector<double> pm1;
    const char* info8;                  // decode_list(llr, 8)
    std::vector<double> pm8;
    bool decodes8, decodes1;            // header.decode(llr) == H at list 8 / list 1
};

static void test_decoder() {
    const hdr::HeaderFields H = test_header();
    const auto coded = hdr::encode(H);
    const std::vector<DecodeCase> cases = {
        {"mild", 1.0, 2654435761LL, 1000003, 3.0,
         "d3a2d7aa952346d891536d0d223d86b0d7eba1b2e5e8f66bd1c623ea980d075a", {0x1.1b15c1f49b4acp+9},
         "8a34634c8e2319799f375d6c2fd28f1700380a173448206dc663b65b65b7525d",
         {0x1.1b15c1f49b4acp+9, 0x1.597cb1d17c910p+9, 0x1.5a7aa3c9f2f81p+9, 0x1.5b56bc9c441e4p+9,
          0x1.5c2eb01a2a0c5p+9, 0x1.5ce1243ca47d1p+9, 0x1.5d7a085100a22p+9, 0x1.5df14dd726822p+9},
         true, true},
        {"list", 0.65, 40503, 65521, 3.0,
         "3aa78e376f2dcdff604043b16d12482c59b6d47a1dfc8879ea335609da5f212a", {0x1.125a829bf5dc1p+10},
         "031fad2437faab28b4169e46efe662fb7f79a564300e8b59e27525b3895ca02e",
         {0x1.ce005a94f1c97p+9, 0x1.dff8acedfe782p+9, 0x1.e1e3926147aa9p+9, 0x1.e481012f3c169p+9,
          0x1.e6c24321a5375p+9, 0x1.e83207d49fe07p+9, 0x1.e847aada9c13bp+9, 0x1.e8de661a6215fp+9},
         true, false},
        {"hard", 0.6, 40503, 65521, 4.0,
         "3aa78e376f2dcdff604043b16d12482c59b6d47a1dfc8879ea335609da5f212a", {0x1.e736075b0ad7fp+9},
         "387421c5039b271101c5cda5b9f28b2595b2ea9795b87216c4115baf80adcd64",
         {0x1.e41e2afc35772p+9, 0x1.e736075b0ad7fp+9, 0x1.fc2384a8a5fdep+9, 0x1.fdf69057d7e0cp+9,
          0x1.ff86e42e0a8f1p+9, 0x1.00499a9c5f1c5p+10, 0x1.00714ca4af68ap+10, 0x1.01a7ae5e905fdp+10},
         false, false},
        {"noise", 0.0, 2246822519LL, 999983, 6.0,
         "6559f18eac04332045d4f7d8efd19e90daaefa040063e6343d3750b6fc40e185", {0x1.488d67e574591p+10},
         "ee683f66011d9e1e4c10ccd18e1462778a9da63b7ac4198ce14c5f28f5119f81",
         {0x1.488d67e574591p+10, 0x1.49ba92a36045ep+10, 0x1.4d77d4db5a431p+10, 0x1.501fda09cd4a4p+10,
          0x1.514943993cea6p+10, 0x1.535f54721ec2ep+10, 0x1.55b849ef4a961p+10, 0x1.55f760aaf61b2p+10},
         false, false},
    };
    for (const auto& c : cases) {
        const std::string n = c.name;
        const auto llr = closed_form_llr(coded, c.scale, c.mult, c.mod, c.amp);
        const auto r1 = pol::decode_list(llr, 1);
        digest_is(r1.info, c.info1, n + " decode_list(L=1) paths");
        pm_close(r1.pm, c.pm1, n + " L=1");
        const auto r8 = pol::decode_list(llr, 8);
        digest_is(r8.info, c.info8, n + " decode_list(L=8) paths");
        pm_close(r8.pm, c.pm8, n + " L=8");
        check::equal(hdr::decode(llr) == std::optional<hdr::HeaderFields>(H), c.decodes8, n + " header decode");
        check::equal(hdr::decode(llr, 1) == std::optional<hdr::HeaderFields>(H), c.decodes1, n + " header decode L=1");
        const auto bits = pol::polar_decode(llr);
        check::equal(bits.has_value(), c.decodes8, n + " polar_decode accepts");
        if (bits) check::is_true(*bits == hdr::pack(H), n + " polar_decode returns the header bits");
        // A caller's acceptance test replaces the header's.
        check::is_true(!pol::polar_decode(llr, 8, [](std::span<const std::uint8_t>) { return false; }),
                       n + " a refusing accept gives nullopt");
        const auto first = pol::polar_decode(llr, 8, [](std::span<const std::uint8_t>) { return true; });
        check::is_true(first && std::equal(first->begin(), first->end(), r8.info.begin()),
                       n + " an accepting accept gives the first path");
    }
}

static void test_decoder_edges() {
    // No information at all: every decision is a tie, broken toward 0 first.
    const std::vector<double> zero(pol::E, 0.0);
    const auto r = pol::decode_list(zero, 8);
    check::equal(r.m, std::size_t(8), "zero LLRs: eight paths");
    check::is_true(std::all_of(r.info.begin(), r.info.begin() + pol::K, [](std::uint8_t b) { return b == 0; }),
                   "zero LLRs: the all-zero path first");
    check::equal(refusal([&] { pol::decode_list(zero, 0); }), std::string("list_size must be >= 1"),
                 "list_size 0 refused");
    auto inf = zero;
    inf[7] = std::numeric_limits<double>::infinity();
    check::equal(refusal([&] { pol::decode_list(inf, 8); }), std::string("LLRs must be finite"), "inf refused");
    auto big = zero;
    big[3] = 1e308;
    big[3 + pol::N] = 1e308;                // finite, but the fold overflows
    check::equal(refusal([&] { pol::decode_list(big, 8); }), std::string("LLRs must be finite"),
                 "fold overflow refused");
    // Zero LLRs on the repeated copies leave the full mother code.
    const hdr::HeaderFields H = test_header();
    const auto coded = hdr::encode(H);
    std::vector<double> llr(pol::E, 0.0);
    for (int e = 0; e < pol::N; ++e) llr[static_cast<std::size_t>(e)] = 6.0 * (1.0 - 2.0 * coded[static_cast<std::size_t>(e)]);
    check::is_true(hdr::decode(llr) == std::optional<hdr::HeaderFields>(H), "repeated copies may be erased");
    // A list larger than 8 keeps every path finite once it has split enough.
    const auto r16 = pol::decode_list(closed_form_llr(coded, 1.0, 2654435761LL, 1000003, 3.0), 16);
    check::equal(r16.m, std::size_t(16), "list 16");
}

int main() {
    try {
        test_info_set();
        test_transform();
        test_rate_match_and_fold();
        test_encode();
        test_decoder();
        test_decoder_edges();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_polar");
}
