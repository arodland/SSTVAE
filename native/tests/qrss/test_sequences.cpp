// native/core/qrss/sequences against digests of the Python reference.
//
// Each digest is hashlib.sha256(np.ascontiguousarray(a).tobytes()) of
// the same call in sstvae/qrss/sequences.py; tests/test_qrss_native_sequences.py
// compares the two implementations directly where Python is available.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "qrss/config.hpp"
#include "qrss/sequences.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
namespace seq = sstvae::qrss::sequences;
using qrss_check::digest_is;

static void test_sha256_fips_vectors() {
    check::equal(util::to_hex(util::sha256(std::string_view(""))),
                 std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"), "sha256('')");
    check::equal(util::to_hex(util::sha256(std::string_view("abc"))),
                 std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "sha256('abc')");
    check::equal(util::to_hex(util::sha256(std::string_view(
                     "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))),
                 std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"), "sha256 448 bits");
    // One million 'a', fed in uneven pieces across block boundaries.
    util::Sha256 h;
    const std::string chunk(997, 'a');
    std::size_t left = 1000000;
    while (left) {
        const std::size_t take = std::min<std::size_t>(left, chunk.size());
        h.update(std::string_view(chunk.data(), take));
        left -= take;
    }
    const auto d = h.digest();
    check::equal(util::to_hex(d),
                 std::string("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"), "sha256 1e6 'a'");
}

static void test_streams() {
    digest_is(seq::preamble_ce(), "41deb0c5a75cfd48a759f005837125a4f062fa857846553f323262758d680bdd",
              "preamble_ce");
    check::equal(seq::preamble_ce().size(), std::size_t(qrss::config::N_PRE), "preamble_ce length");
    digest_is(seq::references_ce(3539), "715b6ad703783d0c23678a3400362778834b0c56fd82c53fe4083c9b58ad7f02",
              "references_ce(3539)");
    digest_is(seq::scrambler(1954752, 50600), "a6f366775e63b7acff30c0aa479daa1e0b3b05c024331fad7f8334286a58c557",
              "scrambler(Q_TEST)");
    digest_is(seq::scrambler((std::uint64_t(1) << 40) + 12345, 50600),
              "d32cdc33fe93f01647dfa322ec87486ebf7b9b4c3572c82826d4eef89d313e98", "scrambler(2^40+12345)");
    digest_is(seq::preamble_l(), "0207b0d82b65e5208eb7935620f40a70c50d4f693fe85e4a7e432a6fb44d7f8a",
              "preamble_l");
    digest_is(seq::references_l(1770), "9011a44942b6bc069c3ea8f82d932d865a8439546a56c6cdc0606c0fe9f6771b",
              "references_l(1770)");
    digest_is(seq::spread_whitener(50600), "3c07eb90a31b78e4beaf1cafe18791d0131c620c0afb34274e65c080b75d5a4d",
              "spread_whitener(50600)");
    const std::uint8_t pre[2] = {1, 2};
    digest_is(seq::sha_bits("QRSSTVAE CE preamble", 1000, std::span<const std::uint8_t>(pre, 2)),
              "0ba44747e6d27a135ee31d26d9f6fe74fa433fc6324ec8a539e9e79c6e4f84bf", "sha_bits with prefix");
}

static void test_prefix_property() {
    // One stream per frame: a shorter count is a prefix of a longer one.
    const auto a = seq::references_ce(165), b = seq::references_ce(3539);
    check::is_true(std::equal(a.begin(), a.end(), b.begin()), "references_ce prefix");
    check::equal(seq::sha_bits("x", 0).size(), std::size_t(0), "sha_bits(0)");
}

static void test_qpsk() {
    const std::vector<std::uint8_t> bits = {0, 0, 0, 1, 1, 0, 1, 1};
    const auto v = seq::qpsk(bits);
    const double r = 1.0 / std::sqrt(2.0);
    check::is_true(v.size() == 4 && v[0] == std::complex<double>(r, r) && v[1] == std::complex<double>(r, -r) &&
                       v[2] == std::complex<double>(-r, r) && v[3] == std::complex<double>(-r, -r),
                   "qpsk mapping");
    bool threw = false;
    try {
        seq::qpsk(std::vector<std::uint8_t>{0, 1, 0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "qpsk refuses an odd count");
}

static void test_slot_clock() {
    check::equal(seq::quarter_hour_count(1759989600), std::int64_t(1955544), "q of 2025-10-09T06:00Z");
    check::equal(seq::quarter_hour_count(-900), std::int64_t(-1), "q before the epoch floors");
    bool threw = false;
    try {
        seq::quarter_hour_count(1759989601);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "off a quarter hour is refused");
    check::equal(seq::quarter_hour_start(1955544), std::int64_t(1759989600), "quarter_hour_start");
}

int main() {
    try {
        test_sha256_fips_vectors();
        test_streams();
        test_prefix_property();
        test_qpsk();
        test_slot_clock();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_sequences");
}
