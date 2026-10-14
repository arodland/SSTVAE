// The passband store's FLAC (native/core/qrss/flac.hpp): our encoder
// against dr_flac, with no Python. The cross-checks against soundfile
// (libFLAC) are tests/test_qrss_native_passband.py's; here the encoder's
// own invariants: lossless round trips over an edge corpus, the CRC
// polynomials against their published check values, MD5 against RFC 1321
// and a pinned digest, CONSTANT subframes for silence, the stereo modes
// actually used, the seek table and partial reads at every offset of a
// small file, and a damaged frame refused rather than skipped.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <atomic>
#include <cmath>

#include "qrss/flac.hpp"
#include "qrss_check.hpp"
#include "util/md5.hpp"

#define CHECK(c) sstvae::check::is_true((c), std::string(#c) + " (line " + std::to_string(__LINE__) + ")")

namespace flac = sstvae::qrss::flac;
using sstvae::util::Md5;

namespace {

std::vector<std::int16_t> round_trip(const std::vector<std::int16_t>& x, int ch, flac::EncodeStats* st = nullptr,
                                     const flac::EncodeOptions& opt = {}) {
    const std::string bytes = flac::encode(x, ch, 4000, opt, st);
    flac::Info info;
    std::vector<std::int16_t> y = flac::decode(bytes, &info);
    CHECK(info.channels == ch);
    CHECK(info.sample_rate == 4000);
    CHECK(info.bits_per_sample == 16);
    CHECK(info.frames == x.size() / static_cast<std::size_t>(ch));
    CHECK(info.md5 == flac::pcm16_md5(x));
    return y;
}

std::vector<std::int16_t> noise(std::size_t n, double sd, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<double> d(0.0, sd);
    std::vector<std::int16_t> x(n);
    for (auto& v : x) v = static_cast<std::int16_t>(std::clamp(std::nearbyint(d(g)), -32767.0, 32767.0));
    return x;
}

void test_md5() {
    auto hex = [](std::string_view s) {
        Md5 m;
        m.update(s);
        return Md5::hex(m.digest());
    };
    // RFC 1321 appendix A.5
    CHECK(hex("") == "d41d8cd98f00b204e9800998ecf8427e");
    CHECK(hex("a") == "0cc175b9c0f1b6a831c399e269772661");
    CHECK(hex("abc") == "900150983cd24fb0d6963f7d28e17f72");
    CHECK(hex("message digest") == "f96b697d7cb7938d525a2f31aaf161d0");
    CHECK(hex("abcdefghijklmnopqrstuvwxyz") == "c3fcd3d76192e4007dfb496cca67e13b");
    CHECK(hex("12345678901234567890123456789012345678901234567890123456789012345678901234567890") ==
          "57edf4a22be3c955ac49da2e2107b67a");
    // fed in pieces that straddle blocks
    std::string big(1000, 'x');
    for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<char>('a' + i % 26);
    Md5 whole;
    whole.update(big);
    Md5 parts;
    for (std::size_t i = 0; i < big.size(); i += 37) parts.update(big.data() + i, std::min<std::size_t>(37, big.size() - i));
    CHECK(Md5::hex(whole.digest()) == Md5::hex(parts.digest()));
    // the FLAC sample convention, pinned from Python:
    // hashlib.md5(np.arange(-500, 500, dtype='<i2').tobytes()).hexdigest()
    std::vector<std::int16_t> ramp;
    for (int v = -500; v < 500; ++v) ramp.push_back(static_cast<std::int16_t>(v));
    CHECK(Md5::hex(flac::pcm16_md5(ramp)) == "4f10c636fd495b2b106654b109f32825");
}

void test_crcs() {
    // the catalogue check values for "123456789": CRC-8 (poly 0x07) 0xF4,
    // CRC-16/UMTS a.k.a. BUYPASS (poly 0x8005, init 0, no reflection) 0xFEE8
    const char* s = "123456789";
    CHECK(flac::crc8(reinterpret_cast<const std::uint8_t*>(s), 9) == 0xF4);
    CHECK(flac::crc16(reinterpret_cast<const std::uint8_t*>(s), 9) == 0xFEE8);
}

void test_corpus() {
    // all zeros, a stereo full hour's worth of blocks scaled down
    {
        std::vector<std::int16_t> z(2 * 50000, 0);
        flac::EncodeStats st;
        CHECK(round_trip(z, 2, &st) == z);
        CHECK(st.subframes[0] == 2 * st.frames);       // every subframe CONSTANT
        CHECK(st.bytes < 2000);
    }
    // full-scale square wave, alternating +-32768 / 32767 (side needs 17 bits)
    {
        std::vector<std::int16_t> x;
        for (int i = 0; i < 9000; ++i) {
            x.push_back(static_cast<std::int16_t>((i / 37) % 2 ? 32767 : -32768));
            x.push_back(static_cast<std::int16_t>((i / 37) % 2 ? -32768 : 32767));
        }
        CHECK(round_trip(x, 2) == x);
        std::vector<std::int16_t> alt;
        for (int i = 0; i < 2 * 10000; ++i) alt.push_back(static_cast<std::int16_t>(i % 2 ? 32767 : -32768));
        CHECK(round_trip(alt, 2) == alt);
        std::vector<std::int16_t> alt2;
        for (int i = 0; i < 2 * 10000; ++i) alt2.push_back(static_cast<std::int16_t>((i / 2) % 2 ? 32767 : -32768));
        CHECK(round_trip(alt2, 2) == alt2);   // L == R alternating: side 0, mid alternating
    }
    // white noise at several levels, mono and stereo
    for (double sd : {0.5, 30.0, 3000.0, 30000.0}) {
        const auto x = noise(2 * 20011, sd, 7);
        CHECK(round_trip(x, 2) == x);
        const auto m = noise(12345, sd, 8);
        CHECK(round_trip(m, 1) == m);
    }
    // one sample; a few; a short last block; exactly one block; one more
    for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5}, std::size_t{16},
                          std::size_t{4095}, std::size_t{4096}, std::size_t{4097}, std::size_t{8192 + 2560}}) {
        const auto x = noise(2 * n, 1000.0, static_cast<unsigned>(n));
        CHECK(round_trip(x, 2) == x);
    }
    // correlated channels (mid/side and left/side must win somewhere)
    {
        const auto a = noise(30000, 2000.0, 3);
        const auto b = noise(30000, 20.0, 4);
        std::vector<std::int16_t> x;
        for (std::size_t i = 0; i < a.size(); ++i) {
            x.push_back(a[i]);
            x.push_back(static_cast<std::int16_t>(std::clamp<int>(a[i] + b[i], -32768, 32767)));
        }
        flac::EncodeStats st;
        CHECK(round_trip(x, 2, &st) == x);
        CHECK(st.stereo[0] < st.frames);    // not left/right throughout
    }
    // smooth signals: the higher fixed orders are used
    {
        std::vector<std::int16_t> x;
        for (int i = 0; i < 20000; ++i) {
            const double t = i / 4000.0;
            x.push_back(static_cast<std::int16_t>(std::nearbyint(20000.0 * std::sin(2 * 3.14159265 * 3.0 * t))));
            x.push_back(static_cast<std::int16_t>(std::nearbyint(15000.0 * std::cos(2 * 3.14159265 * 7.0 * t))));
        }
        flac::EncodeStats st;
        CHECK(round_trip(x, 2, &st) == x);
        CHECK(st.subframes[2 + 3] + st.subframes[2 + 4] > 0);
        CHECK(st.bytes < x.size());         // well under 1 byte a sample
    }
    // a mostly-zero stream with a burst (an hour heard in part)
    {
        std::vector<std::int16_t> x(2 * 100000, 0);
        const auto b = noise(2 * 7000, 800.0, 11);
        std::copy(b.begin(), b.end(), x.begin() + 2 * 41003);
        flac::EncodeStats st;
        CHECK(round_trip(x, 2, &st) == x);
        CHECK(st.subframes[0] >= 2 * (st.frames - 3));
    }
    // options: no stereo, no seek table, other block sizes, order 0 only
    {
        const auto x = noise(2 * 10007, 500.0, 5);
        flac::EncodeOptions o;
        o.stereo_decorrelation = false;
        o.seek_every_frames = 0;
        CHECK(round_trip(x, 2, nullptr, o) == x);
        o.blocksize = 1152;
        o.max_fixed_order = 0;
        o.max_partition_order = 0;
        CHECK(round_trip(x, 2, nullptr, o) == x);
        o.blocksize = 16;
        CHECK(round_trip(x, 2, nullptr, o) == x);
    }
}

void test_partial_reads(const std::filesystem::path& dir) {
    // a small file through the file reader: every offset over two block
    // boundaries, and every length up to a block and a half at a few
    const auto x = noise(2 * 3 * 4096 + 2 * 77, 900.0, 21);
    const std::string path = (dir / "partial.flac").string();
    std::size_t at = 0;
    const flac::Source src = [&](std::uint64_t first, std::size_t n, std::int16_t* out) {
        CHECK(first * 2 == at);
        std::memcpy(out, x.data() + first * 2, n * 2 * sizeof(std::int16_t));
        at += 2 * n;
        return true;
    };
    flac::EncodeOptions o;
    o.seek_every_frames = 1;
    const flac::EncodeStats st = flac::encode_file(path, 4000, 2, x.size() / 2, src, o);
    CHECK(std::filesystem::file_size(path) == st.bytes);
    for (int seek : {1, 0}) {
        std::string p2 = path;
        if (!seek) {
            p2 = (dir / "partial_noseek.flac").string();
            o.seek_every_frames = 0;
            at = 0;
            flac::encode_file(p2, 4000, 2, x.size() / 2, src, o);
        }
        flac::Reader r(p2);
        CHECK(r.info().frames == x.size() / 2);
        const std::uint64_t total = r.info().frames;
        for (std::uint64_t first = 4096 - 40; first < 2 * 4096 + 40; ++first) {
            const auto y = r.read(first, 100);
            CHECK(std::equal(y.begin(), y.end(), x.begin() + static_cast<std::ptrdiff_t>(2 * first)));
            CHECK(y.size() == 200);
        }
        for (std::uint64_t first : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{4095}, std::uint64_t{8191},
                                    total - 1, total - 77}) {
            for (std::uint64_t n = 0; n < 6200; n += 97) {
                const auto y = r.read(first, n);
                const std::uint64_t want = std::min(n, total - first);
                CHECK(y.size() == 2 * want);
                CHECK(std::equal(y.begin(), y.end(), x.begin() + static_cast<std::ptrdiff_t>(2 * first)));
            }
        }
        CHECK(r.read(total, 10).empty());
        CHECK(r.read(total + 5, 10).empty());
        // backwards and forwards again on one reader
        const auto back = r.read(5, 3);
        CHECK(back.size() == 6 && std::equal(back.begin(), back.end(), x.begin() + 10));
    }
}

void test_damage(const std::filesystem::path& dir) {
    // a byte flipped in the middle frame: refused when read through it,
    // never decoded as the next frame's samples
    const auto x = noise(2 * 3 * 4096, 900.0, 31);
    flac::EncodeOptions o;
    o.seek_every_frames = 0;
    std::string bytes = flac::encode(x, 2, 4000, o);
    // find the second frame header after the first: sync 0xFFF8, frame number 1
    std::size_t f2 = std::string::npos;
    for (std::size_t i = 50; i + 4 < bytes.size(); ++i)
        if (static_cast<unsigned char>(bytes[i]) == 0xFF && static_cast<unsigned char>(bytes[i + 1]) == 0xF8 &&
            static_cast<unsigned char>(bytes[i + 4]) == 0x01) {
            f2 = i;
            break;
        }
    CHECK(f2 != std::string::npos);
    bytes[f2 + 400] = static_cast<char>(bytes[f2 + 400] ^ 0x10);
    const std::string path = (dir / "damaged.flac").string();
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
    }
    flac::Reader r(path);
    const auto ok = r.read(0, 4096);
    CHECK(std::equal(ok.begin(), ok.end(), x.begin()));
    bool threw = false;
    try {
        r.read(0, 3 * 4096);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("CRC") != std::string::npos;
    }
    CHECK(threw);
    // not FLAC at all
    const std::string junk = (dir / "junk.flac").string();
    {
        std::FILE* f = std::fopen(junk.c_str(), "wb");
        std::fputs("RIFF not a flac file at all, sorry", f);
        std::fclose(f);
    }
    bool refused = false;
    try {
        flac::Reader j(junk);
    } catch (const std::runtime_error&) {
        refused = true;
    }
    CHECK(refused);
}

void test_cancel(const std::filesystem::path& dir) {
    const auto x = noise(2 * 50000, 900.0, 41);
    std::atomic<bool> stop{false};
    int calls = 0;
    const flac::Source src = [&](std::uint64_t first, std::size_t n, std::int16_t* out) {
        std::memcpy(out, x.data() + first * 2, n * 2 * sizeof(std::int16_t));
        if (++calls == 3) stop = true;
        return true;
    };
    bool cancelled = false;
    try {
        flac::encode_file((dir / "cancel.flac").string(), 4000, 2, x.size() / 2, src, {}, &stop);
    } catch (const flac::Cancelled&) {
        cancelled = true;
    }
    CHECK(cancelled && calls == 3);
    bool source_failed = false;
    try {
        flac::encode_file((dir / "fail.flac").string(), 4000, 2, 100, [](auto, auto, auto) { return false; });
    } catch (const std::runtime_error&) {
        source_failed = true;
    }
    CHECK(source_failed);
}

}  // namespace

int main() {
    std::random_device rd;
    const auto dir = std::filesystem::temp_directory_path() / ("qrss_flac_test_" + std::to_string(rd()));
    std::filesystem::create_directories(dir);
    try {
        test_md5();
        test_crcs();
        test_corpus();
        test_partial_reads(dir);
        test_damage(dir);
        test_cancel(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        std::filesystem::remove_all(dir);
        return 1;
    }
    std::filesystem::remove_all(dir);
    return sstvae::check::report("qrss_flac");
}
