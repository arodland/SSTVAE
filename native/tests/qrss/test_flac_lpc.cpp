// The passband store's FLAC encoder (native/core/qrss/flac.hpp) with its
// LPC subframes, with no Python. Every stream here is read twice: by
// dr_flac, and by a strict decoder of this file's own that parses every
// field and refuses anything outside what the encoder promises -- a FLAC
// subset stream (LPC order <= 12 and partition order <= 8 at the defaults,
// coefficient precision <= 15, a non-negative shift, no wasted bits), the
// CRCs right, the STREAMINFO's MD5 and minimum and maximum frame sizes the
// stream's own, every seek point on a frame, and every LPC prediction sum
// within 32 bits (bps + precision + ilog2(order) <= 32: what libFLAC's
// encoder keeps so that a decoder may predict in 32 bits, as many do).
// The cross-checks against libFLAC (soundfile) and ffmpeg are
// tests/test_qrss_native_flac.py's.
//
// Beyond that: LPC is chosen for band-limited noise and costs less than
// FIXED there; every order 1..32 and every precision 5..15 round trips and
// is used when allowed; the partial_tukey windows pay on a block whose
// halves differ; full scale, impulses, white noise, silence and a partial
// last block, mono and stereo, at extreme options; and a seeded fuzz over
// lengths, contents and options.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/flac.hpp"
#include "qrss_check.hpp"

#define CHECK(c) sstvae::check::is_true((c), std::string(#c) + " (line " + std::to_string(__LINE__) + ")")

namespace flac = sstvae::qrss::flac;

namespace {

// --- a strict decoder ---------------------------------------------------------------------------

struct Malformed : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void require(bool ok, const char* what) {
    if (!ok) throw Malformed(what);
}

class BitReader {
public:
    BitReader(const std::uint8_t* p, std::size_t n, std::size_t at) : p_(p), n_(n), bit_(at * 8) {}
    std::uint32_t bits(int k) {
        std::uint32_t v = 0;
        for (int i = 0; i < k; ++i) {
            require(bit_ < n_ * 8, "read past the end");
            v = (v << 1) | ((p_[bit_ >> 3] >> (7 - (bit_ & 7))) & 1u);
            ++bit_;
        }
        return v;
    }
    std::int64_t sbits(int k) {
        if (k == 0) return 0;
        const std::uint64_t v = bits(k);
        return static_cast<std::int64_t>(v) - ((v >> (k - 1)) & 1 ? (std::int64_t{1} << k) : 0);
    }
    std::uint64_t unary() {
        std::uint64_t q = 0;
        while (bits(1) == 0) ++q;
        return q;
    }
    void align_zero() {
        while (bit_ & 7) require(bits(1) == 0, "nonzero frame padding");
    }
    std::size_t byte() const { return bit_ >> 3; }

private:
    const std::uint8_t* p_;
    std::size_t n_;
    std::size_t bit_;
};

struct Parsed {
    int channels = 0, sample_rate = 0, bps = 0;
    std::uint64_t total = 0;
    std::array<std::uint8_t, 16> md5{};
    std::uint32_t min_block = 0, max_block = 0, min_frame = 0, max_frame = 0;
    std::vector<std::array<std::uint64_t, 3>> seek;   // (sample, offset from the first frame, samples)
    std::vector<std::uint64_t> frame_at;                // each frame's offset from the first frame
    std::vector<std::uint32_t> frame_bytes;
    std::vector<int> frame_n;
    std::vector<std::int16_t> pcm;
    std::array<std::uint64_t, 8> subframes{};           // as EncodeStats counts them
    std::array<std::uint64_t, 33> lpc_orders{};
    std::array<std::uint64_t, 4> stereo{};              // frames by assignment: L/R, L/S, S/R, M/S
    int max_precision = 0;                              // over LPC subframes, as written
    int max_partition_order = 0;
    int max_bound = 0;                                  // max bps + precision + ilog2(order)
    std::int64_t max_sum = 0;                           // max |LPC prediction sum|
};

int ilog2(unsigned v) { return static_cast<int>(std::bit_width(v)) - 1; }

// The residual section of a subframe of n samples, predictor order p.
void read_residual(BitReader& br, int n, int p, std::vector<std::int64_t>& r, Parsed& out) {
    const std::uint32_t method = br.bits(2);
    require(method <= 1, "reserved residual method");
    const int porder = static_cast<int>(br.bits(4));
    require(porder <= 8, "partition order above 8 (dr_flac refuses it)");
    out.max_partition_order = std::max(out.max_partition_order, porder);
    require((n & ((1 << porder) - 1)) == 0, "block size not divisible by the partitions");
    const int s = n >> porder;
    require(s > p, "first partition no longer than the warm-up (dr_flac refuses it)");
    const int pw = method == 0 ? 4 : 5;
    const std::uint32_t esc = method == 0 ? 15u : 31u;
    r.clear();
    for (int j = 0; j < (1 << porder); ++j) {
        const int count = j == 0 ? s - p : s;
        const std::uint32_t k = br.bits(pw);
        if (k == esc) {
            const int raw = static_cast<int>(br.bits(5));
            for (int i = 0; i < count; ++i) r.push_back(br.sbits(raw));
        } else {
            for (int i = 0; i < count; ++i) {
                const std::uint64_t q = br.unary();
                require(q < (std::uint64_t{1} << 40), "runaway Rice code");
                const std::uint64_t u = (q << k) | br.bits(static_cast<int>(k));
                r.push_back(static_cast<std::int64_t>(u >> 1) ^ -static_cast<std::int64_t>(u & 1));
            }
        }
    }
}

// One subframe of n samples at `bps` bits into x.
void read_subframe(BitReader& br, int n, int bps, std::vector<std::int64_t>& x, Parsed& out) {
    require(br.bits(1) == 0, "subframe padding bit");
    const std::uint32_t type = br.bits(6);
    require(br.bits(1) == 0, "wasted bits (never written)");
    x.assign(static_cast<std::size_t>(n), 0);
    std::vector<std::int64_t> r;
    if (type == 0) {
        const std::int64_t v = br.sbits(bps);
        std::fill(x.begin(), x.end(), v);
        ++out.subframes[0];
        return;
    }
    if (type == 1) {
        for (auto& v : x) v = br.sbits(bps);
        ++out.subframes[1];
        return;
    }
    if (type >= 8 && type <= 12) {
        const int p = static_cast<int>(type) - 8;
        require(p < n, "fixed order past the block");
        for (int i = 0; i < p; ++i) x[static_cast<std::size_t>(i)] = br.sbits(bps);
        read_residual(br, n, p, r, out);
        for (int i = p; i < n; ++i) {
            const std::int64_t* y = x.data() + i;
            std::int64_t pred = 0;
            switch (p) {
                case 1: pred = y[-1]; break;
                case 2: pred = 2 * y[-1] - y[-2]; break;
                case 3: pred = 3 * y[-1] - 3 * y[-2] + y[-3]; break;
                case 4: pred = 4 * y[-1] - 6 * y[-2] + 4 * y[-3] - y[-4]; break;
                default: break;
            }
            x[static_cast<std::size_t>(i)] = pred + r[static_cast<std::size_t>(i - p)];
        }
        ++out.subframes[static_cast<std::size_t>(2 + p)];
        return;
    }
    require(type >= 32, "reserved subframe type");
    const int order = static_cast<int>(type) - 31;
    require(order < n, "LPC order past the block");
    for (int i = 0; i < order; ++i) x[static_cast<std::size_t>(i)] = br.sbits(bps);
    const int precision = static_cast<int>(br.bits(4)) + 1;
    require(precision <= 15, "coefficient precision 16 (reserved)");
    const int shift = static_cast<int>(br.sbits(5));
    require(shift >= 0, "negative shift");
    std::array<std::int64_t, 32> q{};
    for (int j = 0; j < order; ++j) q[static_cast<std::size_t>(j)] = br.sbits(precision);
    read_residual(br, n, order, r, out);
    for (int i = order; i < n; ++i) {
        std::int64_t sum = 0;
        for (int j = 0; j < order; ++j) sum += q[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(i - 1 - j)];
        out.max_sum = std::max(out.max_sum, sum < 0 ? -sum : sum);
        x[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(i - order)] + (sum >> shift);
    }
    ++out.subframes[7];
    ++out.lpc_orders[static_cast<std::size_t>(order)];
    out.max_precision = std::max(out.max_precision, precision);
    out.max_bound = std::max(out.max_bound, bps + precision + ilog2(static_cast<unsigned>(order)));
}

std::uint64_t be(const std::uint8_t* p, int n) {
    std::uint64_t v = 0;
    for (int i = 0; i < n; ++i) v = (v << 8) | p[i];
    return v;
}

Parsed parse(const std::string& s) {
    const auto* b = reinterpret_cast<const std::uint8_t*>(s.data());
    const std::size_t len = s.size();
    Parsed out;
    require(len >= 42 && s.compare(0, 4, "fLaC") == 0, "no fLaC marker");
    std::size_t at = 4;
    bool last = false, have_info = false;
    while (!last) {
        require(at + 4 <= len, "truncated metadata");
        last = (b[at] & 0x80) != 0;
        const int type = b[at] & 0x7F;
        const std::size_t n = be(b + at + 1, 3);
        at += 4;
        require(at + n <= len, "truncated metadata block");
        if (type == 0) {
            require(n == 34 && !have_info && at == 8, "STREAMINFO not first or not 34 bytes");
            have_info = true;
            out.min_block = static_cast<std::uint32_t>(be(b + at, 2));
            out.max_block = static_cast<std::uint32_t>(be(b + at + 2, 2));
            out.min_frame = static_cast<std::uint32_t>(be(b + at + 4, 3));
            out.max_frame = static_cast<std::uint32_t>(be(b + at + 7, 3));
            const std::uint64_t packed = be(b + at + 10, 8);
            out.sample_rate = static_cast<int>(packed >> 44);
            out.channels = static_cast<int>((packed >> 41) & 7) + 1;
            out.bps = static_cast<int>((packed >> 36) & 31) + 1;
            out.total = packed & ((std::uint64_t{1} << 36) - 1);
            std::copy(b + at + 18, b + at + 34, out.md5.begin());
        } else if (type == 3) {
            require(n % 18 == 0, "seek table not whole points");
            for (std::size_t k = 0; k < n; k += 18)
                out.seek.push_back({be(b + at + k, 8), be(b + at + k + 8, 8), be(b + at + k + 16, 2)});
        } else {
            require(false, "a metadata block the encoder never writes");
        }
        at += n;
    }
    require(have_info, "no STREAMINFO");
    require(out.bps == 16 && out.min_block == out.max_block && out.min_block >= 16, "STREAMINFO block/sample size");
    const std::size_t first = at;
    const int bs = static_cast<int>(out.max_block);
    std::uint64_t decoded = 0;
    std::vector<std::int64_t> ch[2];
    while (at < len) {
        const std::size_t f0 = at;
        require(at + 6 <= len && b[at] == 0xFF && b[at + 1] == 0xF8, "frame sync (fixed block size)");
        const int bscode = b[at + 2] >> 4, srcode = b[at + 2] & 15;
        const int assign = b[at + 3] >> 4, sscode = (b[at + 3] >> 1) & 7;
        require((b[at + 3] & 1) == 0 && sscode == 4, "sample size 16 in the frame header");
        at += 4;
        // the frame number, UTF-8 coded
        std::uint64_t num = b[at];
        int more = 0;
        if (num >= 0x80) {
            more = std::countl_one(static_cast<std::uint8_t>(num)) - 1;
            require(more >= 1 && more <= 6, "frame number coding");
            num &= 0x3Fu >> more;
        }
        ++at;
        for (int i = 0; i < more; ++i) {
            require(at < len && (b[at] & 0xC0) == 0x80, "frame number continuation");
            num = (num << 6) | (b[at++] & 0x3F);
        }
        require(num == out.frame_at.size(), "frame number out of sequence");
        int n = 0;
        if (bscode == 1) n = 192;
        else if (bscode >= 2 && bscode <= 5) n = 576 << (bscode - 2);
        else if (bscode == 6) n = b[at++] + 1;
        else if (bscode == 7) {
            n = static_cast<int>(be(b + at, 2)) + 1;
            at += 2;
        }
        else if (bscode >= 8) n = 256 << (bscode - 8);
        else require(false, "reserved block size code");
        int fs = 0;
        static constexpr int RATES[12] = {0, 88200, 176400, 192000, 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000};
        if (srcode >= 1 && srcode <= 11) fs = RATES[srcode];
        else if (srcode == 12) fs = b[at++] * 1000;
        else if (srcode == 13 || srcode == 14) {
            fs = static_cast<int>(be(b + at, 2)) * (srcode == 14 ? 10 : 1);
            at += 2;
        }
        else require(false, "sample rate not in the frame header (not subset)");
        require(fs == out.sample_rate, "frame sample rate");
        require(at < len && flac::crc8(b + f0, at - f0) == b[at], "frame header CRC-8");
        ++at;
        const bool is_last = decoded + static_cast<std::uint64_t>(n) >= out.total;
        require(is_last ? decoded + static_cast<std::uint64_t>(n) == out.total : n == bs, "block size");
        require(out.channels == 1 ? assign == 0 : (assign == 1 || (assign >= 8 && assign <= 10)), "channel assignment");
        BitReader br(b, len, at);
        for (int c = 0; c < out.channels; ++c) {
            const bool side = (assign == 8 && c == 1) || (assign == 9 && c == 0) || (assign == 10 && c == 1);
            read_subframe(br, n, side ? 17 : 16, ch[c], out);
        }
        br.align_zero();
        at = br.byte();
        require(at + 2 <= len && flac::crc16(b + f0, at - f0) == be(b + at, 2), "frame CRC-16");
        at += 2;
        if (out.channels == 2) ++out.stereo[static_cast<std::size_t>(assign == 1 ? 0 : assign - 7)];
        for (int i = 0; i < n; ++i) {
            std::int64_t l = ch[0][static_cast<std::size_t>(i)];
            std::int64_t r = out.channels == 2 ? ch[1][static_cast<std::size_t>(i)] : 0;
            if (assign == 8) r = l - r;
            else if (assign == 9) l = l + r;
            else if (assign == 10) {
                const std::int64_t mid = (l * 2) | (r & 1), side = r;
                l = (mid + side) >> 1;
                r = (mid - side) >> 1;
            }
            require(l >= -32768 && l <= 32767 && r >= -32768 && r <= 32767, "decoded sample past 16 bits");
            out.pcm.push_back(static_cast<std::int16_t>(l));
            if (out.channels == 2) out.pcm.push_back(static_cast<std::int16_t>(r));
        }
        out.frame_at.push_back(f0 - first);
        out.frame_bytes.push_back(static_cast<std::uint32_t>(at - f0));
        out.frame_n.push_back(n);
        decoded += static_cast<std::uint64_t>(n);
    }
    require(decoded == out.total, "fewer samples than STREAMINFO says");
    return out;
}

// --- encoding and checking a stream -------------------------------------------------------------

struct Result {
    Parsed parsed;
    flac::EncodeStats stats;
    std::size_t bytes = 0;
};

// x through the encoder at `o`, read back by the strict decoder and by
// dr_flac; every promise above checked. `subset` adds the subset limits.
Result check_stream(const std::vector<std::int16_t>& x, int ch, const flac::EncodeOptions& o, bool subset,
                    const std::string& what) {
    Result res;
    const std::string bytes = flac::encode(x, ch, 4000, o, &res.stats);
    res.bytes = bytes.size();
    try {
        res.parsed = parse(bytes);
    } catch (const Malformed& e) {
        sstvae::check::is_true(false, what + ": malformed stream: " + e.what());
        return res;
    }
    const Parsed& p = res.parsed;
    const auto ok = [&](bool c, const char* name) { sstvae::check::is_true(c, what + ": " + name); };
    ok(p.pcm == x, "strict decode");
    flac::Info info;
    ok(flac::decode(bytes, &info) == x, "dr_flac decode");
    const auto md5 = flac::pcm16_md5(x);
    ok(p.md5 == md5 && info.md5 == md5 && res.stats.md5 == md5, "STREAMINFO MD5");
    ok(p.channels == ch && p.sample_rate == 4000 && p.total == x.size() / static_cast<std::size_t>(ch), "STREAMINFO");
    ok(p.min_block == static_cast<std::uint32_t>(o.blocksize), "STREAMINFO block size");
    if (!p.frame_bytes.empty()) {
        ok(p.min_frame == *std::min_element(p.frame_bytes.begin(), p.frame_bytes.end()) &&
               p.max_frame == *std::max_element(p.frame_bytes.begin(), p.frame_bytes.end()),
           "STREAMINFO frame sizes");
    }
    ok(res.stats.bytes == bytes.size() && res.stats.frames == p.frame_bytes.size(), "stats bytes, frames");
    ok(res.stats.subframes == p.subframes && res.stats.lpc_orders == p.lpc_orders, "stats subframes");
    if (ch == 2 && o.stereo_decorrelation) ok(res.stats.stereo == p.stereo, "stats stereo");
    // seek points: every seek_every_frames'th frame, where it is
    const std::size_t every = o.seek_every_frames > 0 ? static_cast<std::size_t>(o.seek_every_frames) : 0;
    ok(p.seek.size() == (every ? (p.frame_at.size() + every - 1) / every : 0), "seek point count");
    for (std::size_t k = 0; k < p.seek.size(); ++k) {
        const std::size_t f = k * every;
        ok(f < p.frame_at.size() && p.seek[k][0] == f * static_cast<std::uint64_t>(o.blocksize) &&
               p.seek[k][1] == p.frame_at[f] && p.seek[k][2] == static_cast<std::uint64_t>(p.frame_n[f]),
           "seek point on its frame");
    }
    // 32-bit prediction: the bound, and the sums themselves
    ok(p.max_bound <= 32 && p.max_sum <= std::numeric_limits<std::int32_t>::max(), "32-bit prediction");
    for (int k = o.max_lpc_order + 1; k <= 32; ++k) ok(p.lpc_orders[static_cast<std::size_t>(k)] == 0, "order cap");
    ok(p.max_precision <= (o.qlp_precision ? o.qlp_precision : 15), "precision cap");
    ok(p.max_partition_order <= o.max_partition_order, "partition order cap");
    if (subset) {
        ok(o.blocksize <= 4608 && p.max_partition_order <= 8 && p.max_precision <= 15, "subset");
        for (int k = 13; k <= 32; ++k) ok(p.lpc_orders[static_cast<std::size_t>(k)] == 0, "subset LPC order");
    }
    return res;
}

// --- signals ------------------------------------------------------------------------------------

std::int16_t clip16(double v) { return static_cast<std::int16_t>(std::clamp(std::nearbyint(v), -32768.0, 32767.0)); }

// White noise through a two-pole resonator at `f` (cycles per sample) of
// pole radius `r`, scaled to `rms`: band-limited noise.
std::vector<double> resonant(std::size_t n, double f, double r, double rms, std::mt19937& g) {
    std::normal_distribution<double> d(0.0, 1.0);
    std::vector<double> y(n);
    double y1 = 0.0, y2 = 0.0;
    const double a1 = 2 * r * std::cos(2 * 3.14159265358979323846 * f), a2 = -r * r;
    for (std::size_t i = 0; i < n + 1000; ++i) {
        const double v = d(g) + a1 * y1 + a2 * y2;
        y2 = y1;
        y1 = v;
        if (i >= 1000) y[i - 1000] = v;
    }
    double e = 0.0;
    for (double v : y) e += v * v;
    const double k = n ? rms / std::sqrt(e / static_cast<double>(n)) : 0.0;
    for (double& v : y) v *= k;
    return y;
}

std::vector<std::int16_t> stereo(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<std::int16_t> x;
    for (std::size_t i = 0; i < a.size(); ++i) {
        x.push_back(clip16(a[i]));
        x.push_back(clip16(b[i]));
    }
    return x;
}

std::vector<std::int16_t> mono(const std::vector<double>& a) {
    std::vector<std::int16_t> x;
    for (double v : a) x.push_back(clip16(v));
    return x;
}

// Band-limited noise as the passband store holds it: I and Q of a band of
// a few resonances over a weaker floor.
std::vector<std::int16_t> band_noise(std::size_t n, double rms, unsigned seed) {
    std::mt19937 g(seed);
    auto i1 = resonant(n, 0.05, 0.98, rms, g), i2 = resonant(n, 0.21, 0.95, rms * 0.5, g);
    auto q1 = resonant(n, 0.05, 0.98, rms, g), q2 = resonant(n, 0.21, 0.95, rms * 0.5, g);
    for (std::size_t k = 0; k < n; ++k) {
        i1[k] += i2[k];
        q1[k] += q2[k];
    }
    return stereo(i1, q1);
}

// --- tests --------------------------------------------------------------------------------------

void test_residual_kernel() {
    // The encoder's residual kernel against a plain 64-bit reference: every
    // order, 16- and 17-bit samples, both of its instantiations (int16 and
    // int32 sums), every shift, random and extreme coefficients and samples,
    // lengths below and above a block of eight -- predictions past 2^27
    // included (a block formed again in 64 bits) and residuals past 31 bits
    // (refused).
    std::mt19937 g(11);
    auto pick = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(g); };
    int redone = 0, refused = 0, plain = 0;
    for (int t = 0; t < 4000; ++t) {
        const int mode = pick(0, 4);   // 1, 2: the largest coefficients (1: every product at its largest)
        const int order = pick(1, 32);
        const int bps = pick(16, 17);
        const int widest = std::min(15, 32 - bps - ilog2(static_cast<unsigned>(order)));
        const int qbits = mode == 1 || mode == 2 ? widest : pick(1, widest);
        const int n = order + (pick(0, 3) ? pick(0, 40) : pick(41, 400));
        const int shift = mode == 1 ? pick(0, 4) : pick(0, 15);
        const std::int32_t xmin = -(1 << (bps - 1)), xmax = (1 << (bps - 1)) - 1;
        const std::int32_t qmin = -(1 << (qbits - 1)), qmax = (1 << (qbits - 1)) - 1;
        std::vector<std::int32_t> x(static_cast<std::size_t>(n)), q(static_cast<std::size_t>(order));
        for (auto& v : q) {
            v = mode == 1 ? qmin : mode == 2 ? qmax : pick(qmin, qmax);
        }
        for (auto& v : x) {
            if (mode == 1) v = xmin;                       // every product at its largest
            else if (mode == 2) v = pick(0, 9) ? xmax : xmin;
            else if (mode == 3) v = pick(-40, 40);
            else v = pick(xmin, xmax);
        }
        std::vector<std::int32_t> want;
        std::uint64_t want_sum = 0;
        bool want_ok = true, big = false;
        for (int i = order; i < n; ++i) {
            std::int64_t sum = 0;
            for (int j = 0; j < order; ++j)
                sum += static_cast<std::int64_t>(q[static_cast<std::size_t>(j)]) * x[static_cast<std::size_t>(i - 1 - j)];
            CHECK(sum >= std::numeric_limits<std::int32_t>::min() && sum <= std::numeric_limits<std::int32_t>::max());
            const std::int64_t pred = sum >> shift;
            big = big || pred > (1 << 27) || pred < -(1 << 27);
            const std::int64_t v = x[static_cast<std::size_t>(i)] - pred;
            if (v > (1 << 30) - 1 || v < -((1 << 30) - 1)) want_ok = false;
            want.push_back(static_cast<std::int32_t>(v));
            want_sum += static_cast<std::uint64_t>(v < 0 ? -v : v);
        }
        std::vector<std::int16_t> x16;
        const bool narrow = bps == 16 && pick(0, 2) != 0;
        if (narrow)
            for (auto v : x) x16.push_back(static_cast<std::int16_t>(v));
        std::vector<std::int32_t> r{1, 2, 3};
        std::uint64_t got_sum = 12345;
        const bool ok = flac::detail::lpc_residual(x.data(), narrow ? x16.data() : nullptr, n, q.data(), order, shift,
                                                   r, got_sum);
        sstvae::check::is_true(ok == want_ok, "kernel trial " + std::to_string(t) + ": refusal");
        if (ok && want_ok)
            sstvae::check::is_true(r == want && got_sum == want_sum, "kernel trial " + std::to_string(t) + ": residual");
        if (!want_ok) ++refused;
        else if (big) ++redone;
        else ++plain;
    }
    CHECK(redone > 100 && refused > 100 && plain > 1000);
}

void test_lpc_on_band_noise() {
    const auto x = band_noise(5 * 4096 + 1234, 600.0, 1);
    const flac::EncodeOptions d;
    const Result r = check_stream(x, 2, d, true, "band noise");
    CHECK(r.stats.subframes[7] == 2 * r.stats.frames);   // LPC in every subframe
    flac::EncodeOptions fixed_only;
    fixed_only.max_lpc_order = 0;
    const Result f = check_stream(x, 2, fixed_only, true, "band noise, FIXED only");
    CHECK(f.stats.subframes[7] == 0);
    CHECK(r.bytes < f.bytes * 0.93);   // (measured 0.86)
    // mono alike
    std::vector<std::int16_t> m;
    for (std::size_t i = 0; i < x.size(); i += 2) m.push_back(x[i]);
    const Result rm = check_stream(m, 1, d, true, "band noise mono");
    CHECK(rm.stats.subframes[7] == rm.stats.frames);
    // a sine with a little noise: LPC of a low order beats FIXED
    std::mt19937 g(2);
    std::normal_distribution<double> nd(0.0, 3.0);
    std::vector<double> s(30000);
    for (std::size_t i = 0; i < s.size(); ++i) s[i] = 12000.0 * std::sin(0.31 * static_cast<double>(i)) + nd(g);
    const Result rs = check_stream(mono(s), 1, d, true, "sine");
    const Result fs = check_stream(mono(s), 1, fixed_only, true, "sine, FIXED only");
    CHECK(rs.stats.subframes[7] > 0 && rs.bytes < fs.bytes);
    // more orders tried is never worse than one by more than noise, and the estimate's pick alone round trips
    for (int tries : {1, 2, 32}) {
        flac::EncodeOptions o;
        o.lpc_orders_tried = tries;
        const Result t = check_stream(x, 2, o, true, "tries " + std::to_string(tries));
        CHECK(t.bytes <= r.bytes * 1.01 && t.bytes >= r.bytes * 0.98);
    }
}

// An autoregressive process of order `order`: unit Gaussian innovation
// through poles of radius 0.99 (pairs spread over the band, and one real
// pole for an odd order), scaled to an rms of 8000.
std::vector<std::int16_t> autoregressive(int order, std::size_t n, unsigned seed) {
    constexpr double PI = 3.14159265358979323846;
    std::vector<double> poly{1.0};   // the product of the poles' (1 - 2 r cos(w) z^-1 + r^2 z^-2), (1 - r z^-1)
    auto times = [&](std::initializer_list<double> c) {
        std::vector<double> next(poly.size() + c.size() - 1, 0.0);
        for (std::size_t i = 0; i < poly.size(); ++i) {
            std::size_t j = 0;
            for (double v : c) next[i + j++] += poly[i] * v;
        }
        poly = next;
    };
    const double r = 0.99;
    for (int k = 0; k < order / 2; ++k) times({1.0, -2.0 * r * std::cos(PI * (k + 0.5) / (order / 2)), r * r});
    if (order % 2) times({1.0, -r});
    std::mt19937 g(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> y(n + 2000, 0.0);
    for (std::size_t i = 0; i < y.size(); ++i) {
        double v = nd(g);
        for (std::size_t j = 1; j < poly.size() && j <= i; ++j) v -= poly[j] * y[i - j];
        y[i] = v;
    }
    y.erase(y.begin(), y.begin() + 2000);
    double e = 0.0;
    for (double v : y) e += v * v;
    for (double& v : y) v *= 8000.0 / std::sqrt(e / static_cast<double>(y.size()));
    return mono(y);
}

void test_every_order() {
    // Every order 1..32 written and read: an autoregressive process of each
    // order, encoded with that order as the cap, uses it; nothing above a
    // cap is ever used (check_stream).
    for (int order = 1; order <= 32; ++order) {
        const auto x = autoregressive(order, 3 * 4096 + 500, static_cast<unsigned>(order));
        flac::EncodeOptions o;
        o.max_lpc_order = order;
        o.lpc_orders_tried = 1;
        const Result r = check_stream(x, 1, o, order <= 12, "order " + std::to_string(order));
        sstvae::check::is_true(r.stats.lpc_orders[static_cast<std::size_t>(order)] > 0,
                               "order " + std::to_string(order) + " is used");
        o.lpc_orders_tried = 32;
        o.partial_tukey = order % 2 == 0;
        check_stream(x, 1, o, order <= 12, "order " + std::to_string(order) + ", every order tried");
        o.max_lpc_order = 32;
        check_stream(x, 1, o, false, "order " + std::to_string(order) + ", cap 32");
    }
}

void test_every_precision() {
    // every precision, at low and high orders, on 16-bit channels and on a
    // 17-bit side channel (opposite full-scale content: L/S or S/R wins)
    const auto x = band_noise(3 * 4096 + 100, 900.0, 4);
    std::mt19937 g(5);
    auto big = resonant(3 * 4096 + 100, 0.07, 0.995, 14000.0, g);
    auto tiny = resonant(3 * 4096 + 100, 0.2, 0.9, 30.0, g);
    std::vector<double> neg(big.size());
    for (std::size_t i = 0; i < big.size(); ++i) neg[i] = -big[i] + tiny[i];
    const auto side = stereo(big, neg);
    for (int precision = 5; precision <= 15; ++precision) {
        for (int order : {1, 2, 8, 12, 32}) {
            flac::EncodeOptions o;
            o.qlp_precision = precision;
            o.max_lpc_order = order;
            const std::string what = "precision " + std::to_string(precision) + " order " + std::to_string(order);
            const Result r = check_stream(x, 2, o, order <= 12, what);
            CHECK(r.stats.subframes[7] > 0);
            const Result s = check_stream(side, 2, o, order <= 12, what + ", side");
            CHECK(s.stats.subframes[7] > 0);
            CHECK(s.stats.stereo[1] + s.stats.stereo[2] + s.stats.stereo[3] > 0);
        }
    }
    // the precision a 4096 block gets is libFLAC's 12, as written (the
    // fewest bits holding the coefficients: at most 12)
    const Result r = check_stream(x, 2, {}, true, "default precision");
    CHECK(r.parsed.max_precision <= 12 && r.parsed.max_precision >= 10);
}

void test_windows() {
    // Each block holds two different signals, one in each half: the
    // partial_tukey windows see each half alone.
    std::mt19937 g(6);
    const std::size_t n = 6 * 4096;
    auto lo = resonant(n, 0.04, 0.99, 2000.0, g);
    auto hi = resonant(n, 0.37, 0.99, 2000.0, g);
    std::vector<double> a(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = (i % 4096) < 2048 ? lo[i] : hi[i];
    std::vector<double> bursts(n, 0.0);   // silence, then a signal starting in the middle of a block
    for (std::size_t i = 2 * 4096 + 2500; i < n; ++i) bursts[i] = lo[i];
    for (const auto& sig : {a, bursts}) {
        flac::EncodeOptions on, off;
        off.partial_tukey = false;
        const Result r_on = check_stream(mono(sig), 1, on, true, "partial_tukey on");
        const Result r_off = check_stream(mono(sig), 1, off, true, "partial_tukey off");
        CHECK(r_on.bytes < r_off.bytes);
    }
}

void test_extremes() {
    // full scale, impulses, white noise, silence and a partial last block,
    // at the defaults and at every option's far end
    std::vector<flac::EncodeOptions> opts(5);
    opts[1].max_lpc_order = 32;
    opts[1].qlp_precision = 15;
    opts[1].lpc_orders_tried = 32;
    opts[1].max_partition_order = 8;
    opts[2].max_lpc_order = 32;
    opts[2].qlp_precision = 5;
    opts[2].blocksize = 4608;
    opts[2].seek_every_frames = 1;
    opts[3].blocksize = 33;
    opts[3].max_lpc_order = 32;
    opts[3].stereo_decorrelation = false;
    opts[4].blocksize = 9000;   // (FIXED alone, over blocks long enough to fold its sums)
    opts[4].max_lpc_order = 0;
    std::mt19937 g(7);
    std::uniform_int_distribution<int> any(-32768, 32767);
    const std::size_t n = 2 * 9000 + 777;
    std::vector<std::pair<std::string, std::vector<std::int16_t>>> sigs;
    {
        std::vector<std::int16_t> sq, sq2, alt, imp, white, quiet, zero(2 * n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const bool hi = (i / 29) % 2;
            sq.push_back(hi ? 32767 : -32768);
            sq.push_back(hi ? -32768 : 32767);
            sq2.push_back(hi ? 32767 : -32768);
            sq2.push_back(hi ? 32767 : -32768);
            alt.push_back(i % 2 ? 32767 : -32768);
            alt.push_back(i % 3 ? -32768 : 32767);
            const bool spike = i % 997 == 13;
            imp.push_back(spike ? -32768 : 0);
            imp.push_back(i % 1500 == 700 ? 32767 : 0);
            white.push_back(static_cast<std::int16_t>(any(g)));
            white.push_back(static_cast<std::int16_t>(any(g)));
            quiet.push_back(static_cast<std::int16_t>(any(g) % 2));
            quiet.push_back(static_cast<std::int16_t>(any(g) % 3));
        }
        sigs = {{"square opposite", sq}, {"square same", sq2}, {"alternation", alt}, {"impulses", imp},
                {"white", white}, {"quiet", quiet}, {"silence", zero}};
        // a band-limited signal at full scale, clipped
        std::mt19937 g2(8);
        auto loud = resonant(n, 0.13, 0.999, 40000.0, g2);
        auto loud2 = resonant(n, 0.02, 0.9999, 40000.0, g2);
        sigs.emplace_back("clipped band", stereo(loud, loud2));
        // smooth and slow at full scale: Levinson's predictors are large here
        std::vector<double> slow(n), slow2(n);
        for (std::size_t i = 0; i < n; ++i) {
            slow[i] = 32000.0 * std::sin(0.0007 * static_cast<double>(i) * (1.0 + 0.0001 * static_cast<double>(i)));
            slow2[i] = 30000.0 * std::cos(0.003 * static_cast<double>(i));
        }
        sigs.emplace_back("slow full scale", stereo(slow, slow2));
    }
    for (const auto& [name, x] : sigs) {
        for (std::size_t k = 0; k < opts.size(); ++k) {
            const bool subset = opts[k].max_lpc_order <= 12 && opts[k].blocksize <= 4608;
            const Result r = check_stream(x, 2, opts[k], subset, name + " opts " + std::to_string(k));
            if (name == "silence") CHECK(r.stats.subframes[0] == 2 * r.stats.frames);
            if (name == "white") {
                // no predictor helps: within a few bits a sample of verbatim
                CHECK(r.bytes <= x.size() * 2 + r.stats.frames * 64 + 4096);
            }
            std::vector<std::int16_t> m;
            for (std::size_t i = 0; i < x.size(); i += 2) m.push_back(x[i]);
            check_stream(m, 1, opts[k], subset, name + " mono opts " + std::to_string(k));
        }
    }
    // short streams: one sample, a few, one block, one more
    for (std::size_t len : {std::size_t{1}, std::size_t{2}, std::size_t{13}, std::size_t{33}, std::size_t{4096},
                            std::size_t{4097}}) {
        const auto x = band_noise(len, 900.0, static_cast<unsigned>(len));
        check_stream(x, 2, opts[0], true, "length " + std::to_string(len));
        check_stream(x, 2, opts[1], false, "length " + std::to_string(len) + " far options");
    }
}

void test_fuzz() {
    // seeded: lengths (a partial last block nearly always), mono and stereo,
    // every kind of content, every option drawn
    std::mt19937 g(9);
    auto pick = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(g); };
    auto uni = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(g); };
    for (int t = 0; t < 400; ++t) {
        flac::EncodeOptions o;
        const bool defaults = t % 3 == 0;
        if (!defaults) {
            static constexpr int BS[] = {16, 192, 576, 1152, 2048, 4096, 4608, 1000};
            o.blocksize = BS[pick(0, 7)];
            o.max_fixed_order = pick(0, 4);
            o.max_lpc_order = pick(0, 32);
            o.qlp_precision = pick(0, 1) ? 0 : pick(5, 15);
            o.lpc_orders_tried = pick(1, 33);
            o.partial_tukey = pick(0, 1);
            o.max_partition_order = pick(0, 8);
            o.stereo_decorrelation = pick(0, 3) != 0;
            o.seek_every_frames = pick(0, 3);
        }
        const int ch = pick(1, 2);
        const std::size_t n = static_cast<std::size_t>(pick(1, 3 * o.blocksize + 999));
        std::vector<double> c[2];
        const int kind = pick(0, 6);
        for (int k = 0; k < ch; ++k) {
            switch (kind) {
                case 0: c[k] = resonant(n, uni(0.0, 0.5), uni(0.5, 0.9999), uni(1.0, 40000.0), g); break;
                case 1: {   // sine plus noise
                    const double f = uni(0.0, 3.0), amp = uni(0.0, 33000.0), sd = uni(0.0, 100.0);
                    std::normal_distribution<double> nd(0.0, sd > 0 ? sd : 1e-9);
                    c[k].resize(n);
                    for (std::size_t i = 0; i < n; ++i) c[k][i] = amp * std::sin(f * static_cast<double>(i)) + nd(g);
                    break;
                }
                case 2: {   // white
                    std::uniform_real_distribution<double> u(-uni(0.0, 33000.0), uni(0.0, 33000.0));
                    c[k].resize(n);
                    for (auto& v : c[k]) v = u(g);
                    break;
                }
                case 3: {   // squares and impulses at full scale
                    const int per = pick(1, 300);
                    c[k].resize(n);
                    for (std::size_t i = 0; i < n; ++i)
                        c[k][i] = pick(0, 50) == 0 ? (pick(0, 1) ? 32767.0 : -32768.0)
                                                   : ((i / static_cast<std::size_t>(per)) % 2 ? 32767.0 : -32768.0) *
                                                         (k ? -1.0 : 1.0);
                    break;
                }
                case 4: c[k].assign(n, 0.0); break;
                case 5: {   // a burst in silence
                    c[k] = resonant(n, uni(0.0, 0.5), uni(0.8, 0.999), uni(10.0, 5000.0), g);
                    const std::size_t a = static_cast<std::size_t>(pick(0, static_cast<int>(n))),
                                      b = static_cast<std::size_t>(pick(0, static_cast<int>(n)));
                    for (std::size_t i = 0; i < n; ++i)
                        if (i < std::min(a, b) || i >= std::max(a, b)) c[k][i] = 0.0;
                    break;
                }
                default: {   // the passband store's kind
                    const auto x = band_noise(n, uni(10.0, 3000.0), static_cast<unsigned>(pick(0, 1 << 20)));
                    c[k].resize(n);
                    for (std::size_t i = 0; i < n; ++i) c[k][i] = x[2 * i + static_cast<std::size_t>(k)];
                }
            }
        }
        const auto x = ch == 2 ? stereo(c[0], c[1]) : mono(c[0]);
        check_stream(x, ch, o, defaults || (o.max_lpc_order <= 12 && o.blocksize <= 4608),
                     "fuzz " + std::to_string(t) + " kind " + std::to_string(kind) + " n " + std::to_string(n));
    }
}

void test_refusals() {
    const std::vector<std::int16_t> x(100, 0);
    auto refused = [&](auto set) {
        flac::EncodeOptions o;
        set(o);
        try {
            flac::encode(x, 2, 4000, o);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(refused([](flac::EncodeOptions& o) { o.max_lpc_order = 33; }));
    CHECK(refused([](flac::EncodeOptions& o) { o.max_lpc_order = -1; }));
    CHECK(refused([](flac::EncodeOptions& o) { o.qlp_precision = 4; }));
    CHECK(refused([](flac::EncodeOptions& o) { o.qlp_precision = 16; }));
    CHECK(refused([](flac::EncodeOptions& o) { o.qlp_precision = -1; }));
    CHECK(refused([](flac::EncodeOptions& o) { o.lpc_orders_tried = 0; }));
    CHECK(refused([](flac::EncodeOptions& o) { o.max_partition_order = 9; }));
    CHECK(!refused([](flac::EncodeOptions& o) { o.max_lpc_order = 0; }));
    CHECK(!refused([](flac::EncodeOptions& o) { o.lpc_orders_tried = 1000; }));
    // the strict decoder refuses what it should (it is a check only while it can fail)
    const std::string good = flac::encode(band_noise(5000, 500.0, 10), 2, 4000);
    CHECK(parse(good).pcm.size() == 10000);
    for (std::size_t at : {std::size_t{8 + 4}, good.size() / 2, good.size() - 1}) {
        std::string bad = good;
        bad[at] = static_cast<char>(bad[at] ^ 0x04);
        bool threw = false;
        try {
            const Parsed p = parse(bad);
            threw = p.min_frame != *std::min_element(p.frame_bytes.begin(), p.frame_bytes.end());
        } catch (const Malformed&) {
            threw = true;
        }
        sstvae::check::is_true(threw, "a flipped bit at " + std::to_string(at) + " is caught");
    }
}

}  // namespace

int main() {
    try {
        test_residual_kernel();
        test_lpc_on_band_noise();
        test_every_order();
        test_every_precision();
        test_windows();
        test_extremes();
        test_fuzz();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return sstvae::check::report("qrss_flac_lpc");
}
