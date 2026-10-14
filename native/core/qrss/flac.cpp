// FLAC encoder (ours) and reader (dr_flac) for the passband store; see
// flac.hpp for what is written and why.

#include "qrss/flac.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>
#include <system_error>
#include <utility>

#include "qrss/rawfile.hpp"
#include "util/md5.hpp"

// dr_flac: decoding only, through our own callbacks (no stdio: names go
// through rawfile/fileio), no Ogg. CRC checks stay on (see flac.hpp).
#define DR_FLAC_IMPLEMENTATION
#define DR_FLAC_NO_STDIO
#define DR_FLAC_NO_OGG
#define DR_FLAC_NO_WCHAR
#include "dr_flac.h"

namespace sstvae::qrss::flac {

// --- checks ----------------------------------------------------------------------

namespace {

struct CrcTables {
    std::uint8_t c8[256];
    std::uint16_t c16[256];
    CrcTables() {
        for (int i = 0; i < 256; ++i) {
            std::uint8_t c = static_cast<std::uint8_t>(i);
            for (int b = 0; b < 8; ++b) c = static_cast<std::uint8_t>((c & 0x80) ? (c << 1) ^ 0x07 : (c << 1));
            c8[i] = c;
            std::uint16_t d = static_cast<std::uint16_t>(i << 8);
            for (int b = 0; b < 8; ++b) d = static_cast<std::uint16_t>((d & 0x8000) ? (d << 1) ^ 0x8005 : (d << 1));
            c16[i] = d;
        }
    }
};

const CrcTables& crc_tables() {
    static const CrcTables t;   // (a function-local static: thread-safe initialisation)
    return t;
}

}  // namespace

std::uint8_t crc8(const std::uint8_t* p, std::size_t n) {
    const auto& t = crc_tables();
    std::uint8_t c = 0;
    for (std::size_t i = 0; i < n; ++i) c = t.c8[c ^ p[i]];
    return c;
}

std::uint16_t crc16(const std::uint8_t* p, std::size_t n) {
    const auto& t = crc_tables();
    std::uint16_t c = 0;
    for (std::size_t i = 0; i < n; ++i) c = static_cast<std::uint16_t>((c << 8) ^ t.c16[(c >> 8) ^ p[i]]);
    return c;
}

namespace {

void md5_pcm16(util::Md5& m, const std::int16_t* x, std::size_t n) {
    std::uint8_t buf[1024];
    std::size_t i = 0;
    while (i < n) {
        const std::size_t k = std::min<std::size_t>(n - i, sizeof buf / 2);
        for (std::size_t j = 0; j < k; ++j) {
            const auto v = static_cast<std::uint16_t>(x[i + j]);
            buf[2 * j] = static_cast<std::uint8_t>(v & 0xFF);
            buf[2 * j + 1] = static_cast<std::uint8_t>(v >> 8);
        }
        m.update(buf, 2 * k);
        i += k;
    }
}

}  // namespace

std::array<std::uint8_t, 16> pcm16_md5(std::span<const std::int16_t> interleaved) {
    util::Md5 m;
    md5_pcm16(m, interleaved.data(), interleaved.size());
    return m.digest();
}

// --- encoder ---------------------------------------------------------------------

namespace {

class BitWriter {
public:
    explicit BitWriter(std::vector<std::uint8_t>& out) : out_(out) {}

    // The low `bits` (0..32) of v, most significant first.
    void put(std::uint32_t v, int bits) {
        if (bits == 0) return;
        const std::uint64_t m = bits == 32 ? 0xFFFFFFFFull : ((1ull << bits) - 1);
        acc_ = (acc_ << bits) | (v & m);
        n_ += bits;
        while (n_ >= 8) {
            n_ -= 8;
            out_.push_back(static_cast<std::uint8_t>(acc_ >> n_));
        }
    }
    void put_signed(std::int32_t v, int bits) { put(static_cast<std::uint32_t>(v), bits); }
    void zeros(std::uint64_t q) {
        while (q >= 32) {
            put(0, 32);
            q -= 32;
        }
        put(0, static_cast<int>(q));
    }
    // A Rice code: q = u >> k zeros, a one, then the low k bits.
    void rice(std::uint32_t u, int k) {
        const std::uint32_t q = u >> k;
        if (q + 1 + static_cast<std::uint32_t>(k) <= 32) {
            put((1u << k) | (k ? (u & ((1u << k) - 1)) : 0u), static_cast<int>(q) + 1 + k);
        } else {
            zeros(q);
            put(1, 1);
            put(k ? (u & ((1u << k) - 1)) : 0u, k);
        }
    }
    void align() {
        if (n_ & 7) put(0, 8 - (n_ & 7));
    }

private:
    std::vector<std::uint8_t>& out_;
    std::uint64_t acc_ = 0;
    int n_ = 0;
};

inline std::uint32_t zigzag(std::int32_t r) {
    return (static_cast<std::uint32_t>(r) << 1) ^ static_cast<std::uint32_t>(r >> 31);
}

// Bits for a two's-complement field holding every value in [lo, hi].
// (lo <= 0 <= hi: a residual's range always holds 0.)
int signed_bits(std::int64_t lo, std::int64_t hi) {
    if (lo == 0 && hi == 0) return 0;
    const std::uint64_t mag = static_cast<std::uint64_t>(std::max<std::int64_t>(hi, lo < 0 ? ~lo : 0));
    return 1 + static_cast<int>(std::bit_width(mag));
}

int bit_length(std::uint64_t v) { return static_cast<int>(std::bit_width(v)); }

constexpr int SUB_CONSTANT = 0, SUB_VERBATIM = 1, SUB_FIXED = 2;

// One channel's subframe, chosen and costed; `encode_subframe` writes it.
struct Subframe {
    int type = SUB_VERBATIM;
    int order = 0;
    int porder = 0;
    int method = 0;           // 0: 4-bit parameters, 1: 5-bit
    std::uint64_t bits = 0;   // the whole subframe
    std::vector<std::int32_t> residual;
};

// Buffers one stream reuses for every subframe it plans and writes.
struct Scratch {
    std::vector<std::int32_t> r;           // a residual
    std::vector<std::uint32_t> u;          // its zigzag
    std::vector<std::uint64_t> s, s_up;    // [partition][k - wlo]: sum(u >> k)
    std::vector<std::uint64_t> cnt, cnt_up;
    std::vector<std::int64_t> lo, hi, lo_up, hi_up;
};

// sum(u[i] >> k) over [a, b)
std::uint64_t shifted_sum(const std::uint32_t* u, int a, int b, int k) {
    std::uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = a;
    for (; i + 4 <= b; i += 4) {
        s0 += u[i] >> k;
        s1 += u[i + 1] >> k;
        s2 += u[i + 2] >> k;
        s3 += u[i + 3] >> k;
    }
    for (; i < b; ++i) s0 += u[i] >> k;
    return s0 + s1 + s2 + s3;
}

// The lowest Rice parameter in [0, kmax] with the fewest bits for the
// zigzagged residual u[a, b); the bits (parameter field excluded) go in
// `bits`. The cost sum(u >> k) + count (k + 1) is convex in k -- its step
// count - sum(ceil((u >> k) / 2)) never decreases -- so a walk from the
// mean's estimate stops on exactly the parameter an exhaustive search
// finds, ties to the lowest included.
int best_rice(const std::uint32_t* u, int a, int b, int kmax, std::uint64_t& bits) {
    const std::uint64_t count = static_cast<std::uint64_t>(b - a);
    auto cost = [&](int k) { return shifted_sum(u, a, b, k) + count * static_cast<std::uint64_t>(k + 1); };
    const std::uint64_t s0 = shifted_sum(u, a, b, 0);
    int k = count ? std::min(std::max(bit_length(s0 / count) - 1, 0), kmax) : 0;
    std::uint64_t c = cost(k);
    bool moved = false;
    while (k > 0) {
        const std::uint64_t d = cost(k - 1);
        if (d > c) break;
        --k;
        c = d;
        moved = true;
    }
    while (!moved && k < kmax) {
        const std::uint64_t d = cost(k + 1);
        if (d >= c) break;
        ++k;
        c = d;
    }
    bits = c;
    return k;
}

// Cheapest coding of one partition: (bits after the parameter field,
// parameter or -1 for an escape, raw bits for the escape).
struct PartChoice {
    std::uint64_t bits;
    int k;
    int raw;
};

// The escape against a Rice cost already found.
PartChoice with_escape(PartChoice best, std::uint64_t count, std::int64_t lo, std::int64_t hi) {
    const int raw = signed_bits(lo, hi);
    if (raw <= 31) {
        const std::uint64_t b = 5 + count * static_cast<std::uint64_t>(raw);
        if (b < best.bits) best = {b, -1, raw};
    }
    return best;
}

// The partition order and parameter width with the fewest bits for the
// FIXED residual in sc.r of a block of n samples, order p; the residual
// section's bits (method and order fields included) go in `bits`.
void plan_residual(Scratch& sc, int n, int p, int max_porder, int& porder_out, int& method_out,
                   std::uint64_t& bits_out) {
    int rmax = 0;
    while (rmax < max_porder && (n % (1 << (rmax + 1))) == 0 && (n >> (rmax + 1)) > p) ++rmax;
    const std::size_t m = sc.r.size();
    sc.u.resize(m);
    for (std::size_t i = 0; i < m; ++i) sc.u[i] = zigzag(sc.r[i]);
    const std::uint32_t* u = sc.u.data();
    const std::int32_t* r = sc.r.data();
    const int parts = 1 << rmax;
    const int s = n >> rmax;
    const auto P = static_cast<std::size_t>(parts);
    sc.cnt.resize(P);
    sc.lo.resize(P);
    sc.hi.resize(P);
    // Each finest partition's own best parameter. The best for a union of
    // partitions (every coarser order) lies between its parts' best, the
    // costs being convex, so these bound the parameters worth costing at
    // every order; 14, method 0's ceiling, joins them for a window above it.
    int wlo = 30, whi = 0;
    for (int j = 0; j < parts; ++j) {
        const int a = j == 0 ? 0 : j * s - p;
        const int b = (j + 1) * s - p;
        std::int64_t lo = 0, hi = 0;
        for (int i = a; i < b; ++i) {
            lo = std::min<std::int64_t>(lo, r[i]);
            hi = std::max<std::int64_t>(hi, r[i]);
        }
        sc.lo[static_cast<std::size_t>(j)] = lo;
        sc.hi[static_cast<std::size_t>(j)] = hi;
        sc.cnt[static_cast<std::size_t>(j)] = static_cast<std::uint64_t>(b - a);
        std::uint64_t bits = 0;
        const int k = best_rice(u, a, b, 30, bits);
        wlo = std::min(wlo, k);
        whi = std::max(whi, k);
    }
    wlo = std::min(wlo, 14);
    const auto K = static_cast<std::size_t>(whi - wlo + 1);
    sc.s.resize(P * K);
    for (int j = 0; j < parts; ++j) {
        const int a = j == 0 ? 0 : j * s - p;
        const int b = (j + 1) * s - p;
        for (std::size_t kk = 0; kk < K; ++kk)
            sc.s[static_cast<std::size_t>(j) * K + kk] = shifted_sum(u, a, b, wlo + static_cast<int>(kk));
    }
    std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
    std::size_t nparts = P;
    for (int lev = rmax;; --lev) {
        for (int method = 0; method < 2; ++method) {
            std::uint64_t total = 2 + 4;
            const int pw = method == 0 ? 4 : 5;
            const int top = std::min(whi, method == 0 ? 14 : 30);
            for (std::size_t j = 0; j < nparts; ++j) {
                const std::uint64_t* sj = sc.s.data() + j * K;
                const std::uint64_t count = sc.cnt[j];
                PartChoice c{std::numeric_limits<std::uint64_t>::max(), 0, 0};
                for (int k = wlo; k <= top; ++k) {
                    const std::uint64_t bk = sj[k - wlo] + count * static_cast<std::uint64_t>(k + 1);
                    if (bk < c.bits) c = {bk, k, 0};
                }
                total += pw + with_escape(c, count, sc.lo[j], sc.hi[j]).bits;
            }
            if (total < best) {
                best = total;
                porder_out = lev;
                method_out = method;
            }
        }
        if (lev == 0) break;
        const std::size_t up = nparts / 2;
        sc.s_up.resize(up * K);
        sc.cnt_up.resize(up);
        sc.lo_up.resize(up);
        sc.hi_up.resize(up);
        for (std::size_t j = 0; j < up; ++j) {
            for (std::size_t kk = 0; kk < K; ++kk)
                sc.s_up[j * K + kk] = sc.s[2 * j * K + kk] + sc.s[(2 * j + 1) * K + kk];
            sc.cnt_up[j] = sc.cnt[2 * j] + sc.cnt[2 * j + 1];
            sc.lo_up[j] = std::min(sc.lo[2 * j], sc.lo[2 * j + 1]);
            sc.hi_up[j] = std::max(sc.hi[2 * j], sc.hi[2 * j + 1]);
        }
        sc.s.swap(sc.s_up);
        sc.cnt.swap(sc.cnt_up);
        sc.lo.swap(sc.lo_up);
        sc.hi.swap(sc.hi_up);
        nparts = up;
    }
    bits_out = best;
}

// FIXED predictor residual of order P at sample i (i >= P).
template <int P>
inline std::int32_t fixed_at(const std::int32_t* x, int i) {
    if constexpr (P == 0) return x[i];
    else if constexpr (P == 1) return x[i] - x[i - 1];
    else if constexpr (P == 2) return x[i] - 2 * x[i - 1] + x[i - 2];
    else if constexpr (P == 3) return x[i] - 3 * x[i - 1] + 3 * x[i - 2] - x[i - 3];
    else return x[i] - 4 * x[i - 1] + 6 * x[i - 2] - 4 * x[i - 3] + x[i - 4];
}

template <int P>
std::uint64_t fixed_abs_sum(const std::int32_t* x, int from, int n) {
    std::uint64_t e = 0;
    for (int i = from; i < n; ++i) {
        const std::int64_t v = fixed_at<P>(x, i);
        e += static_cast<std::uint64_t>(v < 0 ? -v : v);
    }
    return e;
}

template <int P>
void fixed_fill(const std::int32_t* x, int n, std::int32_t* r) {
    for (int i = P; i < n; ++i) r[i - P] = fixed_at<P>(x, i);
}

// FIXED predictor residual of order p (x has n samples; residual n - p).
void fixed_residual(const std::int32_t* x, int n, int p, std::vector<std::int32_t>& r) {
    r.resize(static_cast<std::size_t>(n - p));
    switch (p) {
        case 0: fixed_fill<0>(x, n, r.data()); break;
        case 1: fixed_fill<1>(x, n, r.data()); break;
        case 2: fixed_fill<2>(x, n, r.data()); break;
        case 3: fixed_fill<3>(x, n, r.data()); break;
        default: fixed_fill<4>(x, n, r.data()); break;
    }
}

std::uint64_t fixed_abs(const std::int32_t* x, int from, int n, int p) {
    switch (p) {
        case 0: return fixed_abs_sum<0>(x, from, n);
        case 1: return fixed_abs_sum<1>(x, from, n);
        case 2: return fixed_abs_sum<2>(x, from, n);
        case 3: return fixed_abs_sum<3>(x, from, n);
        default: return fixed_abs_sum<4>(x, from, n);
    }
}

Subframe plan_subframe(const std::int32_t* x, int n, int bps, const EncodeOptions& opt, Scratch& sc) {
    Subframe sf;
    bool constant = true;
    for (int i = 1; i < n && constant; ++i) constant = x[i] == x[0];
    if (constant) {
        sf.type = SUB_CONSTANT;
        sf.bits = 8 + static_cast<std::uint64_t>(bps);
        return sf;
    }
    sf.type = SUB_VERBATIM;
    sf.bits = 8 + static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(bps);
    // The order with the least sum |residual| over the samples every order
    // predicts (libFLAC's FLAC__fixed_compute_best_predictor rule).
    const int maxp = std::min(opt.max_fixed_order, n - 1);
    if (maxp < 0) return sf;
    std::uint64_t e[5] = {0, 0, 0, 0, 0};
    if (maxp == 4) {
        // all five orders in one pass, each difference from the last one's
        std::int32_t l0 = x[3], l1 = x[3] - x[2], l2 = l1 - (x[2] - x[1]), l3 = l2 - ((x[2] - x[1]) - (x[1] - x[0]));
        for (int i = 4; i < n; ++i) {
            const std::int32_t d0 = x[i], d1 = d0 - l0, d2 = d1 - l1, d3 = d2 - l2, d4 = d3 - l3;
            l0 = d0;
            l1 = d1;
            l2 = d2;
            l3 = d3;
            e[0] += static_cast<std::uint32_t>(d0 < 0 ? -d0 : d0);
            e[1] += static_cast<std::uint32_t>(d1 < 0 ? -d1 : d1);
            e[2] += static_cast<std::uint32_t>(d2 < 0 ? -d2 : d2);
            e[3] += static_cast<std::uint32_t>(d3 < 0 ? -d3 : d3);
            e[4] += static_cast<std::uint32_t>(d4 < 0 ? -d4 : d4);
        }
    } else {
        for (int p = 0; p <= maxp; ++p) e[p] = fixed_abs(x, maxp, n, p);
    }
    int best_p = 0;
    for (int p = 1; p <= maxp; ++p)
        if (e[p] < e[best_p]) best_p = p;
    fixed_residual(x, n, best_p, sc.r);
    int porder = 0, method = 0;
    std::uint64_t rbits = 0;
    plan_residual(sc, n, best_p, opt.max_partition_order, porder, method, rbits);
    const std::uint64_t fbits = 8 + static_cast<std::uint64_t>(best_p) * static_cast<std::uint64_t>(bps) + rbits;
    if (fbits < sf.bits) {
        sf.type = SUB_FIXED;
        sf.order = best_p;
        sf.porder = porder;
        sf.method = method;
        sf.bits = fbits;
        sf.residual = sc.r;
    }
    return sf;
}

void write_subframe(BitWriter& w, const Subframe& sf, const std::int32_t* x, int n, int bps, Scratch& sc) {
    if (sf.type == SUB_CONSTANT) {
        w.put(0x00, 8);
        w.put_signed(x[0], bps);
        return;
    }
    if (sf.type == SUB_VERBATIM) {
        w.put(0x02, 8);
        for (int i = 0; i < n; ++i) w.put_signed(x[i], bps);
        return;
    }
    w.put(static_cast<std::uint32_t>(0x10 | (sf.order << 1)), 8);
    for (int i = 0; i < sf.order; ++i) w.put_signed(x[i], bps);
    // residual: method, partition order, then each partition
    w.put(static_cast<std::uint32_t>(sf.method), 2);
    w.put(static_cast<std::uint32_t>(sf.porder), 4);
    const int parts = 1 << sf.porder;
    const int s = n >> sf.porder;
    const int pw = sf.method == 0 ? 4 : 5;
    const int esc = sf.method == 0 ? 15 : 31;
    const int kmax_m = sf.method == 0 ? 14 : 30;
    const std::int32_t* r = sf.residual.data();
    sc.u.resize(sf.residual.size());
    for (std::size_t i = 0; i < sf.residual.size(); ++i) sc.u[i] = zigzag(r[i]);
    const std::uint32_t* u = sc.u.data();
    for (int j = 0; j < parts; ++j) {
        const int a = j == 0 ? 0 : j * s - sf.order;
        const int b = (j + 1) * s - sf.order;
        std::int64_t lo = 0, hi = 0;
        for (int i = a; i < b; ++i) {
            lo = std::min<std::int64_t>(lo, r[i]);
            hi = std::max<std::int64_t>(hi, r[i]);
        }
        PartChoice c{0, 0, 0};
        c.k = best_rice(u, a, b, kmax_m, c.bits);
        c = with_escape(c, static_cast<std::uint64_t>(b - a), lo, hi);
        if (c.k < 0) {
            w.put(static_cast<std::uint32_t>(esc), pw);
            w.put(static_cast<std::uint32_t>(c.raw), 5);
            for (int i = a; i < b; ++i) w.put_signed(r[i], c.raw);
        } else {
            w.put(static_cast<std::uint32_t>(c.k), pw);
            for (int i = a; i < b; ++i) w.rice(u[i], c.k);
        }
    }
}

int blocksize_code(int n) {
    switch (n) {
        case 192: return 1;
        case 576: return 2;
        case 1152: return 3;
        case 2304: return 4;
        case 4608: return 5;
        case 256: return 8;
        case 512: return 9;
        case 1024: return 10;
        case 2048: return 11;
        case 4096: return 12;
        case 8192: return 13;
        case 16384: return 14;
        case 32768: return 15;
        default: return n <= 256 ? 6 : 7;
    }
}

int samplerate_code(int fs) {
    switch (fs) {
        case 88200: return 1;
        case 176400: return 2;
        case 192000: return 3;
        case 8000: return 4;
        case 16000: return 5;
        case 22050: return 6;
        case 24000: return 7;
        case 32000: return 8;
        case 44100: return 9;
        case 48000: return 10;
        case 96000: return 11;
        default: break;
    }
    if (fs % 1000 == 0 && fs / 1000 <= 255) return 12;
    if (fs <= 65535) return 13;
    if (fs % 10 == 0 && fs / 10 <= 65535) return 14;
    return 0;
}

void put_utf8(std::vector<std::uint8_t>& out, std::uint64_t v) {
    if (v < 0x80) {
        out.push_back(static_cast<std::uint8_t>(v));
        return;
    }
    int len = v < 0x800 ? 2 : v < 0x10000 ? 3 : v < 0x200000 ? 4 : v < 0x4000000 ? 5 : v < 0x80000000ull ? 6 : 7;
    const std::uint8_t lead_mask = len == 7 ? 0xFE : static_cast<std::uint8_t>(0xFF00 >> len);
    std::uint8_t first = static_cast<std::uint8_t>(lead_mask & 0xFF);
    if (len < 7) first |= static_cast<std::uint8_t>(v >> (6 * (len - 1)));
    out.push_back(first);
    for (int i = len - 2; i >= 0; --i) out.push_back(static_cast<std::uint8_t>(0x80 | ((v >> (6 * i)) & 0x3F)));
}

// Where the encoded bytes go.
class Sink {
public:
    virtual ~Sink() = default;
    virtual void write(const std::uint8_t* p, std::size_t n) = 0;
    virtual void patch(std::uint64_t at, const std::uint8_t* p, std::size_t n) = 0;
    virtual void finish() {}
};

class StringSink : public Sink {
public:
    std::string out;
    void write(const std::uint8_t* p, std::size_t n) override { out.append(reinterpret_cast<const char*>(p), n); }
    void patch(std::uint64_t at, const std::uint8_t* p, std::size_t n) override {
        std::memcpy(out.data() + at, p, n);
    }
};

class FileSink : public Sink {
public:
    explicit FileSink(const std::string& path) : path_(path) {
        std::error_code ec;
        f_ = rawfile::RawFile::open(path, rawfile::Mode::Truncate, ec);
        if (!f_.is_open()) throw rawfile::FileError(ec, path);
        buf_.reserve(1 << 20);
    }
    void write(const std::uint8_t* p, std::size_t n) override {
        buf_.insert(buf_.end(), p, p + n);
        if (buf_.size() >= (1u << 20)) flush();
    }
    void patch(std::uint64_t at, const std::uint8_t* p, std::size_t n) override {
        flush();
        std::error_code ec;
        if (!f_.write_at(at, p, n, ec)) throw rawfile::FileError(ec, path_);
    }
    void finish() override {
        flush();
        f_.close();
    }

private:
    void flush() {
        if (buf_.empty()) return;
        std::error_code ec;
        if (!f_.write_at(at_, buf_.data(), buf_.size(), ec)) throw rawfile::FileError(ec, path_);
        at_ += buf_.size();
        buf_.clear();
    }
    std::string path_;
    rawfile::RawFile f_;
    std::vector<std::uint8_t> buf_;
    std::uint64_t at_ = 0;
};

EncodeStats encode_stream(Sink& sink, int fs, int channels, std::uint64_t total, const Source& source,
                          const EncodeOptions& opt, const std::atomic<bool>* cancel) {
    if (channels != 1 && channels != 2) throw std::invalid_argument("FLAC encoder: 1 or 2 channels");
    if (fs < 1 || fs > 655350) throw std::invalid_argument("FLAC encoder: sample rate out of range");
    if (opt.blocksize < 16 || opt.blocksize > 65535) throw std::invalid_argument("FLAC encoder: block size out of range");
    if (opt.max_fixed_order < 0 || opt.max_fixed_order > 4) throw std::invalid_argument("FLAC encoder: fixed order 0..4");
    if (opt.max_partition_order < 0 || opt.max_partition_order > 15)
        throw std::invalid_argument("FLAC encoder: partition order 0..15");
    if (total >= (std::uint64_t{1} << 36)) throw std::invalid_argument("FLAC encoder: too many samples");
    const int bs = opt.blocksize;
    const std::uint64_t nframes = (total + static_cast<std::uint64_t>(bs) - 1) / static_cast<std::uint64_t>(bs);
    const std::uint64_t npoints =
        (opt.seek_every_frames > 0 && nframes > 0)
            ? (nframes + static_cast<std::uint64_t>(opt.seek_every_frames) - 1) / static_cast<std::uint64_t>(opt.seek_every_frames)
            : 0;
    if (npoints * 18 >= (1u << 24)) throw std::invalid_argument("FLAC encoder: seek table too large");

    // Header: marker, STREAMINFO and SEEKTABLE as placeholders, patched at the end.
    std::vector<std::uint8_t> head{'f', 'L', 'a', 'C'};
    head.push_back(npoints ? 0x00 : 0x80);   // STREAMINFO (last block unless a seek table follows)
    head.push_back(0);
    head.push_back(0);
    head.push_back(34);
    const std::size_t streaminfo_at = head.size();
    head.resize(head.size() + 34, 0);
    std::size_t seek_at = 0;
    if (npoints) {
        const std::uint32_t len = static_cast<std::uint32_t>(npoints * 18);
        head.push_back(0x80 | 3);
        head.push_back(static_cast<std::uint8_t>(len >> 16));
        head.push_back(static_cast<std::uint8_t>(len >> 8));
        head.push_back(static_cast<std::uint8_t>(len));
        seek_at = head.size();
        head.resize(head.size() + npoints * 18, 0);
    }
    sink.write(head.data(), head.size());
    const std::uint64_t first_frame_at = head.size();

    EncodeStats stats;
    util::Md5 md5;
    std::uint32_t min_fs = std::numeric_limits<std::uint32_t>::max(), max_fs = 0;
    std::vector<std::uint8_t> seek(npoints * 18);
    std::vector<std::int16_t> pcm(static_cast<std::size_t>(bs) * static_cast<std::size_t>(channels));
    std::vector<std::int32_t> ch[4];
    for (auto& c : ch) c.resize(static_cast<std::size_t>(bs));
    Scratch sc;
    std::vector<std::uint8_t> frame;
    frame.reserve(static_cast<std::size_t>(bs) * 4 + 64);
    std::uint64_t at = 0;   // bytes of frames written so far
    const int srcode = samplerate_code(fs);

    for (std::uint64_t f = 0; f < nframes; ++f) {
        if (cancel && cancel->load(std::memory_order_relaxed)) throw Cancelled();
        const std::uint64_t first = f * static_cast<std::uint64_t>(bs);
        const int n = static_cast<int>(std::min<std::uint64_t>(static_cast<std::uint64_t>(bs), total - first));
        if (!source(first, static_cast<std::size_t>(n), pcm.data()))
            throw std::runtime_error("FLAC encoder: the source failed");
        md5_pcm16(md5, pcm.data(), static_cast<std::size_t>(n) * static_cast<std::size_t>(channels));
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < channels; ++c)
                ch[c][static_cast<std::size_t>(i)] = pcm[static_cast<std::size_t>(i) * channels + c];

        // channel assignment and subframes
        int assign = channels == 1 ? 0 : 1;   // code: 0 mono, 1 L/R, 8 L/S, 9 R/S, 10 M/S
        Subframe sfs[2];
        const std::int32_t* data[2] = {ch[0].data(), channels == 2 ? ch[1].data() : nullptr};
        int bps[2] = {16, 16};
        if (channels == 1) {
            sfs[0] = plan_subframe(ch[0].data(), n, 16, opt, sc);
        } else {
            Subframe l = plan_subframe(ch[0].data(), n, 16, opt, sc);
            Subframe r = plan_subframe(ch[1].data(), n, 16, opt, sc);
            if (!opt.stereo_decorrelation) {
                sfs[0] = std::move(l);
                sfs[1] = std::move(r);
            } else {
                for (int i = 0; i < n; ++i) {
                    const std::int32_t a = ch[0][static_cast<std::size_t>(i)], b = ch[1][static_cast<std::size_t>(i)];
                    ch[2][static_cast<std::size_t>(i)] = (a + b) >> 1;   // mid (arithmetic shift: floor)
                    ch[3][static_cast<std::size_t>(i)] = a - b;          // side, 17 bits
                }
                Subframe m = plan_subframe(ch[2].data(), n, 16, opt, sc);
                Subframe s = plan_subframe(ch[3].data(), n, 17, opt, sc);
                const std::uint64_t cost[4] = {l.bits + r.bits, l.bits + s.bits, s.bits + r.bits, m.bits + s.bits};
                int best = 0;
                for (int k = 1; k < 4; ++k)
                    if (cost[k] < cost[best]) best = k;
                switch (best) {
                    case 0:
                        sfs[0] = std::move(l);
                        sfs[1] = std::move(r);
                        break;
                    case 1:
                        assign = 8;
                        sfs[0] = std::move(l);
                        sfs[1] = std::move(s);
                        data[1] = ch[3].data();
                        bps[1] = 17;
                        break;
                    case 2:
                        assign = 9;
                        sfs[0] = std::move(s);
                        sfs[1] = std::move(r);
                        data[0] = ch[3].data();
                        bps[0] = 17;
                        break;
                    default:
                        assign = 10;
                        sfs[0] = std::move(m);
                        sfs[1] = std::move(s);
                        data[0] = ch[2].data();
                        data[1] = ch[3].data();
                        bps[1] = 17;
                        break;
                }
                ++stats.stereo[static_cast<std::size_t>(best)];
            }
        }

        // frame header
        frame.clear();
        const int bscode = blocksize_code(n);
        frame.push_back(0xFF);
        frame.push_back(0xF8);   // sync, reserved 0, fixed block size
        frame.push_back(static_cast<std::uint8_t>((bscode << 4) | srcode));
        frame.push_back(static_cast<std::uint8_t>((assign << 4) | (4 << 1)));   // 16 bits per sample
        put_utf8(frame, f);
        if (bscode == 6) {
            frame.push_back(static_cast<std::uint8_t>(n - 1));
        } else if (bscode == 7) {
            frame.push_back(static_cast<std::uint8_t>((n - 1) >> 8));
            frame.push_back(static_cast<std::uint8_t>(n - 1));
        }
        if (srcode == 12) {
            frame.push_back(static_cast<std::uint8_t>(fs / 1000));
        } else if (srcode == 13) {
            frame.push_back(static_cast<std::uint8_t>(fs >> 8));
            frame.push_back(static_cast<std::uint8_t>(fs));
        } else if (srcode == 14) {
            frame.push_back(static_cast<std::uint8_t>((fs / 10) >> 8));
            frame.push_back(static_cast<std::uint8_t>(fs / 10));
        }
        frame.push_back(crc8(frame.data(), frame.size()));
        {
            BitWriter w(frame);
            for (int c = 0; c < channels; ++c) {
                write_subframe(w, sfs[c], data[c], n, bps[c], sc);
                const int t = sfs[c].type == SUB_FIXED ? 2 + sfs[c].order : sfs[c].type;
                ++stats.subframes[static_cast<std::size_t>(t)];
            }
            w.align();
        }
        const std::uint16_t c16 = crc16(frame.data(), frame.size());
        frame.push_back(static_cast<std::uint8_t>(c16 >> 8));
        frame.push_back(static_cast<std::uint8_t>(c16));

        if (npoints && f % static_cast<std::uint64_t>(opt.seek_every_frames) == 0) {
            std::uint8_t* p = seek.data() + (f / static_cast<std::uint64_t>(opt.seek_every_frames)) * 18;
            for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(first >> (56 - 8 * i));
            for (int i = 0; i < 8; ++i) p[8 + i] = static_cast<std::uint8_t>(at >> (56 - 8 * i));
            p[16] = static_cast<std::uint8_t>(n >> 8);
            p[17] = static_cast<std::uint8_t>(n);
        }
        sink.write(frame.data(), frame.size());
        at += frame.size();
        min_fs = std::min(min_fs, static_cast<std::uint32_t>(frame.size()));
        max_fs = std::max(max_fs, static_cast<std::uint32_t>(frame.size()));
        ++stats.frames;
    }
    if (nframes == 0) min_fs = 0;

    // STREAMINFO
    std::uint8_t si[34] = {};
    si[0] = static_cast<std::uint8_t>(bs >> 8);
    si[1] = static_cast<std::uint8_t>(bs);
    si[2] = static_cast<std::uint8_t>(bs >> 8);
    si[3] = static_cast<std::uint8_t>(bs);
    si[4] = static_cast<std::uint8_t>(min_fs >> 16);
    si[5] = static_cast<std::uint8_t>(min_fs >> 8);
    si[6] = static_cast<std::uint8_t>(min_fs);
    si[7] = static_cast<std::uint8_t>(max_fs >> 16);
    si[8] = static_cast<std::uint8_t>(max_fs >> 8);
    si[9] = static_cast<std::uint8_t>(max_fs);
    // sample rate (20) | channels-1 (3) | bps-1 (5) | total samples (36) = 64 bits
    const std::uint64_t packed = (static_cast<std::uint64_t>(fs) << 44) |
                                 (static_cast<std::uint64_t>(channels - 1) << 41) | (static_cast<std::uint64_t>(15) << 36) |
                                 total;
    for (int i = 0; i < 8; ++i) si[10 + i] = static_cast<std::uint8_t>(packed >> (56 - 8 * i));
    stats.md5 = md5.digest();
    std::memcpy(si + 18, stats.md5.data(), 16);
    sink.patch(streaminfo_at, si, sizeof si);
    if (npoints) sink.patch(seek_at, seek.data(), seek.size());
    sink.finish();
    stats.bytes = first_frame_at + at;
    return stats;
}

}  // namespace

EncodeStats encode_file(const std::string& path, int sample_rate, int channels, std::uint64_t total_frames,
                        const Source& source, const EncodeOptions& opt, const std::atomic<bool>* cancel) {
    FileSink sink(path);
    return encode_stream(sink, sample_rate, channels, total_frames, source, opt, cancel);
}

std::string encode(std::span<const std::int16_t> interleaved, int channels, int sample_rate,
                   const EncodeOptions& opt, EncodeStats* stats) {
    if (channels < 1 || interleaved.size() % static_cast<std::size_t>(channels) != 0)
        throw std::invalid_argument("FLAC encoder: samples are not whole frames");
    const std::uint64_t total = interleaved.size() / static_cast<std::size_t>(channels);
    StringSink sink;
    const Source src = [&](std::uint64_t first, std::size_t n, std::int16_t* out) {
        std::memcpy(out, interleaved.data() + first * static_cast<std::uint64_t>(channels),
                    n * static_cast<std::size_t>(channels) * sizeof(std::int16_t));
        return true;
    };
    EncodeStats s = encode_stream(sink, sample_rate, channels, total, src, opt, nullptr);
    if (stats) *stats = s;
    return std::move(sink.out);
}

// --- reader ----------------------------------------------------------------------

namespace {

// A byte source for dr_flac (a file through positioned reads, or memory),
// and where the STREAMINFO's MD5 lands: dr_flac hands every callback the
// same user pointer.
struct Stream {
    rawfile::RawFile file;
    const std::string* mem = nullptr;
    std::uint64_t pos = 0;
    std::uint64_t size = 0;
    std::array<std::uint8_t, 16> md5{};
};

size_t on_read(void* user, void* buf, size_t n) {
    auto* s = static_cast<Stream*>(user);
    if (s->pos >= s->size) return 0;
    const std::size_t want = static_cast<std::size_t>(std::min<std::uint64_t>(n, s->size - s->pos));
    std::size_t got;
    if (s->mem) {
        std::memcpy(buf, s->mem->data() + s->pos, want);
        got = want;
    } else {
        std::error_code ec;
        got = s->file.read_at(s->pos, buf, want, ec);
        if (ec) return 0;
    }
    s->pos += got;
    return got;
}

drflac_bool32 on_seek(void* user, int offset, drflac_seek_origin origin) {
    auto* s = static_cast<Stream*>(user);
    std::int64_t base = 0;
    if (origin == DRFLAC_SEEK_CUR) base = static_cast<std::int64_t>(s->pos);
    else if (origin == DRFLAC_SEEK_END) base = static_cast<std::int64_t>(s->size);
    const std::int64_t to = base + offset;
    if (to < 0 || static_cast<std::uint64_t>(to) > s->size) return DRFLAC_FALSE;
    s->pos = static_cast<std::uint64_t>(to);
    return DRFLAC_TRUE;
}

drflac_bool32 on_tell(void* user, drflac_int64* cursor) {
    *cursor = static_cast<drflac_int64>(static_cast<Stream*>(user)->pos);
    return DRFLAC_TRUE;
}

void on_meta(void* user, drflac_metadata* m) {
    if (m->type == DRFLAC_METADATA_BLOCK_TYPE_STREAMINFO)
        std::memcpy(static_cast<Stream*>(user)->md5.data(), m->data.streaminfo.md5, 16);
}

// The PCM frame dr_flac's next read starts at, from its current frame's
// header: what makes a frame skipped for a bad CRC visible (dr_flac skips
// it and goes on, which would shift every later sample by a block).
std::uint64_t decoder_position(const drflac* d) {
    std::uint64_t first = d->currentFLACFrame.header.pcmFrameNumber;
    if (first == 0)
        first = static_cast<std::uint64_t>(d->currentFLACFrame.header.flacFrameNumber) * d->maxBlockSizeInPCMFrames;
    return first + d->currentFLACFrame.header.blockSizeInPCMFrames - d->currentFLACFrame.pcmFramesRemaining;
}

}  // namespace

struct Reader::Impl {
    Stream stream;
    drflac* dec = nullptr;
    Info info;
    std::string what;

    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    ~Impl() {
        if (dec) drflac_close(dec);
    }

    void open() {
        dec = drflac_open_with_metadata(on_read, on_seek, on_tell, on_meta, &stream, nullptr);
        if (!dec) throw std::runtime_error(what + ": not a FLAC file (or a damaged one)");
        info.channels = dec->channels;
        info.sample_rate = static_cast<int>(dec->sampleRate);
        info.bits_per_sample = dec->bitsPerSample;
        info.frames = dec->totalPCMFrameCount;
        info.md5 = stream.md5;
        if (info.bits_per_sample > 16)
            throw std::runtime_error(what + ": " + std::to_string(info.bits_per_sample) +
                                     "-bit FLAC; the passband store holds 16-bit samples");
    }

    std::vector<std::int16_t> read(std::uint64_t first, std::uint64_t n) {
        std::vector<std::int16_t> out;
        if (first >= info.frames || n == 0) return out;
        n = std::min(n, info.frames - first);
        if (!drflac_seek_to_pcm_frame(dec, first))
            throw std::runtime_error(what + ": cannot seek to sample " + std::to_string(first));
        const std::size_t nch = static_cast<std::size_t>(info.channels);
        out.resize(static_cast<std::size_t>(n) * nch);
        std::uint64_t done = 0;
        while (done < n) {
            // never past the end of one FLAC frame per call, so a skipped frame shows
            std::uint64_t chunk = n - done;
            if (dec->currentFLACFrame.pcmFramesRemaining > 0)
                chunk = std::min<std::uint64_t>(chunk, dec->currentFLACFrame.pcmFramesRemaining);
            else if (dec->maxBlockSizeInPCMFrames > 0)
                chunk = std::min<std::uint64_t>(chunk, dec->maxBlockSizeInPCMFrames);
            const drflac_uint64 got =
                drflac_read_pcm_frames_s16(dec, chunk, out.data() + static_cast<std::size_t>(done) * nch);
            if (got == 0) break;
            done += got;
            if (decoder_position(dec) != first + done)
                throw std::runtime_error(what + ": a damaged frame (CRC mismatch) near sample " +
                                         std::to_string(first + done - got));
        }
        out.resize(static_cast<std::size_t>(done) * nch);
        return out;
    }
};

Reader::Reader(const std::string& path) : impl_(std::make_unique<Impl>()) {
    impl_->what = path;
    std::error_code ec;
    impl_->stream.file = rawfile::RawFile::open(path, rawfile::Mode::Read, ec);
    if (!impl_->stream.file.is_open()) throw rawfile::FileError(ec, path);
    const auto sz = impl_->stream.file.size(ec);
    if (!sz) throw rawfile::FileError(ec, path);
    impl_->stream.size = *sz;
    impl_->open();
}

Reader::~Reader() = default;

const Info& Reader::info() const { return impl_->info; }

std::vector<std::int16_t> Reader::read(std::uint64_t first, std::uint64_t n) { return impl_->read(first, n); }

std::vector<std::int16_t> decode(const std::string& bytes, Info* info) {
    Reader::Impl impl;
    impl.what = "FLAC data";
    impl.stream.mem = &bytes;
    impl.stream.size = bytes.size();
    impl.open();
    if (info) *info = impl.info;
    return impl.read(0, impl.info.frames);
}

}  // namespace sstvae::qrss::flac
