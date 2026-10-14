// FLAC encoder (ours) and reader (dr_flac) for the passband store; see
// flac.hpp for what is written and why.

#include "qrss/flac.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
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

constexpr int SUB_CONSTANT = 0, SUB_VERBATIM = 1, SUB_FIXED = 2, SUB_LPC = 3;
constexpr int MAX_LPC_ORDER = 32;

// One residual partition's coding: Rice parameter k, or k = -1 for an
// escape with `raw`-bit samples.
struct PartParam {
    int k = 0;
    int raw = 0;
};

// One channel's subframe, chosen and costed; `write_subframe` writes it.
struct Subframe {
    int type = SUB_VERBATIM;
    int order = 0;
    int porder = 0;
    int method = 0;           // 0: 4-bit parameters, 1: 5-bit
    std::uint64_t bits = 0;   // the whole subframe
    std::vector<std::int32_t> residual;
    std::vector<PartParam> parts;   // per partition, as costed
    int precision = 0;        // LPC: bits per coefficient as written (1..15)
    int shift = 0;            // LPC: the prediction's right shift (0..15)
    std::array<std::int32_t, MAX_LPC_ORDER> coef{};   // LPC: coef[j] multiplies x[i - 1 - j]
};

// An apodization window over a block of n samples, zero outside [lo, hi).
struct Window {
    std::vector<double> w;
    int lo = 0, hi = 0;
    double energy = 0.0;   // sum of w^2: what scales one window's errors to another's
};

void finish_window(Window& win) {
    const int n = static_cast<int>(win.w.size());
    win.lo = 0;
    while (win.lo < n && win.w[static_cast<std::size_t>(win.lo)] == 0.0) ++win.lo;
    win.hi = n;
    while (win.hi > win.lo && win.w[static_cast<std::size_t>(win.hi - 1)] == 0.0) --win.hi;
    win.energy = 0.0;
    for (int i = win.lo; i < win.hi; ++i) win.energy += win.w[static_cast<std::size_t>(i)] * win.w[static_cast<std::size_t>(i)];
}

// libFLAC's tukey(p) window (FLAC__window_tukey): flat, with Hann tapers
// over p/2 of the block at each end.
Window tukey_window(int n, double p) {
    constexpr double PI = 3.14159265358979323846;
    Window win;
    win.w.assign(static_cast<std::size_t>(n), 1.0);
    const int np = static_cast<int>(p / 2.0 * n) - 1;
    if (np > 0) {
        for (int i = 0; i <= np; ++i) {
            win.w[static_cast<std::size_t>(i)] = 0.5 - 0.5 * std::cos(PI * i / np);
            win.w[static_cast<std::size_t>(n - np - 1 + i)] = 0.5 - 0.5 * std::cos(PI * (i + np) / np);
        }
    }
    finish_window(win);
    return win;
}

// libFLAC's partial tukey window (FLAC__window_partial_tukey): tukey(p)
// over the fraction [start, end) of the block, zero elsewhere.
Window partial_tukey_window(int n, double p, double start, double end) {
    constexpr double PI = 3.14159265358979323846;
    Window win;
    win.w.assign(static_cast<std::size_t>(n), 0.0);
    const int start_n = static_cast<int>(start * n), end_n = static_cast<int>(end * n);
    const int np = static_cast<int>(p / 2.0 * (end_n - start_n));
    int i = start_n;
    for (int k = 1; i < start_n + np && i < n; ++i, ++k) win.w[static_cast<std::size_t>(i)] = 0.5 - 0.5 * std::cos(PI * k / np);
    for (; i < end_n - np && i < n; ++i) win.w[static_cast<std::size_t>(i)] = 1.0;
    for (int k = np; i < end_n && i < n; ++i, --k) win.w[static_cast<std::size_t>(i)] = 0.5 - 0.5 * std::cos(PI * k / np);
    finish_window(win);
    return win;
}

// The windows a block is analyzed through, in the order ties go: libFLAC
// -5's tukey(0.5), then, with `partial`, the two of libFLAC's
// partial_tukey(2) (its defaults: overlap 0.1, tukey(0.2) parts), which
// fit a block whose two halves differ -- a signal starting or stopping
// in it.
void make_windows(std::vector<Window>& ws, int n, bool partial) {
    ws.clear();
    ws.push_back(tukey_window(n, 0.5));
    if (partial) {
        constexpr int PARTS = 2;
        constexpr double OVERLAP = 0.1, P = 0.2;
        const double units = 1.0 / (1.0 - OVERLAP) - 1.0;
        for (int m = 0; m < PARTS; ++m)
            ws.push_back(partial_tukey_window(n, P, m / (PARTS + units), (m + 1 + units) / (PARTS + units)));
    }
}

// Buffers one stream reuses for every subframe it plans and writes.
struct Scratch {
    std::vector<std::int32_t> r;           // a residual
    std::vector<std::int32_t> r2;          // another (the best LPC residual so far)
    std::vector<std::uint32_t> u;          // its zigzag
    std::vector<std::uint64_t> s, s_up;    // [partition][k - wlo]: sum(u >> k)
    std::vector<std::uint64_t> cnt, cnt_up;
    std::vector<std::int64_t> lo, hi, lo_up, hi_up;
    std::vector<PartParam> parts, cur;     // the best partition parameters so far, those being costed
    std::vector<Window> windows;           // the apodizations for blocks of `window_n` samples
    int window_n = -1;
    std::vector<double> wx;                // the block, windowed
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
// residual in sc.r of a block of n samples, predictor order p; the
// residual section's bits (method and order fields included) go in `bits`
// and each partition's parameter in sc.parts.
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
            sc.cur.resize(nparts);
            for (std::size_t j = 0; j < nparts; ++j) {
                const std::uint64_t* sj = sc.s.data() + j * K;
                const std::uint64_t count = sc.cnt[j];
                PartChoice c{std::numeric_limits<std::uint64_t>::max(), 0, 0};
                for (int k = wlo; k <= top; ++k) {
                    const std::uint64_t bk = sj[k - wlo] + count * static_cast<std::uint64_t>(k + 1);
                    if (bk < c.bits) c = {bk, k, 0};
                }
                c = with_escape(c, count, sc.lo[j], sc.hi[j]);
                total += pw + c.bits;
                sc.cur[j] = {c.k, c.raw};
            }
            if (total < best) {
                best = total;
                porder_out = lev;
                method_out = method;
                sc.parts.swap(sc.cur);
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

// --- LPC ---------------------------------------------------------------------------
//
// As libFLAC does it (lpc.c, stream_encoder.c): the block through each
// apodization window, its autocorrelation, Levinson-Durbin for the
// predictors of every order with their errors, the orders ranked by the
// bits the error predicts, coefficients quantized with error feedback to a
// precision set by the block size (clamped so that every prediction sum
// fits 32 bits), and the residual then costed exactly. What differs:
// - the predictors of every window are ranked together, each window's
//   error scaled by its energy against the whole-block window's, where
//   libFLAC ranks the orders within a window and costs each window's best;
// - the best few by that rank (lpc_orders_tried) have their residual
//   computed, and the one whose residual's own estimate is smallest is
//   costed exactly -- the Levinson error ranks orders worse than the
//   residual does;
// - the Rice parameters are exact (above), and the precision field is the
//   fewest bits the quantized coefficients need.

// autoc[l] = sum over i of wx[i] wx[i - l], for l = 0..lags-1.
void autocorrelation(const double* wx, int n, int lags, double* autoc) {
    for (int l = 0; l < lags; ++l) {
        double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        const double* y = wx - l;
        int i = l;
        for (; i + 4 <= n; i += 4) {
            a0 += wx[i] * y[i];
            a1 += wx[i + 1] * y[i + 1];
            a2 += wx[i + 2] * y[i + 2];
            a3 += wx[i + 3] * y[i + 3];
        }
        for (; i < n; ++i) a0 += wx[i] * y[i];
        autoc[l] = (a0 + a1) + (a2 + a3);
    }
}

// Levinson-Durbin (FLAC__lpc_compute_lp_coefficients): lp[o][j], the
// predictor of order o + 1 (x[i] ~ sum_j lp[o][j] x[i - 1 - j]), and
// err[o], its error, for o < the returned count (max_order, or fewer when
// the error reaches zero or stops being a non-negative number).
int levinson(const double* autoc, int max_order, double (*lp)[MAX_LPC_ORDER], double* err) {
    double a[MAX_LPC_ORDER];
    double e = autoc[0];
    for (int i = 0; i < max_order; ++i) {
        double r = -autoc[i + 1];
        for (int j = 0; j < i; ++j) r -= a[j] * autoc[i - j];
        r /= e;
        a[i] = r;
        int j = 0;
        for (; j < (i >> 1); ++j) {
            const double t = a[j];
            a[j] += r * a[i - 1 - j];
            a[i - 1 - j] += r * t;
        }
        if (i & 1) a[j] += a[j] * r;
        e *= (1.0 - r * r);
        if (!(e >= 0.0) || !std::isfinite(r)) return i;   // (rounding made it unstable: stop before it)
        for (j = 0; j <= i; ++j) lp[i][j] = -a[j];
        err[i] = e;
        if (e == 0.0) return i + 1;
    }
    return max_order;
}

// Bits per residual sample the error predicts
// (FLAC__lpc_compute_expected_bits_per_residual_sample).
double expected_bits(double err, int n) {
    if (!(err > 0.0)) return 0.0;
    const double b = 0.5 * std::log2(0.5 / n * err);
    return b > 0.0 ? b : 0.0;
}

// The precision libFLAC gives 16-bit coefficients for a block size
// (stream_encoder.c, qlp_coeff_precision 0).
int auto_precision(int blocksize) {
    if (blocksize <= 192) return 7;
    if (blocksize <= 384) return 8;
    if (blocksize <= 576) return 9;
    if (blocksize <= 1152) return 10;
    if (blocksize <= 2304) return 11;
    if (blocksize <= 4608) return 12;
    return 13;
}

// FLAC__lpc_quantize_coefficients: q[0..order) at `precision` bits with
// error feedback, and the shift (0..15). False when nothing quantizes
// (all-zero or non-finite coefficients, or a shift below -16).
bool quantize_lpc(const double* lp, int order, int precision, std::int32_t* q, int& shift) {
    const int p = precision - 1;
    const std::int32_t qmax = (std::int32_t{1} << p) - 1, qmin = -(std::int32_t{1} << p);
    double cmax = 0.0;
    for (int i = 0; i < order; ++i) {
        if (!std::isfinite(lp[i])) return false;
        cmax = std::max(cmax, std::fabs(lp[i]));
    }
    if (!(cmax > 0.0)) return false;
    int e = 0;
    (void)std::frexp(cmax, &e);
    shift = p - (e - 1) - 1;
    if (shift > 15) shift = 15;
    else if (shift < -16) return false;
    // (a negative shift is not allowed in a stream: the coefficients are
    // scaled down instead and the shift written as 0)
    const double scale = std::ldexp(1.0, shift);
    double err = 0.0;
    for (int i = 0; i < order; ++i) {
        err += lp[i] * scale;
        long v = std::lround(err);
        if (v > qmax) v = qmax;
        else if (v < qmin) v = qmin;
        err -= static_cast<double>(v);
        q[i] = static_cast<std::int32_t>(v);
    }
    if (shift < 0) shift = 0;
    return true;
}

int ilog2(unsigned v) { return static_cast<int>(std::bit_width(v)) - 1; }

// Residual of the predictor (q, shift) of `order` on x[0, n) into r
// (n - order samples), with sum |r| into abs_sum. `xs` holds the samples
// the sums read -- x itself, or the same values as int16 for a 16-bit
// channel, which is what lets a baseline (SSE2) build vectorize the
// multiplies -- and `qs` the coefficients in that type. The sums fit 32
// bits by construction: bps + precision + ilog2(order) <= 32, the bound
// libFLAC keeps so that decoders may predict in 32 bits, and the caller's
// precision clamp. False when a residual sample would not fit 31 bits (an
// escape's widest field, and far past any useful predictor).
template <typename T>
bool lpc_residual_t(const std::int32_t* x, const T* xs, int n, const T* qs, int order, int shift,
                    std::vector<std::int32_t>& r, std::uint64_t& abs_sum) {
    const int m = n - order;
    r.resize(static_cast<std::size_t>(m));
    std::int32_t* out = r.data();
    const std::int32_t* target = x + order;
    constexpr std::int64_t LIM = (std::int64_t{1} << 30) - 1;
    // While every prediction is within +-2^27 the residual (a sample of at
    // most 17 bits less it) and eight of their magnitudes fit 32 bits; a
    // block with a larger prediction is formed again in 64.
    constexpr std::int32_t PMAX = std::int32_t{1} << 27;
    std::uint64_t e = 0;
    int i = 0;
    // Eight samples at a time, each step a statement of its own over the
    // eight: the shape a compiler's straight-line vectorizer takes.
    for (; i + 8 <= m; i += 8) {
        std::int32_t a[8] = {};
        for (int j = 0; j < order; ++j) {
            const std::int32_t c = qs[j];
            const T* xj = xs + i + order - 1 - j;
            for (int t = 0; t < 8; ++t) a[t] += c * xj[t];
        }
        std::int32_t p[8], v[8], far[8];
        std::uint32_t av[8];
        for (int t = 0; t < 8; ++t) p[t] = a[t] >> shift;
        for (int t = 0; t < 8; ++t) far[t] = (p[t] > PMAX) | (p[t] < -PMAX);
        for (int t = 0; t < 8; ++t)
            v[t] = static_cast<std::int32_t>(static_cast<std::uint32_t>(target[i + t]) - static_cast<std::uint32_t>(p[t]));
        for (int t = 0; t < 8; ++t) out[i + t] = v[t];
        for (int t = 0; t < 8; ++t) av[t] = v[t] < 0 ? 0u - static_cast<std::uint32_t>(v[t]) : static_cast<std::uint32_t>(v[t]);
        std::int32_t big = 0;
        for (int t = 0; t < 8; ++t) big |= far[t];
        std::uint32_t eb = 0;
        for (int t = 0; t < 8; ++t) eb += av[t];
        if (!big) {
            e += eb;
        } else {
            for (int t = 0; t < 8; ++t) {
                const std::int64_t v = static_cast<std::int64_t>(target[i + t]) - (a[t] >> shift);
                if (v < -LIM || v > LIM) return false;
                out[i + t] = static_cast<std::int32_t>(v);
                e += static_cast<std::uint64_t>(v < 0 ? -v : v);
            }
        }
    }
    for (; i < m; ++i) {
        std::int32_t a = 0;
        for (int j = 0; j < order; ++j) a += static_cast<std::int32_t>(qs[j]) * xs[i + order - 1 - j];
        const std::int64_t v = static_cast<std::int64_t>(target[i]) - (a >> shift);
        if (v < -LIM || v > LIM) return false;
        out[i] = static_cast<std::int32_t>(v);
        e += static_cast<std::uint64_t>(v < 0 ? -v : v);
    }
    abs_sum = e;
    return true;
}

}  // namespace

// lpc_residual_t through x16 (the channel as int16) when there is one.
bool detail::lpc_residual(const std::int32_t* x, const std::int16_t* x16, int n, const std::int32_t* q, int order,
                          int shift, std::vector<std::int32_t>& r, std::uint64_t& abs_sum) {
    if (x16) {
        std::int16_t q16[MAX_LPC_ORDER];
        for (int j = 0; j < order; ++j) q16[j] = static_cast<std::int16_t>(q[j]);   // (precision <= 15 bits)
        return lpc_residual_t<std::int16_t>(x, x16, n, q16, order, shift, r, abs_sum);
    }
    return lpc_residual_t<std::int32_t>(x, x, n, q, order, shift, r, abs_sum);
}

namespace {

// Bits a residual of `count` samples with sum |r| = abs_sum is estimated
// to take as one Rice partition (the parameter from the mean, as
// best_rice starts).
double rice_estimate(std::uint64_t abs_sum, std::uint64_t count) {
    if (count == 0) return 0.0;
    const std::uint64_t u = 2 * abs_sum;   // (the zigzag of r is about 2|r|)
    const int k = std::max(bit_length(u / count) - 1, 0);
    return static_cast<double>(count) * (k + 1) + static_cast<double>(u >> k);
}

// What one look at a channel finds before anything is costed exactly:
// whether it is constant, its FIXED order (the least sum |residual|,
// libFLAC's FLAC__fixed_compute_best_predictor rule) when there is no LPC,
// its LPC predictors of every order through each window, and the subframe
// bits the best of that is estimated to take. The estimates choose the
// channel assignment, so only the two channels coded are costed exactly;
// and the assignment is chosen on the first window alone, the others
// being analyzed only for the channels coded (plan_lpc).
struct FixedLook {
    int order = -1;     // -1: no FIXED order fits the block
    double est = 0.0;   // estimated subframe bits
};

constexpr int MAX_WINDOWS = 3;

struct Analysis {
    bool constant = false;
    FixedLook fixed;                                    // when there is no LPC
    double est = 0.0;                                   // estimated subframe bits
    int windows = 0;                                    // windows analyzed
    int got[MAX_WINDOWS] = {};                          // through window w, predictors of orders 1..got[w]
    double err[MAX_WINDOWS][MAX_LPC_ORDER];             // err[w][o - 1]: the order-o predictor's error
    double lp[MAX_WINDOWS][MAX_LPC_ORDER][MAX_LPC_ORDER];   // lp[w][o - 1]: the order-o predictor
};

FixedLook fixed_look(const std::int32_t* x, int n, int bps, const EncodeOptions& opt) {
    FixedLook f;
    const int maxp = std::min(opt.max_fixed_order, n - 1);
    if (maxp < 0) return f;
    std::uint64_t e[5] = {0, 0, 0, 0, 0};
    if (maxp == 4) {
        // All five orders in one pass, eight samples at a time, each step a
        // statement over the eight (what a straight-line vectorizer takes).
        // The magnitudes collect in 32-bit lanes, folded into e every 1024
        // blocks: a 17-bit channel's fourth difference is under 2^21.
        std::uint32_t lane[5][8] = {};
        auto fold = [&] {
            for (int o = 0; o < 5; ++o)
                for (int t = 0; t < 8; ++t) {
                    e[o] += lane[o][t];
                    lane[o][t] = 0;
                }
        };
        int i = 4, blocks = 0;
        for (; i + 8 <= n; i += 8) {
            std::int32_t d[5][8];
            const std::int32_t* y = x + i;
            for (int t = 0; t < 8; ++t) d[0][t] = y[t];
            for (int t = 0; t < 8; ++t) d[1][t] = y[t] - y[t - 1];
            for (int t = 0; t < 8; ++t) d[2][t] = y[t] - 2 * y[t - 1] + y[t - 2];
            for (int t = 0; t < 8; ++t) d[3][t] = y[t] - 3 * y[t - 1] + 3 * y[t - 2] - y[t - 3];
            for (int t = 0; t < 8; ++t) d[4][t] = y[t] - 4 * y[t - 1] + 6 * y[t - 2] - 4 * y[t - 3] + y[t - 4];
            for (int o = 0; o < 5; ++o)
                for (int t = 0; t < 8; ++t) lane[o][t] += static_cast<std::uint32_t>(d[o][t] < 0 ? -d[o][t] : d[o][t]);
            if (++blocks == 1024) {
                fold();
                blocks = 0;
            }
        }
        fold();
        for (; i < n; ++i) {
            e[0] += static_cast<std::uint32_t>(std::abs(fixed_at<0>(x, i)));
            e[1] += static_cast<std::uint32_t>(std::abs(fixed_at<1>(x, i)));
            e[2] += static_cast<std::uint32_t>(std::abs(fixed_at<2>(x, i)));
            e[3] += static_cast<std::uint32_t>(std::abs(fixed_at<3>(x, i)));
            e[4] += static_cast<std::uint32_t>(std::abs(fixed_at<4>(x, i)));
        }
    } else {
        for (int p = 0; p <= maxp; ++p) e[p] = fixed_abs(x, maxp, n, p);
    }
    int best_p = 0;
    for (int p = 1; p <= maxp; ++p)
        if (e[p] < e[best_p]) best_p = p;
    f.order = best_p;
    // (e counts the samples from maxp on; the estimate scales it to the residual's n - best_p)
    const std::uint64_t counted = static_cast<std::uint64_t>(n - maxp);
    f.est = 8.0 + static_cast<double>(best_p) * bps + 6.0 +
            (counted ? rice_estimate(e[best_p], counted) * (n - best_p) / static_cast<double>(counted) : 0.0);
    return f;
}

// The windows for blocks of n samples, built once per block size.
const std::vector<Window>& windows_for(Scratch& sc, int n, const EncodeOptions& opt) {
    if (sc.window_n != n) {
        make_windows(sc.windows, n, opt.partial_tukey);
        sc.window_n = n;
    }
    return sc.windows;
}

// x through window k of a's windows: its autocorrelation over the window's
// support and the predictors Levinson-Durbin finds in it.
void analyze_window(const std::int32_t* x, int max_order, const Window& win, Scratch& sc, Analysis& a, int k) {
    a.got[k] = 0;
    const int len = win.hi - win.lo;
    if (len <= max_order) return;
    sc.wx.resize(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i)
        sc.wx[static_cast<std::size_t>(i)] = x[win.lo + i] * win.w[static_cast<std::size_t>(win.lo + i)];
    double autoc[MAX_LPC_ORDER + 1] = {};
    autocorrelation(sc.wx.data(), len, max_order + 1, autoc);
    if (!(autoc[0] > 0.0) || !std::isfinite(autoc[0])) return;
    a.got[k] = levinson(autoc, max_order, a.lp[k], a.err[k]);
}

// libFLAC's estimate of an order-o subframe's bits through window k
// (FLAC__lpc_compute_best_order's): the error's expected bits per sample
// for the n - o residual samples, plus the warm-up and coefficients. The
// error is scaled by the window's energy against the first window's, so
// that windows covering part of the block compare with it fairly (the
// first window's estimates are libFLAC's own).
double order_estimate(const Analysis& a, const std::vector<Window>& ws, int k, int o, int n, int bps, int precision) {
    const double scale = ws[static_cast<std::size_t>(k)].energy > 0.0 ? ws[0].energy / ws[static_cast<std::size_t>(k)].energy : 1.0;
    return expected_bits(a.err[k][o - 1] * scale, n) * (n - o) + static_cast<double>(o) * (bps + precision);
}

int base_precision(const EncodeOptions& opt) {
    return opt.qlp_precision > 0 ? opt.qlp_precision : auto_precision(opt.blocksize);
}

void analyze_lpc(const std::int32_t* x, int n, int bps, const EncodeOptions& opt, Scratch& sc, Analysis& a) {
    const int max_order = std::min({opt.max_lpc_order, n - 1, MAX_LPC_ORDER});
    if (max_order < 1) return;
    const std::vector<Window>& ws = windows_for(sc, n, opt);
    analyze_window(x, max_order, ws[0], sc, a, 0);
    a.windows = 1;
    if (a.got[0] < 1) return;
    double best = std::numeric_limits<double>::infinity();
    for (int o = 1; o <= a.got[0]; ++o) best = std::min(best, order_estimate(a, ws, 0, o, n, bps, base_precision(opt)));
    // (8 header bits, 4 + 5 for the precision and shift, 6 for the residual's method and order)
    a.est = std::min(a.est, best + 23.0);
}

Analysis& analyze(const std::int32_t* x, int n, int bps, const EncodeOptions& opt, Scratch& sc, Analysis& a) {
    a.constant = false;
    a.fixed = FixedLook{};
    a.windows = 0;
    a.got[0] = 0;
    bool constant = true;
    for (int i = 1; i < n && constant; ++i) constant = x[i] == x[0];
    if (constant) {
        a.constant = true;
        a.est = 8.0 + bps;
        return a;
    }
    a.est = 8.0 + static_cast<double>(n) * bps;   // VERBATIM
    if (opt.max_lpc_order > 0) analyze_lpc(x, n, bps, opt, sc, a);
    if (a.got[0] == 0) {
        // (with LPC, FIXED is looked at only for the channels coded: plan_subframe)
        a.fixed = fixed_look(x, n, bps, opt);
        if (a.fixed.order >= 0) a.est = std::min(a.est, a.fixed.est);
    }
    return a;
}

// The best LPC subframe for x, into `best` when smaller than its `bits`.
// The remaining windows are analyzed, every (window, order) ranked by its
// estimate, and the first `lpc_orders_tried` quantized and their residuals
// computed; the one whose residual estimates smallest (one Rice partition
// from the mean |residual|, which tracks the exact size far better than
// the Levinson error does) is costed exactly. Returns that estimate
// (infinity when no predictor quantizes).
double plan_lpc(const std::int32_t* x, const std::int16_t* x16, int n, int bps, const EncodeOptions& opt, Scratch& sc,
                Analysis& a, Subframe& best) {
    const int base = base_precision(opt);
    const int max_order = std::min({opt.max_lpc_order, n - 1, MAX_LPC_ORDER});
    const std::vector<Window>& ws = windows_for(sc, n, opt);
    for (int k = a.windows; k < static_cast<int>(ws.size()); ++k)
        analyze_window(x, max_order, ws[static_cast<std::size_t>(k)], sc, a, k);
    a.windows = static_cast<int>(ws.size());
    struct Candidate {
        double est;
        int window, order;
    };
    std::array<Candidate, MAX_WINDOWS * MAX_LPC_ORDER> cand;
    int nc = 0;
    for (int k = 0; k < a.windows; ++k)
        for (int o = 1; o <= a.got[k]; ++o)
            cand[static_cast<std::size_t>(nc++)] = {order_estimate(a, ws, k, o, n, bps, base), k, o};
    std::stable_sort(cand.begin(), cand.begin() + nc, [](const Candidate& p, const Candidate& q) { return p.est < q.est; });
    const int tries = std::min(std::max(opt.lpc_orders_tried, 1), nc);
    std::array<std::int32_t, MAX_LPC_ORDER> q{}, pick_q{};
    double pick_est = std::numeric_limits<double>::infinity();
    int pick_order = 0, pick_shift = 0, pick_need = 0;
    for (int t = 0; t < tries; ++t) {
        const Candidate& c = cand[static_cast<std::size_t>(t)];
        const int order = c.order;
        const int precision = std::min(base, 32 - bps - ilog2(static_cast<unsigned>(order)));
        if (precision < 1) continue;
        int shift = 0;
        if (!quantize_lpc(a.lp[c.window][order - 1], order, precision, q.data(), shift)) continue;
        // the fewest bits that hold every coefficient
        int need = 0;
        for (int j = 0; j < order; ++j) {
            const std::int32_t v = q[static_cast<std::size_t>(j)];
            need = std::max(need, signed_bits(std::min(v, 0), std::max(v, 0)));
        }
        if (need == 0) continue;   // (a zero predictor: FIXED order 0 says it in fewer bits)
        std::uint64_t abs_sum = 0;
        if (!detail::lpc_residual(x, x16, n, q.data(), order, shift, sc.r, abs_sum)) continue;
        const double est = 8.0 + static_cast<double>(order) * (bps + need) + 4 + 5 + 6 +
                           rice_estimate(abs_sum, static_cast<std::uint64_t>(n - order));
        if (est < pick_est) {
            pick_est = est;
            pick_order = order;
            pick_shift = shift;
            pick_need = need;
            pick_q = q;
            sc.r2.swap(sc.r);
        }
    }
    if (pick_order == 0) return pick_est;
    sc.r.swap(sc.r2);
    int porder = 0, method = 0;
    std::uint64_t rbits = 0;
    plan_residual(sc, n, pick_order, opt.max_partition_order, porder, method, rbits);
    const std::uint64_t bits =
        8 + static_cast<std::uint64_t>(pick_order) * static_cast<std::uint64_t>(bps + pick_need) + 4 + 5 + rbits;
    if (bits < best.bits) {
        best.type = SUB_LPC;
        best.order = pick_order;
        best.porder = porder;
        best.method = method;
        best.bits = bits;
        best.residual.swap(sc.r);
        best.parts.swap(sc.parts);
        best.precision = pick_need;
        best.shift = pick_shift;
        best.coef = pick_q;
    }
    return pick_est;
}

// FIXED is costed exactly unless its estimate exceeds the best LPC
// subframe's by this factor, the two estimated alike (one Rice partition
// from the mean |residual|). On the passband store's hours no subframe
// above it wins as FIXED (measured: the same bytes as costing FIXED always).
constexpr double FIXED_MARGIN = 1.1;

// The cheapest subframe for x by exact size: CONSTANT, VERBATIM, LPC or
// FIXED.
void plan_subframe(const std::int32_t* x, const std::int16_t* x16, int n, int bps, const EncodeOptions& opt,
                   Scratch& sc, Analysis& a, Subframe& sf) {
    sf.order = sf.porder = sf.method = sf.precision = sf.shift = 0;
    if (a.constant) {
        sf.type = SUB_CONSTANT;
        sf.bits = 8 + static_cast<std::uint64_t>(bps);
        return;
    }
    sf.type = SUB_VERBATIM;
    sf.bits = 8 + static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(bps);
    double lpc_est = std::numeric_limits<double>::infinity();
    if (a.got[0] > 0) lpc_est = plan_lpc(x, x16, n, bps, opt, sc, a, sf);
    const FixedLook fx = a.got[0] > 0 ? fixed_look(x, n, bps, opt) : a.fixed;
    if (fx.order >= 0 && !(fx.est > FIXED_MARGIN * lpc_est)) {
        const int p = fx.order;
        fixed_residual(x, n, p, sc.r);
        int porder = 0, method = 0;
        std::uint64_t rbits = 0;
        plan_residual(sc, n, p, opt.max_partition_order, porder, method, rbits);
        const std::uint64_t fbits = 8 + static_cast<std::uint64_t>(p) * static_cast<std::uint64_t>(bps) + rbits;
        if (fbits < sf.bits) {
            sf.type = SUB_FIXED;
            sf.order = p;
            sf.porder = porder;
            sf.method = method;
            sf.bits = fbits;
            sf.residual.swap(sc.r);
            sf.parts.swap(sc.parts);
        }
    }
}

void write_subframe(BitWriter& w, const Subframe& sf, const std::int32_t* x, int n, int bps) {
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
    if (sf.type == SUB_FIXED) {
        w.put(static_cast<std::uint32_t>(0x10 | (sf.order << 1)), 8);
        for (int i = 0; i < sf.order; ++i) w.put_signed(x[i], bps);
    } else {
        w.put(static_cast<std::uint32_t>(0x40 | ((sf.order - 1) << 1)), 8);
        for (int i = 0; i < sf.order; ++i) w.put_signed(x[i], bps);
        w.put(static_cast<std::uint32_t>(sf.precision - 1), 4);
        w.put_signed(sf.shift, 5);
        for (int j = 0; j < sf.order; ++j) w.put_signed(sf.coef[static_cast<std::size_t>(j)], sf.precision);
    }
    // residual: method, partition order, then each partition
    w.put(static_cast<std::uint32_t>(sf.method), 2);
    w.put(static_cast<std::uint32_t>(sf.porder), 4);
    const int parts = 1 << sf.porder;
    const int s = n >> sf.porder;
    const int pw = sf.method == 0 ? 4 : 5;
    const int esc = sf.method == 0 ? 15 : 31;
    const std::int32_t* r = sf.residual.data();
    for (int j = 0; j < parts; ++j) {
        const int a = j == 0 ? 0 : j * s - sf.order;
        const int b = (j + 1) * s - sf.order;
        const PartParam c = sf.parts[static_cast<std::size_t>(j)];
        if (c.k < 0) {
            w.put(static_cast<std::uint32_t>(esc), pw);
            w.put(static_cast<std::uint32_t>(c.raw), 5);
            for (int i = a; i < b; ++i) w.put_signed(r[i], c.raw);
        } else {
            w.put(static_cast<std::uint32_t>(c.k), pw);
            for (int i = a; i < b; ++i) w.rice(zigzag(r[i]), c.k);
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
        // To the device before returning: the packer renames this file into
        // place and then deletes the raw hour, so a FLAC still in the page
        // cache at a power cut (ext4's delayed allocation commits a rename to
        // a new name before the data) would lose the hour outright.
        std::error_code ec;
        if (!f_.sync(ec)) throw rawfile::FileError(ec, path_);
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
    if (opt.max_lpc_order < 0 || opt.max_lpc_order > MAX_LPC_ORDER)
        throw std::invalid_argument("FLAC encoder: LPC order 0..32");
    if (opt.qlp_precision != 0 && (opt.qlp_precision < 5 || opt.qlp_precision > 15))
        throw std::invalid_argument("FLAC encoder: coefficient precision 0 or 5..15");
    if (opt.lpc_orders_tried < 1) throw std::invalid_argument("FLAC encoder: LPC orders tried >= 1");
    // (FLAC allows 15, but dr_flac -- the store's reader -- refuses a partition order above 8)
    if (opt.max_partition_order < 0 || opt.max_partition_order > 8)
        throw std::invalid_argument("FLAC encoder: partition order 0..8");
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
    std::vector<std::int32_t> ch[4];   // left, right, mid, side
    for (auto& c : ch) c.resize(static_cast<std::size_t>(bs));
    std::vector<std::int16_t> ch16[3];   // left, right and mid again, as int16 (side takes 17 bits)
    for (auto& c : ch16) c.resize(static_cast<std::size_t>(bs));
    Scratch sc;
    std::vector<Analysis> an(4);
    Subframe sfs[4];   // left, right, mid, side
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
                ch16[c][static_cast<std::size_t>(i)] = pcm[static_cast<std::size_t>(i) * channels + c];
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < n; ++i) ch[c][static_cast<std::size_t>(i)] = ch16[c][static_cast<std::size_t>(i)];

        // channel assignment and subframes. Every candidate channel (left,
        // right, mid, side) is analyzed, the assignment the summed estimates
        // favour is costed exactly beside independent coding (left/right),
        // and the smaller written: never larger than coding the channels
        // independently, and on the passband store's uncorrelated I and Q
        // nearly always just the two exact plans. (libFLAC costs all four
        // channels exactly; that finds about 0.01% more there, for half again
        // the time.)
        int pick = 0;   // 0 L/R, 1 L/S, 2 S/R, 3 M/S (the index into PAIRS and ASSIGN)
        static constexpr int PAIRS[4][2] = {{0, 1}, {0, 3}, {3, 1}, {2, 3}};
        static constexpr int ASSIGN[4] = {1, 8, 9, 10};
        analyze(ch[0].data(), n, 16, opt, sc, an[0]);
        plan_subframe(ch[0].data(), ch16[0].data(), n, 16, opt, sc, an[0], sfs[0]);
        if (channels == 2) {
            analyze(ch[1].data(), n, 16, opt, sc, an[1]);
            plan_subframe(ch[1].data(), ch16[1].data(), n, 16, opt, sc, an[1], sfs[1]);
            if (opt.stereo_decorrelation) {
                for (int i = 0; i < n; ++i) {
                    const std::int32_t a = ch[0][static_cast<std::size_t>(i)], b = ch[1][static_cast<std::size_t>(i)];
                    ch[2][static_cast<std::size_t>(i)] = (a + b) >> 1;   // mid (arithmetic shift: floor)
                    ch16[2][static_cast<std::size_t>(i)] = static_cast<std::int16_t>((a + b) >> 1);
                    ch[3][static_cast<std::size_t>(i)] = a - b;          // side, 17 bits
                }
                analyze(ch[2].data(), n, 16, opt, sc, an[2]);
                analyze(ch[3].data(), n, 17, opt, sc, an[3]);
                const double cost[4] = {an[0].est + an[1].est, an[0].est + an[3].est, an[3].est + an[1].est,
                                        an[2].est + an[3].est};
                int best = 0;
                for (int k = 1; k < 4; ++k)
                    if (cost[k] < cost[best]) best = k;
                if (best != 0) {
                    plan_subframe(ch[3].data(), nullptr, n, 17, opt, sc, an[3], sfs[3]);
                    if (best == 3) plan_subframe(ch[2].data(), ch16[2].data(), n, 16, opt, sc, an[2], sfs[2]);
                    const int* pr = PAIRS[best];
                    if (sfs[pr[0]].bits + sfs[pr[1]].bits < sfs[0].bits + sfs[1].bits) pick = best;
                }
                ++stats.stereo[static_cast<std::size_t>(pick)];
            }
        }
        const int assign = channels == 1 ? 0 : ASSIGN[pick];
        const Subframe* out[2] = {&sfs[PAIRS[pick][0]], channels == 2 ? &sfs[PAIRS[pick][1]] : nullptr};
        const std::int32_t* data[2] = {ch[PAIRS[pick][0]].data(), channels == 2 ? ch[PAIRS[pick][1]].data() : nullptr};
        const int bps[2] = {PAIRS[pick][0] == 3 ? 17 : 16, PAIRS[pick][1] == 3 ? 17 : 16};

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
                const Subframe& sf = *out[c];
                write_subframe(w, sf, data[c], n, bps[c]);
                const int t = sf.type == SUB_FIXED ? 2 + sf.order : sf.type == SUB_LPC ? 7 : sf.type;
                ++stats.subframes[static_cast<std::size_t>(t)];
                if (sf.type == SUB_LPC) ++stats.lpc_orders[static_cast<std::size_t>(sf.order)];
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
        {
            // dr_flac writes its CPU-feature globals in every open
            // (drflac__init_cpu_caps: on x86 the first open only, on ARM
            // every one), so opens are serialised: the packer verifying a
            // FLAC and a reader opening a packed hour never write them at
            // once. (On ARM a decode on one thread may still read the flag
            // another thread's open rewrites with the same value; dr_flac
            // marks that function no_sanitize("thread") for it.)
            static std::mutex open_mu;
            std::lock_guard lk(open_mu);
            dec = drflac_open_with_metadata(on_read, on_seek, on_tell, on_meta, &stream, nullptr);
        }
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
