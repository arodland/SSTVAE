// FLAC encoder (ours) and reader (dr_flac) for the passband store; see
// flac.hpp for what is written and why.

#include "qrss/flac.hpp"

#include <algorithm>
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
int signed_bits(std::int64_t lo, std::int64_t hi) {
    if (lo == 0 && hi == 0) return 0;
    int b = 1;
    while (!(lo >= -(std::int64_t{1} << (b - 1)) && hi <= (std::int64_t{1} << (b - 1)) - 1)) ++b;
    return b;
}

int bit_length(std::uint64_t v) {
    int b = 0;
    while (v) {
        ++b;
        v >>= 1;
    }
    return b;
}

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

struct PartitionStats {
    std::vector<std::uint64_t> s;   // sum(u >> k), k = 0..kcap
    std::uint64_t count = 0;
    std::int64_t lo = 0, hi = 0;    // residual range
};

// Cheapest coding of one partition: (bits after the parameter field,
// parameter or -1 for an escape, raw bits for the escape).
struct PartChoice {
    std::uint64_t bits;
    int k;
    int raw;
};

PartChoice best_partition(const PartitionStats& p, int kcap, int kmax_method) {
    PartChoice best{std::numeric_limits<std::uint64_t>::max(), 0, 0};
    const int top = std::min(kcap, kmax_method);
    for (int k = 0; k <= top; ++k) {
        const std::uint64_t b = p.s[static_cast<std::size_t>(k)] + p.count * static_cast<std::uint64_t>(k + 1);
        if (b < best.bits) best = {b, k, 0};
    }
    const int raw = signed_bits(p.lo, p.hi);
    if (raw <= 31) {
        const std::uint64_t b = 5 + p.count * static_cast<std::uint64_t>(raw);
        if (b < best.bits) best = {b, -1, raw};
    }
    return best;
}

// The partition order and parameter width with the fewest bits for a
// FIXED residual of a block of n samples, order p; the residual section's
// bits (method and order fields included) go in `bits`.
void plan_residual(const std::vector<std::int32_t>& r, int n, int p, int max_porder, int& porder_out,
                   int& method_out, std::uint64_t& bits_out) {
    int rmax = 0;
    while (rmax < max_porder && (n % (1 << (rmax + 1))) == 0 && (n >> (rmax + 1)) > p) ++rmax;
    std::uint32_t umax = 0;
    for (const std::int32_t v : r) umax = std::max(umax, zigzag(v));
    const int kcap = std::min(bit_length(umax), 30);
    const int parts = 1 << rmax;
    const int s = n >> rmax;
    std::vector<PartitionStats> level(static_cast<std::size_t>(parts));
    for (int j = 0; j < parts; ++j) {
        const int a = j == 0 ? 0 : j * s - p;
        const int b = (j + 1) * s - p;
        PartitionStats& st = level[static_cast<std::size_t>(j)];
        st.s.assign(static_cast<std::size_t>(kcap) + 1, 0);
        st.count = static_cast<std::uint64_t>(b - a);
        std::int64_t lo = 0, hi = 0;
        for (int i = a; i < b; ++i) {
            lo = std::min<std::int64_t>(lo, r[static_cast<std::size_t>(i)]);
            hi = std::max<std::int64_t>(hi, r[static_cast<std::size_t>(i)]);
        }
        st.lo = lo;
        st.hi = hi;
        for (int k = 0; k <= kcap; ++k) {
            std::uint64_t sum = 0;
            for (int i = a; i < b; ++i) sum += zigzag(r[static_cast<std::size_t>(i)]) >> k;
            st.s[static_cast<std::size_t>(k)] = sum;
        }
    }
    std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
    for (int lev = rmax;; --lev) {
        for (int method = 0; method < 2; ++method) {
            std::uint64_t total = 2 + 4;
            const int pw = method == 0 ? 4 : 5;
            const int kmax_m = method == 0 ? 14 : 30;
            for (const PartitionStats& st : level) total += pw + best_partition(st, kcap, kmax_m).bits;
            if (total < best) {
                best = total;
                porder_out = lev;
                method_out = method;
            }
        }
        if (lev == 0) break;
        std::vector<PartitionStats> up(level.size() / 2);
        for (std::size_t j = 0; j < up.size(); ++j) {
            const PartitionStats& x = level[2 * j];
            const PartitionStats& y = level[2 * j + 1];
            PartitionStats& z = up[j];
            z.s.resize(x.s.size());
            for (std::size_t k = 0; k < x.s.size(); ++k) z.s[k] = x.s[k] + y.s[k];
            z.count = x.count + y.count;
            z.lo = std::min(x.lo, y.lo);
            z.hi = std::max(x.hi, y.hi);
        }
        level = std::move(up);
    }
    bits_out = best;
}

// FIXED predictor residual of order p (x has n samples; residual n - p).
void fixed_residual(const std::int32_t* x, int n, int p, std::vector<std::int32_t>& r) {
    r.resize(static_cast<std::size_t>(n - p));
    for (int i = p; i < n; ++i) {
        std::int32_t e;
        switch (p) {
            case 0: e = x[i]; break;
            case 1: e = x[i] - x[i - 1]; break;
            case 2: e = x[i] - 2 * x[i - 1] + x[i - 2]; break;
            case 3: e = x[i] - 3 * x[i - 1] + 3 * x[i - 2] - x[i - 3]; break;
            default: e = x[i] - 4 * x[i - 1] + 6 * x[i - 2] - 4 * x[i - 3] + x[i - 4]; break;
        }
        r[static_cast<std::size_t>(i - p)] = e;
    }
}

Subframe plan_subframe(const std::int32_t* x, int n, int bps, const EncodeOptions& opt) {
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
    int best_p = 0;
    std::uint64_t best_e = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::int32_t> r;
    for (int p = 0; p <= maxp; ++p) {
        fixed_residual(x, n, p, r);
        std::uint64_t e = 0;
        for (std::size_t i = static_cast<std::size_t>(maxp - p); i < r.size(); ++i)
            e += static_cast<std::uint64_t>(r[i] < 0 ? -static_cast<std::int64_t>(r[i]) : r[i]);
        if (e < best_e) {
            best_e = e;
            best_p = p;
        }
    }
    fixed_residual(x, n, best_p, r);
    int porder = 0, method = 0;
    std::uint64_t rbits = 0;
    plan_residual(r, n, best_p, opt.max_partition_order, porder, method, rbits);
    const std::uint64_t fbits = 8 + static_cast<std::uint64_t>(best_p) * static_cast<std::uint64_t>(bps) + rbits;
    if (fbits < sf.bits) {
        sf.type = SUB_FIXED;
        sf.order = best_p;
        sf.porder = porder;
        sf.method = method;
        sf.bits = fbits;
        sf.residual = std::move(r);
    }
    return sf;
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
    for (int j = 0; j < parts; ++j) {
        const int a = j == 0 ? 0 : j * s - sf.order;
        const int b = (j + 1) * s - sf.order;
        PartitionStats st;
        std::uint32_t umax = 0;
        std::int64_t lo = 0, hi = 0;
        for (int i = a; i < b; ++i) {
            const std::int32_t v = sf.residual[static_cast<std::size_t>(i)];
            umax = std::max(umax, zigzag(v));
            lo = std::min<std::int64_t>(lo, v);
            hi = std::max<std::int64_t>(hi, v);
        }
        const int kcap = std::min(bit_length(umax), 30);
        st.s.assign(static_cast<std::size_t>(kcap) + 1, 0);
        for (int k = 0; k <= kcap; ++k) {
            std::uint64_t sum = 0;
            for (int i = a; i < b; ++i) sum += zigzag(sf.residual[static_cast<std::size_t>(i)]) >> k;
            st.s[static_cast<std::size_t>(k)] = sum;
        }
        st.count = static_cast<std::uint64_t>(b - a);
        st.lo = lo;
        st.hi = hi;
        const PartChoice c = best_partition(st, kcap, kmax_m);
        if (c.k < 0) {
            w.put(static_cast<std::uint32_t>(esc), pw);
            w.put(static_cast<std::uint32_t>(c.raw), 5);
            for (int i = a; i < b; ++i) w.put_signed(sf.residual[static_cast<std::size_t>(i)], c.raw);
        } else {
            w.put(static_cast<std::uint32_t>(c.k), pw);
            for (int i = a; i < b; ++i) w.rice(zigzag(sf.residual[static_cast<std::size_t>(i)]), c.k);
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
        if (!f_.is_open()) throw std::system_error(ec, path);
        buf_.reserve(1 << 20);
    }
    void write(const std::uint8_t* p, std::size_t n) override {
        buf_.insert(buf_.end(), p, p + n);
        if (buf_.size() >= (1u << 20)) flush();
    }
    void patch(std::uint64_t at, const std::uint8_t* p, std::size_t n) override {
        flush();
        std::error_code ec;
        if (!f_.write_at(at, p, n, ec)) throw std::system_error(ec, path_);
    }
    void finish() override {
        flush();
        f_.close();
    }

private:
    void flush() {
        if (buf_.empty()) return;
        std::error_code ec;
        if (!f_.write_at(at_, buf_.data(), buf_.size(), ec)) throw std::system_error(ec, path_);
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
            sfs[0] = plan_subframe(ch[0].data(), n, 16, opt);
        } else {
            Subframe l = plan_subframe(ch[0].data(), n, 16, opt);
            Subframe r = plan_subframe(ch[1].data(), n, 16, opt);
            if (!opt.stereo_decorrelation) {
                sfs[0] = std::move(l);
                sfs[1] = std::move(r);
            } else {
                for (int i = 0; i < n; ++i) {
                    const std::int32_t a = ch[0][static_cast<std::size_t>(i)], b = ch[1][static_cast<std::size_t>(i)];
                    ch[2][static_cast<std::size_t>(i)] = (a + b) >> 1;   // mid (arithmetic shift: floor)
                    ch[3][static_cast<std::size_t>(i)] = a - b;          // side, 17 bits
                }
                Subframe m = plan_subframe(ch[2].data(), n, 16, opt);
                Subframe s = plan_subframe(ch[3].data(), n, 17, opt);
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
                write_subframe(w, sfs[c], data[c], n, bps[c]);
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
    if (!impl_->stream.file.is_open()) throw std::system_error(ec, path);
    const auto sz = impl_->stream.file.size(ec);
    if (!sz) throw std::system_error(ec, path);
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
