// frontend.PassbandStore: see passband.hpp.

#include "qrss/passband.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <string>

#include "qrss/flac.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/rational.hpp"
#include "qrss/rawfile.hpp"
#include "util/fileio.hpp"

namespace sstvae::qrss::passband {

static_assert(std::endian::native == std::endian::little,
              "the store's files are little-endian int16 (numpy's native order on every target)");

namespace {

constexpr std::uint64_t CHUNK = 1u << 16;   // frames per pass when packing, verifying, unpacking

[[noreturn]] void too_big(const char* what) {
    throw std::overflow_error(std::string("the passband store's ") + what + " is out of the int64 range");
}

// Checked int64 arithmetic (portable: MSVC has no __builtin_*_overflow).
constexpr std::int64_t I64_MAX = std::numeric_limits<std::int64_t>::max();
constexpr std::int64_t I64_MIN = std::numeric_limits<std::int64_t>::min();

std::int64_t mul(std::int64_t a, std::int64_t b, const char* what) {
    bool ovf = false;
    if (a > 0)
        ovf = b > 0 ? a > I64_MAX / b : b < I64_MIN / a;
    else if (a < 0)
        ovf = b > 0 ? a < I64_MIN / b : (b != 0 && b < I64_MAX / a);
    if (ovf) too_big(what);
    return a * b;
}

std::int64_t add(std::int64_t a, std::int64_t b, const char* what) {
    if ((b > 0 && a > I64_MAX - b) || (b < 0 && a < I64_MIN - b)) too_big(what);
    return a + b;
}

std::int64_t sub(std::int64_t a, std::int64_t b, const char* what) {
    if ((b < 0 && a > I64_MAX + b) || (b > 0 && a < I64_MIN + b)) too_big(what);
    return a - b;
}

// divmod(n, d) for d > 0 (floor division, a non-negative remainder).
std::pair<std::int64_t, std::int64_t> floor_divmod(std::int64_t n, std::int64_t d) {
    std::int64_t q = n / d, r = n % d;
    if (r < 0) {
        r += d;
        --q;
    }
    return {q, r};
}

// A float32 from a double, rounded to nearest-even as numpy casts, with an
// overflow to +-inf (the C++ conversion of an out-of-range value is undefined).
float f32(double d) {
    constexpr double midpoint = 0x1.ffffffp+127;   // halfway from FLT_MAX to 2^128: rounds to inf (ties to even)
    if (std::isfinite(d) && std::fabs(d) >= midpoint)
        return d > 0 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
    if (std::isfinite(d) && std::fabs(d) > static_cast<double>(std::numeric_limits<float>::max()))
        return d > 0 ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
    return static_cast<float>(d);
}

// math.floor / math.ceil of a float, as an int: their errors for NaN and infinity.
std::int64_t to_int(double v) {
    if (std::isnan(v)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(v)) throw std::overflow_error("cannot convert float infinity to integer");
    if (v >= 0x1p63 || v < -0x1p63) too_big("slot number");
    return static_cast<std::int64_t>(v);
}

// One part of a sample: rint(part * scale) in float32, clipped, cast (NaN: 0).
std::int16_t quantise_part(float v, float s) {
    float r = std::nearbyint(v * s);
    if (std::isnan(r)) return 0;
    if (r < -32767.0f) r = -32767.0f;
    if (r > 32767.0f) r = 32767.0f;
    return static_cast<std::int16_t>(r);
}

// np.asarray(int16 pair, float32) / scale, assembled as q0 + 1j * q1 does it.
cf dequantise(std::int16_t re, std::int16_t im, float s) {
    const float q0 = static_cast<float>(re) / s;
    const float q1 = static_cast<float>(im) / s;
    // 1j * q1 is complex64 (0 + 1j) * (q1 + 0j); then q0 + 0j plus that.
    const float pr = 0.0f * q1 - 1.0f * 0.0f;
    const float pi = 0.0f * 0.0f + 1.0f * q1;
    return {q0 + pr, 0.0f + pi};
}

std::string join(const std::string& root, const std::string& name) {
    if (root.empty()) return name;
    const char last = root.back();
#ifdef _WIN32
    if (last == '/' || last == '\\') return root + name;
#else
    if (last == '/') return root + name;
#endif
    return root + "/" + name;
}

std::string hour_name(std::int64_t hour, const char* ext) { return "fe_" + std::to_string(hour) + ext; }

// _atomic_write's temporary: "<name>.<pid>.<thread ident>.tmp" beside it.
std::string atomic_tmp(const std::string& root, const std::string& name) {
    return join(root, name + "." + std::to_string(rawfile::process_id()) + "." +
                          std::to_string(rawfile::thread_ident()) + ".tmp");
}

void atomic_write(const std::string& root, const std::string& name, const std::string& text) {
    std::error_code ec;
    const std::string tmp = atomic_tmp(root, name);
    if (!rawfile::write_replace(tmp, join(root, name), text, ec)) throw FileError(ec, tmp);
}

// int(v) of a meta.json value (an int, a float truncated, a bool).
std::int64_t py_int(const pyjson::Value& v, const char* key) {
    using K = pyjson::Value::Kind;
    switch (v.kind()) {
        case K::Int: return v.as_int();
        case K::Bool: return v.as_bool() ? 1 : 0;
        case K::Float: {
            const double f = std::trunc(v.as_double());
            if (std::isnan(f)) throw std::invalid_argument("cannot convert float NaN to integer");
            if (std::isinf(f)) throw std::overflow_error("cannot convert float infinity to integer");
            if (f >= 0x1p63 || f < -0x1p63) too_big(key);
            return static_cast<std::int64_t>(f);
        }
        default: throw std::invalid_argument(std::string("meta.json: ") + key + " is not a number");
    }
}

double py_float(const pyjson::Value& v, const char* key) {
    using K = pyjson::Value::Kind;
    switch (v.kind()) {
        case K::Int:
        case K::Float: return v.as_double();
        case K::Bool: return v.as_bool() ? 1.0 : 0.0;
        default: throw std::invalid_argument(std::string("meta.json: ") + key + " is not a number");
    }
}

void remove_missing_ok(const std::string& path) {
    std::error_code ec;
    if (!util::remove_file(path, ec) && ec != std::errc::no_such_file_or_directory) throw FileError(ec, path);
}

bool is_missing(const std::error_code& ec) { return ec == std::errc::no_such_file_or_directory; }

[[noreturn]] void short_file(const std::string& path) {
    throw std::invalid_argument(path + ": mmap length is greater than file size");
}

}  // namespace

// --- free functions -------------------------------------------------------------

std::int64_t slot_t0(std::int64_t q) { return add(mul(900, q, "slot time"), 1, "slot time"); }

std::vector<Range> merge_ranges(std::vector<Range> rs) {
    std::sort(rs.begin(), rs.end());
    std::vector<Range> out;
    for (const Range& r : rs) {
        if (!out.empty() && r[0] <= out.back()[1])
            out.back()[1] = std::max(out.back()[1], r[1]);
        else
            out.push_back(r);
    }
    return out;
}

std::optional<std::int64_t> hour_of_name(std::string_view name) {
    if (!name.empty() && name.back() == '\n') name.remove_suffix(1);   // `$` before a final newline
    if (name.substr(0, 3) != "fe_") return std::nullopt;
    name.remove_prefix(3);
    std::size_t k = 0;
    while (k < name.size() && name[k] >= '0' && name[k] <= '9') ++k;
    if (k == 0) return std::nullopt;
    const std::string_view ext = name.substr(k);
    if (ext != ".i16" && ext != ".flac") return std::nullopt;
    std::int64_t h = 0;
    for (std::size_t i = 0; i < k; ++i) {
        h = 10 * h + (name[i] - '0');
        if (h > HOUR_MAX) return std::nullopt;   // (Python's int would hold it; no store has such an hour,
    }                                            //  and none here may: hour arithmetic never overflows)
    return h;
}

std::int64_t py_round(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
    double r = std::round(x);   // half away from zero ...
    if (std::fabs(x - std::trunc(x)) == 0.5) r = 2.0 * std::round(x / 2.0);   // ... but ties to even
    if (r >= 0x1p63 || r < -0x1p63) too_big("sample index");
    return static_cast<std::int64_t>(r);
}

// --- the store ------------------------------------------------------------------

struct PassbandStore::Hour {
    std::string path;
    rawfile::RawFile raw;                    // open when the hour is raw
    std::unique_ptr<flac::Reader> packed;    // else this

    // int16 pairs [lo, hi) into out (2 (hi - lo) values).
    void read(std::int64_t lo, std::int64_t hi, std::int16_t* out) {
        const std::size_t n = static_cast<std::size_t>(hi - lo);
        if (raw.is_open()) {
            std::error_code ec;
            const std::size_t want = n * 4;
            const std::size_t got = raw.read_at(static_cast<std::uint64_t>(lo) * 4, out, want, ec);
            if (ec) throw FileError(ec, path);
            if (got != want) short_file(path);
            return;
        }
        const flac::Info& info = packed->info();
        const std::vector<std::int16_t> y = packed->read(static_cast<std::uint64_t>(lo), n);
        const std::size_t nch = static_cast<std::size_t>(info.channels);
        if (nch < 2)
            throw std::out_of_range("index 1 is out of bounds for axis 1 with size " + std::to_string(nch));
        if (y.size() != n * nch)
            throw std::invalid_argument(path + ": could not broadcast " + std::to_string(y.size() / nch) +
                                        " samples into " + std::to_string(n));
        for (std::size_t i = 0; i < n; ++i) {
            out[2 * i] = y[i * nch];
            out[2 * i + 1] = y[i * nch + 1];
        }
    }
};

PassbandStore::PassbandStore(std::string root, StoreOptions opt)
    : root_(std::move(root)),
      fs_(opt.fs),
      scale_(opt.scale),
      keep_hours_(opt.keep_hours),
      per_hour_(0),
      pack_(opt.pack),
      clock_(std::move(opt.clock)) {
    if (root_.empty()) root_ = ".";   // (Path("") is ".")
    std::error_code ec;
    std::string failed;
    if (!rawfile::make_dirs(root_, ec, &failed)) throw FileError(ec, failed);
    const std::string meta = join(root_, "meta.json");
    if (util::exists(meta)) {
        const auto text = rawfile::read_all(meta, ec);
        if (!text) throw FileError(ec, meta);
        const pyjson::Value m = pyjson::loads(*text);
        const std::int64_t fs = py_int(m.at("fs"), "fs");
        const double scale = py_float(m.at("scale"), "scale");
        if (fs > std::numeric_limits<int>::max() || fs < std::numeric_limits<int>::min()) too_big("fs");
        fs_ = static_cast<int>(fs);
        scale_ = scale;
    } else {
        pyjson::Value m = pyjson::Value::object();
        m.set("fs", pyjson::Value::integer(fs_));
        m.set("scale", pyjson::Value::number(scale_));
        atomic_write(root_, "meta.json", pyjson::dumps(m));
    }
    per_hour_ = 3600 * static_cast<std::int64_t>(fs_);
}

PassbandStore::~PassbandStore() {
    closing_.store(true);
    std::thread t;
    {
        std::lock_guard lk(packer_mu_);
        t = std::move(packer_);
    }
    if (t.joinable()) t.join();
}

std::string PassbandStore::raw_path(std::int64_t hour) const { return join(root_, hour_name(hour, ".i16")); }
std::string PassbandStore::index_path(std::int64_t hour) const { return join(root_, hour_name(hour, ".json")); }
std::string PassbandStore::flac_path(std::int64_t hour) const { return join(root_, hour_name(hour, ".flac")); }

double PassbandStore::now() const {
    if (clock_) return clock_();
    const auto d = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration<double>(d).count();
}

void PassbandStore::quantise(std::span<const cf> x, std::vector<std::int16_t>& out) const {
    const float s = f32(scale_);   // (NEP 50: the Python float scale meets a float32 array as float32)
    out.resize(2 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[2 * i] = quantise_part(x[i].real(), s);
        out[2 * i + 1] = quantise_part(x[i].imag(), s);
    }
}

std::vector<Range> PassbandStore::ranges_locked(std::int64_t hour) const {
    const std::string idx = index_path(hour);
    std::error_code ec;
    const auto text = rawfile::read_all(idx, ec);
    if (!text) {
        if (is_missing(ec)) return {};
        throw FileError(ec, idx);
    }
    const pyjson::Value v = pyjson::loads(*text);
    std::vector<Range> out;
    for (const pyjson::Value& r : v.items()) {
        const auto& p = r.items();
        if (p.size() != 2) throw std::invalid_argument(idx + ": a range is not [lo, hi]");
        out.push_back({p[0].as_int("a range's lo"), p[1].as_int("a range's hi")});
    }
    return out;
}

std::shared_ptr<PassbandStore::Hour> PassbandStore::open_hour_locked(std::int64_t hour) const {
    auto h = std::make_shared<Hour>();
    std::error_code ec;
    h->path = raw_path(hour);
    h->raw = rawfile::RawFile::open(h->path, rawfile::Mode::Read, ec);
    if (h->raw.is_open()) {
        const auto size = h->raw.size(ec);
        if (!size) throw FileError(ec, h->path);
        if (*size < static_cast<std::uint64_t>(per_hour_) * 4) short_file(h->path);
        return h;
    }
    if (!is_missing(ec)) throw FileError(ec, h->path);
    h->path = flac_path(hour);
    if (!util::exists(h->path)) return nullptr;
    try {
        h->packed = std::make_unique<flac::Reader>(h->path);
    } catch (const std::system_error& e) {
        if (is_missing(e.code())) return nullptr;   // (gone since the look: as the reference's retry ends)
        throw FileError(e.code(), h->path);
    }
    return h;
}

void PassbandStore::unpack_locked(std::int64_t hour) {
    const std::string src = flac_path(hour);
    flac::Reader r(src);
    const flac::Info& info = r.info();
    const std::size_t nch = static_cast<std::size_t>(info.channels);
    const std::uint64_t per = static_cast<std::uint64_t>(per_hour_);
    const std::uint64_t n = std::min<std::uint64_t>(info.frames, per);
    if (n > 0 && nch != 1 && nch != 2)
        throw std::invalid_argument(src + ": could not broadcast input array from shape (" + std::to_string(n) +
                                    "," + std::to_string(nch) + ") into shape (" + std::to_string(n) + ",2)");
    const std::string tmp =
        join(root_, hour_name(hour, ".i16") + "." + std::to_string(rawfile::process_id()) + ".tmp");
    std::error_code ec;
    try {
        rawfile::RawFile t = rawfile::RawFile::open(tmp, rawfile::Mode::Truncate, ec);
        if (!t.is_open()) throw FileError(ec, tmp);
        if (!t.make_sparse(ec) || !t.set_size(per * 4, ec)) throw FileError(ec, tmp);
        std::vector<std::int16_t> buf;
        for (std::uint64_t first = 0; first < n; first += CHUNK) {
            const std::uint64_t want = std::min<std::uint64_t>(CHUNK, n - first);
            const std::vector<std::int16_t> y = r.read(first, want);
            const std::size_t got = y.size() / nch;
            buf.resize(2 * got);
            for (std::size_t i = 0; i < got; ++i) {
                buf[2 * i] = y[i * nch];
                buf[2 * i + 1] = y[i * nch + (nch - 1)];   // (one channel broadcasts to both)
            }
            // runs of non-zero 1024-sample blocks only, so an unwritten stretch stays a hole
            constexpr std::size_t B = 1024;
            for (std::size_t b = 0; b < got;) {
                auto nonzero = [&](std::size_t at) {
                    const std::size_t e = std::min(got, at + B);
                    return std::any_of(buf.begin() + 2 * at, buf.begin() + 2 * e, [](std::int16_t v) { return v != 0; });
                };
                if (!nonzero(b)) {
                    b += B;
                    continue;
                }
                std::size_t e = b + B;
                while (e < got && nonzero(e)) e += B;
                e = std::min(e, got);
                if (!t.write_at((first + b) * 4, buf.data() + 2 * b, (e - b) * 4, ec)) throw FileError(ec, tmp);
                b = e;
            }
            if (got < want) break;   // (the stream ended before its STREAMINFO said)
        }
        if (!t.sync(ec)) throw FileError(ec, tmp);   // (the reference's mm.flush())
        t.close();
        if (!util::replace_file(tmp, raw_path(hour), ec)) throw FileError(ec, tmp);
    } catch (...) {
        std::error_code ignore;
        util::remove_file(tmp, ignore);
        throw;
    }
    if (!util::remove_file(src, ec)) throw FileError(ec, src);
}

void PassbandStore::write_hour_locked(std::int64_t hour, std::int64_t k, const std::vector<std::int16_t>& q) {
    const std::string raw = raw_path(hour);
    if (!util::exists(raw) && util::exists(flac_path(hour))) unpack_locked(hour);
    std::error_code ec;
    const std::uint64_t bytes = static_cast<std::uint64_t>(per_hour_) * 4;
    rawfile::RawFile f;
    if (!util::exists(raw)) {
        f = rawfile::RawFile::open(raw, rawfile::Mode::Truncate, ec);
        if (!f.is_open()) throw FileError(ec, raw);
        if (!f.make_sparse(ec) || !f.set_size(bytes, ec)) throw FileError(ec, raw);
    } else {
        f = rawfile::RawFile::open(raw, rawfile::Mode::ReadWrite, ec);
        if (!f.is_open()) throw FileError(ec, raw);
        const auto size = f.size(ec);
        if (!size) throw FileError(ec, raw);
        if (*size < bytes) short_file(raw);
    }
    if (!f.write_at(static_cast<std::uint64_t>(k) * 4, q.data(), q.size() * sizeof(std::int16_t), ec))
        throw FileError(ec, raw);
    f.close();
    ++gen_[hour];
    const std::int64_t m = static_cast<std::int64_t>(q.size() / 2);
    std::vector<Range> rs = ranges_locked(hour);
    rs.push_back({k, k + m});
    pyjson::Value out = pyjson::Value::array();
    for (const Range& r : merge_ranges(std::move(rs)))
        out.push_back(pyjson::Value::array({pyjson::Value::integer(r[0]), pyjson::Value::integer(r[1])}));
    atomic_write(root_, hour_name(hour, ".json"), pyjson::dumps(out));
}

void PassbandStore::write(double t0, std::span<const cf> x, bool expire) {
    write_at(py_round(t0 * static_cast<double>(fs_)), x, expire);
}

void PassbandStore::write_at(std::int64_t n0, std::span<const cf> x, bool expire_) {
    const std::int64_t len = static_cast<std::int64_t>(x.size());
    if (len > 0) {
        if (per_hour_ == 0) throw ZeroDivision("integer division or modulo by zero");
        if (per_hour_ < 0) throw std::invalid_argument("negative dimensions are not allowed");
    }
    const std::int64_t end = add(n0, len, "sample index");
    std::vector<std::int16_t> q;
    std::int64_t i = 0;
    while (i < len) {
        const auto [hour, k] = floor_divmod(n0 + i, per_hour_);
        const std::int64_t m = std::min(len - i, per_hour_ - k);
        quantise(x.subspan(static_cast<std::size_t>(i), static_cast<std::size_t>(m)), q);
        {
            std::lock_guard lk(mu_);
            write_hour_locked(hour, k, q);
        }
        i += m;
    }
    if (expire_ && len > 0) {
        const double t = static_cast<double>(end) / static_cast<double>(fs_);
        expire(t);
        maybe_pack(t);
    }
}

Masked PassbandStore::read(double t_start, double t_end) const {
    const std::int64_t a = py_round(t_start * static_cast<double>(fs_));
    const std::int64_t b = py_round(t_end * static_cast<double>(fs_));
    const std::int64_t n = std::max<std::int64_t>(0, sub(b, a, "sample count"));
    Masked out;
    out.x.assign(static_cast<std::size_t>(n), cf(0.0f, 0.0f));
    out.mask.assign(static_cast<std::size_t>(n), 0);
    if (n == 0) return out;
    if (per_hour_ == 0) throw ZeroDivision("integer division or modulo by zero");
    if (per_hour_ < 0) throw std::invalid_argument("negative dimensions are not allowed");
    const float s = f32(scale_);
    std::vector<std::int16_t> buf;
    std::int64_t i = 0;
    while (i < n) {
        const auto [hour, k] = floor_divmod(a + i, per_hour_);
        const std::int64_t m = std::min(n - i, per_hour_ - k);
        std::vector<Range> rs;
        std::shared_ptr<Hour> h;
        {
            std::lock_guard lk(mu_);
            rs = ranges_locked(hour);
            const bool any = std::any_of(rs.begin(), rs.end(), [&](const Range& r) {
                return std::max(r[0], k) < std::min(r[1], k + m);
            });
            if (any) h = open_hour_locked(hour);
        }
        if (h) {
            for (const Range& r : rs) {
                const std::int64_t lo = std::max(r[0], k), hi = std::min(r[1], k + m);
                if (lo >= hi) continue;
                buf.resize(static_cast<std::size_t>(2 * (hi - lo)));
                h->read(lo, hi, buf.data());
                const std::size_t at = static_cast<std::size_t>(i + lo - k);
                for (std::int64_t j = 0; j < hi - lo; ++j) {
                    out.x[at + j] = dequantise(buf[2 * j], buf[2 * j + 1], s);
                    out.mask[at + j] = 1;
                }
            }
        }
        i += m;
    }
    return out;
}

Written PassbandStore::write_capture(const frontend::Capture& cap, bool expire_) {
    if (cap.fs != fs_)
        throw std::invalid_argument("the store holds " + std::to_string(fs_) + " Hz FE, not a " +
                                    std::to_string(cap.fs) + " Hz capture");
    // builtin max(max|re|, max|im|), each np.max(initial=0.0) in float32 (NaN propagating)
    float pre = 0.0f, pim = 0.0f;
    for (const cf& v : cap.fe) {
        const float ar = std::fabs(v.real()), ai = std::fabs(v.imag());
        if (std::isnan(ar) || ar > pre) pre = std::isnan(pre) ? pre : ar;
        if (std::isnan(ai) || ai > pim) pim = std::isnan(pim) ? pim : ai;
    }
    const float best = (pim > pre) ? pim : pre;   // max(a, b): b only when b > a
    const double peak = static_cast<double>(best) * scale_;
    double gain = 1.0;
    std::vector<cf> scaled;
    std::span<const cf> x(cap.fe);
    if (peak > 16383.0) {
        const double l = std::ceil(std::log2(peak / 16383.0));
        if (std::isinf(l)) throw std::overflow_error("cannot convert float infinity to integer");
        gain = std::ldexp(1.0, -static_cast<int>(std::min(l, 4000.0)));
        const float g = f32(gain);
        scaled.resize(cap.fe.size());
        for (std::size_t i = 0; i < cap.fe.size(); ++i) {
            // complex64 * float32: the textbook product with (g + 0j)
            const float a = cap.fe[i].real(), b = cap.fe[i].imag();
            scaled[i] = cf(a * g - b * 0.0f, a * 0.0f + b * g);
        }
        x = scaled;
    }
    const std::int64_t n0 =
        sub(mul(slot_t0(cap.q), fs_, "sample index"), py_round(cap.t0_index), "sample index");
    write_at(n0, x, expire_);
    return {static_cast<double>(n0) / static_cast<double>(fs_), gain};
}

SlotCapture PassbandStore::capture(std::int64_t q, double dur_s, double lead_s, double tail_s) const {
    const std::int64_t t0 = slot_t0(q);
    const double fs = static_cast<double>(fs_);
    const std::int64_t base = mul(t0, fs_, "sample index");
    const std::int64_t a = sub(base, py_round(lead_s * fs), "sample index");
    const std::int64_t b = add(base, py_round((dur_s + tail_s) * fs), "sample index");
    if (fs_ == 0) throw ZeroDivision("division by zero");
    Masked r = read(static_cast<double>(a) / fs, static_cast<double>(b) / fs);
    SlotCapture out;
    const auto first = std::find(r.mask.begin(), r.mask.end(), std::uint8_t{1});
    if (first == r.mask.end()) return out;
    const std::int64_t len = static_cast<std::int64_t>(r.mask.size());
    const std::int64_t k0 = base - a;
    const std::int64_t k1 = std::min(len, add(k0, py_round(dur_s * fs), "sample index"));
    if (k1 > k0) {
        // mask[k0:k1] with Python's slice bounds (a negative start counts from the end)
        auto norm = [len](std::int64_t v) { return v < 0 ? std::max<std::int64_t>(0, v + len) : std::min(v, len); };
        const std::int64_t s0 = norm(k0), s1 = norm(k1);
        if (s1 > s0) {
            const auto c = std::count(r.mask.begin() + s0, r.mask.begin() + s1, std::uint8_t{1});
            out.coverage = static_cast<double>(c) / static_cast<double>(s1 - s0);
        } else {
            out.coverage = std::numeric_limits<double>::quiet_NaN();   // np.mean of an empty slice
        }
    }
    const std::int64_t i = first - r.mask.begin();
    const std::int64_t j = len - (std::find(r.mask.rbegin(), r.mask.rend(), std::uint8_t{1}) - r.mask.rbegin());
    frontend::Capture c;
    c.fe.assign(r.x.begin() + i, r.x.begin() + j);
    c.q = q;
    c.t0_index = static_cast<double>(k0 - i);
    c.fs = fs_;
    out.capture = std::move(c);
    return out;
}

std::vector<std::int64_t> PassbandStore::slots(double dur_s) const {
    std::set<std::int64_t> out;
    for (const std::int64_t hour : hours()) {
        const std::int64_t h0 = mul(hour, 3600, "hour");
        const double lo = static_cast<double>(h0) - dur_s - 1.0;
        const std::int64_t hi = add(h0, 3600, "hour");
        const std::int64_t qlo = to_int(std::floor(lo / 900.0));
        const std::int64_t qhi = to_int(std::ceil(static_cast<double>(hi) / 900.0));
        for (std::int64_t q = qlo; q <= qhi; ++q) {
            const std::int64_t t = slot_t0(q);
            if (static_cast<double>(t) + dur_s > static_cast<double>(h0) && t < hi) out.insert(q);
        }
    }
    return {out.begin(), out.end()};
}

std::vector<std::int64_t> PassbandStore::hours() const {
    std::error_code ec;
    const std::vector<std::string> names = util::list_dir(root_, ec);
    if (ec) throw FileError(ec, root_);
    std::set<std::int64_t> out;
    for (const std::string& n : names)
        if (const auto h = hour_of_name(n)) out.insert(*h);
    return {out.begin(), out.end()};
}

std::vector<std::int64_t> PassbandStore::raw_hours() const {
    std::vector<std::int64_t> out;
    for (const std::int64_t h : hours())
        if (util::exists(raw_path(h))) out.push_back(h);
    return out;
}

std::vector<std::int64_t> PassbandStore::expire(std::optional<double> when) {
    const double t = when ? *when : now();
    const double cut = t - keep_hours_ * 3600.0;
    std::vector<std::int64_t> gone;
    std::lock_guard lk(mu_);
    for (const std::int64_t hour : hours()) {
        // (hour + 1) * 3600 <= cut, an int against a float (exact while |hour| < 2^53 / 3600)
        const std::int64_t end = mul(add(hour, 1, "hour"), 3600, "hour");
        if (static_cast<double>(end) <= cut) {
            remove_missing_ok(raw_path(hour));
            remove_missing_ok(index_path(hour));
            remove_missing_ok(flac_path(hour));
            gen_.erase(hour);
            gone.push_back(hour);
        }
    }
    return gone;
}

bool PassbandStore::pack_hour(std::int64_t hour) {
    std::lock_guard pk(pack_mu_);
    const std::string raw = raw_path(hour);
    if (!util::exists(raw)) return false;
    const std::string tmp =
        join(root_, hour_name(hour, ".flac") + "." + std::to_string(rawfile::process_id()) + ".tmp");
    bool ok = false;
    try {
        std::error_code ec;
        const auto before = rawfile::mtime_ns(raw, ec);
        if (!before) throw FileError(ec, raw);
        std::uint64_t gen0 = 0;
        {
            std::lock_guard lk(mu_);
            const auto g = gen_.find(hour);
            gen0 = g == gen_.end() ? 0 : g->second;
        }
        rawfile::RawFile f = rawfile::RawFile::open(raw, rawfile::Mode::Read, ec);
        if (!f.is_open()) throw FileError(ec, raw);
        const auto size = f.size(ec);
        if (!size) throw FileError(ec, raw);
        const std::uint64_t per = static_cast<std::uint64_t>(per_hour_);
        if (*size < per * 4) short_file(raw);
        const flac::Source source = [&](std::uint64_t first, std::size_t n, std::int16_t* out) {
            std::error_code e;
            return f.read_at(first * 4, out, n * 4, e) == n * 4 && !e;
        };
        flac::encode_file(tmp, fs_, 2, per, source, {}, &closing_);
        // decode it back and compare with the raw file, sample for sample
        bool same = false;
        {
            flac::Reader r(tmp);
            const flac::Info& info = r.info();
            same = info.channels == 2 && info.frames == per && info.bits_per_sample == 16;
            std::vector<std::int16_t> mine;
            for (std::uint64_t first = 0; same && first < per; first += CHUNK) {
                if (closing_.load()) throw flac::Cancelled();
                const std::uint64_t n = std::min<std::uint64_t>(CHUNK, per - first);
                const std::vector<std::int16_t> y = r.read(first, n);
                mine.resize(static_cast<std::size_t>(2 * n));
                if (!source(first, static_cast<std::size_t>(n), mine.data())) throw FileError(ec, raw);
                same = y.size() == mine.size() && std::equal(y.begin(), y.end(), mine.begin());
            }
        }
        f.close();
        std::lock_guard lk(mu_);
        if (same) {
            const auto after = rawfile::mtime_ns(raw, ec);
            if (!after) throw FileError(ec, raw);
            const auto g = gen_.find(hour);
            const std::uint64_t gen1 = g == gen_.end() ? 0 : g->second;
            if (*after == *before && gen1 == gen0) {   // (else written to meanwhile: packed again later)
                if (!util::replace_file(tmp, flac_path(hour), ec)) throw FileError(ec, tmp);
                if (!util::remove_file(raw, ec)) throw FileError(ec, raw);
                ok = true;
            }
        }
    } catch (const std::system_error&) {
    } catch (const std::runtime_error&) {   // (a FLAC failure, Cancelled, an overflow)
    } catch (const std::invalid_argument&) {
    }
    std::error_code ignore;
    util::remove_file(tmp, ignore);
    return ok;
}

void PassbandStore::maybe_pack(double when) {
    std::lock_guard lk(packer_mu_);
    if (!pack_ || when < pack_check_ || packer_running_ || closing_.load()) return;
    pack_check_ = when + PB_PACK_CHECK_S;
    std::vector<std::int64_t> due;
    for (const std::int64_t h : raw_hours())
        if (static_cast<double>((h + 1) * 3600) + PB_PACK_AFTER_S <= when) due.push_back(h);
    if (due.empty()) return;
    if (packer_.joinable()) packer_.join();   // (finished: packer_running_ is false)
    packer_running_ = true;
    packer_started_ = true;
    packer_ = std::thread([this, due = std::move(due)] {
        for (const std::int64_t h : due) {
            if (closing_.load()) break;
            try {
                pack_hour(h);
            } catch (...) {   // (nothing may escape a thread; pack_hour reports by returning false)
            }
        }
        {
            std::lock_guard l(packer_mu_);
            packer_running_ = false;
        }
        packer_cv_.notify_all();
    });
}

bool PassbandStore::packer_started() const {
    std::lock_guard lk(packer_mu_);
    return packer_started_;
}

bool PassbandStore::packer_alive() const {
    std::lock_guard lk(packer_mu_);
    return packer_running_;
}

bool PassbandStore::join_packer(std::optional<double> timeout_s) {
    std::unique_lock lk(packer_mu_);
    if (!timeout_s) {
        packer_cv_.wait(lk, [this] { return !packer_running_; });
        return true;
    }
    const double t = std::max(0.0, std::isnan(*timeout_s) ? 0.0 : std::min(*timeout_s, 1e9));
    return packer_cv_.wait_for(lk, std::chrono::duration<double>(t), [this] { return !packer_running_; });
}

}  // namespace sstvae::qrss::passband
