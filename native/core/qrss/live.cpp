// The live listener's pieces below the listener (live.hpp).

#include "qrss/live.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "math/errors.hpp"
#include "qrss/frontend.hpp"
#include "qrss/pynum.hpp"
#include "util/fileio.hpp"

namespace sstvae::qrss::live {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr std::int64_t FE_FS = config::FE_FS;

// Python's floor modulo of an int by a positive int.
std::int64_t floor_mod(std::int64_t a, std::int64_t n) {
    const std::int64_t r = a % n;
    return r < 0 ? r + n : r;
}

// int(round(v)) for a float: half to even. ValueError / OverflowError as
// Python's round() raises them; past int64 (where Python's int grows) an
// OverflowError too.
std::int64_t py_round_int(double v) {
    if (std::isnan(v)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(v)) throw std::overflow_error("cannot convert float infinity to integer");
    const double r = std::nearbyint(v);
    if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0))
        throw std::overflow_error("int too large to convert to int64");
    return static_cast<std::int64_t>(r);
}

// int(math.ceil(v)), clamped far outside any array's range: callers only
// compare it with 0 and a length, so the clamp keeps every comparison.
std::int64_t ceil_clamped(double v) {
    if (std::isnan(v)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(v)) throw std::overflow_error("cannot convert float infinity to integer");
    const double c = std::ceil(v);
    constexpr double big = 4.0e18;
    if (c > big) return static_cast<std::int64_t>(big);
    if (c < -big) return -static_cast<std::int64_t>(big);
    return static_cast<std::int64_t>(c);
}

// mean(ok[lo:hi]) as numpy computes it for bools (a count over n, exact);
// NaN for an empty slice, as np.mean of one is.
double mean_of(std::span<const std::uint8_t> ok, std::int64_t lo, std::int64_t hi) {
    if (hi <= lo) return std::numeric_limits<double>::quiet_NaN();
    std::int64_t c = 0;
    for (std::int64_t i = lo; i < hi; ++i) c += ok[static_cast<std::size_t>(i)] ? 1 : 0;
    return static_cast<double>(c) / static_cast<double>(hi - lo);
}

bool any_of(std::span<const std::uint8_t> ok, std::int64_t lo, std::int64_t hi) {
    for (std::int64_t i = lo; i < hi; ++i)
        if (ok[static_cast<std::size_t>(i)]) return true;
    return false;
}

// days since 1970-01-01 -> (year, month, day), proleptic Gregorian
// (H. Hinnant's civil_from_days), for any int64 day count a time_t gives.
struct Civil {
    std::int64_t y;
    int m;
    int d;
};
Civil civil_from_days(std::int64_t z) {
    z += 719468;
    const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const std::int64_t doe = z - era * 146097;
    const std::int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const std::int64_t y = yoe + era * 400;
    const std::int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const std::int64_t mp = (5 * doy + 2) / 153;
    const int d = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
    const int m = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
    return {y + (m <= 2 ? 1 : 0), m, d};
}

std::string two(int v) {
    std::string s = std::to_string(v);
    return v < 10 ? "0" + s : s;
}

// --- the dial file's text ---------------------------------------------------------

void put_replacement(std::string& out) { out += "\xEF\xBF\xBD"; }

bool cont(unsigned char c, unsigned char lo = 0x80, unsigned char hi = 0xBF) { return c >= lo && c <= hi; }

// bytes.decode("utf-8", errors="replace"), as UTF-8: each maximal invalid
// subpart one U+FFFD (Unicode's recommended practice, which CPython follows).
std::string decode_replace(std::string_view in) {
    std::string out;
    out.reserve(in.size());
    const std::size_t n = in.size();
    std::size_t i = 0;
    auto at = [&](std::size_t k) { return static_cast<unsigned char>(in[k]); };
    while (i < n) {
        const unsigned char c = at(i);
        if (c < 0x80) {
            out += static_cast<char>(c);
            ++i;
            continue;
        }
        int need = 0;
        unsigned char lo = 0x80, hi = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) {
            need = 1;
        } else if (c == 0xE0) {
            need = 2;
            lo = 0xA0;
        } else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) {
            need = 2;
        } else if (c == 0xED) {
            need = 2;
            hi = 0x9F;
        } else if (c == 0xF0) {
            need = 3;
            lo = 0x90;
        } else if (c >= 0xF1 && c <= 0xF3) {
            need = 3;
        } else if (c == 0xF4) {
            need = 3;
            hi = 0x8F;
        } else {
            put_replacement(out);
            ++i;
            continue;
        }
        std::size_t k = i + 1;
        int got = 0;
        bool ok = true;
        while (got < need) {
            if (k >= n || !cont(at(k), got == 0 ? lo : 0x80, got == 0 ? hi : 0xBF)) {
                ok = false;
                break;
            }
            ++k;
            ++got;
        }
        if (ok) {
            out.append(in.substr(i, k - i));
        } else {
            put_replacement(out);   // the lead and the continuations that fit: one subpart
        }
        i = k;
    }
    return out;
}

// str.splitlines() over UTF-8 text: \n, \r, \r\n, \v, \f, \x1c, \x1d,
// \x1e, U+0085, U+2028, U+2029 end a line; no empty last line.
std::vector<std::string_view> split_lines(std::string_view s) {
    std::vector<std::string_view> out;
    const std::size_t n = s.size();
    std::size_t start = 0, i = 0;
    auto at = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    while (i < n) {
        const unsigned char c = at(i);
        std::size_t len = 0;
        if (c == '\n' || c == 0x0B || c == 0x0C || c == 0x1C || c == 0x1D || c == 0x1E) {
            len = 1;
        } else if (c == '\r') {
            len = (i + 1 < n && s[i + 1] == '\n') ? 2 : 1;
        } else if (c == 0xC2 && i + 1 < n && at(i + 1) == 0x85) {
            len = 2;
        } else if (c == 0xE2 && i + 2 < n && at(i + 1) == 0x80 && (at(i + 2) == 0xA8 || at(i + 2) == 0xA9)) {
            len = 3;
        }
        if (len) {
            out.push_back(s.substr(start, i - start));
            i += len;
            start = i;
        } else {
            ++i;
        }
    }
    if (start < n) out.push_back(s.substr(start));
    return out;
}

// float(v) of a JSON value: a number, a bool, or a numeric string; nullopt
// where float() raises (ValueError, TypeError).
std::optional<double> as_float(const pyjson::Value& v) {
    using K = pyjson::Value::Kind;
    switch (v.kind()) {
        case K::Bool:
            return v.as_bool() ? 1.0 : 0.0;
        case K::Int:
            return static_cast<double>(v.as_int());
        case K::Float:
            return v.as_double();
        case K::String:
            return pynum::py_float(v.as_string());
        default:
            return std::nullopt;
    }
}

std::string type_of(const pyjson::Value& v) {
    using K = pyjson::Value::Kind;
    switch (v.kind()) {
        case K::Null:
            return "NoneType";
        case K::Array:
            return "list";
        case K::Object:
            return "dict";
        case K::Bool:
            return "bool";
        case K::Int:
            return "int";
        case K::Float:
            return "float";
        default:
            return "str";
    }
}

// --- the ring's storage, in contiguous runs -------------------------------------------

// fn(ring position, offset into the run, count) over [lo, hi) mod n.
template <class F>
void each_run(std::int64_t lo, std::int64_t hi, std::int64_t n, F&& fn) {
    std::int64_t k = lo;
    while (k < hi) {
        const std::int64_t pos = floor_mod(k, n);
        const std::int64_t len = std::min(hi - k, n - pos);
        fn(pos, k - lo, len);
        k += len;
    }
}

}  // namespace

// --- small things ---------------------------------------------------------------------------

std::int64_t slot_t0(std::int64_t q) { return config::SLOT_S * q + config::T0_OFFSET_S; }

std::string utc_iso(double t) {
    if (std::isnan(t)) throw std::invalid_argument("Invalid value NaN (not a number)");
    // CPython's _PyTime_DoubleToDenominator(t, 1e6, ROUND_HALF_EVEN)
    double intpart = 0.0;
    double frac = std::modf(t, &intpart);
    frac *= 1e6;
    double r = std::round(frac);
    if (std::fabs(frac - r) == 0.5) r = 2.0 * std::round(frac / 2.0);
    if (r >= 1e6) {
        intpart += 1.0;
    } else if (r < 0) {
        intpart -= 1.0;
    }
    if (!(intpart >= -9223372036854775808.0 && intpart < 9223372036854775808.0))
        throw std::overflow_error("timestamp out of range for platform time_t");
    return utc_iso_int(static_cast<std::int64_t>(intpart));
}

std::string utc_iso_int(std::int64_t secs) {
    std::int64_t days = secs / 86400;
    std::int64_t sod = secs % 86400;
    if (sod < 0) {
        sod += 86400;
        --days;
    }
    const Civil c = civil_from_days(days);
    // gmtime's struct tm holds year - 1900 in an int
    if (c.y - 1900 > INT_MAX || c.y - 1900 < INT_MIN)
        throw std::system_error(std::make_error_code(std::errc::value_too_large));
    if (c.y < 1 || c.y > 9999) throw std::invalid_argument("year " + std::to_string(c.y) + " is out of range");
    const int hh = static_cast<int>(sod / 3600), mm = static_cast<int>(sod / 60 % 60), ss = static_cast<int>(sod % 60);
    return std::to_string(c.y) + "-" + two(c.m) + "-" + two(c.d) + "T" + two(hh) + ":" + two(mm) + ":" + two(ss) + "Z";
}

double frame_seconds(const frame::FrameSpec& spec) { return spec.keyed_end_pos() * spec.t_sym(); }

double frame_seconds(Selection specs) {
    if (specs.empty()) throw std::invalid_argument("max() iterable argument is empty");
    double best = frame_seconds(specs[0]);
    for (std::size_t i = 1; i < specs.size(); ++i) {
        const double s = frame_seconds(specs[i]);
        if (s > best) best = s;   // max keeps the first of equals
    }
    return best;
}

bool in_band(double f_hz, int waveform) {
    const double lo = waveform == config::WAVEFORM_L ? f_hz - L_BELOW_HZ : f_hz;
    return config::CARRIER_BAND_HZ[0] <= lo && f_hz <= config::CARRIER_BAND_HZ[1];
}

double centre_of(double f_hz, int waveform) {
    return waveform == config::WAVEFORM_L ? f_hz + receiver::L_EXCLUDE_OFFSET_HZ : f_hz;
}

int waveform_id(std::string_view name) {
    for (const auto& nv : rxconfig::live::WAVEFORM_OF_)
        if (nv.name == name) return static_cast<int>(nv.value);
    return config::WAVEFORM_CE;
}

std::int64_t waveform_id(const pyjson::Value& w) {
    using K = pyjson::Value::Kind;
    switch (w.kind()) {
        case K::String:
            return waveform_id(w.as_string());
        case K::Bool:
            return w.as_bool() ? 1 : 0;
        case K::Int:
            return w.as_int();
        case K::Float: {
            const double f = w.as_double();
            if (std::isnan(f)) throw std::invalid_argument("cannot convert float NaN to integer");
            if (std::isinf(f)) throw std::overflow_error("cannot convert float infinity to integer");
            const double t = std::trunc(f);
            if (!(t >= -9223372036854775808.0 && t < 9223372036854775808.0))
                throw std::overflow_error("int too large to convert to int64");
            return static_cast<std::int64_t>(t);
        }
        default:
            throw sstvae::math::TypeError("int() argument must be a string, a bytes-like object or a real number, not '" +
                                  type_of(w) + "'");
    }
}

std::optional<int> frame_waveform(std::string_view name) {
    try {
        return static_cast<int>(frame::get_any(name).waveform());
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

std::int64_t due_waveform(const pyjson::Value& d) {
    if (d.kind() == pyjson::Value::Kind::Object) {
        if (const pyjson::Value* f = d.find("frame"); f && f->kind() == pyjson::Value::Kind::String)
            if (const std::optional<int> w = frame_waveform(f->as_string())) return *w;
        if (const pyjson::Value* w = d.find("waveform")) return waveform_id(*w);
    }
    return config::WAVEFORM_CE;
}

// --- the FE ring ----------------------------------------------------------------------------

FeRing::FeRing(double span_s) {
    const double v = span_s * static_cast<double>(FE_FS);
    if (std::isnan(v)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(v)) throw std::overflow_error("cannot convert float infinity to integer");
    const double t = std::trunc(v);
    if (t < 0) throw std::invalid_argument("negative dimensions are not allowed");
    if (!(t < 9223372036854775808.0)) throw std::overflow_error("int too large to convert to int64");
    n_ = static_cast<std::int64_t>(t);
    x_.assign(static_cast<std::size_t>(n_), cf(0.0f, 0.0f));
    ok_.assign(static_cast<std::size_t>(n_), 0);
}

std::optional<std::int64_t> FeRing::end() const {
    std::shared_lock lock(mu_);
    return end_;
}

void FeRing::clear_locked(std::int64_t a, std::int64_t b) {
    b = std::min(b, a + n_);
    if (b <= a || n_ == 0) return;
    each_run(a, b, n_, [&](std::int64_t pos, std::int64_t, std::int64_t len) {
        std::fill_n(x_.begin() + pos, len, cf(0.0f, 0.0f));
        std::fill_n(ok_.begin() + pos, len, std::uint8_t{0});
    });
}

void FeRing::clear(std::int64_t a, std::int64_t b) {
    std::unique_lock lock(mu_);
    clear_locked(a, b);
}

void FeRing::write(std::int64_t n0, std::span<const cf> x) {
    std::unique_lock lock(mu_);
    const auto len = static_cast<std::int64_t>(x.size());
    if (len > n_) {
        n0 += len - n_;
        // x[-n:]: all of it when n is 0, as Python's x[-0:] is
        if (n_ > 0) x = x.last(static_cast<std::size_t>(n_));
    }
    if (end_ && n0 > *end_) clear_locked(*end_, n0);   // a hole: what was there is older than the ring
    const auto m = static_cast<std::int64_t>(x.size());
    if (m > 0 && n_ == 0) throw std::out_of_range("index 0 is out of bounds for axis 0 with size 0");
    each_run(n0, n0 + m, n_, [&](std::int64_t pos, std::int64_t off, std::int64_t cnt) {
        std::copy_n(x.begin() + off, cnt, x_.begin() + pos);
        std::fill_n(ok_.begin() + pos, cnt, std::uint8_t{1});
    });
    end_ = std::max(end_.value_or(0), n0 + m);
}

FeSpan FeRing::read(std::int64_t a, std::int64_t b) const {
    if (b < a) throw std::invalid_argument("negative dimensions are not allowed");
    FeSpan out;
    out.x.resize(static_cast<std::size_t>(b - a));
    out.ok.resize(static_cast<std::size_t>(b - a));
    read_into(a, out.x, out.ok);
    return out;
}

void FeRing::read_into(std::int64_t a, std::span<cf> x, std::span<std::uint8_t> ok) const {
    if (x.size() != ok.size())
        throw std::invalid_argument("read_into: " + std::to_string(x.size()) + " samples and " +
                                    std::to_string(ok.size()) + " flags");
    std::fill(x.begin(), x.end(), cf(0.0f, 0.0f));
    std::fill(ok.begin(), ok.end(), std::uint8_t{0});
    const std::int64_t b = a + static_cast<std::int64_t>(x.size());
    std::shared_lock lock(mu_);
    if (!end_) return;
    const std::int64_t lo = std::max(a, *end_ - n_), hi = std::min(b, *end_);
    if (hi > lo) {
        each_run(lo, hi, n_, [&](std::int64_t pos, std::int64_t off, std::int64_t cnt) {
            std::copy_n(x_.begin() + pos, cnt, x.begin() + (lo - a) + off);
            std::copy_n(ok_.begin() + pos, cnt, ok.begin() + (lo - a) + off);
        });
    }
}

std::vector<cf> FeRing::samples() const {
    std::shared_lock lock(mu_);
    return x_;
}

std::vector<std::uint8_t> FeRing::written() const {
    std::shared_lock lock(mu_);
    return ok_;
}

Erase spans(std::span<const std::uint8_t> mask, double fs, double origin) {
    Erase out;
    const std::size_t n = mask.size();
    std::size_t i = 0;
    while (i < n) {
        if (mask[i]) {
            ++i;
            continue;
        }
        const std::size_t s = i;
        while (i < n && !mask[i]) ++i;
        out.emplace_back((static_cast<double>(s) - origin) / fs, (static_cast<double>(i) - origin) / fs);
    }
    return out;
}

// --- a slot's capture -------------------------------------------------------------------------

CaptureSpan capture_span(std::int64_t q, double frame_s, double lead_s, double tail_s) {
    const std::int64_t t0 = slot_t0(q);
    CaptureSpan s;
    s.a = t0 * FE_FS - py_round_int(lead_s * static_cast<double>(FE_FS));
    s.b = t0 * FE_FS + py_round_int((frame_s + tail_s) * static_cast<double>(FE_FS));
    s.origin = t0 * FE_FS - s.a;
    return s;
}

SlotRead capture_read(const FeRing& ring, std::int64_t q, Selection sel, double lead_s, double tail_s) {
    const CaptureSpan s = capture_span(q, frame_seconds(sel), lead_s, tail_s);
    FeSpan r = ring.read(s.a, s.b);
    return {std::move(r.x), std::move(r.ok), s.origin};
}

SlotRead capture_read(const FeReader& reader, std::int64_t q, Selection sel, double lead_s, double tail_s) {
    const CaptureSpan s = capture_span(q, frame_seconds(sel), lead_s, tail_s);
    FeSpan r = reader(s.a, s.b);
    const auto want = static_cast<std::size_t>(s.b - s.a);
    if (r.x.size() != want || r.ok.size() != want)
        throw std::invalid_argument("a reader returned " + std::to_string(r.x.size()) + " samples and " +
                                    std::to_string(r.ok.size()) + " flags for " + std::to_string(want));
    return {std::move(r.x), std::move(r.ok), s.origin};
}

std::vector<cf> default_noise(std::int64_t seed, std::size_t n) {
    // splitmix64 seeds xoshiro256**
    std::uint64_t sm = static_cast<std::uint64_t>(seed);
    auto splitmix = [&sm]() {
        std::uint64_t z = (sm += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    std::uint64_t s[4] = {splitmix(), splitmix(), splitmix(), splitmix()};
    auto rotl = [](std::uint64_t v, int k) { return (v << k) | (v >> (64 - k)); };
    auto next = [&]() {
        const std::uint64_t result = rotl(s[1] * 5, 7) * 9;
        const std::uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    };
    const double scale = std::sqrt(0.5 / std::log(2.0));
    constexpr double two_pi = 6.283185307179586;
    std::vector<cf> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double u1 = 1.0 - static_cast<double>(next() >> 11) * 0x1.0p-53;   // (0, 1]
        const double u2 = static_cast<double>(next() >> 11) * 0x1.0p-53;         // [0, 1)
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double th = two_pi * u2;
        const double re = r * std::cos(th);
        const double im = r * std::sin(th);
        out[i] = cf(static_cast<float>(scale * re), static_cast<float>(scale * im));
    }
    return out;
}

SlotCapture capture_prepare(std::span<const cf> x, std::span<const std::uint8_t> ok, std::int64_t origin,
                            std::int64_t q, double frame_s, std::optional<std::int64_t> seed, const PrepareSeams& seams) {
    if (x.size() != ok.size())
        throw std::invalid_argument("x has " + std::to_string(x.size()) + " samples and ok " +
                                    std::to_string(ok.size()));
    const std::size_t n = x.size();
    // The blanker and the inverse AGC look at most about 0.6 s either side of
    // a sample, and everything after the newest heard sample becomes noise,
    // so they run only up to CAPTURE_MARGIN_S past it (the reference's
    // reasoning: the same capture, bit for bit).
    std::size_t h = 0;
    for (std::size_t i = n; i > 0; --i)
        if (ok[i - 1]) {
            h = i;
            break;
        }
    const auto margin = static_cast<std::size_t>(CAPTURE_MARGIN_S * static_cast<double>(FE_FS));
    const std::size_t m = std::min(n, h + margin);
    std::vector<cf> fe(n, cf(0.0f, 0.0f));
    std::vector<float> keep(n, 1.0f);
    {
        frontend::Blanked b = seams.blank ? seams.blank(x.first(m)) : frontend::blank(x.first(m));
        if (b.fe.size() != m || b.keep.size() != m)
            throw std::invalid_argument("blank gave " + std::to_string(b.fe.size()) + " samples for " +
                                        std::to_string(m));
        std::copy(b.fe.begin(), b.fe.end(), fe.begin());
        std::copy(b.keep.begin(), b.keep.end(), keep.begin());
    }
    for (std::size_t i = 0; i < n; ++i) keep[i] = keep[i] * (ok[i] ? 1.0f : 0.0f);   // keep * ok, float32
    {
        const std::span<const cf> fe_m = std::span<const cf>(fe).first(m);
        const std::span<const float> keep_m = std::span<const float>(keep).first(m);
        frontend::Normalised nm = seams.normalise ? seams.normalise(fe_m, keep_m) : frontend::normalise(fe_m, keep_m);
        if (nm.fe.size() != m)
            throw std::invalid_argument("normalise gave " + std::to_string(nm.fe.size()) + " samples for " +
                                        std::to_string(m));
        std::copy(nm.fe.begin(), nm.fe.end(), fe.begin());
    }
    std::size_t nmiss = 0;
    for (std::size_t i = 0; i < n; ++i) nmiss += ok[i] ? 0 : 1;
    if (nmiss) {
        const std::int64_t sd = seed.value_or(q);
        const std::vector<cf> z = seams.noise ? seams.noise(sd, nmiss) : default_noise(sd, nmiss);
        if (z.size() != nmiss)
            throw std::invalid_argument("the noise gave " + std::to_string(z.size()) + " samples for " +
                                        std::to_string(nmiss));
        std::size_t j = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (!ok[i]) {
                fe[i] = z[j++];
                keep[i] = 1.0f;
            }
    }
    SlotCapture out;
    out.prep.fe = std::move(fe);
    out.prep.keep = std::move(keep);
    out.prep.q = q;
    out.prep.t0_index = static_cast<double>(origin);
    out.prep.fs = FE_FS;
    out.erase = spans(ok, static_cast<double>(FE_FS), static_cast<double>(origin));
    const auto len = static_cast<std::int64_t>(n);
    const std::int64_t k1 = std::min(len, origin + py_round_int(frame_s * static_cast<double>(FE_FS)));
    if (k1 > origin) {
        const auto [lo, hi] = py_slice(len, origin, k1);
        out.heard = mean_of(ok, lo, hi);
    } else {
        out.heard = 0.0;
    }
    return out;
}

SlotCapture capture_prepare(std::span<const cf> x, std::span<const std::uint8_t> ok, std::int64_t origin,
                            std::int64_t q, Selection sel, std::optional<std::int64_t> seed, const PrepareSeams& seams) {
    return capture_prepare(x, ok, origin, q, frame_seconds(sel), seed, seams);
}

SlotCapture slot_capture(const FeRing& ring, std::int64_t q, Selection sel, std::optional<std::int64_t> seed,
                         double lead_s, double tail_s, const PrepareSeams& seams) {
    const SlotRead r = capture_read(ring, q, sel, lead_s, tail_s);
    return capture_prepare(r.x, r.ok, r.origin, q, sel, seed, seams);
}

// --- the end-of-slot receive ----------------------------------------------------------------------

std::pair<std::int64_t, std::int64_t> py_slice(std::int64_t n, std::int64_t start, std::int64_t stop) {
    auto norm = [n](std::int64_t i) {
        if (i < 0) {
            i += n;
            return i < 0 ? std::int64_t{0} : i;
        }
        return i > n ? n : i;
    };
    const std::int64_t lo = norm(start), hi = norm(stop);
    return {lo, std::max(lo, hi)};
}

FinishGate finish_gate(std::span<const std::uint8_t> ok, std::int64_t origin, double frame_s, bool have_dets,
                       bool have_hints, double min_heard) {
    const auto n = static_cast<std::int64_t>(ok.size());
    FinishGate g;
    g.k1 = std::min(n, origin + py_round_int(frame_s * static_cast<double>(FE_FS)));
    const auto [lo, hi] = py_slice(n, origin, g.k1);
    const bool nothing = !any_of(ok, lo, hi);
    if (nothing || (!have_dets && !have_hints && mean_of(ok, lo, hi) < min_heard)) {
        g.receive = false;
        g.heard = g.k1 > origin ? mean_of(ok, lo, hi) : 0.0;
        return g;
    }
    g.receive = true;
    return g;
}

FinishBranch finish_branch(double heard, bool have_dets, bool have_hints, double min_heard) {
    if (heard >= min_heard) return FinishBranch::Search;
    if ((have_dets || have_hints) && heard > 0.0) return FinishBranch::Known;
    return FinishBranch::Nothing;
}

api::ReceiveConfig finish_config(std::vector<frame::FrameSpec> frames, std::string estimator) {
    api::ReceiveConfig c;
    c.frames = std::move(frames);
    c.estimator = std::move(estimator);
    c.max_candidates = FINISH_MAX_CANDIDATES;
    c.budget_s = FINISH_BUDGET_S;
    return c;
}

std::vector<double> known_hz(std::span<const types::Detection> dets, bool joint) {
    std::vector<double> out;
    out.reserve(dets.size());
    for (const types::Detection& d : dets) out.push_back(joint ? receiver::centre_hz(d) : d.f_hz);
    return out;
}

FinishResult finish_receive(std::span<const cf> x, std::span<const std::uint8_t> ok, std::int64_t origin,
                            std::int64_t q, const api::ReceiveConfig& config, std::span<const types::Detection> dets,
                            std::span<const acquire::Hint> hints, const FinishOptions& opt) {
    const double frame_s = frame_seconds(config.frames);
    const FinishGate g = finish_gate(ok, origin, frame_s, !dets.empty(), !hints.empty(), opt.min_heard);
    FinishResult out;
    if (!g.receive) {
        out.heard = g.heard;   // nothing to receive
        return out;
    }
    SlotCapture cap = capture_prepare(x, ok, origin, q, frame_s, opt.seed, opt.seams);
    out.heard = cap.heard;
    switch (finish_branch(cap.heard, !dets.empty(), !hints.empty(), opt.min_heard)) {
        case FinishBranch::Search: {
            api::ReceiveConfig c = config;
            c.live_only = false;
            c.erase_s = cap.erase;
            c.round_b = true;
            c.hints.assign(hints.begin(), hints.end());
            out.passes = api::receive_slot(cap.prep, c);
            break;
        }
        case FinishBranch::Known: {
            std::vector<types::Detection> all(dets.begin(), dets.end());
            if (!hints.empty()) {
                api::ReceiveConfig c = config;
                c.live_only = true;
                c.erase_s = cap.erase;
                c.hints.assign(hints.begin(), hints.end());
                const std::vector<double> known = known_hz(dets, receiver::is_joint(config.frames));
                const std::vector<types::Detection> more = api::detect(cap.prep, c, known);
                all.insert(all.end(), more.begin(), more.end());
            }
            api::ReceiveConfig c = config;
            c.live_only = false;
            c.erase_s = cap.erase;
            c.round_b = true;
            c.max_candidates.reset();
            c.budget_s.reset();
            c.hints.clear();
            out.passes = api::receive_slot(cap.prep, c, std::move(all));
            break;
        }
        case FinishBranch::Nothing:
            break;
    }
    return out;
}

// --- the radio's dial ---------------------------------------------------------------------------------

std::vector<DialPoint> parse_dial_text(std::string_view bytes) {
    const std::string text = decode_replace(bytes);
    std::vector<DialPoint> pts;
    for (const std::string_view line : split_lines(text)) {
        pyjson::Value d;
        try {
            d = pyjson::loads(line);
        } catch (const std::exception&) {
            continue;   // json's ValueError (and an int past int64, which this reader cannot hold)
        }
        if (d.kind() != pyjson::Value::Kind::Object) continue;   // d["t"] on a list or a number: TypeError
        const pyjson::Value* tv = d.find("t");
        const pyjson::Value* hv = d.find("dial_hz");
        if (!tv) continue;   // KeyError
        const std::optional<double> t = as_float(*tv);
        if (!t || !hv) continue;
        const std::optional<double> hzf = as_float(*hv);
        if (!hzf || std::isnan(*hzf)) continue;   // round(nan): ValueError
        if (std::isinf(*hzf)) continue;           // round(inf): OverflowError out of the reference's refresh
        const double r = std::nearbyint(*hzf);
        if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0)) continue;   // past int64
        const auto hz = static_cast<std::int64_t>(r);
        if (std::isfinite(*t) && hz > 0) pts.push_back({*t, hz});
    }
    std::stable_sort(pts.begin(), pts.end(), [](const DialPoint& a, const DialPoint& b) { return a.t < b.t; });
    std::vector<DialPoint> out;
    for (const DialPoint& p : pts)
        if (out.empty() || out.back().dial_hz != p.dial_hz) out.push_back(p);
    return out;
}

DialLog::DialLog(std::string path) : path_(std::move(path)) {}

void DialLog::refresh() {
    std::lock_guard refresh_lock(refresh_mu_);
    const std::optional<std::filesystem::path> fp = util::fs_path(path_);
    if (!fp) throw std::invalid_argument("embedded null byte");   // Path.stat()'s ValueError
    std::error_code ec;
    auto clear = [this] {
        std::lock_guard lock(mu_);
        sig_.reset();
        points_.clear();
    };
    const std::filesystem::file_status st = std::filesystem::status(*fp, ec);
    if (ec || !std::filesystem::exists(st)) return clear();
    const auto mtime = std::filesystem::last_write_time(*fp, ec);
    if (ec) return clear();
    std::uintmax_t size = 0;
    if (std::filesystem::is_regular_file(st)) {
        size = std::filesystem::file_size(*fp, ec);
        if (ec) return clear();
    }
    const Signature sig{static_cast<std::int64_t>(mtime.time_since_epoch().count()), size};
    {
        std::lock_guard lock(mu_);
        if (sig_ && *sig_ == sig) return;
    }
    std::string bytes;
    {
        util::File f = util::open_file(path_, "rb");
        if (!f) return;   // read_text's OSError: keep what was read before
        char buf[1 << 15];
        std::size_t got = 0;
        while ((got = std::fread(buf, 1, sizeof buf, f.get())) > 0) bytes.append(buf, got);
        if (std::ferror(f.get())) return;   // a directory, say
    }
    std::vector<DialPoint> pts = parse_dial_text(bytes);
    std::lock_guard lock(mu_);
    sig_ = sig;
    points_ = std::move(pts);
}

std::vector<DialPoint> DialLog::points() const {
    std::lock_guard lock(mu_);
    return points_;
}

void DialLog::set_points(std::vector<DialPoint> points) {
    std::lock_guard lock(mu_);
    points_ = std::move(points);
}

std::vector<DialSpan> DialLog::spans(double a, double b, const DialGuard& guard) const {
    const std::vector<DialPoint> pts = points();
    return dial_spans(pts, a, b, guard);
}

DialEpochs DialLog::by_dial(double a, double b, const DialGuard& guard) const {
    const std::vector<DialPoint> pts = points();
    return dial_by_dial(pts, a, b, guard);
}

std::optional<std::int64_t> DialLog::dial_at(double t, const DialGuard& guard) const {
    return span_at(t, guard).dial;
}

DialSpan DialLog::span_at(double t, const DialGuard& guard) const {
    const std::vector<DialPoint> pts = points();
    return dial_span_at(pts, t, guard);
}

std::vector<DialSpan> dial_spans(std::span<const DialPoint> pts, double a, double b, const DialGuard& guard) {
    if (pts.empty()) return {DialSpan{-INF, INF, std::nullopt}};
    std::vector<DialSpan> out;
    const std::size_t n = pts.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double start = i == 0 ? -INF : pts[i].t;
        const double end = i + 1 < n ? pts[i + 1].t : INF;
        if (end <= a || start >= b) continue;
        const double lo = i ? start + guard[1] : -INF;
        const double hi = i + 1 < n ? end - guard[0] : INF;
        if (hi > lo) out.push_back({lo, hi, pts[i].dial_hz});
    }
    return out;
}

DialEpochs dial_by_dial(std::span<const DialPoint> pts, double a, double b, const DialGuard& guard) {
    DialEpochs out;
    for (const DialSpan& s : dial_spans(pts, a, b, guard)) {
        auto it = std::find_if(out.begin(), out.end(), [&](const auto& e) { return e.first == s.dial; });
        if (it == out.end()) {
            out.emplace_back(s.dial, std::vector<std::pair<double, double>>{});
            it = out.end() - 1;
        }
        it->second.emplace_back(s.lo, s.hi);
    }
    return out;
}

DialSpan dial_span_at(std::span<const DialPoint> pts, double t, const DialGuard& guard) {
    if (pts.empty()) return {-INF, INF, std::nullopt};
    // np.searchsorted(times, t, side="right") - 1, at least 0 (a NaN t
    // sorts after everything, as numpy's does)
    const auto it = std::upper_bound(pts.begin(), pts.end(), t, [](double v, const DialPoint& p) { return v < p.t; });
    const std::size_t found = static_cast<std::size_t>(it - pts.begin());
    const std::size_t i = found == 0 ? 0 : found - 1;
    const double lo = i ? pts[i].t + guard[1] : -INF;
    const double hi = i + 1 < pts.size() ? pts[i + 1].t - guard[0] : INF;
    return {lo, hi, pts[i].dial_hz};
}

bool same_dial(std::optional<std::int64_t> a, std::optional<std::int64_t> b) { return !a || !b || *a == *b; }

std::optional<std::vector<std::pair<std::int64_t, std::int64_t>>> restrict_ranges(
    std::int64_t n, std::int64_t origin, std::int64_t q, std::span<const std::pair<double, double>> spans_) {
    for (const auto& [lo, hi] : spans_)
        if (lo == -INF && hi == INF) return std::nullopt;
    const auto t0 = static_cast<double>(slot_t0(q));
    std::vector<std::pair<std::int64_t, std::int64_t>> out;
    for (const auto& [lo, hi] : spans_) {
        const std::int64_t k_lo = lo == -INF ? 0 : origin + ceil_clamped((lo - t0) * static_cast<double>(FE_FS));
        const std::int64_t k_hi = hi == INF ? n : origin + ceil_clamped((hi - t0) * static_cast<double>(FE_FS));
        const std::int64_t a = std::max<std::int64_t>(0, k_lo), b = std::min(n, k_hi);
        if (b > a) out.emplace_back(a, b);
    }
    return out;
}

std::vector<std::uint8_t> restrict(std::span<const std::uint8_t> ok, std::int64_t origin, std::int64_t q,
                                   std::span<const std::pair<double, double>> spans_) {
    const auto ranges = restrict_ranges(static_cast<std::int64_t>(ok.size()), origin, q, spans_);
    if (!ranges) return std::vector<std::uint8_t>(ok.begin(), ok.end());
    std::vector<std::uint8_t> out(ok.size(), 0);
    for (const auto& [a, b] : *ranges) std::copy(ok.begin() + a, ok.begin() + b, out.begin() + a);
    return out;
}

// --- tiles ------------------------------------------------------------------------------------

namespace {

pyjson::Value num(double v) { return std::isfinite(v) ? pyjson::Value::number(v) : pyjson::Value::null(); }
pyjson::Value num(const std::optional<double>& v) { return v ? num(*v) : pyjson::Value::null(); }
pyjson::Value str(const std::string& s) { return pyjson::Value::string(s); }
pyjson::Value str(const std::optional<std::string>& s) { return s ? str(*s) : pyjson::Value::null(); }
pyjson::Value integer(std::int64_t i) { return pyjson::Value::integer(i); }
pyjson::Value integer(const std::optional<std::int64_t>& i) { return i ? integer(*i) : pyjson::Value::null(); }

using K = pyjson::Value::Kind;

// A field's JSON value, read as Tile(**...) would take it; false: not its type.
bool read(const pyjson::Value& v, std::string& out) {
    if (v.kind() != K::String) return false;
    out = v.as_string();
    return true;
}
bool read(const pyjson::Value& v, std::optional<std::string>& out) {
    if (v.is_null()) {
        out.reset();
        return true;
    }
    std::string s;
    if (!read(v, s)) return false;
    out = std::move(s);
    return true;
}
bool read(const pyjson::Value& v, std::int64_t& out) {
    if (v.kind() == K::Int) {
        out = v.as_int();
        return true;
    }
    if (v.kind() == K::Bool) {
        out = v.as_bool() ? 1 : 0;
        return true;
    }
    return false;
}
bool read(const pyjson::Value& v, std::optional<std::int64_t>& out) {
    if (v.is_null()) {
        out.reset();
        return true;
    }
    std::int64_t i = 0;
    if (!read(v, i)) return false;
    out = i;
    return true;
}
bool read(const pyjson::Value& v, double& out) {
    if (v.kind() != K::Int && v.kind() != K::Float && v.kind() != K::Bool) return false;
    out = v.kind() == K::Bool ? (v.as_bool() ? 1.0 : 0.0) : v.as_double();
    return true;
}
bool read(const pyjson::Value& v, std::optional<double>& out) {
    if (v.is_null()) {
        out.reset();
        return true;
    }
    double d = 0.0;
    if (!read(v, d)) return false;
    out = d;
    return true;
}
bool read(const pyjson::Value& v, std::vector<std::optional<double>>& out) {
    if (v.kind() != K::Array) return false;
    out.clear();
    for (const pyjson::Value& e : v.items()) {
        std::optional<double> d;
        if (!read(e, d)) return false;
        out.push_back(d);
    }
    return true;
}

}  // namespace

pyjson::Value Tile::to_json() const {
    pyjson::Value o = pyjson::Value::object();
    o.set("id", str(id));
    o.set("q", integer(q));
    o.set("slot_utc", str(slot_utc));
    o.set("frame", str(frame));
    o.set("f_hz", num(f_hz));
    o.set("dial_hz", integer(dial_hz));
    o.set("status", str(status));
    o.set("snr_db", num(snr_db));
    o.set("z_ref", num(z_ref));
    o.set("callsign", str(callsign));
    o.set("grid", str(grid));
    o.set("picture_id", str(picture_id));
    o.set("mode", str(mode));
    o.set("segment", integer(segment));
    o.set("cw_text", str(cw_text));
    o.set("progress", num(progress));
    o.set("heard", num(heard));
    o.set("received", num(received));
    o.set("mean_w_db", num(mean_w_db));
    pyjson::Value conf = pyjson::Value::array();
    for (const std::optional<double>& c : confidence) conf.push_back(num(c));
    o.set("confidence", std::move(conf));
    o.set("passes", integer(passes));
    o.set("image", str(image));
    o.set("image_rev", integer(image_rev));
    o.set("updated", num(updated));
    o.set("note", str(note));
    o.set("uid", str(uid));
    o.set("waveform", str(waveform));
    return o;
}

std::optional<Tile> Tile::from_json(const pyjson::Value& v) {
    if (v.kind() != K::Object) return std::nullopt;
    Tile t;
    for (const char* need : {"id", "q", "slot_utc", "frame", "f_hz"})
        if (!v.find(need)) return std::nullopt;   // Tile(**...)'s TypeError: a required field missing
    bool ok = true;
    auto field = [&](const char* name, auto& dst) {
        if (const pyjson::Value* f = v.find(name)) ok = ok && read(*f, dst);
    };
    field("id", t.id);
    field("q", t.q);
    field("slot_utc", t.slot_utc);
    field("frame", t.frame);
    field("f_hz", t.f_hz);
    field("dial_hz", t.dial_hz);
    field("status", t.status);
    field("snr_db", t.snr_db);
    field("z_ref", t.z_ref);
    field("callsign", t.callsign);
    field("grid", t.grid);
    field("picture_id", t.picture_id);
    field("mode", t.mode);
    field("segment", t.segment);
    field("cw_text", t.cw_text);
    field("progress", t.progress);
    field("heard", t.heard);
    field("received", t.received);
    field("mean_w_db", t.mean_w_db);
    field("confidence", t.confidence);
    field("passes", t.passes);
    field("image", t.image);
    field("image_rev", t.image_rev);
    field("updated", t.updated);
    field("note", t.note);
    field("uid", t.uid);
    field("waveform", t.waveform);
    if (!ok) return std::nullopt;
    return t;
}

Tile* SlotState::tile(std::string_view tid) {
    for (auto& [k, t] : tiles)
        if (k == tid) return &t;
    return nullptr;
}

const Tile* SlotState::tile(std::string_view tid) const {
    for (const auto& [k, t] : tiles)
        if (k == tid) return &t;
    return nullptr;
}

std::vector<frame::FrameSpec> LiveConfig::specs() const {
    if (!waveforms) return {spec};
    std::vector<std::int64_t> ws(waveforms->begin(), waveforms->end());
    std::sort(ws.begin(), ws.end());
    ws.erase(std::unique(ws.begin(), ws.end()), ws.end());
    const bool bad = ws.empty() || std::any_of(ws.begin(), ws.end(), [](std::int64_t w) {
                         return w != config::WAVEFORM_CE && w != config::WAVEFORM_L;
                     });
    if (bad) {
        std::string shown = "(";
        for (std::size_t i = 0; i < waveforms->size(); ++i) {
            if (i) shown += ", ";
            shown += std::to_string((*waveforms)[i]);
        }
        shown += waveforms->size() == 1 ? ",)" : ")";
        throw std::invalid_argument("waveforms " + shown + ": one or more of " +
                                    std::to_string(config::WAVEFORM_CE) + ", " + std::to_string(config::WAVEFORM_L));
    }
    std::vector<frame::FrameSpec> out;
    for (const std::int64_t w : ws) out.push_back(frame::twin(spec, static_cast<int>(w)));
    return out;
}

std::vector<frame::FrameSpec> LiveConfig::selection() const {
    std::vector<frame::FrameSpec> sp = specs();
    if (receiver::is_joint(sp)) return sp;
    return {sp[0]};
}

bool LiveConfig::joint() const { return receiver::is_joint(specs()); }

}  // namespace sstvae::qrss::live
