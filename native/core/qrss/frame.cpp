#include "qrss/frame.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>

#include "qrss/sequences.hpp"

namespace sstvae::qrss::frame {

namespace {

using std::int64_t;

constexpr int64_t REF = config::REF_PERIOD;
constexpr std::size_t N_BITS = static_cast<std::size_t>(config::N_HDR_BITS);

// Python's // and its ceiling counterpart -(-a // b), for b > 0.
int64_t floordiv(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}
int64_t ceildiv(int64_t a, int64_t b) { return -floordiv(-a, b); }

std::string str(int64_t v) { return std::to_string(v); }

// The counts the length searches accept, so nothing in them overflows.
void check_count(int64_t n, const char* what) {
    if (n > MAX_COUNT || n < -MAX_COUNT)
        throw std::invalid_argument(std::string(what) + " " + str(n) +
                                    " is too large (|n| <= 2**53 here)");
}

// The smallest D >= start with g(D) >= n, for g nondecreasing in steps of
// 0 or 1 -- what `while g(D) < n: D += 1` returns, found by doubling and
// bisection so that a large count cannot spin.
template <typename G>
int64_t first_at_least(const G& g, int64_t start, int64_t n) {
    if (g(start) >= n) return start;
    int64_t lo = start, step = 1, hi = start + 1;
    while (g(hi) < n) {
        lo = hi;
        step *= 2;
        hi = start + step;
    }
    while (hi - lo > 1) {
        const int64_t mid = lo + (hi - lo) / 2;
        (g(mid) >= n ? hi : lo) = mid;
    }
    return hi;
}

std::string join_names(config::PresetGroup group) {
    std::string out;
    for (const FrameSpec& s : presets(group)) {
        if (!out.empty()) out += ", ";
        out += s.name();
    }
    return out;
}

// Decodes one UTF-8 sequence at s[i], advancing i; a malformed byte is
// returned as itself.
std::uint32_t next_code_point(std::string_view s, std::size_t& i) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    const std::size_t extra = b0 >= 0xF0 ? 3 : b0 >= 0xE0 ? 2 : b0 >= 0xC0 ? 1 : 0;
    if (extra == 0 || i + extra >= s.size()) {
        ++i;
        return b0;
    }
    std::uint32_t cp = b0 & (0x3Fu >> extra);
    for (std::size_t k = 1; k <= extra; ++k) {
        const auto b = static_cast<unsigned char>(s[i + k]);
        if ((b & 0xC0) != 0x80) {
            ++i;
            return b0;
        }
        cp = (cp << 6) | (b & 0x3Fu);
    }
    i += extra + 1;
    return cp;
}

// Whether DIGITS x 10^(exp10 - len + 1), i.e. d.ddd x 10^exp10, reads back as
// v. An integer mantissa, so no decimal point and no locale is involved.
bool reads_back(const std::string& digits, int exp10, double v) {
    const std::string t = digits + "e" + std::to_string(exp10 - static_cast<int>(digits.size()) + 1);
    return std::strtod(t.c_str(), nullptr) == v;
}

// The p-digit decimal one unit in the last place above (+1) or below (-1)
// d.ddd x 10^exp10, renormalised to p digits.
void step_digits(std::string& d, int& exp10, int dir) {
    const std::size_t p = d.size();
    if (dir > 0) {
        std::size_t k = p;
        while (k > 0 && d[k - 1] == '9') d[--k] = '0';
        if (k == 0) {
            d = "1" + std::string(p - 1, '0');  // 9.99 -> 10.0 = 1.00 x 10^(e+1)
            ++exp10;
        } else {
            ++d[k - 1];
        }
    } else {
        if (d == "1" + std::string(p - 1, '0')) {
            d = std::string(p, '9');  // 1.00 -> 0.999 = 9.99 x 10^(e-1)
            --exp10;
            return;
        }
        std::size_t k = p;
        while (k > 0 && d[k - 1] == '0') d[--k] = '9';
        --d[k - 1];
    }
}

// The shortest decimal digits that read back as v (finite, > 0), the one
// nearest v among them: what Python's repr prints. printf gives the nearest
// p-digit decimal; when it misses v's rounding interval but some p-digit
// decimal is inside it (the interval is lopsided at a power of two), that
// one is its neighbour on the far side. Not std::to_chars, which libc++
// withholds below macOS 13.3.
std::string shortest_digits(double v, int& decpt) {
    for (int p = 1; p <= 17; ++p) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.*e", p - 1, v);
        const char* e = std::strchr(buf, 'e');
        std::string digits;
        for (const char* c = buf; c < e; ++c)
            if (*c >= '0' && *c <= '9') digits += *c;
        const int exp10 = std::atoi(e + 1);
        if (reads_back(digits, exp10, v)) {
            decpt = exp10 + 1;
            return digits;
        }
        for (const int dir : {+1, -1}) {
            std::string d = digits;
            int x = exp10;
            step_digits(d, x, dir);
            if (reads_back(d, x, v)) {
                while (d.size() > 1 && d.back() == '0') d.pop_back();
                decpt = x + 1;
                return d;
            }
        }
    }
    char buf[64];  // unreachable: 17 significant digits always read back
    std::snprintf(buf, sizeof buf, "%.16e", v);
    const char* e = std::strchr(buf, 'e');
    std::string digits;
    for (const char* c = buf; c < e; ++c)
        if (*c >= '0' && *c <= '9') digits += *c;
    decpt = std::atoi(e + 1) + 1;
    return digits;
}

}  // namespace

// --- Python's text forms ----------------------------------------------------

std::string py_repr(std::string_view s) {
    // str.__repr__: single quotes unless the text has one and no double.
    const bool has_single = s.find('\'') != std::string_view::npos;
    const bool has_double = s.find('"') != std::string_view::npos;
    const char quote = (has_single && !has_double) ? '"' : '\'';
    std::string out(1, quote);
    static const char* hex = "0123456789abcdef";
    std::size_t i = 0;
    while (i < s.size()) {
        const std::size_t at = i;
        const std::uint32_t cp = next_code_point(s, i);
        if (cp == static_cast<std::uint32_t>(quote) || cp == '\\') {
            out += '\\';
            out += static_cast<char>(cp);
        } else if (cp == '\t') {
            out += "\\t";
        } else if (cp == '\n') {
            out += "\\n";
        } else if (cp == '\r') {
            out += "\\r";
        } else if (cp < 0x20 || (cp >= 0x7F && cp <= 0xA0) || cp == 0xAD) {
            // Not printable (str.isprintable); escaped as \xNN.
            out += "\\x";
            out += hex[(cp >> 4) & 0xF];
            out += hex[cp & 0xF];
        } else {
            out.append(s.substr(at, i - at));
        }
    }
    out += quote;
    return out;
}

std::string py_repr(double v) {
    // float.__repr__: the shortest round-tripping digits, fixed notation for
    // decimal exponents -4 < e <= 16 (".0" when integral), else d.ddde+XX.
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";
    int decpt = 0;  // value = 0.DIGITS x 10^decpt
    const std::string digits = shortest_digits(std::fabs(v), decpt);
    const int nd = static_cast<int>(digits.size());
    std::string out;
    if (decpt <= -4 || decpt > 16) {
        out = digits.substr(0, 1);
        if (nd > 1) out += "." + digits.substr(1);
        const int ex = decpt - 1;
        std::string es = std::to_string(ex < 0 ? -ex : ex);
        if (es.size() < 2) es = "0" + es;
        out += std::string("e") + (ex < 0 ? "-" : "+") + es;
    } else if (decpt <= 0) {
        out = "0." + std::string(static_cast<std::size_t>(-decpt), '0') + digits;
    } else if (decpt >= nd) {
        out = digits + std::string(static_cast<std::size_t>(decpt - nd), '0') + ".0";
    } else {
        out = digits.substr(0, static_cast<std::size_t>(decpt)) + "." +
              digits.substr(static_cast<std::size_t>(decpt));
    }
    return v < 0 ? "-" + out : out;
}

std::string shape_repr(std::span<const std::int64_t> shape) {
    std::string out = "(";
    for (std::size_t k = 0; k < shape.size(); ++k) {
        if (k) out += ", ";
        out += str(shape[k]);
    }
    if (shape.size() == 1) out += ",";
    return out + ")";
}

// --- lengths ------------------------------------------------------------------

int64_t refs_below(int64_t j) {
    check_count(j, "j");
    return ceildiv(j, REF);
}

int64_t data_symbols(int64_t n_data) {
    // L - ceil(L/16) never exceeds 15L/16, so counting up from
    // n_data + n_data//16 finds the smallest L.
    check_count(n_data, "n_data");
    const auto g = [](int64_t L) { return L - ceildiv(L, REF); };
    return first_at_least(g, n_data + floordiv(n_data, REF), n_data);
}

int64_t data_symbols_l(int64_t n_payload, int64_t j0) {
    // L's header is not a whole number of reference periods, so the data
    // section need not start on a reference.
    check_count(n_payload, "n_payload");
    check_count(j0, "j0");
    const int64_t r0 = ceildiv(j0, REF);
    const auto g = [&](int64_t D) { return D - (ceildiv(j0 + D, REF) - r0); };
    int64_t D = first_at_least(g, n_payload + floordiv(n_payload, REF), n_payload);
    // while D > 0 and g(D - 1) >= n: D -= 1  (g nondecreasing: down to the
    // smallest D with g(D) >= n, or to 0)
    if (D > 0 && g(D - 1) >= n_payload) {
        if (g(0) >= n_payload) return 0;
        int64_t lo = 0, hi = D - 1;  // g(lo) < n <= g(hi)
        while (hi - lo > 1) {
            const int64_t mid = lo + (hi - lo) / 2;
            (g(mid) >= n_payload ? hi : lo) = mid;
        }
        D = hi;
    }
    return D;
}

// --- FrameSpec ------------------------------------------------------------------

FrameSpec FrameSpec::make(Waveform waveform, std::string name, int64_t n_pre, int64_t n_hdr,
                          int64_t n_data, std::vector<int64_t> cw_after,
                          std::optional<double> hdr_rho) {
    if (waveform != Waveform::CE && waveform != Waveform::L)
        throw std::invalid_argument("waveform must be " + str(config::WAVEFORM_CE) + " (CE) or " +
                                    str(config::WAVEFORM_L) + " (L), not " +
                                    str(static_cast<int>(waveform)));
    const bool is_l = waveform == Waveform::L;
    const double rho = hdr_rho ? *hdr_rho : (n_hdr != 0 ? config::SPREAD_RHO : 0.0);
    if (!(0.0 <= rho && rho < 1.0))
        throw std::invalid_argument("hdr_rho must be in [0, 1), not " + py_repr(rho));
    const int64_t want_pre = is_l ? config::L_N_PRE : config::N_PRE;
    const int64_t want_hdr = is_l ? config::L_N_HDR : config::N_HDR;
    if (n_pre != want_pre)
        throw std::invalid_argument("n_pre must be " + str(want_pre) + ", not " + str(n_pre));
    if (n_hdr != 0 && n_hdr != want_hdr)
        throw std::invalid_argument("n_hdr must be 0 or " + str(want_hdr) + ", not " + str(n_hdr));
    if (is_l) {
        if (n_data <= 0 || n_data % 2)
            throw std::invalid_argument("n_data must be positive and even, not " + str(n_data));
    } else if (n_data <= 0) {
        throw std::invalid_argument("n_data must be positive, not " + str(n_data));
    }
    check_count(n_data, "n_data");

    FrameSpec s;
    s.waveform_ = waveform;
    s.name_ = std::move(name);
    s.n_pre_ = n_pre;
    s.n_hdr_ = n_hdr;
    s.n_data_ = n_data;
    s.cw_after_ = std::move(cw_after);
    s.hdr_rho_ = rho;
    s.n_data_sym_ = is_l ? data_symbols_l(n_data / 2, n_hdr) : data_symbols(n_data);

    const int64_t lo = n_pre + n_hdr, hi = s.n_sym();
    const int64_t* prev = nullptr;
    for (const int64_t& c : s.cw_after_) {
        if (!(lo < c && c <= hi))
            throw std::invalid_argument(s.name_ + ": window after stream symbol " + str(c) +
                                        " is outside (" + str(lo) + ", " + str(hi) + "]");
        if (prev && c <= *prev)
            throw std::invalid_argument(s.name_ + ": cw_after must be strictly increasing");
        prev = &c;
    }
    return s;
}

double FrameSpec::t_sym() const { return waveform_ == Waveform::L ? config::L_T_SYM : config::T_SYM; }

int64_t FrameSpec::cw_window() const {
    return waveform_ == Waveform::L ? config::L_CW_WINDOW : config::CW_WINDOW;
}

int64_t FrameSpec::n_data_payload() const { return waveform_ == Waveform::L ? n_data_ / 2 : n_data_; }

int64_t FrameSpec::n_ref() const { return ceildiv(n_sym() - n_pre_, REF); }

double FrameSpec::duration_s() const { return static_cast<double>(n_pos()) * t_sym(); }

std::vector<int64_t> FrameSpec::win_start_pos() const {
    std::vector<int64_t> out(cw_after_.size());
    for (std::size_t w = 0; w < cw_after_.size(); ++w)
        out[w] = cw_after_[w] + cw_window() * static_cast<int64_t>(w);
    return out;
}

bool FrameSpec::last_window_ends_frame() const { return !cw_after_.empty() && cw_after_.back() == n_sym(); }

double FrameSpec::keyed_end_pos() const {
    if (last_window_ends_frame()) return static_cast<double>(n_pos()) - 0.5;
    return static_cast<double>(pos_of(n_sym() - 1) + config::SPAN);
}

int64_t FrameSpec::pos_of(int64_t s) const {
    // np.searchsorted(cw_after, s, side="right"): windows at or before s.
    const auto n_before =
        static_cast<std::uint64_t>(std::upper_bound(cw_after_.begin(), cw_after_.end(), s) - cw_after_.begin());
    return static_cast<int64_t>(static_cast<std::uint64_t>(s) +
                                static_cast<std::uint64_t>(cw_window()) * n_before);
}

std::vector<std::pair<double, double>> FrameSpec::window_spans_s() const {
    std::vector<std::pair<double, double>> out;
    const double t = t_sym();
    for (const int64_t p : win_start_pos())
        out.emplace_back((static_cast<double>(p) - 0.5) * t, (static_cast<double>(p + cw_window()) - 0.5) * t);
    return out;
}

bool operator==(const FrameSpec& a, const FrameSpec& b) {
    return a.waveform_ == b.waveform_ && a.name_ == b.name_ && a.n_pre_ == b.n_pre_ &&
           a.n_hdr_ == b.n_hdr_ && a.n_data_ == b.n_data_ && a.cw_after_ == b.cw_after_ &&
           a.hdr_rho_ == b.hdr_rho_;
}

// --- presets --------------------------------------------------------------------

const std::vector<FrameSpec>& presets(config::PresetGroup group) {
    static const std::array<std::vector<FrameSpec>, 4> tables = [] {
        std::array<std::vector<FrameSpec>, 4> t;
        for (const config::PresetDef& d : config::FRAME_PRESETS) {
            std::vector<int64_t> cw(d.cw_after.begin(), d.cw_after.begin() + d.n_win);
            t[static_cast<std::size_t>(d.group)].push_back(
                FrameSpec::make(static_cast<Waveform>(d.waveform), std::string(d.name), d.n_pre, d.n_hdr,
                                d.n_data, std::move(cw), d.hdr_rho));
        }
        return t;
    }();
    return tables.at(static_cast<std::size_t>(group));
}

std::string unknown_frame(std::string_view shown, Waveform waveform) {
    if (waveform == Waveform::L)
        return "unknown L frame " + std::string(shown) + "; one of " + join_names(config::PresetGroup::LPresets);
    return "unknown frame " + std::string(shown) + "; one of " + join_names(config::PresetGroup::Presets);
}

const FrameSpec& get(std::string_view name) {
    for (const auto group : {config::PresetGroup::Presets, config::PresetGroup::Legacy,
                             config::PresetGroup::Prototypes})
        for (const FrameSpec& s : presets(group))
            if (s.name() == name) return s;
    throw std::invalid_argument(unknown_frame(py_repr(name), Waveform::CE));
}

const FrameSpec& get_l(std::string_view name) {
    for (const FrameSpec& s : presets(config::PresetGroup::LPresets))
        if (s.name() == name) return s;
    throw std::invalid_argument(unknown_frame(py_repr(name), Waveform::L));
}

const FrameSpec& get_any(std::string_view name) {
    if (name.substr(0, 2) == "l-") return get_l(name);
    return get(name);
}

FrameSpec twin(const FrameSpec& spec, int waveform) {
    if (static_cast<int>(spec.waveform()) == waveform) return spec;
    if (waveform == config::WAVEFORM_L) return get_l("l-" + spec.name());
    return get_any(spec.name().size() >= 2 ? std::string_view(spec.name()).substr(2) : std::string_view());
}

FrameSpec plain(const FrameSpec& spec) {
    if (spec.hdr_rho() == 0) return spec;
    for (const FrameSpec& v1 : presets(config::PresetGroup::Legacy))
        if (v1.waveform() == spec.waveform() && v1.n_pre() == spec.n_pre() && v1.n_hdr() == spec.n_hdr() &&
            v1.n_data() == spec.n_data() && v1.cw_after() == spec.cw_after())
            return v1;
    return FrameSpec::make(spec.waveform(), spec.name() + "-v1", spec.n_pre(), spec.n_hdr(), spec.n_data(),
                           spec.cw_after(), 0.0);
}

// --- layout ---------------------------------------------------------------------

void require_waveform(const FrameSpec& spec, Waveform waveform, std::string_view what) {
    if (spec.waveform() != waveform)
        throw WrongWaveform(std::string(what) + " is " + (waveform == Waveform::L ? "L's" : "CE's") + "; " +
                            py_repr(spec.name()) + " is waveform " + str(static_cast<int>(spec.waveform())));
}

namespace {

Layout build_layout(const FrameSpec& spec) {
    const bool is_l = spec.waveform() == Waveform::L;
    const int64_t n_sym = spec.n_sym(), n_pre = spec.n_pre(), n_hdr = spec.n_hdr();
    Layout lay;
    lay.waveform = spec.waveform();
    lay.is_ref.assign(static_cast<std::size_t>(n_sym), 0);
    lay.pos.resize(static_cast<std::size_t>(n_sym));
    std::vector<int64_t> hdr_nonref;
    for (int64_t s = 0; s < n_sym; ++s) {
        const int64_t j = s - n_pre;
        const bool is_ref = j >= 0 && j % REF == 0;
        lay.is_ref[static_cast<std::size_t>(s)] = is_ref ? 1 : 0;
        lay.pos[static_cast<std::size_t>(s)] = spec.pos_of(s);
        if (s < n_pre) {
            lay.pre.push_back(s);
        } else if (s < n_pre + n_hdr) {
            (is_ref ? lay.hdr_ref : hdr_nonref).push_back(s);
        } else {
            (is_ref ? lay.data_ref : lay.data).push_back(s);
        }
        if (is_ref) lay.ref.push_back(s);
    }
    if (static_cast<int64_t>(lay.data.size()) != spec.n_data_payload())
        throw std::logic_error(str(static_cast<int64_t>(lay.data.size())) + " data slots for " +
                               str(spec.n_data()) + " latents");

    // known_idx = pre ++ ref (++ CE's spare), then a stable sort by index.
    std::vector<int64_t> idx = lay.pre;
    idx.insert(idx.end(), lay.ref.begin(), lay.ref.end());
    std::vector<std::size_t> order;
    if (is_l) {
        if (n_hdr && static_cast<int64_t>(hdr_nonref.size()) != N_HDR_SLOTS_L)
            throw std::logic_error(str(static_cast<int64_t>(hdr_nonref.size())) + " header symbols for " +
                                   str(config::N_HDR_BITS) + " bits");
        lay.hdr_sym = std::move(hdr_nonref);
        std::vector<std::complex<double>> val = sequences::preamble_l();
        const auto refs = sequences::references_l(lay.ref.size());
        val.insert(val.end(), refs.begin(), refs.end());
        order.resize(idx.size());
        std::iota(order.begin(), order.end(), std::size_t(0));
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return idx[a] < idx[b]; });
        for (const std::size_t k : order) {
            lay.known_idx.push_back(idx[k]);
            lay.known_val_l.push_back(val[k]);
        }
    } else {
        if (n_hdr) {
            if (hdr_nonref.size() != N_BITS + 1)
                throw std::logic_error(str(static_cast<int64_t>(hdr_nonref.size())) + " header slots for " +
                                       str(config::N_HDR_BITS) + " bits");
            lay.hdr_bits.assign(hdr_nonref.begin(), hdr_nonref.begin() + static_cast<std::ptrdiff_t>(N_BITS));
            lay.hdr_spare.assign(hdr_nonref.begin() + static_cast<std::ptrdiff_t>(N_BITS), hdr_nonref.end());
        }
        idx.insert(idx.end(), lay.hdr_spare.begin(), lay.hdr_spare.end());
        std::vector<std::int8_t> val = sequences::preamble_ce();
        const auto refs = sequences::references_ce(lay.ref.size());
        val.insert(val.end(), refs.begin(), refs.end());
        val.insert(val.end(), lay.hdr_spare.size(), std::int8_t(1));
        order.resize(idx.size());
        std::iota(order.begin(), order.end(), std::size_t(0));
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return idx[a] < idx[b]; });
        for (const std::size_t k : order) {
            lay.known_idx.push_back(idx[k]);
            lay.known_val_ce.push_back(val[k]);
        }
    }
    const int64_t w = spec.cw_window();
    for (const int64_t p : spec.win_start_pos()) lay.win_pos.push_back({p, p + w - 1});
    return lay;
}

}  // namespace

std::shared_ptr<const Layout> layout(const FrameSpec& spec) {
    constexpr std::size_t CACHED = 16;
    static std::mutex mu;
    static std::vector<std::pair<FrameSpec, std::shared_ptr<const Layout>>> cache;  // most recent last
    {
        std::lock_guard<std::mutex> lock(mu);
        for (std::size_t k = 0; k < cache.size(); ++k) {
            if (cache[k].first == spec) {
                std::rotate(cache.begin() + static_cast<std::ptrdiff_t>(k),
                            cache.begin() + static_cast<std::ptrdiff_t>(k) + 1, cache.end());
                return cache.back().second;
            }
        }
    }
    auto lay = std::make_shared<const Layout>(build_layout(spec));
    std::lock_guard<std::mutex> lock(mu);
    cache.emplace_back(spec, lay);
    if (cache.size() > CACHED) cache.erase(cache.begin());
    return lay;
}

// --- assembling -------------------------------------------------------------------

HdrBits HdrBits::of(std::span<const std::uint8_t> bits) {
    HdrBits b;
    b.values.assign(bits.begin(), bits.end());
    b.len = bits.size();
    b.one_d = true;
    return b;
}

void check_hdr_bits(const FrameSpec& spec, const HdrBits* bits) {
    if (spec.has_header()) {
        if (!bits)
            throw std::invalid_argument("frame " + py_repr(spec.name()) + " has a header: hdr_bits is required");
        bool ok = bits->one_d && bits->values.size() == N_BITS;
        for (std::size_t i = 0; ok && i < bits->values.size(); ++i)
            ok = bits->values[i] == 0.0 || bits->values[i] == 1.0;
        if (!ok) throw std::invalid_argument("hdr_bits must be " + str(config::N_HDR_BITS) + " values of 0/1");
    } else if (bits && bits->len) {
        throw std::invalid_argument("frame " + py_repr(spec.name()) + " has no header but hdr_bits were given");
    }
}

void check_x_shape(const FrameSpec& spec, std::span<const std::int64_t> shape) {
    if (!(shape.size() == 1 && shape[0] == spec.n_data()))
        throw std::invalid_argument("x has shape " + shape_repr(shape) + ", expected (" + str(spec.n_data()) +
                                    ",)");
}

void check_y_symbols(const FrameSpec& spec, std::int64_t last_dim) {
    if (last_dim != spec.n_sym())
        throw std::invalid_argument("y_sym has " + str(last_dim) + " symbols, expected " + str(spec.n_sym()));
}

std::vector<double> spread_signs(const FrameSpec& spec, const HdrBits* bits) {
    if (!bits || !bits->one_d || bits->values.size() != N_BITS)
        throw std::invalid_argument("hdr_bits must be " + str(config::N_HDR_BITS) + " values of 0/1");
    const auto n = static_cast<std::size_t>(spec.n_data());
    const std::vector<std::int8_t> w = sequences::spread_whitener(n);
    const std::vector<double>& b = bits->values;
    std::vector<double> out(n);
    // numpy: w * (1.0 - 2.0 * b[arange(n) % 2474])
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<double>(w[i]) * (1.0 - 2.0 * b[i % N_BITS]);
    return out;
}

std::vector<double> spread_symbols(const FrameSpec& spec, const HdrBits* bits) {
    const std::vector<double> h = spread_signs(spec, bits);  // the right-hand side first, as Python
    require_waveform(spec, Waveform::CE, "frame.spread_symbols");
    const auto lay = layout(spec);
    std::vector<double> out(static_cast<std::size_t>(spec.n_sym()), 0.0);
    const double s = std::sqrt(spec.hdr_rho());
    for (std::size_t i = 0; i < h.size(); ++i) out[static_cast<std::size_t>(lay->data[i])] = s * h[i];
    return out;
}

std::vector<double> assemble(const FrameSpec& spec, const HdrBits* bits, std::span<const double> x) {
    require_waveform(spec, Waveform::CE, "frame.assemble");
    const auto lay = layout(spec);
    const std::int64_t xs[1] = {static_cast<std::int64_t>(x.size())};
    check_x_shape(spec, xs);
    check_hdr_bits(spec, bits);
    std::vector<double> out(static_cast<std::size_t>(spec.n_sym()), 0.0);
    for (std::size_t k = 0; k < lay->known_idx.size(); ++k)
        out[static_cast<std::size_t>(lay->known_idx[k])] = static_cast<double>(lay->known_val_ce[k]);
    if (spec.n_hdr())
        for (std::size_t i = 0; i < lay->hdr_bits.size(); ++i)
            out[static_cast<std::size_t>(lay->hdr_bits[i])] = 1.0 - 2.0 * bits->values[i];
    for (std::size_t i = 0; i < x.size(); ++i) out[static_cast<std::size_t>(lay->data[i])] = x[i];
    if (spec.hdr_rho() > 0) {
        // out[data] += sqrt(rho) * h: the latent, then the header phase on top.
        const std::vector<double> h = spread_signs(spec, bits);
        const double s = std::sqrt(spec.hdr_rho());
        for (std::size_t i = 0; i < h.size(); ++i) {
            double& o = out[static_cast<std::size_t>(lay->data[i])];
            o = o + s * h[i];
        }
    }
    return out;
}

std::vector<double> data_latents(const FrameSpec& spec, std::span<const double> x, const HdrBits* bits) {
    const std::int64_t xs[1] = {static_cast<std::int64_t>(x.size())};
    check_x_shape(spec, xs);
    if (spec.hdr_rho() == 0) return std::vector<double>(x.begin(), x.end());
    const double rho = spec.hdr_rho();
    const std::vector<double> h = spread_signs(spec, bits);
    const double s1 = std::sqrt(rho), s2 = std::sqrt(1.0 + rho);
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = (x[i] + s1 * h[i]) / s2;
    return out;
}

std::vector<std::complex<double>> pair(std::span<const double> v) {
    if (v.size() % 2)
        throw std::invalid_argument("pair takes an even number of values, not shape (" +
                                    str(static_cast<int64_t>(v.size())) + ",)");
    // numpy evaluates (v[0::2] + 1j * v[1::2]) / np.sqrt(2.0) as complex
    // arithmetic throughout: 1j*b is (0*b - 1*0, 0*0 + 1*b), adding the real
    // a makes (a + re, 0 + im), and dividing by the real s is its complex
    // division -- rat = 0/s, scl = 1/(s + 0*rat), then each part times scl
    // -- which is a multiplication by 1/sqrt(2), not a division by sqrt(2).
    const double s = std::sqrt(2.0);
    const double rat = 0.0 / s;
    const double scl = 1.0 / (s + 0.0 * rat);
    std::vector<std::complex<double>> out(v.size() / 2);
    for (std::size_t k = 0; k < out.size(); ++k) {
        const double a = v[2 * k], b = v[2 * k + 1];
        const double t_re = 0.0 * b - 1.0 * 0.0;
        const double t_im = 0.0 * 0.0 + 1.0 * b;
        const double u_re = a + t_re;
        const double u_im = 0.0 + t_im;
        out[k] = {(u_re + u_im * rat) * scl, (u_im - u_re * rat) * scl};
    }
    return out;
}

std::vector<double> unpair(std::span<const std::complex<double>> z) {
    const double s = std::sqrt(2.0);
    std::vector<double> out(2 * z.size());
    for (std::size_t k = 0; k < z.size(); ++k) {
        out[2 * k] = s * z[k].real();
        out[2 * k + 1] = s * z[k].imag();
    }
    return out;
}

std::vector<std::complex<double>> assemble_l(const FrameSpec& spec, const HdrBits* bits,
                                             std::span<const double> x) {
    require_waveform(spec, Waveform::L, "lframe.assemble");
    const auto lay = layout(spec);
    std::vector<std::complex<double>> out(static_cast<std::size_t>(spec.n_sym()), {0.0, 0.0});
    for (std::size_t k = 0; k < lay->known_idx.size(); ++k)
        out[static_cast<std::size_t>(lay->known_idx[k])] = lay->known_val_l[k];
    check_hdr_bits(spec, bits);
    if (spec.n_hdr()) {
        std::vector<std::uint8_t> b(bits->values.size());
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = bits->values[i] == 1.0 ? 1 : 0;
        const auto q = sequences::qpsk(b);
        for (std::size_t j = 0; j < q.size(); ++j) out[static_cast<std::size_t>(lay->hdr_sym[j])] = q[j];
    }
    const auto p = pair(data_latents(spec, x, bits));
    for (std::size_t k = 0; k < p.size(); ++k) out[static_cast<std::size_t>(lay->data[k])] = p[k];
    return out;
}

std::pair<std::vector<double>, std::vector<double>> disassemble_l(std::span<const std::complex<double>> y,
                                                                  const FrameSpec& spec) {
    require_waveform(spec, Waveform::L, "lframe.disassemble");
    const auto lay = layout(spec);
    const auto n = static_cast<std::size_t>(spec.n_sym());
    return {unpair(gather(y, n, lay->hdr_sym)), unpair(gather(y, n, lay->data))};
}

}  // namespace sstvae::qrss::frame
