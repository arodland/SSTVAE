// em.py's shared pieces (see em.hpp). The statements follow the reference
// in its order; comments name the Python they mirror.

#include "qrss/em.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "math/special.hpp"
#include "qrss/morse.hpp"
#include "qrss/npsum.hpp"
#include "qrss/precoder.hpp"
#include "qrss/pyjson.hpp"

namespace sstvae::qrss::em {

namespace {

using pyjson::Value;

constexpr double INF = std::numeric_limits<double>::infinity();

// numpy's broadcast refusal for two 1-D operands of different lengths.
[[noreturn]] void broadcast_error(std::size_t a, std::size_t b) {
    throw std::invalid_argument("operands could not be broadcast together with shapes (" + std::to_string(a) +
                                ",) (" + std::to_string(b) + ",) ");
}

// Python's math.log(x): ValueError "math domain error" for x <= 0 (and
// -0.0, which Python treats as 0); NaN and inf through.
double py_log(double x) {
    if (x <= 0.0) throw std::invalid_argument("math domain error");
    return std::log(x);
}

// Python's math.lgamma(x) with its exceptions (ValueError at a pole,
// OverflowError when a finite x overflows).
double py_lgamma_checked(double x) {
    const double r = math::py_lgamma(x);
    if (std::isinf(r) && std::isfinite(x)) {
        if (x == std::floor(x) && x <= 0.0) throw std::invalid_argument("math domain error");
        throw std::overflow_error("math range error");
    }
    return r;
}

// max(x, 1.0) / min(x, 0.0) as Python's builtins pick: the first argument
// unless the second compares greater (smaller), so a NaN first argument wins.
double py_max(double x, double y) { return y > x ? y : x; }
double py_min(double x, double y) { return y < x ? y : x; }

// HeaderFields(**h) of a stored pass's header dict (PassResult.save's keys).
header::HeaderFields header_from_json(const Value& h) {
    header::HeaderFields f;
    f.callsign = h.at("callsign").as_string("callsign");
    const Value& g = h.at("grid");
    if (!g.is_null()) f.grid = g.as_string("grid");
    f.picture_id = store::py_int(h.at("picture_id"));
    f.mode = store::py_int(h.at("mode"));
    f.segment = store::py_int(h.at("segment"));
    f.codec_id = store::py_int(h.at("codec_id"));
    f.version = store::py_int(h.at("version"));
    f.reserved = store::py_int(h.at("reserved"));
    return f;
}

// A member's "uid" as store.pass_info / load_pass take it.
std::string member_uid(const Value& m) { return m.at("uid").as_string("uid"); }

}  // namespace

// --- the hooks ----------------------------------------------------------------------------

void check_cancel(const Hooks& hooks) {
    if (hooks.cancel != nullptr && hooks.cancel->load(std::memory_order_relaxed)) throw Cancelled("cancelled");
}

void log_line(const Hooks& hooks, const std::string& line) {
    if (hooks.log) hooks.log(line);
}

void report_progress(const Hooks& hooks, std::int64_t done, std::int64_t total, const std::string& what) {
    if (hooks.progress) hooks.progress(done, total, what);
}

// --- references ---------------------------------------------------------------------------

Lmmse lmmse_latents(std::span<const double> S, std::span<const double> W) {
    // W = np.maximum(W, 0.0) (NaN propagates); a = S / (1.0 + W) broadcast
    // as numpy does (a length-1 operand stretches); r = 1.0 / (1.0 + W)
    // has W's own length.
    std::size_t n = S.size();
    if (S.size() != W.size()) {
        if (S.size() == 1) n = W.size();
        else if (W.size() != 1) broadcast_error(S.size(), W.size());
    }
    Lmmse out;
    out.a.resize(n);
    out.r.resize(W.size());
    std::vector<double> d(W.size());   // 1.0 + W
    for (std::size_t i = 0; i < W.size(); ++i) {
        const double w = (W[i] >= 0.0 || std::isnan(W[i])) ? W[i] : 0.0;
        d[i] = 1.0 + w;
        out.r[i] = 1.0 / d[i];
    }
    for (std::size_t i = 0; i < n; ++i) out.a[i] = S[S.size() == 1 ? 0 : i] / d[W.size() == 1 ? 0 : i];
    return out;
}

types::EmPrior prior_from(std::span<const double> S, std::span<const double> W, std::int64_t q) {
    const Lmmse l = lmmse_latents(S, W);
    if (q < 0) throw std::invalid_argument("q must fit in uint64, not " + std::to_string(q));
    types::EmPrior p;
    p.x_hat = precoder::precode(std::span<const double>(l.a), static_cast<std::uint64_t>(q), l.a.size());
    p.v = precoder::block_mean(std::span<const double>(l.r), l.r.size());
    return p;
}

types::EmPrior em_prior(const store::Accumulator& acc, const std::string& uid, store::Store& store, std::int64_t q,
                        const frame::FrameSpec& spec) {
    const auto [S_, W_] = acc.loo(uid, store);
    const std::int64_t n = spec.n_data();
    if (static_cast<std::int64_t>(S_.size()) != n)
        throw std::invalid_argument("pass " + uid + " holds " + std::to_string(S_.size()) + " latents; frame " +
                                    frame::py_repr(spec.name()) + " sends " + std::to_string(n));
    return prior_from(S_, W_, q);
}

double acc_mean_w_db(const store::Accumulator& acc) {
    std::vector<double> heard;
    for (const float w : acc.W)
        if (w > 0) heard.push_back(static_cast<double>(w));
    if (heard.empty()) return -INF;
    return 10.0 * std::log10(npsum::mean(std::span<const double>(heard)));
}

// --- what every member pass is known to have sent ------------------------------------------

std::optional<header::HeaderFields> known_header(store::Store& store, const store::Accumulator& acc,
                                                 std::int64_t segment) {
    const store::Key& key = acc.key;   // (acc.key[0], int(acc.key[1]))
    std::optional<header::HeaderFields> found;
    // llr_by_seg: a dict, so the segments in order of first appearance
    std::vector<std::pair<std::int64_t, std::vector<double>>> llr_by_seg;
    for (const Value& m : acc.members) {
        const store::PassInfo info = store.pass_info(member_uid(m));
        const Value& h = info.header;
        if (!h.is_null()) {
            // (h["callsign"], int(h["picture_id"])) == key: the tuple is built
            // (int() evaluated) before it is compared
            const Value& call = h.at("callsign");
            const std::int64_t pid = store::py_int(h.at("picture_id"));
            if (store::equals(call, key.callsign) && pid == key.picture_id) {
                found = header_from_json(h);
                break;
            }
        }
        const std::vector<double>& llr = info.hdr_llr;
        bool any = false;   // np.any(llr != 0): NaN counts
        for (const double v : llr)
            if (v != 0.0) {
                any = true;
                break;
            }
        if (llr.size() == static_cast<std::size_t>(config::N_HDR_BITS) && any) {
            const std::int64_t g = store::py_int(m.at("segment"));
            std::vector<double>* slot = nullptr;
            for (auto& [seg, sum] : llr_by_seg)
                if (seg == g) slot = &sum;
            if (slot == nullptr) {
                // llr_by_seg.get(g, 0.0) + llr
                std::vector<double> first(llr.size());
                for (std::size_t i = 0; i < llr.size(); ++i) first[i] = 0.0 + llr[i];
                llr_by_seg.emplace_back(g, std::move(first));
            } else {
                for (std::size_t i = 0; i < llr.size(); ++i) (*slot)[i] = (*slot)[i] + llr[i];
            }
        }
    }
    if (!found) {
        for (const auto& [seg, llr] : llr_by_seg) {
            (void)seg;
            std::optional<header::HeaderFields> h = header::decode(llr);
            if (h && h->callsign == key.callsign && h->picture_id == key.picture_id) {
                found = std::move(h);
                break;
            }
        }
    }
    if (!found) return std::nullopt;
    if (!(0 <= segment && segment <= found->mode)) return std::nullopt;
    found->segment = segment;
    return found;
}

std::optional<receiver::CwKnown> known_keying(store::Store& store, const store::Accumulator& acc,
                                              const std::optional<std::string>& call, std::int64_t waveform) {
    if (!call) return std::nullopt;
    try {
        morse::check_callsign(*call);
    } catch (const std::invalid_argument&) {   // except ValueError
        return std::nullopt;
    }
    std::int64_t fsk = 0, ook = 0;
    for (const Value& m : acc.members) {
        const types::PassResult p = store.load_pass(member_uid(m));
        if (static_cast<std::int64_t>(pass_spec(p).waveform()) != waveform) continue;
        if (p.cw && p.cw->agrees.value_or(false) && p.cw->z_match > config::Z_ACCEPT) {
            if (p.cw->keying == "fsk") ++fsk;
            else if (p.cw->keying == "ook") ++ook;
            // any other keying string gets a vote of its own, which decides nothing
        }
    }
    if (fsk == 0 && ook == 0) return std::nullopt;
    receiver::CwKnown out;
    out.units = morse::keying_units(*call);
    out.ook = waveform != config::WAVEFORM_L && ook > fsk;
    return out;
}

// --- a stored pass ------------------------------------------------------------------------

frame::FrameSpec pass_spec(const types::PassResult& p) {
    frame::FrameSpec spec = frame::get_any(p.frame);
    if (spec.waveform() == frame::Waveform::CE && p.header && p.header->version == 1) spec = frame::plain(spec);
    return spec;
}

bool is_l(const types::PassResult& p) { return pass_spec(p).waveform() == frame::Waveform::L; }

void ce_only(const frame::FrameSpec& spec, std::string_view what) {
    if (spec.waveform() != frame::Waveform::CE)
        throw std::invalid_argument(std::string(what) + " is CE-only (design 8.3): frame " +
                                    frame::py_repr(spec.name()) + " is waveform L");
}

types::FreqPath path_of(const track::TrackResult& tr) {
    if (tr.pos.empty()) throw std::out_of_range("index 0 is out of bounds for axis 0 with size 0");
    const double p0 = static_cast<double>(tr.pos.front());
    const double p1 = static_cast<double>(tr.pos.back());
    const double start = tr.t_pos_s(std::span<const double>(&p0, 1))[0] - 10.0;
    const double stop = tr.t_pos_s(std::span<const double>(&p1, 1))[0] + 10.0;
    // np.arange(start, stop, 1.0): ceil((stop - start) / 1.0) points (none
    // when that is not positive); start, then start + 1.0, then the fill
    // loop's start + i * delta with delta = (start + 1.0) - start.
    const double len_d = std::ceil((stop - start) / 1.0);
    if (std::isnan(len_d)) throw std::invalid_argument("arange: cannot compute length");
    std::size_t n = 0;
    if (len_d > 0) {
        if (len_d > 9.2e18) throw std::invalid_argument("Maximum allowed size exceeded");
        n = static_cast<std::size_t>(len_d);
    }
    types::FreqPath out;
    out.t_s.resize(n);
    if (n > 0) out.t_s[0] = start;
    if (n > 1) out.t_s[1] = start + 1.0;
    if (n > 2) {
        const double delta = out.t_s[1] - out.t_s[0];
        for (std::size_t i = 2; i < n; ++i) out.t_s[i] = start + static_cast<double>(i) * delta;
    }
    out.f_hz = tr.freq(std::span<const double>(out.t_s));
    out.weight.assign(n, 1.0);
    return out;
}

types::FreqPath flat(double f_hz) {
    return {{-20.0, 0.0, 1800.0}, {f_hz, f_hz, f_hz}, {1.0, 1.0, 1.0}};
}

// --- the template search's statistic -------------------------------------------------------

double gauss_z(double log_p) {
    if (log_p >= std::log(0.5)) return 0.0;
    if (log_p > -700.0) return math::norm_isf(std::exp(log_p));
    // Mills ratio, to well under 0.01 in Z this far out
    double z = std::sqrt(-2.0 * log_p);
    const double s2pi = std::sqrt(2 * 3.141592653589793);   // math.sqrt(2 * math.pi)
    for (int k = 0; k < 4; ++k) z = std::sqrt(py_max(-2.0 * log_p - 2.0 * py_log(z * s2pi), 1.0));
    return z;
}

double template_z(double lam, std::int64_t n_chunks, double n_cells) {
    const double K = static_cast<double>(n_chunks);
    double log_p = math::gamma_logsf(lam, K);
    if (!std::isfinite(log_p)) {
        // far out the tail: log of the leading term x^(K-1) e^-x / Gamma(K)
        const double km1 = static_cast<double>(n_chunks - 1);
        const double lg = py_log(lam);
        log_p = km1 * lg - lam - py_lgamma_checked(K);
    }
    log_p += py_log(py_max(n_cells, 1.0));
    return gauss_z(py_min(log_p, 0.0));
}

}  // namespace sstvae::qrss::em
