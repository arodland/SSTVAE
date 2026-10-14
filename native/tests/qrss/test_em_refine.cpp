// native/core/qrss/em_refine (em.py's pass_path, rereceive and em_refine)
// without Python, against values pinned from the reference by
// gen_em_refine_pins.py (em_refine_pins.hpp; the formulas are in that
// script's docstring).
//
// The passes are synthetic CH-rate captures (a stored pass's form) of one
// latent vector precoded into three TINY slots and one SHORT, received by
// receive_pass here as in the generator (the captures' digests say the
// inputs are the generator's to the bit). What is pinned is the reference's
// results, held to their primitives' rounding (1e-5 of max(1, |value|): the
// tracker, the fit and the demodulator agree to ~1e-7, and these are a few
// of them composed), every decision exactly (pass_path's choice, the
// branches, lengths, flags, estimator, em_round, which rounds ran and what
// they logged) and the log's numbers to 0.011 dB (they are printed "%+.2f").
//
// The rest is the C++'s own: the hooks (progress per pass and its total;
// the "converged" call when the rounds stop early; none for rounds 0),
// cancel (before the call, from inside rereceive, and set from the progress
// hook -- the next poll throws Cancelled and every file of the store still
// loads, with the passes re-received before it stored), the EmSeams seam
// (called in member order with the path it handed back last round, a pass
// without a capture skipped, a nullopt pass not stored) and the refusals.
// tests/test_qrss_native_em_refine.py compares against the reference
// directly, on simulated passes.
//
//   PYTHONPATH=. python native/tests/qrss/gen_em_refine_pins.py   # em_refine_pins.hpp

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "em_refine_pins.hpp"
#include "qrss/ce.hpp"
#include "qrss/config.hpp"
#include "qrss/em.hpp"
#include "qrss/em_refine.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/morse.hpp"
#include "qrss/precoder.hpp"
#include "qrss/rational.hpp"
#include "qrss/receiver.hpp"
#include "qrss/store.hpp"
#include "qrss/track.hpp"
#include "qrss/types.hpp"
#include "qrss_check.hpp"
#include "util/fileio.hpp"

using namespace sstvae;
namespace em = sstvae::qrss::em;
namespace rc = sstvae::qrss::receiver;
namespace fr = sstvae::qrss::frame;
namespace hd = sstvae::qrss::header;
namespace qs = sstvae::qrss::store;
namespace types = sstvae::qrss::types;
namespace cfg = sstvae::qrss::config;
namespace P = em_refine_pins;
namespace fs = std::filesystem;
using cd = std::complex<double>;
using qrss::Rational;

namespace {

constexpr double TOL = 1e-5;     // the primitives' rounding, composed
constexpr double LOG_TOL = 0.011;   // a "%+.2f" number against the reference's
double worst_rel = 0.0;

template <typename E, typename F>
void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("threw another type: ") + e.what());
        return;
    }
    check::fail(what, "did not throw");
}

// |got - want| <= tol * max(1, |want|)
void near(double got, double want, const std::string& what, double tol = TOL) {
    const double rel = std::abs(got - want) / std::max(1.0, std::abs(want));
    const bool ok = rel <= tol;
    if (tol == TOL && std::isfinite(rel)) worst_rel = std::max(worst_rel, rel);
    check::is_true(ok, what);
    if (!ok) std::fprintf(stderr, "   %s: got %.17g want %.17g\n", what.c_str(), got, want);
}

void near_all(std::span<const double> got, const double* want, std::size_t n, const std::string& what) {
    check::equal(got.size(), n, what + ": length");
    for (std::size_t i = 0; i < std::min(got.size(), n); ++i) near(got[i], want[i], what + "[" + std::to_string(i) + "]");
}

double sum_of(std::span<const float> v) {
    double s = 0.0;
    for (const float x : v) s += static_cast<double>(x);
    return s;
}
double sum_of(std::span<const double> v) {
    double s = 0.0;
    for (const double x : v) s += x;
    return s;
}

// --- the fixture (gen_em_refine_pins.py) --------------------------------------------------------

std::vector<double> hnoise(std::int64_t n, std::uint64_t seed) {
    constexpr std::uint64_t M32 = 0xFFFFFFFFull;
    std::vector<double> out(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        std::uint64_t h = (static_cast<std::uint64_t>(i) * 0x9E3779B1ull + seed) & M32;
        h ^= h >> 15;
        h = (h * 0x85EBCA77ull) & M32;
        h ^= h >> 13;
        h = (h * 0xC2B2AE3Dull) & M32;
        h ^= h >> 16;
        out[static_cast<std::size_t>(i)] = static_cast<double>(h) / 4294967296.0 - 0.5;
    }
    return out;
}

// noise_of(n, scale, k): seeds 8k + 1 .. 8k + 8
std::vector<cd> noise_of(std::int64_t n, double scale, std::uint64_t k) {
    const std::uint64_t s = 8 * k;
    const auto u1 = hnoise(n, s + 1), u2 = hnoise(n, s + 2), u3 = hnoise(n, s + 3), u4 = hnoise(n, s + 4);
    const auto u5 = hnoise(n, s + 5), u6 = hnoise(n, s + 6), u7 = hnoise(n, s + 7), u8 = hnoise(n, s + 8);
    std::vector<cd> out(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double re = ((u1[i] + u2[i]) + u3[i]) + u4[i];
        const double im = ((u5[i] + u6[i]) + u7[i]) + u8[i];
        out[i] = cd(re * scale, im * scale);
    }
    return out;
}

hd::HeaderFields test_header() {
    hd::HeaderFields h;
    h.callsign = "K1ABC";
    h.grid = "FN42";
    h.picture_id = 0x12345678;
    h.mode = 0;
    h.segment = 0;
    h.codec_id = 0xD1D8;
    return h;
}

const qs::Key KEY{"K1ABC", 0x12345678};

// air(spec): a[k] = +1 where (5 k + 1) % 3 == 0, else -1
std::vector<double> air(const fr::FrameSpec& spec) {
    std::vector<double> a;
    for (std::int64_t k = 0; k < spec.n_data(); ++k) a.push_back((k * 5 + 1) % 3 == 0 ? 1.0 : -1.0);
    return a;
}

rc::Prepared cap(const fr::FrameSpec& spec, std::int64_t n, std::int64_t k) {
    const std::int64_t q = P::Q + 7 * k;
    const std::vector<std::uint8_t> bits = hd::encode(test_header());
    const fr::HdrBits hb = fr::HdrBits::of(bits);
    const std::vector<double> data = qrss::precoder::precode(air(spec), static_cast<std::uint64_t>(q));
    const auto sym = fr::assemble(spec, spec.n_hdr() ? &hb : nullptr, data);
    const auto phi = qrss::ce::phase_grid(250, -3000, n, sym, spec);
    const auto noise = noise_of(n, P::NOISE, static_cast<std::uint64_t>(k));
    rc::Prepared p;
    const double half_milli = 0.5 * 0.001;
    for (std::int64_t i = 0; i < n; ++i) {
        const double di = static_cast<double>(i);
        const double sq = (di / 250) * (di / 250);
        const double ph = phi[static_cast<std::size_t>(i)] + 2 * std::numbers::pi * (0.02 * di / 250 + half_milli * sq);
        const cd z = cd(std::cos(ph), std::sin(ph)) + noise[static_cast<std::size_t>(i)];
        p.fe.emplace_back(static_cast<float>(z.real()), static_cast<float>(z.imag()));
    }
    p.q = q;
    p.t0_index = 3002.5;
    p.fs = cfg::CH_FS;
    return p;
}

types::Detection det_of() {
    types::Detection d;
    d.f_hz = 1500.0;
    d.path = types::FreqPath{{-10.0, 0.0, 1800.0}, {1500.0, 1500.0, 1500.0}, {1.0, 1.0, 1.0}};
    d.z_ref = std::numeric_limits<double>::quiet_NaN();
    d.method = "tbd";
    d.waveform = 0;
    return d;
}

types::PassResult receive(const rc::Prepared& prep, const fr::FrameSpec& spec, const std::string& what) {
    rc::PassArgs a;
    a.f_mix_hz = Rational(0);
    rc::PassOutcome o = rc::receive_pass(&prep, spec, det_of(), a);
    if (!o.pass) throw std::runtime_error(what + ": receive_pass refused the fixture");
    return std::move(*o.pass);
}

fs::path fresh_dir(const std::string& name) {
    const fs::path p = fs::temp_directory_path() /
                       ("sstvae_test_em_refine_" + name + "_" +
                        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(p);
    return p;
}

struct Cleanup {
    fs::path dir;
    ~Cleanup() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

qs::Store::Seams clock_t0() {
    qs::Store::Seams s;
    s.clock = [] { return P::T0; };
    return s;
}

// A store at dir/qrss holding the passes, attached to KEY in order.
void fill(qs::Store& st, const std::vector<types::PassResult>& passes) {
    for (const types::PassResult& p : passes) {
        st.add_pass(p);
        st.attach(KEY, p.uid, 0, "test", 0, 0xD1D8);
    }
}

// --- one pass against its pins ------------------------------------------------------------------

struct Span {
    const double* p;
    std::size_t n;
};

struct PassPins {
    Span f, report;
    std::int64_t flags, nz;
    Span z_at, w_at;
    double z_sum, w_sum, psi_sum;
    std::int64_t nllr;
    double llr_abs;
    const char* estimator;
    std::int64_t em_round, hdr;
    double mean_w_db;
    std::int64_t path_n;
    Span path_t;
    double path_f_sum;
};

#define SPAN(X) Span{P::X, P::X##_N}
#define PASS_PINS(T)                                                                                                 \
    PassPins{SPAN(T##_F),      SPAN(T##_REPORT), P::T##_FLAGS,      P::T##_NZ,        SPAN(T##_Z_AT),                 \
             SPAN(T##_W_AT),   P::T##_Z_SUM,     P::T##_W_SUM,      P::T##_PSI_SUM,   P::T##_NLLR,                    \
             P::T##_LLR_ABS,   P::T##_ESTIMATOR, P::T##_EM_ROUND,   P::T##_HDR,       P::T##_MEAN_W_DB,               \
             P::T##_PATH_N,    SPAN(T##_PATH_T), P::T##_PATH_F_SUM}

void check_pass(const em::Rereceived& r, const PassPins& w, const std::string& tag) {
    check::is_true(r.pass.has_value() && r.path.has_value(), tag + ": a pass and its path");
    if (!r.pass || !r.path) return;
    const types::PassResult& p = *r.pass;
    const double f[] = {p.f_hz, p.timing.tau0, p.timing.ppm, p.timing.gamma, p.timing.z};
    near_all(f, w.f.p, w.f.n, tag + ": f_hz, timing");
    const types::TrackReport& rp = p.report;
    const double rep[] = {rp.offset_hz, rp.drift_hz_per_min, rp.wander_hz_rms, rp.doppler_hz,
                          rp.ppm,       rp.snr2500_db,       rp.z_ref,         rp.kappa};
    near_all(rep, w.report.p, w.report.n, tag + ": report");
    check::equal(static_cast<std::int64_t>(rp.slip_free) + 2 * static_cast<std::int64_t>(rp.suspect), w.flags,
                 tag + ": slip_free + 2 suspect");
    check::equal(static_cast<std::int64_t>(p.z.size()), w.nz, tag + ": len(z)");
    check::equal(static_cast<std::int64_t>(p.w.size()), w.nz, tag + ": len(w)");
    for (std::size_t j = 0; j < w.z_at.n && p.z.size() == static_cast<std::size_t>(w.nz); ++j) {
        const std::size_t i = P::IDX[j];
        near(p.z[i], w.z_at.p[j], tag + ": z[" + std::to_string(i) + "]");
        near(p.w[i], w.w_at.p[j], tag + ": w[" + std::to_string(i) + "]");
    }
    near(sum_of(p.z), w.z_sum, tag + ": sum z");
    near(sum_of(p.w), w.w_sum, tag + ": sum w");
    near(sum_of(p.psi), w.psi_sum, tag + ": sum psi");
    check::equal(static_cast<std::int64_t>(p.hdr_llr.size()), w.nllr, tag + ": len(hdr_llr)");
    double llr_abs = 0.0;
    for (const float x : p.hdr_llr) llr_abs += std::abs(static_cast<double>(x));
    near(llr_abs, w.llr_abs, tag + ": sum |hdr_llr|");
    check::equal(p.estimator, std::string(w.estimator), tag + ": estimator");
    check::equal(p.em_round, w.em_round, tag + ": em_round");
    check::equal(static_cast<std::int64_t>(p.header.has_value()), w.hdr, tag + ": header");
    near(rc::mean_w_db(p), w.mean_w_db, tag + ": mean W dB");
    const types::FreqPath& pth = *r.path;
    check::equal(static_cast<std::int64_t>(pth.t_s.size()), w.path_n, tag + ": path length");
    if (!pth.t_s.empty()) {
        const double t[] = {pth.t_s.front(), pth.t_s.back()};
        near_all(t, w.path_t.p, w.path_t.n, tag + ": path t_s ends");
    }
    near(sum_of(pth.f_hz), w.path_f_sum, tag + ": path sum f_hz");
}

// --- the fixture, received once -----------------------------------------------------------------

struct Fixture {
    std::vector<types::PassResult> tiny;   // k = 0, 1, 2
    types::PassResult short_;
};

Fixture build() {
    check::current_step = "fixture";
    const auto& TINY = fr::get("tiny");
    const auto& SHORT = fr::get("short");
    Fixture fx;
    const char* digests[] = {P::CAP_TINY_0, P::CAP_TINY_1, P::CAP_TINY_2};
    const double mean_w[] = {P::TINY_0_MEAN_W_DB, P::TINY_1_MEAN_W_DB, P::TINY_2_MEAN_W_DB};
    for (std::int64_t k = 0; k < 3; ++k) {
        const rc::Prepared prep = cap(TINY, P::N_TINY, k);
        const std::string tag = "TINY pass " + std::to_string(k);
        qrss_check::digest_is(prep.fe, digests[k], tag + ": the capture");
        fx.tiny.push_back(receive(prep, TINY, tag));
        near(rc::mean_w_db(fx.tiny.back()), mean_w[k], tag + ": mean W dB as received");
    }
    const rc::Prepared sp = cap(SHORT, P::N_SHORT, 0);
    qrss_check::digest_is(sp.fe, P::CAP_SHORT, "SHORT: the capture");
    fx.short_ = receive(sp, SHORT, "SHORT");
    check::equal(static_cast<std::int64_t>(fx.short_.header.has_value()), P::SHORT_HAS_HDR, "SHORT: header");
    return fx;
}

// --- the tests ---------------------------------------------------------------------------------

void test_pass_path(const Fixture& fx) {
    check::current_step = "pass_path";
    const types::PassResult& p0 = fx.tiny[0];
    em::PassPathTrace tr;
    const types::FreqPath best = em::pass_path(p0, nullptr, &tr);
    near(tr.z_line, P::PP_Z[0], "pass_path: Z_ref of the line");
    near(tr.z_refined, P::PP_Z[1], "pass_path: Z_ref of the refinement");
    check::equal(static_cast<std::int64_t>(tr.choice), P::PP_CHOICE, "pass_path: which won");
    check::is_true(best.f_hz == (tr.choice ? tr.refined : tr.line).f_hz, "pass_path: returns the winner");
    check::equal(static_cast<std::int64_t>(best.t_s.size()), P::PP_N, "pass_path: length");
    if (best.t_s.size() == static_cast<std::size_t>(P::PP_N)) {
        const double t[] = {best.t_s.front(), best.t_s.back()};
        near_all(t, P::PP_T, P::PP_T_N, "pass_path: t_s ends");
        for (std::size_t j = 0; j < P::PP_F_AT_N; ++j)
            near(best.f_hz[P::PP_IDX[j]], P::PP_F_AT[j], "pass_path: f_hz[" + std::to_string(P::PP_IDX[j]) + "]");
    }
    near(sum_of(best.f_hz), P::PP_F_SUM, "pass_path: sum f_hz");
    // given the pass's own channel, the same path to the bit
    const fr::FrameSpec spec = em::pass_spec(p0);
    const types::Detection det = rc::detection_from_pass(p0);
    const auto chan = std::make_shared<const qrss::track::ChanCapture>(
        rc::channel_for(rc::capture_from_pass(p0), spec, det, {}, p0.f_mix_hz));
    const types::FreqPath given = em::pass_path(p0, chan);
    check::is_true(given.t_s == best.t_s && given.f_hz == best.f_hz, "pass_path: chan given, the same path");
    // CE only
    types::PassResult pl = p0;
    pl.frame = "l-tiny";
    pl.waveform = cfg::WAVEFORM_L;
    throws<std::invalid_argument>([&] { em::pass_path(pl); }, P::ERR_L_PATH, "pass_path: an L pass");
    const em::Rereceived rl = em::rereceive(pl, std::nullopt);
    check::is_true(!rl.pass && !rl.path, "rereceive: an L pass, nothing");
    types::PassResult nc = p0;
    nc.ch.clear();
    const em::Rereceived rn = em::rereceive(nc, std::nullopt);
    check::is_true(!rn.pass && !rn.path, "rereceive: no capture, nothing");
}

void test_rereceive(const Fixture& fx) {
    const fs::path dir = fresh_dir("rereceive");
    Cleanup cleanup{dir};
    qs::Store st(util::path_text(dir / "qrss"), clock_t0());
    fill(st, fx.tiny);
    const types::PassResult& p0 = fx.tiny[0];
    const qs::Accumulator acc = st.accumulator(KEY);
    near(em::acc_mean_w_db(acc), P::ACC_MEAN_W_DB, "the accumulator's mean W dB");

    check::current_step = "rereceive plain";
    check_pass(em::rereceive(p0, std::nullopt), PASS_PINS(RR_PLAIN), "rereceive(TINY 0, None)");

    check::current_step = "rereceive prior";
    const types::EmPrior prior0 = em::em_prior(acc, p0.uid, st, p0.q, em::pass_spec(p0));
    near(sum_of(prior0.v), P::PRIOR0_V_SUM, "em_prior: sum v");
    near(sum_of(prior0.x_hat), P::PRIOR0_X_SUM, "em_prior: sum x_hat");
    em::RereceiveArgs a1;
    a1.em_round = 1;
    check_pass(em::rereceive(p0, prior0, a1), PASS_PINS(RR_PRIOR), "rereceive(TINY 0, em_prior, em_round=1)");

    // SHORT: the formula prior, the header, the keying
    const auto& SHORT = fr::get("short");
    const std::vector<double> a = air(SHORT);
    const std::size_t n = a.size();
    std::vector<double> W(n, 2.0), S(n);
    for (std::size_t i = 0; i < n; ++i) S[i] = W[i] * (a[i] + (i % 4 == 0 ? 0.25 : -0.25));
    const types::EmPrior pr_s = em::prior_from(S, W, fx.short_.q);
    em::RereceiveArgs as;
    as.hdr = test_header();
    as.cw_known = rc::CwKnown{qrss::morse::keying_units("K1ABC"), false};
    as.em_round = 2;
    check::current_step = "rereceive SHORT split";
    check_pass(em::rereceive(fx.short_, pr_s, as), PASS_PINS(RR_SHORT_SPLIT), "rereceive(SHORT, split)");
    check::current_step = "rereceive SHORT no split";
    as.split = false;
    check_pass(em::rereceive(fx.short_, pr_s, as), PASS_PINS(RR_SHORT_NOSPLIT), "rereceive(SHORT, no split)");
    check::current_step = "rereceive SHORT no prior";
    as.split = true;
    check_pass(em::rereceive(fx.short_, std::nullopt, as), PASS_PINS(RR_SHORT_NOPRIOR), "rereceive(SHORT, None)");

    // cancel: before any tracker run (the path is given, so the first poll
    // is the first thing rereceive does after reading the capture)
    check::current_step = "rereceive cancel";
    std::atomic<bool> flag{true};
    em::Hooks h;
    h.cancel = &flag;
    em::RereceiveArgs ac;
    ac.path = det_of().path;
    throws<em::Cancelled>([&] { em::rereceive(p0, std::nullopt, ac, h); }, "cancelled", "rereceive: cancel");
    throws<em::Cancelled>([&] { em::rereceive(fx.short_, pr_s, as, h); }, "cancelled", "rereceive SHORT: cancel");
}

void test_em_refine(const Fixture& fx) {
    check::current_step = "em_refine";
    const fs::path dir = fresh_dir("refine");
    Cleanup cleanup{dir};
    qs::Store st(util::path_text(dir / "qrss"), clock_t0());
    fill(st, fx.tiny);
    const std::vector<std::string> uids = st.accumulator(KEY).uids();
    std::vector<std::string> log;
    struct Call {
        std::int64_t done, total;
        std::string what;
    };
    std::vector<Call> calls;
    em::Hooks h;
    h.log = [&](const std::string& s) { log.push_back(s); };
    h.progress = [&](std::int64_t d, std::int64_t t, const std::string& w) { calls.push_back({d, t, w}); };
    em::EmRefineArgs args;
    args.rounds = 3;
    const std::vector<double> hist = em::em_refine(st, KEY, args, h);
    near_all(hist, P::EMR_HIST, P::EMR_HIST_N, "em_refine: history");
    check::equal(static_cast<std::int64_t>(log.size()), P::EMR_NLOG, "em_refine: log lines");
    for (std::size_t i = 0; i < std::min(log.size(), static_cast<std::size_t>(P::EMR_NLOG)); ++i) {
        // "round <r> pass <uid>: mean W <+x.xx> -> <+y.yy> dB"
        const std::string want_head = "round " + std::to_string(P::EMR_LOG_ROUND[i]) + " pass " +
                                      uids[static_cast<std::size_t>(P::EMR_LOG_MEMBER[i])] + ": mean W ";
        const std::string& s = log[i];
        const bool head = s.rfind(want_head, 0) == 0;
        check::is_true(head, "em_refine: log line " + std::to_string(i) + " names round and pass");
        if (!head) {
            std::fprintf(stderr, "   got '%s'\n", s.c_str());
            continue;
        }
        const std::string rest = s.substr(want_head.size());
        const auto arrow = rest.find(" -> ");
        const bool shape = arrow != std::string::npos && rest.size() > 3 && rest.substr(rest.size() - 3) == " dB";
        check::is_true(shape, "em_refine: log line " + std::to_string(i) + " shape");
        if (!shape) continue;
        const std::string b = rest.substr(0, arrow), af = rest.substr(arrow + 4, rest.size() - 3 - arrow - 4);
        check::is_true((b[0] == '+' || b[0] == '-') && b.size() - b.find('.') == 3, "log: before is %+.2f");
        check::is_true((af[0] == '+' || af[0] == '-') && af.size() - af.find('.') == 3, "log: after is %+.2f");
        near(std::stod(b), P::EMR_LOG_BEFORE[i], "log " + std::to_string(i) + ": before", LOG_TOL);
        near(std::stod(af), P::EMR_LOG_AFTER[i], "log " + std::to_string(i) + ": after", LOG_TOL);
    }
    const qs::Accumulator acc = st.accumulator(KEY);
    check::equal(acc.members.size(), std::size_t{3}, "em_refine: members");
    for (std::size_t i = 0; i < acc.members.size(); ++i) {
        check::equal(qs::py_int(acc.members[i].at("em_round")), P::EMR_EM_ROUND[i],
                     "em_refine: member " + std::to_string(i) + " em_round");
        near(acc.members[i].at("mean_w_db").as_double(), P::EMR_MEMBER_W_DB[i],
             "em_refine: member " + std::to_string(i) + " mean W dB");
        // the pass file says the same
        const types::PassResult p = st.load_pass(uids[i]);
        check::equal(p.em_round, P::EMR_EM_ROUND[i], "em_refine: pass file " + std::to_string(i) + " em_round");
    }
    // progress: one call per pass re-received, total rounds x 3; then, as
    // the rounds stopped early, (done, done, "converged after round <r>")
    const std::int64_t ran = static_cast<std::int64_t>(hist.size()) - 1;
    check::equal(static_cast<std::int64_t>(calls.size()), P::EMR_NLOG + (ran < args.rounds ? 1 : 0),
                 "em_refine: progress calls");
    for (std::size_t i = 0; i < calls.size() && i < static_cast<std::size_t>(P::EMR_NLOG); ++i) {
        check::equal(calls[i].done, static_cast<std::int64_t>(i + 1), "progress: done");
        check::equal(calls[i].total, args.rounds * 3, "progress: total");
        check::equal(calls[i].what,
                     "round " + std::to_string(P::EMR_LOG_ROUND[i]) + " pass " +
                         uids[static_cast<std::size_t>(P::EMR_LOG_MEMBER[i])],
                     "progress: what");
    }
    if (ran < args.rounds && !calls.empty()) {
        check::equal(calls.back().done, P::EMR_NLOG, "progress: converged, done");
        check::equal(calls.back().total, P::EMR_NLOG, "progress: converged, total = done");
        check::equal(calls.back().what, "converged after round " + std::to_string(ran), "progress: converged, what");
    }

    check::current_step = "em_refine rounds 0";
    calls.clear();
    log.clear();
    args.rounds = 0;
    const std::vector<double> h0 = em::em_refine(st, KEY, args, h);
    near_all(h0, P::EMR0_HIST, P::EMR0_HIST_N, "em_refine(rounds=0): history");
    check::is_true(calls.empty() && log.empty(), "em_refine(rounds=0): no progress, no log");
    args.rounds = -2;
    check::equal(em::em_refine(st, KEY, args, h).size(), std::size_t{1}, "em_refine(rounds=-2): history");
    check::is_true(calls.empty() && log.empty(), "em_refine(rounds=-2): no progress, no log");

    check::current_step = "em_refine refusals";
    throws<qs::KeyError>([&] { em::em_refine(st, qs::Key{"NOPE", 7}, {}, h); }, std::string("KeyError: ") + P::ERR_NO_ACC,
                         "em_refine: no accumulator");
}

// Every file under root loads in a fresh Store; no temporary file left.
void store_loads(const fs::path& root, const std::string& what) {
    qs::Store st(util::path_text(root), clock_t0());
    int passes = 0, accs = 0;
    for (const auto& e : fs::recursive_directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        const std::string name = util::path_text(e.path().filename());
        check::is_true(name.find(".tmp") == std::string::npos, what + ": no temporary file (" + name + ")");
        const std::string dir = util::path_text(e.path().parent_path().filename());
        if (dir == "passes" && e.path().extension() == ".npz") {
            st.load_pass(util::path_text(e.path().stem()));
            ++passes;
        }
        if (dir == "acc" && e.path().extension() == ".json") ++accs;
    }
    check::equal(passes, 3, what + ": pass files");
    check::equal(accs, 1, what + ": accumulator files");
    const qs::Accumulator acc = st.accumulator(KEY);
    check::equal(acc.members.size(), std::size_t{3}, what + ": members");
}

void test_cancel(const Fixture& fx) {
    check::current_step = "em_refine cancel";
    {
        // set before the call: nothing is read past the accumulator
        const fs::path dir = fresh_dir("cancel0");
        Cleanup cleanup{dir};
        qs::Store st(util::path_text(dir / "qrss"), clock_t0());
        fill(st, fx.tiny);
        std::atomic<bool> flag{true};
        em::Hooks h;
        h.cancel = &flag;
        throws<em::Cancelled>([&] { em::em_refine(st, KEY, {}, h); }, "cancelled", "em_refine: cancel before");
        const qs::Accumulator acc = st.accumulator(KEY);
        for (const auto& m : acc.members) check::equal(qs::py_int(m.at("em_round")), std::int64_t{0}, "untouched");
        store_loads(dir / "qrss", "cancel before");
    }
    {
        // set from the progress hook after the first pass: the next poll throws
        const fs::path dir = fresh_dir("cancel1");
        Cleanup cleanup{dir};
        qs::Store st(util::path_text(dir / "qrss"), clock_t0());
        fill(st, fx.tiny);
        const std::vector<std::string> uids = st.accumulator(KEY).uids();
        std::atomic<bool> flag{false};
        std::vector<std::string> log;
        int progress = 0;
        em::Hooks h;
        h.cancel = &flag;
        h.log = [&](const std::string& s) { log.push_back(s); };
        h.progress = [&](std::int64_t, std::int64_t, const std::string&) {
            ++progress;
            flag.store(true);
        };
        throws<em::Cancelled>([&] { em::em_refine(st, KEY, {}, h); }, "cancelled", "em_refine: cancel from progress");
        check::equal(progress, 1, "cancel from progress: one pass done");
        check::equal(log.size(), std::size_t{1}, "cancel from progress: one line logged");
        const qs::Accumulator acc = st.accumulator(KEY);
        check::equal(qs::py_int(acc.members[0].at("em_round")), std::int64_t{1}, "cancel: the first stored");
        check::equal(qs::py_int(acc.members[1].at("em_round")), std::int64_t{0}, "cancel: the second not");
        check::equal(qs::py_int(acc.members[2].at("em_round")), std::int64_t{0}, "cancel: the third not");
        store_loads(dir / "qrss", "cancel from progress");
        qs::Store fresh(util::path_text(dir / "qrss"), clock_t0());
        check::equal(fresh.load_pass(uids[0]).em_round, std::int64_t{1}, "cancel: the first's file re-received");
        check::equal(fresh.load_pass(uids[1]).em_round, std::int64_t{0}, "cancel: the second's file as it was");
    }
}

void test_seams(const Fixture& fx) {
    check::current_step = "em_refine seams";
    const fs::path dir = fresh_dir("seams");
    Cleanup cleanup{dir};
    qs::Store st(util::path_text(dir / "qrss"), clock_t0());
    fill(st, fx.tiny);
    // a fourth member without a capture (a merged pass's form): skipped
    types::PassResult nc = fx.tiny[2];
    nc.ch.clear();
    nc.q = P::Q + 40;
    nc.uid = types::pass_uid(nc.q, nc.f_hz, nc.z);
    st.add_pass(nc);
    st.attach(KEY, nc.uid, 0, "test", 0, 0xD1D8);
    const std::vector<std::string> uids = st.accumulator(KEY).uids();
    check::equal(uids.size(), std::size_t{4}, "seams: four members");
    struct Call {
        std::string uid;
        std::int64_t round;
        bool has_path;
        double path_t0;
        bool hdr, cw;
        std::optional<std::string> estimator;
    };
    std::vector<Call> calls;
    em::EmSeams seams;
    seams.rereceive = [&](const types::PassResult& p, const types::EmPrior& prior,
                          const std::optional<hd::HeaderFields>& hdr, const std::optional<rc::CwKnown>& cw,
                          const std::optional<std::string>& est, std::int64_t r,
                          const std::optional<types::FreqPath>& path) {
        check::equal(prior.x_hat.size(), p.z.size(), "seams: the prior's length");
        calls.push_back({p.uid, r, path.has_value(), path ? path->t_s.front() : 0.0, hdr.has_value(), cw.has_value(),
                         est});
        em::Rereceived out;
        // the path handed back names the round and the member, so the next
        // round's call can be checked to receive it
        const double tag = 100.0 * static_cast<double>(r) + static_cast<double>(calls.size());
        out.path = types::FreqPath{{tag, tag + 1.0}, {1500.0, 1500.0}, {1.0, 1.0}};
        if (p.uid == uids[1]) return out;   // no pass: not stored, not logged
        types::PassResult np = p;
        np.em_round = r;
        out.pass = std::move(np);
        return out;
    };
    std::vector<std::string> log;
    std::vector<std::int64_t> totals;
    em::Hooks h;
    h.log = [&](const std::string& s) { log.push_back(s); };
    h.progress = [&](std::int64_t, std::int64_t t, const std::string&) { totals.push_back(t); };
    em::EmRefineArgs args;
    args.rounds = 2;
    args.tol_db = -std::numeric_limits<double>::infinity();   // a flat history runs every round
    args.estimator = "plain";
    const std::vector<double> hist = em::em_refine(st, KEY, args, h, seams);
    check::equal(hist.size(), std::size_t{3}, "seams: history");
    check::equal(calls.size(), std::size_t{6}, "seams: calls (3 members with a capture x 2 rounds)");
    for (std::size_t i = 0; i < calls.size(); ++i) {
        const std::size_t m = i % 3;
        check::equal(calls[i].uid, uids[m], "seams: call " + std::to_string(i) + " member order");
        check::equal(calls[i].round, static_cast<std::int64_t>(i / 3 + 1), "seams: call round");
        check::is_true(!calls[i].hdr, "seams: TINY has no header to know");
        check::is_true(calls[i].estimator == std::optional<std::string>("plain"), "seams: the estimator handed on");
        if (i < 3) {
            check::is_true(!calls[i].has_path, "seams: round 1 measures its own path");
        } else {
            check::is_true(calls[i].has_path, "seams: round 2 gets last round's path");
            check::equal(calls[i].path_t0, 100.0 + static_cast<double>(m + 1), "seams: the member's own path");
        }
    }
    check::equal(log.size(), std::size_t{4}, "seams: two stored a round, logged");
    check::equal(totals.size(), std::size_t{4}, "seams: progress per stored pass");
    for (const std::int64_t t : totals) check::equal(t, std::int64_t{6}, "seams: total = 2 rounds x 3 with a capture");
    const qs::Accumulator acc = st.accumulator(KEY);
    check::equal(qs::py_int(acc.members[0].at("em_round")), std::int64_t{2}, "seams: member 0 stored");
    check::equal(qs::py_int(acc.members[1].at("em_round")), std::int64_t{0}, "seams: member 1 never stored");
    check::equal(qs::py_int(acc.members[3].at("em_round")), std::int64_t{0}, "seams: no capture, skipped");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(600.0, "qrss_em_refine");
    try {
        const Fixture fx = build();
        test_pass_path(fx);
        test_rereceive(fx);
        test_em_refine(fx);
        test_cancel(fx);
        test_seams(fx);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL in %s: %s\n", check::current_step.load(), e.what());
        return 1;
    }
    std::printf("worst relative error against the pins: %.3g\n", worst_rel);
    return check::report("qrss_em_refine");
}
