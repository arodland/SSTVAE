// native/core/qrss/receiver (the per-pass half of sstvae/qrss/receiver.py)
// against values pinned from the Python reference by gen_receiver_pins.py
// (receiver_pins.hpp; its docstring has the formulas).
//
// The passes are synthetic CH-rate captures -- a stored pass's form, so
// receive_pass goes through prepare's pass-through, channel_for's CH branch
// and gate V -- of TINY, SHORT, l-tiny and l-short at about -6 dB: the
// frame (frame::assemble + ce::phase_grid, or frame::assemble_l +
// lwave::composite) plus an integer-hash noise, both computed here to the
// bit (the captures' digests say so). What a pass is pinned to is its
// primitives' rounding (the tracker, the fit, the demodulator: 1e-6) and
// its decisions exactly (a pass or none, the header, the callsign verdict,
// the trackers' q, the report's flags). decode_header, confidence_db,
// mean_w_db, centre_hz, detection_from_pass and the refusals are bitwise.
//
// Also here: a raw FE capture through prepare (blank, normalise) and
// channel_for, the seams (each wrapped around the C++ itself: the same pass
// to the bit, every seam called), four threads, and the round-B keying of a
// CE frame (a (192,) mask broadcast to the windows: a SHORT pass's round B
// refused it before the C++ broadcast it).
//
// tests/test_qrss_native_receiver.py compares against the reference
// directly, on simulated passes.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/frame.hpp"
#include "qrss/frontend.hpp"
#include "qrss/header.hpp"
#include "qrss/ltrack_timing.hpp"
#include "qrss/lwave.hpp"
#include "qrss/rational.hpp"
#include "qrss/receiver.hpp"
#include "qrss/track_timing.hpp"
#include "qrss/types.hpp"
#include "qrss_check.hpp"
#include "receiver_pins.hpp"

using namespace sstvae;
namespace rc = sstvae::qrss::receiver;
namespace fr = sstvae::qrss::frame;
namespace fe = sstvae::qrss::frontend;
namespace tk = sstvae::qrss::track;
namespace lw = sstvae::qrss::lwave;
namespace hd = sstvae::qrss::header;
namespace types = sstvae::qrss::types;
namespace cfg = sstvae::qrss::config;
namespace P = receiver_pins;
using cd = std::complex<double>;
using cf = std::complex<float>;
using qrss::Rational;

namespace {

constexpr std::int64_t Q = 1954752;

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
void near(double got, double want, double tol, const std::string& what) {
    const bool ok = std::abs(got - want) <= tol * std::max(1.0, std::abs(want));
    check::is_true(ok, what);
    if (!ok) std::fprintf(stderr, "   %s: got %.17g want %.17g\n", what.c_str(), got, want);
}

// --- the inputs ------------------------------------------------------------------------------

// hnoise(n, seed): a 32-bit integer hash of (i, seed), over 2^32, minus 1/2.
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

std::vector<cd> noise_of(std::int64_t n, double scale) {
    const auto u1 = hnoise(n, 1), u2 = hnoise(n, 2), u3 = hnoise(n, 3), u4 = hnoise(n, 4);
    const auto u5 = hnoise(n, 5), u6 = hnoise(n, 6), u7 = hnoise(n, 7), u8 = hnoise(n, 8);
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

hd::HeaderFields test_header_v1() {
    hd::HeaderFields h;
    h.callsign = "W1AW";
    h.grid = std::nullopt;
    h.picture_id = 0x0BADF00D;
    h.mode = 1;
    h.segment = 1;
    h.codec_id = 0x1234;
    h.version = 1;
    return h;
}

std::vector<double> latents(const fr::FrameSpec& spec, double a) {
    std::vector<double> x;
    for (std::int64_t k = 0; k < spec.n_data(); ++k) x.push_back((k * 5 + 1) % 3 == 0 ? a : -a);
    return x;
}

rc::Prepared ce_cap(const fr::FrameSpec& spec, std::int64_t n) {
    const std::vector<std::uint8_t> bits = hd::encode(test_header());
    const fr::HdrBits hb = fr::HdrBits::of(bits);
    const auto sym = fr::assemble(spec, spec.n_hdr() ? &hb : nullptr, latents(spec, 1.0));
    const auto phi = qrss::ce::phase_grid(250, -3000, n, sym, spec);
    const auto noise = noise_of(n, 0.75);
    rc::Prepared p;
    const double half_milli = 0.5 * 0.001;
    for (std::int64_t i = 0; i < n; ++i) {
        const double di = static_cast<double>(i);
        const double sq = (di / 250) * (di / 250);
        const double ph = phi[static_cast<std::size_t>(i)] + 2 * std::numbers::pi * (0.02 * di / 250 + half_milli * sq);
        const cd z = cd(std::cos(ph), std::sin(ph)) + noise[static_cast<std::size_t>(i)];
        p.fe.emplace_back(static_cast<float>(z.real()), static_cast<float>(z.imag()));
    }
    p.q = Q;
    p.t0_index = 3002.5;
    p.fs = cfg::CH_FS;
    return p;
}

rc::Prepared l_cap(const fr::FrameSpec& spec, std::int64_t n) {
    const std::vector<std::uint8_t> bits = hd::encode(test_header());
    const fr::HdrBits hb = fr::HdrBits::of(bits);
    const auto sym = fr::assemble_l(spec, spec.n_hdr() ? &hb : nullptr, latents(spec, 0.5));
    const auto sig = lw::composite(sym, spec, cfg::CH_FS, -3000, n, nullptr, 0.0);
    const auto noise = noise_of(n, 0.75);
    rc::Prepared p;
    for (std::size_t i = 0; i < sig.size(); ++i) {
        const cd z = sig[i] + noise[i];
        p.fe.emplace_back(static_cast<float>(z.real()), static_cast<float>(z.imag()));
    }
    p.q = Q;
    p.t0_index = 3000.0;
    p.fs = cfg::CH_FS;
    return p;
}

types::FreqPath flat_path() { return types::FreqPath{{-10.0, 0.0, 1800.0}, {1500.0, 1500.0, 1500.0}, {1.0, 1.0, 1.0}}; }

types::Detection det_of(std::int64_t wf, double f = 1500.0) {
    types::Detection d;
    d.f_hz = f;
    d.path = flat_path();
    d.z_ref = std::numeric_limits<double>::quiet_NaN();
    d.method = "tbd";
    d.waveform = wf;
    return d;
}

// --- a pass against its pins -------------------------------------------------------------------

struct Span {
    const double* p;
    std::size_t n;
};

struct PassPins {
    std::int64_t ntrack;
    Span track_q, track_keying_n;
    std::int64_t plain;
    Span f, report;
    const std::int64_t* flags;
    std::int64_t nz;
    Span z_at, w_at;
    double z_sum, w_sum;
    std::int64_t nllr;
    double llr_abs;
    std::int64_t npsi, psi_zero;
    double psi_sum, ch_t0;
    const std::int64_t* f_mix;
    const char* uid_head;
    const char* frame;
    double mean_w_db;
    Span conf;
    const char* hdr_call = nullptr;
    const char* hdr_grid = nullptr;
    const std::int64_t* hdr_ints = nullptr;
    bool cw = false;
    bool cw_has_text = false;
    const char* cw_text = nullptr;
    const char* cw_keying = nullptr;
    std::int64_t cw_agrees = -1;
    double cw_z = 0.0;
};

#define SPAN(X) Span{P::X, P::X##_N}
#define PASS_PINS(T)                                                                                               \
    PassPins{P::T##_NTRACK,  SPAN(T##_TRACK_Q), SPAN(T##_TRACK_KEYING_N), P::T##_PLAIN,     SPAN(T##_F),     SPAN(T##_REPORT),               \
             P::T##_FLAGS,   P::T##_NZ,         SPAN(T##_Z_AT),   SPAN(T##_W_AT),  P::T##_Z_SUM,                   \
             P::T##_W_SUM,   P::T##_NLLR,       P::T##_LLR_ABS,   P::T##_NPSI,     P::T##_PSI_ZERO,                \
             P::T##_PSI_SUM, P::T##_CH_T0,      P::T##_F_MIX,     P::T##_UID_HEAD, P::T##_FRAME,                   \
             P::T##_MEAN_W_DB, SPAN(T##_CONF)}

PassPins with_hdr(PassPins p, const char* call, const char* grid, const std::int64_t* ints) {
    p.hdr_call = call;
    p.hdr_grid = grid;
    p.hdr_ints = ints;
    return p;
}

PassPins with_cw(PassPins p, std::int64_t has_text, const char* t, const char* keying, std::int64_t agrees,
                 double z) {
    p.cw = true;
    p.cw_has_text = has_text != 0;
    p.cw_text = t;
    p.cw_keying = keying;
    p.cw_agrees = agrees;
    p.cw_z = z;
    return p;
}

constexpr std::size_t IDX[] = {0, 1, 7, 100, 511, 1023};

void check_conf(const std::vector<std::optional<double>>& got, Span want, double tol, const std::string& what) {
    check::equal(got.size(), want.n, what + " length");
    for (std::size_t i = 0; i < std::min(got.size(), want.n); ++i) {
        const bool none = std::isnan(want.p[i]);
        check::equal(!got[i].has_value(), none, what + " None at " + std::to_string(i));
        if (got[i] && !none) {
            const bool ok = std::abs(*got[i] - want.p[i]) <= tol;
            check::is_true(ok, what + " at " + std::to_string(i));
            if (!ok) std::fprintf(stderr, "   %s[%zu]: got %.17g want %.17g\n", what.c_str(), i, *got[i], want.p[i]);
        }
    }
}

void check_pass(const rc::PassOutcome& o, const PassPins& w, const std::string& tag) {
    check::is_true(o.pass.has_value(), tag + ": a pass");
    if (!o.pass) return;
    const types::PassResult& p = *o.pass;
    check::equal(static_cast<std::int64_t>(o.tracks.size()), w.ntrack, tag + ": tracks");
    for (std::size_t i = 0; i < std::min(o.tracks.size(), w.track_q.n); ++i)
        check::equal(o.tracks[i]->q, w.track_q.p[i], tag + ": track q " + std::to_string(i));
    for (std::size_t i = 0; i < std::min(o.tracks.size(), w.track_keying_n.n); ++i) {
        const auto& k = o.tracks[i]->keying;
        check::equal(k ? static_cast<double>(k->size()) : -1.0, w.track_keying_n.p[i],
                     tag + ": track keying size " + std::to_string(i));
    }
    check::equal(static_cast<std::int64_t>(o.plain), w.plain, tag + ": plain");
    check::equal(p.frame, std::string(w.frame), tag + ": frame");
    check::equal(p.header.has_value(), w.hdr_call != nullptr, tag + ": a header");
    if (p.header && w.hdr_call) {
        check::equal(p.header->callsign, std::string(w.hdr_call), tag + ": callsign");
        check::equal(p.header->grid.value_or(""), std::string(w.hdr_grid), tag + ": grid");
        const std::int64_t got[] = {p.header->picture_id, p.header->mode,    p.header->segment,
                                    p.header->codec_id,   p.header->version, p.header->reserved};
        for (std::size_t i = 0; i < 6; ++i) check::equal(got[i], w.hdr_ints[i], tag + ": header field " + std::to_string(i));
    }
    check::equal(p.cw.has_value(), w.cw, tag + ": a callsign reading");
    if (p.cw && w.cw) {
        check::equal(p.cw->text.has_value(), w.cw_has_text, tag + ": cw text or None");
        if (p.cw->text) check::equal(*p.cw->text, std::string(w.cw_text), tag + ": cw text");
        check::equal(p.cw->keying, std::string(w.cw_keying), tag + ": cw keying");
        check::equal(p.cw->agrees ? static_cast<std::int64_t>(*p.cw->agrees) : std::int64_t{-1}, w.cw_agrees,
                     tag + ": cw agrees");
        near(p.cw->z_match, w.cw_z, 1e-6, tag + ": cw z_match");
    }
    const double f[] = {p.f_hz, p.timing.tau0, p.timing.ppm, p.timing.gamma, p.timing.z};
    const char* fn[] = {"f_hz", "tau0", "ppm", "gamma", "timing z"};
    for (std::size_t i = 0; i < 5; ++i) near(f[i], w.f.p[i], 1e-6, tag + ": " + fn[i]);
    const types::TrackReport& r = p.report;
    const double rep[] = {r.offset_hz, r.drift_hz_per_min, r.wander_hz_rms, r.doppler_hz,
                          r.ppm,       r.snr2500_db,       r.z_ref,         r.kappa};
    for (std::size_t i = 0; i < 8; ++i) near(rep[i], w.report.p[i], 1e-6, tag + ": report " + std::to_string(i));
    check::equal(static_cast<std::int64_t>(r.slip_free), w.flags[0], tag + ": slip_free");
    check::equal(static_cast<std::int64_t>(r.suspect), w.flags[1], tag + ": suspect");
    check::equal(static_cast<std::int64_t>(p.z.size()), w.nz, tag + ": n z");
    check::equal(p.w.size(), p.z.size(), tag + ": n w");
    if (static_cast<std::int64_t>(p.z.size()) == w.nz) {
        for (std::size_t i = 0; i < 6; ++i) {
            near(p.z[IDX[i]], w.z_at.p[i], 1e-5, tag + ": z at " + std::to_string(IDX[i]));
            near(p.w[IDX[i]], w.w_at.p[i], 1e-5, tag + ": w at " + std::to_string(IDX[i]));
        }
        double zs = 0.0, ws = 0.0;
        for (std::size_t i = 0; i < p.z.size(); ++i) {
            zs += p.z[i];
            ws += p.w[i];
        }
        near(zs, w.z_sum, 1e-4, tag + ": sum z");
        near(ws, w.w_sum, 1e-6, tag + ": sum w");
    }
    check::equal(static_cast<std::int64_t>(p.hdr_llr.size()), w.nllr, tag + ": n hdr_llr");
    double la = 0.0;
    for (const float x : p.hdr_llr) la += std::abs(static_cast<double>(x));
    near(la, w.llr_abs, 1e-6, tag + ": sum |hdr_llr|");
    check::equal(static_cast<std::int64_t>(p.psi.size()), w.npsi, tag + ": n psi");
    std::int64_t pz = 0;
    double ps = 0.0;
    for (const float x : p.psi) {
        pz += x == 0.0f;
        ps += x;
    }
    check::equal(pz, w.psi_zero, tag + ": psi zeros");
    near(ps, w.psi_sum, 1e-6, tag + ": sum psi");
    check::equal(p.ch_t0_index, w.ch_t0, tag + ": ch_t0_index");
    check::equal(p.ch_fs, std::int64_t{cfg::CH_FS}, tag + ": ch_fs");
    check::equal(p.f_mix_hz.num(), w.f_mix[0], tag + ": f_mix numerator");
    check::equal(p.f_mix_hz.den(), w.f_mix[1], tag + ": f_mix denominator");
    check::equal(p.uid.substr(0, p.uid.rfind('_')), std::string(w.uid_head), tag + ": uid head");
    check::equal(p.q, Q, tag + ": q");
    near(rc::mean_w_db(p), w.mean_w_db, 1e-6, tag + ": mean_w_db");
    // confidence_db rounds to 0.1 dB: a 1e-7 change of a bin's mean could
    // move one by a step, never more
    check_conf(rc::confidence_db(p), w.conf, 0.1 + 1e-9, tag + ": confidence_db");
}

rc::PassArgs args_ch() {
    rc::PassArgs a;
    a.f_mix_hz = Rational(0);
    a.return_tracks = true;
    return a;
}

rc::PassOutcome run(const rc::Prepared& prep, const fr::FrameSpec& spec, const rc::PassArgs& a,
                    const rc::ReceiverSeams& s = {}) {
    return rc::receive_pass(&prep, spec, det_of(static_cast<std::int64_t>(spec.waveform())), a, s);
}

// --- the tests --------------------------------------------------------------------------------

void test_passes() {
    {
        const auto n = noise_of(17000, 0.75);
        qrss_check::digest_is(n, P::NOISE, "the hash noise");
    }
    const auto& TINY = fr::get("tiny");
    const auto& SHORT = fr::get("short");
    const auto& LTINY = fr::get_l("l-tiny");
    const auto& LSHORT = fr::get_l("l-short");
    const rc::Prepared tiny = ce_cap(TINY, 17000), short_ = ce_cap(SHORT, 72000);
    const rc::Prepared ltiny = l_cap(LTINY, 17500), lshort = l_cap(LSHORT, 72000);
    qrss_check::digest_is(tiny.fe, P::CAP_TINY, "the TINY capture");
    qrss_check::digest_is(short_.fe, P::CAP_SHORT, "the SHORT capture");
    qrss_check::digest_is(ltiny.fe, P::CAP_LTINY, "the l-tiny capture");
    qrss_check::digest_is(lshort.fe, P::CAP_LSHORT, "the l-short capture");
    static_assert(P::TINY_HAS == 1 && P::SHORT_HAS == 1 && P::LTINY_HAS == 1 && P::LSHORT_HAS == 1);
    static_assert(P::TINY_HDR == 0 && P::TINY_CW == 0 && P::LTINY_HDR == 0 && P::LTINY_CW == 0);
    static_assert(P::SHORT_HDR == 1 && P::SHORT_CW == 1 && P::LSHORT_HDR == 1 && P::LSHORT_CW == 1);

    check_pass(run(tiny, TINY, args_ch()), PASS_PINS(TINY), "TINY");
    check_pass(run(short_, SHORT, args_ch()),
               with_cw(with_hdr(PASS_PINS(SHORT), P::SHORT_HDR_CALL, P::SHORT_HDR_GRID, P::SHORT_HDR_INTS),
                       P::SHORT_CW_HAS_TEXT, P::SHORT_CW_TEXT, P::SHORT_CW_KEYING, P::SHORT_CW_AGREES, P::SHORT_CW_Z),
               "SHORT");
    check_pass(run(ltiny, LTINY, args_ch()), PASS_PINS(LTINY), "l-tiny");
    check_pass(run(lshort, LSHORT, args_ch()),
               with_cw(with_hdr(PASS_PINS(LSHORT), P::LSHORT_HDR_CALL, P::LSHORT_HDR_GRID, P::LSHORT_HDR_INTS),
                       P::LSHORT_CW_HAS_TEXT, P::LSHORT_CW_TEXT, P::LSHORT_CW_KEYING, P::LSHORT_CW_AGREES,
                       P::LSHORT_CW_Z),
               "l-short");

    // the variants
    rc::PassArgs a = args_ch();
    a.erase_s = {{10.0, 20.0}};
    a.estimator = "plain";
    static_assert(P::TINY_ERASED_HDR == 0 && P::TINY_ERASED_CW == 0);
    check_pass(run(tiny, TINY, a), PASS_PINS(TINY_ERASED), "TINY erased, plain");
    a = args_ch();
    a.round_b = false;
    static_assert(P::SHORT_A_HDR == 1 && P::SHORT_A_CW == 1);
    check_pass(run(short_, SHORT, a),
               with_cw(with_hdr(PASS_PINS(SHORT_A), P::SHORT_A_HDR_CALL, P::SHORT_A_HDR_GRID, P::SHORT_A_HDR_INTS),
                       P::SHORT_A_CW_HAS_TEXT, P::SHORT_A_CW_TEXT, P::SHORT_A_CW_KEYING, P::SHORT_A_CW_AGREES,
                       P::SHORT_A_CW_Z),
               "SHORT round A only");
    a = args_ch();
    a.exclude_hz = {1530.0};
    a.erase_s = {{5.0, 12.5}};
    static_assert(P::LTINY_EX_HDR == 0 && P::LTINY_EX_CW == 0);
    check_pass(run(ltiny, LTINY, a), PASS_PINS(LTINY_EX), "l-tiny, a neighbour excluded, erased");

    // return_tracks off: the same pass, no tracks
    a = args_ch();
    a.return_tracks = false;
    const rc::PassOutcome nt = run(tiny, TINY, a);
    const rc::PassOutcome wt = run(tiny, TINY, args_ch());
    check::is_true(nt.pass && nt.tracks.empty(), "return_tracks off: no tracks");
    check::is_true(nt.pass && wt.pass && nt.pass->z == wt.pass->z && nt.pass->w == wt.pass->w &&
                       nt.pass->uid == wt.pass->uid,
                   "return_tracks off: the same pass to the bit");

    // the noise alone: gate V refuses
    static_assert(P::NOISE_TINY_HAS == 0 && P::NOISE_LTINY_HAS == 0);
    for (const auto* spec : {&TINY, &LTINY}) {
        rc::Prepared nz;
        for (const cd z : noise_of(17500, 0.75)) nz.fe.emplace_back(static_cast<float>(z.real()), static_cast<float>(z.imag()));
        nz.q = Q;
        nz.t0_index = spec->waveform() == fr::Waveform::L ? 3000.0 : 3002.5;
        nz.fs = cfg::CH_FS;
        const rc::PassOutcome o = run(nz, *spec, args_ch());
        check::is_true(!o.pass && o.tracks.empty(), "noise alone: refused (" + spec->name() + ")");
    }
}

void test_seams() {
    const auto& TINY = fr::get("tiny");
    const auto& LTINY = fr::get_l("l-tiny");
    for (const auto* spec : {&TINY, &LTINY}) {
        const bool L = spec->waveform() == fr::Waveform::L;
        const rc::Prepared prep = L ? l_cap(*spec, 17500) : ce_cap(*spec, 17000);
        const rc::PassOutcome ref = run(prep, *spec, args_ch());
        int n_prep = 0, n_chan = 0, n_verify = 0, n_verify_l = 0, n_psd = 0;
        rc::ReceiverSeams s;
        s.prepare = [&](rc::CaptureRef c) {
            ++n_prep;
            return rc::prepare(c);
        };
        s.channel_for = [&](const rc::Prepared& p, const fr::FrameSpec& sp, const types::Detection& d,
                            std::span<const double> ex, const std::optional<Rational>& fm) {
            ++n_chan;
            return std::make_shared<const tk::ChanCapture>(rc::channel_for(p, sp, d, ex, fm));
        };
        s.noise_psd = [&](std::span<const cf> ch, std::int64_t fs, std::span<const double> ex, double win_s,
                          std::optional<double> audio_hz) {
            ++n_psd;
            fe::NoisePsdOptions o;
            o.audio_hz = audio_hz;
            return fe::noise_psd(ch, fs, ex, win_s, o);
        };
        s.verify = [&](std::shared_ptr<const tk::ChanCapture> chan, const fr::FrameSpec& sp, const types::Detection& d,
                       std::span<const std::pair<double, double>> er) {
            ++n_verify;
            tk::VerifyArgs v;
            v.erase_s.assign(er.begin(), er.end());
            v.noise_psd = s.noise_psd;
            return tk::verify(std::move(chan), sp, d, v);
        };
        s.verify_l = [&](std::shared_ptr<const tk::ChanCapture> chan, const fr::FrameSpec& sp,
                         const types::Detection& d, std::span<const std::pair<double, double>> er) {
            ++n_verify_l;
            tk::VerifyArgs v;
            v.erase_s.assign(er.begin(), er.end());
            v.noise_psd = s.noise_psd;
            return qrss::ltrack::verify_l(std::move(chan), sp, d, v);
        };
        const rc::PassOutcome got = run(prep, *spec, args_ch(), s);
        const std::string what = "seams (" + spec->name() + ")";
        check::is_true(got.pass && ref.pass, what + ": a pass");
        if (got.pass && ref.pass) {
            check::is_true(got.pass->z == ref.pass->z && got.pass->w == ref.pass->w &&
                               got.pass->psi == ref.pass->psi && got.pass->uid == ref.pass->uid &&
                               got.pass->f_hz == ref.pass->f_hz && got.pass->timing.tau0 == ref.pass->timing.tau0,
                           what + ": the same pass to the bit");
        }
        check::equal(n_prep, 1, what + ": prepare called once");
        check::equal(n_chan, 1, what + ": channel_for called once");
        check::equal(n_verify, L ? 0 : 1, what + ": verify");
        check::equal(n_verify_l, L ? 1 : 0, what + ": verify_l");
        check::is_true(n_psd >= 1, what + ": noise_psd reached");
    }
}

void test_threads() {
    const auto& TINY = fr::get("tiny");
    const rc::Prepared prep = ce_cap(TINY, 17000);
    const rc::PassOutcome ref = run(prep, TINY, args_ch());
    std::vector<std::optional<types::PassResult>> got(4);
    std::vector<std::thread> ts;
    for (std::size_t i = 0; i < got.size(); ++i)
        ts.emplace_back([&, i] { got[i] = run(prep, TINY, args_ch()).pass; });
    for (auto& t : ts) t.join();
    for (const auto& g : got)
        check::is_true(g && ref.pass && g->z == ref.pass->z && g->uid == ref.pass->uid, "four threads: the same pass");
}

void test_refusals() {
    const auto& TINY = fr::get("tiny");
    const auto& LTINY = fr::get_l("l-tiny");
    const rc::Prepared none;
    throws<std::invalid_argument>([&] { (void)rc::receive_pass(&none, LTINY, det_of(cfg::WAVEFORM_CE)); },
                                  P::ERR_CE_AS_L, "a CE detection as l-tiny");
    throws<std::invalid_argument>([&] { (void)rc::receive_pass(&none, TINY, det_of(cfg::WAVEFORM_L)); },
                                  P::ERR_L_AS_CE, "an L detection as TINY");
    throws<std::invalid_argument>([&] { (void)rc::receive_pass(&none, TINY, det_of(7)); }, P::ERR_7_AS_CE,
                                  "a waveform 7 detection");
}

void test_fe_capture() {
    constexpr std::int64_t N_FE = 80 * cfg::FE_FS;
    const auto noise = noise_of(N_FE, 0.25);
    fe::Capture cap;
    for (std::int64_t k = 0; k < N_FE; ++k) {
        const double dk = static_cast<double>(k);
        const double x = 2 * std::numbers::pi * 0.25 * dk / cfg::FE_FS;
        cd z = cd(0.5 * std::cos(x), 0.5 * std::sin(x)) + noise[static_cast<std::size_t>(k)];
        if (k == 100000) z += 50.0;
        if (k == 200001) z += cd(40.0, -30.0);
        cap.fe.emplace_back(static_cast<float>(z.real()), static_cast<float>(z.imag()));
    }
    cap.q = Q;
    cap.t0_index = 48000.0;
    qrss_check::digest_is(cap.fe, P::FE_CAP, "the raw FE capture");
    const rc::Prepared prep = rc::prepare(cap);
    check::is_true(prep.keep.has_value(), "prepare: a keep indicator");
    check::equal(prep.fs, std::int64_t{cfg::FE_FS}, "prepare: fs");
    check::equal(prep.q, Q, "prepare: q");
    check::equal(prep.t0_index, 48000.0, "prepare: t0_index");
    double power = 0.0, keep = 0.0;
    for (const cf z : prep.fe) power += std::norm(cd(z));
    for (const float k : *prep.keep) keep += k;
    near(power, P::PREP_POWER, 1e-9, "prepare: power");
    near(keep, P::PREP_KEEP_SUM, 1e-9, "prepare: sum keep");
    for (std::size_t i = 0; i < P::FE_IDX_N; ++i) {
        const auto j = static_cast<std::size_t>(P::FE_IDX[i]);
        near(prep.fe[j].real(), P::PREP_RE[i], 1e-6, "prepare: re at " + std::to_string(j));
        near(prep.fe[j].imag(), P::PREP_IM[i], 1e-6, "prepare: im at " + std::to_string(j));
        near((*prep.keep)[j], P::PREP_KEEP_AT[i], 1e-6, "prepare: keep at " + std::to_string(j));
    }
    // the shared_ptr form: a Prepared passes through, owning nothing
    const std::shared_ptr<const rc::Prepared> same = rc::prepare(rc::CaptureRef(&prep));
    check::is_true(same.get() == &prep, "prepare of a Prepared: itself");

    const auto& TINY = fr::get("tiny");
    const tk::ChanCapture chan = rc::channel_for(prep, TINY, det_of(cfg::WAVEFORM_CE));
    check::equal(static_cast<std::int64_t>(chan.ch.size()), P::CHAN_N, "channel_for: length");
    check::equal(chan.t0_index, P::CHAN_T0, "channel_for: t0_index");
    check::equal(chan.f_mix.num(), P::CHAN_MIX[0], "channel_for: f_mix numerator");
    check::equal(chan.f_mix.den(), P::CHAN_MIX[1], "channel_for: f_mix denominator");
    double cp = 0.0, scale = 0.0;
    for (const cf z : chan.ch) {
        cp += std::norm(cd(z));
        scale = std::max(scale, std::abs(cd(z)));
    }
    near(cp, P::CHAN_POWER, 1e-6, "channel_for: power");
    for (std::size_t i = 0; i < P::CH_IDX_N && static_cast<std::int64_t>(chan.ch.size()) == P::CHAN_N; ++i) {
        const auto j = static_cast<std::size_t>(P::CH_IDX[i]);
        near(chan.ch[j].real() / scale, P::CHAN_RE[i] / scale, 1e-6, "channel_for: re at " + std::to_string(j));
        near(chan.ch[j].imag() / scale, P::CHAN_IM[i] / scale, 1e-6, "channel_for: im at " + std::to_string(j));
    }
    types::Detection dt = det_of(cfg::WAVEFORM_CE);
    types::Timing tm;
    tm.tau0 = 250.5;
    tm.cov = math::Matrix::identity(2);
    dt.timing = tm;
    const std::vector<double> ex = {1520.0, 1480.5};
    const tk::ChanCapture chan2 = rc::channel_for(prep, TINY, dt, ex);
    check::equal(static_cast<std::int64_t>(chan2.ch.size()), P::CHAN2_N, "channel_for with a timing: length");
    check::equal(chan2.t0_index, P::CHAN2_T0, "channel_for with a timing: t0_index");
    check::equal(chan2.f_mix.num(), P::CHAN2_MIX[0], "channel_for with a timing: f_mix numerator");
    check::equal(chan2.f_mix.den(), P::CHAN2_MIX[1], "channel_for with a timing: f_mix denominator");
    check::is_true(chan2.exclude_hz == ex, "channel_for: exclude_hz kept");
    double cp2 = 0.0;
    for (const cf z : chan2.ch) cp2 += std::norm(cd(z));
    near(cp2, P::CHAN2_POWER, 1e-6, "channel_for with a timing: power");

    rc::Prepared ch_rate;
    ch_rate.fe.assign(10, cf(0.0f, 0.0f));
    ch_rate.q = Q;
    ch_rate.t0_index = 2.5;
    ch_rate.fs = cfg::CH_FS;
    const tk::ChanCapture chan3 = rc::channel_for(ch_rate, TINY, det_of(cfg::WAVEFORM_CE, 1503.3));
    check::equal(chan3.f_mix.num(), P::CHAN3_MIX[0], "channel_for at the CH rate: default mix numerator");
    check::equal(chan3.f_mix.den(), P::CHAN3_MIX[1], "channel_for at the CH rate: default mix denominator");
    check::equal(chan3.t0_index, 2.5, "channel_for at the CH rate: t0_index");
    check::equal(chan3.ch.size(), std::size_t{10}, "channel_for at the CH rate: the capture as it is");

    // the raw capture through receive_pass: a carrier, no frame -- refused
    const rc::PassOutcome o = rc::receive_pass(&cap, TINY, det_of(cfg::WAVEFORM_CE));
    check::is_true(!o.pass, "a carrier without a frame: refused");
}

std::vector<double> noisy(const hd::HeaderFields& h) {
    const std::vector<std::uint8_t> bits = hd::encode(h);
    std::vector<double> out;
    for (std::size_t k = 0; k < bits.size(); ++k) {
        const double s = 1.0 - 2.0 * bits[k];
        out.push_back(4.0 * s + static_cast<double>((static_cast<std::int64_t>(k) * 7) % 19 - 9) * 0.5);
    }
    return out;
}

void test_decode_header() {
    std::vector<double> garbage;
    for (std::int64_t k = 0; k < 2474; ++k) garbage.push_back(static_cast<double>((k * 5) % 11 - 5) * 1.0);
    const std::vector<double> n2 = noisy(test_header()), n1 = noisy(test_header_v1());
    auto vec = [&](const std::string& name) -> const std::vector<double>* {
        if (name == "noisy2") return &n2;
        if (name == "noisy1") return &n1;
        if (name == "garbage") return &garbage;
        return nullptr;
    };
    constexpr std::size_t n_cases = sizeof(P::HDR_CASE_FRAMES) / sizeof(P::HDR_CASE_FRAMES[0]);
    static_assert(P::HDR_CASES_N == 8 * n_cases);
    for (std::size_t c = 0; c < n_cases; ++c) {
        const fr::FrameSpec& spec = fr::get_any(P::HDR_CASE_FRAMES[c]);
        const std::vector<double>* sum = vec(P::HDR_CASE_VECS[3 * c]);
        // the sum as the pass's float32 LLRs
        std::vector<double> llr;
        for (const double x : *sum) llr.push_back(static_cast<float>(x));
        const rc::HeaderDecode h =
            rc::decode_header(llr, vec(P::HDR_CASE_VECS[3 * c + 1]), vec(P::HDR_CASE_VECS[3 * c + 2]), spec);
        const std::int64_t* want = P::HDR_CASES + 8 * c;
        const std::string what = "decode_header case " + std::to_string(c) + " (" + spec.name() + ")";
        check::equal(h.header.has_value(), want[0] != -1, what + ": a header");
        if (h.header && want[0] != -1) {
            const std::int64_t got[] = {h.header->picture_id, h.header->mode,    h.header->segment,
                                        h.header->codec_id,   h.header->version, h.header->reserved};
            for (std::size_t i = 0; i < 6; ++i) check::equal(got[i], want[i], what + ": field " + std::to_string(i));
        }
        check::equal(static_cast<std::int64_t>(h.block), want[6], what + ": the block's LLRs kept");
        check::equal(static_cast<std::int64_t>(h.plain), want[7], what + ": frame.plain");
        // (decode_header_l itself does not ask has_header: the reference's
        // _decode_header_l decodes an l-tiny sum as readily as any)
        if (spec.waveform() == fr::Waveform::L && spec.has_header()) {
            const auto hl = rc::decode_header_l(llr, vec(P::HDR_CASE_VECS[3 * c + 1]),
                                                vec(P::HDR_CASE_VECS[3 * c + 2]), spec);
            check::is_true(hl == h.header, what + ": decode_header_l agrees");
        }
    }
}

types::PassResult synthetic_pass(const fr::FrameSpec& spec) {
    types::PassResult p;
    p.frame = spec.name();
    p.waveform = static_cast<std::int64_t>(spec.waveform());
    for (std::int64_t i = 0; i < spec.n_data(); ++i)
        p.w.push_back(i % 13 == 0 ? 0.0f : static_cast<float>(static_cast<double>((i * 5) % 7) / 4.0 + 0.125));
    for (std::int64_t j = 0; j < 2474; ++j) p.hdr_llr.push_back(static_cast<float>(((j * 3) % 23 - 11) * 0.75));
    for (std::int64_t k = 0; k < spec.n_pos(); ++k) p.psi.push_back((k / 50) % 9 == 4 ? 0.0f : 1.0f);
    return p;
}

void test_confidence() {
    struct Frame {
        const char* name;
        const double* cases[5];
        std::size_t n[5];
        Span bare;
        double mean_w_db;
    };
#define CONF_FRAME(T, name)                                                                                     \
    Frame {                                                                                                     \
        name, {P::CONF_##T##_0, P::CONF_##T##_1, P::CONF_##T##_2, P::CONF_##T##_3, P::CONF_##T##_4},            \
            {P::CONF_##T##_0_N, P::CONF_##T##_1_N, P::CONF_##T##_2_N, P::CONF_##T##_3_N, P::CONF_##T##_4_N},    \
            SPAN(CONF_##T##_BARE), P::MEAN_W_DB_##T                                                              \
    }
    const Frame frames[] = {CONF_FRAME(SHORT, "short"), CONF_FRAME(LSHORT, "l-short"), CONF_FRAME(TINY, "tiny"),
                            CONF_FRAME(FULL, "full")};
    static_assert(P::CONF_N_N == 5);
    for (const Frame& f : frames) {
        const fr::FrameSpec& spec = fr::get_any(f.name);
        const types::PassResult p = synthetic_pass(spec);
        for (std::size_t c = 0; c < 5; ++c) {
            std::optional<double> el;
            if (!std::isnan(P::CONF_ELAPSED[c])) el = P::CONF_ELAPSED[c];
            check_conf(rc::confidence_db(p, nullptr, el, P::CONF_N[c]), Span{f.cases[c], f.n[c]}, 0.0,
                       std::string("confidence_db ") + f.name + " case " + std::to_string(c));
        }
        // a spec of the other waveform: the pass's own frame
        const fr::FrameSpec& other = fr::get_any(spec.waveform() == fr::Waveform::L ? "short" : "l-short");
        check_conf(rc::confidence_db(p, &other), Span{f.cases[0], f.n[0]}, 0.0,
                   std::string("confidence_db, another waveform's spec: ") + f.name);
        const std::vector<double> w(p.w.begin(), p.w.end());
        check_conf(rc::confidence_db(spec, w, nullptr, nullptr), f.bare, 0.0,
                   std::string("confidence_db, no psi, no hdr_llr: ") + f.name);
        check::equal(rc::mean_w_db(p), f.mean_w_db, std::string("mean_w_db ") + f.name);
    }
    const types::PassResult ps = synthetic_pass(fr::get("short"));
    throws<std::invalid_argument>([&] { (void)rc::confidence_db(ps, nullptr, std::nullopt, -1); }, P::ERR_CONF_M1,
                                  "confidence_db n = -1");
    throws<std::invalid_argument>([&] { (void)rc::confidence_db(ps, nullptr, std::nullopt, -2); }, P::ERR_CONF_M2,
                                  "confidence_db n = -2");
    throws<std::invalid_argument>([&] { (void)rc::confidence_db(ps, nullptr, std::nullopt, 0); }, P::ERR_CONF_0,
                                  "confidence_db n = 0");
    const std::vector<double> zero(5, 0.0), mix = {1e-31, 3.0, 0.5};
    check::equal(rc::mean_w_db(zero), P::MEAN_W_DB_ZERO, "mean_w_db of zeros: the floor");
    check::equal(rc::mean_w_db(mix), P::MEAN_W_DB_MIX, "mean_w_db of a mix");
}

void test_small() {
    check::equal(rc::centre_hz(1500.25), P::CENTRE_CE, "centre_hz CE");
    check::equal(rc::centre_hz(1500.25, cfg::WAVEFORM_L), P::CENTRE_L, "centre_hz L");
    types::PassResult p = synthetic_pass(fr::get("short"));
    p.f_hz = 1500.75;
    p.timing.tau0 = 12.25;
    p.report.offset_hz = 1500.5;
    p.report.drift_hz_per_min = 0.75;
    p.report.z_ref = 12.5;
    const types::Detection d = rc::detection_from_pass(p, 2.5);
    check::is_true(d.path.f_hz == std::vector<double>(P::DFP_F, P::DFP_F + P::DFP_F_N), "detection_from_pass: path f");
    check::is_true(d.path.t_s == std::vector<double>(P::DFP_T, P::DFP_T + P::DFP_T_N), "detection_from_pass: path t");
    check::is_true(d.path.weight == std::vector<double>(3, 1.0), "detection_from_pass: weights");
    check::equal(d.f_hz, 1500.75, "detection_from_pass: f_hz");
    check::equal(d.z_ref, 12.5, "detection_from_pass: z_ref");
    check::equal(d.lead_in_s, 2.5, "detection_from_pass: lead_in_s");
    check::equal(d.method, std::string("template"), "detection_from_pass: method");
    check::is_true(d.timing && d.timing->tau0 == 12.25, "detection_from_pass: timing");
    const rc::Prepared c = rc::capture_from_pass(p);
    check::is_true(c.fs == p.ch_fs && c.q == p.q && !c.keep && c.t0_index == p.ch_t0_index,
                   "capture_from_pass: the pass's CH capture");

    // frames / is_joint / single / ce_hints
    const std::vector<fr::FrameSpec> both = {fr::get_l("l-short"), fr::get("short")};
    const auto f = rc::frames(both);
    check::is_true(f.size() == 2 && f[0].name() == "short" && f[1].name() == "l-short", "frames: CE's first");
    check::is_true(rc::is_joint(both) && !rc::is_joint(fr::get("short")) && rc::is_joint(fr::get_l("l-tiny")),
                   "is_joint");
    const std::vector<fr::FrameSpec> one = {fr::get("short")};
    check::equal(rc::single(one).name(), std::string("short"), "_single");
    const std::vector<qrss::acquire::Hint> hints = {{1500.0, cfg::WAVEFORM_CE}, {1520.0, cfg::WAVEFORM_L},
                                                    {1540.0, cfg::WAVEFORM_CE}};
    const auto ch = rc::ce_hints(hints);
    check::is_true(ch.size() == 2 && ch[0].f_hz == 1500.0 && ch[1].f_hz == 1540.0, "_ce_hints: the CE ones");
    check::equal(rc::waveform_name(cfg::WAVEFORM_L), std::string("L"), "waveform_name L");
    check::equal(rc::waveform_name(9), std::string("9"), "waveform_name of an unknown waveform");
}

}  // namespace

int main() {
    try {
        test_small();
        test_decode_header();
        test_confidence();
        test_refusals();
        test_fe_capture();
        test_passes();
        test_seams();
        test_threads();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_receiver");
}
