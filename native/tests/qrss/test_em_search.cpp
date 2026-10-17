// native/core/qrss/em_search (the template search of sstvae/qrss/em.py)
// without Python, on inputs pinned by gen_em_search_pins.py
// (em_search_pins.hpp; the formulas are in that script): search_chan's cell
// exactly and Lambda to 1e-12 relative on formula captures and templates;
// chunk_template's starts exactly, energies and samples to 1e-12 (of the
// largest |E|) and the empty span's refusal; template_search on a formula
// store (two SHORT passes as segments 0 and 1 of one picture) against
// formula 250 Hz captures holding that picture's template (segment,
// candidate, lag, clock step and frequency bin chosen) or noise alone, and
// one raw FE-rate noise capture: the detection or none, Z and Lambda to
// 1e-9 relative, the chunks, cells and winning cell exactly. Then the
// false-alarm check with no reference: more noise captures, none
// detected. tests/test_qrss_native_em_search.py compares the same functions
// against Python directly (receive_template and retro_detect only there).
//
//   PYTHONPATH=. python native/tests/qrss/gen_em_search_pins.py   # em_search_pins.hpp

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "em_search_pins.hpp"
#include "infra_testgen.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/ce.hpp"
#include "qrss/em.hpp"
#include "qrss/em_search.hpp"
#include "qrss/frame.hpp"
#include "qrss/frontend.hpp"
#include "qrss/header.hpp"
#include "qrss/morse.hpp"
#include "qrss/rational.hpp"
#include "qrss/store.hpp"
#include "qrss/track.hpp"
#include "qrss/types.hpp"
#include "qrss_check.hpp"
#include "util/fileio.hpp"

using namespace sstvae;
namespace em = sstvae::qrss::em;
namespace qs = sstvae::qrss::store;
namespace types = sstvae::qrss::types;
namespace frame = sstvae::qrss::frame;
namespace P = em_search_pins;
namespace fs = std::filesystem;
using cplx = std::complex<double>;

namespace {

constexpr double PI = 3.141592653589793;
double worst_chan = 0.0, worst_ct = 0.0, worst_z = 0.0, worst_lam = 0.0;

std::string hex(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%a", x);
    return buf;
}

// |got - want| <= tol |want| (exact when want is 0); the worst ratio kept.
void near(double got, double want, double tol, const std::string& what, double& worst) {
    if (want == 0.0) {
        check::is_true(got == 0.0, what + ": got " + hex(got) + ", want 0");
        return;
    }
    const double rel = std::abs(got - want) / std::abs(want);
    worst = std::max(worst, rel);
    check::is_true(rel <= tol, what + ": got " + hex(got) + ", want " + hex(want));
}

std::vector<double> units(std::uint64_t seed, std::size_t n) {
    const auto z = qrss_testgen::splitmix(seed, n);
    std::vector<double> u(n);
    for (std::size_t i = 0; i < n; ++i) u[i] = qrss_testgen::unit(z[i]);
    return u;
}

// gen_em_search_pins.noise: (u_2i - 1/2) + j (u_2i+1 - 1/2)
std::vector<cplx> noise(std::uint64_t seed, std::size_t n) {
    const std::vector<double> u = units(seed, 2 * n);
    std::vector<cplx> z(n);
    for (std::size_t i = 0; i < n; ++i) z[i] = {u[2 * i] - 0.5, u[2 * i + 1] - 0.5};
    return z;
}

std::uint32_t h32(std::uint64_t in) {
    std::uint32_t x = static_cast<std::uint32_t>(in);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

fs::path fresh_dir(const std::string& name) {
    const fs::path p = fs::temp_directory_path() /
                       ("sstvae_test_em_search_" + name + "_" +
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

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// --- search_chan --------------------------------------------------------------------------------

void test_search_chan() {
    for (const P::Chan& c : P::CHANS) {
        check::current_step = c.name;
        const std::string name = std::string("search_chan ") + c.name;
        std::vector<cplx> z = noise(c.seed, static_cast<std::size_t>(c.n_z));
        const std::size_t K = c.starts.size(), chunk = static_cast<std::size_t>(c.chunk);
        const std::vector<double> v = units(c.seed + 1, 2 * K * chunk);
        em::ChunkTemplate t;
        t.chunk = c.chunk;
        t.starts = c.starts;
        t.E.resize(K * chunk);
        for (std::size_t i = 0; i < K * chunk; ++i) t.E[i] = {v[2 * i] - 0.5, v[2 * i + 1] - 0.5};
        for (std::size_t k = 0; k < K; ++k) t.en.push_back(1.0 + static_cast<double>(k) / 4.0);
        if (c.embed != 0.0) {
            for (std::size_t k = 0; k < K; ++k)
                for (std::size_t j = 0; j < chunk; ++j) {
                    const std::int64_t n = c.starts[k] + c.embed_lag + static_cast<std::int64_t>(j);
                    if (n < 0 || n >= c.n_z) continue;
                    const cplx e = t.E[k * chunk + j];
                    z[static_cast<std::size_t>(n)] = {z[static_cast<std::size_t>(n)].real() + c.embed * e.real(),
                                                      z[static_cast<std::size_t>(n)].imag() + c.embed * e.imag()};
                }
        }
        const em::ChanPeak pk = em::search_chan(z, c.n0, t, c.lag_max, c.m_max, c.drift);
        check::equal(pk.lag, c.lag, name + ": lag");
        check::equal(pk.m, c.m, name + ": m");
        near(pk.lam, c.lam, 1e-12, name + ": lam", worst_chan);
    }
}

// --- chunk_template -------------------------------------------------------------------------------

void test_chunk_template() {
    int c = 0;
    for (const P::ChunkT& ct : P::CHUNK_TEMPLATE) {
        check::current_step = ct.name;
        const std::string name = std::string("chunk_template ") + ct.name;
        const frame::FrameSpec& spec = frame::get(ct.spec);
        const std::size_t n = static_cast<std::size_t>(spec.n_sym());
        std::vector<double> mu = units(800 + static_cast<std::uint64_t>(c), n);
        std::vector<double> nu = units(900 + static_cast<std::uint64_t>(c), n);
        for (std::size_t i = 0; i < n; ++i) {
            mu[i] = 2.0 * mu[i] - 1.0;
            nu[i] = i % 7 == 0 ? 0.0 : nu[i] / 2.0;
        }
        ++c;
        std::optional<qrss::ce::Keying> keying;
        if (std::string(ct.keying) != "") {
            const std::vector<std::uint8_t> u = qrss::morse::keying_units(P::CALL);
            keying = qrss::ce::window_keying(std::span<const std::uint8_t>(u), spec);
        }
        const bool ook = std::string(ct.keying) == "ook";
        em::ChunkTemplate t;
        try {
            t = em::chunk_template(spec, mu, nu, keying ? &*keying : nullptr, ook, ct.t0, ct.ppm, ct.n_ch, ct.chunk);
        } catch (const std::out_of_range& e) {
            check::equal(std::string(e.what()), std::string(ct.error), name + ": refusal");
            continue;
        }
        check::equal(std::string(""), std::string(ct.error), name + ": no refusal");
        check::is_true(t.starts == ct.starts, name + ": starts (" + std::to_string(t.starts.size()) + " against " +
                                                  std::to_string(ct.starts.size()) + " pinned)");
        check::equal(t.en.size(), ct.en.size(), name + ": energies");
        for (std::size_t k = 0; k < std::min(t.en.size(), ct.en.size()); ++k)
            near(t.en[k], ct.en[k], 1e-12, name + ": en[" + std::to_string(k) + "]", worst_ct);
        double scale = 0.0;
        for (const cplx& e : t.E) scale = std::max(scale, std::abs(e));
        const std::size_t K = t.size(), chunk = static_cast<std::size_t>(ct.chunk);
        std::size_t s = 0;
        for (std::size_t k = 0; k < K && K == ct.starts.size(); ++k) {
            if (k % 8 != 0 && k != K - 1) continue;
            for (const std::size_t j : {std::size_t{0}, chunk / 2, chunk - 1}) {
                const cplx e = t.E[k * chunk + j];
                for (const double got : {e.real(), e.imag()}) {
                    const double want = ct.samples.at(s++);
                    const double err = std::abs(got - want) / scale;
                    worst_ct = std::max(worst_ct, err);
                    check::is_true(err <= 1e-12, name + ": E[" + std::to_string(k) + "][" + std::to_string(j) +
                                                     "] " + hex(got) + ", want " + hex(want));
                }
            }
        }
        check::equal(s, ct.samples.size(), name + ": every pinned sample checked");
    }
}

// --- template_search -------------------------------------------------------------------------------

qrss::header::HeaderFields hdr_of(std::int64_t g) {
    qrss::header::HeaderFields h;
    h.callsign = P::CALL;
    h.grid = std::string(P::GRID);
    h.picture_id = P::PICTURE_ID;
    h.mode = P::MODE;
    h.segment = g;
    h.codec_id = P::CODEC_ID;
    return h;
}

// gen_em_search_pins.make_pass
types::PassResult make_pass(std::int64_t g) {
    const frame::FrameSpec& spec = frame::get("short");
    const std::size_t n = static_cast<std::size_t>(spec.n_data());
    types::PassResult p;
    p.z.resize(n);
    p.w.assign(n, static_cast<float>(P::PASS_W));
    const auto ug = static_cast<std::uint64_t>(g);
    for (std::size_t i = 0; i < n; ++i) {
        const double a = (static_cast<double>(h32(i + 1000003u * (10u + ug) + 77u) >> 27) - 15.5) / 8.0;
        const double nz = (static_cast<double>(h32(i + 1000003u * (27u + ug)) >> 28) - 7.5) / 8.0;
        p.z[i] = static_cast<float>(a + nz);
    }
    const qrss::header::HeaderFields h = hdr_of(g);
    const std::vector<std::uint8_t> bits = qrss::header::encode(h);
    p.hdr_llr.resize(bits.size());
    for (std::size_t j = 0; j < bits.size(); ++j) p.hdr_llr[j] = static_cast<float>(4.0 * (1.0 - 2.0 * bits[j]));
    p.header = h;
    const std::size_t gi = static_cast<std::size_t>(g);
    p.q = P::Q0 - 1 - g;
    p.f_hz = P::PASS_F[gi];
    p.uid = types::pass_uid(p.q, p.f_hz, p.z);
    p.frame = "short";
    p.waveform = qrss::config::WAVEFORM_CE;
    p.timing = {0.0, P::PASS_PPM[gi], 0.0, 10.0, math::Matrix(2, 2, {1.0, 0.0, 0.0, 1.0})};
    p.report = {P::PASS_F[gi], P::PASS_DRIFT[gi], 0.0, 0.0, 0.0, -20.0, 10.0, 1.0, true, false};
    p.ch_fs = 250;
    p.ch_t0_index = 0.0;
    p.f_mix_hz = qrss::Rational(1500);
    p.psi.assign(4, 1.0f);
    p.estimator = "joint";
    p.em_round = 0;
    return p;
}

qs::Key the_key() { return {P::CALL, P::PICTURE_ID}; }

// template_search's own classes for segment g at slot q (gen_em_search_pins.classes_of)
qrss::track::Classes classes_of(qs::Store& st, const frame::FrameSpec& spec, std::int64_t g, std::int64_t q) {
    const qs::Accumulator acc = st.accumulator(the_key());
    const std::span<const std::uint16_t> idx = qs::canonical_index(g, spec.n_data());
    const std::size_t base = static_cast<std::size_t>(g) * qs::GROUP_LATENTS;
    std::vector<double> S(idx.size()), W(idx.size());
    for (std::size_t i = 0; i < idx.size(); ++i) {
        S[i] = static_cast<double>(acc.S[base + idx[i]]);
        W[i] = static_cast<double>(acc.W[base + idx[i]]);
    }
    const types::EmPrior prior = em::prior_from(S, W, q);
    const std::optional<qrss::header::HeaderFields> h = em::known_header(st, acc, g);
    std::optional<frame::HdrBits> bits;
    if (h && spec.has_header()) bits = frame::HdrBits::of(qrss::header::encode(*h));
    return qrss::track::make_classes(spec, bits ? &*bits : nullptr, &prior);
}

// gen_em_search_pins.ch_capture: the 250 Hz capture as a stored pass
types::PassResult ch_capture(qs::Store& st, const P::Search& ts) {
    const frame::FrameSpec& spec = frame::get(ts.spec);
    const double dur = spec.keyed_end_pos() * qrss::config::T_SYM;
    const auto n_ch = static_cast<std::int64_t>(P::T0_CH + dur * qrss::config::CH_FS + 2500);
    const std::vector<cplx> nz = noise(ts.seed, static_cast<std::size_t>(n_ch));
    std::vector<double> re(nz.size()), im(nz.size());
    for (std::size_t i = 0; i < nz.size(); ++i) {
        re[i] = nz[i].real();
        im[i] = nz[i].imag();
    }
    if (ts.amp != 0.0) {
        const qrss::track::Classes cl = classes_of(st, spec, ts.seg, P::CAP_Q);
        std::optional<qrss::ce::Keying> keying;
        if (spec.n_win() != 0) {
            const std::vector<std::uint8_t> u = qrss::morse::keying_units(P::CALL);
            keying = qrss::ce::window_keying(std::span<const std::uint8_t>(u), spec);
        }
        const std::size_t cand = static_cast<std::size_t>(ts.cand);
        const double ppm_step = em::TS_SLIP_SAMPLES / (dur * qrss::config::CH_FS) * 1e6;
        const double ppm = P::PASS_PPM[cand] + static_cast<double>(ts.k) * ppm_step;
        const em::ChunkTemplate t = em::chunk_template(spec, cl.mu, cl.nu, keying ? &*keying : nullptr, false,
                                                       P::T0_CH + static_cast<double>(ts.lag), ppm, n_ch, 1);
        std::vector<cplx> sig(nz.size());
        for (std::size_t i = 0; i < t.size(); ++i) sig[static_cast<std::size_t>(t.starts[i])] = t.E[i];
        const double f_ch = P::PASS_F[cand] - (1500.0 + static_cast<double>(ts.f_mix));
        const double drift = P::PASS_DRIFT[cand];
        for (std::int64_t n = 0; n < n_ch; ++n) {
            const auto i = static_cast<std::size_t>(n);
            const double tt = (static_cast<double>(n) - P::T0_CH) / 250.0;
            const double x =
                f_ch * tt + ((drift / 60.0) * tt) * tt / 2.0 + static_cast<double>(ts.m0 * n) / 2048.0;
            const double r = x - std::floor(x);
            const double c = std::cos(2.0 * PI * r), d = std::sin(2.0 * PI * r);
            re[i] = ts.amp * (sig[i].real() * c - sig[i].imag() * d) + nz[i].real();
            im[i] = ts.amp * (sig[i].real() * d + sig[i].imag() * c) + nz[i].imag();
        }
    }
    types::PassResult p = make_pass(0);
    p.uid = std::string("cap-") + ts.name;
    p.q = P::CAP_Q;
    p.ch.resize(nz.size());
    for (std::size_t i = 0; i < nz.size(); ++i) p.ch[i] = {static_cast<float>(re[i]), static_cast<float>(im[i])};
    p.ch_fs = qrss::config::CH_FS;
    p.ch_t0_index = P::T0_CH;
    p.f_mix_hz = qrss::Rational(ts.f_mix);
    return p;
}

// gen_em_search_pins.fe_capture: raw noise at the FE rate
qrss::frontend::Capture fe_capture(std::uint64_t seed, const frame::FrameSpec& spec) {
    const auto n = static_cast<std::int64_t>(P::T0_FE + spec.keyed_end_pos() * qrss::config::T_SYM *
                                                            qrss::config::FE_FS + 40000);
    const std::vector<cplx> nz = noise(seed, static_cast<std::size_t>(n));
    qrss::frontend::Capture c;
    c.fe.resize(nz.size());
    for (std::size_t i = 0; i < nz.size(); ++i)
        c.fe[i] = {static_cast<float>(nz[i].real()), static_cast<float>(nz[i].imag())};
    c.q = P::CAP_Q;
    c.t0_index = P::T0_FE;
    c.fs = qrss::config::FE_FS;
    return c;
}

em::SearchOptions options_of(const P::Search& ts) {
    em::SearchOptions o;
    o.freqs = ts.freqs;
    if (ts.has_segments) o.segments = ts.segments;
    o.tau_s = ts.tau_s;
    o.df_hz = ts.df_hz;
    o.ppm_steps = ts.ppm_steps;
    o.z_accept = ts.z_accept;
    return o;
}

void make_store(qs::Store& st) {
    for (std::int64_t g = 0; g < 2; ++g) {
        const types::PassResult p = make_pass(g);
        st.add_pass(p);
        st.attach(the_key(), p.uid, g, "test", P::MODE, P::CODEC_ID);
    }
}

void test_template_search(qs::Store& st) {
    for (const P::Search& ts : P::TEMPLATE_SEARCH) {
        check::current_step = ts.name;
        const std::string name = std::string("template_search ") + ts.name;
        const frame::FrameSpec& spec = frame::get(ts.spec);
        const auto t0 = std::chrono::steady_clock::now();
        em::SearchResult r;
        if (ts.fe) {
            const qrss::frontend::Capture cap = fe_capture(ts.seed, spec);
            r = em::template_search(st, the_key(), em::SearchCap(&cap), spec, options_of(ts));
        } else {
            const types::PassResult cap = ch_capture(st, ts);
            r = em::template_search(st, the_key(), em::SearchCap(&cap), spec, options_of(ts));
        }
        std::printf("  %-18s Z %7.3f%s (%.2f s)\n", ts.name, r.z, r.det ? " detected" : "", seconds_since(t0));
        check::equal(r.det.has_value(), ts.det, name + ": detection");
        near(r.z, ts.z, 1e-9, name + ": Z", worst_z);
        check::equal(r.info.has_value(), ts.has_info, name + ": info");
        if (!r.info || !ts.has_info) continue;
        const em::SearchInfo& in = *r.info;
        near(in.lam, ts.lam, 1e-9, name + ": lam", worst_lam);
        check::equal(in.chunks, ts.chunks, name + ": chunks");
        check::equal(hex(in.cells), hex(ts.cells), name + ": cells");
        check::equal(in.segment, ts.segment, name + ": segment");
        check::equal(hex(in.f_hz), hex(ts.f_hz), name + ": f_hz");
        check::equal(hex(in.ppm), hex(ts.ppm), name + ": ppm");
        check::equal(hex(in.tau0), hex(ts.tau0), name + ": tau0");
        check::equal(hex(in.drift), hex(ts.drift), name + ": drift");
        if (r.det) {
            check::equal(hex(r.det->f_hz), hex(in.f_hz), name + ": det.f_hz");
            check::equal(hex(r.det->z_ref), hex(r.z), name + ": det.z_ref");
            check::is_true(r.det->timing && r.det->timing->tau0 == in.tau0 && r.det->timing->ppm == in.ppm,
                           name + ": det.timing");
            check::equal(r.det->method, std::string("template"), name + ": det.method");
        }
    }
}

// No reference: more noise-only 250 Hz captures (TINY) are never detected,
// and their Z stays well under the gate (40 of them measured at most 1.1
// against the reference, 2026-10-14).
void test_false_alarms(qs::Store& st) {
    check::current_step = "false alarms";
    const frame::FrameSpec& spec = frame::get("tiny");
    double worst = 0.0;
    for (std::uint64_t seed = 5000; seed < 5008; ++seed) {
        P::Search ts = P::TEMPLATE_SEARCH[0];
        ts.name = "false-alarm";
        ts.seed = seed;
        ts.amp = 0.0;
        const types::PassResult cap = ch_capture(st, ts);
        const em::SearchResult r = em::template_search(st, the_key(), em::SearchCap(&cap), spec);
        check::is_true(!r.det, "noise seed " + std::to_string(seed) + " detected (Z " + hex(r.z) + ")");
        worst = std::max(worst, r.z);
    }
    check::is_true(worst < 4.0, "the largest Z on noise is " + hex(worst));
    std::printf("  largest Z over 8 more noise captures: %.3f\n", worst);
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(600.0, "qrss_em_search");
    try {
        test_search_chan();
        test_chunk_template();
        const fs::path dir = fresh_dir("store");
        Cleanup cleanup{dir};
        qs::Store::Seams seams;
        seams.clock = [] { return P::T0; };
        qs::Store st(util::path_text(dir / "qrss"), seams);
        make_store(st);
        test_template_search(st);
        test_false_alarms(st);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL in %s: %s\n", check::current_step.load(), e.what());
        return 1;
    }
    std::printf("worst relative error: search_chan lam %.3g, chunk_template %.3g, Z %.3g, lam %.3g\n", worst_chan,
                worst_ct, worst_z, worst_lam);
    return check::report("qrss_em_search");
}
