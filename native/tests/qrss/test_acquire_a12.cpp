// native/core/qrss/acquire_a12 against sstvae/qrss/acquire.py's A1, A2
// and lead_in_span, on one synthetic capture both sides can build: a CE
// TINY frame with zero data (frame.assemble + ce.baseband, both bit-exact
// ports) plus splitmix noise, in complex64, the frame 0.4 s early against
// nominal t0. A1 is pinned to the bit (statistic digest, candidates in
// hex), the tables by digest, A2's hit to tolerance (its chunk sums are a
// BLAS zgemm in the reference), the lead-in span exactly; then the
// reference's errors and the thread-safety of the cached tables.
// tests/test_qrss_native_acquire_a12.py compares against Python directly.
// The pins (scratch script, values printed with float.hex / repr):
//
//   def capture(lead_in_s, seed):
//       n = 40 * FE_FS
//       sym = frame.assemble(frame.TINY, None, np.zeros(frame.TINY.n_data))
//       z = ce.baseband(sym, frame.TINY, FE_FS, -12.0, n, lead_in_s=lead_in_s)
//       u = splitmix(seed, 2 * n)                                  # infra_testgen.hpp
//       noise = (unit(u[:n]) - 0.5) * 9.8 + 1j * ((unit(u[n:]) - 0.5) * 9.8)
//       return (z + noise).astype(np.complex64)
//   x, t0 = capture(0.0, 1201), 12.0 * FE_FS + 0.4 * FE_FS
//   cs = A.carrier_line_stats(x, t0); lines = A.carrier_lines(x, t0, stats=cs)
//   A.hint_line(cs, f), A.pilot_lines(cs, lines), A.preamble_search(x, t0, lines[0], kind=k),
//   A.lead_in_span(...), sha256(a.tobytes()) of every table.

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "infra_testgen.hpp"
#include "math/nplog.hpp"
#include "qrss/acquire_a12.hpp"
#include "qrss/ce.hpp"
#include "qrss/frame.hpp"
#include "qrss/pyjson.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace acq = sstvae::qrss::acquire;
namespace types = sstvae::qrss::types;
namespace rxa = sstvae::qrss::rxconfig::acquire;
namespace config = sstvae::qrss::config;
using cf = std::complex<float>;
using cd = std::complex<double>;

constexpr double FE = 4000.0;

static std::vector<cf> capture(double lead_in_s, std::uint64_t seed) {
    const auto n = static_cast<std::int64_t>(40 * FE);
    const auto& spec = qrss::frame::get("tiny");
    const std::vector<double> zeros(static_cast<std::size_t>(spec.n_data()), 0.0);
    const std::vector<double> sym = qrss::frame::assemble(spec, nullptr, zeros);
    qrss::ce::BasebandOptions opt;
    opt.lead_in_s = lead_in_s;
    const std::vector<cd> z = qrss::ce::baseband(sym, spec, FE, -12.0, n, opt);
    const auto un = static_cast<std::size_t>(n);
    const auto u = qrss_testgen::splitmix(seed, 2 * un);
    std::vector<cf> x(un);
    for (std::size_t i = 0; i < un; ++i) {
        const double nr = (qrss_testgen::unit(u[i]) - 0.5) * 9.8;
        const double ni = (qrss_testgen::unit(u[un + i]) - 0.5) * 9.8;
        const cd v = z[i] + cd(nr, ni);
        x[i] = cf(static_cast<float>(v.real()), static_cast<float>(v.imag()));
    }
    return x;
}

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("wrong exception: ") + e.what());
        return;
    }
    check::fail(what, "did not throw");
}

static void line_is(const types::CarrierCandidate& c, double f, double drift, double metric, const std::string& what) {
    check::equal(c.f_hz, f, what + " f_hz");
    check::equal(c.drift_hz_per_min, drift, what + " drift");
    check::equal(c.metric, metric, what + " metric");
}

static void near(double got, double want, double tol, const std::string& what) {
    check::close(std::vector<double>{got}, std::vector<double>{want}, tol, what);
}

static void test_a1(const std::vector<cf>& x, double t0) {
    const acq::CarrierStats cs = acq::carrier_line_stats(x, t0);
    check::equal(cs.stat.rows, std::size_t(13), "stat rows");
    check::equal(cs.stat.cols, std::size_t(9601), "stat cols");
    qrss_check::digest_is(cs.stat.data, "4819a4aac1efb309e789c5cfdb9c6cd1f6aacc6225050cfa7fc318787d56dbcc",
                          "A1 statistic, bit for bit");
    qrss_check::digest_is(cs.f_hz, "f69f010fdf4babbd1c0d6d73a98478a6f1f09a4674e2cd1f55b7b97b696787fc", "A1 bins");
    check::equal(cs.M, std::int64_t(11), "M");

    const auto lines = acq::carrier_lines(cs);
    check::equal(lines.size(), std::size_t(7), "carrier_lines count");
    line_is(lines.at(0), 0x1.76ffbdffaf800p+10, 0.0, 0x1.f816200000000p+10, "line 0");
    line_is(lines.at(1), 0x1.79d0084e3f700p+10, -1.0, 0x1.9499180000000p+5, "line 1");
    line_is(lines.at(2), 0x1.742f99b58a800p+10, 0.5, 0x1.782eb40000000p+5, "line 2");
    line_is(lines.at(3), 0x1.f5027b180c000p+10, 1.0, 0x1.e84c440000000p+4, "line 3");
    // the same lines from the capture directly, and from complex128
    std::vector<cd> x64(x.begin(), x.end());
    check::is_true(acq::carrier_line_stats(x64, t0).stat == cs.stat, "complex128 capture, same statistic");

    bool refined = true;
    line_is(acq::hint_line(cs, 1500.3, FE, rxa::A1_PFA, rxa::HINT_SPAN_HZ, &refined), 0x1.76ffbdffaf800p+10, 0.0,
            0x1.f816200000000p+10, "hint at the carrier");
    check::is_true(refined, "hint at the carrier is a cell");
    line_is(acq::hint_line(cs, 2000.0, FE, rxa::A1_PFA, rxa::HINT_SPAN_HZ, &refined), 2000.0, 0.0,
            0x1.1798900000000p+4, "hint under threshold");
    check::is_true(!refined, "hint under threshold is the hint");
    line_is(acq::hint_line(cs, 100.0), 100.0, 0.0, 0.0, "hint out of band");

    const auto pil = acq::pilot_lines(cs, lines);
    check::equal(pil.size(), std::size_t(5), "pilot_lines count");
    line_is(pil.at(0), 0x1.7abf037eab000p+10, -2.5, 0x1.50bb380000000p+5, "pilot 0");
    line_is(pil.at(1), 0x1.f6fc62c974000p+10, -1.5, 0x1.e74de60000000p+4, "pilot 1");
    line_is(pil.at(2), 0x1.84c7b4bcf0000p+8, 2.0, 0x1.712e520000000p+4, "pilot 2");

    // _cell_line indexes the drift only when it survives
    const types::CarrierCandidate kept = acq::cell_line(cs, 0, 13, -1e9);
    check::equal(kept.drift_hz_per_min, 0.0, "cell_line, an out-of-range drift replaced by the straight one");
    throws<std::out_of_range>([&] { (void)acq::cell_line(cs, 0, 13, 1e9); },
                              "index 13 is out of bounds for axis 0 with size 13", "cell_line, surviving drift");
    throws<std::out_of_range>([&] { (void)acq::cell_line(cs, 9601, 0, 10.0); },
                              "index 9601 is out of bounds for axis 1 with size 9601", "cell_line, bin");
    const std::vector<double> none;
    throws<std::invalid_argument>([&] { (void)acq::carrier_line_stats(x, t0, FE, rxa::A1_SPAN_S, none); },
                                  "zero-size array to reduction operation maximum which has no identity",
                                  "carrier_line_stats, no drifts");
}

static void test_tables() {
    qrss_check::digest_is(acq::notch_taps(), "b58532339cb2b35e2a7d170a2c4344f19b18d1bd83af8cfa5d00af3df133075e", "notch");
    qrss_check::digest_is(acq::mf_taps(), "20beeeefc133593140f36afd10cd4fc599f168614aa1dde2c636170a2b9fb56d", "mf");
    qrss_check::digest_is(acq::l_mf_taps(), "9a213f55043818b0f53493d04b34304015e2778b15cf26eb2c91c783ad28c685", "l mf");
    qrss_check::digest_is(acq::mf_taps(500), "a8430ede131e1d05b41d6ca38dee00474b2c238c5f59f3ebfdd785644c7bee75", "mf 500");
    std::vector<double> ce_seq;
    for (const cd& v : acq::ce_pre().seq) ce_seq.push_back(v.real());
    qrss_check::digest_is(ce_seq, "71ff3707fadd10dd65834c63723f7c6f7790fc00f617ce82cf3b073f2a6e70e0", "CE preamble");
    qrss_check::digest_is(acq::l_pre().seq, "0207b0d82b65e5208eb7935620f40a70c50d4f693fe85e4a7e432a6fb44d7f8a", "L preamble");
    check::is_true(acq::ce_pre().real && !acq::l_pre().real, "CE's preamble is real, L's complex");
    check::equal(acq::ce_pre().n_chunks(), std::int64_t(10), "CE n_chunks");
    check::equal(acq::l_pre().n_chunks(), std::int64_t(10), "L n_chunks");
    check::equal(acq::l_pre().chunk, std::int64_t(33), "L chunk");
    check::equal(acq::ce_pre().name(), std::string("CE"), "CE name");
    check::equal(acq::l_pre().name(), std::string("L"), "L name");
    check::is_true(&acq::preamble_kind(0) == &acq::ce_pre() && &acq::preamble_kind(1) == &acq::l_pre(),
                   "PREAMBLE_KINDS");
    throws<qrss::pyjson::KeyError>([] { (void)acq::preamble_kind(7); }, "KeyError: 7", "PREAMBLE_KINDS[7]");
    const std::vector<double> df(rxa::A2_DF_HZ.begin(), rxa::A2_DF_HZ.end());
    const acq::Templates ce = acq::templates(df, acq::ce_pre());
    std::vector<double> ce_b;
    for (const cd& v : ce.b.b) ce_b.push_back(v.real());
    qrss_check::digest_is(ce_b, "eaef2407ee25e6cc77003ca16a0b5574685cc839798552aadff4229bcc53575c", "CE chunks");
    qrss_check::digest_is(ce.W, "4657a342bbbf51e417f174045dca622196013717a56d624792ced339d14f89fc", "CE templates W");
    qrss_check::digest_is(ce.norm, "db36d825ea67b11191d0a4c6d1f89a9a91d2fd5833a79709896365071ddc4bd0", "CE norm");
    const acq::Templates lt = acq::templates(df, acq::l_pre());
    qrss_check::digest_is(lt.b.b, "b3f3ba7efcb15e00ef729b4f3786f560d44a010a9db8b82c293dd6e8efcde431", "L chunks");
    qrss_check::digest_is(lt.W, "dbc3b5f3bc6a734fcf5fa4906392b4c8389a6aeb12f27e1a9d5d3e5c0e669b89", "L templates W");
    qrss_check::digest_is(lt.norm, "ce2e781bd1bbfd10d649534688ecdc9685540f9a31dd8ce48f644dcc94020b54", "L norm");
    // numpy's float32 log (its SIMD kernel), [v.hex() for v in np.log(np.float32([...]))]
    const float in[] = {0.3f, 7.0f, 1e-9f, 2.0f, 0.7071067f, 123456.7f, 3.4e38f, 1.5e-45f};
    const double want[] = {-0x1.3437900000000p+0, 0x1.f2272c0000000p+0, -0x1.4b92800000000p+4,
                           0x1.62e4300000000p-1,  -0x1.62e4360000000p-2, 0x1.77281c0000000p+3,
                           0x1.62e3560000000p+6,  -0x1.9d1da00000000p+6};
    for (int i = 0; i < 8; ++i)
        check::equal(static_cast<double>(math::np::log_f32(in[i])), want[i], "np.log float32 " + std::to_string(i));
    check::is_true(std::isinf(math::np::log_f32(0.0f)) && math::np::log_f32(0.0f) < 0, "log 0");
    check::is_true(std::isnan(math::np::log_f32(-1.0f)), "log -1");
    // _phase_line on five chunk sums: [repr(v) for v in A._phase_line(S, 0.6)]
    const std::vector<cd> S = {{1, 1}, {0.5, -2}, {-1, 0.25}, {2, 2}, {0.1, 3}};
    bool fitted = false;
    const auto [a, b, sd] = acq::phase_line(S, 0.6, &fitted);
    check::is_true(fitted, "phase_line fitted");
    near(a, -1.012134320680961, 1e-14, "phase_line a");
    near(b, 0.4649309266347542, 1e-14, "phase_line b");
    near(sd, 0.3918062550436878, 1e-14, "phase_line sd_b");
    const std::vector<cd> zero(4, cd(0, 0));
    const auto [a0, b0, sd0] = acq::phase_line(zero, 0.6, &fitted);
    check::is_true(!fitted && a0 == 0.0 && b0 == 0.0 && std::isinf(sd0), "phase_line without weight");
}

static void hit_near(const acq::PreambleHit& h, double f, double tau0, double phase, double lam, double cell_lam,
                     double p, bool passed, std::int64_t groups, double sd, std::int64_t waveform,
                     const std::string& what) {
    near(h.f_hz, f, 1e-7, what + " f_hz");
    check::equal(h.drift_hz_per_min, 0.0, what + " drift");
    near(h.tau0_s, tau0, 1e-6, what + " tau0_s");
    near(h.phase, phase, 1e-9, what + " phase");
    near(h.lam, lam, 1e-9 * lam, what + " lam");
    near(h.cell_lam, cell_lam, 1e-9 * cell_lam, what + " cell_lam");
    near(h.p_value, p, 1e-6 * p, what + " p_value");
    check::equal(h.passed, passed, what + " passed");
    check::equal(h.n_groups, groups, what + " n_groups");
    check::is_true(h.f_mix.has_value() && h.f_mix->num() == 0 && h.f_mix->den() == 1, what + " f_mix");
    near(h.tau0_sd_s, sd, 1e-6 * sd, what + " tau0_sd_s");
    check::equal(h.waveform, waveform, what + " waveform");
    check::equal(h.chunk_sums ? h.chunk_sums->size() : 0, std::size_t(10), what + " chunk_sums");
}

static void test_a2(const std::vector<cf>& x, double t0) {
    const auto lines = acq::carrier_lines(acq::carrier_line_stats(x, t0));
    const types::CarrierCandidate& top = lines.at(0);
    // the channel and its grid
    const acq::A2Channel chan = acq::preamble_channel(x, t0, top.f_hz, top.drift_hz_per_min);
    check::equal(chan.t0_ch, 0x1.09a0000000000p+11, "t0_ch");
    near(chan.sigma2, 0.4532741714168457, 1e-12, "sigma2");
    const acq::Sums s = acq::preamble_sums(chan);
    std::size_t best = 0;
    for (std::size_t i = 1; i < s.lam.size(); ++i)
        if (s.lam[i] > s.lam[best]) best = i;
    check::equal(best / s.n_tau, std::size_t(10), "Lambda argmax df");
    check::equal(best % s.n_tau, std::size_t(525), "Lambda argmax tau");
    near(s.lam[best], 543.1000079307213, 1e-9 * 543.1, "Lambda max");
    check::equal(s.taus[525], -100.0, "tau offset");

    const acq::PreambleHit ce = acq::preamble_search(x, t0, top);
    hit_near(ce, 1500.0007352213397, -0.3999859723016677, -0.07860512177431533, 545.6062855769717,
             533.2657444271699, 5.43809922966975e-229, true, 2, 0.0004561193995226714, 0, "CE hit");
    const acq::PreambleHit l = acq::preamble_search(x, t0, top, FE, rxa::A2_PFA, acq::l_pre());
    hit_near(l, 1499.7392196498981, 0.25923938964178606, -0.9245395089805991, 18.05802451822522, 10.128419531425257,
             0.00015971410731339684, false, 1, 0.008901252372626048, 1, "L hit on a CE carrier");
    // one channelisation for both kinds
    const acq::A2Input inp = acq::a2_input(x, t0, top.f_hz, top.drift_hz_per_min);
    const acq::PreambleHit ce2 = acq::preamble_search(inp, top);
    check::equal(ce2.tau0_s, ce.tau0_s, "from _a2_input, the same hit");
    check::equal(ce2.p_value, ce.p_value, "from _a2_input, the same p");

    const types::Detection det = ce.to_detection(100.0);
    near(det.timing->tau0, -99.99649307541692, 1e-6 * 250, "to_detection tau0");
    near(det.timing->cov(0, 0), 0.013002806663807645, 1e-6 * 0.013, "to_detection cov");
    check::equal(det.method, std::string("preamble"), "to_detection method");
    check::equal(acq::lead_in_span(x, t0, ce.f_hz, ce.drift_hz_per_min, ce.tau0_s), 0.0, "no lead-in");
    check::equal(acq::lead_in_span(x, t0, l.f_hz, l.drift_hz_per_min, l.tau0_s), 1.0, "a lead-in of noise, 1 s");

    // a 6 s lead-in
    const std::vector<cf> xl = capture(6.0, 1202);
    const acq::PreambleHit hl = acq::preamble_search(xl, t0, types::CarrierCandidate{1500.0, 0.0, 0.0, std::nullopt});
    near(hl.f_hz, 1499.9998265073996, 1e-7, "lead-in hit f_hz");
    near(hl.tau0_s, -0.40023539625034155, 1e-6, "lead-in hit tau0_s");
    check::equal(acq::lead_in_span(xl, t0, hl.f_hz, hl.drift_hz_per_min, hl.tau0_s), 6.0, "lead-in 6 s");
    check::equal(acq::lead_in_span(xl, t0, hl.f_hz, hl.drift_hz_per_min, hl.tau0_s, FE, 40.0), 0.0,
                 "lead-in, z_min 40");
    check::equal(acq::lead_in_span(xl, t0, hl.f_hz + 30.0, hl.drift_hz_per_min, hl.tau0_s), 0.0,
                 "lead-in 30 Hz off");
    // the noise_psd seam: called with the channel, CH_FS, the FE rate and the audio centre
    std::int64_t seen_fs = 0, seen_in = 0;
    double seen_audio = 0.0;
    const acq::NoisePsdFn spy = [&](std::span<const cf> ch, std::int64_t fs, std::int64_t fs_in, double audio) {
        seen_fs = fs;
        seen_in = fs_in;
        seen_audio = audio;
        return std::vector<float>(5, ch.empty() ? 1.0f : 0.5f);
    };
    (void)acq::lead_in_span(xl, t0, hl.f_hz, 0.0, hl.tau0_s, FE, rxa::LEAD_Z, config::T_SYM, spy);
    check::equal(seen_fs, std::int64_t(250), "noise_psd fs");
    check::equal(seen_in, std::int64_t(4000), "noise_psd fs_in");
    check::equal(seen_audio, 1500.0, "noise_psd audio_hz");
}

static void test_threads(const std::vector<cf>& x, double t0) {
    // the tables are built once whoever asks first; the search is pure
    const types::CarrierCandidate c{1500.0, 0.0, 0.0, std::nullopt};
    const acq::PreambleHit want = acq::preamble_search(x, t0, c);
    std::vector<int> bad(4, 0);
    std::vector<std::thread> pool;
    for (int t = 0; t < 4; ++t)
        pool.emplace_back([&, t] {
            const auto& taps = t % 2 ? acq::mf_taps() : acq::l_mf_taps();
            if (taps.empty()) ++bad[static_cast<std::size_t>(t)];
            const acq::PreambleHit h = acq::preamble_search(x, t0, c, FE, rxa::A2_PFA,
                                                            t % 2 ? acq::ce_pre() : acq::ce_pre());
            if (h.tau0_s != want.tau0_s || h.lam != want.lam || h.p_value != want.p_value)
                ++bad[static_cast<std::size_t>(t)];
        });
    for (auto& th : pool) th.join();
    int nbad = 0;
    for (const int b : bad) nbad += b;
    check::equal(nbad, 0, "preamble_search from 4 threads");
}

int main() {
    try {
        const std::vector<cf> x = capture(0.0, 1201);
        qrss_check::digest_is(x, "23438ab54e64dcd00accc29f12831eadb789a38e4235fdcfc1ba56d1a9c05230", "the capture");
        const double t0 = 12.0 * FE + 0.4 * FE;
        test_tables();
        test_a1(x, t0);
        test_a2(x, t0);
        test_threads(x, t0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_acquire_a12");
}
