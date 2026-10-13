// native/core/qrss/frontend against values pinned from the Python
// reference (sstvae/qrss/frontend.py, numpy 2.5.3, scipy 1.18.1). The
// inputs are splitmix64 counters (infra_testgen.hpp), so each pin can be
// reproduced from Python:
//
//   x   = unit(splitmix(77, 5000)) * 2 - 1                      # audio
//   fe0 = (unit(splitmix(101, 12000)) * 2 - 1
//          + 1j * (unit(splitmix(102, 12000)) * 2 - 1)).astype(np.complex64)
//   fe0[[3000, 3001, 7000, 11999]] *= 40                        # impulses
//   b_fe, b_keep = frontend.blank(fe0)
//   n_fe, n_lev = frontend.normalise(b_fe, b_keep)
//   p = (unit(splitmix(103, 9000)) * 5).astype(np.float32); p[::17] = np.round(p[::17])
//   chn = (unit(splitmix(104, 7500)) * 2 - 1
//          + 1j * (unit(splitmix(105, 7500)) * 2 - 1)).astype(np.complex64)
//
// Digests are hashlib.sha256(a.tobytes()); the masks' digest is of every
// _fe_flat(4000, 250, 2500, a).view(np.uint8) for a in 300..440, 2560..2700
// concatenated. tests/test_qrss_native_frontend.py compares against the
// reference directly, over far more inputs.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "infra_testgen.hpp"
#include "qrss/frontend.hpp"
#include "qrss/rational.hpp"
#include "qrss/rx_config.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace fe = sstvae::qrss::frontend;
using cf = std::complex<float>;
using cd = std::complex<double>;

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("threw the wrong type: ") + e.what());
        return;
    }
    check::fail(what, "did not throw");
}

template <typename T>
static void near(std::complex<T> got, cd want, double tol, const std::string& what) {
    ++check::checks;
    const double d = std::abs(cd(got.real(), got.imag()) - want);
    if (!(d <= tol)) {
        char buf[200];
        std::snprintf(buf, sizeof buf, "got {%.17g, %.17g}, want {%.17g, %.17g}", static_cast<double>(got.real()),
                      static_cast<double>(got.imag()), want.real(), want.imag());
        check::fail(what, buf);
    }
}

static void near(double got, double want, double rtol, const std::string& what) {
    ++check::checks;
    if (!(std::abs(got - want) <= rtol * std::abs(want))) {
        char buf[200];
        std::snprintf(buf, sizeof buf, "got %.17g, want %.17g (rtol %g)", got, want, rtol);
        check::fail(what, buf);
    }
}

static std::vector<double> audio() {
    const auto z = qrss_testgen::splitmix(77, 5000);
    std::vector<double> x(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) x[i] = qrss_testgen::unit(z[i]) * 2 - 1;
    return x;
}

static std::vector<cf> complex_noise(std::uint64_t seed_re, std::uint64_t seed_im, std::size_t n) {
    const auto a = qrss_testgen::splitmix(seed_re, n), b = qrss_testgen::splitmix(seed_im, n);
    std::vector<cf> out(n);
    for (std::size_t i = 0; i < n; ++i)   // complex128 first, then the complex64 cast, as numpy
        out[i] = cf(static_cast<float>(qrss_testgen::unit(a[i]) * 2 - 1),
                    static_cast<float>(qrss_testgen::unit(b[i]) * 2 - 1));
    return out;
}

static std::vector<cf> impulsive() {
    auto x = complex_noise(101, 102, 12000);
    for (const std::size_t i : {3000, 3001, 7000, 11999}) x[i] = cf(x[i].real() * 40.0f, x[i].imag() * 40.0f);
    return x;
}

static void test_taps() {
    const auto& t = fe::fe_taps();
    const auto& want = qrss::rxconfig::frontend::FE_TAPS_;
    check::equal(t.size(), want.size(), "255 taps");
    bool same = t.size() == want.size();
    for (std::size_t i = 0; same && i < t.size(); ++i) same = t[i] == want[i];
    check::is_true(same, "the FE taps are firwin(255, 1250, kaiser 8) to the bit");
    check::equal(fe::FE_DECIM, 2, "FE_DECIM");
    check::equal(fe::fe_hz(1830.25), 330.25, "fe_hz");
}

static void test_audio_to_fe() {
    const auto x = audio();
    const auto y = fe::audio_to_fe(x);
    check::equal(y.size(), std::size_t{2500}, "audio_to_fe length");
    near(y[0], {-0.08469869196414948, -0.05580603703856468}, 1e-6, "audio_to_fe[0]");
    near(y[1], {0.05847383663058281, 0.10030677914619446}, 1e-6, "audio_to_fe[1]");
    near(y[1234], {0.873684823513031, -0.3723170757293701}, 1e-6, "audio_to_fe[1234]");
    near(y[2499], {-0.03896944969892502, -0.006080583203583956}, 1e-6, "audio_to_fe[2499]");
    check::equal(fe::audio_to_fe(std::vector<double>{}).size(), std::size_t{0}, "audio_to_fe of nothing");
    check::equal(fe::audio_to_fe(std::vector<double>(7, 0.25)).size(), std::size_t{4}, "odd length: ceil(n/2)");

    std::vector<cd> fe128(y.begin(), y.end());
    const auto back = fe::fe_to_audio(fe128);
    check::equal(back.size(), std::size_t{5000}, "fe_to_audio length");
    const std::pair<std::size_t, double> pins[] = {{0, -0.08395082565788813},
                                                   {1, -0.10055475566958175},
                                                   {777, -0.3451431900943057},
                                                   {2500, 0.193955834504549},
                                                   {4999, 0.0060458036157388356}};
    for (const auto& [i, v] : pins) near(back[i], v, 1e-12, "fe_to_audio[" + std::to_string(i) + "]");
}

static void test_stream() {
    // StreamFE over uneven chunks equals the batch conversion away from the
    // stream's ends; every output sample is produced once, in order.
    const auto z = qrss_testgen::splitmix(120, 40000);
    std::vector<double> x(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) x[i] = qrss_testgen::unit(z[i]) * 2 - 1;
    const auto batch = fe::audio_to_fe(x);
    fe::StreamFE s(1000);
    check::equal(s.block(), std::int64_t{992}, "block rounded down to ALIGN");
    check::is_true(!s.next_index().has_value(), "no next index before the first push");
    std::size_t a = 0, k = 0;
    std::int64_t expect = fe::HALO / 2;
    double worst = 0.0;
    bool contiguous = true;
    const std::size_t sizes[] = {1, 300, 4096, 17, 999, 5000, 3};
    while (a < x.size()) {
        const std::size_t n = std::min(sizes[k++ % 7], x.size() - a);
        for (const auto& c : s.push(static_cast<std::int64_t>(a), std::span<const double>(x.data() + a, n))) {
            contiguous = contiguous && c.index == expect;
            for (std::size_t i = 0; i < c.fe.size(); ++i) {
                const auto j = static_cast<std::size_t>(c.index) + i;
                worst = std::max(worst, static_cast<double>(std::abs(c.fe[i] - batch[j])));
            }
            expect = c.index + static_cast<std::int64_t>(c.fe.size());
        }
        a += n;
    }
    check::is_true(contiguous, "StreamFE output is contiguous from HALO / 2");
    check::is_true(expect > 19000, "StreamFE produced all but the pending tail");
    check::is_true(worst <= 1e-6, "StreamFE equals the batch conversion (" + std::to_string(worst) + ")");
    check::equal(*s.next_index(), std::int64_t{40000}, "next_index follows the stream");
    // a gap restarts the stream on the ALIGN grid
    fe::StreamFE r;
    r.push(0, std::span<const double>(x.data(), 100));
    r.push(1003, std::span<const double>(x.data(), 100));
    check::equal(*r.a0(), std::int64_t{1008}, "a restart drops samples up to the next multiple of ALIGN");
    check::equal(r.pending().size(), std::size_t{95}, "and keeps the rest");
}

static void test_blank_normalise() {
    const auto x = impulsive();
    const auto b = fe::blank(x);
    qrss_check::digest_is(b.fe, "802d50833fdb3f223790b7da0522e04f4143310239ac6d447d65d4b771d3bb74", "blank fe");
    qrss_check::digest_is(b.keep, "6c1c47d02a11fdf256d55e114ac94ff124ef29937ae555796def90e7e0c1c42c", "blank keep");
    std::size_t dead = 0, taper = 0;
    for (const float k : b.keep) {
        dead += k == 0.0f;
        taper += k > 0.0f && k < 1.0f;
    }
    check::equal(dead, std::size_t{44}, "blanked samples");
    check::equal(taper, std::size_t{20}, "taper samples");
    const auto n = fe::normalise(b.fe, std::span<const float>(b.keep));
    qrss_check::digest_is(n.fe, "0f81b51d064f4f3b9e61e8bf7ba87948e88834e6ab37c3fbee504ad7c1e7c1f2", "normalise fe");
    qrss_check::digest_is(n.level, "57086244b141d7ccf1f02cf1e5e1a1866e0aedccde1b36daa7d8ecd7927d3e1b",
                          "normalise level");

    throws<std::invalid_argument>([&] { fe::blank(std::span<const cf>(x.data(), 50)); },
                                  "array of sample points is empty", "blank of 50 samples (np.interp's refusal)");
    check::equal(fe::blank(std::span<const cf>()).fe.size(), std::size_t{0}, "blank of nothing");
    const std::vector<float> short_keep(99, 1.0f);
    throws<std::out_of_range>(
        [&] { fe::normalise(std::span<const cf>(x.data(), 100), std::span<const float>(short_keep)); },
        "boolean index did not match indexed array along axis 0; size of axis is 100 but size of corresponding "
        "boolean axis is 99",
        "normalise with a keep of the wrong length");
    throws<std::invalid_argument>([] { fe::interp_to_samples(5, {}, {}); }, "array of sample points is empty",
                                  "interp with no centres");
    throws<std::invalid_argument>([] { fe::interp_to_samples(-1, {}, {}); }, "negative dimensions are not allowed",
                                  "interp to a negative length");
}

static void test_running_median() {
    const auto z = qrss_testgen::splitmix(103, 9000);
    std::vector<float> p(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) p[i] = static_cast<float>(qrss_testgen::unit(z[i]) * 5);
    for (std::size_t i = 0; i < p.size(); i += 17) p[i] = std::nearbyint(p[i]);
    auto r = fe::running_median(p, 4000, 1.0, 0.05);
    check::equal(r.centres.size(), std::size_t{45}, "running_median centres");
    qrss_check::digest_is(r.centres, "de7275db23547d4b503d50431f5a7880c6e3e4c044a58d484b79f7d003cdb7ee", "centres");
    qrss_check::digest_is(r.medians, "41511e13d0aef5f783adaa076207617ab4c7aa6541f2af6643a93a6d873569c9", "medians");
    r = fe::running_median(p, 1000, 0.1, 0.0001);
    check::equal(r.centres.size(), std::size_t{9000}, "running_median centres, fine hop");
    qrss_check::digest_is(r.centres, "205e3c9fc70045854739417b3df65f19bd02744fd00d6e0c5e4d96f149083c73",
                          "centres, fine hop");
    qrss_check::digest_is(r.medians, "8e97cc91fcabb748c6d30bff84dda14b7e1c441037e8d00898d3b99c032455a7",
                          "medians, fine hop");
}

static void test_mix() {
    const std::pair<double, std::pair<std::int64_t, std::int64_t>> q[] = {{-1169.75, {-4679, 4}},
                                                                         {0.5 / 64, {0, 1}},   // half to even
                                                                         {1.5 / 64, {1, 32}},
                                                                         {-2.5 / 64, {-1, 32}},
                                                                         {330.3 - 1500, {-74861, 64}}};
    for (const auto& [f, nd] : q) {
        const qrss::Rational r = fe::quantise_mix(f);
        check::equal(r.num(), nd.first, "quantise_mix numerator " + std::to_string(f));
        check::equal(r.den(), nd.second, "quantise_mix denominator " + std::to_string(f));
    }
    throws<std::invalid_argument>([] { fe::quantise_mix(std::nan("")); }, "cannot convert float NaN to integer",
                                  "quantise_mix NaN");
    throws<std::overflow_error>([] { fe::quantise_mix(HUGE_VAL); }, "cannot convert float infinity to integer",
                                "quantise_mix inf");

    const auto ph = fe::mix_phasor(12345, 5, qrss::Rational(-4679, 4), 4000);
    const cd want[] = {{0.6328742597706853, 0.7742545907643733},
                       {-0.9136519097388281, 0.4064974634983498},
                       {-0.1513902124981477, -0.9884740783448829},
                       {0.9934328103795266, 0.11441700599751596},
                       {-0.372137451742818, 0.9281776322506171}};
    check::equal(ph.size(), std::size_t{5}, "mix_phasor length");
    for (std::size_t i = 0; i < 5; ++i) near(ph[i], want[i], 1e-15, "mix_phasor[" + std::to_string(i) + "]");
    // exact turns: the phasor repeats with the mix's period (4000 * 4 samples here)
    const auto a = fe::mix_phasor(7, 3, qrss::Rational(-4679, 4), 4000);
    const auto b = fe::mix_phasor(7 + 16000 * 1000003LL, 3, qrss::Rational(-4679, 4), 4000);
    check::is_true(a == b, "mix_phasor is periodic to the bit at any distance");
    throws<std::invalid_argument>([] { fe::mix_phasor(0, 3, qrss::Rational(1, 128), 4000); },
                                  "f_mix 1/128 is not on the 1/64 Hz grid", "mix_phasor off the grid");

    const auto x = impulsive();
    const auto c = fe::channelise(std::span<const cf>(x.data(), 8000), 4000, -1169.75, 250, 160);
    check::equal(c.f_mix.num(), std::int64_t{-4679}, "channelise f_mix");
    check::equal(c.f_mix.den(), std::int64_t{4}, "channelise f_mix den");
    check::equal(c.ch.size(), std::size_t{500}, "channelise length");
    near(c.ch[0], {-0.042121682316064835, -0.1456756293773651}, 1e-7, "channelise[0]");
    near(c.ch[100], {-0.07992371171712875, 0.26786044239997864}, 1e-7, "channelise[100]");
    near(c.ch[499], {-0.059379272162914276, -0.018619107082486153}, 1e-7, "channelise[499]");
    throws<std::invalid_argument>([&] { fe::channelise(std::span<const cf>(x.data(), 100), 4000, 10.0, 300); },
                                  "fs_out 300 must divide fs_in 4000", "channelise to a rate that does not divide");

    const auto keep = fe::blank(x).keep;
    const auto ck = fe::channelise_keep(std::span<const float>(keep));
    check::equal(ck.size(), std::size_t{750}, "channelise_keep length");
    const std::pair<std::size_t, double> kp[] = {
        {0, 0.53125}, {186, 1.099552035331726}, {187, 0.2822946012020111}, {400, 1.0}, {749, 0.7670009136199951}};
    for (const auto& [i, v] : kp) near(ck[i], v, 1e-7, "channelise_keep[" + std::to_string(i) + "]");
}

static void test_noise_tables() {
    const auto nr = fe::noise_response(4000, 250, 2500);
    check::equal(nr->size(), std::size_t{2500}, "noise_response length");
    near((*nr)[0], 1.0000000000040241, 1e-14, "noise_response[0]");
    near((*nr)[100], 0.9999977680104436, 1e-14, "noise_response[100]");
    near((*nr)[1250], 0.4999968902842892, 1e-14, "noise_response[1250]");
    near((*nr)[2400], 0.9999977680104436, 1e-14, "noise_response[2400]");
    check::is_true(fe::noise_response(4000, 250, 2500) == nr, "noise_response is cached");
    check::is_true(*fe::noise_response(250, 250, 16) == std::vector<double>(16, 1.0), "ones at the channel rate");

    std::vector<std::uint8_t> all;
    std::size_t flat = 0;
    std::vector<int> audio_hz;
    for (int a = 300; a <= 440; ++a) audio_hz.push_back(a);
    for (int a = 2560; a <= 2700; ++a) audio_hz.push_back(a);
    for (const int a : audio_hz) {
        const auto m = fe::fe_flat(4000, 250, 2500, a);
        all.insert(all.end(), m.begin(), m.end());
        for (const auto v : m) flat += v;
    }
    qrss_check::digest_is(all, "cd464fb6a04b76469c52763e65e2f2d87fd8279a57bd63d57093483b5e144a24",
                          "the noise-floor masks at the band's edges");
    check::equal(flat, std::size_t{99138}, "flat bins over the edges");

    // the caches answer several threads at once with one value
    std::vector<std::thread> threads;
    std::atomic<int> agree{0};
    const auto want = fe::fe_flat(4000, 250, 3000, 333.5);
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 20; ++i) agree += fe::fe_flat(4000, 250, 3000, 333.5) == want;
        });
    for (auto& t : threads) t.join();
    check::equal(agree.load(), 80, "fe_flat from four threads");
    const auto r = fe::fe_response(4000, 250, 2500, 1500.0);
    const auto fp = fe::fe_power(4000, 250, 2500, 1500.0, fe::NOISE_BAND);
    std::vector<bool> band(r.size(), false);
    for (const auto j : fp->idx) band[static_cast<std::size_t>(j)] = true;
    bool ones = !fp->idx.empty(), ratio = true;
    for (std::size_t j = 0; j < r.size(); ++j) ones = ones && (band[j] || r[j] == 1.0);
    for (std::size_t k = 0; k < fp->idx.size(); ++k)
        ratio = ratio && r[static_cast<std::size_t>(fp->idx[k])] == fp->num[k] / fp->den[k];
    check::is_true(ones, "fe_response is 1 outside the band's bins");
    check::is_true(ratio, "fe_response is num / den on them");
}

static void test_noise_psd() {
    const auto ch = complex_noise(104, 105, 7500);
    struct Case {
        bool old;
        std::optional<double> audio;
        std::vector<double> ex;
        double want[3];
    };
    const Case cases[] = {
        {false, 330.25, {100.0}, {0.008950632996857166, 0.010234138928353786, 0.008966822177171707}},
        {false, std::nullopt, {}, {0.0030280929058790207, 0.003529155161231756, 0.003105281153693795}},
        {false, 2690.0, {-60.0}, {0.004297719802707434, 0.003729450050741434, 0.0029433262534439564}},
        {false, 1500.0, {60.0}, {0.0031624911352992058, 0.003799597965553403, 0.0028984961099922657}},
        {false, 330.25, {}, {0.0028884101193398237, 0.002887614304199815, 0.003193024080246687}},
        {false, 2695.0, {100.0, -100.0}, {0.002847229829058051, 0.0035019563511013985, 0.0024242615327239037}},
        {true, 330.25, {100.0}, {0.002549721859395504, 0.0034484020434319973, 0.0024074031971395016}},
        {true, std::nullopt, {}, {0.0030280929058790207, 0.003529155161231756, 0.003105281153693795}},
        {true, 2690.0, {-60.0}, {0.0032929887529462576, 0.003587082028388977, 0.003341000061482191}},
        {true, 1500.0, {60.0}, {0.0031624911352992058, 0.003799597965553403, 0.0028984961099922657}},
        {true, 330.25, {}, {0.0030280929058790207, 0.003529155161231756, 0.003105281153693795}},
        {true, 2695.0, {100.0, -100.0}, {0.003023088676854968, 0.0031852000392973423, 0.0029515856876969337}},
    };
    for (const Case& c : cases) {
        fe::NoisePsdOptions o;
        o.audio_hz = c.audio;
        o.old_edge_noise = c.old;
        const auto v = fe::noise_psd(ch, 250, c.ex, 10.0, o);
        const std::string what = std::string("noise_psd ") + (c.old ? "old " : "") +
                                 (c.audio ? std::to_string(*c.audio) : "None") + " " + std::to_string(c.ex.size());
        check::equal(v.size(), std::size_t{3}, what + " windows");
        for (std::size_t i = 0; i < 3 && i < v.size(); ++i)
            near(v[i], c.want[i], 1e-9, what + " [" + std::to_string(i) + "]");
    }
    fe::NoisePsdOptions o;
    o.audio_hz = 1000.0;
    o.old_edge_noise = false;
    const auto v = fe::noise_psd(ch, 250, {}, 7.0, o);
    const double w7[] = {0.0032499812077730894, 0.002894042292609811, 0.0030088601633906364, 0.0028070767875760794,
                         0.0031927817035466433};
    check::equal(v.size(), std::size_t{5}, "noise_psd 7 s windows");
    for (std::size_t i = 0; i < 5 && i < v.size(); ++i) near(v[i], w7[i], 1e-9, "noise_psd 7 s");

    fe::NoisePsdOptions band;
    band.band_hz = {200.0, 210.0};
    band.old_edge_noise = false;
    throws<std::invalid_argument>([&] { fe::noise_psd(ch, 250, {}, 10.0, band); }, "noise_psd: no bin in the band",
                                  "noise_psd above Nyquist");
    throws<qrss::ZeroDivision>([] { fe::noise_psd(std::span<const cf>(), 250); }, "integer division or modulo by zero",
                               "noise_psd of nothing");
}

int main() {
    try {
        test_taps();
        test_audio_to_fe();
        test_stream();
        test_blank_normalise();
        test_running_median();
        test_mix();
        test_noise_tables();
        test_noise_psd();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_frontend");
}
