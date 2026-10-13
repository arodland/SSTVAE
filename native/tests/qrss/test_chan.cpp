// native/core/qrss/chan (track.make_chan, ChanCapture.noise_psd_at and
// noise_floor) against values pinned from the Python reference. Inputs
// are splitmix64 counters (infra_testgen.hpp):
//
//   fe0 = (unit(splitmix(101, 12000)) * 2 - 1
//          + 1j * (unit(splitmix(102, 12000)) * 2 - 1)).astype(np.complex64)
//   fe0[[3000, 3001, 7000, 11999]] *= 40
//   b_fe, b_keep = frontend.blank(fe0)
//   cap = track.make_chan(b_fe, b_keep, 4000.37, 4000, 330.25, -1.5, 2.25, (400.0,))
//   big = (unit(splitmix(111, 160000)) * 2 - 1
//          + 1j * (unit(splitmix(112, 160000)) * 2 - 1)).astype(np.complex64)
//   cap = track.make_chan(big, None, 40000.0, 4000, 812.4, -20.0, 25.0, (700.0,))
//   cap.noise_floor(), cap._psd, cap.noise_psd_at(t)
//
// The noise table's seam (NoisePsdFn) is checked with a stub: its
// arguments, the reference's fill-in rule, and one computation however
// many threads ask. tests/test_qrss_native_chan.py compares against the
// reference directly.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "infra_testgen.hpp"
#include "qrss/chan.hpp"
#include "qrss/frontend.hpp"
#include "qrss/rational.hpp"
#include "qrss/rules.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace track = sstvae::qrss::track;
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

static void near(double got, double want, double rtol, double atol, const std::string& what) {
    ++check::checks;
    if (!(std::abs(got - want) <= atol + rtol * std::abs(want))) {
        char buf[200];
        std::snprintf(buf, sizeof buf, "got %.17g, want %.17g", got, want);
        check::fail(what, buf);
    }
}

static void near(cf got, cd want, double tol, const std::string& what) {
    ++check::checks;
    if (!(std::abs(cd(got.real(), got.imag()) - want) <= tol)) {
        char buf[200];
        std::snprintf(buf, sizeof buf, "got {%.9g, %.9g}, want {%.17g, %.17g}", static_cast<double>(got.real()),
                      static_cast<double>(got.imag()), want.real(), want.imag());
        check::fail(what, buf);
    }
}

static std::vector<cf> complex_noise(std::uint64_t seed_re, std::uint64_t seed_im, std::size_t n) {
    const auto a = qrss_testgen::splitmix(seed_re, n), b = qrss_testgen::splitmix(seed_im, n);
    std::vector<cf> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = cf(static_cast<float>(qrss_testgen::unit(a[i]) * 2 - 1),
                    static_cast<float>(qrss_testgen::unit(b[i]) * 2 - 1));
    return out;
}

static void test_make_chan() {
    auto x = complex_noise(101, 102, 12000);
    for (const std::size_t i : {3000, 3001, 7000, 11999}) x[i] = cf(x[i].real() * 40.0f, x[i].imag() * 40.0f);
    const auto b = fe::blank(x);
    const std::vector<double> ex{400.0};
    const auto c = track::make_chan(b.fe, std::span<const float>(b.keep), 4000.37, 4000, 330.25, -1.5, 2.25, ex);
    check::equal(c.ch.size(), std::size_t{938}, "make_chan length");
    check::equal(c.fs, 250, "make_chan fs");
    check::is_true(c.f_mix == qrss::Rational(-4679, 4), "make_chan f_mix");
    check::equal(c.t0_index, 375.023125, "make_chan t0_index");
    check::is_true(c.exclude_hz == ex, "make_chan exclude_hz");
    check::is_true(c.ch[0] == cf(0, 0) && c.ch[937] == cf(0, 0), "zero padding past both ends");
    near(c.ch[125], {0.1306237280368805, -0.07702641189098358}, 1e-7, "make_chan ch[125]");
    near(c.ch[500], {-0.05852373689413071, 0.03819442167878151}, 1e-7, "make_chan ch[500]");
    check::is_true(c.keep.has_value() && c.keep->size() == 938, "make_chan keep");
    if (c.keep) {
        near((*c.keep)[0], 0.0, 0, 0, "keep[0]");
        near((*c.keep)[125], 0.53125, 1e-7, 0, "keep[125]");
        near((*c.keep)[500], 1.0, 1e-7, 0, "keep[500]");
        near((*c.keep)[937], 0.0, 0, 0, "keep[937]");
    }
    check::is_true(!c.psd.get(), "no table until asked");

    const auto d = track::make_chan(b.fe, std::nullopt, 100.0, 4000, 1000.0, 2.0, 3.0);
    check::equal(d.ch.size(), std::size_t{251}, "make_chan 2 length");
    check::is_true(d.f_mix == qrss::Rational(-500, 1), "make_chan 2 f_mix");
    check::equal(d.t0_index, -499.75, "make_chan 2 t0_index");
    std::size_t nz = 0;
    for (const cf v : d.ch) nz += v != cf(0, 0);
    check::equal(nz, std::size_t{251}, "make_chan 2: all inside the capture");
    check::is_true(!d.keep.has_value(), "make_chan 2: no keep");

    const std::span<const cf> s(b.fe);
    throws<qrss::ZeroDivision>([&] { track::make_chan(s, std::nullopt, 0, 100, 900.0, 0.0, 1.0); },
                               "integer modulo by zero", "fs_fe below CH_FS");
    throws<std::invalid_argument>([&] { track::make_chan(s, std::nullopt, 0, 4000, 900.0, 1.0, -1.0); },
                                  "negative dimensions are not allowed", "a negative span");
    throws<std::invalid_argument>([&] { track::make_chan(s, std::nullopt, std::nan(""), 4000, 900.0, 0.0, 1.0); },
                                  "cannot convert float NaN to integer", "a NaN t0");
    throws<std::overflow_error>([&] { track::make_chan(s, std::nullopt, 0, 4000, 900.0, -HUGE_VAL, 1.0); },
                                "cannot convert float infinity to integer", "an infinite time");
    throws<std::invalid_argument>([&] { track::make_chan(s, std::nullopt, 0, 4100, 900.0, 0.0, 1.0); },
                                  "fs_out 250 must divide fs_in 4100", "a rate CH_FS does not divide");
}

static void test_noise_table() {
    const auto big = complex_noise(111, 112, 160000);
    const std::vector<double> ex{700.0};
    const double t_want[] = {-15.0, -5.0, 5.0, 15.0, 20.0};
    const double p_want[] = {0.00014782065409235656, 0.00017795649182517081, 0.00014805138926021755,
                             0.0001475899334764108, 0.00013808862422592938};
    const double at_t[] = {-30.0, 0.0, 3.3, 40.0};
    const double at_want[] = {0.00014782065409235656, 0.00016300394054269418, 0.00015313525669625962,
                              0.00013808862422592938};
    for (const bool old : {false, true}) {   // the same at 812 Hz under both rules
        qrss::rules::set_old(qrss::rules::EDGE_NOISE, old);
        const std::string tag = old ? " (old rule)" : "";
        const auto c = track::make_chan(big, std::nullopt, 40000.0, 4000, 812.4, -20.0, 25.0, ex);
        check::equal(c.ch.size(), std::size_t{11250}, "big length" + tag);
        check::is_true(c.f_mix == qrss::Rational(-22003, 32), "big f_mix" + tag);
        check::equal(c.t0_index, 5000.0, "big t0_index" + tag);
        near(track::noise_floor(c), 0.00014782065409235656, 1e-9, 0, "noise_floor" + tag);
        const auto t = track::noise_psd_table(c);
        check::equal(t->t_s.size(), std::size_t{5}, "five windows" + tag);
        for (std::size_t i = 0; i < 5 && i < t->t_s.size(); ++i) {
            check::equal(t->t_s[i], t_want[i], "window centre " + std::to_string(i) + tag);
            near(t->psd[i], p_want[i], 1e-9, 0, "window psd " + std::to_string(i) + tag);
        }
        for (std::size_t i = 0; i < 4; ++i)
            near(track::noise_psd_at(c, at_t[i]), at_want[i], 1e-9, 0, "noise_psd_at " + std::to_string(at_t[i]) + tag);
        const auto v = track::noise_psd_at(c, std::span<const double>(at_t));
        check::equal(v.size(), std::size_t{4}, "noise_psd_at over an array" + tag);
        for (std::size_t i = 0; i < 4 && i < v.size(); ++i)
            check::equal(v[i], track::noise_psd_at(c, at_t[i]), "array = scalar " + std::to_string(i) + tag);
        // a copy carries the table
        const track::ChanCapture copy = c;
        check::is_true(copy.psd.get() == c.psd.get(), "a copy holds the same table" + tag);
    }
    qrss::rules::set_old(qrss::rules::EDGE_NOISE, false);
}

static void test_seam() {
    // the reference's seam test: ones with the first 3 s zero; the stub's
    // PSD [2, 0, 4] keeps only the last window (the first is 70% padding,
    // the second reads 0), so every window takes its 4
    std::vector<cf> ch(250 * 25, cf(1, 0));
    std::fill(ch.begin(), ch.begin() + 750, cf(0, 0));
    track::ChanCapture c;
    c.ch = ch;
    c.fs = 250;
    c.t0_index = 10.0;
    c.f_mix = qrss::Rational(-4679, 4);
    c.exclude_hz = {400.0, 250.0};
    std::atomic<int> calls{0};
    std::vector<double> seen_ex;
    double seen_win = 0, seen_audio = 0;
    std::int64_t seen_fs = 0;
    std::size_t seen_n = 0;
    const track::NoisePsdFn stub = [&](std::span<const cf> x, std::int64_t fs, std::span<const double> exclude,
                                       double win_s, std::optional<double> audio_hz) {
        ++calls;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));   // let the other threads pile up
        seen_n = x.size();
        seen_fs = fs;
        seen_ex.assign(exclude.begin(), exclude.end());
        seen_win = win_s;
        seen_audio = audio_hz.value_or(-1);
        return std::vector<float>{2.0f, 0.0f, 4.0f};
    };
    std::vector<std::thread> threads;
    std::atomic<int> agree{0};
    for (int i = 0; i < 8; ++i)
        threads.emplace_back([&] { agree += track::noise_psd_at(c, 50.0, stub) == 4.0; });
    for (auto& t : threads) t.join();
    check::equal(calls.load(), 1, "the table is computed once for eight threads");
    check::equal(agree.load(), 8, "and every thread reads it");
    check::equal(seen_n, ch.size(), "the seam gets the channel");
    check::equal(seen_fs, std::int64_t{250}, "and its rate");
    check::is_true(seen_ex == std::vector<double>{400.0 - 330.25, 250.0 - 330.25},
                   "exclude_hz relative to the mix's audio frequency");
    check::equal(seen_win, track::NOISE_WIN_S, "win_s = NOISE_WIN_S");
    check::equal(seen_audio, 330.25, "audio_hz = the mix's audio frequency");
    const auto t = c.psd.get();
    check::is_true(t && t->psd == std::vector<double>{4.0, 4.0, 4.0}, "bad windows take the good ones' median");
    check::is_true(t && t->t_s == std::vector<double>{(1250 - 10.0) / 250, (3750 - 10.0) / 250, (5000 - 10.0) / 250},
                   "window centres in receiver time");
    check::equal(track::noise_floor(c, stub), 4.0, "noise_floor");
    check::equal(calls.load(), 1, "and noise_floor uses the cached table");

    // a table already in the slot is used as it is
    track::ChanCapture planted = c;
    planted.psd.set(std::make_shared<const track::NoisePsd>(track::NoisePsd{{0.0, 10.0}, {1.0, 3.0}}));
    check::equal(track::noise_psd_at(planted, 5.0), 2.0, "a planted table");
    check::equal(track::noise_floor(planted), 2.0, "its median");
}

static void test_edges() {
    // silence: no good window, every window max(median, 1e-30)
    track::ChanCapture z;
    z.ch.assign(250 * 30, cf(0, 0));
    check::equal(track::noise_floor(z), 1e-30, "the floor of silence");
    // an empty channel: noise_psd refuses first
    track::ChanCapture e;
    throws<qrss::ZeroDivision>([&] { track::noise_psd_at(e, 0.0); }, "integer division or modulo by zero",
                               "an empty channel");
    // no window from the seam: np.interp's refusal, after the empty table is cached
    track::ChanCapture n;
    n.ch.assign(100, cf(1, 0));
    const track::NoisePsdFn none = [](std::span<const cf>, std::int64_t, std::span<const double>, double,
                                      std::optional<double>) { return std::vector<float>{}; };
    throws<std::invalid_argument>([&] { track::noise_psd_at(n, 0.0, none); }, "array of sample points is empty",
                                  "no window");
    check::is_true(n.psd.get() && n.psd.get()->t_s.empty(), "the empty table is cached all the same");
}

int main() {
    try {
        test_make_chan();
        test_noise_table();
        test_seam();
        test_edges();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_chan");
}
