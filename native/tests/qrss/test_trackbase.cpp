// native/core/qrss/trackbase against sstvae/qrss/track.py: the process-noise
// maps and the timing maps pinned from Python, PathFn (spline and line)
// pinned from Python with its linear continuation, the refusals, the PSD
// cache slot under concurrent use, MfTable's check and TrackResult's
// accessors. tests/test_qrss_native_trackbase.py compares against Python
// directly. The pins:
//
//   track.q_of_spread(0.5), track.spread_of_q(1e-6), track.q_of_spread(0.5, 0.060625)
//   tm = Timing(tau0=1234.5, ppm=-37.5, gamma=12.25, z=0.0, cov=np.zeros((3, 3)))
//   track.rx_index(tm, [-10.0, 0.0, 777.0, 5000.0], 5000)
//   track._tau_of_rx(tm, [-300.0, 0.0, 12345.0, 40000.0], 5000)
//   tt = np.linspace(0, 120, 25); ff = 1500 + 0.01 * tt + 0.2 * np.sin(tt / 15)
//   fn = track.PathFn(tt, ff, np.linspace(0.5, 2, 25))
//   fn._spl.t, fn._line, fn([-30.0, 0.0, 17.5, 60.0, 119.0, 120.0, 150.0])
//   fl = track.PathFn([0.0, 10.0, 30.0], [1500.0, 1500.5, 1501.0]); fl._line, fl([-5.0, 12.0, 40.0])

#include <atomic>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "math/errors.hpp"
#include "qrss/trackbase.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace track = sstvae::qrss::track;

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    }
    check::fail(what, "did not throw");
}

static std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> v(static_cast<std::size_t>(n));
    const double step = (b - a) / (n - 1);
    for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = i * step + a;
    v.back() = b;
    return v;
}

static void test_maps() {
    check::equal(track::q_of_spread(0.5), 0.00010795055983215056, "q_of_spread");
    check::equal(track::spread_of_q(1e-6), 0.10500944698846705, "spread_of_q");
    check::equal(track::q_of_spread(0.5, 0.060625), 0.0008636044786572045, "q_of_spread at L's period");
    const auto g = track::q_grid();
    for (std::size_t i = 0; i < g.size(); ++i)
        check::equal(g[i], track::Q_GRID[i], "q_grid() is the generated Q_GRID, " + std::to_string(i));

    const qrss::types::Timing tm{1234.5, -37.5, 12.25, 0.0, math::Matrix(3, 3)};
    check::close(track::rx_index(tm, std::vector<double>{-10.0, 0.0, 777.0, 5000.0}, 5000),
                 std::vector<double>{1158.721640796875, 1234.5, 7122.778144592813, 39135.9541015625}, 0.0,
                 "rx_index");
    check::close(track::tau_of_rx(tm, std::vector<double>{-300.0, 0.0, 12345.0, 40000.0}, 5000),
                 std::vector<double>{-202.50096691673494, -162.91091801173806, 1466.04383940038, 5113.948182676499},
                 0.0, "_tau_of_rx");
    check::equal(track::rx_index(tm, 777.0, 5000), 7122.778144592813, "rx_index, scalar");
}

static void test_pathfn() {
    const auto tt = linspace(0.0, 120.0, 25);
    std::vector<double> ff(tt.size());
    for (std::size_t i = 0; i < tt.size(); ++i) ff[i] = 1500 + 0.01 * tt[i] + 0.2 * std::sin(tt[i] / 15);
    const auto w = linspace(0.5, 2.0, 25);
    const track::PathFn fn(tt, ff, std::span<const double>(w));
    check::is_true(fn.spline().has_value(), "a spline");
    check::close(fn.spline()->t,
                 std::vector<double>{0, 0, 0, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 120, 120, 120},
                 0.0, "knots every KNOTS_S");
    check::close(fn.line(), std::vector<double>{0.010427841930243499, 1499.9937406021634}, 1e-12, "line");
    const std::vector<double> at = {-30.0, 0.0, 17.5, 60.0, 119.0, 120.0, 150.0};
    const std::vector<double> want = {1499.3002164570908, 1500.0000017025488, 1500.358891449027, 1500.4485971350205,
                                      1501.3894011613884, 1501.3978732633332, 1501.638167324588};
    check::close(fn(at), want, 1e-10, "PathFn values, both continuations");
    for (std::size_t i = 0; i < at.size(); ++i)
        check::equal(fn(at[i]), fn(std::vector<double>{at[i]})[0], "scalar is the vector's");
    // the continuation is linear: the second difference outside vanishes
    const double d2 = fn(-50.0) - 2 * fn(-40.0) + fn(-30.0);
    check::is_true(std::abs(d2) < 1e-9, "linear before lo");
    // from_parts reproduces it without refitting
    const auto again = track::PathFn::from_parts(fn.lo(), fn.hi(), fn.line(), fn.spline());
    check::close(again(at), fn(at), 0.0, "from_parts");

    // three points: the line only
    const track::PathFn fl(std::vector<double>{0.0, 10.0, 30.0}, std::vector<double>{1500.0, 1500.5, 1501.0});
    check::is_true(!fl.spline().has_value(), "3 points: a line");
    check::close(fl.line(), std::vector<double>{0.0321428571428492, 1500.0714285714291}, 1e-12, "3-point line");
    check::close(fl(std::vector<double>{-5.0, 12.0, 40.0}),
                 std::vector<double>{1499.910714285715, 1500.4571428571433, 1501.3571428571431}, 1e-10, "line values");
    // one point: flat
    const track::PathFn one(std::vector<double>{5.0}, std::vector<double>{1501.0});
    check::equal(one(100.0), 1501.0, "one point: flat");
    // a gap of empty knot intervals: no finite spline, the line
    std::vector<double> tg, fg;
    for (int i = 0; i < 20; ++i) {
        tg.push_back(i * 1.5);
        fg.push_back(1500.0 + 0.001 * i);
    }
    for (int i = 0; i < 20; ++i) {
        tg.push_back(200.0 + i * 3.0);
        fg.push_back(1500.0 + 0.001 * (200.0 + i * 3.0));
    }
    check::is_true(!track::PathFn(tg, fg).spline().has_value(), "knots without data: the line");

    throws<std::out_of_range>([] { track::PathFn(std::vector<double>{}, std::vector<double>{}); },
                              "index 0 is out of bounds for axis 0 with size 0", "no points");
    throws<std::invalid_argument>(
        [] { track::PathFn(std::vector<double>{std::nan("")}, std::vector<double>{1.0}); },
        "cannot convert float NaN to integer", "one NaN point");
    throws<math::LinAlgError>(
        [] { track::PathFn(std::vector<double>{std::nan(""), std::nan("")}, std::vector<double>{1.0, 2.0}); },
        "SVD did not converge in Linear Least Squares", "NaN times");
    throws<qrss::ZeroDivision>(
        [] { track::PathFn(std::vector<double>{1.0, 2.0}, std::vector<double>{1.0, 2.0}, std::nullopt, 0.0); },
        "float floor division by zero", "knots_s 0");
    throws<std::invalid_argument>(
        [] { track::PathFn(std::vector<double>{1.0, 2.0}, std::vector<double>{1.0}); },
        "PathFn: t, f and w must have the same length", "length mismatch");

    const qrss::types::FreqPath path{tt, ff, w};
    check::close(track::freq_path_spline(path)(at), fn(at), 0.0, "freq_path_spline");
}

static void test_psd_slot() {
    track::ChanCapture c;
    std::atomic<int> computed{0};
    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<const track::NoisePsd>> got(8);
    for (std::size_t i = 0; i < got.size(); ++i)
        threads.emplace_back([&, i] {
            got[i] = c.psd.get_or_compute([&] {
                ++computed;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                return track::NoisePsd{{0.0, 10.0}, {1.0, 2.0}};
            });
        });
    for (auto& t : threads) t.join();
    check::equal(computed.load(), 1, "computed once under 8 threads");
    for (const auto& g : got) check::is_true(g.get() == got[0].get(), "every thread sees the one value");
    const track::ChanCapture copy = c;
    check::is_true(copy.psd.get().get() == got[0].get(), "a copy shares the value");

    c.f_mix = qrss::Rational(96017, 64);
    check::equal(c.mix_audio_hz(), 1500.0 + 96017.0 / 64.0, "mix_audio_hz");
    c.t0_index = 100.0;
    c.fs = 250;
    check::equal(c.t_s(350.0), 1.0, "t_s");
}

static void test_records() {
    throws<std::invalid_argument>([] { track::MfTable(0.03, 7.5, 2, std::vector<double>(10), 1.0); },
                                  "tab must hold 2049 taps, not 10", "MfTable size");
    const track::MfTable mt(0.03, 7.5, 2, std::vector<double>(2049, 0.5), 1.0);
    check::equal(mt.size(), std::size_t{2049}, "MfTable size");
    check::is_true(!mt.is_complex(), "real table");
    const track::MfTable copy = mt;
    check::is_true(copy.tab().data() == mt.tab().data(), "copies share the taps");

    const auto spec = qrss::frame::get("short");
    auto chan = std::make_shared<const track::ChanCapture>();
    const qrss::types::Timing tm{1234.5, 3.0, 0.5, 9.0, math::Matrix::identity(3)};
    const track::PathFn fn(std::vector<double>{0.0, 1.0}, std::vector<double>{1500.0, 1500.0});
    track::TrackResult tr(spec, chan, tm, fn);
    throws<std::out_of_range>([&] { (void)tr.p_lo(); }, "index 0 is out of bounds for axis 0 with size 0",
                              "p_lo of an empty grid");
    for (std::int64_t p = -40; p < 60; ++p) tr.pos.push_back(p);
    check::equal(tr.p_lo(), std::int64_t{-40}, "p_lo");
    check::equal(tr.gi(std::int64_t{0}), std::int64_t{40}, "gi");
    tr.q = 1e-6;
    check::equal(tr.doppler_hz(), track::spread_of_q(1e-6, spec.t_sym()), "doppler_hz");
    const auto ts = tr.t_pos_s();
    check::equal(ts[40], track::rx_index(tm, 0.0, spec.n_pos(), qrss::config::CH_FS, spec.t_sym()) / 250.0,
                 "t_pos_s");
    check::is_true(std::isnan(tr.z_ref), "z_ref defaults to NaN");
    track::LTrackResult lt(qrss::frame::get_l("l-short"), chan, tm, fn);
    const track::TrackResult& base = lt;
    check::is_true(dynamic_cast<const track::LTrackResult*>(&base) != nullptr, "LTrackResult is a TrackResult");
    check::equal(lt.delta == std::complex<double>(1.0, 0.0), true, "delta defaults to 1");
}

int main() {
    try {
        test_maps();
        test_pathfn();
        test_psd_slot();
        test_records();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_trackbase");
}
