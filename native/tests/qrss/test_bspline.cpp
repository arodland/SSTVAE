// native/core/math/bspline against scipy 1.18.1: make_lsq_spline,
// BSpline evaluation (with extrapolation) and derivative pinned from
// scipy, LAPACK's dlartg to the bit, and scipy's refusals with their
// texts. tests/test_qrss_native_math.py compares against scipy directly,
// on random fits. The pins:
//
//   x = np.linspace(0, 10, 20); y = np.sin(x)
//   t = np.r_[[0.0] * 4, 3.0, 5.0, [10.0] * 4]
//   spl = interpolate.make_lsq_spline(x, y, t, k=3, w=np.linspace(1, 2, 20))
//   pts = [-2.0, 0.0, 1.3, 3.0, 4.4, 9.9, 10.0, 12.5]
//   spl.c, spl(pts), spl.derivative().t, .c, spl.derivative()(pts)
//   scipy.linalg.lapack.dlartg(f, g)

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "math/bspline.hpp"
#include "qrss_check.hpp"

using namespace sstvae;

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
    for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = a + i * step;   // np.linspace's a + i*step
    v.back() = b;
    return v;
}

static const std::vector<double> KNOTS = {0.0, 0.0, 0.0, 0.0, 3.0, 5.0, 10.0, 10.0, 10.0, 10.0};
static const std::vector<double> PTS = {-2.0, 0.0, 1.3, 3.0, 4.4, 9.9, 10.0, 12.5};

static void test_fit() {
    const auto x = linspace(0.0, 10.0, 20);
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = std::sin(x[i]);
    const auto w = linspace(1.0, 2.0, 20);
    const math::BSpline spl = math::make_lsq_spline(x, y, KNOTS, 3, std::span<const double>(w));
    check::close(spl.c,
                 std::vector<double>{-0.16877230919116296, 2.0298853946561732, -0.0530218555651799,
                                     -1.7428583440782752, 2.8833786775940293, -0.7724782998515429},
                 1e-13, "make_lsq_spline coefficients");
    check::close(spl.t, KNOTS, 0.0, "knots kept");
    check::close(spl(PTS),
                 std::vector<double>{-10.317758039563804, -0.16877230919116296, 1.0637454590012638,
                                     -0.023927263462120596, -0.7637332853045403, -0.5613998623960355,
                                     -0.7724782998515429, -12.717016515163605},
                 1e-12, "evaluation, extrapolating both sides");
    const math::BSpline d = spl.derivative();
    check::close(d.t, std::vector<double>{0.0, 0.0, 0.0, 3.0, 5.0, 10.0, 10.0, 10.0}, 0.0, "derivative knots");
    check::equal(d.k, 2, "derivative degree");
    check::close(d.c,
                 std::vector<double>{2.1986577038473363, -1.2497443501328118, -0.5069509465539286,
                                     1.9826730092881306, -2.1935141864673433, 0.0, 0.0, 0.0},
                 1e-12, "derivative coefficients (padded to len(t))");
    check::close(d(PTS),
                 std::vector<double>{8.527228485210857, 2.1986577038473363, -0.05873607821818804,
                                     -0.804068307985482, -0.18514415526487982, -2.0288484946456675,
                                     -2.1935141864673433, -7.858323887562053},
                 1e-12, "derivative evaluation");
    check::is_true(std::isnan(spl(std::nan(""))), "spl(nan) is nan");
}

static void test_dlartg() {
    struct Case {
        double f, g, c, s, r;
    };
    const Case cases[] = {
        {3.0, 4.0, 0.6, 0.8, 5.0},
        {0.0, -2.0, 0.0, -1.0, 2.0},
        {-5.0, 0.0, 1.0, 0.0, -5.0},
        {1e-300, 1e-300, 0.7071067811865475, 0.7071067811865475, 1.4142135623730952e-300},
        {1e300, -3e300, 0.3162277660168379, -0.9486832980505138, 3.1622776601683795e+300},
        {-7.0, 2.0, 0.9615239476408232, -0.27472112789737807, -7.280109889280518},
    };
    for (const Case& k : cases) {
        const auto g = math::dlartg(k.f, k.g);
        const std::string what = "dlartg(" + std::to_string(k.f) + ", " + std::to_string(k.g) + ")";
        check::equal(g.c, k.c, what + " c");
        check::equal(g.s, k.s, what + " s");
        check::equal(g.r, k.r, what + " r");
    }
}

static void test_refusals() {
    const auto x = linspace(0.0, 10.0, 20);
    std::vector<double> y(x.size(), 1.0);
    std::vector<double> bad = x;
    bad[3] = std::nan("");
    throws<std::invalid_argument>([&] { math::make_lsq_spline(bad, y, KNOTS, 3); },
                                  "Array must not contain infs or nans.", "NaN x");
    throws<std::invalid_argument>(
        [&] {
            math::make_lsq_spline(std::vector<double>(x.begin(), x.begin() + 3),
                                  std::vector<double>(y.begin(), y.begin() + 3), KNOTS, 3);
        },
        "Need more x points.", "3 points");
    throws<std::invalid_argument>([&] { math::make_lsq_spline(x, y, KNOTS, -1); }, "Expect non-negative k.",
                                  "k < 0");
    std::vector<double> t_desc = KNOTS;
    std::swap(t_desc[4], t_desc[5]);
    throws<std::invalid_argument>([&] { math::make_lsq_spline(x, y, t_desc, 3); },
                                  "Expect t to be a 1D strictly increasing sequence.", "decreasing knots");
    throws<std::invalid_argument>(
        [&] { math::make_lsq_spline(x, std::vector<double>(y.begin(), y.end() - 1), KNOTS, 3); },
        "Shapes of x (20,) and y (19,) are incompatible", "y length");
    std::vector<double> x_desc = x;
    std::swap(x_desc[4], x_desc[5]);
    throws<std::invalid_argument>([&] { math::make_lsq_spline(x_desc, y, KNOTS, 3); },
                                  "Expect x to be a 1D non-decreasing sequence.", "unsorted x");
    // too few points for the coefficients: fpback's refusal
    throws<std::invalid_argument>(
        [&] {
            math::make_lsq_spline(std::vector<double>{0.0, 1.0, 2.0, 3.0, 10.0},
                                  std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0},
                                  std::vector<double>{0, 0, 0, 0, 1, 2, 3, 10, 10, 10, 10}, 3);
        },
        "nc = 7 > m = 5", "nc > m");
    // internal repeated knots: not differentiable that many times
    const math::BSpline rep{{0, 0, 0, 0, 5, 5, 5, 5, 10, 10, 10, 10}, {1, 2, 3, 4, 5, 6, 7, 8}, 3};
    throws<std::invalid_argument>([&] { (void)rep.derivative(1); },
                                  "The spline has internal repeated knots and is not differentiable 1 times",
                                  "repeated knots");
}

int main() {
    try {
        test_fit();
        test_dlartg();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_bspline");
}
