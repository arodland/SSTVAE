// sstvae_native.qrss.math: bindings of native/core/math (numpy's median,
// mean, average, interp, polyfit, polyval and linalg routines, scipy's
// make_lsq_spline and BSpline), for the parity tests. The receiver's
// modules call these from C++; nothing in the reference is substituted by
// them directly.
//
//   median(x), mean(x)              float32 input stays float32 (a numpy scalar)
//   average(a, weights)             float64
//   interp(x, xp, fp, left=None, right=None)
//   polyfit(x, y, deg, rcond=None, w=None), polyval(p, x)
//   lstsq(a, b, rcond=None) -> (x, residuals, rank, s)
//   solve(a, b), inv(a), eigh(a) -> (w, v), eigvalsh(a)
//   make_lsq_spline(x, y, t, k=3, w=None) -> (t, c, k)
//   bspline_eval(t, c, k, x), bspline_derivative(t, c, k, nu=1) -> (t, c, k)
//   dlartg(f, g) -> (c, s, r)
//
// Errors are numpy's and scipy's types (rx_types.hpp's translators):
// numpy.linalg.LinAlgError, ValueError, TypeError, ZeroDivisionError.

#include "qrss/rx_types.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "math/npstats.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace sm = sstvae::math;

struct Doubles {
    std::vector<double> v;
    py::tuple shape;
};

Doubles doubles(const py::handle& o) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o, py::arg("dtype") = "float64"));
    if (!a) throw py::error_already_set();
    return {from_numpy<double>(a), py::tuple(a.attr("shape"))};
}

py::object shaped(const std::vector<double>& v, const py::tuple& shape) {
    py::object a = to_numpy(v).attr("reshape")(shape);
    if (shape.empty()) return a[py::tuple()];
    return a;
}

std::optional<double> opt_double(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    return rx::as_double(o);
}

bool is_float32(const py::handle& o) {
    py::object a = rx::np_module().attr("asarray")(o);
    return py::str(a.attr("dtype")).cast<std::string>() == "float32";
}

py::object stat(const py::object& x, double (*f64)(std::span<const double>), float (*f32)(std::span<const float>)) {
    if (is_float32(x)) {
        const std::vector<float> v = rx::vec<float>(x);
        return rx::np_module().attr("float32")(f32(v));
    }
    const std::vector<double> v = rx::vec<double>(x);
    return rx::np_module().attr("float64")(f64(v));
}

sm::BSpline spline(const py::object& t, const py::object& c, int k) {
    return sm::BSpline{rx::vec<double>(t), rx::vec<double>(c), k};
}

py::tuple spline_tuple(const sm::BSpline& s) { return py::make_tuple(to_numpy(s.t), to_numpy(s.c), s.k); }

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("math", "numpy/scipy numerics used by the receiver (native/core/math)");

    m.def("median", [](const py::object& x) {
        return stat(x, static_cast<double (*)(std::span<const double>)>(&sm::median),
                    static_cast<float (*)(std::span<const float>)>(&sm::median));
    }, py::arg("x"));
    m.def("mean", [](const py::object& x) {
        return stat(x, static_cast<double (*)(std::span<const double>)>(&sm::mean),
                    static_cast<float (*)(std::span<const float>)>(&sm::mean));
    }, py::arg("x"));

    m.def("average", [](const py::object& a, const py::object& w) {
        const std::vector<double> av = rx::vec<double>(a), wv = rx::vec<double>(w);
        return rx::np_module().attr("float64")(sm::average(av, wv));
    }, py::arg("a"), py::arg("weights"));

    m.def("interp", [](const py::object& x, const py::object& xp, const py::object& fp, const py::object& left,
                       const py::object& right) {
        const Doubles d = doubles(x);
        const std::vector<double> xv = rx::vec<double>(xp), fv = rx::vec<double>(fp);
        return shaped(sm::interp(std::span<const double>(d.v), xv, fv, opt_double(left), opt_double(right)),
                      d.shape);
    }, py::arg("x"), py::arg("xp"), py::arg("fp"), py::arg("left") = py::none(), py::arg("right") = py::none());

    m.def("polyfit", [](const py::object& x, const py::object& y, int deg, const py::object& rcond,
                        const py::object& w) {
        const std::vector<double> xv = rx::vec<double>(x), yv = rx::vec<double>(y);
        std::optional<std::vector<double>> wv;
        if (!w.is_none()) wv = rx::vec<double>(w);
        std::optional<std::span<const double>> ws;
        if (wv) ws = std::span<const double>(*wv);
        return to_numpy(sm::polyfit(xv, yv, deg, ws, opt_double(rcond)));
    }, py::arg("x"), py::arg("y"), py::arg("deg"), py::arg("rcond") = py::none(), py::arg("w") = py::none());
    m.def("polyval", [](const py::object& p, const py::object& x) {
        const std::vector<double> pv = rx::vec<double>(p);
        const Doubles d = doubles(x);
        return shaped(sm::polyval(pv, std::span<const double>(d.v)), d.shape);
    }, py::arg("p"), py::arg("x"));

    m.def("lstsq", [](const py::object& a, const py::object& b, const py::object& rcond) {
        const sm::Matrix am = rx::matrix(a);
        const std::vector<double> bv = rx::vec<double>(b);
        const sm::LstsqResult r = sm::lstsq(am, bv, opt_double(rcond));
        return py::make_tuple(to_numpy(r.x), to_numpy(r.residuals), r.rank, to_numpy(r.s));
    }, py::arg("a"), py::arg("b"), py::arg("rcond") = py::none());
    m.def("solve", [](const py::object& a, const py::object& b) {
        const std::vector<double> bv = rx::vec<double>(b);
        return to_numpy(sm::solve(rx::matrix(a), bv));
    }, py::arg("a"), py::arg("b"));
    m.def("inv", [](const py::object& a) { return rx::to_py(sm::inv(rx::matrix(a))); }, py::arg("a"));
    m.def("eigh", [](const py::object& a) {
        const sm::EighResult r = sm::eigh(rx::matrix(a));
        return py::make_tuple(to_numpy(r.w), rx::to_py(r.v));
    }, py::arg("a"));
    m.def("eigvalsh", [](const py::object& a) { return to_numpy(sm::eigvalsh(rx::matrix(a))); }, py::arg("a"));

    m.def("make_lsq_spline", [](const py::object& x, const py::object& y, const py::object& t, int k,
                                const py::object& w) {
        const std::vector<double> xv = rx::vec<double>(x), yv = rx::vec<double>(y), tv = rx::vec<double>(t);
        std::optional<std::vector<double>> wv;
        if (!w.is_none()) wv = rx::vec<double>(w);
        std::optional<std::span<const double>> ws;
        if (wv) ws = std::span<const double>(*wv);
        return spline_tuple(sm::make_lsq_spline(xv, yv, tv, k, ws));
    }, py::arg("x"), py::arg("y"), py::arg("t"), py::arg("k") = 3, py::arg("w") = py::none());
    m.def("bspline_eval", [](const py::object& t, const py::object& c, int k, const py::object& x) {
        const sm::BSpline s = spline(t, c, k);
        const Doubles d = doubles(x);
        return shaped(s(std::span<const double>(d.v)), d.shape);
    }, py::arg("t"), py::arg("c"), py::arg("k"), py::arg("x"));
    m.def("bspline_derivative", [](const py::object& t, const py::object& c, int k, int nu) {
        return spline_tuple(spline(t, c, k).derivative(nu));
    }, py::arg("t"), py::arg("c"), py::arg("k"), py::arg("nu") = 1);
    m.def("dlartg", [](double f, double g) {
        const sm::Givens r = sm::dlartg(f, g);
        return py::make_tuple(r.c, r.s, r.r);
    }, py::arg("f"), py::arg("g"));
}

const Registrar reg("math", 5, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
