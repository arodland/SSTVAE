// sstvae_native.qrss.cubic_spline: native/core/math/cubic_spline, scipy's
// CubicSpline of complex values with not-a-knot ends, as
// acquire.preamble_search uses it. tests/qrss_native/acquire_stats.py
// substitutes CubicSpline for the name acquire.py imports from
// scipy.interpolate, so the class keeps scipy's call signature:
//
//   CubicSpline(x, y, axis=0, bc_type="not-a-knot", extrapolate=None)
//   spline(x, nu=0, extrapolate=None) -> complex128, x's shape
//   spline.x, spline.c (shape (4, n-1)), spline.axis
//
// Only what the receiver uses is ported: axis 0 of a 1-D complex y,
// bc_type "not-a-knot", extrapolation on. Anything else raises
// NotImplementedError rather than computing something else; a real y is
// refused too (scipy's real path divides where its complex path
// multiplies by a reciprocal, so it is not the same arithmetic).

#include "qrss/rx_types.hpp"

#include <algorithm>
#include <complex>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "math/cubic_spline.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace sm = sstvae::math;

[[noreturn]] void not_implemented(const std::string& what) {
    PyErr_SetString(PyExc_NotImplementedError, ("native CubicSpline: " + what).c_str());
    throw py::error_already_set();
}

std::shared_ptr<sm::CubicSpline> make(const py::object& x, const py::object& y, int axis, const py::object& bc_type,
                                      const py::object& extrapolate) {
    py::module_ np = rx::np_module();
    py::array xa = py::array::ensure(np.attr("asarray")(x));
    py::array ya = py::array::ensure(np.attr("asarray")(y));
    if (!xa || !ya) throw py::error_already_set();
    if (np.attr("iscomplexobj")(xa).cast<bool>()) throw py::value_error("`x` must contain real values.");
    if (!np.attr("iscomplexobj")(ya).cast<bool>()) not_implemented("only complex y is ported");
    if (!py::isinstance<py::str>(bc_type) || bc_type.cast<std::string>() != "not-a-knot")
        not_implemented("only bc_type='not-a-knot' is ported");
    if (!extrapolate.is_none() && !rx::as_bool(extrapolate)) not_implemented("only extrapolate=True is ported");
    if (ya.ndim() != 1 || (axis != 0 && axis != -1)) not_implemented("only a 1-D y along axis 0 is ported");
    if (xa.ndim() != 1) throw py::value_error("`x` must be 1-dimensional.");
    const std::vector<double> xv = rx::vec<double>(xa);
    const std::vector<std::complex<double>> yv = rx::vec<std::complex<double>>(ya);
    return std::make_shared<sm::CubicSpline>(xv, yv);
}

py::object call(const sm::CubicSpline& s, const py::object& x, int nu, const py::object& extrapolate) {
    if (!extrapolate.is_none() && !rx::as_bool(extrapolate)) not_implemented("only extrapolate=True is ported");
    py::array xa = py::array::ensure(rx::np_module().attr("asarray")(x, py::arg("dtype") = "float64"));
    if (!xa) throw py::error_already_set();
    const std::vector<double> xv = rx::vec<double>(xa);
    const std::vector<std::complex<double>> out = s(std::span<const double>(xv), nu);
    py::array_t<std::complex<double>> res(std::vector<py::ssize_t>(xa.shape(), xa.shape() + xa.ndim()));
    std::copy(out.begin(), out.end(), res.mutable_data());
    return res;
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("cubic_spline", "scipy.interpolate.CubicSpline, complex not-a-knot (native/core/math)");
    py::class_<sm::CubicSpline, std::shared_ptr<sm::CubicSpline>>(m, "CubicSpline")
        .def(py::init(&make), py::arg("x"), py::arg("y"), py::arg("axis") = 0, py::arg("bc_type") = "not-a-knot",
             py::arg("extrapolate") = py::none())
        .def("__call__", &call, py::arg("x"), py::arg("nu") = 0, py::arg("extrapolate") = py::none())
        .def_property_readonly("x", [](const sm::CubicSpline& s) { return to_numpy(s.x()); })
        .def_property_readonly("c", [](const sm::CubicSpline& s) {
            const std::size_t m = s.size() - 1;
            py::array_t<std::complex<double>> c(std::vector<py::ssize_t>{4, static_cast<py::ssize_t>(m)});
            std::copy(s.c().begin(), s.c().end(), c.mutable_data());
            return c;
        })
        .def_property_readonly("axis", [](const sm::CubicSpline&) { return 0; })
        .def_property_readonly("extrapolate", [](const sm::CubicSpline&) { return true; });
}

const Registrar reg("cubic_spline", 6, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
