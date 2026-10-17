// sstvae_native.qrss.special, continued: the normal and Gamma tails the
// template search's statistic stands on (native/core/math/special: ndtri,
// norm_isf, gamma_logsf) and Python's math.lgamma (CPython's own Lanczos,
// not the C library's), for the parity tests
// (tests/test_qrss_native_em.py). Added to the submodule special.cpp
// creates (def_submodule returns the module already in sys.modules), so
// it registers after it.
//
//   special.ndtri(y)            scipy.special.ndtri
//   special.norm_isf(q)         scipy.stats.norm.isf(q) (loc 0, scale 1)
//   special.gamma_logsf(x, a)   scipy.stats.gamma.logsf(x, a) (loc 0, scale 1)
//   special.py_lgamma(x)        math.lgamma(x), +inf where Python raises
//                               (a pole: ValueError; an overflow:
//                               OverflowError)
//
// Each broadcasts like a ufunc over float64; a 0-d result is a numpy
// float64 scalar.

#include "qrss/rx_types.hpp"

#include <vector>

#include "math/special.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace sm = sstvae::math;

using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

DArray as_f64(const py::object& x) {
    return DArray::ensure(rx::np_module().attr("asarray")(x, py::arg("dtype") = "float64"));
}

py::object finish(const py::array_t<double>& out) {
    if (out.ndim() == 0) return rx::np_module().attr("float64")(out.attr("item")());
    return out;
}

py::array_t<double> like(const DArray& a) {
    return py::array_t<double>(std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim()));
}

template <typename F>
py::object unary(const py::object& x, F f) {
    const DArray a = as_f64(x);
    py::array_t<double> out = like(a);
    const double* s = a.data();
    double* d = out.mutable_data();
    for (py::ssize_t i = 0; i < a.size(); ++i) d[i] = f(s[i]);
    return finish(out);
}

template <typename F>
py::object binary(const py::object& x, const py::object& y, F f) {
    py::tuple b = rx::np_module().attr("broadcast_arrays")(as_f64(x), as_f64(y));
    const DArray a = DArray::ensure(b[0]);
    const DArray c = DArray::ensure(b[1]);
    py::array_t<double> out = like(a);
    const double* sa = a.data();
    const double* sc = c.data();
    double* d = out.mutable_data();
    for (py::ssize_t i = 0; i < a.size(); ++i) d[i] = f(sa[i], sc[i]);
    return finish(out);
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("special");
    m.def("ndtri", [](const py::object& y) { return unary(y, [](double v) { return sm::ndtri(v); }); },
          py::arg("y"));
    m.def("norm_isf", [](const py::object& qq) { return unary(qq, [](double v) { return sm::norm_isf(v); }); },
          py::arg("q"));
    m.def("gamma_logsf", [](const py::object& x, const py::object& a) {
        return binary(x, a, [](double u, double v) { return sm::gamma_logsf(u, v); });
    }, py::arg("x"), py::arg("a"));
    m.def("py_lgamma", [](const py::object& x) { return unary(x, [](double v) { return sm::py_lgamma(v); }); },
          py::arg("x"));
}

const Registrar reg("special_tail", 7, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
