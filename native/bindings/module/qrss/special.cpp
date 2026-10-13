// sstvae_native.qrss.special and sstvae_native.qrss.ndfilter: bindings of
// native/core/math/special (scipy.special's incomplete Gamma functions and
// the cephes pieces under them) and native/core/math/ndfilter
// (scipy.ndimage.median_filter, 1-D, mode "reflect"), for the parity tests.
// The receiver calls these from C++ (acquire_stats); nothing in the
// reference is substituted by them directly.
//
//   special.gammaincc(a, x), gammainc(a, x), gammainccinv(a, y),
//   gammaincinv(a, y)        broadcast like the ufuncs, float64; a 0-d
//                            result is a numpy float64 scalar
//   special.gammaincc_many(a, x)   one shape, a 1-D array of x (the
//                            receiver's vectorised call)
//   special.lgam(x), gamma(x), erfc(x), zeta(x, q)   likewise
//   ndfilter.median_filter(x, size, mode="reflect")  1-D float64 or
//                            float32 (dtype kept); only mode "reflect"
//                            and an odd size (ValueError otherwise)

#include "qrss/rx_types.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "math/ndfilter.hpp"
#include "math/special.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace sm = sstvae::math;

using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::object finish(const py::array_t<double>& out) {
    if (out.ndim() == 0) return rx::np_module().attr("float64")(out.attr("item")());
    return out;
}

py::array_t<double> like(const DArray& a) {
    return py::array_t<double>(std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim()));
}

DArray as_f64(const py::object& x) {
    return DArray::ensure(rx::np_module().attr("asarray")(x, py::arg("dtype") = "float64"));
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

void bind_special(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("special", "scipy.special's incomplete Gamma functions (native/core/math/special)");
    m.def("gammaincc", [](const py::object& a, const py::object& x) {
        return binary(a, x, [](double u, double v) { return sm::gammaincc(u, v); });
    }, py::arg("a"), py::arg("x"));
    m.def("gammainc", [](const py::object& a, const py::object& x) {
        return binary(a, x, [](double u, double v) { return sm::gammainc(u, v); });
    }, py::arg("a"), py::arg("x"));
    m.def("gammainccinv", [](const py::object& a, const py::object& y) {
        return binary(a, y, [](double u, double v) { return sm::gammainccinv(u, v); });
    }, py::arg("a"), py::arg("y"));
    m.def("gammaincinv", [](const py::object& a, const py::object& y) {
        return binary(a, y, [](double u, double v) { return sm::gammaincinv(u, v); });
    }, py::arg("a"), py::arg("y"));
    m.def("gammaincc_many", [](double a, const py::object& x) {
        const std::vector<double> xv = rx::vec<double>(x);
        return to_numpy(sm::gammaincc(a, std::span<const double>(xv)));
    }, py::arg("a"), py::arg("x"));
    m.def("lgam", [](const py::object& x) { return unary(x, [](double v) { return sm::lgam(v); }); }, py::arg("x"));
    m.def("gamma", [](const py::object& x) { return unary(x, [](double v) { return sm::gamma(v); }); }, py::arg("x"));
    m.def("erfc", [](const py::object& x) { return unary(x, [](double v) { return sm::erfc(v); }); }, py::arg("x"));
    m.def("zeta", [](const py::object& x, const py::object& qq) {
        return binary(x, qq, [](double u, double v) { return sm::zeta(u, v); });
    }, py::arg("x"), py::arg("q"));
}

void bind_ndfilter(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("ndfilter", "scipy.ndimage.median_filter, 1-D reflect (native/core/math/ndfilter)");
    m.def("median_filter", [](const py::object& x, int size, const std::string& mode) -> py::object {
        if (mode != "reflect")
            throw py::value_error("median_filter: only mode 'reflect' is ported, not '" + mode + "'");
        py::array a = py::array::ensure(rx::np_module().attr("asarray")(x));
        if (!a) throw py::error_already_set();
        if (a.ndim() != 1)
            throw py::value_error("median_filter: only a 1-D array is ported, not " + std::to_string(a.ndim()) + "-D");
        if (py::str(a.attr("dtype")).cast<std::string>() == "float32") {
            const std::vector<float> v = rx::vec<float>(a);
            return to_numpy(sm::median_filter(std::span<const float>(v), size));
        }
        const std::vector<double> v = rx::vec<double>(a);
        return to_numpy(sm::median_filter(std::span<const double>(v), size));
    }, py::arg("x"), py::arg("size"), py::arg("mode") = "reflect");
}

const Registrar reg_special("special", 6, &bind_special);
const Registrar reg_ndfilter("ndfilter", 6, &bind_ndfilter);

}  // namespace
}  // namespace sstvae_bindings::qrss
