// sstvae_native.qrss.npfft: native/core/math/npfft (numpy.fft's fft,
// ifft, rfft, irfft, fftfreq, rfftfreq), for the parity tests -- the
// receiver's modules call these from C++.
//
//   fft(a, n=None, axis=-1), ifft(a, n=None, axis=-1)   1-D, or 2-D
//       along the last axis (fft only); complex128 out
//   rfft(a, n=None), irfft(a, n=None)   1-D
//   fftfreq(n, d=1.0), rfftfreq(n, d=1.0)
//
// Only the default norm is ported; anything else raises
// NotImplementedError.

#include "qrss/rx_types.hpp"

#include <complex>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "math/npfft.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace sm = sstvae::math;
using cd = std::complex<double>;

[[noreturn]] void not_implemented(const std::string& what) {
    PyErr_SetString(PyExc_NotImplementedError, ("native npfft: " + what).c_str());
    throw py::error_already_set();
}

std::optional<std::int64_t> opt_n(const py::object& n) {
    if (n.is_none()) return std::nullopt;
    return rx::as_int(n);
}

void check_norm(const py::object& norm) {
    if (!norm.is_none() && norm.cast<std::string>() != "backward") not_implemented("only norm='backward' is ported");
}

py::array ensure(const py::object& a, const char* dtype) {
    py::array arr = py::array::ensure(rx::np_module().attr("asarray")(a, py::arg("dtype") = dtype));
    if (!arr) throw py::error_already_set();
    return arr;
}

py::object c2c(const py::object& a, const py::object& n, int axis, const py::object& norm, bool forward) {
    check_norm(norm);
    py::array arr = ensure(a, "complex128");
    if (arr.ndim() == 1) {
        if (axis != -1 && axis != 0) throw py::value_error("axis out of range");
        const std::vector<cd> x = rx::vec<cd>(arr);
        return to_numpy(forward ? sm::fft(std::span<const cd>(x), opt_n(n)) : sm::ifft(std::span<const cd>(x), opt_n(n)));
    }
    if (arr.ndim() == 2 && (axis == -1 || axis == 1) && forward) {
        const std::vector<cd> x = rx::vec<cd>(arr);
        const auto rows = static_cast<std::size_t>(arr.shape(0)), cols = static_cast<std::size_t>(arr.shape(1));
        const std::vector<cd> out = sm::fft_rows(x, rows, cols, opt_n(n));
        const std::size_t npts = rows ? out.size() / rows : (n.is_none() ? cols : static_cast<std::size_t>(rx::as_int(n)));
        py::array_t<cd> res(std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(npts)});
        std::copy(out.begin(), out.end(), res.mutable_data());
        return res;
    }
    not_implemented("only 1-D, or fft along the last axis of 2-D, is ported");
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("npfft", "numpy.fft through pocketfft (native/core/math/npfft)");
    m.def("fft", [](const py::object& a, const py::object& n, int axis, const py::object& norm) {
        return c2c(a, n, axis, norm, true);
    }, py::arg("a"), py::arg("n") = py::none(), py::arg("axis") = -1, py::arg("norm") = py::none());
    m.def("ifft", [](const py::object& a, const py::object& n, int axis, const py::object& norm) {
        return c2c(a, n, axis, norm, false);
    }, py::arg("a"), py::arg("n") = py::none(), py::arg("axis") = -1, py::arg("norm") = py::none());
    m.def("rfft", [](const py::object& a, const py::object& n, const py::object& norm) {
        check_norm(norm);
        py::array arr = ensure(a, "float64");
        if (arr.ndim() != 1) not_implemented("only 1-D rfft is ported");
        const std::vector<double> x = rx::vec<double>(arr);
        return to_numpy(sm::rfft(std::span<const double>(x), opt_n(n)));
    }, py::arg("a"), py::arg("n") = py::none(), py::arg("norm") = py::none());
    m.def("irfft", [](const py::object& a, const py::object& n, const py::object& norm) {
        check_norm(norm);
        py::array arr = ensure(a, "complex128");
        if (arr.ndim() != 1) not_implemented("only 1-D irfft is ported");
        const std::vector<cd> x = rx::vec<cd>(arr);
        return to_numpy(sm::irfft(std::span<const cd>(x), opt_n(n)));
    }, py::arg("a"), py::arg("n") = py::none(), py::arg("norm") = py::none());
    m.def("fftfreq", [](std::int64_t n, double d) { return to_numpy(sm::fftfreq(n, d)); }, py::arg("n"),
          py::arg("d") = 1.0);
    m.def("rfftfreq", [](std::int64_t n, double d) { return to_numpy(sm::rfftfreq(n, d)); }, py::arg("n"),
          py::arg("d") = 1.0);
}

const Registrar reg("npfft", 6, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
