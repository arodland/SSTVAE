// sstvae_native.qrss.npfft: native/core/math/npfft (numpy.fft's fft,
// ifft, rfft, irfft, fftfreq, rfftfreq), for the parity tests -- the
// receiver's modules call these from C++.
//
//   fft(a, n=None, axis=-1), ifft(a, n=None, axis=-1)   1-D, or 2-D
//       along the last axis (fft only); complex128 out, complex64 for
//       complex64/float32 input (numpy 2.x computes those in single)
//   rfft(a, n=None), irfft(a, n=None)   1-D, precision likewise
//   fftfreq(n, d=1.0), rfftfreq(n, d=1.0)
//
// Only the default norm is ported; anything else raises
// NotImplementedError.

#include "qrss/rx_types.hpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "math/npfft.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace sm = sstvae::math;

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

py::array ensure(const py::object& a) {
    py::array arr = py::array::ensure(rx::np_module().attr("asarray")(a));
    if (!arr) throw py::error_already_set();
    return arr;
}

// numpy's precision choice: result_type(a, 1j) for the complex transforms,
// result_type(a.real, 1.0) for irfft's output -- complex64/float32 (and
// float16) run single precision, everything else double; longdouble is
// not ported.
bool single(const py::array& arr) {
    const std::string dt = py::str(rx::np_module().attr("result_type")(arr.attr("dtype"), std::complex<double>(0, 1)))
                               .cast<std::string>();
    if (dt == "complex64") return true;
    if (dt == "complex128") return false;
    not_implemented("only single and double precision are ported, not " + dt);
}

template <typename T>
py::object c2c_t(const py::array& arr, const py::object& n, int axis, bool forward) {
    using C = std::complex<T>;
    if (arr.ndim() == 1) {
        if (axis != -1 && axis != 0) throw py::value_error("axis out of range");
        const std::vector<C> x = rx::vec<C>(arr);
        return to_numpy(forward ? sm::fft(std::span<const C>(x), opt_n(n)) : sm::ifft(std::span<const C>(x), opt_n(n)));
    }
    if constexpr (std::is_same_v<T, double>) {
        if (arr.ndim() == 2 && (axis == -1 || axis == 1) && forward) {
            // complex128: copy each row into the result, cropped or zero-padded,
            // and transform it there -- one copy, where fft_rows' vector API
            // costs three (the batch was 3x numpy's time for 150 x 16000)
            using In = py::array_t<C, py::array::c_style | py::array::forcecast>;
            const In in = In::ensure(arr);
            if (!in) throw py::error_already_set();
            const auto rows = static_cast<std::size_t>(in.shape(0)), cols = static_cast<std::size_t>(in.shape(1));
            const std::int64_t want = n.is_none() ? static_cast<std::int64_t>(cols) : rx::as_int(n);
            if (want < 1)
                throw py::value_error("Invalid number of FFT data points (" + std::to_string(want) + ") specified.");
            const auto npts = static_cast<std::size_t>(want);
            py::array_t<C> res(std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(npts)});
            C* d = res.mutable_data();
            const C* src = in.data();
            const std::size_t used = std::min(npts, cols);
            for (std::size_t r = 0; r < rows; ++r) {
                C* row = d + r * npts;
                std::copy_n(src + r * cols, used, row);
                std::fill(row + used, row + npts, C(0, 0));
                sm::fft_inplace(std::span<C>(row, npts));
            }
            return res;
        }
    }
    if (arr.ndim() == 2 && (axis == -1 || axis == 1) && forward) {
        const std::vector<C> x = rx::vec<C>(arr);
        const auto rows = static_cast<std::size_t>(arr.shape(0)), cols = static_cast<std::size_t>(arr.shape(1));
        const std::vector<C> out = sm::fft_rows(std::span<const C>(x), rows, cols, opt_n(n));
        const std::size_t npts = rows ? out.size() / rows : (n.is_none() ? cols : static_cast<std::size_t>(rx::as_int(n)));
        py::array_t<C> res(std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(npts)});
        std::copy(out.begin(), out.end(), res.mutable_data());
        return res;
    }
    not_implemented("only 1-D, or fft along the last axis of 2-D, is ported");
}

py::object c2c(const py::object& a, const py::object& n, int axis, const py::object& norm, bool forward) {
    check_norm(norm);
    py::array arr = ensure(a);
    return single(arr) ? c2c_t<float>(arr, n, axis, forward) : c2c_t<double>(arr, n, axis, forward);
}

template <typename T>
py::object rfft_t(const py::array& arr, const py::object& n) {
    const std::vector<T> x = rx::vec<T>(arr);
    return to_numpy(sm::rfft(std::span<const T>(x), opt_n(n)));
}

template <typename T>
py::object irfft_t(const py::array& arr, const py::object& n) {
    const std::vector<std::complex<T>> x = rx::vec<std::complex<T>>(arr);
    return to_numpy(sm::irfft(std::span<const std::complex<T>>(x), opt_n(n)));
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
        py::array arr = ensure(a);
        if (arr.ndim() != 1) not_implemented("only 1-D rfft is ported");
        if (rx::np_module().attr("iscomplexobj")(arr).cast<bool>()) not_implemented("rfft of complex input");
        return single(arr) ? rfft_t<float>(arr, n) : rfft_t<double>(arr, n);
    }, py::arg("a"), py::arg("n") = py::none(), py::arg("norm") = py::none());
    m.def("irfft", [](const py::object& a, const py::object& n, const py::object& norm) {
        check_norm(norm);
        py::array arr = ensure(a);
        if (arr.ndim() != 1) not_implemented("only 1-D irfft is ported");
        return single(arr) ? irfft_t<float>(arr, n) : irfft_t<double>(arr, n);
    }, py::arg("a"), py::arg("n") = py::none(), py::arg("norm") = py::none());
    m.def("fftfreq", [](std::int64_t n, double d) { return to_numpy(sm::fftfreq(n, d)); }, py::arg("n"),
          py::arg("d") = 1.0);
    m.def("rfftfreq", [](std::int64_t n, double d) { return to_numpy(sm::rfftfreq(n, d)); }, py::arg("n"),
          py::arg("d") = 1.0);
}

const Registrar reg("npfft", 6, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
