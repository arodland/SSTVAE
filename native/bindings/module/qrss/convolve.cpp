// sstvae_native.qrss.convolve: bindings of native/core/math/convolve
// (scipy 1.18's oaconvolve, fftconvolve, freqz with an array worN and
// ndimage.maximum_filter1d), for the parity tests. The receiver's modules
// call these from C++; nothing in the reference is substituted by them.
//
//   oaconvolve(in1, in2, mode="full"), fftconvolve(in1, in2, mode="full")
//                     1-D, 'full' or 'same'; complex128 when either operand
//                     is complex, else float64 (scipy's result dtypes)
//   freqz(b, worN, fs=2*pi) -> (w, h)   worN an array of frequencies
//   maximum_filter1d(input, size)       bool or uint8, mode 'reflect'
//   next_fast_len(target, real=False), calc_oa_lens(s1, s2) ->
//       (block_size, overlap or None, in1_step, in2_step), lambertw_m1(x)
//
// What scipy accepts and this does not (2-D operands, mode 'valid', axes,
// a scalar worN, other filter modes) raises NotImplementedError naming it.

#include "qrss/rx_types.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "math/convolve.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace sm = sstvae::math;
using cd = std::complex<double>;

[[noreturn]] void not_implemented(const std::string& what) {
    PyErr_SetString(PyExc_NotImplementedError, what.c_str());
    throw py::error_already_set();
}

sm::ConvMode conv_mode(const std::string& mode) {
    if (mode == "full") return sm::ConvMode::Full;
    if (mode == "same") return sm::ConvMode::Same;
    if (mode == "valid") not_implemented("convolve: mode 'valid' is not ported");
    throw py::value_error("acceptable mode flags are 'valid', 'same', or 'full'");
}

py::array one_d(const py::handle& o) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o));
    if (!a) throw py::error_already_set();
    if (a.ndim() != 1) not_implemented("convolve: only 1-D operands are ported");
    return a;
}

bool is_complex(const py::array& a) { return py::str(a.dtype().attr("kind")).cast<std::string>() == "c"; }

template <bool OA>
py::object convolve(const py::object& in1, const py::object& in2, const std::string& mode_name) {
    const sm::ConvMode mode = conv_mode(mode_name);
    const py::array a = one_d(in1), b = one_d(in2);
    const bool ca = is_complex(a), cb = is_complex(b);
    auto run = [&](auto x, auto y) {
        py::gil_scoped_release nogil;
        if constexpr (OA)
            return sm::oaconvolve(std::span(x), std::span(y), mode);
        else
            return sm::fftconvolve(std::span(x), std::span(y), mode);
    };
    if (!ca && !cb) {
        const auto x = rx::vec<double>(a), y = rx::vec<double>(b);
        return to_numpy(run(std::span<const double>(x), std::span<const double>(y)));
    }
    if (ca && cb) {
        const auto x = rx::vec<cd>(a), y = rx::vec<cd>(b);
        return to_numpy(run(std::span<const cd>(x), std::span<const cd>(y)));
    }
    if (ca) {
        const auto x = rx::vec<cd>(a);
        const auto y = rx::vec<double>(b);
        return to_numpy(run(std::span<const cd>(x), std::span<const double>(y)));
    }
    const auto x = rx::vec<double>(a);
    const auto y = rx::vec<cd>(b);
    return to_numpy(run(std::span<const double>(x), std::span<const cd>(y)));
}

void bind(py::module_& qrss) {
    rx::register_translators();
    py::module_ m = qrss.def_submodule("convolve", "native/core/math/convolve (scipy's FFT convolutions)");

    m.def("oaconvolve", &convolve<true>, py::arg("in1"), py::arg("in2"), py::arg("mode") = "full");
    m.def("fftconvolve", &convolve<false>, py::arg("in1"), py::arg("in2"), py::arg("mode") = "full");

    m.def("freqz", [](const py::object& b, const py::object& worN, double fs) {
        py::array w = py::array::ensure(rx::np_module().attr("asarray")(worN, py::arg("dtype") = "float64"));
        if (!w || w.ndim() != 1) not_implemented("freqz: only a 1-D array worN is ported");
        const auto bv = rx::vec<double>(b);
        const auto f = rx::vec<double>(w);
        std::vector<cd> h;
        {
            py::gil_scoped_release nogil;
            h = sm::freqz(bv, f, fs);
        }
        // scipy hands back its own w: 2 pi f / fs, then times fs / (2 pi)
        std::vector<double> wr(f.size());
        for (std::size_t i = 0; i < f.size(); ++i)
            wr[i] = (2 * std::numbers::pi * f[i] / fs) * (fs / (2 * std::numbers::pi));
        return py::make_tuple(to_numpy(wr), to_numpy(h));
    }, py::arg("b"), py::arg("worN"), py::arg("fs") = 2 * std::numbers::pi);

    m.def("maximum_filter1d", [](const py::object& input, std::int64_t size) {
        py::array a = py::array::ensure(rx::np_module().attr("asarray")(input));
        if (!a) throw py::error_already_set();
        if (a.ndim() != 1) not_implemented("maximum_filter1d: only 1-D input is ported");
        const std::string dt = py::str(a.dtype()).cast<std::string>();
        if (dt != "bool" && dt != "uint8") not_implemented("maximum_filter1d: only bool and uint8 input is ported");
        const auto x = from_numpy<std::uint8_t>(a.attr("view")("uint8"));
        const auto y = sm::maximum_filter1d(x, size);
        py::object out = to_numpy(y);
        return dt == "bool" ? out.attr("view")("bool") : out;
    }, py::arg("input"), py::arg("size"));

    m.def("next_fast_len", &sm::next_fast_len, py::arg("target"), py::arg("real") = false);
    m.def("calc_oa_lens", [](std::int64_t s1, std::int64_t s2) {
        const sm::OaLens l = sm::calc_oa_lens(s1, s2);
        return py::make_tuple(l.block_size, l.overlap < 0 ? py::object(py::none()) : py::object(py::int_(l.overlap)),
                              l.in1_step, l.in2_step);
    }, py::arg("s1"), py::arg("s2"));
    m.def("lambertw_m1", &sm::lambertw_m1, py::arg("x"));
}

const Registrar reg("convolve", 6, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
