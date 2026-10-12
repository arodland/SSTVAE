// sstvae_native.qrss.infra: the port's numeric and file infrastructure
// (native/core/qrss/{half,npsum,crc32,rational,npz}), bound for the parity
// tests in tests/test_qrss_native_infra.py. None of it replaces a
// sstvae.qrss attribute -- these are the numpy, zlib and fractions
// behaviours the ported modules rely on -- so there are no substitution
// rows; the tests compare each against the Python library it mirrors.

#include "qrss/registry.hpp"

#include <complex>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/crc32.hpp"
#include "qrss/half.hpp"
#include "qrss/npsum.hpp"
#include "qrss/npz.hpp"
#include "qrss/rational.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace half = sstvae::qrss::half;
namespace npsum = sstvae::qrss::npsum;
namespace npz = sstvae::qrss::npz;
using sstvae::qrss::Rational;

std::string dtype_str(const py::array& a) { return py::str(a.attr("dtype").attr("str")).cast<std::string>(); }

std::vector<py::ssize_t> shape_of(const py::array& a) {
    return std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim());
}

// A C-contiguous T view of `obj` (copied only if it is not one already),
// requiring its dtype to be exactly T's.
template <typename T>
py::array_t<T, py::array::c_style> exact(const py::array& a, const char* fn) {
    if (!a.dtype().is(py::dtype::of<T>()))
        throw py::type_error(std::string(fn) + ": unexpected dtype " + dtype_str(a));
    return py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(a);
}

template <typename T>
py::array zero_d(T v) {
    py::array_t<T> out(std::vector<py::ssize_t>{});
    *out.mutable_data() = v;
    return out;
}

template <typename In, typename Out, typename F>
py::array elementwise(const py::array& a, F f) {
    auto in = py::array_t<In, py::array::c_style | py::array::forcecast>::ensure(a);
    py::array_t<Out> out(shape_of(in));
    const In* src = in.data();
    Out* dst = out.mutable_data();
    for (py::ssize_t i = 0; i < in.size(); ++i) dst[i] = f(src[i]);
    return out;
}

std::int64_t to_i64(const py::handle& v) {
    const py::int_ i = py::reinterpret_borrow<py::object>(v).attr("__index__")();
    const py::int_ hi(std::numeric_limits<std::int64_t>::max());
    if (i > hi || i < -hi) throw std::overflow_error("Rational holds int64 numerators and denominators");
    return i.cast<std::int64_t>();
}

py::object fraction_of(const Rational& r) {
    return py::module_::import("fractions").attr("Fraction")(r.num(), r.den());
}

// --- npz <-> numpy ---

npz::Array to_array(const py::handle& obj) {
    py::array a = py::module_::import("numpy").attr("asarray")(obj);
    const std::string d = dtype_str(a);
    const auto t = npz::from_descr(d);
    if (!t) throw py::value_error("npz: dtype '" + d + "' is not supported");
    npz::Array out;
    out.dtype = *t;
    for (py::ssize_t k = 0; k < a.ndim(); ++k) out.shape.push_back(static_cast<std::size_t>(a.shape(k)));
    const std::string raw = a.attr("tobytes")().cast<std::string>();
    out.bytes.assign(raw.begin(), raw.end());
    return out;
}

py::array to_numpy(const npz::Array& a) {
    std::vector<py::ssize_t> shape(a.shape.begin(), a.shape.end());
    return py::array(py::dtype(std::string(npz::descr(a.dtype))), shape, a.bytes.data());
}

npz::Npz to_npz(const py::dict& arrays) {
    npz::Npz z;
    for (const auto& [k, v] : arrays) z.add(py::str(k).cast<std::string>(), to_array(v));
    return z;
}

py::dict to_dict(const npz::Npz& z) {
    py::dict out;
    for (const auto& [k, a] : z.members()) out[py::str(k)] = to_numpy(a);
    return out;
}

std::span<const std::uint8_t> as_span(const std::string& s) {
    return {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
}

py::bytes as_bytes(const std::vector<std::uint8_t>& v) {
    return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("infra", "Port of numpy/zlib/fractions behaviour (native/core/qrss)");

    py::register_local_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const sstvae::qrss::ZeroDivision& e) {
            PyErr_SetString(PyExc_ZeroDivisionError, e.what());
        }
    });

    // --- half ---
    m.def("half_from_double", [](const py::array& x) {
        return elementwise<double, std::uint16_t>(x, [](double v) { return half::from_double(v); });
    }, py::arg("x"), "x.astype(np.float16).view(np.uint16), from float64");
    m.def("half_from_float", [](const py::array& x) {
        return elementwise<float, std::uint16_t>(x, [](float v) { return half::from_float(v); });
    }, py::arg("x"), "x.astype(np.float16).view(np.uint16), from float32");
    m.def("half_to_double", [](const py::array& h) {
        return elementwise<std::uint16_t, double>(h, [](std::uint16_t v) { return half::to_double(v); });
    }, py::arg("bits"));
    m.def("half_to_float", [](const py::array& h) {
        return elementwise<std::uint16_t, float>(h, [](std::uint16_t v) { return half::to_float(v); });
    }, py::arg("bits"));

    // --- npsum ---
    m.def("np_sum", [](const py::array& x) -> py::array {
        const std::string d = dtype_str(x);
        if (d == "<f8") {
            auto a = exact<double>(x, "np_sum");
            return zero_d(npsum::sum(std::span<const double>(a.data(), a.size())));
        }
        if (d == "<f4") {
            auto a = exact<float>(x, "np_sum");
            return zero_d(npsum::sum(std::span<const float>(a.data(), a.size())));
        }
        if (d == "<c16") {
            auto a = exact<std::complex<double>>(x, "np_sum");
            return zero_d(npsum::sum(std::span<const std::complex<double>>(a.data(), a.size())));
        }
        if (d == "<c8") {
            auto a = exact<std::complex<float>>(x, "np_sum");
            return zero_d(npsum::sum(std::span<const std::complex<float>>(a.data(), a.size())));
        }
        throw py::type_error("np_sum: float64, float32, complex128 or complex64, not " + d);
    }, py::arg("x"), "np.sum(x) of a 1-D array, same dtype");

    m.def("np_mean", [](const py::array& x) -> py::array {
        const std::string d = dtype_str(x);
        if (d == "<f8") {
            auto a = exact<double>(x, "np_mean");
            return zero_d(npsum::mean(std::span<const double>(a.data(), a.size())));
        }
        if (d == "<f4") {
            auto a = exact<float>(x, "np_mean");
            return zero_d(npsum::mean(std::span<const float>(a.data(), a.size())));
        }
        if (d == "|b1") {
            auto a = exact<bool>(x, "np_mean");
            return zero_d(npsum::mean_as_double(std::span<const bool>(a.data(), a.size())));
        }
        if (d == "|u1") {
            auto a = exact<std::uint8_t>(x, "np_mean");
            return zero_d(npsum::mean_as_double(std::span<const std::uint8_t>(a.data(), a.size())));
        }
        if (d == "<i8") {
            auto a = exact<std::int64_t>(x, "np_mean");
            return zero_d(npsum::mean_as_double(std::span<const std::int64_t>(a.data(), a.size())));
        }
        throw py::type_error("np_mean: float64, float32, bool, uint8 or int64, not " + d);
    }, py::arg("x"), "np.mean(x) of a 1-D array");

    m.def("np_sum_as", [](const py::array& x, const std::string& dtype) -> py::array {
        const std::string d = dtype_str(x);
        auto run = [&](auto acc) -> py::array {
            using Acc = decltype(acc);
            if (d == "<f8") {
                auto a = exact<double>(x, "np_sum_as");
                return zero_d(npsum::sum_as<Acc>(std::span<const double>(a.data(), a.size())));
            }
            if (d == "<f4") {
                auto a = exact<float>(x, "np_sum_as");
                return zero_d(npsum::sum_as<Acc>(std::span<const float>(a.data(), a.size())));
            }
            if (d == "|b1") {
                auto a = exact<bool>(x, "np_sum_as");
                return zero_d(npsum::sum_as<Acc>(std::span<const bool>(a.data(), a.size())));
            }
            if (d == "<i8") {
                auto a = exact<std::int64_t>(x, "np_sum_as");
                return zero_d(npsum::sum_as<Acc>(std::span<const std::int64_t>(a.data(), a.size())));
            }
            throw py::type_error("np_sum_as: float64, float32, bool or int64 input, not " + d);
        };
        if (dtype == "float64") return run(double{});
        if (dtype == "float32") return run(float{});
        throw py::value_error("np_sum_as: dtype must be 'float64' or 'float32'");
    }, py::arg("x"), py::arg("dtype"), "np.sum(x, dtype=dtype) where x's dtype differs (buffered cast)");

    m.def("np_mean_as_double", [](const py::array& x) -> py::array {
        auto a = exact<float>(x, "np_mean_as_double");
        return zero_d(npsum::mean_as_double(std::span<const float>(a.data(), a.size())));
    }, py::arg("x"), "np.mean(x32, dtype=np.float64)");

    // --- crc32 ---
    m.def("crc32", [](const py::bytes& data, const py::int_& value) {
        const std::string s = data;
        // zlib.crc32 takes `value` as an unsigned int, bitwise (masked).
        const auto v = static_cast<std::uint32_t>(
            py::int_(value.attr("__and__")(py::int_(0xFFFFFFFFu))).cast<std::uint64_t>());
        return sstvae::qrss::crc32(std::string_view(s), v);
    }, py::arg("data"), py::arg("value") = 0);

    // --- Rational ---
    py::class_<Rational>(m, "Rational")
        .def(py::init([](const py::object& num, const py::object& den) { return Rational(to_i64(num), to_i64(den)); }),
             py::arg("numerator") = 0, py::arg("denominator") = 1)
        .def_static("from_float", &Rational::from_double, py::arg("x"))
        // Fraction(x).limit_denominator(max_denominator), x a float.
        .def_static("from_float_limited", [](double x, const py::object& max_den) {
            const py::int_ i = max_den.attr("__index__")();
            if (i < py::int_(1)) return Rational::from_double_limited(x, 0);   // NaN/inf first, as Python
            if (i > py::int_(std::numeric_limits<std::int64_t>::max())) {
                (void)Rational::from_double_limited(x, 1);                      // the same order
                throw std::overflow_error("Rational: max_denominator out of int64 range");
            }
            return Rational::from_double_limited(x, i.cast<std::int64_t>());
        }, py::arg("x"), py::arg("max_denominator") = 1000000)
        .def_static("parse", [](const std::string& s) { return Rational::parse(s); }, py::arg("text"))
        .def_property_readonly("numerator", &Rational::num)
        .def_property_readonly("denominator", &Rational::den)
        .def("limit_denominator", [](const Rational& r, const py::object& max_den) {
            const py::int_ i = max_den.attr("__index__")();
            if (i < py::int_(1)) throw py::value_error("max_denominator should be at least 1");
            const py::int_ cap(std::numeric_limits<std::int64_t>::max());
            return r.limit_denominator(i > cap ? std::numeric_limits<std::int64_t>::max() : i.cast<std::int64_t>());
        }, py::arg("max_denominator") = 1000000)
        .def("__str__", &Rational::str)
        .def("__repr__", [](const Rational& r) {
            return "Rational(" + std::to_string(r.num()) + ", " + std::to_string(r.den()) + ")";
        })
        .def("__float__", &Rational::to_double)
        .def("__int__", &Rational::trunc)
        .def("__floor__", &Rational::floor)
        .def("__ceil__", &Rational::ceil)
        .def("__round__", [](const Rational& r) { return r.round(); })
        .def("as_fraction", &fraction_of)
        .def("__neg__", [](const Rational& a) { return -a; })
        .def("__add__", [](const Rational& a, const Rational& b) { return a + b; })
        .def("__sub__", [](const Rational& a, const Rational& b) { return a - b; })
        .def("__mul__", [](const Rational& a, const Rational& b) { return a * b; })
        .def("__truediv__", [](const Rational& a, const Rational& b) { return a / b; })
        .def("__eq__", [](const Rational& a, const Rational& b) { return a == b; })
        .def("__lt__", [](const Rational& a, const Rational& b) { return a < b; })
        .def("__le__", [](const Rational& a, const Rational& b) { return a <= b; })
        .def("__gt__", [](const Rational& a, const Rational& b) { return a > b; })
        .def("__ge__", [](const Rational& a, const Rational& b) { return a >= b; })
        .def("__hash__", [](const Rational& r) { return py::hash(fraction_of(r)); });

    // --- npy / npz ---
    m.def("npy_bytes", [](const py::object& a) { return as_bytes(npz::npy_bytes(to_array(a))); }, py::arg("array"),
          "np.save(buf, array); buf.getvalue()");
    m.def("parse_npy", [](const py::bytes& b) {
        const std::string s = b;
        return to_numpy(npz::parse_npy(as_span(s)));
    }, py::arg("data"));
    m.def("npz_bytes", [](const py::dict& arrays, unsigned dos_time, unsigned dos_date) {
        if (dos_time > 0xffff || dos_date > 0xffff) throw py::value_error("DOS time and date are 16-bit");
        return as_bytes(npz::npz_bytes(to_npz(arrays), npz::DosTime{static_cast<std::uint16_t>(dos_time),
                                                                    static_cast<std::uint16_t>(dos_date)}));
    }, py::arg("arrays"), py::arg("dos_time") = 0, py::arg("dos_date") = 0x21,
          "np.savez(buf, **arrays); buf.getvalue(), stamped with the given DOS time");
    m.def("parse_npz", [](const py::bytes& b) {
        const std::string s = b;
        return to_dict(npz::parse_npz(as_span(s)));
    }, py::arg("data"), "dict(np.load(io.BytesIO(data)))");
    m.def("save_npy", [](const std::string& path, const py::object& a) { npz::save_npy(path, to_array(a)); },
          py::arg("path"), py::arg("array"));
    m.def("load_npy", [](const std::string& path) { return to_numpy(npz::load_npy(path)); }, py::arg("path"));
    m.def("save_npz", [](const std::string& path, const py::dict& arrays) { npz::save_npz(path, to_npz(arrays)); },
          py::arg("path"), py::arg("arrays"));
    m.def("load_npz", [](const std::string& path) { return to_dict(npz::load_npz(path)); }, py::arg("path"));
    m.def("npz_to_double", [](const py::object& a) {
        const auto v = to_array(a).to_double();
        return py::array_t<double>(static_cast<py::ssize_t>(v.size()), v.data());
    }, py::arg("array"), "a.astype(np.float64).ravel() via npz::Array::to_double");
    m.def("npz_to_int64", [](const py::object& a) { return to_array(a).to_int64(); }, py::arg("array"),
          "int(a) via npz::Array::to_int64");
}

const Registrar reg("infra", 0, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
