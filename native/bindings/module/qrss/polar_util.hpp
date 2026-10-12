// Argument conversions shared by the polar and header bindings
// (bindings/module/qrss/{polar,header}.cpp): numpy's own semantics for the
// checks the reference makes on its inputs, so a refusal (or a None) comes
// back for exactly the inputs Python refuses.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "qrss/registry.hpp"

namespace sstvae_bindings::qrss::polar_util {

inline py::module_ numpy() { return py::module_::import("numpy"); }

// operator.index(obj): Python ints, numpy integers and bools; TypeError
// otherwise, with Python's own message.
inline py::int_ index(const py::handle& obj) {
    PyObject* r = PyNumber_Index(obj.ptr());
    if (!r) throw py::error_already_set();
    return py::reinterpret_steal<py::int_>(r);
}

// A Python int clamped into int64. Every QRSS integer check is a range
// check, so a value beyond int64 fails it exactly as the original would.
inline std::int64_t clamp_i64(const py::int_& v) {
    const py::int_ lo(std::numeric_limits<std::int64_t>::min());
    const py::int_ hi(std::numeric_limits<std::int64_t>::max());
    if (v < lo) return std::numeric_limits<std::int64_t>::min();
    if (hi < v) return std::numeric_limits<std::int64_t>::max();
    return v.cast<std::int64_t>();
}

inline bool truthy(const py::handle& obj) {
    const int t = PyObject_IsTrue(obj.ptr());
    if (t < 0) throw py::error_already_set();
    return t != 0;
}

inline std::string shape_str(const py::array& a) { return py::str(a.attr("shape")).cast<std::string>(); }

// The values of `a` as 0/1 bytes (C order), or nullopt when
// `np.any((a != 0) & (a != 1))` -- the reference's test, numpy's semantics
// (so 1.0 and True are 1, 0.5 and NaN are refused).
inline std::optional<std::vector<std::uint8_t>> bits01(const py::array& a) {
    const char kind = a.dtype().kind();
    if (kind == 'b' || kind == 'i' || kind == 'u') {
        auto v = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(a);
        if (!v) throw py::error_already_set();
        std::vector<std::uint8_t> out(static_cast<std::size_t>(v.size()));
        const std::int64_t* p = v.data();
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (p[i] != 0 && p[i] != 1) return std::nullopt;
            out[i] = static_cast<std::uint8_t>(p[i]);
        }
        return out;
    }
    py::module_ op = py::module_::import("operator");
    py::object bad = numpy().attr("any")(op.attr("and_")(op.attr("ne")(a, 0), op.attr("ne")(a, 1)));
    if (truthy(bad)) return std::nullopt;
    py::array u = a.attr("astype")(numpy().attr("uint8"));
    auto v = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>::ensure(u);
    return std::vector<std::uint8_t>(v.data(), v.data() + v.size());
}

// np.asarray(obj, dtype=np.float64) as a vector, refused unless it is
// shape (n,) with the reference's message "<what>, got shape <shape>".
inline std::vector<double> float_vector(const py::object& obj, py::ssize_t n, const std::string& what) {
    py::array a = numpy().attr("asarray")(obj, py::arg("dtype") = numpy().attr("float64"));
    if (a.ndim() != 1 || a.shape(0) != n) throw py::value_error(what + ", got shape " + shape_str(a));
    return from_numpy<double>(a);
}

// A 2-D uint8 array (rows x cols) holding `v`.
inline py::array_t<std::uint8_t> to_numpy_2d(const std::vector<std::uint8_t>& v, py::ssize_t rows,
                                             py::ssize_t cols) {
    py::array_t<std::uint8_t> out({rows, cols});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

}  // namespace sstvae_bindings::qrss::polar_util
