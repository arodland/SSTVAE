// Argument conversions shared by the polar and header bindings
// (bindings/module/qrss/{polar,header}.cpp): numpy's own semantics for the
// checks the reference makes on its inputs, so a refusal (or a None) comes
// back for exactly the inputs Python refuses.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "qrss/polar.hpp"
#include "qrss/registry.hpp"

namespace sstvae_bindings::qrss::polar_util {

inline py::module_ numpy() { return py::module_::import("numpy"); }

inline py::object builtin(const char* name) { return py::module_::import("builtins").attr(name); }

// A new reference from the C API, or the pending Python error.
inline py::object steal(PyObject* r) {
    if (!r) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(r);
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

// `bool(a <op> b)`: Python's own comparison (reflection included) and truthiness.
inline bool compare(const py::handle& a, const py::handle& b, int op) {
    return truthy(steal(PyObject_RichCompare(a.ptr(), b.ptr(), op)));
}

// f"{obj}": what the reference's refusal texts print.
inline std::string fmt(const py::handle& obj) { return builtin("format")(obj).cast<std::string>(); }

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

// polar.decode_list's argument handling, in the reference's order:
//
//     if list_size < 1: ValueError          Python's own `<` and truthiness
//     mother = fold(llr)                    np.asarray(llr, float64), its shape
//     np.all(np.isfinite(mother))           "LLRs must be finite"
//     _List(list_size)                      np.full(list_size, np.inf); pm[0] = 0.0
//     np.broadcast_to(mother, (list_size, N))
//
// so a float or a bool list size is refused by numpy's own shape
// conversion, after the LLRs (as there), and the list size the recursion
// runs with is the broadcast's first dimension, as `_scl` reads it.
struct DecodeArgs {
    std::vector<double> llr;
    std::size_t list_size = 0;
};

inline DecodeArgs decode_args(const py::object& llr, const py::object& list_size) {
    namespace pol = sstvae::qrss::polar;
    if (compare(list_size, py::int_(1), Py_LT)) throw py::value_error("list_size must be >= 1");
    DecodeArgs out;
    out.llr = float_vector(llr, pol::E, "expected " + std::to_string(pol::E) + " coded LLRs");
    for (double v : pol::fold(out.llr))
        if (!std::isfinite(v)) throw py::value_error("LLRs must be finite");
    py::module_ np = numpy();
    py::object pm = np.attr("full")(list_size, np.attr("inf"));
    pm[py::int_(0)] = py::float_(0.0);
    py::array alpha = np.attr("broadcast_to")(np.attr("zeros")(pol::N), py::make_tuple(list_size, pol::N));
    out.list_size = static_cast<std::size_t>(alpha.shape(0));
    return out;
}

// A 2-D uint8 array (rows x cols) holding `v`.
inline py::array_t<std::uint8_t> to_numpy_2d(const std::vector<std::uint8_t>& v, py::ssize_t rows,
                                             py::ssize_t cols) {
    py::array_t<std::uint8_t> out({rows, cols});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

}  // namespace sstvae_bindings::qrss::polar_util
