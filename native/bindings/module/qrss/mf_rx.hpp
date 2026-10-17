// The matched filter's binding, shared by ce_rx.cpp (ce.matched_filter) and
// lwave_rxf.cpp (lwave.matched_filter), which differ only in the pulse.
//
// matched_filter(z, fs, n0, pos, chunk=2048), in the reference's order:
// z = np.asarray(z) (complex dtypes give complex128 out, anything else
// float64, numpy's result_type with float64), fs through _grid_step (int(fs),
// "fs must divide 16000, not {fs}"), pos = np.asarray(pos, int64) and
// len(pos) (a 0-d pos: TypeError "len() of unsized object"), then
// range(0, len(pos), chunk)'s own refusal of a zero chunk ("range() arg 3
// must not be zero"; a negative chunk evaluates nothing: zeros). Only then,
// and only when the loop body runs (some pos, chunk > 0), are n0 and z
// looked at: idx = nn - n0 with nn int64 must come out integer, or numpy
// refuses the gather ("arrays used as indices must be of integer (or
// boolean) type" -- a float n0, even 3.0, never truncates), and len(z)
// refuses a 0-d z. The result does not otherwise depend on chunk. z and
// pos must be 1-D (the reference's arithmetic assumes it; anything else
// there gives a meaningless shape or a broadcast error).

#pragma once

#include "qrss/registry.hpp"

#include <pybind11/complex.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sstvae_bindings::qrss::mf_rx {

using cd = std::complex<double>;

inline const py::object& numpy_mod() {
    static py::object* np = new py::object(py::module_::import("numpy"));
    return *np;
}

// int(x) as int64.
inline std::int64_t int_of(const py::handle& h) {
    py::object i = py::reinterpret_steal<py::object>(PyNumber_Long(h.ptr()));
    if (!i) throw py::error_already_set();
    const long long v = PyLong_AsLongLong(i.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

// operator.index(x) as int64 (what range() takes).
inline std::int64_t index_of(const py::handle& h) {
    py::object i = py::reinterpret_steal<py::object>(PyNumber_Index(h.ptr()));
    if (!i) throw py::error_already_set();
    const long long v = PyLong_AsLongLong(i.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

// idx = nn - n0 as the reference evaluates it, nn being an int64 array:
// the subtraction's own refusal propagates unchanged (None, a str, a Python
// int past int64), and a result that is not an integer array is numpy's
// refusal of it as an index.
inline std::int64_t n0_of(const py::handle& h) {
    const py::object nn = numpy_mod().attr("zeros")(1, py::arg("dtype") = "int64");
    py::object r = py::reinterpret_steal<py::object>(PyNumber_Subtract(nn.ptr(), h.ptr()));
    if (!r) throw py::error_already_set();
    const py::array ra = py::array::ensure(r);
    const char kind = ra ? ra.dtype().kind() : 'O';
    if (kind != 'i' && kind != 'u' && kind != 'b') {
        PyErr_SetString(PyExc_IndexError, "arrays used as indices must be of integer (or boolean) type");
        throw py::error_already_set();
    }
    return index_of(h);
}

[[noreturn]] inline void unsized() {
    PyErr_SetString(PyExc_TypeError, "len() of unsized object");
    throw py::error_already_set();
}

// `real(z, fs, n0, pos)` / `cplx(...)` are the core's two overloads;
// `step(fs)` its _grid_step (refusing fs first, as the reference does).
template <typename Step, typename Real, typename Cplx>
py::object matched_filter(const py::object& z_in, const py::object& fs_in, const py::object& n0_in,
                          const py::object& pos_in, const py::object& chunk_in, Step&& step, Real&& real,
                          Cplx&& cplx) {
    const py::array z = numpy_mod().attr("asarray")(z_in);
    const std::int64_t fs = int_of(fs_in);
    step(fs);
    const py::array pos_a = numpy_mod().attr("asarray")(pos_in, py::arg("dtype") = "int64");
    if (pos_a.ndim() == 0) unsized();   // np.zeros(len(pos), ...)
    const auto n_pos = static_cast<std::size_t>(pos_a.shape(0));
    const std::int64_t chunk = index_of(chunk_in);
    if (chunk == 0) {
        PyErr_SetString(PyExc_ValueError, "range() arg 3 must not be zero");
        throw py::error_already_set();
    }
    const bool is_complex = z.dtype().kind() == 'c';
    if (chunk < 0 || n_pos == 0) {   // range(0, len(pos), chunk) is empty: out stays zeros
        if (is_complex) return to_numpy(std::vector<cd>(n_pos));
        return to_numpy(std::vector<double>(n_pos));
    }
    // The loop body: p, first, nn, mm, then idx = nn - n0, then len(z).
    if (pos_a.ndim() != 1)
        throw py::value_error("matched_filter: pos must be 1-D, not " + std::to_string(pos_a.ndim()) + "-D");
    const std::vector<std::int64_t> pos = from_numpy<std::int64_t>(pos_a);
    const std::int64_t n0 = n0_of(n0_in);
    if (z.ndim() == 0) unsized();
    if (z.ndim() != 1) throw py::value_error("matched_filter: z must be 1-D, not " + std::to_string(z.ndim()) + "-D");
    if (is_complex) {
        const std::vector<cd> zz = from_numpy<cd>(numpy_mod().attr("asarray")(z, py::arg("dtype") = "complex128"));
        return to_numpy(cplx(std::span<const cd>(zz), fs, n0, std::span<const std::int64_t>(pos)));
    }
    const std::vector<double> zz = from_numpy<double>(numpy_mod().attr("asarray")(z, py::arg("dtype") = "float64"));
    return to_numpy(real(std::span<const double>(zz), fs, n0, std::span<const std::int64_t>(pos)));
}

}  // namespace sstvae_bindings::qrss::mf_rx
