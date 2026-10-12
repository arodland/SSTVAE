// sstvae_native.qrss.precoder: bindings of native/core/qrss/precoder.
// Signatures, dtypes, shapes and error texts follow sstvae/qrss/precoder.py,
// so tests/qrss_native/precoder.py can assign these over the reference's
// functions. Inputs go through numpy exactly as the reference converts
// them (np.array / np.asarray with dtype float64), and ints through int(),
// so what is accepted, and how it fails, is numpy's and Python's own.

#include "qrss/registry.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/precoder.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace pc = sstvae::qrss::precoder;

using Shape = std::vector<py::ssize_t>;

std::string str_of(const py::handle& h) { return py::str(h).cast<std::string>(); }

py::object numpy() { return py::module_::import("numpy"); }

// np.array(v, dtype=np.float64, copy=copy) laid out in C order: a private
// buffer the binding may write when `copy`.
py::array_t<double> float64(const py::object& v, bool copy) {
    py::object np = numpy();
    py::object a = copy ? np.attr("array")(v, py::arg("dtype") = np.attr("float64"), py::arg("copy") = true,
                                           py::arg("order") = "C")
                        : np.attr("asarray")(v, py::arg("dtype") = np.attr("float64"));
    auto out = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
    if (!out) throw py::error_already_set();
    return out;
}

Shape shape_of(const py::array& a) { return Shape(a.shape(), a.shape() + a.ndim()); }

// a.shape[-1], with Python's error for a 0-d array.
std::size_t last_axis(const py::array& a) {
    if (a.ndim() == 0) throw py::index_error("tuple index out of range");
    return static_cast<std::size_t>(a.shape(a.ndim() - 1));
}

// int(x): PyNumber_Long, which is what Python's int() calls, so a float
// truncates, a str parses, a numpy integer converts and a bool (or any int
// subclass) becomes a plain int -- and what fails, fails as int() does.
py::int_ as_int(const py::object& x) {
    PyObject* r = PyNumber_Long(x.ptr());
    if (!r) throw py::error_already_set();
    return py::reinterpret_steal<py::int_>(r);
}

// scrambler's q = int(q) and its range check.
std::uint64_t slot_q(const py::object& q) {
    const py::int_ qi = as_int(q);
    return to_u64(qi, "q must fit in uint64, not " + str_of(qi));
}

// precoder._block_sizes(int(n)): ValueError below zero; a count no list
// can hold fails as Python's list repetition does.
std::size_t block_count(const py::object& n_obj) {
    const py::int_ n = as_int(n_obj);
    if (n < py::int_(0)) throw py::value_error("n must be >= 0, not " + str_of(n));
    const py::object blocks = n.attr("__floordiv__")(py::int_(pc::BLOCK));
    if (blocks > py::module_::import("sys").attr("maxsize")) {
        PyErr_SetString(PyExc_OverflowError, "cannot fit 'int' into an index-sized integer");
        throw py::error_already_set();
    }
    if (!(n < py::int_(1).attr("__lshift__")(64))) {
        PyErr_NoMemory();
        throw py::error_already_set();
    }
    return n.cast<std::size_t>();
}

template <typename F>
auto or_no_memory(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const std::length_error&) {
        PyErr_NoMemory();
        throw py::error_already_set();
    }
}

py::array_t<double> shaped(const std::vector<double>& v, const Shape& shape) {
    py::array_t<double> out(shape);
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("precoder", "Port of sstvae.qrss.precoder");
    m.attr("BLOCK") = py::int_(pc::BLOCK);

    m.def("block_sizes", [](const py::object& n) {
        const std::size_t count = block_count(n);
        py::list out;
        for (std::size_t s : or_no_memory([&] { return pc::block_sizes(count); })) out.append(py::int_(s));
        return out;
    }, py::arg("n"));

    // Private in the reference; bound for the parity test only.
    m.def("_runs", [](const py::object& n) {
        py::list out;
        for (const pc::Run& r : pc::runs(block_count(n)))
            out.append(py::make_tuple(r.start, r.m, r.count));
        return py::tuple(out);
    }, py::arg("n"));

    m.def("wht", [](const py::object& v) {
        py::array_t<double> out = float64(v, true);
        const std::size_t n = last_axis(out);
        pc::wht_inplace(std::span<double>(out.mutable_data(), static_cast<std::size_t>(out.size())), n);
        return out;
    }, py::arg("v"));

    m.def("precode", [](const py::object& a_air, const py::object& q) {
        const py::array_t<double> a = float64(a_air, false);
        const std::size_t n = last_axis(a);
        const std::uint64_t qq = slot_q(q);
        return shaped(pc::precode(std::span<const double>(a.data(), static_cast<std::size_t>(a.size())), qq, n),
                      shape_of(a));
    }, py::arg("a_air"), py::arg("q"));

    m.def("unprecode", [](const py::object& y_in, const py::object& q) {
        const py::array_t<double> y = float64(y_in, false);
        const std::size_t n = last_axis(y);
        const std::uint64_t qq = slot_q(q);
        return shaped(pc::unprecode(std::span<const double>(y.data(), static_cast<std::size_t>(y.size())), qq, n),
                      shape_of(y));
    }, py::arg("y"), py::arg("q"));

    m.def("block_mean", [](const py::object& v_in) {
        const py::array_t<double> v = float64(v_in, false);
        const std::size_t n = last_axis(v);
        return shaped(pc::block_mean(std::span<const double>(v.data(), static_cast<std::size_t>(v.size())), n),
                      shape_of(v));
    }, py::arg("v"));

    m.def("block_index", [](const py::object& n) {
        const std::size_t count = block_count(n);
        return to_numpy(or_no_memory([&] { return pc::block_index(count); }));
    }, py::arg("n"));
}

const Registrar reg("precoder", 20, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
