// How the QRSSTVAE port's bindings find their way into sstvae_native.
//
// The port lands one module at a time, and often several modules at
// once from separate branches, so no shared file lists them: each
// bindings/module/qrss/<module>.cpp registers itself with a static
// `Registrar`, and module.cpp calls `bind_all` once. The files are
// compiled straight into the extension (not into a static library),
// which is what guarantees their static initialisers run.
//
// Each module binds into `sstvae_native.qrss.<module>`, keeping the
// Python module's function names and signatures exactly: the pytest
// substitution (tests/qrss_native/) assigns these over the reference's
// attributes, and a binding that "improved" an interface would break
// the mechanism, as CLAUDE.md says of the SSTVAE bindings.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sstvae_bindings::qrss {

namespace py = pybind11;

using BindFn = void (*)(py::module_& qrss);

struct Entry {
    const char* name;
    int order;   // lower binds first; ties by name. Purely for determinism.
    BindFn fn;
};

std::vector<Entry>& registry();

struct Registrar {
    Registrar(const char* name, int order, BindFn fn) { registry().push_back({name, order, fn}); }
};

// Creates `m.qrss`, sets `__qrss_modules__` to the registered names, and
// binds each module in order.
void bind_all(py::module_& m);

// --- conversions shared by the QRSS bindings --------------------------------

// A 1-D numpy array holding a copy of `v`.
template <typename T>
py::array_t<T> to_numpy(std::span<const T> v, bool writeable = true) {
    py::array_t<T> out(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    if (!writeable) out.attr("setflags")(py::arg("write") = false);
    return out;
}

template <typename T>
py::array_t<T> to_numpy(const std::vector<T>& v, bool writeable = true) {
    return to_numpy(std::span<const T>(v), writeable);
}

// The contents of a numpy array (any shape, any dtype numpy can cast to
// T without error) as a flat vector, C order.
template <typename T>
std::vector<T> from_numpy(const py::handle& obj) {
    auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!a) throw py::type_error("expected an array-like");
    return std::vector<T>(a.data(), a.data() + a.size());
}

// Python ints of any size, as numpy would see them for a uint64 field:
// out of range raises ValueError with `what`.
inline std::uint64_t to_u64(const py::int_& v, const std::string& what) {
    py::object zero = py::int_(0);
    py::object top = py::int_(1).attr("__lshift__")(64);
    if (v < zero || !(v < top)) throw py::value_error(what);
    return v.cast<std::uint64_t>();
}

}  // namespace sstvae_bindings::qrss
