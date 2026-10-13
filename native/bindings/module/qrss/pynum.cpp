// sstvae_native.qrss.pynum: Python's int() and float() on text
// (native/core/qrss/pynum), which the sstvae-qrss command line parses its
// numbers with. Like infra, no sstvae.qrss attribute is replaced: the
// parity tests (tests/test_qrss_native_cli.py) compare these with int() and
// float() themselves.

#include "qrss/registry.hpp"

#include <optional>
#include <string>

#include "qrss/picture_chain.hpp"
#include "qrss/pynum.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace pn = sstvae::qrss::pynum;
using picture_chain::str_from_utf8;
using picture_chain::utf8_of;

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("pynum", "Python's int() and float() on text (native/core/qrss/pynum)");

    // int(text, base) for base 10 / 16: (str(value), value, or None past
    // int64), or None where int() raises ValueError.
    m.def("py_int", [](const py::str& text, int base) -> py::object {
        const std::optional<pn::PyInt> v = pn::py_int(utf8_of(text), base);
        if (!v) return py::none();
        return py::make_tuple(v->decimal, v->value ? py::object(py::int_(*v->value)) : py::object(py::none()));
    }, py::arg("text"), py::arg("base"));

    m.def("invalid_int_literal", [](const py::str& text, int base) {
        return str_from_utf8(pn::invalid_int_literal(utf8_of(text), base));
    }, py::arg("text"), py::arg("base"));

    // float(text), or None where float() raises ValueError.
    m.def("py_float", [](const py::str& text) -> py::object {
        const std::optional<double> v = pn::py_float(utf8_of(text));
        return v ? py::object(py::float_(*v)) : py::object(py::none());
    }, py::arg("text"));
}

const Registrar reg("pynum", 1, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
