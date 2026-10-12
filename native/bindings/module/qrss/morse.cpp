// sstvae_native.qrss.morse: bindings of native/core/qrss/morse.
// Signatures, dtypes and ValueError texts follow sstvae/qrss/morse.py, so
// tests/qrss_native/morse.py can assign these over the reference's
// functions.
//
// A str crosses as UTF-8 with "surrogatepass", so every Python string,
// lone surrogates included, reaches the C++ as the code points Python
// sees (qrss/pytext.hpp decodes that form), and the refusal texts quote
// it identically.

#include "qrss/registry.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "qrss/config.hpp"
#include "qrss/morse.hpp"
#include "qrss/pytext.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace mo = sstvae::qrss::morse;
namespace qc = sstvae::qrss::config;

std::string utf8_of(const py::handle& s) {
    py::object b = py::reinterpret_steal<py::object>(PyUnicode_AsEncodedString(s.ptr(), "utf-8", "surrogatepass"));
    if (!b) throw py::error_already_set();
    return b.cast<std::string>();
}

py::str str_of_utf8(const std::string& s) {
    py::object o = py::reinterpret_steal<py::object>(
        PyUnicode_DecodeUTF8(s.data(), static_cast<py::ssize_t>(s.size()), "surrogatepass"));
    if (!o) throw py::error_already_set();
    return py::reinterpret_borrow<py::str>(o);
}

// morse.check_callsign's type check, then the text as UTF-8.
std::string callsign_of(const py::object& callsign) {
    if (!py::isinstance<py::str>(callsign))
        throw py::value_error("callsign must be a str, not " +
                              py::str(py::type::of(callsign).attr("__name__")).cast<std::string>());
    return utf8_of(callsign);
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("morse", "Port of sstvae.qrss.morse");

    py::dict table;
    for (const qc::MorseCode& c : qc::MORSE) table[py::str(std::string(1, c.ch))] = py::str(std::string(c.code));
    m.attr("MORSE") = table;
    m.attr("MAX_CALL_CHARS") = qc::MAX_CALL_CHARS;

    m.def("check_callsign", [](const py::object& callsign) {
        return str_of_utf8(mo::check_callsign(callsign_of(callsign)));
    }, py::arg("callsign"));

    m.def("morse_keying", [](const py::object& callsign) {
        return to_numpy(mo::morse_keying(callsign_of(callsign)));
    }, py::arg("callsign"));

    m.def("keying_units", [](const py::object& callsign) {
        return to_numpy(mo::keying_units(callsign_of(callsign)));
    }, py::arg("callsign"));

    // qrss/pytext, bound for the parity test only: repr() of a str, and
    // str.isprintable() of each code point of a uint32 array.
    m.def("_repr", [](const py::str& s) {
        return str_of_utf8(sstvae::qrss::pytext::repr(std::string_view(utf8_of(s))));
    }, py::arg("s"));
    m.def("_isprintable", [](const py::object& cps) {
        const std::vector<std::uint32_t> v = from_numpy<std::uint32_t>(cps);
        std::vector<std::uint8_t> out(v.size());
        for (std::size_t i = 0; i < v.size(); ++i)
            out[i] = sstvae::qrss::pytext::isprintable(static_cast<char32_t>(v[i])) ? 1 : 0;
        return to_numpy(out).attr("astype")("bool");
    }, py::arg("cps"));
}

const Registrar reg("morse", 20, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
