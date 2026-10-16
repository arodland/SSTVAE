// sstvae_native.qrss.morse: bindings of native/core/qrss/morse.
// Signatures, dtypes and ValueError texts follow sstvae/qrss/morse.py, so
// tests/qrss_native/morse.py can assign these over the reference's
// functions.
//
// A str crosses as UTF-8 with "surrogatepass", so every Python string,
// lone surrogates included, reaches the C++ as the code points Python
// sees (qrss/pytext.hpp decodes that form) and is counted identically.
//
// A refusal is worded here, from the core's CallsignError fields, with
// Python's own repr() of the caller's object: morse.py quotes
// `{callsign!r}`, and a str subclass words that differently from a str
// (np.str_ reads "np.str_('k1abc')"), which no C++ repr of the text can
// know. That also makes the binding's text exact where pytext simplifies
// (an unassigned code point, which Python escapes).

#include "qrss/registry.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "qrss/qrss_config.hpp"
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

// Raise ValueError(msg) with the str itself, which may hold lone
// surrogates that a std::string message could not carry.
[[noreturn]] void value_error(const py::str& msg) {
    PyErr_SetObject(PyExc_ValueError, msg.ptr());
    throw py::error_already_set();
}

// morse.check_callsign's type check, then the text as UTF-8.
std::string callsign_of(const py::object& callsign) {
    if (!py::isinstance<py::str>(callsign))
        value_error(py::str("callsign must be a str, not {}").format(py::type::of(callsign).attr("__name__")));
    return utf8_of(callsign);
}

// One code point (a lone surrogate included) as a str.
py::str chr_of(char32_t cp) {
    const Py_UCS4 c = static_cast<Py_UCS4>(cp);
    py::object o = py::reinterpret_steal<py::object>(PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, &c, 1));
    if (!o) throw py::error_already_set();
    return py::reinterpret_borrow<py::str>(o);
}

// f(), with a CallsignError re-raised in morse.py's words.
template <typename F>
auto refusing_as_python(const py::object& callsign, F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const mo::CallsignError& e) {
        switch (e.why()) {
        case mo::CallsignError::Why::length:
            value_error(py::str("callsign {!r} must be 1 to {} characters").format(callsign, qc::MAX_CALL_CHARS));
        case mo::CallsignError::Why::characters: {
            py::list bad;
            for (char32_t c : e.bad()) bad.append(chr_of(c));
            value_error(py::str("callsign {!r} has characters outside [A-Z0-9/]: {}").format(callsign, bad));
        }
        case mo::CallsignError::Why::units:
            value_error(py::str("callsign {!r} needs {} Morse units, more than the window's {}")
                            .format(callsign, e.units(), qc::CW_ROOM_UNITS));
        }
        throw;
    }
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("morse", "Port of sstvae.qrss.morse");

    py::dict table;
    for (const qc::MorseCode& c : qc::MORSE) table[py::str(std::string(1, c.ch))] = py::str(std::string(c.code));
    m.attr("MORSE") = table;
    m.attr("MAX_CALL_CHARS") = qc::MAX_CALL_CHARS;

    m.def("check_callsign", [](const py::object& callsign) {
        const std::string text = callsign_of(callsign);
        return str_of_utf8(refusing_as_python(callsign, [&] { return mo::check_callsign(text); }));
    }, py::arg("callsign"));

    m.def("morse_keying", [](const py::object& callsign) {
        const std::string text = callsign_of(callsign);
        return to_numpy(refusing_as_python(callsign, [&] { return mo::morse_keying(text); }));
    }, py::arg("callsign"));

    m.def("keying_units", [](const py::object& callsign) {
        const std::string text = callsign_of(callsign);
        return to_numpy(refusing_as_python(callsign, [&] { return mo::keying_units(text); }));
    }, py::arg("callsign"));

    // The core's own refusal text (pytext's quoting), bound for the parity
    // test only: what a C++ caller is shown. None when accepted.
    m.def("_core_refusal", [](const py::str& callsign) -> py::object {
        try {
            mo::keying_units(utf8_of(callsign));
        } catch (const mo::CallsignError& e) {
            return str_of_utf8(e.what());
        }
        return py::none();
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
