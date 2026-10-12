// sstvae_native.qrss.header: bindings of native/core/qrss/header.
// Names and ValueError texts follow sstvae/qrss/header.py. HeaderFields is
// a Python dataclass, so these take its eight fields as arguments and
// return them as a tuple; tests/qrss_native/header.py (pack, unpack,
// encode, ...) and header_rx.py (decode) adapt to and from the dataclass.
//
// The binding makes the reference's checks with Python's own operations on
// what the caller passed -- `int()` for picture_id and codec_id, `<=`,
// `in` and `!=` for mode, segment, version and reserved, in
// _check_fields' order -- and then converts each field with `int()`, as
// _int_to_bits does, so a float, a numpy scalar, a bool or an int of any
// size is accepted, refused or converted exactly as there (ints clamped
// into int64, past which every check refuses). The refusal texts are built
// from Python's own repr()/format() of the arguments, so they are the
// reference's character for character.

#include "qrss/registry.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/header.hpp"
#include "qrss/polar.hpp"
#include "qrss/polar_util.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace hdr = sstvae::qrss::header;
using namespace polar_util;

std::string utf8(const py::handle& s) {
    return py::bytes(s.attr("encode")("utf-8", "surrogatepass")).cast<std::string>();
}

std::string repr_of(const py::handle& o) { return py::repr(o).cast<std::string>(); }
std::string str_of(const py::handle& o) { return py::str(o).cast<std::string>(); }

// A grid argument for the C++: None, a str (upper-cased by Python, as the
// reference does before its regex, so a non-ASCII case mapping agrees), or
// anything else as a value the check refuses. The message text is repr(g).
std::optional<std::string> grid_arg(const py::handle& g, std::optional<std::string>& text) {
    if (g.is_none()) return std::nullopt;
    text = repr_of(g);
    if (py::isinstance<py::str>(g)) return utf8(g.attr("upper")());
    return std::string("\x01");
}

struct Converted {
    hdr::HeaderFields h;
    hdr::FieldText text;
};

// header._check_fields, then pack's own conversions, on the Python objects.
Converted fields(const py::object& callsign, const py::object& grid, const py::object& picture_id,
                 const py::object& mode, const py::object& segment, const py::object& codec_id,
                 const py::object& version, const py::object& reserved) {
    Converted c;
    c.text.callsign = repr_of(callsign);
    c.h.callsign = py::isinstance<py::str>(callsign) ? utf8(callsign) : std::string();
    if (!py::isinstance<py::str>(callsign) || !hdr::valid_callsign(c.h.callsign))
        throw py::value_error("callsign must be 1-8 characters of [A-Z0-9/], got " + *c.text.callsign);
    const py::object to_int = builtin("int");
    const struct {
        const char* name;
        const py::object* obj;
        std::int64_t hi;
    } ranged[] = {{"picture_id", &picture_id, std::int64_t(1) << 32}, {"codec_id", &codec_id, std::int64_t(1) << 16}};
    for (const auto& r : ranged) {
        const std::int64_t v = clamp_i64(to_int(*r.obj));       // 0 <= int(x) < hi
        if (!(0 <= v && v < r.hi)) throw py::value_error(std::string(r.name) + " out of range: " + fmt(*r.obj));
    }
    const py::int_ zero(0);
    if (!(compare(zero, mode, Py_LE) && compare(mode, py::int_(hdr::MAX_MODE), Py_LE)))
        throw py::value_error("mode must be 0.." + std::to_string(hdr::MAX_MODE) + ", got " + fmt(mode));
    if (!(compare(zero, segment, Py_LE) && compare(segment, mode, Py_LE)))
        throw py::value_error("segment must be 0..mode (" + fmt(mode) + "), got " + fmt(segment));
    py::tuple known(sstvae::qrss::config::FORMAT_VERSIONS.size());
    for (std::size_t i = 0; i < sstvae::qrss::config::FORMAT_VERSIONS.size(); ++i)
        known[i] = sstvae::qrss::config::FORMAT_VERSIONS[i];
    const int in = PySequence_Contains(known.ptr(), version.ptr());
    if (in < 0) throw py::error_already_set();
    if (!in) throw py::value_error("version must be one of " + str_of(known) + ", got " + fmt(version));
    if (compare(reserved, zero, Py_NE)) throw py::value_error("reserved must be 0, got " + fmt(reserved));
    // pack: the grid, then int() of each field in table order.
    c.h.grid = grid_arg(grid, c.text.grid);
    (void)hdr::grid_encode(c.h.grid ? std::optional<std::string_view>(*c.h.grid) : std::nullopt, c.text.grid);
    const std::pair<const py::object*, std::int64_t*> ints[] = {
        {&picture_id, &c.h.picture_id}, {&mode, &c.h.mode}, {&segment, &c.h.segment},
        {&codec_id, &c.h.codec_id}, {&version, &c.h.version}, {&reserved, &c.h.reserved}};
    for (const auto& [obj, dst] : ints) *dst = clamp_i64(to_int(*obj));
    c.text.picture_id = fmt(picture_id);
    c.text.mode = fmt(mode);
    c.text.segment = fmt(segment);
    c.text.codec_id = fmt(codec_id);
    c.text.version = fmt(version);
    c.text.reserved = fmt(reserved);
    return c;
}

py::object as_tuple(const std::optional<hdr::HeaderFields>& h) {
    if (!h) return py::none();
    py::object grid = h->grid ? py::object(py::str(*h->grid)) : py::object(py::none());
    return py::make_tuple(h->callsign, grid, h->picture_id, h->mode, h->segment, h->codec_id, h->version,
                          h->reserved);
}

// unpack(bits, versions) on 142 bits already known to be 0/1: the CRC, then
// Python's own `version in versions` -- consulted only once the CRC has
// passed, as the reference does, so a generator, a str or a non-container
// behaves (and fails) the same -- then the field checks.
std::optional<hdr::HeaderFields> unpack_py(std::span<const std::uint8_t> b, const py::object& versions) {
    const int c0 = hdr::offset("crc").first;
    const auto crc = hdr::crc16(b.first(static_cast<std::size_t>(c0)));
    if (!std::equal(crc.begin(), crc.end(), b.begin() + c0)) return std::nullopt;
    const auto [v0, v1] = hdr::offset("version");
    int version = 0;
    for (int i = v0; i < v1; ++i) version = (version << 1) | b[static_cast<std::size_t>(i)];
    const int in = PySequence_Contains(versions.ptr(), py::int_(version).ptr());
    if (in < 0) throw py::error_already_set();
    std::vector<int> allowed;
    if (in) allowed.push_back(version);
    return hdr::unpack(b, allowed);
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("header", "Port of sstvae.qrss.header");
    m.attr("CALL_CHARS") = hdr::CALL_CHARS;
    m.attr("GRID_NONE") = hdr::GRID_NONE;
    m.attr("MAX_MODE") = hdr::MAX_MODE;
    py::dict off;
    for (const auto& f : hdr::FIELDS) {
        const auto [a, b] = hdr::offset(f.name);
        off[py::str(std::string(f.name))] = py::make_tuple(a, b);
    }
    m.attr("OFFSET") = off;

    // 0/1 bits (bool, integer or float, 1-D) go to the C++ beacon CRC; any
    // other input runs the reference's register over Python ints --
    // `for b in bits: reg = ((reg << 1) & 0xFFFF) | int(b)` and numpy's
    // shift for the 16 bits out -- so it is computed, or refused, exactly as
    // there (a 2, a 0.5, a 2-D array, a 0-d one).
    m.def("crc16", [](const py::object& bits) -> py::object {
        py::array a = numpy().attr("asarray")(bits);
        const char kind = a.dtype().kind();
        if (a.ndim() == 1 && (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f'))
            if (const auto v = bits01(a)) return to_numpy(hdr::crc16(*v));
        const py::object to_int = builtin("int");
        const py::int_ one(1), fifteen(15), mask(0xFFFF), poly(0x1021);
        py::object reg = py::int_(0xFFFF);
        for (py::handle b : a) {
            const py::object top = steal(PyNumber_And(steal(PyNumber_Rshift(reg.ptr(), fifteen.ptr())).ptr(), one.ptr()));
            const py::object low = steal(PyNumber_And(steal(PyNumber_Lshift(reg.ptr(), one.ptr())).ptr(), mask.ptr()));
            reg = steal(PyNumber_Or(low.ptr(), to_int(b).ptr()));
            if (truthy(top)) reg = steal(PyNumber_Xor(reg.ptr(), poly.ptr()));
        }
        py::module_ np = numpy();
        const py::object shifts = np.attr("arange")(15, -1, -1);
        const py::object out = steal(PyNumber_And(steal(PyNumber_Rshift(reg.ptr(), shifts.ptr())).ptr(), one.ptr()));
        return out.attr("astype")(np.attr("int64")).attr("astype")(np.attr("uint8"));
    }, py::arg("bits"));

    m.def("grid_encode", [](const py::object& g) {
        std::optional<std::string> text;
        const auto s = grid_arg(g, text);
        return hdr::grid_encode(s ? std::optional<std::string_view>(*s) : std::nullopt, text);
    }, py::arg("g"));

    m.def("grid_decode", [](const py::object& v) -> py::object {
        py::int_ iv = py::module_::import("builtins").attr("int")(v);   // v = int(v)
        const std::int64_t code = clamp_i64(iv);
        // Clamped (or at an int64 bound): no grid code, and the text is Python's.
        if (code == std::numeric_limits<std::int64_t>::min() || code == std::numeric_limits<std::int64_t>::max())
            throw py::value_error("invalid grid code " + str_of(iv));
        const auto g = hdr::grid_decode(code);
        if (!g) return py::none();
        return py::str(*g);
    }, py::arg("v"));

    m.def("pack", [](const py::object& callsign, const py::object& grid, const py::object& picture_id,
                     const py::object& mode, const py::object& segment, const py::object& codec_id,
                     const py::object& version, const py::object& reserved) {
        const Converted c = fields(callsign, grid, picture_id, mode, segment, codec_id, version, reserved);
        return to_numpy(hdr::pack(c.h, c.text));
    }, py::arg("callsign"), py::arg("grid"), py::arg("picture_id"), py::arg("mode"), py::arg("segment"),
       py::arg("codec_id"), py::arg("version") = sstvae::qrss::config::FORMAT_VERSION, py::arg("reserved") = 0);

    m.def("encode", [](const py::object& callsign, const py::object& grid, const py::object& picture_id,
                       const py::object& mode, const py::object& segment, const py::object& codec_id,
                       const py::object& version, const py::object& reserved) {
        const Converted c = fields(callsign, grid, picture_id, mode, segment, codec_id, version, reserved);
        return to_numpy(hdr::encode(c.h, c.text));
    }, py::arg("callsign"), py::arg("grid"), py::arg("picture_id"), py::arg("mode"), py::arg("segment"),
       py::arg("codec_id"), py::arg("version") = sstvae::qrss::config::FORMAT_VERSION, py::arg("reserved") = 0);

    // (callsign, grid, picture_id, mode, segment, codec_id, version, reserved) or None.
    m.def("unpack", [](const py::object& bits, const py::object& versions) -> py::object {
        py::array a = numpy().attr("asarray")(bits);
        if (a.ndim() != 1 || a.shape(0) != sstvae::qrss::config::N_INFO_BITS) return py::none();
        const auto v = bits01(a);
        if (!v) return py::none();
        return as_tuple(unpack_py(*v, versions));
    }, py::arg("bits"), py::arg("versions") = py::make_tuple(sstvae::qrss::config::FORMAT_VERSION));

    // The reference's flow: `tuple(versions) == (FORMAT_VERSION,)` picks
    // polar_decode's default acceptance (header::unpack with the default
    // versions), anything else accepts by unpack(b, versions); the result
    // is unpack(bits, versions). `versions` is the caller's object
    // throughout, so an iterator consumed by the tuple() is empty for every
    // later `in`, as there.
    m.def("decode", [](const py::object& llr, const py::object& list_size, const py::object& versions) -> py::object {
        const bool plain = py::tuple(versions).equal(py::make_tuple(sstvae::qrss::config::FORMAT_VERSION));
        const DecodeArgs a = decode_args(llr, list_size);
        sstvae::qrss::polar::Accept accept;
        if (!plain)
            accept = [&versions](std::span<const std::uint8_t> b) { return unpack_py(b, versions).has_value(); };
        const auto bits = sstvae::qrss::polar::polar_decode(a.llr, a.list_size, accept);
        if (!bits) return py::none();
        return as_tuple(unpack_py(*bits, versions));
    }, py::arg("llr"), py::arg("list_size") = 8,
       py::arg("versions") = py::make_tuple(sstvae::qrss::config::FORMAT_VERSION));
}

const Registrar reg("header", 51, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
