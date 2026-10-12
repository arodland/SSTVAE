// sstvae_native.qrss.header: bindings of native/core/qrss/header.
// Names and ValueError texts follow sstvae/qrss/header.py. HeaderFields is
// a Python dataclass, so these take its eight fields as arguments and
// return them as a tuple; tests/qrss_native/header.py (pack, unpack,
// encode, ...) and header_rx.py (decode) adapt to and from the dataclass.
//
// Integers are taken through operator.index (Python ints of any size,
// numpy integers, bools) and clamped into int64, which every check here
// refuses exactly as it would the original; the refusal texts are built
// from Python's own str()/repr() of what the caller passed
// (header::FieldText), so they are the reference's character for
// character.

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

Converted fields(const py::object& callsign, const py::object& grid, const py::object& picture_id,
                 const py::object& mode, const py::object& segment, const py::object& codec_id,
                 const py::object& version, const py::object& reserved) {
    Converted c;
    c.text.callsign = repr_of(callsign);
    c.h.callsign = py::isinstance<py::str>(callsign) ? utf8(callsign) : std::string();
    c.h.grid = grid_arg(grid, c.text.grid);
    const std::array<std::pair<const py::object*, std::int64_t*>, 6> ints{{
        {&picture_id, &c.h.picture_id}, {&mode, &c.h.mode}, {&segment, &c.h.segment},
        {&codec_id, &c.h.codec_id}, {&version, &c.h.version}, {&reserved, &c.h.reserved}}};
    for (const auto& [obj, dst] : ints) *dst = clamp_i64(index(*obj));
    c.text.picture_id = str_of(picture_id);
    c.text.mode = str_of(mode);
    c.text.segment = str_of(segment);
    c.text.codec_id = str_of(codec_id);
    c.text.version = str_of(version);
    c.text.reserved = str_of(reserved);
    return c;
}

py::object as_tuple(const std::optional<hdr::HeaderFields>& h) {
    if (!h) return py::none();
    py::object grid = h->grid ? py::object(py::str(*h->grid)) : py::object(py::none());
    return py::make_tuple(h->callsign, grid, h->picture_id, h->mode, h->segment, h->codec_id, h->version,
                          h->reserved);
}

// `version in versions` for every value the 4-bit field can hold, by
// Python's own `in` over tuple(versions).
std::vector<int> versions_of(const py::object& versions) {
    py::tuple vs = py::tuple(versions);
    std::vector<int> out;
    for (int k = 0; k < 16; ++k)
        if (vs.contains(py::int_(k))) out.push_back(k);
    return out;
}

std::size_t list_size_of(const py::object& list_size) {
    py::int_ ls = index(list_size);
    if (ls < py::int_(1)) throw py::value_error("list_size must be >= 1");
    return ls.cast<std::size_t>();
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

    // Over 0/1 bits only (the reference's documented domain); see crc16 in
    // core/qrss/header.hpp.
    m.def("crc16", [](const py::object& bits) {
        py::array a = numpy().attr("asarray")(bits);
        if (a.ndim() != 1) throw py::type_error("crc16 takes a 1-D array of bits, not shape " + shape_str(a));
        auto v = bits01(a);
        if (!v) throw py::value_error("crc16 takes bits of 0 or 1");
        return to_numpy(hdr::crc16(*v));
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
    // `versions` is consulted as the reference consults it: Python's own
    // `version in versions`, made only once the CRC has passed (so a
    // generator, a str or a non-container behaves, and fails, the same).
    m.def("unpack", [](const py::object& bits, const py::object& versions) -> py::object {
        py::array a = numpy().attr("asarray")(bits);
        if (a.ndim() != 1 || a.shape(0) != sstvae::qrss::config::N_INFO_BITS) return py::none();
        const auto v = bits01(a);
        if (!v) return py::none();
        const std::span<const std::uint8_t> b(*v);
        const int c0 = hdr::offset("crc").first;
        const auto crc = hdr::crc16(b.first(static_cast<std::size_t>(c0)));
        if (!std::equal(crc.begin(), crc.end(), b.begin() + c0)) return py::none();
        const auto [v0, v1] = hdr::offset("version");
        int version = 0;
        for (int i = v0; i < v1; ++i) version = (version << 1) | b[static_cast<std::size_t>(i)];
        std::vector<int> allowed;
        if (truthy(py::module_::import("operator").attr("contains")(versions, version))) allowed.push_back(version);
        return as_tuple(hdr::unpack(b, allowed));
    }, py::arg("bits"), py::arg("versions") = py::make_tuple(sstvae::qrss::config::FORMAT_VERSION));

    m.def("decode", [](const py::object& llr, const py::object& list_size, const py::object& versions) -> py::object {
        const std::vector<int> vs = versions_of(versions);
        const std::size_t L = list_size_of(list_size);
        const std::vector<double> v = float_vector(
            llr, sstvae::qrss::config::N_HDR_BITS,
            "expected " + std::to_string(sstvae::qrss::config::N_HDR_BITS) + " coded LLRs");
        return as_tuple(hdr::decode(v, L, vs));
    }, py::arg("llr"), py::arg("list_size") = 8,
       py::arg("versions") = py::make_tuple(sstvae::qrss::config::FORMAT_VERSION));
}

const Registrar reg("header", 51, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
