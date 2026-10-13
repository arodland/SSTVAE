// Conversions shared by the picture, beaconfile and tx bindings
// (bindings/module/qrss/{picture,beaconfile,tx}.cpp).
//
// The reference passes its own dataclasses around -- StoredPicture,
// BeaconFile, HeaderFields -- and those stay the objects Python callers
// see. So these bindings take any object with the dataclass's attributes
// (the reference's own objects pass straight through) and hand results
// back as dicts / tuples of fields, which tests/qrss_native/{picture,
// beaconfile,tx}.py wrap into the dataclasses.
//
// Integers are read with int() and clamped into int64 (every check on them
// is a range check, so a value past int64 is refused as the original is);
// strings cross as UTF-8.

#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "qrss/beaconfile.hpp"
#include "qrss/frame.hpp"
#include "qrss/half.hpp"
#include "qrss/header.hpp"
#include "qrss/picture.hpp"
#include "qrss/polar_util.hpp"
#include "qrss/registry.hpp"
#include "qrss/tx.hpp"

namespace sstvae_bindings::qrss::picture_chain {

namespace pc = sstvae::qrss::picture;
namespace bfm = sstvae::qrss::beaconfile;
namespace hdr = sstvae::qrss::header;
namespace half = sstvae::qrss::half;
using polar_util::builtin;
using polar_util::clamp_i64;
using polar_util::numpy;
using polar_util::steal;
using polar_util::truthy;

// int(obj) as int64 (clamped).
inline std::int64_t int64_of(const py::handle& obj) {
    return clamp_i64(py::reinterpret_borrow<py::int_>(steal(PyNumber_Long(obj.ptr()))));
}

// str(obj): how an f-string prints an argument the reference quotes.
inline std::string str_of(const py::handle& obj) { return py::str(obj).cast<std::string>(); }

// float(obj), as Python's float() reads it (a str, a Fraction, an int).
inline double float_of(const py::handle& obj) {
    return PyFloat_AsDouble(steal(PyNumber_Float(obj.ptr())).ptr());
}

inline std::string utf8_of(const py::handle& s) {
    return py::bytes(steal(PyUnicode_AsEncodedString(s.ptr(), "utf-8", "surrogatepass"))).cast<std::string>();
}

inline py::str str_from_utf8(const std::string& s) {
    return py::reinterpret_borrow<py::str>(
        steal(PyUnicode_DecodeUTF8(s.data(), static_cast<py::ssize_t>(s.size()), "surrogatepass")));
}

// A str attribute as UTF-8 (TypeError for anything else).
inline std::string text_of(const py::handle& s, const char* what) {
    if (!py::isinstance<py::str>(s))
        throw py::type_error(std::string(what) + " must be a str, not " +
                             py::type::of(s).attr("__name__").cast<std::string>());
    return utf8_of(s);
}

inline std::optional<std::string> grid_of(const py::handle& g) {
    if (g.is_none()) return std::nullopt;
    return text_of(g, "grid");
}

inline std::vector<std::int64_t> shape_of(const py::array& a) {
    return std::vector<std::int64_t>(a.shape(), a.shape() + a.ndim());
}

// np.asarray(s) as float16 values, exactly as s.astype("<f2") gives them:
// float16 as it is, float32 and float64 rounded once by half.hpp (numpy's
// own conversions), any other dtype through numpy's astype.
inline pc::Segment segment_of(const py::array& a) {
    const std::string d = a.dtype().attr("str").cast<std::string>();
    pc::Segment out(static_cast<std::size_t>(a.size()));
    if (d == "<f4") {
        auto v = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(a);
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = pc::Half{half::from_float(v.data()[i])};
        return out;
    }
    if (d == "<f8") {
        auto v = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = half::to_half(v.data()[i]);
        return out;
    }
    py::object h = d == "<f2" ? py::object(a) : a.attr("astype")("<f2");
    py::array c = numpy().attr("ascontiguousarray")(h).attr("view")("<u2");
    if (!out.empty()) std::memcpy(out.data(), c.data(), out.size() * sizeof(pc::Half));
    return out;
}

// A float16 numpy array of `v` with `shape`.
inline py::array half_array(const std::vector<pc::Half>& v, const std::vector<py::ssize_t>& shape) {
    py::array out(py::dtype("float16"), shape);
    if (!v.empty()) std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(pc::Half));
    return out;
}

// --- StoredPicture ------------------------------------------------------------------

// picture_id(codec_id, mode, segs) with the reference's checks in its order
// (mode, codec ID, segment count, each segment's shape), the segments
// converted as s.astype("<f2"). Returns the ID and fills `segs_out`.
inline std::uint32_t picture_id_of(const py::handle& codec_id, const py::handle& mode, const py::handle& segs,
                                   std::vector<pc::Segment>* segs_out = nullptr) {
    const std::int64_t m = pc::check_mode(int64_of(mode), str_of(mode));
    const std::int64_t c = int64_of(codec_id);
    pc::check_codec_id(c, str_of(codec_id));
    const py::list items = builtin("list")(segs);
    pc::check_segment_count(m, items.size());
    std::vector<pc::Segment> conv;
    for (const py::handle s : items) {
        const py::array a = numpy().attr("asarray")(s);
        pc::check_segment_shape(shape_of(a));
        conv.push_back(segment_of(a));
    }
    const std::uint32_t pid = pc::picture_id(c, m, conv);
    if (segs_out) *segs_out = std::move(conv);
    return pid;
}

// StoredPicture.check() on any object with its fields; the checked picture.
inline pc::StoredPicture checked_picture(const py::handle& sp) {
    pc::StoredPicture out;
    const std::uint32_t pid = picture_id_of(sp.attr("codec_id"), sp.attr("mode"), sp.attr("segs"), &out.segs);
    out.mode = int64_of(sp.attr("mode"));
    out.codec_id = int64_of(sp.attr("codec_id"));
    out.picture_id = int64_of(sp.attr("picture_id"));
    pc::check_id(out.picture_id, pid);
    return out;
}

// The fields of any object with StoredPicture's attributes, unchecked (as
// the transmit path reads them: it never calls check()).
inline pc::StoredPicture picture_of(const py::handle& sp) {
    pc::StoredPicture out;
    out.mode = int64_of(sp.attr("mode"));
    out.codec_id = int64_of(sp.attr("codec_id"));
    out.picture_id = int64_of(sp.attr("picture_id"));
    for (const py::handle s : py::reinterpret_borrow<py::iterable>(sp.attr("segs")))
        out.segs.push_back(segment_of(numpy().attr("asarray")(s)));
    return out;
}

inline py::dict picture_dict(const pc::StoredPicture& sp) {
    py::dict d;
    d["mode"] = py::int_(sp.mode);
    d["codec_id"] = py::int_(sp.codec_id);
    py::list segs;
    for (const pc::Segment& s : sp.segs) segs.append(half_array(s, {static_cast<py::ssize_t>(s.size())}));
    d["segs"] = segs;
    d["picture_id"] = py::int_(sp.picture_id);
    return d;
}

// --- HeaderFields ---------------------------------------------------------------------

inline hdr::HeaderFields header_of(const py::handle& h) {
    hdr::HeaderFields out;
    out.callsign = text_of(h.attr("callsign"), "callsign");
    out.grid = grid_of(h.attr("grid"));
    out.picture_id = int64_of(h.attr("picture_id"));
    out.mode = int64_of(h.attr("mode"));
    out.segment = int64_of(h.attr("segment"));
    out.codec_id = int64_of(h.attr("codec_id"));
    out.version = int64_of(h.attr("version"));
    out.reserved = int64_of(h.attr("reserved"));
    return out;
}

inline std::optional<hdr::HeaderFields> opt_header_of(const py::handle& h) {
    if (h.is_none()) return std::nullopt;
    return header_of(h);
}

// HeaderFields' eight fields, in its order.
inline py::tuple header_tuple(const hdr::HeaderFields& h) {
    py::object grid = h.grid ? py::object(str_from_utf8(*h.grid)) : py::object(py::none());
    return py::make_tuple(str_from_utf8(h.callsign), grid, h.picture_id, h.mode, h.segment, h.codec_id,
                          h.version, h.reserved);
}

// --- BeaconFile ------------------------------------------------------------------------

// 0/1 values of a 1-D array; any other value becomes 2 (never a bit, so
// compared with one it is a mismatch, as the reference's array_equal
// finds), and a shape other than 1-D an empty vector (never the right
// length).
inline std::vector<std::uint8_t> bits_of(const py::handle& obj) {
    const py::array a = numpy().attr("asarray")(obj);
    if (a.ndim() != 1) return {};
    const char kind = a.dtype().kind();
    if (kind != 'b' && kind != 'i' && kind != 'u' && kind != 'f') {
        // np.array_equal would compare these elementwise; nothing of another
        // kind equals a bit vector here.
        return std::vector<std::uint8_t>(static_cast<std::size_t>(a.size()), 2);
    }
    auto v = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(v.size()));
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double x = v.data()[i];
        out[i] = x == 0.0 ? 0 : x == 1.0 ? 1 : 2;
    }
    return out;
}

// The latents as int8, and -- when they are not a 1-D int8 array -- how
// BeaconFile.check() prints them ("float64(50600,)"); the values are then
// numpy's cast, for the transmit path, which reads any dtype.
struct Latents {
    std::vector<std::int8_t> values;
    std::optional<std::string> not_int8;
};

inline Latents latents_of(const py::handle& obj) {
    const py::array a = numpy().attr("asarray")(obj);
    Latents out;
    if (!(a.dtype().attr("str").cast<std::string>() == "|i1" && a.ndim() == 1))
        out.not_int8 = str_of(a.dtype()) + py::str(a.attr("shape")).cast<std::string>();
    py::array c = numpy().attr("ascontiguousarray")(a.attr("astype")("int8")).attr("reshape")(-1);
    out.values.resize(static_cast<std::size_t>(c.size()));
    if (!out.values.empty()) std::memcpy(out.values.data(), c.data(), out.values.size());
    return out;
}

// Any object with BeaconFile's attributes; `not_int8` reports latents that
// are not a 1-D int8 array (the C++ BeaconFile holds int8 only).
inline bfm::BeaconFile beacon_of(const py::handle& bf, std::optional<std::string>* not_int8 = nullptr) {
    bfm::BeaconFile out;
    out.waveform = int64_of(bf.attr("waveform"));
    out.mode = int64_of(bf.attr("mode"));
    out.segment = int64_of(bf.attr("segment"));
    out.picture_id = int64_of(bf.attr("picture_id"));
    out.codec_id = int64_of(bf.attr("codec_id"));
    out.callsign = text_of(bf.attr("callsign"), "callsign");
    out.grid = grid_of(bf.attr("grid"));
    out.ook = truthy(bf.attr("ook"));
    out.keying = bits_of(bf.attr("keying"));
    out.hdr_bits = bits_of(bf.attr("hdr_bits"));
    Latents lat = latents_of(bf.attr("latents_i8"));
    out.latents_i8 = std::move(lat.values);
    if (not_int8) *not_int8 = std::move(lat.not_int8);
    return out;
}

// BeaconFile.check() on such an object: the C++ check, and the reference's
// dtype refusal in its place (after every other field) for latents that
// are not int8.
inline bfm::BeaconFile checked_beacon(const py::handle& bf) {
    std::optional<std::string> not_int8;
    bfm::BeaconFile out = beacon_of(bf, &not_int8);
    if (not_int8) {
        bfm::BeaconFile probe = out;
        probe.latents_i8.assign(1, 0);
        probe.check();
        throw py::value_error("latents must be 1.." + std::to_string(pc::SENT) + " int8 values, not " + *not_int8);
    }
    out.check();
    return out;
}

template <typename T>
py::array_t<T> array_of(const std::vector<T>& v) {
    return to_numpy(std::span<const T>(v));
}

inline py::dict beacon_dict(const bfm::BeaconFile& bf) {
    py::dict d;
    d["waveform"] = py::int_(bf.waveform);
    d["mode"] = py::int_(bf.mode);
    d["segment"] = py::int_(bf.segment);
    d["picture_id"] = py::int_(bf.picture_id);
    d["codec_id"] = py::int_(bf.codec_id);
    d["callsign"] = str_from_utf8(bf.callsign);
    d["grid"] = bf.grid ? py::object(str_from_utf8(*bf.grid)) : py::object(py::none());
    d["ook"] = py::bool_(bf.ook);
    d["keying"] = array_of(bf.keying);
    d["hdr_bits"] = array_of(bf.hdr_bits);
    d["latents_i8"] = array_of(bf.latents_i8);
    return d;
}

// --- sources --------------------------------------------------------------------------

inline bool is_beacon(const py::handle& src) { return py::hasattr(src, "latents_i8"); }
inline bool is_picture(const py::handle& src) { return py::hasattr(src, "segs"); }

// A transmit source: a BeaconFile-like object, else a StoredPicture-like
// one, else tx._segment_and_header's TypeError.
inline sstvae::qrss::tx::Source source_of(const py::handle& src) {
    if (is_beacon(src)) return beacon_of(src);
    if (is_picture(src)) return picture_of(src);
    throw py::type_error("source must be a StoredPicture or a BeaconFile, not " + str_of(py::type::of(src)));
}

// bytes(data), as the reference's from_bytes reads its argument.
inline std::vector<std::uint8_t> bytes_of(const py::handle& data) {
    const py::bytes b = py::reinterpret_borrow<py::bytes>(steal(PyBytes_FromObject(data.ptr())));
    char* p = nullptr;
    py::ssize_t n = 0;
    if (PyBytes_AsStringAndSize(b.ptr(), &p, &n) != 0) throw py::error_already_set();
    return std::vector<std::uint8_t>(reinterpret_cast<const std::uint8_t*>(p),
                                     reinterpret_cast<const std::uint8_t*>(p) + n);
}

inline py::bytes as_bytes(const std::vector<std::uint8_t>& v) {
    return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
}

// open(path, "rb").read() / open(path, "wb"): Python's own file handling,
// so a missing file or a bad path fails with Python's OSError, as np.load
// and open() do in the reference.
inline std::vector<std::uint8_t> read_file(const py::handle& path) {
    py::object f = builtin("open")(path, "rb");
    py::object data;
    try {
        data = f.attr("read")();
    } catch (...) {
        f.attr("close")();
        throw;
    }
    f.attr("close")();
    return bytes_of(data);
}

}  // namespace sstvae_bindings::qrss::picture_chain
