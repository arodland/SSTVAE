// sstvae_native.qrss.tx: bindings of native/core/qrss/tx (the slot
// functions; transmit_audio comes with the CE and L audio). Names,
// signatures (keyword-only arguments included), dtypes and ValueError
// texts follow sstvae/qrss/tx.py.
//
// A source is any object with BeaconFile's attributes (latents_i8, ...)
// or StoredPicture's (segs, ...) -- the reference's own dataclasses pass
// straight through -- and a header any object with HeaderFields'. A frame
// is a native FrameSpec or the reference's FrameSpec / LFrameSpec, through
// sstvae_native.qrss.frame.of. tests/qrss_native/tx.py wraps the one
// HeaderFields that comes back.

#include "qrss/registry.hpp"

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "qrss/frame.hpp"
#include "qrss/picture_chain.hpp"
#include "qrss/tx.hpp"

namespace sstvae_bindings::qrss {
namespace {

using namespace picture_chain;
namespace fr = sstvae::qrss::frame;
namespace txm = sstvae::qrss::tx;

// sstvae_native.qrss.frame.of, looked up once the frame binding exists (it
// binds first). Leaked on purpose: nothing may run its destructor after
// the interpreter is gone.
py::object* frame_of_fn = nullptr;

fr::FrameSpec spec_of(const py::handle& spec) {
    if (py::isinstance<fr::FrameSpec>(spec)) return spec.cast<fr::FrameSpec>();
    return (*frame_of_fn)(spec).cast<fr::FrameSpec>();
}

std::optional<std::int64_t> opt_int(const py::handle& v) {
    if (v.is_none()) return std::nullopt;
    return int64_of(v);
}

// The `segment` argument as _segment_and_header reads it: int(segment) for
// a stored picture, and for a beacon file Python's `segment not in (None,
// src.segment)` -- an equality test, so 1.0 is the file's segment 1 and 1.5
// is not (refused quoting str(1.5), which `opt_shown` supplies).
std::optional<std::int64_t> segment_arg(const txm::Source& s, const py::handle& segment) {
    if (segment.is_none()) return std::nullopt;
    if (const auto* bf = std::get_if<bfm::BeaconFile>(&s)) {
        const py::int_ own(bf->segment);
        if (segment.ptr() == own.ptr()) return bf->segment;
        const int eq = PyObject_RichCompareBool(segment.ptr(), own.ptr(), Py_EQ);
        if (eq < 0) throw py::error_already_set();
        return eq ? bf->segment : bf->segment + 1;    // + 1: any segment but the file's
    }
    return int64_of(segment);
}

std::optional<std::string> opt_shown(const py::handle& v) {
    if (v.is_none()) return std::nullopt;
    return str_of(v);
}

// use_int8 as the reference reads it: `use_int8 is False` refuses a beacon
// file, and its truth value picks the int8 version of a stored picture --
// so None and a falsy value other than False both mean "as stored".
std::optional<bool> use_int8_of(const py::handle& v) {
    if (v.is_none()) return std::nullopt;
    if (v.ptr() == Py_False) return false;
    if (truthy(v)) return true;
    return std::nullopt;
}

py::object symbols_array(const txm::Symbols& s) {
    if (s.waveform == fr::Waveform::L) return to_numpy(std::span<const std::complex<double>>(s.l));
    return to_numpy(std::span<const double>(s.ce));
}

py::object opt_bits(const std::optional<std::vector<std::uint8_t>>& v) {
    if (!v) return py::none();
    return to_numpy(std::span<const std::uint8_t>(*v));
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("tx", "Port of sstvae.qrss.tx (the slot functions)");
    frame_of_fn = new py::object(q.attr("frame").attr("of"));
    const py::object full = q.attr("frame").attr("get")("full");

    // (segment, header, hdr bits, keying, ook). The header is the object
    // given, for a stored picture, or the beacon file's own fields as a
    // tuple (the adapter makes it a HeaderFields).
    m.def("_segment_and_header", [](const py::object& src, const py::object& segment, const py::object& header) {
        const txm::Source s = source_of(src);
        const txm::SegmentAndHeader r =
            txm::segment_and_header(s, segment_arg(s, segment), opt_header_of(header), opt_shown(segment));
        py::object h = py::none();
        if (std::holds_alternative<bfm::BeaconFile>(s))
            h = header_tuple(*r.header);
        else if (r.header)
            h = header;
        return py::make_tuple(r.segment, h, opt_bits(r.hdr_bits), opt_bits(r.keying), py::bool_(r.ook));
    }, py::arg("src"), py::arg("segment"), py::arg("header"));

    m.def("slot_air_latents", [](const py::object& src, const py::object& segment, const py::object& use_int8) {
        const txm::Source s = is_beacon(src) ? txm::Source(beacon_of(src)) : txm::Source(picture_of(src));
        return to_numpy(txm::slot_air_latents(s, opt_int(segment), use_int8_of(use_int8)));
    }, py::arg("src"), py::arg("segment") = py::none(), py::arg("use_int8") = py::none());

    m.def("slot_symbols", [](const py::object& src, const py::object& qv, const py::object& spec,
                             const py::object& segment, const py::object& header, const py::object& use_int8) {
        const txm::Source s = source_of(src);
        const fr::FrameSpec f = spec_of(spec);
        const txm::SlotPlan plan =
            txm::plan_symbols(s, f, segment_arg(s, segment), opt_header_of(header), use_int8_of(use_int8),
                              opt_shown(segment));
        // precode's int(q) and its range check come last, as in the reference.
        const py::int_ qi = py::reinterpret_borrow<py::int_>(steal(PyNumber_Long(qv.ptr())));
        const std::uint64_t qq = to_u64(qi, "q must fit in uint64, not " + str_of(qi));
        return symbols_array(txm::assemble_symbols(plan, f, qq));
    }, py::arg("src"), py::arg("q"), py::arg("spec") = full, py::kw_only(), py::arg("segment") = py::none(),
       py::arg("header") = py::none(), py::arg("use_int8") = py::none());

    m.def("slot_keying", [](const py::object& src, const py::object& header) -> py::object {
        const txm::Source s = is_beacon(src) ? txm::Source(beacon_of(src)) : txm::Source(sstvae::qrss::picture::StoredPicture{});
        return opt_bits(txm::slot_keying(s, opt_header_of(header)));
    }, py::arg("src"), py::arg("header") = py::none());

    m.def("slot_samples", [](const py::object& spec, const py::object& pre_s, const py::object& post_s,
                             const py::object& fs) {
        return txm::slot_samples(spec_of(spec), float_of(pre_s), float_of(post_s), float_of(fs));
    }, py::arg("spec"), py::arg("pre_s") = 12.0, py::arg("post_s") = 3.0,
       py::arg("fs") = sstvae::qrss::config::FS);

    m.def("check_pilot", [](const py::object& carrier_hz) {
        return txm::check_pilot(float_of(carrier_hz), str_of(carrier_hz));
    }, py::arg("carrier_hz"));
}

const Registrar reg("tx", 62, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
