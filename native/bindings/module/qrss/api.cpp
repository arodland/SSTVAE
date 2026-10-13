// sstvae_native.qrss.api: the in-process transmit API (native/core/qrss/api),
// for its tests. Not a port of a Python module, so there is no
// substitution row: what it is compared with is qrss_transmit.py itself
// (tests/test_qrss_native_api.py).
//
// `src` is a StoredPicture / BeaconFile (the reference's dataclasses, or
// anything with their fields) or a path, read as qrss_transmit.py reads
// its input (api::read_source). The request's keywords are
// api::PassRequest's fields with qrss_transmit.py's defaults.
//
// The codec half (open_encoder) is bound from qrss/codec/api.cpp, only
// with the codec.

#include "qrss/registry.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "qrss/api.hpp"
#include "qrss/picture_chain.hpp"
#include "qrss/polar_util.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace api = sstvae::qrss::api;
using namespace picture_chain;

api::Waveform waveform_of(const std::string& name) {
    const auto w = api::waveform_named(name);
    if (!w) throw py::value_error("waveform must be 'CE' or 'L', not " + py::repr(py::str(name)).cast<std::string>());
    return *w;
}

api::Source src_of(const py::handle& src) {
    if (py::isinstance<py::str>(src) || py::hasattr(src, "__fspath__")) {
        const std::string path = utf8_of(py::module_::import("os").attr("fspath")(src));
        py::gil_scoped_release unlock;
        return api::read_source(path);
    }
    return source_of(src);
}

std::optional<std::string> opt_text(const py::handle& v, const char* what) {
    if (v.is_none()) return std::nullopt;
    return text_of(v, what);
}

api::PassRequest request_of(std::int64_t q, const std::string& waveform, double freq_hz, const py::handle& callsign,
                            const py::handle& grid, std::int64_t segment, const std::string& frame,
                            double lead_in_s, bool ook) {
    api::PassRequest r;
    r.q = q;
    r.waveform = waveform_of(waveform);
    r.freq_hz = freq_hz;
    r.callsign = opt_text(callsign, "callsign");
    r.grid = opt_text(grid, "grid");
    r.segment = segment;
    r.frame = frame;
    r.lead_in_s = lead_in_s;
    r.ook = ook;
    return r;
}

// The request's keywords, after `src` and `q`.
#define SSTVAE_API_REQUEST_ARGS                                                                        \
    py::arg("src"), py::arg("q"), py::kw_only(), py::arg("waveform") = "CE",                           \
        py::arg("freq_hz") = api::CARRIER_DEFAULT_HZ, py::arg("callsign") = py::none(),                \
        py::arg("grid") = py::none(), py::arg("segment") = 0, py::arg("frame") = "full",               \
        py::arg("lead_in_s") = 0.0, py::arg("ook") = false

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("api", "The in-process QRSSTVAE transmit API (native/core/qrss/api.hpp)");

    m.attr("SAMPLE_RATE") = api::SAMPLE_RATE;
    m.attr("SLOT_S") = api::SLOT_S;
    m.attr("T0_OFFSET_S") = api::T0_OFFSET_S;
    m.attr("AUDIO_LEAD_S") = api::AUDIO_LEAD_S;
    m.attr("AUDIO_TAIL_S") = api::AUDIO_TAIL_S;
    m.attr("PASS_S") = api::PASS_S;
    m.attr("CARRIER_MIN_HZ") = api::CARRIER_MIN_HZ;
    m.attr("CARRIER_MAX_HZ") = api::CARRIER_MAX_HZ;
    m.attr("CARRIER_DEFAULT_HZ") = api::CARRIER_DEFAULT_HZ;
    m.attr("LEAD_IN_MAX_S") = api::LEAD_IN_MAX_S;
    py::list frames;
    for (std::string_view f : api::FRAMES) frames.append(py::str(std::string(f)));
    m.attr("FRAMES") = py::tuple(frames);

    m.def("mode_index", [](const std::string& name) -> py::object {
        const auto i = api::mode_index(name);
        return i ? py::object(py::int_(*i)) : py::object(py::none());
    }, py::arg("name"));
    m.def("passes_for", [](const std::string& mode) { return api::passes_for(mode); }, py::arg("mode"));
    m.def("frame_name", [](const std::string& waveform, const std::string& frame) {
        return api::frame_spec(waveform_of(waveform), frame).name();
    }, py::arg("waveform"), py::arg("frame") = "full");

    m.def("parse_slot", [](const py::str& text) { return api::parse_slot(utf8_of(text)); }, py::arg("text"));
    m.def("slot_iso", [](std::int64_t qq) { return api::slot_iso(qq); }, py::arg("q"));
    m.def("slot_of", [](double t) { return api::slot_of(t); }, py::arg("quarter_hour_unix_s"));
    m.def("audio_start_s", [](std::int64_t qq) { return api::audio_start_s(qq); }, py::arg("q"));
    m.def("pass_samples", [](const std::string& waveform, const std::string& frame) {
        return api::pass_samples(waveform_of(waveform), frame);
    }, py::arg("waveform") = "CE", py::arg("frame") = "full");
    m.def("audio_end_s", [](std::int64_t qq, const std::string& waveform, const std::string& frame) {
        return api::audio_end_s(qq, waveform_of(waveform), frame);
    }, py::arg("q"), py::arg("waveform") = "CE", py::arg("frame") = "full");

    // The source as the API reads it: the picture's or beacon file's fields.
    m.def("read_source", [](const py::object& path) -> py::dict {
        const api::Source s = src_of(path);
        if (const auto* sp = std::get_if<pc::StoredPicture>(&s)) return picture_dict(*sp);
        return beacon_dict(std::get<sstvae::qrss::beaconfile::BeaconFile>(s));
    }, py::arg("path"));

    m.def("problem", [](const py::object& src, std::int64_t qq, const std::string& waveform, double freq_hz,
                        const py::object& callsign, const py::object& grid, std::int64_t segment,
                        const std::string& frame, double lead_in_s, bool ook) {
        const api::Source s = src_of(src);
        const api::PassRequest r = request_of(qq, waveform, freq_hz, callsign, grid, segment, frame, lead_in_s, ook);
        return api::problem(s, r);
    }, SSTVAE_API_REQUEST_ARGS);

    // float32 at 8 kHz; ValueError with problem()'s text.
    m.def("pass_audio", [](const py::object& src, std::int64_t qq, const std::string& waveform, double freq_hz,
                           const py::object& callsign, const py::object& grid, std::int64_t segment,
                           const std::string& frame, double lead_in_s, bool ook) {
        const api::Source s = src_of(src);
        const api::PassRequest r = request_of(qq, waveform, freq_hz, callsign, grid, segment, frame, lead_in_s, ook);
        std::vector<float> a;
        {
            py::gil_scoped_release unlock;
            a = api::pass_audio(s, r);
        }
        return to_numpy(std::span<const float>(a));
    }, SSTVAE_API_REQUEST_ARGS);

    // encode_picture with a Python encoder: (3, 480, 640) float32 -> the
    // full flat latents. `rgb` is (H, W, 3) uint8, any size.
    m.def("encode_picture", [](const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& rgb,
                               std::int64_t mode, std::int64_t codec_id, const py::object& encoder) {
        if (rgb.ndim() != 3 || rgb.shape(2) != 3) throw py::value_error("rgb must be (H, W, 3) uint8");
        sstvae::images::Picture p(static_cast<int>(rgb.shape(1)), static_cast<int>(rgb.shape(0)));
        std::copy(rgb.data(), rgb.data() + p.rgb.size(), p.rgb.begin());
        const api::Encoder enc = [&encoder](const sstvae::images::ImageArray& img) {
            py::array_t<float> a({py::ssize_t{3}, static_cast<py::ssize_t>(img.height),
                                  static_cast<py::ssize_t>(img.width)});
            std::copy(img.chw.begin(), img.chw.end(), a.mutable_data());
            const py::array_t<double, py::array::c_style | py::array::forcecast> flat = encoder(a);
            return std::vector<double>(flat.data(), flat.data() + flat.size());
        };
        return picture_dict(api::encode_picture(p, mode, codec_id, enc));
    }, py::arg("rgb"), py::arg("mode"), py::arg("codec_id"), py::arg("encoder"));
}

#undef SSTVAE_API_REQUEST_ARGS

const Registrar reg("api", 90, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
