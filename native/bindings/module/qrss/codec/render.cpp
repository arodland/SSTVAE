// sstvae_native.qrss.codec_render: native/core/qrss/codec/render (the
// decoder side of the codec glue: render.load / check_codec_id /
// decoder_codec_id / render, and qrss_receive.py render_pass's placement of
// a pass), for tests/test_qrss_native_cli_receive.py to compare with the
// reference. Built only with SSTVAE's codec (this directory's glob).
//
// No rows: every reference function here takes a loaded codec object
// (render.render(codec, ...), check_codec_id(codec, ...)), which nothing
// native can stand in for; the tests compare outcomes instead.
//
//   check_codec_id(have, codec_id, force=False) -> str | None, or raises
//       sstvae.qrss.render.CodecMismatch (the reference's class and text)
//   pass_planes(z, w, segment) -> (S, W) float64 (3, 52800)
//   decoder_codec_id(path=None, precision="fp32") -> int | None
//   render(S, W, mode, path=None, precision="fp32") -> uint8 (480, 640, 3)

#include "qrss/registry.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "qrss/codec/codec.hpp"
#include "qrss/codec/render.hpp"
#include "qrss/picture_chain.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace qc = sstvae::qrss::codec;
using namespace picture_chain;

[[noreturn]] void raise_as(PyObject* type, const std::string& what) {
    PyErr_SetString(type, what.c_str());
    throw py::error_already_set();
}

template <typename F>
auto translated(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const qc::CodecMismatch& e) {
        py::object cls = py::module_::import("sstvae.qrss.render").attr("CodecMismatch");
        PyErr_SetObject(cls.ptr(), py::str(e.what()).ptr());
        throw py::error_already_set();
    } catch (const qc::OrtError& e) {
        raise_as(PyExc_RuntimeError, e.python_class() + ": " + e.what());
    } catch (const sstvae::checkpoint::CheckpointError& e) {
        PyErr_SetObject(PyExc_SystemExit, py::str(e.what()).ptr());
        throw py::error_already_set();
    }
}

std::optional<std::int64_t> opt_int(const py::handle& v) {
    if (v.is_none()) return std::nullopt;
    return int64_of(v);
}

std::optional<std::string> opt_text(const py::handle& v) {
    if (v.is_none()) return std::nullopt;
    return utf8_of(py::module_::import("os").attr("fspath")(v));
}

std::vector<double> flat_f64(const py::handle& a) {
    const auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
    if (!arr) throw py::type_error("expected an array of numbers");
    return std::vector<double>(arr.data(), arr.data() + arr.size());
}

std::vector<float> flat_f32(const py::handle& a) {
    const auto arr = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(a);
    if (!arr) throw py::type_error("expected an array of numbers");
    return std::vector<float>(arr.data(), arr.data() + arr.size());
}

py::array picture_array(const sstvae::images::Picture& p) {
    py::array_t<std::uint8_t> out({static_cast<py::ssize_t>(p.height), static_cast<py::ssize_t>(p.width),
                                   py::ssize_t{3}});
    std::copy(p.rgb.begin(), p.rgb.end(), out.mutable_data());
    return out;
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("codec_render", "Port of sstvae.qrss.render's codec half (native/core/qrss/codec)");

    m.def("check_codec_id", [](const py::object& have, const py::object& codec_id, bool force) -> py::object {
        const auto h = opt_int(have), c = opt_int(codec_id);
        return translated([&]() -> py::object {
            const auto w = qc::check_codec_id(h, c, force);
            return w ? py::object(str_from_utf8(*w)) : py::object(py::none());
        });
    }, py::arg("have"), py::arg("codec_id"), py::arg("force") = false);

    m.def("pass_planes", [](const py::object& z, const py::object& w, std::int64_t segment) {
        const std::vector<float> zz = flat_f32(z), ww = flat_f32(w);
        auto [S, W] = qc::pass_planes(zz, ww, segment);
        auto shaped = [](const std::vector<double>& v) {
            py::array_t<double> out({py::ssize_t{3}, static_cast<py::ssize_t>(v.size() / 3)});
            std::copy(v.begin(), v.end(), out.mutable_data());
            return out;
        };
        return py::make_tuple(shaped(S), shaped(W));
    }, py::arg("z"), py::arg("w"), py::arg("segment"));

    m.def("decoder_codec_id", [](const py::object& path, const std::string& precision) -> py::object {
        const auto p = opt_text(path);
        return translated([&]() -> py::object {
            std::optional<std::int64_t> id;
            {
                py::gil_scoped_release unlock;
                id = qc::open_decoder(p, precision).codec_id;
            }
            return id ? py::object(py::int_(*id)) : py::object(py::none());
        });
    }, py::arg("path") = py::none(), py::arg("precision") = "fp32");

    m.def("render", [](const py::object& S, const py::object& W, std::int64_t mode, const py::object& path,
                       const std::string& precision) {
        const std::vector<double> s = flat_f64(S), w = flat_f64(W);
        const auto p = opt_text(path);
        return translated([&] {
            sstvae::images::Picture pic;
            {
                py::gil_scoped_release unlock;
                pic = qc::render(qc::open_decoder(p, precision), s, w, mode);
            }
            return picture_array(pic);
        });
    }, py::arg("S"), py::arg("W"), py::arg("mode"), py::arg("path") = py::none(), py::arg("precision") = "fp32");
}

const Registrar reg("codec_render", 96, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
