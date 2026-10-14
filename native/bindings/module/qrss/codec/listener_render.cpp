// sstvae_native.qrss.listener.ListenerRenderer: native/core/qrss/codec/
// listener_render (live.py's _codec and the codec calls of its _render),
// the renderer a LiveListener draws its pictures with when SSTVAE's codec
// is built (this directory's glob). Registered after the listener module,
// into it.
//
//   ListenerRenderer(precision="fp32", model=None)
//       .loaded, .decoder_codec_id, .error, .model

#include "qrss/registry.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "qrss/codec/listener_render.hpp"
#include "qrss/fspath.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace qc = sstvae::qrss::codec;
namespace lv = sstvae::qrss::live;

std::optional<std::string> opt_path(const py::handle& v) {
    if (v.is_none()) return std::nullopt;
    return fspath::pathlib_path(v).bytes;
}

py::object opt_str(const std::optional<std::string>& s) {
    return s ? py::object(py::str(PyUnicode_DecodeUTF8(s->data(), static_cast<py::ssize_t>(s->size()), "replace"),
                                  false))
             : py::object(py::none());
}

void bind(py::module_& q) {
    py::module_ m = q.attr("listener").cast<py::module_>();
    py::class_<qc::ListenerRenderer, lv::Renderer, std::shared_ptr<qc::ListenerRenderer>>(
        m, "ListenerRenderer", "The codec-backed renderer (native/core/qrss/codec/listener_render)")
        .def(py::init([](const std::string& precision, const py::object& model) {
                 return std::make_shared<qc::ListenerRenderer>(precision, opt_path(model));
             }),
             py::arg("precision") = "fp32", py::arg("model") = py::none())
        .def_property_readonly("loaded", [](const qc::ListenerRenderer& r) {
            py::gil_scoped_release unlock;
            return r.loaded();
        })
        .def_property_readonly("decoder_codec_id", [](const qc::ListenerRenderer& r) -> py::object {
            std::optional<std::int64_t> id;
            {
                py::gil_scoped_release unlock;
                id = r.decoder_codec_id();
            }
            return id ? py::object(py::int_(*id)) : py::object(py::none());
        })
        .def_property_readonly("error", [](const qc::ListenerRenderer& r) {
            std::optional<std::string> e;
            {
                py::gil_scoped_release unlock;
                e = r.error();
            }
            return opt_str(e);
        })
        .def_property_readonly("model", [](const qc::ListenerRenderer& r) {
            return opt_str(r.model());
        });
}

const Registrar reg("listener_codec", 98, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
