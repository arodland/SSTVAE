// sstvae_native.qrss.api.open_encoder: the codec half of the in-process
// transmit API (native/core/qrss/codec/api.hpp), only with the codec.
//
//   enc = open_encoder(model=None, precision="fp16", codec_id=None)
//   enc.codec_id, enc.onnx_path
//   enc.encode_picture(rgb, mode)   # (H, W, 3) uint8 -> StoredPicture fields
//   enc.encode_file(path, mode)     # as qrss_encode.py reads its image
//
// Refusals: SystemExit for resolve_onnx's (as the codec binding raises
// them), FileNotFoundError for a directory with neither part, RuntimeError
// "<onnxruntime's class>: <message>" for an artifact onnxruntime refuses,
// ValueError otherwise.

#include "qrss/registry.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "qrss/api.hpp"
#include "qrss/codec/api.hpp"
#include "qrss/codec/codec.hpp"
#include "qrss/fspath.hpp"
#include "qrss/picture_chain.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace api = sstvae::qrss::api;
namespace qc = sstvae::qrss::codec;
using namespace picture_chain;

template <typename F>
auto translated(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const qc::NoModelFile& e) {
        fspath::raise_text(PyExc_FileNotFoundError, e.what());
    } catch (const sstvae::checkpoint::CheckpointError& e) {
        fspath::raise_text(PyExc_SystemExit, e.what());
    } catch (const qc::OrtError& e) {
        fspath::raise_text(PyExc_RuntimeError, e.python_class() + ": " + e.what());
    } catch (const std::overflow_error&) {
        throw;
    } catch (const std::range_error&) {
        throw;
    } catch (const std::runtime_error& e) {
        // the image layer's ("cannot read <path>: ..."), the name decoded
        // as Python decodes names
        fspath::raise_text(PyExc_RuntimeError, e.what());
    }
}

void bind(py::module_& q) {
    py::module_ m = py::reinterpret_borrow<py::module_>(q.attr("api"));

    py::class_<api::OpenEncoder>(m, "OpenEncoder")
        .def_readonly("codec_id", &api::OpenEncoder::codec_id)
        .def_property_readonly("onnx_path",
                               [](const api::OpenEncoder& enc) { return fspath::fs_text(enc.onnx_path); })
        .def("encode_picture", [](const api::OpenEncoder& enc,
                                  const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& rgb,
                                  std::int64_t mode) {
            if (rgb.ndim() != 3 || rgb.shape(2) != 3) throw py::value_error("rgb must be (H, W, 3) uint8");
            sstvae::images::Picture p(static_cast<int>(rgb.shape(1)), static_cast<int>(rgb.shape(0)));
            std::copy(rgb.data(), rgb.data() + p.rgb.size(), p.rgb.begin());
            return translated([&] {
                sstvae::qrss::picture::StoredPicture sp;
                {
                    py::gil_scoped_release unlock;
                    sp = api::encode_picture(p, mode, enc.codec_id, enc.encode);
                }
                return picture_dict(sp);
            });
        }, py::arg("rgb"), py::arg("mode"))
        .def("encode_file", [](const api::OpenEncoder& enc, const py::object& path, std::int64_t mode) {
            const std::string p = fspath::fs_path(path).bytes;   // Image.open's: str, bytes or os.PathLike
            return translated([&] {
                sstvae::qrss::picture::StoredPicture sp;
                {
                    py::gil_scoped_release unlock;
                    sp = api::encode_file(p, mode, enc);
                }
                return picture_dict(sp);
            });
        }, py::arg("path"), py::arg("mode"));

    m.def("open_encoder", [](const py::object& model, const std::string& precision, const py::object& codec_id) {
        std::optional<std::string> mdl;
        if (!model.is_none()) mdl = fspath::pathlib_path(model).bytes;   // --model, made a Path
        std::optional<std::int64_t> cid;
        if (!codec_id.is_none()) cid = int64_of(codec_id);
        return translated([&] { return api::open_encoder(mdl, precision, cid); });
    }, py::arg("model") = py::none(), py::arg("precision") = "fp16", py::arg("codec_id") = py::none());
}

// After "api" (90), which creates the submodule.
const Registrar reg("api_codec", 96, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
