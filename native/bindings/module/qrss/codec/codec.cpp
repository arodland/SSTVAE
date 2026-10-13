// sstvae_native.qrss.codec, and picture.codec_id_from_onnx: bindings of
// native/core/qrss/codec (the codec glue, plan stage 1 item 16).
//
// Compiled into the module only when sstvae_qrss_codec exists, i.e. with
// SSTVAE's codec (onnxruntime): this directory is outside the qrss/*.cpp
// glob for that reason (native/bindings/module/CMakeLists.txt).
//
// picture.codec_id_from_onnx keeps the reference's name, signature and
// outcomes: an int, None for missing or unreadable metadata,
// FileNotFoundError for a directory without either part, and onnxruntime's
// own exception class (onnxruntime.capi.onnxruntime_pybind11_state.*, the
// Python binding's message text) for a file onnxruntime refuses. Where the
// reference raises SystemExit (checkpoint.resolve_onnx), so do these.
//
// encode_picture is qrss_encode.py's work after its arguments -- load the
// image, encode, StoredPicture.from_latents -- returning the StoredPicture's
// fields as a dict, as the picture bindings do.

#include "qrss/registry.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#include "qrss/codec/codec.hpp"
#include "qrss/picture_chain.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace qc = sstvae::qrss::codec;
using namespace picture_chain;

[[noreturn]] void raise_text(PyObject* type, const std::string& what) {
    PyErr_SetString(type, what.c_str());
    throw py::error_already_set();
}

// os.fspath(obj) as a str (pathlib's own TypeError for anything else).
std::string fspath(const py::handle& obj) {
    py::object p = py::module_::import("os").attr("fspath")(obj);
    if (!py::isinstance<py::str>(p))
        raise_text(PyExc_TypeError,
                   "argument should be a str or an os.PathLike object where __fspath__ returns a str, not " +
                       py::repr(py::type::of(p).attr("__name__")).cast<std::string>());
    return utf8_of(p);
}

std::optional<std::string> opt_path(const py::handle& obj) {
    if (obj.is_none()) return std::nullopt;
    return fspath(obj);
}

// The C++ refusals as the reference raises them.
template <typename F>
auto translated(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const qc::OrtError& e) {
        // onnxruntime's own class when its Python binding is there to name it.
        PyObject* type = PyExc_RuntimeError;
        py::object cls;
        try {
            py::module_ m = py::module_::import("onnxruntime.capi.onnxruntime_pybind11_state");
            if (py::hasattr(m, e.python_class().c_str())) {
                cls = m.attr(e.python_class().c_str());
                type = cls.ptr();
            }
        } catch (py::error_already_set&) {
            PyErr_Clear();
        }
        raise_text(type, e.what());
    } catch (const qc::NoModelFile& e) {
        raise_text(PyExc_FileNotFoundError, e.what());
    } catch (const sstvae::checkpoint::CheckpointError& e) {
        // checkpoint.resolve_onnx's SystemExit (ModelError included).
        PyErr_SetObject(PyExc_SystemExit, py::str(e.what()).ptr());
        throw py::error_already_set();
    }
}

py::object optional_int(const std::optional<std::int64_t>& v) {
    return v ? py::object(py::int_(*v)) : py::object(py::none());
}

void bind(py::module_& q) {
    py::module_ pic = py::hasattr(q, "picture") ? py::reinterpret_borrow<py::module_>(q.attr("picture"))
                                                : q.def_submodule("picture", "Port of sstvae.qrss.picture");
    py::module_ m = q.def_submodule("codec", "Port of the QRSS codec glue (native/core/qrss/codec)");

    auto codec_id_from_onnx = [](const py::object& path_or_dir, const py::object& precision) {
        const std::string p = fspath(path_or_dir);
        const std::string prec = str_of(precision);
        return translated([&] { return optional_int(qc::codec_id_from_onnx(p, prec)); });
    };
    pic.def("codec_id_from_onnx", codec_id_from_onnx, py::arg("path_or_dir"), py::arg("precision") = "fp16");
    m.def("codec_id_from_onnx", codec_id_from_onnx, py::arg("path_or_dir"), py::arg("precision") = "fp16");

    m.def("source_sha256", [](const py::object& path) {
        const std::string p = fspath(path);
        return translated([&]() -> py::object {
            const auto sha = qc::source_sha256(p);
            return sha ? py::object(str_from_utf8(*sha)) : py::object(py::none());
        });
    }, py::arg("path"));

    m.def("codec_id_of_sha", [](const py::str& sha) { return optional_int(qc::codec_id_of_sha(utf8_of(sha))); },
          py::arg("sha"));

    // checkpoint.resolve_onnx(part, path=None, precision="fp16"), with an
    // explicit path (SystemExit with the reference's texts). No path is the
    // native cache, which is not the reference's.
    m.def("resolve_onnx", [](const std::string& part, const py::object& path, const std::string& precision) {
        const auto p = opt_path(path);
        return translated([&] { return qc::resolve_onnx(part, p, precision); });
    }, py::arg("part"), py::arg("path") = py::none(), py::arg("precision") = "fp16");

    // Python's int(text, base) for base 10 / 16: (str(value), value or
    // None past int64), or None where int() raises ValueError.
    m.def("_py_int", [](const py::str& text, int base) -> py::object {
        const auto v = qc::py_int(utf8_of(text), base);
        if (!v) return py::none();
        return py::make_tuple(v->decimal, optional_int(v->value));
    }, py::arg("text"), py::arg("base"));
    m.def("_invalid_int_literal", [](const py::str& text, int base) {
        return str_from_utf8(qc::invalid_int_literal(utf8_of(text), base));
    }, py::arg("text"), py::arg("base"));
    m.def("_pathlib_str", [](const py::str& text) { return str_from_utf8(qc::pathlib_str(utf8_of(text))); },
          py::arg("text"));

    // images.load_image(path): (3, 480, 640) float32 in [0, 1].
    m.def("load_image", [](const py::object& path) {
        const std::string p = fspath(path);
        sstvae::images::ImageArray a;
        {
            py::gil_scoped_release unlock;
            a = qc::load_image(p);
        }
        py::array_t<float> out({py::ssize_t{3}, static_cast<py::ssize_t>(a.height), static_cast<py::ssize_t>(a.width)});
        std::copy(a.chw.begin(), a.chw.end(), out.mutable_data());
        return out;
    }, py::arg("path"));

    // qrss_encode.py's encode: the image file -> StoredPicture fields, with
    // the encoder from --model / --precision (None: the native cache).
    m.def("encode_picture", [](const py::object& image, const py::object& mode, const py::object& codec_id,
                               const py::object& model, const std::string& precision) {
        const std::string img = fspath(image);
        const std::int64_t mm = pc::check_mode(int64_of(mode), str_of(mode));
        const std::int64_t cid = int64_of(codec_id);
        pc::check_codec_id(cid, str_of(codec_id));
        const auto mdl = opt_path(model);
        return translated([&] {
            pc::StoredPicture sp;
            {
                py::gil_scoped_release unlock;
                sp = qc::encode_picture(img, mm, cid, qc::onnx_encoder(mdl, precision));
            }
            return picture_dict(sp);
        });
    }, py::arg("image"), py::arg("mode"), py::arg("codec_id"), py::arg("model") = py::none(),
       py::arg("precision") = "fp16");
}

const Registrar reg("codec", 95, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
