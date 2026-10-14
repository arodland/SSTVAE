// A path argument as Python's open() takes one, for the bindings that hand
// a file name to the C++ (native/core/util/fileio.hpp opens it).
//
// open(path) accepts a str, bytes or an os.PathLike, and turns it into the
// bytes the system gets with os.fsencode: on POSIX the file-system encoding
// with "surrogateescape" (so a str holding the surrogate escapes of a name
// that is not UTF-8 means exactly that name's bytes), on Windows UTF-8 with
// "surrogatepass" (PEP 529), which fileio.hpp turns into UTF-16 the same way.
// That is fs_path: os.fspath, then PyUnicode_FSConverter, which is what
// open() itself calls, so the refusals are open()'s: TypeError "expected
// str, bytes or os.PathLike object, not int", ValueError "embedded null
// byte". pybind11's std::string caster is not this: it decodes a str
// strictly (a surrogate escape is a TypeError) and takes no PathLike.
//
// A refusal names the file as Python's OSError does, by os.fspath(path)
// (`name`: a str, or bytes for bytes), and a message naming it in text
// decodes the bytes with os.fsdecode (fs_text), never strictly.

#ifndef SSTVAE_BINDINGS_QRSS_FSPATH_HPP
#define SSTVAE_BINDINGS_QRSS_FSPATH_HPP

#include <pybind11/pybind11.h>

#include <cstring>
#include <string>

namespace sstvae_bindings::qrss::fspath {

namespace py = pybind11;

struct FsPath {
    py::object name;     // os.fspath(obj): what an OSError's filename is
    std::string bytes;   // os.fsencode(name): what the C++ opens
};

// os.fspath(obj) and os.fsencode of it, as open(obj) reads its argument.
inline FsPath fs_path(const py::handle& obj) {
    PyObject* p = PyOS_FSPath(obj.ptr());
    if (!p) throw py::error_already_set();
    FsPath out;
    out.name = py::reinterpret_steal<py::object>(p);
    PyObject* enc = nullptr;
    if (!PyUnicode_FSConverter(out.name.ptr(), &enc)) throw py::error_already_set();
    const py::object b = py::reinterpret_steal<py::object>(enc);
    char* data = nullptr;
    Py_ssize_t n = 0;
    if (PyBytes_AsStringAndSize(b.ptr(), &data, &n) != 0) throw py::error_already_set();
    out.bytes.assign(data, static_cast<std::size_t>(n));
    return out;
}

// pathlib.Path(obj)'s reading of its argument, then os.fsencode: os.fspath
// of it (obj itself when that refuses) must be a str, else pathlib's
// TypeError "argument should be a str or an os.PathLike object where
// __fspath__ returns a str, not 'bytes'". For the references that make a
// Path of what they are given (checkpoint.resolve_onnx, codec_id_from_onnx).
inline FsPath pathlib_path(const py::handle& obj) {
    py::object p;
    if (PyObject* r = PyOS_FSPath(obj.ptr())) {
        p = py::reinterpret_steal<py::object>(r);
    } else if (PyErr_ExceptionMatches(PyExc_TypeError)) {
        PyErr_Clear();
        p = py::reinterpret_borrow<py::object>(obj);
    } else {
        throw py::error_already_set();
    }
    if (!PyUnicode_Check(p.ptr())) {
        PyErr_Format(PyExc_TypeError,
                     "argument should be a str or an os.PathLike object where __fspath__ returns a str, not %R",
                     py::str(py::type::of(p).attr("__name__")).ptr());
        throw py::error_already_set();
    }
    return fs_path(p);
}

// fs_path(str(obj)): a reference that calls str(path) before opening it
// (receiver.capture_from_wav), so a bytes path is its repr "b'...'" there
// and here alike.
inline FsPath str_path(const py::handle& obj) { return fs_path(py::str(obj)); }

// os.fsdecode(bytes): a file name as text, for a message.
inline py::str fs_text(const std::string& bytes) {
    PyObject* s = PyUnicode_DecodeFSDefaultAndSize(bytes.data(), static_cast<Py_ssize_t>(bytes.size()));
    if (!s) throw py::error_already_set();
    return py::reinterpret_steal<py::str>(s);
}

// OSError(err, strerror(err), name), raised as the subclass the errno picks
// (FileNotFoundError, IsADirectoryError, ...): what open() raises.
[[noreturn]] inline void raise_os_error(int err, const py::handle& name) {
    PyObject* exc = PyObject_CallFunction(PyExc_OSError, "isO", err, std::strerror(err), name.ptr());
    if (!exc) throw py::error_already_set();
    PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(exc)), exc);
    Py_DECREF(exc);
    throw py::error_already_set();
}

// `type`(text) for a C++ message that may name a file by its bytes: the
// whole text decoded with os.fsdecode, which is UTF-8's strict decoding
// wherever the text is UTF-8 and gives a name that is not its own str back.
[[noreturn]] inline void raise_text(PyObject* type, const std::string& text) {
    const py::str t = fs_text(text);
    PyErr_SetObject(type, t.ptr());
    throw py::error_already_set();
}

// RuntimeError(text): a C++ refusal naming a file, its name decoded with
// os.fsdecode rather than as strict UTF-8 (which a name that is not UTF-8
// would turn into a UnicodeDecodeError raised from the error path).
[[noreturn]] inline void raise_runtime_error(const std::string& before, const std::string& path_bytes,
                                             const std::string& after = {}) {
    const py::str name = fs_text(path_bytes);
    PyObject* text = PyUnicode_FromFormat("%s%U%s", before.c_str(), name.ptr(), after.c_str());
    if (!text) throw py::error_already_set();
    PyErr_SetObject(PyExc_RuntimeError, text);
    Py_DECREF(text);
    throw py::error_already_set();
}

}  // namespace sstvae_bindings::qrss::fspath

#endif  // SSTVAE_BINDINGS_QRSS_FSPATH_HPP
