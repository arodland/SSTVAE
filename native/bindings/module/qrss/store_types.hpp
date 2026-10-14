// Conversions for the multi-pass store's bindings (store.cpp, associate.cpp):
// JSON values (pyjson::Value) to and from Python's json types, accumulator
// keys, the store's records as the dicts store.py returns, npz arrays, and
// the Python-side Store object (`PyStore`: the core Store plus the Python
// seams it calls).
//
// JSON, Python -> C++ (`to_value`) reads what json.dumps writes: None, bool,
// int (int64: OverflowError past it, where Python's are unbounded), float
// (and its subclasses, np.float64 among them), str, list and tuple, dict
// (keys converted as json.dumps converts them: str as is, a float by its
// repr, True/False/None as "true"/"false"/"null", an int in decimal; any
// other key is TypeError "keys must be str, int, float, bool or None, not
// X"). Anything else is TypeError "Object of type X is not JSON
// serializable", or with `Unknown::as_float` float(x), as json.dumps(...,
// default=float) does for the log (an np.int64, an np.float32, an np.bool_
// all become floats there, which is why a merge's "accepted" is 1.0 or 0.0).
// C++ -> Python (`to_py`) gives what json.loads gives.
//
// Errors raise as store.py's would: npz::FileError (a file that cannot be
// opened, read, written, synced, replaced or removed) becomes OSError
// (errno, strerror, file name) as the errno picks its subclass
// (FileNotFoundError, PermissionError, ...); pyjson::KeyError KeyError (the
// translators rx_types.hpp registers); std::invalid_argument ValueError.
//
// THREADS and the GIL: every binding releases the GIL around the core call
// (`nogil`) and builds no Python object while it is released; the seams
// (clock, replace, zw_cache, soft_header) take the GIL back to call Python.
// Nothing holds the GIL while waiting on a Store's mutex, so a thread inside
// the store (mutex held, waiting for the GIL in a seam) and a thread calling
// it (GIL released before the mutex is taken) cannot deadlock.

#pragma once

#include "qrss/rx_types.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "qrss/fspath.hpp"
#include "qrss/npz.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/store.hpp"

namespace sstvae_bindings::qrss::st {

namespace store = sstvae::qrss::store;
namespace pyjson = sstvae::qrss::pyjson;
namespace npz = sstvae::qrss::npz;
using Value = pyjson::Value;

// --- errors ---------------------------------------------------------------------

[[noreturn]] inline void raise_file_error(const npz::FileError& e) {
    fspath::raise_os_error(e.err ? e.err : EIO, fspath::fs_text(e.path));
}

// f() with the GIL released; a FileError raised as OSError (with the GIL
// held again). f must build no Python object.
template <typename F>
auto nogil(F&& f) -> decltype(f()) {
    try {
        py::gil_scoped_release release;
        return f();
    } catch (const npz::FileError& e) {
        raise_file_error(e);
    }
}

// --- JSON values ----------------------------------------------------------------

enum class Unknown { refuse, as_float };

inline std::string class_name(const py::handle& o) {
    return py::str(py::type::of(o).attr("__name__")).cast<std::string>();
}

inline std::string float_repr(double x) {
    if (std::isnan(x)) return "NaN";
    if (std::isinf(x)) return x > 0 ? "Infinity" : "-Infinity";
    return py::repr(py::float_(x)).cast<std::string>();
}

inline std::string json_key(const py::handle& k) {
    if (PyUnicode_Check(k.ptr())) return rx::as_string(k);
    if (PyFloat_Check(k.ptr())) return float_repr(PyFloat_AsDouble(k.ptr()));
    if (k.ptr() == Py_True) return "true";
    if (k.ptr() == Py_False) return "false";
    if (k.is_none()) return "null";
    if (PyLong_Check(k.ptr())) return py::str(py::reinterpret_steal<py::object>(PyNumber_Long(k.ptr()))).cast<std::string>();
    throw py::type_error("keys must be str, int, float, bool or None, not " + class_name(k));
}

inline Value to_value(const py::handle& o, Unknown unknown = Unknown::refuse, int depth = 0) {
    if (depth > 900) throw py::value_error("Circular reference detected");
    PyObject* p = o.ptr();
    if (o.is_none()) return Value::null();
    if (p == Py_True) return Value::boolean(true);
    if (p == Py_False) return Value::boolean(false);
    if (PyLong_Check(p)) {
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(p, &overflow);
        if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
        if (overflow) throw std::overflow_error("int too large for a JSON value here (int64)");
        return Value::integer(v);
    }
    if (PyFloat_Check(p)) return Value::number(PyFloat_AsDouble(p));
    if (PyUnicode_Check(p)) return Value::string(rx::as_string(o));
    if (PyList_Check(p) || PyTuple_Check(p)) {
        std::vector<Value> items;
        for (const py::handle h : o) items.push_back(to_value(h, unknown, depth + 1));
        return Value::array(std::move(items));
    }
    if (PyDict_Check(p)) {
        Value d = Value::object();
        for (const auto& [k, v] : py::reinterpret_borrow<py::dict>(o)) d.set(json_key(k), to_value(v, unknown, depth + 1));
        return d;
    }
    if (unknown == Unknown::as_float) return Value::number(rx::as_double(o));
    throw py::type_error("Object of type " + class_name(o) + " is not JSON serializable");
}

inline py::object to_py(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Null: return py::none();
        case Value::Kind::Bool: return py::bool_(v.as_bool());
        case Value::Kind::Int: return py::int_(v.as_int());
        case Value::Kind::Float: return py::float_(v.as_double());
        case Value::Kind::String: return rx::to_str(v.as_string());
        case Value::Kind::Array: {
            py::list out;
            for (const Value& x : v.items()) out.append(to_py(x));
            return out;
        }
        case Value::Kind::Object: {
            py::dict out;
            for (std::size_t k = 0; k < v.keys().size(); ++k) out[rx::to_str(v.keys()[k])] = to_py(v.values()[k]);
            return out;
        }
    }
    return py::none();
}

inline py::list to_py(const std::vector<Value>& v) {
    py::list out;
    for (const Value& x : v) out.append(to_py(x));
    return out;
}

inline std::vector<Value> values_of(const py::handle& o) {
    std::vector<Value> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(o))) out.push_back(to_value(h));
    return out;
}

inline py::object opt_py(const std::optional<Value>& v) { return v ? to_py(*v) : py::object(py::none()); }

// --- keys -----------------------------------------------------------------------

// `call, pid = key` then call.replace('/', '_') and int(pid) (key_name):
// Python's unpacking and attribute errors.
inline store::Key key_of(const py::handle& k) {
    PyObject* it = PyObject_GetIter(k.ptr());
    if (!it) {
        PyErr_Clear();
        throw py::type_error("cannot unpack non-iterable " + class_name(k) + " object");
    }
    const py::object iter = py::reinterpret_steal<py::object>(it);
    std::vector<py::object> parts;
    while (PyObject* x = PyIter_Next(iter.ptr())) {
        parts.push_back(py::reinterpret_steal<py::object>(x));
        if (parts.size() > 2) throw py::value_error("too many values to unpack (expected 2)");
    }
    if (PyErr_Occurred()) throw py::error_already_set();
    if (parts.size() < 2)
        throw py::value_error("not enough values to unpack (expected 2, got " + std::to_string(parts.size()) + ")");
    if (PyBytes_Check(parts[0].ptr()) || PyByteArray_Check(parts[0].ptr()))
        throw py::type_error("a bytes-like object is required, not 'str'");   // bytes.replace('/', '_')
    if (!PyUnicode_Check(parts[0].ptr()))
        throw py::attribute_error("'" + class_name(parts[0]) + "' object has no attribute 'replace'");
    return {rx::as_string(parts[0]), rx::as_int(parts[1])};
}

// (str(key[0]), int(key[1])): Accumulator.empty's reading.
inline store::Key key_index(const py::handle& k) {
    const py::object o = py::reinterpret_borrow<py::object>(k);
    return {rx::as_string(py::str(o[py::int_(0)])), rx::as_int(o[py::int_(1)])};
}

inline py::tuple key_py(const store::Key& k) { return py::make_tuple(rx::to_str(k.callsign), k.picture_id); }

inline py::object opt_key(const std::optional<store::Key>& k) { return k ? py::object(key_py(*k)) : py::object(py::none()); }

inline py::list keys_py(const std::vector<store::Key>& v) {
    py::list out;
    for (const store::Key& k : v) out.append(key_py(k));
    return out;
}

// --- arrays ---------------------------------------------------------------------

// A pathlib.Path of a file name the core spells in bytes.
inline py::object path_py(const std::string& bytes) {
    static py::object* cls = new py::object(py::module_::import("pathlib").attr("Path"));
    return (*cls)(fspath::fs_text(bytes));
}

inline npz::Array npz_array(const py::handle& obj) {
    py::array a = rx::np_module().attr("ascontiguousarray")(obj);
    const std::string d = py::str(a.attr("dtype").attr("str")).cast<std::string>();
    const auto t = npz::from_descr(d);
    if (!t) throw py::value_error("a foreign part's dtype '" + d + "' is not one the store can write");
    npz::Array out;
    out.dtype = *t;
    for (py::ssize_t k = 0; k < a.ndim(); ++k) out.shape.push_back(static_cast<std::size_t>(a.shape(k)));
    const std::string raw = a.attr("tobytes")().cast<std::string>();
    out.bytes.assign(raw.begin(), raw.end());
    return out;
}

inline py::array npz_py(const npz::Array& a) {
    std::vector<py::ssize_t> shape(a.shape.begin(), a.shape.end());
    return py::array(py::dtype(std::string(npz::descr(a.dtype))), shape, a.bytes.data());
}

// A float32 plane (3, 52800) viewing v, kept alive by `base` (the Accumulator).
inline py::array plane_view(std::vector<float>& v, const py::handle& base) {
    if (v.size() != store::PLANE) {
        py::array_t<float> out(static_cast<py::ssize_t>(v.size()));
        std::copy(v.begin(), v.end(), out.mutable_data());
        return out;
    }
    return py::array(py::dtype::of<float>(),
                     std::vector<py::ssize_t>{store::N_GROUPS, store::GROUP_LATENTS},
                     std::vector<py::ssize_t>{static_cast<py::ssize_t>(store::GROUP_LATENTS * sizeof(float)),
                                              static_cast<py::ssize_t>(sizeof(float))},
                     v.data(), base);
}

// np.asarray(o, float32) holding PLANE values, copied into v in place (so
// views of v stay valid); numpy's reshape refusal otherwise.
inline void set_plane(std::vector<float>& v, const py::handle& o) {
    const std::vector<float> x = rx::vec<float>(o);
    if (x.size() != store::PLANE)
        throw py::value_error("cannot reshape array of size " + std::to_string(x.size()) + " into shape (" +
                              std::to_string(store::N_GROUPS) + "," + std::to_string(store::GROUP_LATENTS) + ")");
    if (v.size() != store::PLANE) v.resize(store::PLANE);
    std::copy(x.begin(), x.end(), v.begin());
}

inline py::array plane_py(const std::vector<double>& v) {
    py::array_t<double> out(std::vector<py::ssize_t>{store::N_GROUPS, store::GROUP_LATENTS});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

// --- the store's records ----------------------------------------------------------

inline py::dict info_py(const store::PassInfo& i) {
    py::dict d;
    d["uid"] = rx::to_str(i.uid);
    d["q"] = i.q;
    d["f_hz"] = i.f_hz;
    d["header"] = to_py(i.header);
    d["hdr_llr"] = rx::arr(i.hdr_llr);
    d["z_ref"] = i.z_ref;
    d["em_round"] = i.em_round;
    d["waveform"] = i.waveform;
    d["frame"] = to_py(i.frame);
    return d;
}

inline py::dict due_py(const store::Due& r) {
    py::dict d;
    d["key"] = key_py(r.key);
    d["segment"] = r.segment;
    d["f_hz"] = r.f_hz;
    d["q"] = r.q;
    d["waveform"] = r.waveform;
    d["frame"] = to_py(r.frame);
    return d;
}

// --- the Python Store -------------------------------------------------------------

// sstvae_native.qrss.store.Store: the core Store and the Python objects its
// seams call. `core` is declared last so it is destroyed first: its seams
// point back at this object.
struct PyStore {
    py::object clock;      // called for every timestamp (time.time by default)
    py::object replace;    // None: the C++ os.replace; else called as replace(src, dst)
    py::object zw_cache;   // None: ZW_CACHE; an int; or called for the cache size at each read
    std::unique_ptr<store::Store> core;
};

}  // namespace sstvae_bindings::qrss::st
