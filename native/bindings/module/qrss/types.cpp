// sstvae_native.qrss.types: bindings of native/core/qrss/types (the
// receiver's shared records, sstvae/qrss/types.py).
//
// The records cross as rx_types.hpp converts them: in, any object with
// the reference dataclass's attributes (or a dict of them); out, plain
// dicts with a "__type__" key that tests/qrss_native/_rx.py's `to_ref`
// turns back into the reference's dataclasses.
//
//   pass_uid(q, f_hz, z) -> str                  types.pass_uid
//   save_pass(p, path)                           PassResult.save
//   load_pass(path) -> dict                      PassResult.load
//   pass_bytes(p) -> bytes                       the .npz save writes
//   parse_pass(data) -> dict                     load of those bytes
//   meta_json(p) -> str                          json.dumps(meta) as save builds it
//   roundtrip(obj, kind=None) -> dict            Python -> C++ -> Python, for the
//                                                parity tests (kind: the type's
//                                                name, else obj's class name)

#include "qrss/rx_types.hpp"

#include "qrss/fspath.hpp"
#include "qrss/npz.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sstvae_bindings::qrss {
namespace {

namespace t = sstvae::qrss::types;

py::bytes as_bytes(const std::vector<std::uint8_t>& v) {
    return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
}

std::vector<std::uint8_t> bytes_of(const py::handle& o) {
    py::buffer b = py::reinterpret_borrow<py::buffer>(o);
    const py::buffer_info info = b.request();
    if (info.ndim > 1 || (info.ndim == 1 && info.itemsize != 1 && info.size != 0))
        throw py::type_error("expected a bytes-like object");
    const auto* p = static_cast<const std::uint8_t*>(info.ptr);
    return std::vector<std::uint8_t>(p, p + info.size * info.itemsize);
}

py::object roundtrip(const py::object& obj, const py::object& kind_o) {
    const std::string kind = kind_o.is_none() ? rx::type_name(obj) : py::str(kind_o).cast<std::string>();
    if (kind == "Timing") return rx::to_py(rx::timing(obj));
    if (kind == "FreqPath") return rx::to_py(rx::freq_path(obj));
    if (kind == "CarrierCandidate") return rx::to_py(rx::carrier_candidate(obj));
    if (kind == "Detection") return rx::to_py(rx::detection(obj));
    if (kind == "TrackReport") return rx::to_py(rx::track_report(obj));
    if (kind == "CwIdResult") return rx::to_py(rx::cw_result(obj));
    if (kind == "EmPrior") return rx::to_py(rx::em_prior(obj));
    if (kind == "HeaderFields") return rx::to_py(rx::header_fields(obj));
    if (kind == "PassResult") return rx::to_py(rx::pass_result(obj));
    throw py::value_error("types.roundtrip: no conversion for " + kind);
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("types", "Port of sstvae/qrss/types.py (native/core/qrss/types)");

    m.def("pass_uid", [](const py::object& q_, double f_hz, const py::object& z) {
        const std::vector<float> zz = rx::vec<float>(z);
        return rx::to_str(t::pass_uid(rx::as_int(q_), f_hz, zz));
    }, py::arg("q"), py::arg("f_hz"), py::arg("z"));

    // a path as open() takes one (fspath.hpp); a file the C++ cannot open,
    // read or write is RuntimeError("cannot open <name>") (the rows raise
    // Python's own OSError from it)
    m.def("save_pass", [](const py::object& p, const py::object& path) {
        const t::PassResult r = rx::pass_result(p);
        const fspath::FsPath f = fspath::fs_path(path);
        try {
            r.save(f.bytes);
        } catch (const sstvae::qrss::npz::FileError& e) {
            fspath::raise_runtime_error(e.verb + " ", f.bytes);
        }
    }, py::arg("p"), py::arg("path"));
    m.def("load_pass", [](const py::object& path) {
        const fspath::FsPath f = fspath::fs_path(path);
        try {
            return rx::to_py(t::PassResult::load(f.bytes));
        } catch (const sstvae::qrss::npz::FileError& e) {
            fspath::raise_runtime_error(e.verb + " ", f.bytes);
        }
    }, py::arg("path"));
    m.def("pass_bytes", [](const py::object& p) { return as_bytes(rx::pass_result(p).bytes()); }, py::arg("p"));
    m.def("parse_pass", [](const py::object& data) {
        const std::vector<std::uint8_t> b = bytes_of(data);
        return rx::to_py(t::PassResult::parse(b));
    }, py::arg("data"));
    m.def("meta_json", [](const py::object& p) { return rx::to_str(rx::pass_result(p).meta_json()); },
          py::arg("p"));
    m.def("roundtrip", &roundtrip, py::arg("obj"), py::arg("kind") = py::none());
}

const Registrar reg("types", 65, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
