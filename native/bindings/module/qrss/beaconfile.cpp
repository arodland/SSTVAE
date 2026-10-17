// sstvae_native.qrss.beaconfile: bindings of native/core/qrss/beaconfile.
// Names, signatures and ValueError texts follow sstvae/qrss/beaconfile.py.
// BeaconFile, StoredPicture and HeaderFields stay the reference's
// dataclasses: these take any object with their attributes and return a
// BeaconFile's fields as a dict (HeaderFields' as a tuple), which
// tests/qrss_native/beaconfile.py wraps (see picture_chain.hpp). Files are
// opened with Python's open(), so a bad path fails exactly as there.

#include "qrss/registry.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "qrss/beaconfile.hpp"
#include "qrss/picture_chain.hpp"

namespace sstvae_bindings::qrss {
namespace {

using namespace picture_chain;

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("beaconfile", "Port of sstvae.qrss.beaconfile");
    m.attr("MAGIC") = py::bytes(std::string(bfm::MAGIC));
    m.attr("FILE_VERSION") = bfm::FILE_VERSION;
    m.attr("FLAG_OOK") = bfm::FLAG_OOK;
    m.attr("HEAD_BYTES") = bfm::HEAD_BYTES;
    m.attr("KEYING_BYTES") = bfm::KEYING_BYTES;
    m.attr("HDR_BYTES") = bfm::HDR_BYTES;
    m.attr("LATENT_OFFSET") = bfm::LATENT_OFFSET;

    // LATENT_OFFSET + int(n_latents) + 4: the C++ in int64, Python's own
    // ints past it (the reference's int has no limit).
    m.def("file_size", [](const py::object& n_latents) -> py::object {
        const py::int_ n = py::reinterpret_borrow<py::int_>(steal(PyNumber_Long(n_latents.ptr())));
        const std::int64_t v = clamp_i64(n);
        if (v < std::numeric_limits<std::int64_t>::max() - static_cast<std::int64_t>(bfm::LATENT_OFFSET + 4) &&
            v > std::numeric_limits<std::int64_t>::min())
            return py::int_(bfm::file_size(v));
        return steal(PyNumber_Add(steal(PyNumber_Add(py::int_(bfm::LATENT_OFFSET).ptr(), n.ptr())).ptr(),
                                  py::int_(4).ptr()));
    }, py::arg("n_latents") = pc::SENT);

    // BeaconFile.header_fields(self) -> HeaderFields' eight fields.
    m.def("header_fields", [](const py::object& self) { return header_tuple(beacon_of(self).header_fields()); },
          py::arg("self"));

    m.def("check", [](const py::object& self) { (void)checked_beacon(self); }, py::arg("self"));

    m.def("to_bytes", [](const py::object& self) { return as_bytes(checked_beacon(self).to_bytes()); },
          py::arg("self"));

    // BeaconFile.from_bytes(data) -> the fields.
    m.def("from_bytes", [](const py::object& data) { return beacon_dict(bfm::BeaconFile::from_bytes(bytes_of(data))); },
          py::arg("data"));

    // write(path, bf): the file opened (and truncated) first, then the bytes,
    // as the reference -- a refused BeaconFile leaves an empty file.
    m.def("write", [](const py::object& path, const py::object& bf) {
        py::object f = builtin("open")(path, "wb");
        try {
            f.attr("write")(as_bytes(checked_beacon(bf).to_bytes()));
        } catch (...) {
            f.attr("close")();
            throw;
        }
        f.attr("close")();
    }, py::arg("path"), py::arg("bf"));

    m.def("read", [](const py::object& path) { return beacon_dict(bfm::BeaconFile::from_bytes(read_file(path))); },
          py::arg("path"));

    // from_picture(sp, segment, header, ook=False) -> the fields.
    m.def("from_picture", [](const py::object& sp, const py::object& segment, const py::object& header,
                             const py::object& ook) {
        const pc::StoredPicture s = checked_picture(sp);   // sp.check() first
        const std::int64_t seg = int64_of(segment);
        return beacon_dict(bfm::from_picture(s, seg, header_of(header), truthy(ook)));
    }, py::arg("sp"), py::arg("segment"), py::arg("header"), py::arg("ook") = false);

    // Python's repr() of bytes, as the refusals print it (for the parity test).
    m.def("_bytes_repr", [](const py::bytes& b) {
        const std::string s = b;
        return bfm::bytes_repr(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()));
    }, py::arg("b"));
}

const Registrar reg("beaconfile", 61, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
