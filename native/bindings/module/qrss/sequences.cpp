// sstvae_native.qrss.sequences: bindings of native/core/qrss/sequences.
// Signatures, dtypes, read-only flags and ValueError texts follow
// sstvae/qrss/sequences.py, so tests/qrss_native/sequences.py can assign
// these over the reference's functions.

#include "qrss/registry.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "qrss/sequences.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace seq = sstvae::qrss::sequences;

std::size_t count(const py::int_& n) {
    if (n < py::int_(0)) throw py::value_error("n must be >= 0, not " + py::str(n).cast<std::string>());
    return n.cast<std::size_t>();
}

std::string as_bytes(const py::object& b) {
    return py::bytes(b).cast<std::string>();
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("sequences", "Port of sstvae.qrss.sequences");

    m.def("sha_bits", [](const py::object& domain, const py::int_& n, const py::object& prefix) {
        const std::string d = as_bytes(domain), p = as_bytes(prefix);
        auto bits = seq::sha_bits(d, count(n), std::span<const std::uint8_t>(
                                                   reinterpret_cast<const std::uint8_t*>(p.data()), p.size()));
        return to_numpy(bits);
    }, py::arg("domain"), py::arg("n"), py::arg("prefix") = py::bytes(""));

    // Any shape, as numpy's elementwise 1 - 2*b is.
    m.def("pm1", [](const py::object& bits) {
        auto a = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>::ensure(bits);
        if (!a) throw py::type_error("pm1 takes an array-like of bits");
        std::vector<std::uint8_t> flat(a.data(), a.data() + a.size());
        auto v = seq::pm1(flat);
        py::array_t<std::int8_t> out(std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim()));
        std::copy(v.begin(), v.end(), out.mutable_data());
        return out;
    }, py::arg("bits"));

    m.def("qpsk", [](const py::object& bits) {
        auto a = py::array::ensure(bits);
        if (!a || a.ndim() != 1 || a.size() % 2) {
            std::string shape = a ? py::str(a.attr("shape")).cast<std::string>() : "()";
            throw py::value_error("qpsk takes an even number of bits, not shape " + shape);
        }
        return to_numpy(seq::qpsk(from_numpy<std::uint8_t>(a)));
    }, py::arg("bits"));

    m.def("preamble_ce", [] { return to_numpy(seq::preamble_ce(), false); });
    m.def("references_ce", [](const py::int_& n_ref) {
        return to_numpy(seq::references_ce(count(n_ref)), false);
    }, py::arg("n_ref"));
    m.def("preamble_l", [] { return to_numpy(seq::preamble_l(), false); });
    m.def("references_l", [](const py::int_& n_ref) {
        return to_numpy(seq::references_l(count(n_ref)), false);
    }, py::arg("n_ref"));
    m.def("spread_whitener", [](const py::int_& n) {
        return to_numpy(seq::spread_whitener(count(n)), false);
    }, py::arg("n"));
    m.def("scrambler", [](const py::int_& qv, const py::int_& n) {
        const std::uint64_t q64 = to_u64(qv, "q must fit in uint64, not " + py::str(qv).cast<std::string>());
        return to_numpy(seq::scrambler(q64, count(n)));
    }, py::arg("q"), py::arg("n"));
    m.def("quarter_hour_count_unix", [](std::int64_t t) {
        try {
            return seq::quarter_hour_count(t);
        } catch (const std::invalid_argument& e) {
            throw py::value_error(e.what());
        }
    }, py::arg("unix_seconds"));
}

const Registrar reg("sequences", 10, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
