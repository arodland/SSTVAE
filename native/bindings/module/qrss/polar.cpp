// sstvae_native.qrss.polar: bindings of native/core/qrss/polar.
// Signatures, dtypes and ValueError texts follow sstvae/qrss/polar.py, so
// tests/qrss_native/polar.py (encoder) and polar_rx.py (decoder) can assign
// these over the reference's functions. Inputs go through the same numpy
// conversions the reference makes (np.array/np.asarray), so what is refused
// -- and numpy's own errors for a shape the reference cannot reshape or
// index -- matches.

#include "qrss/registry.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/polar.hpp"
#include "qrss/polar_util.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace pol = sstvae::qrss::polar;
using namespace polar_util;

std::string tuple_text(const std::vector<py::ssize_t>& dims) {
    // numpy's reshape message prints a shape as "(a,b,c)", no spaces.
    std::string s = "(";
    for (std::size_t i = 0; i < dims.size(); ++i) s += (i ? "," : "") + std::to_string(dims[i]);
    return s + ")";
}

std::vector<double> coded_llrs(const py::object& llr) {
    return float_vector(llr, pol::E, "expected " + std::to_string(pol::E) + " coded LLRs");
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("polar", "Port of sstvae.qrss.polar");
    m.attr("N") = pol::N;
    m.attr("K") = pol::K;
    m.attr("E") = pol::E;
    m.attr("N_STAGES") = pol::N_STAGES;
    py::tuple info(pol::K);
    for (int t = 0; t < pol::K; ++t) info[static_cast<std::size_t>(t)] = pol::INFO_SET[static_cast<std::size_t>(t)];
    m.attr("INFO_SET") = info;
    m.def("info_set_ok", &pol::info_set_ok);

    // Any leading shape, over the last axis; values XORed as bytes.
    m.def("polar_transform", [](const py::object& v) {
        py::array x = numpy().attr("array")(v, py::arg("dtype") = numpy().attr("uint8"), py::arg("copy") = true);
        if (x.ndim() == 0) throw py::index_error("tuple index out of range");
        std::vector<py::ssize_t> shape(x.shape(), x.shape() + x.ndim());
        const py::ssize_t n = shape.back();
        // A size-0 array reshapes into any size-0 shape, so only a
        // non-empty one can fail the reference's reshape.
        for (py::ssize_t h = n / 2; h >= 1 && x.size() != 0; h /= 2)
            if (n % (2 * h)) {
                std::vector<py::ssize_t> dims(shape.begin(), shape.end() - 1);
                dims.insert(dims.end(), {n / (2 * h), 2, h});
                throw py::value_error("cannot reshape array of size " + std::to_string(x.size()) +
                                      " into shape " + tuple_text(dims));
            }
        auto c = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>::ensure(x);
        py::array_t<std::uint8_t> out(shape);
        if (c.size() != 0) std::memcpy(out.mutable_data(), c.data(), static_cast<std::size_t>(c.size()));
        pol::polar_transform_rows(std::span<std::uint8_t>(out.mutable_data(), static_cast<std::size_t>(out.size())),
                                  static_cast<std::size_t>(n));
        return out;
    }, py::arg("v"));

    // np.asarray(x)[..., _RM]: any dtype, any leading shape.
    m.def("rate_match", [](const py::object& x) -> py::object {
        py::array a = numpy().attr("asarray")(x);
        if (a.ndim() == 0)
            throw py::index_error("too many indices for array: array is 0-dimensional, but 1 were indexed");
        const py::ssize_t n = a.shape(a.ndim() - 1);
        if (n < pol::N)
            throw py::index_error("index " + std::to_string(n) + " is out of bounds for axis " +
                                  std::to_string(a.ndim() - 1) + " with size " + std::to_string(n));
        std::vector<py::ssize_t> shape(a.shape(), a.shape() + a.ndim());
        shape.back() = pol::E;
        if (a.dtype().attr("hasobject").cast<bool>()) {
            // References, not bytes: let numpy copy them, through our index map.
            py::list idx;
            for (int e = 0; e < pol::E; ++e) idx.append(pol::mother_index(e));
            return a.attr("take")(idx, py::arg("axis") = -1);
        }
        py::array c = numpy().attr("ascontiguousarray")(a);
        py::array out(c.dtype(), shape);
        const std::size_t item = static_cast<std::size_t>(c.itemsize());
        const std::size_t rows = static_cast<std::size_t>(out.size()) / static_cast<std::size_t>(pol::E);
        const auto* src = static_cast<const std::uint8_t*>(c.data());
        auto* dst = static_cast<std::uint8_t*>(out.mutable_data());
        for (std::size_t r = 0; r < rows; ++r)
            for (int e = 0; e < pol::E; ++e)
                std::memcpy(dst + (r * pol::E + static_cast<std::size_t>(e)) * item,
                            src + (r * static_cast<std::size_t>(n) + static_cast<std::size_t>(pol::mother_index(e))) * item,
                            item);
        return out;
    }, py::arg("x"));

    m.def("fold", [](const py::object& llr) { return to_numpy(pol::fold(coded_llrs(llr))); }, py::arg("llr"));

    m.def("polar_encode", [](const py::object& info) {
        py::array a = numpy().attr("asarray")(info);
        if (a.ndim() != 1 || a.shape(0) != pol::K)
            throw py::value_error("expected " + std::to_string(pol::K) + " information bits, got shape " +
                                  shape_str(a));
        auto bits = bits01(a);
        if (!bits) throw py::value_error("information bits must be 0 or 1");
        return to_numpy(pol::polar_encode(*bits));
    }, py::arg("info"));

    m.def("decode_list", [](const py::object& llr, const py::object& list_size) {
        const DecodeArgs a = decode_args(llr, list_size);
        const pol::ListResult r = pol::decode_list(a.llr, a.list_size);
        return py::make_tuple(to_numpy_2d(r.info, static_cast<py::ssize_t>(r.m), pol::K), to_numpy(r.pm));
    }, py::arg("llr"), py::arg("list_size") = 8);

    // accept=None is the header's own check, as in the reference. A Python
    // `accept` sees each path as an array, and the one it accepts is the
    // array returned -- the same object, as the reference returns the row
    // it handed over (so whatever `accept` did to it shows).
    m.def("polar_decode", [](const py::object& llr, const py::object& list_size, const py::object& accept) -> py::object {
        const DecodeArgs a = decode_args(llr, list_size);
        pol::Accept acc;
        py::object accepted;
        if (!accept.is_none())
            acc = [&accept, &accepted](std::span<const std::uint8_t> bits) {
                py::object row = to_numpy(bits);
                const bool ok = truthy(accept(row));
                if (ok) accepted = row;
                return ok;
            };
        const auto bits = pol::polar_decode(a.llr, a.list_size, acc);
        if (!bits) return py::none();
        if (accepted) return accepted;
        return to_numpy(*bits);
    }, py::arg("llr"), py::arg("list_size") = 8, py::arg("accept") = py::none());
}

const Registrar reg("polar", 50, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
