// sstvae_native.qrss.ce_rx: bindings of native/core/qrss/ce_rx (waveform
// CE, the receive side the receiver uses). Names, arguments, defaults and
// texts follow sstvae/qrss/ce.py, so tests/qrss_native/ce_rx.py can assign
// these over the reference's:
//
//   mean_template(tau_pos, mu, nu, spec, keying=None, ook=False,
//                 keyed_from_pos=None) -> complex128, tau_pos's shape
//   _mean_template_grid(fs, n0, n, mu, nu, spec, keying=None, ook=False)
//   matched_filter(z, fs, n0, pos, chunk=2048) (mf_rx.hpp)
//
// The transmit side's bindings (ce.cpp) are not touched; the keying
// conversion is the same as theirs (window_keying's rules).

#include "qrss/registry.hpp"

#include <pybind11/complex.h>

#include <complex>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/ce_rx.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/mf_rx.hpp"
#include "qrss/rx_types.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace ce = sstvae::qrss::ce;
namespace cr = sstvae::qrss::ce_rx;
namespace fr = sstvae::qrss::frame;
namespace cfg = sstvae::qrss::config;
using cd = std::complex<double>;

[[noreturn]] void raise(PyObject* type, const std::string& what) {
    PyErr_SetString(type, what.c_str());
    throw py::error_already_set();
}

// np.asarray(x, float64) with its shape.
struct Doubles {
    std::vector<double> v;
    std::vector<std::int64_t> shape;
};
Doubles doubles(const py::handle& obj) {
    py::array a = mf_rx::numpy_mod().attr("asarray")(obj, py::arg("dtype") = "float64");
    Doubles d;
    d.shape.assign(a.shape(), a.shape() + a.ndim());
    d.v = from_numpy<double>(a);
    return d;
}

// _check_sym(x, spec, name): its values, refusing any shape but (n_sym,).
std::vector<double> checked(const py::handle& x, const fr::FrameSpec& spec, const char* name) {
    Doubles d = doubles(x);
    ce::check_sym(spec, d.shape, name);
    return std::move(d.v);
}

// window_keying(keying, spec) as ce.cpp's binding converts it: None, else
// the validated (n_win, 192) keying.
std::optional<ce::Keying> keying_of(const py::handle& keying, const fr::FrameSpec& spec) {
    if (keying.is_none()) return std::nullopt;
    const py::object np = mf_rx::numpy_mod();
    py::array k = np.attr("asarray")(keying);
    std::vector<std::int64_t> shape(k.shape(), k.shape() + k.ndim());
    const char kind = k.dtype().kind();
    if (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f') {
        const std::vector<double> v = from_numpy<double>(np.attr("asarray")(k, py::arg("dtype") = "float64"));
        return ce::window_keying(std::span<const double>(v), shape, spec);
    }
    std::vector<std::int64_t> sh = shape;
    if (sh.size() == 1) sh = {spec.n_win(), sh[0]};
    bool ok = sh.size() == 2 && sh[0] == spec.n_win() && sh[1] == cfg::CW_UNITS;
    if (ok) {
        py::object b = np.attr("broadcast_to")(k, py::make_tuple(sh[0], sh[1]));
        auto ne = [&](int v) {
            py::object r = py::reinterpret_steal<py::object>(PyObject_RichCompare(b.ptr(), py::int_(v).ptr(), Py_NE));
            if (!r) throw py::error_already_set();
            return r;
        };
        ok = !py::bool_(np.attr("any")(ne(0) & ne(1))).cast<bool>();
    }
    if (!ok) raise(PyExc_ValueError, ce::keying_refusal(spec, shape));
    const std::vector<double> v =
        from_numpy<double>(np.attr("asarray")(k.attr("astype")("uint8"), py::arg("dtype") = "float64"));
    return ce::window_keying(std::span<const double>(v), shape, spec);
}

const ce::Keying* ptr(const std::optional<ce::Keying>& k) { return k ? &*k : nullptr; }

// v in `shape` (0-d for a scalar tau).
py::array shaped(const std::vector<cd>& v, const std::vector<std::int64_t>& shape) {
    py::array_t<cd> out(std::vector<py::ssize_t>(shape.begin(), shape.end()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("ce_rx", "Port of sstvae.qrss.ce, the receive side the receiver uses");

    m.def("mean_template", [](const py::object& tau_pos, const py::object& mu, const py::object& nu,
                              const py::object& spec, const py::object& keying, const py::object& ook,
                              const py::object& keyed_from_pos) -> py::object {
        const Doubles tau = doubles(tau_pos);
        const fr::FrameSpec s = rx::spec(spec);
        const std::vector<double> mv = checked(mu, s, "sym");
        const std::vector<double> nv = checked(nu, s, "var");
        std::optional<double> lo;
        if (!keyed_from_pos.is_none()) lo = rx::as_double(keyed_from_pos);
        const auto k = keying_of(keying, s);
        const std::vector<cd> out = cr::mean_template(tau.v, mv, nv, s, ptr(k), rx::as_bool(ook), lo);
        py::array a = shaped(out, tau.shape);
        if (tau.shape.empty()) return a[py::tuple()];   // numpy's scalar for a 0-d tau
        return a;
    }, py::arg("tau_pos"), py::arg("mu"), py::arg("nu"), py::arg("spec"), py::arg("keying") = py::none(),
       py::arg("ook") = false, py::arg("keyed_from_pos") = py::none());

    m.def("_mean_template_grid", [](const py::object& fs, const py::object& n0, const py::object& n,
                                    const py::object& mu, const py::object& nu, const py::object& spec,
                                    const py::object& keying, const py::object& ook) {
        const std::int64_t a = mf_rx::int_of(n0), b = mf_rx::int_of(n);
        const std::int64_t f = mf_rx::int_of(fs);
        ce::grid_step(f);
        const fr::FrameSpec s = rx::spec(spec);
        const std::vector<double> mv = checked(mu, s, "sym");
        const std::vector<double> nv = checked(nu, s, "var");
        const auto k = keying_of(keying, s);
        return to_numpy(cr::mean_template_grid(f, a, b, mv, nv, s, ptr(k), rx::as_bool(ook)));
    }, py::arg("fs"), py::arg("n0"), py::arg("n"), py::arg("mu"), py::arg("nu"), py::arg("spec"),
       py::arg("keying") = py::none(), py::arg("ook") = false);

    m.def("matched_filter", [](const py::object& z, const py::object& fs, const py::object& n0,
                               const py::object& pos, const py::object& chunk) {
        return mf_rx::matched_filter(
            z, fs, n0, pos, chunk, [](std::int64_t f) { return ce::grid_step(f); },
            [](std::span<const double> zz, std::int64_t f, std::int64_t a, std::span<const std::int64_t> p) {
                return cr::matched_filter(zz, f, a, p);
            },
            [](std::span<const cd> zz, std::int64_t f, std::int64_t a, std::span<const std::int64_t> p) {
                return cr::matched_filter(zz, f, a, p);
            });
    }, py::arg("z"), py::arg("fs"), py::arg("n0"), py::arg("pos"), py::arg("chunk") = 2048);
}

const Registrar reg("ce_rx", 72, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
