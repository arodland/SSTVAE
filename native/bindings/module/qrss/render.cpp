// sstvae_native.qrss.render: bindings of native/core/qrss/render (the
// codec boundary's arithmetic). Names, arguments and texts follow
// sstvae/qrss/render.py, so tests/qrss_native/render.py can assign these
// over the reference's:
//
//   g_shrink(W) -> float64, W's shape
//   reference_w(W, mode) -> float
//   decoder_planes(S, W, mode) -> (latents float32[158400], weights float32[158400])
//
// S and W are anything np.asarray(x, float64) takes, of any shape with
// 3 * 52800 elements (the reference reshapes them). render(),
// render_accumulator() and the model lookup (the codec's side) are not here.

#include "qrss/registry.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "qrss/render.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace rd = sstvae::qrss::render;

const py::object& np_mod() {
    static py::object* np = new py::object(py::module_::import("numpy"));
    return *np;
}

// int(x) as int64.
std::int64_t int_of(const py::handle& h) {
    py::object i = py::reinterpret_steal<py::object>(PyNumber_Long(h.ptr()));
    if (!i) throw py::error_already_set();
    const long long v = PyLong_AsLongLong(i.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

py::array float64(const py::handle& x) { return np_mod().attr("asarray")(x, py::arg("dtype") = "float64"); }

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("render", "Port of sstvae.qrss.render's decoder planes");
    m.attr("G_MAX") = rd::G_MAX;
    m.attr("G_KNEE") = rd::G_KNEE;
    m.attr("N_GROUPS") = rd::N_GROUPS;
    m.attr("GROUP_LATENTS") = rd::GROUP_LATENTS;

    m.def("g_shrink", [](const py::object& W) -> py::object {
        const py::array a = float64(W);
        const std::vector<double> w = from_numpy<double>(a);
        const std::vector<double> g = rd::g_shrink(w);
        py::array_t<double> out(std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim()));
        std::copy(g.begin(), g.end(), out.mutable_data());
        if (a.ndim() == 0) return out[py::tuple()];
        return out;
    }, py::arg("W"));

    m.def("reference_w", [](const py::object& W, const py::object& mode) {
        const std::vector<double> w = from_numpy<double>(float64(W));
        return rd::reference_w(w, int_of(mode));
    }, py::arg("W"), py::arg("mode"));

    m.def("decoder_planes", [](const py::object& S, const py::object& W, const py::object& mode) {
        const std::int64_t md = int_of(mode);
        if (md < 0 || md >= rd::N_GROUPS) {   // before S and W are converted, as the reference
            const std::vector<double> none;
            (void)rd::decoder_planes(std::span<const double>(none), std::span<const double>(none), md);
        }
        const std::vector<double> s = from_numpy<double>(float64(S));
        const std::vector<double> w = from_numpy<double>(float64(W));
        const auto [lat, wt] = rd::decoder_planes(std::span<const double>(s), std::span<const double>(w), md);
        return py::make_tuple(to_numpy(lat), to_numpy(wt));
    }, py::arg("S"), py::arg("W"), py::arg("mode"));
}

const Registrar reg("render", 74, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
