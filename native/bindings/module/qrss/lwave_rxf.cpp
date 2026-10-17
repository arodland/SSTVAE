// sstvae_native.qrss.lwave_rxf: bindings of native/core/qrss/lwave_rx
// (waveform L, the receive side the receiver uses). A module of its own so
// the transmit side's sstvae_native.qrss.lwave (lwave.cpp) is not edited;
// tests/qrss_native/lwave_rx.py assigns these over sstvae.qrss.lwave's:
//
//   PULSE_NORM                                  float, computed here
//   matched_filter(z, fs, n0, pos, chunk=2048)  (mf_rx.hpp)

#include "qrss/registry.hpp"

#include <complex>
#include <cstdint>
#include <span>

#include "qrss/lwave.hpp"
#include "qrss/lwave_rx.hpp"
#include "qrss/mf_rx.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace lw = sstvae::qrss::lwave;
namespace lr = sstvae::qrss::lwave_rx;
using cd = std::complex<double>;

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("lwave_rxf", "Port of sstvae.qrss.lwave, the receive side the receiver uses");
    m.attr("PULSE_NORM") = lr::pulse_norm();

    m.def("matched_filter", [](const py::object& z, const py::object& fs, const py::object& n0,
                               const py::object& pos, const py::object& chunk) {
        return mf_rx::matched_filter(
            z, fs, n0, pos, chunk, [](std::int64_t f) { return lw::grid_step(f); },
            [](std::span<const double> zz, std::int64_t f, std::int64_t a, std::span<const std::int64_t> p) {
                return lr::matched_filter(zz, f, a, p);
            },
            [](std::span<const cd> zz, std::int64_t f, std::int64_t a, std::span<const std::int64_t> p) {
                return lr::matched_filter(zz, f, a, p);
            });
    }, py::arg("z"), py::arg("fs"), py::arg("n0"), py::arg("pos"), py::arg("chunk") = 2048);
}

const Registrar reg("lwave_rxf", 73, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
