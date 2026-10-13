// sstvae_native.qrss.ldemod: bindings of native/core/qrss/ldemod (waveform
// L's symbol extraction, kappa calibration and header LLRs). Names,
// arguments, defaults and texts follow sstvae/qrss/ldemod.py, so
// tests/qrss_native/ldemod.py can assign these over the reference's:
//
//   d_pass_x(spec) -> float
//   symbol_estimates_l(tr, stream_idx, gain) -> (r complex128, s2n)
//   data_estimates_l(tr) -> (y_x, s2n_x, extra)
//   header_estimates_l(tr) -> (y_h, s2n_h)
//   _combine(ests) -> (kappa, suspect)
//   kappa_estimates_l(tr) -> [(kappa, variance), ...]
//   calibrate_l(tr) -> (kappa, suspect)
//   extract_l(m, track, spec, q, estimator="joint", *, calibrate_kappa=True)
//       -> (z f32, w f32, llr f32, diag)
//
// Tracks as demod_rx.hpp reads them; the AssertionError for a CE frame
// comes before the track is read, as the reference's _l_only does.

#include "qrss/demod_rx.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include "qrss/demod.hpp"
#include "qrss/ldemod.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace dm = sstvae::qrss::demod;
namespace ld = sstvae::qrss::ldemod;
namespace fr = sstvae::qrss::frame;
using namespace demod_rx;

// _l_only(spec, what) for a non-L spec: the core's refusal, raised before
// anything else is read.
void l_only(const fr::FrameSpec& s, int which) {
    if (s.waveform() == fr::Waveform::L) return;
    dm::TrackView v;
    v.spec = &s;
    if (which == 0) (void)ld::kappa_estimates_l(v);
    if (which == 1) (void)ld::calibrate_l(v);
    (void)ld::extract_l(v, s, 0);
}

py::tuple estimates_to_py(const dm::DataEstimates& de) {
    return py::make_tuple(to_numpy(de.y), to_numpy(de.s2n), de.extra);
}

void bind(py::module_& q) {
    rx::register_translators();
    register_wrong_waveform();
    py::module_ m = q.def_submodule("ldemod", "Port of sstvae.qrss.ldemod (waveform L's demodulator)");

    m.attr("G_REF_L") = ld::G_REF_L;
    m.attr("G_HDR_L") = ld::G_HDR_L;
    m.attr("G_DATA_L") = ld::G_DATA_L;
    m.attr("G_PRE_L") = ld::G_PRE_L;
    m.attr("D_PASS_L") = ld::D_PASS_L;
    m.attr("D_REF_L") = ld::D_REF_L;
    m.attr("D_HDR_L") = ld::D_HDR_L;
    m.attr("D_PRE_L") = ld::D_PRE_L;
    m.attr("D_PASS_LX") = ld::D_PASS_LX;

    m.def("d_pass_x", [](const py::object& spec) {
        // D_PASS_L * (1.0 + float(spec.hdr_rho)): only the rho is read
        return ld::D_PASS_L * (1.0 + rx::as_double(rx::field(spec, "hdr_rho")));
    }, py::arg("spec"));

    m.def("symbol_estimates_l", [](const py::object& tr, const py::object& stream_idx, const py::object& gain) {
        const Track t = read_track(tr, BASE);
        const std::vector<std::int64_t> idx = rx::vec<std::int64_t>(stream_idx);
        const ld::SymbolEstimatesL se = ld::symbol_estimates_l(t.view(), idx, rx::as_double(gain));
        return py::make_tuple(to_numpy(se.r), to_numpy(se.s2n));
    }, py::arg("tr"), py::arg("stream_idx"), py::arg("gain"));

    m.def("data_estimates_l", [](const py::object& tr) {
        const Track t = read_track(tr, L_HDR);
        return estimates_to_py(ld::data_estimates_l(t.view()));
    }, py::arg("tr"));

    m.def("header_estimates_l", [](const py::object& tr) {
        const Track t = read_track(tr, BASE);
        const ld::HeaderEstimates he = ld::header_estimates_l(t.view());
        return py::make_tuple(to_numpy(he.y), to_numpy(he.s2n));
    }, py::arg("tr"));

    m.def("_combine", [](const py::object& ests) {
        std::vector<std::pair<double, double>> e;
        for (const py::handle kv : py::reinterpret_borrow<py::iterable>(ests)) {
            const py::tuple t = py::reinterpret_borrow<py::object>(kv).cast<py::tuple>();
            if (t.size() != 2) throw py::value_error("_combine: each estimate is (kappa, variance)");
            e.emplace_back(rx::as_double(t[0]), rx::as_double(t[1]));
        }
        const dm::Calibration c = ld::combine(e);
        return py::make_tuple(c.kappa, c.suspect);
    }, py::arg("ests"));

    m.def("kappa_estimates_l", [](const py::object& tr) {
        fr::FrameSpec s = track_spec(tr);
        l_only(s, 0);
        const Track t = read_track(tr, std::move(s), L_MEAS | L_HDR);
        py::list out;
        for (const auto& [k, v] : ld::kappa_estimates_l(t.view())) out.append(py::make_tuple(k, v));
        return out;
    }, py::arg("tr"));

    m.def("calibrate_l", [](const py::object& tr) {
        fr::FrameSpec s = track_spec(tr);
        l_only(s, 1);
        const Track t = read_track(tr, std::move(s), L_MEAS | L_HDR);
        const dm::Calibration c = ld::calibrate_l(t.view());
        return py::make_tuple(c.kappa, c.suspect);
    }, py::arg("tr"));

    m.def("extract_l", [](const py::object& mf, const py::object& track, const py::object& spec,
                          const py::object& qq, const py::object& estimator, const py::object& calibrate_kappa) {
        const fr::FrameSpec s = rx::spec(spec);
        l_only(s, 2);
        replace_m(mf, track);
        const bool cal = rx::as_bool(calibrate_kappa);
        const Track t = read_track(track, L_MEAS | L_HDR);
        const std::uint64_t qv = q_of(qq);
        const Estimator e = estimator_of(estimator);
        const dm::Extraction ex = ld::extract_l(t.view(), s, qv, e.name, cal, e.shown);
        return extraction_to_py(ex, estimator);
    }, py::arg("m"), py::arg("track"), py::arg("spec"), py::arg("q"), py::arg("estimator") = "joint",
       py::kw_only(), py::arg("calibrate_kappa") = true);
}

const Registrar reg("ldemod", 71, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
