// sstvae_native.qrss.demod: bindings of native/core/qrss/demod (waveform
// CE's symbol extraction, kappa calibration, block estimators and header
// LLRs). Names, arguments, defaults and texts follow sstvae/qrss/demod.py,
// so tests/qrss_native/demod.py can assign these over the reference's:
//
//   symbol_estimates(tr, stream_idx) -> (y, s2n, s2)
//   data_estimates(tr) -> (y, s2n, extra)
//   spread_llr(y, s2, spec) -> float64[2474]
//   calibrate(tr) -> (kappa, suspect)
//   block_estimate(y, s2, q, estimator="joint") -> (z, v)
//   extract(m, track, spec, q, estimator="joint", *, calibrate_kappa=True)
//       -> (z f32, w f32, llr f32, diag)
//
// A track is anything with the reference's attributes (demod_rx.hpp says
// which each function reads); a spec a native FrameSpec or the
// reference's. The AssertionError for an L frame is raised before the
// track is read, as the reference's assert is its first statement.

#include "qrss/demod_rx.hpp"

#include <cstdint>
#include <span>
#include <vector>

#include "qrss/demod.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace dm = sstvae::qrss::demod;
namespace fr = sstvae::qrss::frame;
using namespace demod_rx;

void bind(py::module_& q) {
    rx::register_translators();
    register_wrong_waveform();
    py::module_ m = q.def_submodule("demod", "Port of sstvae.qrss.demod (waveform CE's demodulator)");

    m.attr("D_PASS") = dm::D_PASS;
    m.attr("D_REF") = dm::D_REF;
    m.attr("KAPPA_RANGE") = py::make_tuple(dm::KAPPA_RANGE[0], dm::KAPPA_RANGE[1]);
    m.attr("KAPPA_PRIOR_SD") = dm::KAPPA_PRIOR_SD;
    m.attr("KAPPA_S2_MAX") = dm::KAPPA_S2_MAX;
    m.attr("KAPPA_S2_REL") = dm::KAPPA_S2_REL;
    m.attr("U_MIN_REL") = dm::U_MIN_REL;

    m.def("symbol_estimates", [](const py::object& tr, const py::object& stream_idx) {
        const Track t = read_track(tr, BASE);
        const std::vector<std::int64_t> idx = rx::vec<std::int64_t>(stream_idx);
        const dm::SymbolEstimates se = dm::symbol_estimates(t.view(), idx);
        return py::make_tuple(to_numpy(se.y), to_numpy(se.s2n), to_numpy(se.s2));
    }, py::arg("tr"), py::arg("stream_idx"));

    m.def("data_estimates", [](const py::object& tr) {
        const Track t = read_track(tr, CE_CLASSES);
        const dm::DataEstimates de = dm::data_estimates(t.view());
        return py::make_tuple(to_numpy(de.y), to_numpy(de.s2n), de.extra);
    }, py::arg("tr"));

    m.def("spread_llr", [](const py::object& y, const py::object& s2, const py::object& spec) {
        const fr::FrameSpec s = rx::spec(spec);
        const std::vector<double> yv = rx::vec<double>(y), sv = rx::vec<double>(s2);
        return to_numpy(dm::spread_llr(yv, sv, s));
    }, py::arg("y"), py::arg("s2"), py::arg("spec"));

    m.def("calibrate", [](const py::object& tr) {
        fr::FrameSpec s = track_spec(tr);
        if (s.waveform() != fr::Waveform::CE) {   // the assert, before anything else is read
            dm::TrackView v;
            v.spec = &s;
            (void)dm::calibrate(v);
        }
        const Track t = read_track(tr, std::move(s), CE_CLASSES | CE_TEMPLATE);
        const dm::Calibration c = dm::calibrate(t.view());
        return py::make_tuple(c.kappa, c.suspect);
    }, py::arg("tr"));

    m.def("block_estimate", [](const py::object& y, const py::object& s2, const py::object& qq,
                               const py::object& estimator) {
        const std::vector<double> yv = rx::vec<double>(y), sv = rx::vec<double>(s2);
        const std::uint64_t qv = q_of(qq);
        const Estimator e = estimator_of(estimator);
        const dm::BlockEstimate be = dm::block_estimate(yv, sv, qv, e.name, e.shown);
        return py::make_tuple(to_numpy(be.z), to_numpy(be.v));
    }, py::arg("y"), py::arg("s2"), py::arg("q"), py::arg("estimator") = "joint");

    m.def("extract", [](const py::object& mf, const py::object& track, const py::object& spec,
                        const py::object& qq, const py::object& estimator, const py::object& calibrate_kappa) {
        const fr::FrameSpec s = rx::spec(spec);
        if (s.waveform() != fr::Waveform::CE) {
            const dm::TrackView none{};
            (void)dm::extract(none, s, 0);   // raises the reference's AssertionError
        }
        replace_m(mf, track);
        const bool cal = rx::as_bool(calibrate_kappa);
        const Track t = read_track(track, CE_CLASSES | CE_TEMPLATE);
        const std::uint64_t qv = q_of(qq);
        const Estimator e = estimator_of(estimator);
        const dm::Extraction ex = dm::extract(t.view(), s, qv, e.name, cal, e.shown);
        return extraction_to_py(ex, estimator);
    }, py::arg("m"), py::arg("track"), py::arg("spec"), py::arg("q"), py::arg("estimator") = "joint",
       py::kw_only(), py::arg("calibrate_kappa") = true);
}

const Registrar reg("demod", 70, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
