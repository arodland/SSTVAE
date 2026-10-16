// sstvae_native.qrss.track_timing: bindings of native/core/qrss/track_timing
// (the rest of sstvae/qrss/track.py: the coarse timing fit, gate V and the
// frequency-path measurements). Names, arguments and defaults follow the
// reference's, so tests/qrss_native/track_timing.py can assign them over it:
//
//   fit_timing(ch, ch_fs, path, spec, tmpl_fn=None, tau0_init=None,
//              ppm_range=250.0, *, t0_index, tau_range=None, z_d=None, n_best=1)
//       -> "Timing" dict (n_best <= 1) or a list of them
//   _fit_timing_tries(<fit_timing's arguments>)
//       -> (fit_timing(n_best=1), fit_timing(n_best)) from one grid
//   _segment_timing(x, sig2, kn, t0abs, delta, sps, n_pos, seg_s=SEG_TIMING_S)
//       -> (t0abs, delta, score)
//   verify(ch, ch_fs, spec, cand, *, fit=True, tau_range=None, erase_s=())
//       -> "Detection" dict or None
//   _carrier_lines(z, chan, spec, seg_s, span_hz, *, med_band=None, ratio=10.0)
//       -> (t, f, ratio) arrays
//   refine_path(chan, spec, path, seg_s=PATH_SEG_S, span_hz=PATH_SPAN_HZ, *,
//               med_band=None, ratio=10.0, fine_ratio=PATH_FINE_RATIO,
//               fine_share=PATH_FINE_SHARE) -> ("FreqPath" dict, unchanged)
//   line_path(chan, spec, path) -> ("FreqPath" dict, unchanged)
//       unchanged: the reference returns `path` itself (the row hands it back)
//   _flat_path(path) -> "FreqPath" dict
//   verify_track(chan, spec, cand, *, fit=True, tau_range=None, erase_s=())
//       -> ("TrackResult" dict with chan None, same_timing): same_timing says
//          the result's timing is cand's own (no fit ran and the tracker kept it)
//
// and, for the parity tests, what fit_timing computes on the way:
//
//   _fit_timing_trace(<fit_timing's arguments>) -> (timings, trace dict)
//   _fit_statistic(z_d, lp_hz, fs=CH_FS)        Im(MF{z_d conj(LP(z_d))})
//   _fit_grid(x, kn, base, nlag, frac0, deltas, sps, norm) -> [n_delta, nlag]
//
// fit_timing with z_d None derotates through the module's own
// freq_path_spline and derotate (sstvae.qrss.track's, whatever implements
// them), as the reference does; a `path` that is not a FreqPath is called
// as it is. The CE-only functions refuse an L frame (AssertionError) before
// reading anything else. verify / verify_track fill the capture's noise
// table first as the tracker binding does (chan.noise_psd_at(0.0) when
// chan._psd is None). _parabola is not bound here: the row is
// acquire_stats' (the same function, with numpy's float32 typing).

#include "qrss/rx_types.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "qrss/qrss_config.hpp"
#include "qrss/demod_rx.hpp"
#include "qrss/frame.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/track.hpp"
#include "qrss/track_timing.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace tk = sstvae::qrss::track;
namespace fr = sstvae::qrss::frame;
namespace cfg = sstvae::qrss::config;
namespace rtk = sstvae::qrss::rxconfig::track;
namespace types = sstvae::qrss::types;
using cd = std::complex<double>;

py::module_ reference() { return py::module_::import("sstvae.qrss.track"); }

// chan.noise_psd_at(0.0) when chan._psd is still None (see track.cpp's).
void ensure_psd(const py::handle& chan) {
    if (py::isinstance<py::dict>(chan) || !py::hasattr(chan, "_psd")) return;
    if (chan.attr("_psd").is_none()) chan.attr("noise_psd_at")(0.0);
}

std::shared_ptr<const tk::ChanCapture> chan_full(const py::handle& chan) {
    ensure_psd(chan);
    return std::make_shared<const tk::ChanCapture>(rx::chan_capture(chan));
}

// What _carrier_lines reads of a capture: fs and t0_index (its t_s).
tk::ChanCapture chan_times(const py::handle& o) {
    tk::ChanCapture c;
    c.fs = static_cast<int>(rx::as_int(rx::field(o, "fs")));
    c.t0_index = rx::as_double(rx::field(o, "t0_index"));
    return c;
}

std::optional<std::pair<double, double>> band_of(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    py::sequence s = py::reinterpret_borrow<py::sequence>(o);
    return std::make_pair(rx::as_double(s[0]), rx::as_double(s[1]));
}

[[noreturn]] void raise(PyObject* type, const std::string& what) {
    PyErr_SetString(type, what.c_str());
    throw py::error_already_set();
}

// erase_s as the reference's erase reads it: nothing when falsy, else
// `for a, b in spans` with CPython 3.13's refusals and messages.
std::vector<std::pair<double, double>> spans_of(const py::object& o) {
    std::vector<std::pair<double, double>> out;
    if (!rx::as_bool(o)) return out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(o))) {
        if (!py::isinstance<py::iterable>(h))
            raise(PyExc_TypeError, std::string("cannot unpack non-iterable ") + Py_TYPE(h.ptr())->tp_name + " object");
        py::list items;
        for (const py::handle x : py::reinterpret_borrow<py::iterable>(py::iter(h))) {
            items.append(x);
            if (items.size() > 2) raise(PyExc_ValueError, "too many values to unpack (expected 2)");
        }
        if (items.size() < 2)
            raise(PyExc_ValueError,
                  "not enough values to unpack (expected 2, got " + std::to_string(items.size()) + ")");
        out.emplace_back(rx::as_double(items[0]), rx::as_double(items[1]));
    }
    return out;
}

py::array to_numpy2(const std::vector<double>& v, std::size_t rows, std::size_t cols) {
    py::array_t<double> a({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
    return a;
}

// fit_timing's arguments, read in the reference's order.
struct FitCall {
    fr::FrameSpec spec;
    int fs = 0;
    std::vector<cd> z_d;
    std::optional<tk::Known> known;
    tk::FitTimingArgs args;
    double t0_index = 0.0;
};

FitCall fit_call(const py::object& ch, const py::object& ch_fs, const py::object& path, const py::object& spec,
                 const py::object& tmpl_fn, const py::object& tau0_init, const py::object& ppm_range,
                 const py::object& t0_index, const py::object& tau_range, const py::object& z_d,
                 const py::object& n_best) {
    FitCall c{rx::spec(spec), 0, {}, std::nullopt, {}, 0.0};
    tk::ce_only(c.spec, "track.fit_timing", "ltrack.fit_timing_l");
    c.fs = static_cast<int>(rx::as_int(ch_fs));
    py::object zd = z_d;
    if (zd.is_none()) {
        py::module_ T = reference();
        py::object fn = py::isinstance(path, T.attr("FreqPath")) ? T.attr("freq_path_spline")(path) : path;
        zd = T.attr("derotate")(ch, fn);
    }
    c.z_d = rx::vec<cd>(zd);
    if (!tmpl_fn.is_none() && py::isinstance(tmpl_fn, reference().attr("_Known"))) c.known = rx::known(tmpl_fn);
    if (!tau0_init.is_none()) c.args.tau0_init = rx::as_double(tau0_init);
    c.args.ppm_range = rx::as_double(ppm_range);
    if (!tau_range.is_none()) c.args.tau_range = rx::as_double(tau_range);
    c.args.n_best = rx::as_int(n_best);
    c.t0_index = rx::as_double(t0_index);
    return c;
}

std::vector<types::Timing> run_fit(FitCall& c, tk::FitTimingTrace* trace) {
    if (c.known) c.args.known = &*c.known;
    py::gil_scoped_release nogil;
    return tk::fit_timing(std::span<const cd>(c.z_d), c.fs, c.spec, c.t0_index, c.args, trace);
}

py::object timings_py(const std::vector<types::Timing>& t, std::int64_t n_best) {
    if (n_best <= 1) return rx::to_py(t.front());
    py::list out;
    for (const auto& x : t) out.append(rx::to_py(x));
    return out;
}

py::dict trace_py(const tk::FitTimingTrace& tr) {
    py::dict d;
    d["base"] = tr.base;
    d["nlag"] = tr.nlag;
    d["nd"] = tr.nd;
    d["frac0"] = tr.frac0;
    d["step"] = tr.step;
    d["sig2"] = rx::arr(tr.sig2);
    py::list peaks;
    for (const auto& p : tr.peaks) {
        py::dict e;
        e["lp"] = p.lp;
        e["i"] = p.i;
        e["j"] = p.j;
        e["di"] = p.di;
        e["dj"] = p.dj;
        e["zmax"] = p.zmax;
        e["t0abs_in"] = p.t0abs_in;
        e["delta_in"] = p.delta_in;
        e["t0abs"] = p.t0abs;
        e["delta"] = p.delta;
        e["score"] = p.score;
        peaks.append(e);
    }
    d["peaks"] = peaks;
    return d;
}

tk::VerifyArgs verify_args(const py::object& fit, const py::object& tau_range, const py::object& erase_s) {
    tk::VerifyArgs a;
    a.fit = rx::as_bool(fit);
    if (!tau_range.is_none()) a.tau_range = rx::as_double(tau_range);
    a.erase_s = spans_of(erase_s);
    return a;
}

// The same doubles bit for bit (a NaN z or covariance entry included: with
// fit=False the reference hands back cand.timing itself whatever it holds,
// and a == comparison would call a NaN timing a new one).
bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
bool same_timing(const types::Timing& a, const types::Timing& b) {
    if (!(same_bits(a.tau0, b.tau0) && same_bits(a.ppm, b.ppm) && same_bits(a.gamma, b.gamma) &&
          same_bits(a.z, b.z) && a.cov.rows == b.cov.rows && a.cov.cols == b.cov.cols &&
          a.cov.data.size() == b.cov.data.size()))
        return false;
    for (std::size_t i = 0; i < a.cov.data.size(); ++i)
        if (!same_bits(a.cov.data[i], b.cov.data[i])) return false;
    return true;
}

void bind(py::module_& q) {
    rx::register_translators();
    demod_rx::register_wrong_waveform();   // an L frame's AssertionError
    py::module_ m = q.def_submodule("track_timing",
                                    "native/core/qrss/track_timing (sstvae.qrss.track's timing fit and gate V)");

    m.def("fit_timing", [](const py::object& ch, const py::object& ch_fs, const py::object& path,
                           const py::object& spec, const py::object& tmpl_fn, const py::object& tau0_init,
                           const py::object& ppm_range, const py::object& t0_index, const py::object& tau_range,
                           const py::object& z_d, const py::object& n_best) {
        FitCall c = fit_call(ch, ch_fs, path, spec, tmpl_fn, tau0_init, ppm_range, t0_index, tau_range, z_d, n_best);
        const std::vector<types::Timing> t = run_fit(c, nullptr);
        return timings_py(t, c.args.n_best);
    }, py::arg("ch"), py::arg("ch_fs"), py::arg("path"), py::arg("spec"), py::arg("tmpl_fn") = py::none(),
       py::arg("tau0_init") = py::none(), py::arg("ppm_range") = 250.0, py::kw_only(), py::arg("t0_index"),
       py::arg("tau_range") = py::none(), py::arg("z_d") = py::none(), py::arg("n_best") = 1);

    m.def("_fit_timing_tries", [](const py::object& ch, const py::object& ch_fs, const py::object& path,
                                  const py::object& spec, const py::object& tmpl_fn, const py::object& tau0_init,
                                  const py::object& ppm_range, const py::object& t0_index,
                                  const py::object& tau_range, const py::object& z_d, const py::object& n_best) {
        FitCall c = fit_call(ch, ch_fs, path, spec, tmpl_fn, tau0_init, ppm_range, t0_index, tau_range, z_d, n_best);
        if (c.known) c.args.known = &*c.known;
        tk::FitTimingTries t;
        {
            py::gil_scoped_release nogil;
            t = tk::fit_timing_tries(std::span<const cd>(c.z_d), c.fs, c.spec, c.t0_index, c.args);
        }
        return py::make_tuple(rx::to_py(t.best), timings_py(t.tries, c.args.n_best));
    }, py::arg("ch"), py::arg("ch_fs"), py::arg("path"), py::arg("spec"), py::arg("tmpl_fn") = py::none(),
       py::arg("tau0_init") = py::none(), py::arg("ppm_range") = 250.0, py::kw_only(), py::arg("t0_index"),
       py::arg("tau_range") = py::none(), py::arg("z_d") = py::none(), py::arg("n_best") = 1);

    m.def("_fit_timing_trace", [](const py::object& ch, const py::object& ch_fs, const py::object& path,
                                  const py::object& spec, const py::object& tmpl_fn, const py::object& tau0_init,
                                  const py::object& ppm_range, const py::object& t0_index,
                                  const py::object& tau_range, const py::object& z_d, const py::object& n_best) {
        FitCall c = fit_call(ch, ch_fs, path, spec, tmpl_fn, tau0_init, ppm_range, t0_index, tau_range, z_d, n_best);
        tk::FitTimingTrace tr;
        const std::vector<types::Timing> t = run_fit(c, &tr);
        py::list ts;
        for (const auto& x : t) ts.append(rx::to_py(x));
        return py::make_tuple(ts, trace_py(tr));
    }, py::arg("ch"), py::arg("ch_fs"), py::arg("path"), py::arg("spec"), py::arg("tmpl_fn") = py::none(),
       py::arg("tau0_init") = py::none(), py::arg("ppm_range") = 250.0, py::kw_only(), py::arg("t0_index"),
       py::arg("tau_range") = py::none(), py::arg("z_d") = py::none(), py::arg("n_best") = 1);

    m.def("_fit_statistic", [](const py::object& z_d, double lp_hz, int fs) {
        const std::vector<cd> z = rx::vec<cd>(z_d);
        std::vector<double> x;
        {
            py::gil_scoped_release nogil;
            x = tk::fit_statistic(z, lp_hz, fs);
        }
        return rx::arr(x);
    }, py::arg("z_d"), py::arg("lp_hz"), py::arg("fs") = cfg::CH_FS);

    m.def("_fit_grid", [](const py::object& x, const py::object& kn, std::int64_t base, std::int64_t nlag,
                          double frac0, const py::object& deltas, double sps, double norm) {
        const std::vector<double> xv = rx::vec<double>(x);
        const tk::Known k = rx::known(kn);
        const std::vector<double> d = rx::vec<double>(deltas);
        std::vector<double> Z;
        {
            py::gil_scoped_release nogil;
            Z = tk::fit_grid(xv, k, base, nlag, frac0, d, sps, norm);
        }
        return to_numpy2(Z, d.size(), nlag < 0 ? 0 : static_cast<std::size_t>(nlag));
    }, py::arg("x"), py::arg("kn"), py::arg("base"), py::arg("nlag"), py::arg("frac0"), py::arg("deltas"),
       py::arg("sps"), py::arg("norm"));

    m.def("_segment_timing", [](const py::object& x, const py::object& sig2, const py::object& kn,
                                const py::object& t0abs, const py::object& delta, const py::object& sps,
                                const py::object& n_pos, const py::object& seg_s) {
        const std::vector<double> xv = rx::vec<double>(x);
        const double s2 = rx::as_double(sig2);
        const tk::Known k = rx::known(kn);
        const double t0 = rx::as_double(t0abs), d = rx::as_double(delta), sp = rx::as_double(sps);
        const std::int64_t np_ = rx::as_int(n_pos);
        const double sg = rx::as_double(seg_s);
        tk::SegmentTiming r;
        {
            py::gil_scoped_release nogil;
            r = tk::segment_timing(xv, s2, k, t0, d, sp, np_, sg);
        }
        return py::make_tuple(r.t0abs, r.delta, r.score);
    }, py::arg("x"), py::arg("sig2"), py::arg("kn"), py::arg("t0abs"), py::arg("delta"), py::arg("sps"),
       py::arg("n_pos"), py::arg("seg_s") = rtk::SEG_TIMING_S);

    m.def("verify", [](const py::object& ch, const py::object& ch_fs, const py::object& spec,
                       const py::object& cand, const py::object& fit, const py::object& tau_range,
                       const py::object& erase_s) -> py::object {
        (void)ch_fs;   // the reference does not read it either
        const fr::FrameSpec s = rx::spec(spec);
        tk::ce_only(s, "track.verify", "ltrack.verify_l");
        const auto c = chan_full(ch);
        const types::Detection d = rx::detection(cand);
        const tk::VerifyArgs a = verify_args(fit, tau_range, erase_s);
        std::optional<types::Detection> out;
        {
            py::gil_scoped_release nogil;
            out = tk::verify(c, s, d, a);
        }
        if (!out) return py::none();
        return rx::to_py(*out);
    }, py::arg("ch"), py::arg("ch_fs"), py::arg("spec"), py::arg("cand"), py::kw_only(), py::arg("fit") = true,
       py::arg("tau_range") = py::none(), py::arg("erase_s") = py::tuple());

    m.def("_carrier_lines", [](const py::object& z, const py::object& chan, const py::object& spec,
                               const py::object& seg_s, const py::object& span_hz, const py::object& med_band,
                               const py::object& ratio) {
        const std::vector<cd> zv = rx::vec<cd>(z);
        const tk::ChanCapture c = chan_times(chan);
        const fr::FrameSpec s = rx::spec(spec);
        const double sg = rx::as_double(seg_s), sp = rx::as_double(span_hz), r = rx::as_double(ratio);
        const auto band = band_of(med_band);
        tk::CarrierLines L;
        {
            py::gil_scoped_release nogil;
            L = tk::carrier_lines(zv, c, s, sg, sp, band, r);
        }
        return py::make_tuple(rx::arr(L.t), rx::arr(L.f), rx::arr(L.w));
    }, py::arg("z"), py::arg("chan"), py::arg("spec"), py::arg("seg_s"), py::arg("span_hz"), py::kw_only(),
       py::arg("med_band") = py::none(), py::arg("ratio") = 10.0);

    m.def("refine_path", [](const py::object& chan, const py::object& spec, const py::object& path,
                            const py::object& seg_s, const py::object& span_hz, const py::object& med_band,
                            const py::object& ratio, const py::object& fine_ratio, const py::object& fine_share) {
        const tk::ChanCapture c = rx::chan_capture(chan);
        const fr::FrameSpec s = rx::spec(spec);
        const types::FreqPath p = rx::freq_path(path);
        tk::RefinePathArgs a;
        a.seg_s = rx::as_double(seg_s);
        a.span_hz = rx::as_double(span_hz);
        a.med_band = band_of(med_band);
        a.ratio = rx::as_double(ratio);
        a.fine_ratio = rx::as_double(fine_ratio);
        a.fine_share = rx::as_double(fine_share);
        std::optional<tk::PathResult> r;
        {
            py::gil_scoped_release nogil;
            r.emplace(tk::refine_path(c, s, p, a));
        }
        return py::make_tuple(rx::to_py(r->path), r->unchanged);
    }, py::arg("chan"), py::arg("spec"), py::arg("path"), py::arg("seg_s") = rtk::PATH_SEG_S,
       py::arg("span_hz") = rtk::PATH_SPAN_HZ, py::kw_only(), py::arg("med_band") = py::none(),
       py::arg("ratio") = 10.0, py::arg("fine_ratio") = rtk::PATH_FINE_RATIO,
       py::arg("fine_share") = rtk::PATH_FINE_SHARE);

    m.def("line_path", [](const py::object& chan, const py::object& spec, const py::object& path) {
        const tk::ChanCapture c = rx::chan_capture(chan);
        const fr::FrameSpec s = rx::spec(spec);
        const types::FreqPath p = rx::freq_path(path);
        std::optional<tk::PathResult> r;
        {
            py::gil_scoped_release nogil;
            r.emplace(tk::line_path(c, s, p));
        }
        return py::make_tuple(rx::to_py(r->path), r->unchanged);
    }, py::arg("chan"), py::arg("spec"), py::arg("path"));

    m.def("_flat_path", [](const py::object& path) { return rx::to_py(tk::flat_path(rx::freq_path(path))); },
          py::arg("path"));

    m.def("verify_track", [](const py::object& chan, const py::object& spec, const py::object& cand,
                             const py::object& fit, const py::object& tau_range, const py::object& erase_s) {
        const fr::FrameSpec s = rx::spec(spec);
        tk::ce_only(s, "track.verify_track", "ltrack.verify_track_l");
        const auto c = chan_full(chan);
        const types::Detection d = rx::detection(cand);
        const tk::VerifyArgs a = verify_args(fit, tau_range, erase_s);
        std::optional<tk::TrackResult> r;
        {
            py::gil_scoped_release nogil;
            r.emplace(tk::verify_track(c, s, d, a));
        }
        const bool same = !a.fit && d.timing && same_timing(r->timing, *d.timing);
        r->chan = nullptr;   // the row puts the caller's object back
        return py::make_tuple(rx::to_py(*r), same);
    }, py::arg("chan"), py::arg("spec"), py::arg("cand"), py::kw_only(), py::arg("fit") = true,
       py::arg("tau_range") = py::none(), py::arg("erase_s") = py::tuple());
}

const Registrar reg("track_timing", 77, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
