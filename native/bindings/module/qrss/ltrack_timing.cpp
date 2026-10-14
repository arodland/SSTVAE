// sstvae_native.qrss.ltrack_timing: bindings of native/core/qrss/ltrack_timing
// (the rest of sstvae/qrss/ltrack.py: L's coarse timing fit and gate V).
// Names, arguments and defaults follow the reference's, so
// tests/qrss_native/ltrack_timing.py can assign them over it:
//
//   _lag_phase(lag), _clock_phase(pos, d)        complex128 of the argument's
//                                                shape (0-d: a numpy scalar)
//   fit_timing_l(ch, ch_fs, path, spec, tmpl_fn=None, tau0_init=None,
//                ppm_range=250.0, *, t0_index, tau_range=None, z_d=None, n_best=1)
//       -> "Timing" dict (n_best <= 1) or a list of them
//   _fit_timing_tries_l(<fit_timing_l's arguments>)
//       -> (fit_timing_l(n_best=1), fit_timing_l(n_best)) from one grid
//   _segment_timing_l(x, sig2, kn, t0abs, delta, sps, n_pos, n_ref,
//                     seg_s=SEG_TIMING_S) -> (t0abs, delta, score)
//   refine_path_l(chan, spec, path) -> ("FreqPath" dict, unchanged)
//       unchanged: the reference returns `path` itself (the row hands it back)
//   verify_track_l(chan, spec, cand, *, fit=True, tau_range=None, erase_s=())
//       -> ("LTrackResult" dict with chan None, same_timing): same_timing says
//          the result's timing is cand's own (no fit ran and the tracker kept it)
//   resid_ratio(tr, c_known, gain, dist) -> float
//   verify_l(ch, ch_fs, spec, cand, *, fit=True, tau_range=None, erase_s=())
//       -> "Detection" dict or None
//
// and, for the parity tests, what fit_timing_l computes on the way:
//
//   _fit_timing_l_trace(<fit_timing_l's arguments>) -> (timings, trace dict)
//   _fit_mixed(z_d, n_ref, fs=CH_FS)            MF{z_d conj(lag_phase(n - n_ref))}
//   _fit_statistic_l(z_d, n_ref, lp_hz, fs=CH_FS)   that times conj(LP(z_d))
//   _fit_grid_l(x, kn, base, nlag, frac0, deltas, sps, norm) -> [n_delta, nlag]
//
// fit_timing_l with z_d None derotates through ltrack's own
// freq_path_spline and derotate (whatever implements them), as the
// reference does; a `path` that is not a FreqPath is called as it is. The
// L-only functions refuse a CE frame (AssertionError) before reading
// anything else; resid_ratio meets a CE frame where the reference does, in
// lframe.layout. verify_l / verify_track_l fill the capture's noise table
// first as the tracker binding does (chan.noise_psd_at(0.0) when chan._psd
// is None).

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

#include "qrss/config.hpp"
#include "qrss/demod_rx.hpp"
#include "qrss/frame.hpp"
#include "qrss/ltrack.hpp"
#include "qrss/ltrack_timing.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/track.hpp"
#include "qrss/track_timing.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace tk = sstvae::qrss::track;
namespace lt = sstvae::qrss::ltrack;
namespace fr = sstvae::qrss::frame;
namespace cfg = sstvae::qrss::config;
namespace rlt = sstvae::qrss::rxconfig::ltrack;
namespace types = sstvae::qrss::types;
using cd = std::complex<double>;

py::module_ reference() { return py::module_::import("sstvae.qrss.ltrack"); }

[[noreturn]] void raise(PyObject* type, const std::string& what) {
    PyErr_SetString(type, what.c_str());
    throw py::error_already_set();
}

// chan.noise_psd_at(0.0) when chan._psd is still None (see track.cpp's).
void ensure_psd(const py::handle& chan) {
    if (py::isinstance<py::dict>(chan) || !py::hasattr(chan, "_psd")) return;
    if (chan.attr("_psd").is_none()) chan.attr("noise_psd_at")(0.0);
}

std::shared_ptr<const tk::ChanCapture> chan_full(const py::handle& chan) {
    ensure_psd(chan);
    return std::make_shared<const tk::ChanCapture>(rx::chan_capture(chan));
}

// np.asarray(o, dtype=np.float64): its values (flattened) and shape.
struct Doubles {
    std::vector<double> v;
    std::vector<py::ssize_t> shape;
};
Doubles doubles(const py::handle& o) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o, py::arg("dtype") = "float64"));
    if (!a) throw py::error_already_set();
    return {rx::vec<double>(a), std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim())};
}

// A complex128 array of `shape` (a numpy scalar for a 0-d one, as numpy's
// elementwise arithmetic hands back).
py::object shaped_result(const std::vector<cd>& v, const std::vector<py::ssize_t>& shape) {
    py::array_t<cd> out(shape);
    if (!v.empty()) std::memcpy(static_cast<void*>(out.mutable_data()), v.data(), v.size() * sizeof(cd));
    if (shape.empty()) return out.attr("__getitem__")(py::tuple());
    return out;
}

py::array to_numpy2(const std::vector<double>& v, std::size_t rows, std::size_t cols) {
    py::array_t<double> a({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
    return a;
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

// lframe.layout on a CE frame, as resid_ratio meets it: the header's symbol
// count is checked first, and a CE FrameSpec has no n_data_payload.
void l_layout(const fr::FrameSpec& s) {
    if (s.waveform() == fr::Waveform::L) return;
    if (const std::int64_t n_hdr = s.n_hdr()) {
        const std::int64_t slots = n_hdr - (n_hdr + cfg::REF_PERIOD - 1) / cfg::REF_PERIOD;
        if (slots != cfg::N_HDR_BITS / 2)
            raise(PyExc_AssertionError,
                  std::to_string(slots) + " header symbols for " + std::to_string(cfg::N_HDR_BITS) + " bits");
    }
    raise(PyExc_AttributeError, "'FrameSpec' object has no attribute 'n_data_payload'");
}

tk::PathFn no_path() { return tk::PathFn::from_parts(0.0, 0.0, {0.0, 0.0}, std::nullopt); }

// fit_timing_l's arguments, read in the reference's order.
struct FitCall {
    fr::FrameSpec spec;
    int fs = 0;
    std::vector<cd> z_d;
    std::optional<tk::Known> known;
    lt::FitTimingLArgs args;
    double t0_index = 0.0;
};

FitCall fit_call(const py::object& ch, const py::object& ch_fs, const py::object& path, const py::object& spec,
                 const py::object& tmpl_fn, const py::object& tau0_init, const py::object& ppm_range,
                 const py::object& t0_index, const py::object& tau_range, const py::object& z_d,
                 const py::object& n_best) {
    FitCall c{rx::spec(spec), 0, {}, std::nullopt, {}, 0.0};
    lt::l_only(c.spec, "ltrack.fit_timing_l");
    c.fs = static_cast<int>(rx::as_int(ch_fs));
    py::object zd = z_d;
    if (zd.is_none()) {
        py::module_ L = reference();
        py::object fn = py::isinstance(path, L.attr("FreqPath")) ? L.attr("freq_path_spline")(path) : path;
        zd = L.attr("derotate")(ch, fn);
    }
    if (!tmpl_fn.is_none() && py::isinstance(tmpl_fn, py::module_::import("sstvae.qrss.track").attr("_Known")))
        c.known = rx::known(tmpl_fn);
    c.z_d = rx::vec<cd>(zd);
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
    return lt::fit_timing_l(std::span<const cd>(c.z_d), c.fs, c.spec, c.t0_index, c.args, trace);
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

py::object fit_timing_l_py(const py::object& ch, const py::object& ch_fs, const py::object& path,
                           const py::object& spec, const py::object& tmpl_fn, const py::object& tau0_init,
                           const py::object& ppm_range, const py::object& t0_index, const py::object& tau_range,
                           const py::object& z_d, const py::object& n_best) {
    FitCall c = fit_call(ch, ch_fs, path, spec, tmpl_fn, tau0_init, ppm_range, t0_index, tau_range, z_d, n_best);
    const std::vector<types::Timing> t = run_fit(c, nullptr);
    return timings_py(t, c.args.n_best);
}

py::object fit_timing_tries_l_py(const py::object& ch, const py::object& ch_fs, const py::object& path,
                                 const py::object& spec, const py::object& tmpl_fn, const py::object& tau0_init,
                                 const py::object& ppm_range, const py::object& t0_index,
                                 const py::object& tau_range, const py::object& z_d, const py::object& n_best) {
    FitCall c = fit_call(ch, ch_fs, path, spec, tmpl_fn, tau0_init, ppm_range, t0_index, tau_range, z_d, n_best);
    if (c.known) c.args.known = &*c.known;
    tk::FitTimingTries t;
    {
        py::gil_scoped_release nogil;
        t = lt::fit_timing_tries_l(std::span<const cd>(c.z_d), c.fs, c.spec, c.t0_index, c.args);
    }
    return py::make_tuple(rx::to_py(t.best), timings_py(t.tries, c.args.n_best));
}

py::object fit_timing_l_trace_py(const py::object& ch, const py::object& ch_fs, const py::object& path,
                                 const py::object& spec, const py::object& tmpl_fn, const py::object& tau0_init,
                                 const py::object& ppm_range, const py::object& t0_index,
                                 const py::object& tau_range, const py::object& z_d, const py::object& n_best) {
    FitCall c = fit_call(ch, ch_fs, path, spec, tmpl_fn, tau0_init, ppm_range, t0_index, tau_range, z_d, n_best);
    tk::FitTimingTrace tr;
    const std::vector<types::Timing> t = run_fit(c, &tr);
    py::list ts;
    for (const auto& x : t) ts.append(rx::to_py(x));
    return py::make_tuple(ts, trace_py(tr));
}

// What resid_ratio reads of a track: spec, pos, classes.known, psi, m, u, N, P.
double resid_ratio_py(const py::object& tr, const py::object& c_known, const py::object& gain,
                      const py::object& dist) {
    const fr::FrameSpec s = rx::spec(rx::field(tr, "spec"));
    l_layout(s);
    tk::LTrackResult t(s, nullptr, types::Timing{}, no_path());
    tk::LClasses cl;
    cl.known = rx::vec_bool(rx::field(rx::field(tr, "classes"), "known"));
    t.classes = std::move(cl);
    t.pos = rx::vec<std::int64_t>(rx::field(tr, "pos"));
    t.psi = rx::vec_or_empty<double>(rx::field(tr, "psi"));
    t.m = rx::vec_or_empty<cd>(rx::field(tr, "m"));
    t.u = rx::vec_or_empty<cd>(rx::field(tr, "u"));
    t.N = rx::vec_or_empty<double>(rx::field(tr, "N"));
    t.P = rx::vec_or_empty<double>(rx::field(tr, "P"));
    const std::vector<cd> c = rx::vec<cd>(c_known);
    const std::vector<double> g = rx::vec<double>(gain), d = rx::vec<double>(dist);
    py::gil_scoped_release nogil;
    return lt::resid_ratio(t, c, g, d);
}

void bind(py::module_& q) {
    rx::register_translators();
    demod_rx::register_wrong_waveform();   // a CE frame's AssertionError
    py::module_ m = q.def_submodule("ltrack_timing",
                                    "native/core/qrss/ltrack_timing (sstvae.qrss.ltrack's timing fit and gate V)");

    m.def("_lag_phase", [](const py::object& lag) {
        const Doubles d = doubles(lag);
        std::vector<cd> out;
        {
            py::gil_scoped_release nogil;
            out = lt::lag_phase(std::span<const double>(d.v));
        }
        return shaped_result(out, d.shape);
    }, py::arg("lag"));

    m.def("_clock_phase", [](const py::object& pos, const py::object& d) {
        const Doubles p = doubles(pos);
        const double dv = rx::as_double(d);
        std::vector<cd> out;
        {
            py::gil_scoped_release nogil;
            out = lt::clock_phase(std::span<const double>(p.v), dv);
        }
        return shaped_result(out, p.shape);
    }, py::arg("pos"), py::arg("d"));

    m.def("fit_timing_l", &fit_timing_l_py, py::arg("ch"), py::arg("ch_fs"), py::arg("path"), py::arg("spec"),
          py::arg("tmpl_fn") = py::none(), py::arg("tau0_init") = py::none(), py::arg("ppm_range") = 250.0,
          py::kw_only(), py::arg("t0_index"), py::arg("tau_range") = py::none(), py::arg("z_d") = py::none(),
          py::arg("n_best") = 1);

    m.def("_fit_timing_tries_l", &fit_timing_tries_l_py, py::arg("ch"), py::arg("ch_fs"), py::arg("path"),
          py::arg("spec"), py::arg("tmpl_fn") = py::none(), py::arg("tau0_init") = py::none(),
          py::arg("ppm_range") = 250.0, py::kw_only(), py::arg("t0_index"), py::arg("tau_range") = py::none(),
          py::arg("z_d") = py::none(), py::arg("n_best") = 1);

    m.def("_fit_timing_l_trace", &fit_timing_l_trace_py, py::arg("ch"), py::arg("ch_fs"), py::arg("path"),
          py::arg("spec"), py::arg("tmpl_fn") = py::none(), py::arg("tau0_init") = py::none(),
          py::arg("ppm_range") = 250.0, py::kw_only(), py::arg("t0_index"), py::arg("tau_range") = py::none(),
          py::arg("z_d") = py::none(), py::arg("n_best") = 1);

    m.def("_fit_mixed", [](const py::object& z_d, double n_ref, int fs) {
        const std::vector<cd> z = rx::vec<cd>(z_d);
        std::vector<cd> y;
        {
            py::gil_scoped_release nogil;
            y = lt::fit_mixed(z, n_ref, fs);
        }
        return rx::arr(y);
    }, py::arg("z_d"), py::arg("n_ref"), py::arg("fs") = cfg::CH_FS);

    m.def("_fit_statistic_l", [](const py::object& z_d, double n_ref, double lp_hz, int fs) {
        const std::vector<cd> z = rx::vec<cd>(z_d);
        std::vector<cd> x;
        {
            py::gil_scoped_release nogil;
            x = lt::fit_statistic_l(z, n_ref, lp_hz, fs);
        }
        return rx::arr(x);
    }, py::arg("z_d"), py::arg("n_ref"), py::arg("lp_hz"), py::arg("fs") = cfg::CH_FS);

    m.def("_fit_grid_l", [](const py::object& x, const py::object& kn, std::int64_t base, std::int64_t nlag,
                            double frac0, const py::object& deltas, double sps, double norm) {
        const std::vector<cd> xv = rx::vec<cd>(x);
        const tk::Known k = rx::known(kn);
        const std::vector<double> d = rx::vec<double>(deltas);
        std::vector<double> Z;
        {
            py::gil_scoped_release nogil;
            Z = lt::fit_grid_l(xv, k, base, nlag, frac0, d, sps, norm);
        }
        return to_numpy2(Z, d.size(), nlag < 0 ? 0 : static_cast<std::size_t>(nlag));
    }, py::arg("x"), py::arg("kn"), py::arg("base"), py::arg("nlag"), py::arg("frac0"), py::arg("deltas"),
       py::arg("sps"), py::arg("norm"));

    m.def("_segment_timing_l", [](const py::object& x, const py::object& sig2, const py::object& kn,
                                  const py::object& t0abs, const py::object& delta, const py::object& sps,
                                  const py::object& n_pos, const py::object& n_ref, const py::object& seg_s) {
        const std::vector<cd> xv = rx::vec<cd>(x);
        const double s2 = rx::as_double(sig2);
        const tk::Known k = rx::known(kn);
        const double t0 = rx::as_double(t0abs), d = rx::as_double(delta), sp = rx::as_double(sps);
        const std::int64_t np_ = rx::as_int(n_pos);
        const double nr = rx::as_double(n_ref), sg = rx::as_double(seg_s);
        tk::SegmentTiming r;
        {
            py::gil_scoped_release nogil;
            r = lt::segment_timing_l(xv, s2, k, t0, d, sp, np_, nr, sg);
        }
        return py::make_tuple(r.t0abs, r.delta, r.score);
    }, py::arg("x"), py::arg("sig2"), py::arg("kn"), py::arg("t0abs"), py::arg("delta"), py::arg("sps"),
       py::arg("n_pos"), py::arg("n_ref"), py::arg("seg_s") = rlt::SEG_TIMING_S);

    m.def("refine_path_l", [](const py::object& chan, const py::object& spec, const py::object& path) {
        const tk::ChanCapture c = rx::chan_capture(chan);
        const fr::FrameSpec s = rx::spec(spec);
        const types::FreqPath p = rx::freq_path(path);
        std::optional<tk::PathResult> r;
        {
            py::gil_scoped_release nogil;
            r.emplace(lt::refine_path_l(c, s, p));
        }
        return py::make_tuple(rx::to_py(r->path), r->unchanged);
    }, py::arg("chan"), py::arg("spec"), py::arg("path"));

    m.def("verify_track_l", [](const py::object& chan, const py::object& spec, const py::object& cand,
                               const py::object& fit, const py::object& tau_range, const py::object& erase_s) {
        const fr::FrameSpec s = rx::spec(spec);
        lt::l_only(s, "ltrack.verify_track_l");
        const auto c = chan_full(chan);
        const types::Detection d = rx::detection(cand);
        const tk::VerifyArgs a = verify_args(fit, tau_range, erase_s);
        std::optional<tk::LTrackResult> r;
        {
            py::gil_scoped_release nogil;
            r.emplace(lt::verify_track_l(c, s, d, a));
        }
        const bool same = !a.fit && d.timing && same_timing(r->timing, *d.timing);
        r->chan = nullptr;   // the row puts the caller's object back
        return py::make_tuple(rx::to_py(static_cast<const tk::TrackResult&>(*r)), same);
    }, py::arg("chan"), py::arg("spec"), py::arg("cand"), py::kw_only(), py::arg("fit") = true,
       py::arg("tau_range") = py::none(), py::arg("erase_s") = py::tuple());

    m.def("resid_ratio", &resid_ratio_py, py::arg("tr"), py::arg("c_known"), py::arg("gain"), py::arg("dist"));

    m.def("verify_l", [](const py::object& ch, const py::object& ch_fs, const py::object& spec,
                         const py::object& cand, const py::object& fit, const py::object& tau_range,
                         const py::object& erase_s) -> py::object {
        (void)ch_fs;   // the reference does not read it either
        const fr::FrameSpec s = rx::spec(spec);
        lt::l_only(s, "ltrack.verify_l");
        const auto c = chan_full(ch);
        const types::Detection d = rx::detection(cand);
        const tk::VerifyArgs a = verify_args(fit, tau_range, erase_s);
        std::optional<types::Detection> out;
        {
            py::gil_scoped_release nogil;
            out = lt::verify_l(c, s, d, a);
        }
        if (!out) return py::none();
        return rx::to_py(*out);
    }, py::arg("ch"), py::arg("ch_fs"), py::arg("spec"), py::arg("cand"), py::kw_only(), py::arg("fit") = true,
       py::arg("tau_range") = py::none(), py::arg("erase_s") = py::tuple());
}

const Registrar reg("ltrack_timing", 81, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
