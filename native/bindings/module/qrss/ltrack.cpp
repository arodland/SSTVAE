// sstvae_native.qrss.ltrack: bindings of native/core/qrss/ltrack (the
// waveform L tracker of sstvae/qrss/ltrack.py). Names, arguments and
// defaults follow the reference's, so tests/qrss_native/ltrack.py can
// assign them over it:
//
//   _l_only(spec, what)
//   pilot_table() -> "MfTable" dict, its tab a read-only view of the C++ table
//   _pilot_taps()
//   pilot_filter(x, tp)                  complex for a complex x, else float64;
//                                        tp's shape
//   data_mix(z_d, chan, timing, n_pos, idx=None)
//   data_mf(z_d, chan, timing, pos, n_pos)
//   make_classes_l(spec, hdr_bits=None, prior=None) -> "LClasses" dict, its
//                                        hdr_bits np.asarray(hdr_bits, uint8)
//   _distortion(spec)
//   data_templates(spec, grid, classes, withhold_known=False) -> 5-tuple
//   _pilot_gain(tau, spec, lead_in_s=0.0), pilot_wave(tau, spec, keying=None,
//                                        lead_in_s=0.0)   tau's shape
//   _q_near_span(q, grid) -> (lo, hi)    the row slices the grid itself
//   _model_init(tr, spec, classes, withhold_known, keying, lead, n_lead) -> dict
//                                        of _Model's arrays (the row sets
//                                        tr.meas, tr.meas_pl)
//   _model_streams(model, Pu_loc, delta=1, data=True, pilot=True, data_mask=None)
//   _model_fused(...same...), _model_pu_initial(model)
//                                        model: any object with _Model's
//                                        arrays and its .tr, .spec, read now
//   _pu_local(u, P, measured), _hold_ends(u, P, R)
//   _lead_supported_l(chan, z_d, tm, spec, lead)
//   _fit_delta(model, v, Pv, Pu_loc, delta0=1+0j)
//   track_l(chan, spec, det, classes=None, cw_known=None, prior=None, *,
//           timing=None, freq=None, withhold_known=False, outer=2, refine=True,
//           lead_in_s=None, z_d=None)
//       -> ("LTrackResult" dict with chan None, keying (the reference's
//          np.asarray(k, uint8) or None), same_freq, same_timing, same_zd):
//          same_* say the result's path / timing / z_d are the ones passed in
//          (or det's), so the row can hand back those very objects
//   refine_timing_l(tr, iters=3, u_ref=None, P_ref=None) -> "Timing" dict or None
//   report_l(tr, kappa=1.0, suspect=False) -> "TrackReport" dict
//   known_symbols_l(spec, hdr_bits=None) -> "Known" dict
//
// A track is read as the reference reads it: by attribute (or a dict's
// key), and only the fields the function uses. The capture's noise table
// is its `_psd`: when that is still None, `track_l` and `_lead_supported_l`
// call chan.noise_psd_at(0.0) first (whatever implements it fills it), so
// the C++ interpolates the very table the reference would; `report_l` asks
// tr.chan.noise_floor().

#include "qrss/rx_types.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
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
#include "qrss/lwave.hpp"
#include "qrss/rational.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/track.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace lt = sstvae::qrss::ltrack;
namespace tk = sstvae::qrss::track;
namespace lw = sstvae::qrss::lwave;
namespace fr = sstvae::qrss::frame;
namespace cfg = sstvae::qrss::config;
namespace types = sstvae::qrss::types;
using cd = std::complex<double>;

[[noreturn]] void raise(PyObject* type, const std::string& what) {
    PyErr_SetString(type, what.c_str());
    throw py::error_already_set();
}

py::object numpy() { return rx::np_module(); }

py::array asarray(const py::handle& o) {
    py::array a = py::array::ensure(numpy().attr("asarray")(o));
    if (!a) throw py::error_already_set();
    return a;
}

std::vector<py::ssize_t> pyshape(const py::array& a) { return std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim()); }

template <typename T>
py::array shaped(const std::vector<T>& v, const std::vector<py::ssize_t>& shape) {
    py::array_t<T> out(shape);
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

// `a * b` evaluated by Python: raises exactly what the reference's product
// with a None operand raises.
[[noreturn]] void python_product(const py::handle& a, const py::handle& b) {
    py::object r = py::reinterpret_steal<py::object>(PyNumber_Multiply(a.ptr(), b.ptr()));
    if (!r) throw py::error_already_set();
    raise(PyExc_TypeError, "unsupported operand type(s) for *");   // (numpy raises before this)
}

[[noreturn]] void not_subscriptable() { raise(PyExc_TypeError, "'NoneType' object is not subscriptable"); }

// --- what the reference reads -------------------------------------------------

// A Timing as rx_index reads it: tau0, ppm and gamma.
types::Timing timing_lite(const py::handle& o) {
    if (o.is_none()) raise(PyExc_AttributeError, "'NoneType' object has no attribute 'tau0'");
    types::Timing t;
    t.tau0 = rx::as_double(rx::field(o, "tau0"));
    t.ppm = rx::as_double(rx::field(o, "ppm"));
    t.gamma = rx::as_double(rx::field(o, "gamma"));
    return t;
}

double t0_of(const py::handle& chan) {
    if (chan.is_none()) raise(PyExc_AttributeError, "'NoneType' object has no attribute 't0_index'");
    return rx::as_double(rx::field(chan, "t0_index"));
}

bool is_pathfn(const py::handle& o) {
    if (py::isinstance<tk::PathFn>(o)) return true;
    if (py::hasattr(o, "_native") && py::isinstance<tk::PathFn>(o.attr("_native"))) return true;
    return py::hasattr(o, "_spl") && py::hasattr(o, "_line") && py::hasattr(o, "lo") && py::hasattr(o, "hi");
}

tk::PathFn path_of(const py::handle& o) {
    if (!is_pathfn(o))
        raise(PyExc_TypeError, "the C++ tracker needs a track.PathFn as its frequency path, not " +
                                   std::string(py::str(py::type::of(o).attr("__name__"))));
    return rx::path_fn(o);
}

tk::PathFn no_path() { return tk::PathFn::from_parts(0.0, 0.0, {0.0, 0.0}, std::nullopt); }

// chan.noise_psd_at(0.0) when chan._psd is still None: fills the table by
// whatever implements the method (the reference caches it there too).
void ensure_psd(const py::handle& chan) {
    if (py::isinstance<py::dict>(chan) || !py::hasattr(chan, "_psd")) return;
    if (chan.attr("_psd").is_none()) chan.attr("noise_psd_at")(0.0);
}

std::shared_ptr<const tk::ChanCapture> chan_full(const py::handle& chan) {
    ensure_psd(chan);
    return std::make_shared<const tk::ChanCapture>(rx::chan_capture(chan));
}

// A capture holding only its t0_index (what data_mf reads of tr.chan).
std::shared_ptr<const tk::ChanCapture> chan_t0(const py::handle& o) {
    if (o.is_none()) return nullptr;
    auto c = std::make_shared<tk::ChanCapture>();
    c->t0_index = rx::as_double(rx::field(o, "t0_index"));
    return c;
}

std::shared_ptr<const std::vector<cd>> zd_of(const py::handle& o) {
    return std::make_shared<const std::vector<cd>>(rx::vec<cd>(o));
}

// LClasses as data_templates and the tracker read them: mu, nu, known, gain.
tk::LClasses classes_lite(const py::handle& o) {
    tk::LClasses c;
    c.mu = rx::vec<cd>(rx::field(o, "mu"));
    c.nu = rx::vec<double>(rx::field(o, "nu"));
    c.known = rx::vec_bool(rx::field(o, "known"));
    c.gain = rx::vec<double>(rx::field(o, "gain"));
    return c;
}

std::optional<types::EmPrior> prior_of(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    return types::EmPrior{rx::vec<double>(rx::field(o, "x_hat")), rx::vec<double>(rx::field(o, "v"))};
}

// The keying argument as lwave reads it: np.asarray(keying).
std::optional<lw::Keying> lkeying(const py::object& obj) {
    if (obj.is_none()) return std::nullopt;
    py::array a = asarray(obj);
    lw::Keying k;
    k.shape.assign(a.shape(), a.shape() + a.ndim());
    const char kind = a.dtype().kind();
    if (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f') {
        k.values = rx::vec<double>(a);
    } else if (kind == 'c') {
        const auto re = rx::vec<double>(a.attr("real")), im = rx::vec<double>(a.attr("imag"));
        k.values.resize(re.size());
        for (std::size_t i = 0; i < re.size(); ++i)
            k.values[i] = im[i] != 0 ? std::numeric_limits<double>::quiet_NaN() : re[i];
    } else {
        k.values.assign(static_cast<std::size_t>(a.size()), std::numeric_limits<double>::quiet_NaN());
    }
    return k;
}

const lw::Keying* ptr(const std::optional<lw::Keying>& k) { return k ? &*k : nullptr; }

// lwave's lead-in refusal with the reference's f-string text of lead_in_s.
template <typename F>
auto with_lead(const py::handle& lead_obj, F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const lw::LeadInRefusal&) {
        throw py::value_error(lw::lead_in_refusal(py::str(lead_obj).cast<std::string>()));
    }
}

// `a, b = obj` as CPython 3.13 unpacks it.
std::pair<py::object, py::object> unpack2(const py::handle& obj) {
    if (!py::isinstance<py::iterable>(obj))
        raise(PyExc_TypeError, std::string("cannot unpack non-iterable ") + Py_TYPE(obj.ptr())->tp_name + " object");
    py::list items;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(obj))) {
        items.append(h);
        if (items.size() > 2) raise(PyExc_ValueError, "too many values to unpack (expected 2)");
    }
    if (items.size() < 2)
        raise(PyExc_ValueError, "not enough values to unpack (expected 2, got " + std::to_string(items.size()) + ")");
    return {items[0], items[1]};
}

// _keying_tuple(cw_known): (None or np.asarray(k, uint8), bool(ook)).
std::pair<py::object, bool> keying_tuple(const py::object& cw_known) {
    if (cw_known.is_none()) return {py::none(), false};
    const auto [k, ook] = unpack2(cw_known);
    py::object arr = k.is_none() ? py::object(py::none()) : numpy().attr("asarray")(k, py::arg("dtype") = "uint8");
    return {arr, rx::as_bool(ook)};
}

// hdr_bits as make_classes_l / known_symbols_l take them: the uint8 array
// np.asarray(hdr_bits, dtype=np.uint8) is (Python object, its values).
struct Bits {
    py::object arr;
    std::vector<std::uint8_t> values;
};

std::optional<Bits> bits_u8(const py::object& obj) {
    if (obj.is_none()) return std::nullopt;
    Bits b;
    b.arr = numpy().attr("asarray")(obj, py::arg("dtype") = "uint8");
    b.values = from_numpy<std::uint8_t>(numpy().attr("ravel")(b.arr));
    return b;
}

// sequences.qpsk's refusal of bits that are not 1-D (the C++ takes the values).
void qpsk_shape(const Bits& b) {
    const py::array a = asarray(b.arr);
    if (a.ndim() != 1)
        raise(PyExc_ValueError, "qpsk takes an even number of bits, not shape " +
                                    py::str(a.attr("shape")).cast<std::string>());
}

// --- tables -----------------------------------------------------------------------

py::array view(std::span<const double> v) {
    py::array_t<double> a(std::vector<py::ssize_t>{static_cast<py::ssize_t>(v.size())},
                          std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(double))}, v.data(),
                          py::capsule(v.data(), [](void*) {}));
    a.attr("setflags")(py::arg("write") = false);
    return a;
}

// --- results ----------------------------------------------------------------------

bool same_bits(double a, double b) { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

bool same_vec(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!same_bits(a[i], b[i])) return false;
    return true;
}

bool same_path(const tk::PathFn& a, const tk::PathFn& b) {
    if (!same_bits(a.lo(), b.lo()) || !same_bits(a.hi(), b.hi()) || !same_vec(a.line(), b.line())) return false;
    if (a.spline().has_value() != b.spline().has_value()) return false;
    if (!a.spline()) return true;
    return same_vec(a.spline()->t, b.spline()->t) && same_vec(a.spline()->c, b.spline()->c) &&
           a.spline()->k == b.spline()->k;
}

bool same_timing(const types::Timing& a, const types::Timing& b) {
    return same_bits(a.tau0, b.tau0) && same_bits(a.ppm, b.ppm) && same_bits(a.gamma, b.gamma) &&
           same_bits(a.z, b.z) && a.cov.rows == b.cov.rows && a.cov.cols == b.cov.cols && same_vec(a.cov.data, b.cov.data);
}

cd as_complex(const py::handle& o) {
    py::object c = py::module_::import("builtins").attr("complex")(o);
    return c.cast<cd>();
}

// --- the model ----------------------------------------------------------------------

lt::Model model_of(const py::handle& m) {
    lt::Model md;
    md.c = rx::vec<cd>(m.attr("c"));
    md.nu = rx::vec<double>(m.attr("nu"));
    md.gain = rx::vec<double>(m.attr("gain"));
    md.dist = rx::vec<double>(m.attr("dist"));
    md.meas_d = rx::vec_bool(m.attr("meas_d"));
    md.meas_p = rx::vec_bool(m.attr("meas_p"));
    md.k_pl = rx::vec<double>(m.attr("k_pl"));
    return md;
}

// The model's track, with the fields streams/fused/pu_initial read.
struct ModelTrack {
    py::object tr;
    tk::LTrackResult t;
};

ModelTrack model_track(const py::handle& m) {
    const py::object tr = m.attr("tr");
    ModelTrack out{tr, tk::LTrackResult(rx::spec(m.attr("spec")), nullptr, types::Timing{}, no_path())};
    tk::LTrackResult& t = out.t;
    t.pos = rx::vec<std::int64_t>(rx::field(tr, "pos"));
    t.psi = rx::vec_or_empty<double>(rx::field(tr, "psi"));
    t.m = rx::vec_or_empty<cd>(rx::field(tr, "m"));
    t.N = rx::vec_or_empty<double>(rx::field(tr, "N"));
    t.c_pl = rx::opt_vec<cd>(rx::field(tr, "c_pl"));
    t.psi_pl = rx::opt_vec<double>(rx::field(tr, "psi_pl"));
    t.m_pl = rx::opt_vec<cd>(rx::field(tr, "m_pl"));
    t.N_pl = rx::opt_vec<double>(rx::field(tr, "N_pl"));
    return out;
}

// streams' None refusals in the reference's order: c * psi, m[ok], N[ok];
// c_pl * psi_pl, m_pl[ok], N_pl[ok].
void streams_refusals(const py::handle& m, const py::object& tr, bool data, bool pilot) {
    if (data) {
        const py::object psi = rx::field(tr, "psi");
        if (psi.is_none()) python_product(m.attr("c"), psi);
        if (rx::field(tr, "m").is_none() || rx::field(tr, "N").is_none()) not_subscriptable();
    }
    if (pilot) {
        const py::object c_pl = rx::field(tr, "c_pl"), psi_pl = rx::field(tr, "psi_pl");
        if (c_pl.is_none() || psi_pl.is_none()) python_product(c_pl, psi_pl);
        if (rx::field(tr, "m_pl").is_none() || rx::field(tr, "N_pl").is_none()) not_subscriptable();
    }
}

std::optional<std::vector<std::uint8_t>> mask_of(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    return rx::vec_bool(o);
}

py::object model_init_py(const py::object& tr, const py::object& spec, const py::object& classes,
                         const py::object& withhold_known, const py::object& keying, const py::object& lead,
                         const py::object& n_lead) {
    const fr::FrameSpec s = rx::spec(spec);
    tk::LTrackResult t(s, nullptr, types::Timing{}, no_path());
    t.pos = rx::vec<std::int64_t>(rx::field(tr, "pos"));
    const tk::LClasses c = classes_lite(classes);
    const lw::Keying dummy;   // _Model reads only whether keying is None
    const double ld = rx::as_double(lead);
    const lt::Model md = with_lead(lead, [&] {
        return lt::Model(t, s, c, rx::as_bool(withhold_known), keying.is_none() ? nullptr : &dummy, ld,
                         rx::as_int(n_lead));
    });
    py::dict d;
    d["c_all"] = rx::arr(md.c_all);
    d["c"] = rx::arr(md.c);
    d["nu"] = rx::arr(md.nu);
    d["gain"] = rx::arr(md.gain);
    d["dist"] = rx::arr(md.dist);
    d["is_sym"] = rx::arr_bool(md.is_sym);
    d["is_known"] = rx::arr_bool(md.is_known);
    d["meas_d"] = rx::arr_bool(md.meas_d);
    d["need"] = rx::arr_bool(md.need);
    d["meas_p"] = rx::arr_bool(md.meas_p);
    d["measured"] = rx::arr_bool(md.measured);
    d["k_pl"] = rx::arr(md.k_pl);
    return d;
}

py::object model_streams_py(const py::object& model, const py::object& pu_loc, const py::object& delta,
                            const py::object& data, const py::object& pilot, const py::object& data_mask) {
    const bool dt = rx::as_bool(data), pl = rx::as_bool(pilot);
    const lt::Model md = model_of(model);
    const ModelTrack mt = model_track(model);
    streams_refusals(model, mt.tr, dt, pl);
    const std::vector<double> pu = rx::vec<double>(pu_loc);
    const cd dl = as_complex(delta);
    const auto mask = mask_of(data_mask);
    lt::Streams s;
    {
        py::gil_scoped_release nogil;
        s = md.streams(mt.t, pu, dl, dt, pl, mask ? &*mask : nullptr);
    }
    py::list out;
    out.append(py::make_tuple(rx::arr(s.mu_d), rx::arr(s.R_d)));
    out.append(py::make_tuple(rx::arr(s.mu_p), rx::arr(s.R_p)));
    return out;
}

py::object model_fused_py(const py::object& model, const py::object& pu_loc, const py::object& delta,
                          const py::object& data, const py::object& pilot, const py::object& data_mask) {
    const bool dt = rx::as_bool(data), pl = rx::as_bool(pilot);
    const lt::Model md = model_of(model);
    const ModelTrack mt = model_track(model);
    streams_refusals(model, mt.tr, dt, pl);
    const std::vector<double> pu = rx::vec<double>(pu_loc);
    const cd dl = as_complex(delta);
    const auto mask = mask_of(data_mask);
    lt::Fused f;
    {
        py::gil_scoped_release nogil;
        f = md.fused(mt.t, pu, dl, dt, pl, mask ? &*mask : nullptr);
    }
    return py::make_tuple(rx::arr(f.mu), rx::arr(f.R));
}

double model_pu_initial_py(const py::object& model) {
    const lt::Model md = model_of(model);
    const ModelTrack mt = model_track(model);
    const py::object c_pl = rx::field(mt.tr, "c_pl"), psi_pl = rx::field(mt.tr, "psi_pl");
    if (c_pl.is_none() || psi_pl.is_none()) python_product(c_pl, psi_pl);
    const py::object psi = rx::field(mt.tr, "psi");
    try {
        return md.pu_initial(mt.t);
    } catch (const std::invalid_argument&) {
        // the data part's c * psi with psi None (after the pilot part, as the reference)
        if (psi.is_none()) python_product(model.attr("c"), psi);
        throw;
    }
}

// --- the tracker ------------------------------------------------------------------

py::object track_l_py(const py::object& chan, const py::object& spec, const py::object& det, const py::object& classes,
                      const py::object& cw_known, const py::object& prior, const py::object& timing,
                      const py::object& freq, const py::object& withhold_known, const py::object& outer,
                      const py::object& refine, const py::object& lead_in_s, const py::object& z_d) {
    const fr::FrameSpec s = rx::spec(spec);
    lt::l_only(s, "ltrack.track_l");
    lt::TrackLArgs a;
    if (classes.is_none()) {
        const auto p = prior_of(prior);
        a.classes = lt::make_classes_l(s, nullptr, p ? &*p : nullptr);
    } else {
        a.classes = classes_lite(classes);
    }
    const auto [k, ook] = keying_tuple(cw_known);
    if (ook) raise(PyExc_ValueError, "waveform L keys its callsign by FSK only: cw_known must say ook=False");
    a.keying = lkeying(k);
    types::Detection d;
    if (freq.is_none())
        d.path = rx::freq_path(rx::field(det, "path"));
    else
        a.freq = path_of(freq);
    const py::object tm = timing.is_none() ? rx::field(det, "timing") : timing;
    if (tm.is_none()) raise(PyExc_ValueError, "track_l needs a timing (fit_timing_l, or gate V's)");
    a.timing = rx::timing(tm);
    const py::object lead = lead_in_s.is_none() ? rx::field(det, "lead_in_s") : lead_in_s;
    a.lead_in_s = rx::as_double(lead);
    a.withhold_known = rx::as_bool(withhold_known);
    a.outer = static_cast<int>(rx::as_int(outer));
    a.refine = rx::as_bool(refine);
    if (!z_d.is_none()) a.z_d = zd_of(z_d);
    const auto c = chan_full(chan);
    std::optional<tk::LTrackResult> r;
    {
        py::gil_scoped_release nogil;
        r.emplace(lt::track_l(c, s, d, a));
    }
    const bool same_f = a.freq && same_path(r->freq, *a.freq);
    const bool same_t = same_timing(r->timing, *a.timing);
    const bool same_z = a.z_d && r->z_d == a.z_d;
    r->chan = nullptr;   // the row puts the caller's object back
    return py::make_tuple(rx::to_py(static_cast<const tk::TrackResult&>(*r)), k, same_f, same_t, same_z);
}

py::object refine_timing_l_py(const py::object& tr, const py::object& iters, const py::object& u_ref,
                              const py::object& P_ref) {
    const fr::FrameSpec s = rx::spec(rx::field(tr, "spec"));
    lt::l_only(s, "ltrack.refine_timing_l");
    tk::LTrackResult t(s, chan_t0(rx::field(tr, "chan")), rx::timing(rx::field(tr, "timing")), no_path());
    t.pos = rx::vec<std::int64_t>(rx::field(tr, "pos"));
    const py::object cl = rx::field(tr, "classes");
    if (!cl.is_none()) {
        if (py::isinstance<py::dict>(cl) ? py::reinterpret_borrow<py::dict>(cl).contains("gain")
                                         : py::hasattr(cl, "gain")) {
            tk::LClasses c;
            c.known = rx::vec_bool(rx::field(cl, "known"));
            c.gain = rx::vec<double>(rx::field(cl, "gain"));
            t.classes = std::move(c);
        } else {
            tk::Classes c;
            c.known = rx::vec_bool(rx::field(cl, "known"));
            t.classes = std::move(c);
        }
    }
    t.psi = rx::vec_or_empty<double>(rx::field(tr, "psi"));
    t.c = rx::vec_or_empty<cd>(rx::field(tr, "c"));
    t.N = rx::vec_or_empty<double>(rx::field(tr, "N"));
    t.pu = rx::vec_or_empty<double>(rx::field(tr, "pu"));
    t.u_pl = rx::opt_vec<cd>(rx::field(tr, "u_pl"));
    t.P_pl = rx::opt_vec<double>(rx::field(tr, "P_pl"));
    if (u_ref.is_none() && !t.u_pl) t.u = rx::vec_or_empty<cd>(rx::field(tr, "u"));
    if (P_ref.is_none() && !t.u_pl) t.P = rx::vec_or_empty<double>(rx::field(tr, "P"));
    t.delta = as_complex(rx::field(tr, "delta"));
    const py::object zd = rx::field(tr, "z_d");
    if (!zd.is_none()) t.z_d = zd_of(zd);
    std::optional<std::vector<cd>> u;
    std::optional<std::vector<double>> P;
    if (!u_ref.is_none()) u = rx::vec<cd>(u_ref);
    if (!P_ref.is_none()) P = rx::vec<double>(P_ref);
    const int it = static_cast<int>(rx::as_int(iters));
    std::optional<types::Timing> out;
    try {
        py::gil_scoped_release nogil;
        out = lt::refine_timing_l(t, it, u ? &*u : nullptr, P ? &*P : nullptr);
    } catch (const std::invalid_argument& e) {
        // what the reference raises reading tr.classes.gain of CE classes,
        // tr.chan.t0_index of None or len(None) of z_d
        const std::string w = e.what();
        if (w == "'Classes' object has no attribute 'gain'") raise(PyExc_AttributeError, w);
        if (w.find("no channel") != std::string::npos)
            raise(PyExc_AttributeError, "'NoneType' object has no attribute 't0_index'");
        if (w.find("no z_d") != std::string::npos) raise(PyExc_TypeError, "object of type 'NoneType' has no len()");
        throw;
    }
    if (!out) return py::none();
    return rx::to_py(*out);
}

void bind(py::module_& q) {
    rx::register_translators();
    demod_rx::register_wrong_waveform();   // a CE frame's AssertionError
    py::module_ m = q.def_submodule("ltrack", "native/core/qrss/ltrack (sstvae.qrss.ltrack's tracker)");

    m.def("_l_only", [](const py::object& spec, const std::string& what) { lt::l_only(rx::spec(spec), what); },
          py::arg("spec"), py::arg("what"));

    m.def("pilot_table", [] {
        const tk::MfTable& t = lt::pilot_table();
        py::dict d = rx::typed("MfTable");
        d["t_sym"] = t.t_sym();
        d["sps"] = t.sps();
        d["half"] = t.half();
        d["tab"] = view(t.tab());
        d["g0"] = t.g0();
        return d;
    });

    m.def("_pilot_taps", [] { return to_numpy(lt::pilot_taps()); });

    m.def("pilot_filter", [](const py::object& x, const py::object& tp) -> py::object {
        const py::array xa = asarray(x);
        if (xa.ndim() != 1) raise(PyExc_ValueError, "in1 and in2 should have the same dimensionality");
        const py::array ta = asarray(numpy().attr("asarray")(tp, py::arg("dtype") = "float64"));
        const std::vector<double> t = rx::vec<double>(ta);
        if (xa.dtype().kind() == 'c') {
            const std::vector<cd> v = rx::vec<cd>(xa);
            std::vector<cd> out;
            {
                py::gil_scoped_release nogil;
                out = lt::pilot_filter(std::span<const cd>(v), t);
            }
            return shaped(out, pyshape(ta));
        }
        const std::vector<double> v = rx::vec<double>(xa);
        std::vector<double> out;
        {
            py::gil_scoped_release nogil;
            out = lt::pilot_filter(std::span<const double>(v), t);
        }
        return shaped(out, pyshape(ta));
    }, py::arg("x"), py::arg("tp"));

    m.def("data_mix", [](const py::object& z_d, const py::object& chan, const py::object& timing,
                         const py::object& n_pos, const py::object& idx) {
        std::optional<std::vector<std::int64_t>> ix;
        if (!idx.is_none()) ix = rx::vec<std::int64_t>(idx);
        const double t0 = t0_of(chan);
        const types::Timing tm = timing_lite(timing);
        const std::int64_t n = rx::as_int(n_pos);
        const std::vector<cd> z = rx::vec<cd>(z_d);
        std::vector<cd> out;
        {
            py::gil_scoped_release nogil;
            if (ix)
                out = lt::data_mix(z, t0, tm, n, std::span<const std::int64_t>(*ix));
            else
                out = lt::data_mix(z, t0, tm, n);
        }
        return to_numpy(out);
    }, py::arg("z_d"), py::arg("chan"), py::arg("timing"), py::arg("n_pos"), py::arg("idx") = py::none());

    m.def("data_mf", [](const py::object& z_d, const py::object& chan, const py::object& timing,
                        const py::object& pos, const py::object& n_pos) {
        const std::vector<double> p = rx::vec<double>(pos);
        if (p.empty()) return to_numpy(std::vector<cd>{});
        const double t0 = t0_of(chan);
        const types::Timing tm = timing_lite(timing);
        const std::int64_t n = rx::as_int(n_pos);
        const std::vector<cd> z = rx::vec<cd>(z_d);
        std::vector<cd> out;
        {
            py::gil_scoped_release nogil;
            out = lt::data_mf(z, t0, tm, p, n);
        }
        return to_numpy(out);
    }, py::arg("z_d"), py::arg("chan"), py::arg("timing"), py::arg("pos"), py::arg("n_pos"));

    m.def("make_classes_l", [](const py::object& spec, const py::object& hdr_bits, const py::object& prior) {
        const fr::FrameSpec s = rx::spec(spec);
        lt::l_only(s, "ltrack.make_classes_l");
        const auto b = bits_u8(hdr_bits);
        if (b) {
            if (s.n_hdr()) qpsk_shape(*b);
            else if (s.hdr_rho() > 0 && asarray(b->arr).ndim() != 1)
                raise(PyExc_ValueError, "hdr_bits must be " + std::to_string(cfg::N_HDR_BITS) + " values of 0/1");
        }
        const auto p = prior_of(prior);
        py::dict d = rx::to_py(lt::make_classes_l(s, b ? &b->values : nullptr, p ? &*p : nullptr));
        if (b) d["hdr_bits"] = b->arr;
        return d;
    }, py::arg("spec"), py::arg("hdr_bits") = py::none(), py::arg("prior") = py::none());

    m.def("_distortion", [](const py::object& spec) { return to_numpy(lt::distortion(rx::spec(spec))); },
          py::arg("spec"));

    m.def("data_templates", [](const py::object& spec, const py::object& grid, const py::object& classes,
                               const py::object& withhold_known) {
        const fr::FrameSpec s = rx::spec(spec);
        const std::vector<std::int64_t> g = rx::vec<std::int64_t>(grid);
        const tk::LClasses c = classes_lite(classes);
        const lt::DataTemplates t = lt::data_templates(s, g, c, rx::as_bool(withhold_known));
        return py::make_tuple(rx::arr(t.c), rx::arr(t.nu), rx::arr(t.gain), rx::arr(t.dist), rx::arr_bool(t.is_sym));
    }, py::arg("spec"), py::arg("grid"), py::arg("classes"), py::arg("withhold_known") = false);

    m.def("_pilot_gain", [](const py::object& tau, const py::object& spec, const py::object& lead_in_s) {
        const py::array ta = asarray(numpy().attr("asarray")(tau, py::arg("dtype") = "float64"));
        const std::vector<double> t = rx::vec<double>(ta);
        const fr::FrameSpec s = rx::spec(spec);
        const double lead = rx::as_double(lead_in_s);
        return shaped(with_lead(lead_in_s, [&] { return lt::pilot_gain(t, s, lead); }), pyshape(ta));
    }, py::arg("tau"), py::arg("spec"), py::arg("lead_in_s") = 0.0);

    m.def("pilot_wave", [](const py::object& tau, const py::object& spec, const py::object& keying,
                           const py::object& lead_in_s) {
        const py::array ta = asarray(numpy().attr("asarray")(tau, py::arg("dtype") = "float64"));
        const std::vector<double> t = rx::vec<double>(ta);
        const fr::FrameSpec s = rx::spec(spec);
        const auto k = lkeying(keying);
        const double lead = rx::as_double(lead_in_s);
        return shaped(with_lead(lead_in_s, [&] { return lt::pilot_wave(t, s, ptr(k), lead); }), pyshape(ta));
    }, py::arg("tau"), py::arg("spec"), py::arg("keying") = py::none(), py::arg("lead_in_s") = 0.0);

    m.def("_q_near_span", [](const py::object& q, const py::object& grid) {
        const std::vector<double> g = rx::vec<double>(numpy().attr("array")(grid));
        const auto [lo, hi] = lt::q_near_span(rx::as_double(q), g);
        return py::make_tuple(lo, hi);
    }, py::arg("q"), py::arg("grid"));

    m.def("_model_init", &model_init_py, py::arg("tr"), py::arg("spec"), py::arg("classes"),
          py::arg("withhold_known"), py::arg("keying"), py::arg("lead"), py::arg("n_lead"));
    m.def("_model_streams", &model_streams_py, py::arg("model"), py::arg("Pu_loc"),
          py::arg("delta") = cd(1.0, 0.0), py::arg("data") = true, py::arg("pilot") = true,
          py::arg("data_mask") = py::none());
    m.def("_model_fused", &model_fused_py, py::arg("model"), py::arg("Pu_loc"), py::arg("delta") = cd(1.0, 0.0),
          py::arg("data") = true, py::arg("pilot") = true, py::arg("data_mask") = py::none());
    m.def("_model_pu_initial", &model_pu_initial_py, py::arg("model"));

    m.def("_pu_local", [](const py::object& u, const py::object& P, const py::object& measured) {
        const std::vector<cd> uv = rx::vec<cd>(u);
        const std::vector<double> pv = rx::vec<double>(P);
        const std::vector<std::uint8_t> mv = rx::vec_bool(measured);
        lt::PuLocal r;
        {
            py::gil_scoped_release nogil;
            r = lt::pu_local(uv, pv, mv);
        }
        return py::make_tuple(rx::arr(r.loc), r.g);
    }, py::arg("u"), py::arg("P"), py::arg("measured"));

    m.def("_hold_ends", [](const py::object& u, const py::object& P, const py::object& R) -> py::object {
        const std::vector<double> rv = rx::vec<double>(R);
        const bool complex_u = asarray(u).dtype().kind() == 'c';
        std::vector<cd> uv = rx::vec<cd>(u);
        std::vector<double> pv = rx::vec<double>(P);
        if (!lt::hold_ends(uv, pv, rv)) return py::make_tuple(u, P);
        py::object uo;
        if (complex_u) {
            uo = rx::arr(uv);
        } else {
            std::vector<double> re(uv.size());
            for (std::size_t i = 0; i < uv.size(); ++i) re[i] = uv[i].real();
            uo = rx::arr(re);
        }
        return py::make_tuple(uo, rx::arr(pv));
    }, py::arg("u"), py::arg("P"), py::arg("R"));

    m.def("_lead_supported_l", [](const py::object& chan, const py::object& z_d, const py::object& tm,
                                  const py::object& spec, const py::object& lead) {
        const double l = rx::as_double(lead);
        const fr::FrameSpec s = rx::spec(spec);
        // n < 4 + PILOT_REACH: True before anything else is read
        // (and a NaN or infinite count's refusal, also before anything is read)
        if (!(std::floor(std::min(l, sstvae::qrss::rxconfig::track::LEAD_KEEP_S) / cfg::L_T_SYM) >=
              4 + sstvae::qrss::rxconfig::ltrack::PILOT_REACH))
            return with_lead(lead, [&] {
                return lt::lead_supported_l(tk::ChanCapture{}, {}, types::Timing{}, s, l);
            });
        ensure_psd(chan);
        tk::ChanCapture c;
        c.t0_index = t0_of(chan);
        const py::object psd = rx::field(chan, "_psd");
        py::tuple pt = py::tuple(psd);
        if (pt.size() != 2) raise(PyExc_ValueError, "ChanCapture._psd must be (t_s, psd)");
        c.psd.set(std::make_shared<const tk::NoisePsd>(tk::NoisePsd{rx::vec<double>(pt[0]), rx::vec<double>(pt[1])}));
        const std::vector<cd> z = rx::vec<cd>(z_d);
        const types::Timing t = timing_lite(tm);
        return with_lead(lead, [&] {
            py::gil_scoped_release nogil;
            return lt::lead_supported_l(c, z, t, s, l);
        });
    }, py::arg("chan"), py::arg("z_d"), py::arg("tm"), py::arg("spec"), py::arg("lead"));

    m.def("_fit_delta", [](const py::object& model, const py::object& v, const py::object& Pv,
                           const py::object& Pu_loc, const py::object& delta0) {
        const lt::Model md = model_of(model);
        const ModelTrack mt = model_track(model);
        streams_refusals(model, mt.tr, true, false);
        const std::vector<cd> vv = rx::vec<cd>(v);
        const std::vector<double> pv = rx::vec<double>(Pv), pu = rx::vec<double>(Pu_loc);
        const cd d0 = as_complex(delta0);
        cd out;
        {
            py::gil_scoped_release nogil;
            out = lt::fit_delta(md, mt.t, vv, pv, pu, d0);
        }
        return out;
    }, py::arg("model"), py::arg("v"), py::arg("Pv"), py::arg("Pu_loc"), py::arg("delta0") = cd(1.0, 0.0));

    m.def("track_l", &track_l_py, py::arg("chan"), py::arg("spec"), py::arg("det"), py::arg("classes") = py::none(),
          py::arg("cw_known") = py::none(), py::arg("prior") = py::none(), py::kw_only(),
          py::arg("timing") = py::none(), py::arg("freq") = py::none(), py::arg("withhold_known") = false,
          py::arg("outer") = 2, py::arg("refine") = true, py::arg("lead_in_s") = py::none(),
          py::arg("z_d") = py::none());

    m.def("refine_timing_l", &refine_timing_l_py, py::arg("tr"), py::arg("iters") = 3, py::arg("u_ref") = py::none(),
          py::arg("P_ref") = py::none());

    m.def("report_l", [](const py::object& tr, const py::object& kappa, const py::object& suspect) {
        const fr::FrameSpec s = rx::spec(rx::field(tr, "spec"));
        tk::LTrackResult t(s, nullptr, rx::timing(rx::field(tr, "timing")), path_of(rx::field(tr, "freq")));
        t.pos = rx::vec<std::int64_t>(rx::field(tr, "pos"));
        t.u = rx::vec_or_empty<cd>(rx::field(tr, "u"));
        t.pu = rx::vec_or_empty<double>(rx::field(tr, "pu"));
        t.q = rx::as_double(rx::field(tr, "q"));
        t.z_ref = rx::as_double(rx::field(tr, "z_ref"));
        const double k = rx::as_double(kappa);
        const bool sp = rx::as_bool(suspect);
        const double nf = rx::as_double(rx::field(tr, "chan").attr("noise_floor")());
        return rx::to_py(lt::report_l(t, k, sp, nf));
    }, py::arg("tr"), py::arg("kappa") = 1.0, py::arg("suspect") = false);

    m.def("known_symbols_l", [](const py::object& spec, const py::object& hdr_bits) {
        const fr::FrameSpec s = rx::spec(spec);
        lt::l_only(s, "ltrack.known_symbols_l");
        std::optional<Bits> b;
        if (!hdr_bits.is_none() && s.n_hdr()) {
            b = bits_u8(hdr_bits);
            qpsk_shape(*b);
        }
        return rx::to_py(lt::known_symbols_l(s, b ? &b->values : nullptr));
    }, py::arg("spec"), py::arg("hdr_bits") = py::none());
}

const Registrar reg("ltrack", 80, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
