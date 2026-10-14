// sstvae_native.qrss.receiver_slot: bindings of native/core/qrss/receiver_slot
// (the slot half of sstvae/qrss/receiver.py). Names and arguments follow
// the reference's; tests/qrss_native/receiver_slot.py wraps them with the
// reference's defaults and assigns them over it:
//
//   GHOST_HZ, GHOST_T, GHOST_Z_RATIO, ALIAS_HZ, ALIAS_RATIO, L_OWN_BELOW_HZ,
//   L_OWN_ABOVE_HZ, CE_RESID_MIN
//   _ghost_of(f_hz, tau0, g_f_hz, g_tau0, t_sym)
//   _aliases(dets, t_sym), _ghosts(dets, t_sym)     the caller's objects kept
//   _suspect_neighbour(p, out), _suspect_beside(p, out)
//   _a3_lines(cands) -> [(line, centre, Z)], _owned(d, found, a3)
//   _ce_resid_ratio(tr)
//   candidates(prep, spec, live_only, hints, typed)
//   detect(prep, spec, live_only, max_candidates, budget_s, erase_s, cands,
//          skip_hz, hints, typed)
//   _detect_joint(prep, specs, live_only, max_candidates, budget_s, erase_s,
//                 cands, skip_hz, hints, typed)
//   _gate(prep, spec, d, others, erase_s, typed)
//   receive_slot(cap, spec, live_only, estimator, erase_s, round_b, dets,
//                max_candidates, budget_s, hints, typed)
//   _receive_slot_joint(prep, specs, live_only, estimator, erase_s, round_b,
//                       dets, max_candidates, budget_s, hints, typed)
//   capture_from_wav(path, q, start_unix, typed) -> frontend.Capture
//   _read_wav(path) -> float64 (wavread.hpp: wavio.read_wav; tests only)
//   receive_wav(path, q, start_unix, spec, estimator)   (the C++ throughout)
//
// Two ways to run, one orchestration (receiver_slot_flow.hpp):
//
//   typed=True (the row says every stage the slot receiver reaches is the
//     native row, so nothing Python is to be called): the arguments become
//     the C++ records and the whole call runs in C++ with the GIL released
//     -- receiver_slot.hpp's functions, which call the C++ stages directly.
//     Results come back as "__type__" dicts (the row's to_ref makes them
//     the reference's). An argument that does not convert (a test's stub
//     prep, a detection without z_ref, a list holding one detection twice,
//     an np.int64 max_candidates ...) falls through to the second way.
//   otherwise: the same orchestration over the Python objects themselves,
//     every stage the module attribute the reference calls, resolved at
//     each call through receiver's own globals (receiver.acquire.acquire,
//     receiver.track.verify, receiver.time.monotonic, receiver.rules.old,
//     ...), so a test's replacement or spy is called with the reference's
//     arguments in the reference's order, and every object a stage returns
//     is handed on as it is.
//
// The rules (_aliases, _ghosts, _ghost_of, _owned, _a3_lines,
// _suspect_neighbour, _suspect_beside) are the C++ in both ways: the
// orchestration applies them itself, as the reference's functions would,
// and does not look them up in the module.

#include "qrss/rx_types.hpp"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "qrss/acquire.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/fspath.hpp"
#include "qrss/frontend.hpp"
#include "qrss/receiver.hpp"
#include "qrss/receiver_slot.hpp"
#include "qrss/receiver_slot_flow.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/wavread.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace rc = sstvae::qrss::receiver;
namespace fl = sstvae::qrss::receiver::flow;
namespace fr = sstvae::qrss::frame;
namespace aq = sstvae::qrss::acquire;
namespace tk = sstvae::qrss::track;
namespace fe = sstvae::qrss::frontend;
namespace cfg = sstvae::qrss::config;
namespace types = sstvae::qrss::types;
namespace wav = sstvae::qrss::wav;
using namespace pybind11::literals;

// --- small things ----------------------------------------------------------------------------

bool is_seq(const py::handle& o) { return py::isinstance<py::tuple>(o) || py::isinstance<py::list>(o); }

// getattr(o, name, default) (a dict's key for the C++'s own records)
py::object attr_or(const py::handle& o, const char* name, py::object dflt) {
    if (py::isinstance<py::dict>(o)) {
        py::dict d = py::reinterpret_borrow<py::dict>(o);
        return d.contains(name) ? py::object(d[name]) : dflt;
    }
    return py::getattr(o, name, dflt);
}

void set_field(const py::handle& o, const char* name, const py::object& v) {
    if (py::isinstance<py::dict>(o))
        py::reinterpret_borrow<py::dict>(o)[name] = v;
    else
        py::setattr(o, name, v);
}

py::object steal(PyObject* o) {
    if (!o) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(o);
}

const py::object& ce_int() {
    static py::object* v = new py::object(py::int_(cfg::WAVEFORM_CE));
    return *v;
}
const py::object& l_int() {
    static py::object* v = new py::object(py::int_(cfg::WAVEFORM_L));
    return *v;
}

// list(iterable) as a vector of its objects
std::vector<py::object> objects(const py::handle& o) {
    std::vector<py::object> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(o)))
        out.push_back(py::reinterpret_borrow<py::object>(h));
    return out;
}

// `line, cc, zz = h`, as Python unpacks it (at most four items taken)
std::array<py::object, 3> unpack3(const py::handle& h) {
    PyObject* it = PyObject_GetIter(h.ptr());
    if (!it) {
        PyTypeObject* t = Py_TYPE(h.ptr());
        if (PyErr_ExceptionMatches(PyExc_TypeError) && t->tp_iter == nullptr && !PySequence_Check(h.ptr())) {
            PyErr_Clear();
            throw py::type_error(std::string("cannot unpack non-iterable ") + t->tp_name + " object");
        }
        throw py::error_already_set();
    }
    const py::object iter = steal(it);
    std::array<py::object, 3> out;
    for (std::size_t k = 0; k < 3; ++k) {
        PyObject* v = PyIter_Next(it);
        if (!v) {
            if (PyErr_Occurred()) throw py::error_already_set();
            throw py::value_error("not enough values to unpack (expected 3, got " + std::to_string(k) + ")");
        }
        out[k] = steal(v);
    }
    if (PyObject* extra = PyIter_Next(it)) {
        Py_DECREF(extra);
        throw py::value_error("too many values to unpack (expected 3)");
    }
    if (PyErr_Occurred()) throw py::error_already_set();
    return out;
}

py::list list_of(const std::vector<py::object>& v) {
    py::list out;
    for (const py::object& o : v) out.append(o);
    return out;
}

py::tuple floats_tuple(const std::vector<double>& v) {
    py::tuple out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = py::float_(v[i]);
    return out;
}

// erase_s as the reference unpacks it: `for a, b in spans`, float(a), float(b)
std::vector<std::pair<double, double>> spans_of(const py::object& o) {
    std::vector<std::pair<double, double>> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(o))) {
        const std::vector<py::object> ab = objects(h);
        if (ab.size() != 2) throw py::value_error("an erase span is (a, b)");
        out.emplace_back(rx::as_double(ab[0]), rx::as_double(ab[1]));
    }
    return out;
}

// --- the Python policy -------------------------------------------------------------------------

// receiver_slot_flow.hpp's policy over the Python objects: every stage the
// module attribute the reference reaches, through receiver's own globals.
struct Py {
    using Det = py::object;
    using Pass = py::object;
    using Prep = py::object;
    using Cap = py::object;
    using AcqCap = py::object;
    using Sel = py::object;
    using Spec = py::object;
    using Chan = py::object;
    using Track = py::object;
    using FrameMap = py::object;
    using WKey = py::object;
    using Others = py::object;
    using Cands = py::object;
    using LiveOnly = py::object;
    using Estimator = py::object;
    using Erase = py::object;
    using RoundB = py::object;
    using MaxC = py::object;
    using Budget = py::object;
    using Hints = py::object;
    using Skip = py::object;

    py::module_ R = py::module_::import("sstvae.qrss.receiver");

    // receiver's global `name` (a function, or a module it imported)
    py::object g(const char* name) const { return R.attr(name); }

    // values
    static double f_hz(const py::object& d) { return rx::as_double(rx::field(d, "f_hz")); }
    double centre(const py::object& d) const { return rx::as_double(g("centre_hz")(d)); }
    static bool method_is(const py::object& d, const char* m) { return rx::field(d, "method").equal(py::str(m)); }
    static double a3_z(const py::object& d) {
        return rx::as_double(attr_or(d, "a3_z", py::float_(std::numeric_limits<double>::quiet_NaN())));
    }
    static double z_ref(const py::object& d) { return rx::as_double(rx::field(d, "z_ref")); }
    static double tau0(const py::object& d) { return rx::as_double(rx::field(rx::field(d, "timing"), "tau0")); }
    static double lead_in(const py::object& d) { return rx::as_double(rx::field(d, "lead_in_s")); }
    static void set_lead_in(py::object& d, const py::object& x) { set_field(d, "lead_in_s", x); }
    static bool is_l(const py::object& d) { return attr_or(d, "waveform", ce_int()).equal(l_int()); }
    static double p_f_hz(const py::object& x) { return f_hz(x); }
    static double p_tau0(const py::object& x) { return tau0(x); }
    static double p_z_ref(const py::object& x) { return rx::as_double(rx::field(rx::field(x, "report"), "z_ref")); }
    static bool p_suspect(const py::object& x) { return rx::as_bool(rx::field(rx::field(x, "report"), "suspect")); }
    double p_centre(const py::object& x) const { return centre(x); }
    static bool same_waveform(const py::object& a, const py::object& b) {
        return rx::field(a, "waveform").equal(rx::field(b, "waveform"));
    }

    // frames
    bool is_joint(const py::object& sel) const { return rx::as_bool(g("is_joint")(sel)); }
    py::object frames(const py::object& sel) const { return g("frames")(sel); }
    py::object single(const py::object& sel) const { return g("_single")(sel); }
    py::object frame_map(const py::object& specs) const { return g("acquire").attr("frame_map")(specs); }
    std::optional<py::object> smap_get(const py::object& smap, const py::object& d) const {
        py::object sp = smap.attr("get")(attr_or(d, "waveform", ce_int()));
        if (sp.is_none()) return std::nullopt;
        return sp;
    }
    std::vector<std::pair<py::object, py::object>> smap_items(const py::object& smap) const {
        std::vector<std::pair<py::object, py::object>> out;
        for (const py::handle kv : py::reinterpret_borrow<py::iterable>(py::iter(smap.attr("items")()))) {
            const std::vector<py::object> p = objects(kv);
            out.emplace_back(p.at(0), p.at(1));
        }
        return out;
    }
    static bool waveform_eq(const py::object& d, const py::object& w) { return rx::field(d, "waveform").equal(w); }
    py::object spec_for(const py::object& smap, const py::object& specs, const py::object& d) const {
        py::object w = attr_or(d, "waveform", ce_int());
        const int in = PySequence_Contains(smap.ptr(), w.ptr());
        if (in < 0) throw py::error_already_set();
        if (in) return smap[w];
        return g("lframe").attr("twin")(specs[py::int_(0)], w);
    }
    static double t_sym(const py::object& spec) { return rx::as_double(rx::field(spec, "t_sym")); }
    static bool spec_is_l(const py::object& spec) { return rx::field(spec, "waveform").equal(l_int()); }

    // arguments
    static bool given(const py::object& c) { return c && !c.is_none(); }
    static std::vector<py::object> listed(const py::object& c) { return objects(c); }
    py::object ce_hints(const py::object& h) const { return g("_ce_hints")(h); }
    static std::vector<double> floats(const py::object& skip) {
        std::vector<double> out;
        for (const py::handle f : py::reinterpret_borrow<py::iterable>(py::iter(skip))) out.push_back(rx::as_double(f));
        return out;
    }
    std::vector<double> centres(const py::object& skip) const { return floats(g("_centres")(skip)); }
    py::object monotonic() const { return g("time").attr("monotonic")(); }
    double now() const { return rx::as_double(monotonic()); }
    std::optional<double> deadline(const py::object& budget) const {
        if (budget.is_none()) return std::nullopt;
        py::object t = monotonic();
        return rx::as_double(steal(PyNumber_Add(t.ptr(), budget.ptr())));
    }
    static bool reached(const py::object& maxc, std::int64_t taken) {
        if (maxc.is_none()) return false;
        const int r = PyObject_RichCompareBool(py::int_(taken).ptr(), maxc.ptr(), Py_GE);
        if (r < 0) throw py::error_already_set();
        return r != 0;
    }
    bool old_gating() const {
        py::object rules = g("rules");
        return rx::as_bool(rules.attr("old")(rules.attr("GATING")));
    }
    static py::object others_f(const std::vector<py::object>& dets, std::size_t i) {
        py::list o;
        for (const py::object& x : dets)
            if (!x.is(dets[i])) o.append(rx::field(x, "f_hz"));
        return py::tuple(o);
    }
    py::object others_c(const std::vector<py::object>& dets, std::size_t i) const {
        py::list o;
        for (const py::object& x : dets)
            if (!x.is(dets[i])) o.append(g("centre_hz")(x));
        return py::tuple(o);
    }

    // stages
    py::object capture_of(const py::object& prep) const {
        return g("frontend").attr("Capture")(rx::field(prep, "fe"), rx::field(prep, "q"), rx::field(prep, "t0_index"),
                                            rx::field(prep, "fs"));
    }
    std::vector<py::object> acquire_dets(const py::object& cap, const py::object& x, const py::object& live_only,
                                         const py::object& hints) const {
        py::object acq = g("acquire").attr("acquire")(cap, x, "live_only"_a = live_only, "preprocessed"_a = true,
                                                     "hints"_a = hints);
        return objects(acq.attr("detections")(x));
    }
    std::vector<py::object> candidates(const py::object& prep, const py::object& sel, const py::object& live_only,
                                       const py::object& hints) const {
        return objects(g("candidates")(prep, sel, live_only, hints));
    }
    py::object channel_for(const py::object& prep, const py::object& spec, const py::object& d,
                           const std::vector<double>& others) const {
        return g("channel_for")(prep, spec, d, floats_tuple(others));
    }
    static std::optional<py::object> opt(py::object o) {
        if (o.is_none()) return std::nullopt;
        return o;
    }
    std::optional<py::object> verify(const py::object& chan, const py::object& spec, const py::object& d,
                                     const py::object& erase) const {
        return opt(g("track").attr("verify")(chan, cfg::CH_FS, spec, d, "erase_s"_a = erase));
    }
    std::optional<py::object> verify_l(const py::object& chan, const py::object& spec, const py::object& d,
                                       const py::object& erase) const {
        return opt(g("ltrack").attr("verify_l")(chan, cfg::CH_FS, spec, d, "erase_s"_a = erase));
    }
    std::optional<py::object> verify_track(const py::object& chan, const py::object& spec, const py::object& v,
                                           const py::object& erase) const {
        return opt(g("track").attr("verify_track")(chan, spec, v, "fit"_a = false, "erase_s"_a = erase));
    }
    double ce_resid_ratio(const py::object& tr) const { return rx::as_double(g("_ce_resid_ratio")(tr)); }
    py::object lead_in_span(const py::object& prep, const py::object& v, std::optional<double> t_sym) const {
        py::object fn = g("acquire").attr("lead_in_span");
        py::object pfe = rx::field(prep, "fe"), t0 = rx::field(prep, "t0_index"), f = rx::field(v, "f_hz");
        py::object tau = rx::field(rx::field(v, "timing"), "tau0");
        py::object tau_s = steal(PyNumber_TrueDivide(tau.ptr(), py::int_(cfg::CH_FS).ptr()));
        py::object fs = rx::field(prep, "fs");
        if (t_sym) return fn(pfe, t0, f, 0.0, tau_s, fs, "t_sym"_a = *t_sym);
        return fn(pfe, t0, f, 0.0, tau_s, fs);
    }
    std::optional<py::object> gate(const py::object& prep, const py::object& spec, const py::object& d,
                                   const std::vector<double>& others, const py::object& erase) const {
        return opt(g("_gate")(prep, spec, d, floats_tuple(others), erase));
    }
    std::vector<py::object> detect(const py::object& prep, const py::object& sel, const py::object& live_only,
                                   const py::object& maxc, const py::object& budget, const py::object& erase,
                                   const py::object& hints) const {
        return objects(g("detect")(prep, sel, live_only, maxc, budget, "erase_s"_a = erase, "hints"_a = hints));
    }
    std::vector<py::object> detect_joint(const py::object& prep, const py::object& specs,
                                         const fl::DetectArgs<const Py>& a) const {
        return objects(g("_detect_joint")(prep, specs, a.live_only, a.max_candidates, a.budget_s, a.erase_s, a.cands,
                                          a.skip_hz, a.hints));
    }
    std::vector<py::object> receive_slot_joint(const py::object& prep, const py::object& specs,
                                               const fl::SlotArgs<const Py>& a) const {
        return objects(g("_receive_slot_joint")(prep, specs, a.live_only, a.estimator, "erase_s"_a = a.erase_s,
                                                "round_b"_a = a.round_b, "dets"_a = a.dets,
                                                "max_candidates"_a = a.max_candidates, "budget_s"_a = a.budget_s,
                                                "hints"_a = a.hints));
    }
    py::object prepare(const py::object& cap) const { return g("prepare")(cap); }
    std::optional<py::object> receive_pass(const py::object& prep, const py::object& spec, const py::object& d,
                                           const py::object& est, const py::object& others, const py::object& round_b,
                                           const py::object& erase) const {
        return opt(g("receive_pass")(prep, spec, d, "estimator"_a = est, "exclude_hz"_a = others,
                                     "round_b"_a = round_b, "erase_s"_a = erase));
    }
};

fl::DetectArgs<const Py> py_detect_args(const py::object& live_only, const py::object& maxc, const py::object& budget,
                                        const py::object& erase, const py::object& cands, const py::object& skip,
                                        const py::object& hints) {
    fl::DetectArgs<const Py> a;
    a.live_only = live_only;
    a.max_candidates = maxc;
    a.budget_s = budget;
    a.erase_s = erase;
    a.cands = cands;
    a.skip_hz = skip;
    a.hints = hints;
    return a;
}

fl::SlotArgs<const Py> py_slot_args(const py::object& live_only, const py::object& estimator, const py::object& erase,
                                    const py::object& round_b, const py::object& dets, const py::object& maxc,
                                    const py::object& budget, const py::object& hints) {
    fl::SlotArgs<const Py> a;
    a.live_only = live_only;
    a.estimator = estimator;
    a.erase_s = erase;
    a.round_b = round_b;
    a.dets = dets;
    a.max_candidates = maxc;
    a.budget_s = budget;
    a.hints = hints;
    return a;
}

// --- the typed way: the arguments as C++ records ----------------------------------------------

// Raised by a conversion that is to fall through to the Python way.
struct Untyped {};

template <class F>
auto convert(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const py::error_already_set&) {
        throw Untyped{};
    } catch (const std::exception&) {
        throw Untyped{};
    }
}

std::vector<fr::FrameSpec> selection(const py::object& spec) {
    std::vector<fr::FrameSpec> out;
    if (is_seq(spec)) {
        for (const py::handle h : py::reinterpret_borrow<py::iterable>(spec)) out.push_back(rx::spec(h));
    } else {
        out.push_back(rx::spec(spec));
    }
    return out;
}

// A Prepared as the native records hold it: complex64 samples (anything else
// would be cast, where the reference's acquisition reads them as they are).
rc::Prepared typed_prep(const py::object& prep) {
    if (!rx::is_prepared(prep)) throw Untyped{};
    const py::object a = rx::np_module().attr("asarray")(rx::field(prep, "fe"));
    if (py::str(a.attr("dtype")).cast<std::string>() != "complex64") throw Untyped{};
    const py::object keep = rx::field(prep, "keep");
    if (!keep.is_none()) {
        const std::string dt = py::str(rx::np_module().attr("asarray")(keep).attr("dtype")).cast<std::string>();
        if (dt != "float32" && dt != "bool") throw Untyped{};
    }
    return rx::prepared(prep);
}

std::vector<aq::Hint> typed_hints(const py::object& hints) {
    std::vector<aq::Hint> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(hints))) {
        aq::Hint x;
        if (is_seq(h)) {
            const std::vector<py::object> fw = objects(h);
            if (fw.size() != 2) throw Untyped{};
            x.f_hz = rx::as_double(fw[0]);
            x.waveform = rx::as_int(fw[1]);
        } else {
            x.f_hz = rx::as_double(h);
        }
        out.push_back(x);
    }
    return out;
}

// Detections, each its own object (one listed twice is the Python way's:
// `g is not d` tells them apart only there).
std::vector<types::Detection> typed_dets(const py::object& dets) {
    const std::vector<py::object> objs = objects(dets);
    for (std::size_t i = 0; i < objs.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (objs[i].is(objs[j])) throw Untyped{};
    std::vector<types::Detection> out;
    for (const py::object& o : objs) out.push_back(rx::detection(o));
    return out;
}

std::optional<std::int64_t> typed_maxc(const py::object& m) {
    if (m.is_none()) return std::nullopt;
    if (!py::isinstance<py::int_>(m)) throw Untyped{};
    return rx::as_int(m);
}

std::optional<double> typed_budget(const py::object& b) {
    if (b.is_none()) return std::nullopt;
    if (!(py::isinstance<py::int_>(b) || py::isinstance<py::float_>(b))) throw Untyped{};
    return rx::as_double(b);
}

// skip_hz: frequencies (CE) or _centres (joint)
std::vector<double> typed_skip(const py::object& skip, bool joint) {
    std::vector<double> out;
    const py::module_ np = rx::np_module();
    for (const py::handle f : py::reinterpret_borrow<py::iterable>(py::iter(skip))) {
        const bool number = py::isinstance<py::int_>(f) || py::isinstance<py::float_>(f) ||
                            py::isinstance(f, np.attr("floating")) || py::isinstance(f, np.attr("integer"));
        if (number || !joint) {
            out.push_back(rx::as_double(f));
        } else {
            const py::object wf = attr_or(f, "waveform", ce_int());
            out.push_back(rc::centre_hz(rx::as_double(rx::field(f, "f_hz")),
                                        wf.equal(l_int()) ? cfg::WAVEFORM_L : cfg::WAVEFORM_CE));
        }
    }
    return out;
}

py::list dets_py(const std::vector<types::Detection>& v) {
    py::list out;
    for (const auto& d : v) out.append(rx::to_py(d));
    return out;
}

py::list passes_py(const std::vector<types::PassResult>& v) {
    py::list out;
    for (const auto& p : v) out.append(rx::to_py(p));
    return out;
}

rc::DetectOptions detect_options(const py::object& live_only, const py::object& maxc, const py::object& budget,
                                 const py::object& erase, const py::object& cands, const py::object& skip,
                                 const py::object& hints, bool joint) {
    rc::DetectOptions o;
    o.live_only = rx::as_bool(live_only);
    o.max_candidates = typed_maxc(maxc);
    o.budget_s = typed_budget(budget);
    o.erase_s = spans_of(erase);
    if (!cands.is_none()) o.cands = typed_dets(cands);
    o.skip_hz = typed_skip(skip, joint);
    o.hints = typed_hints(hints);
    return o;
}

rc::SlotOptions slot_options(const py::object& live_only, const py::object& estimator, const py::object& erase,
                             const py::object& round_b, const py::object& dets, const py::object& maxc,
                             const py::object& budget, const py::object& hints) {
    rc::SlotOptions o;
    o.live_only = rx::as_bool(live_only);
    if (!py::isinstance<py::str>(estimator)) throw Untyped{};
    o.estimator = rx::as_string(estimator);
    o.erase_s = spans_of(erase);
    o.round_b = rx::as_bool(round_b);
    if (!dets.is_none()) o.dets = typed_dets(dets);
    o.max_candidates = typed_maxc(maxc);
    o.budget_s = typed_budget(budget);
    o.hints = typed_hints(hints);
    return o;
}

// --- the entry points ---------------------------------------------------------------------------

py::object candidates_py(const py::object& prep, const py::object& spec, const py::object& live_only,
                         const py::object& hints, bool typed) {
    if (typed) {
        try {
            const rc::Prepared p = convert([&] { return typed_prep(prep); });
            const std::vector<fr::FrameSpec> sel = convert([&] { return selection(spec); });
            const bool lo = convert([&] { return rx::as_bool(live_only); });
            const std::vector<aq::Hint> h = convert([&] { return typed_hints(hints); });
            std::vector<types::Detection> r;
            {
                py::gil_scoped_release nogil;
                r = rc::candidates(p, sel, lo, h);
            }
            return dets_py(r);
        } catch (const Untyped&) {
        }
    }
    const Py p;
    return list_of(fl::candidates(p, prep, spec, live_only, hints));
}

py::object detect_py(const py::object& prep, const py::object& spec, const py::object& live_only,
                     const py::object& maxc, const py::object& budget, const py::object& erase,
                     const py::object& cands, const py::object& skip, const py::object& hints, bool typed,
                     bool joint_entry) {
    if (typed) {
        try {
            const rc::Prepared p = convert([&] { return typed_prep(prep); });
            const std::vector<fr::FrameSpec> sel = convert([&] { return selection(spec); });
            const bool joint = joint_entry || convert([&] { return rc::is_joint(std::span<const fr::FrameSpec>(sel)); });
            const rc::DetectOptions o =
                convert([&] { return detect_options(live_only, maxc, budget, erase, cands, skip, hints, joint); });
            std::vector<types::Detection> r;
            {
                py::gil_scoped_release nogil;
                r = joint_entry ? rc::detect_joint(p, sel, o) : rc::detect(p, sel, o);
            }
            return dets_py(r);
        } catch (const Untyped&) {
        }
    }
    const Py p;
    const fl::DetectArgs<const Py> a = py_detect_args(live_only, maxc, budget, erase, cands, skip, hints);
    if (joint_entry) return list_of(fl::detect_joint(p, prep, spec, a));
    return list_of(fl::detect(p, prep, spec, a));
}

py::object gate_py(const py::object& prep, const py::object& spec, const py::object& d, const py::object& others,
                   const py::object& erase, bool typed) {
    if (typed) {
        try {
            const rc::Prepared p = convert([&] { return typed_prep(prep); });
            const fr::FrameSpec s = convert([&] { return rx::spec(spec); });
            const types::Detection det = convert([&] { return rx::detection(d); });
            const std::vector<double> oth = convert([&] { return Py::floats(others); });
            const rc::Erase e = convert([&] { return spans_of(erase); });
            std::optional<types::Detection> r;
            {
                py::gil_scoped_release nogil;
                r = rc::gate(p, s, det, oth, e);
            }
            return r ? py::object(rx::to_py(*r)) : py::object(py::none());
        } catch (const Untyped&) {
        }
    }
    const Py p;
    // _gate(prep, spec, d, others, erase_s): `others` as the caller gave them
    // (the flow sorts no further: the reference hands them on as they are)
    const std::vector<double> oth = Py::floats(others);
    const std::optional<py::object> v = fl::gate(p, prep, spec, py::object(d), oth, erase);
    return v ? *v : py::object(py::none());
}

py::object receive_slot_py(const py::object& cap, const py::object& spec, const py::object& live_only,
                           const py::object& estimator, const py::object& erase, const py::object& round_b,
                           const py::object& dets, const py::object& maxc, const py::object& budget,
                           const py::object& hints, bool typed, bool joint_entry) {
    if (typed) {
        try {
            const std::vector<fr::FrameSpec> sel = convert([&] { return selection(spec); });
            const rc::SlotOptions o =
                convert([&] { return slot_options(live_only, estimator, erase, round_b, dets, maxc, budget, hints); });
            std::optional<rc::Prepared> prep;
            std::optional<fe::Capture> raw;
            if (joint_entry || rx::is_prepared(cap))
                prep.emplace(convert([&] { return typed_prep(cap); }));
            else
                raw.emplace(convert([&] { return rx::capture(cap); }));
            std::vector<types::PassResult> r;
            {
                py::gil_scoped_release nogil;
                if (joint_entry)
                    r = rc::receive_slot_joint(*prep, sel, o);
                else
                    r = rc::receive_slot(prep ? rc::CaptureRef(&*prep) : rc::CaptureRef(&*raw), sel, o);
            }
            return passes_py(r);
        } catch (const Untyped&) {
        }
    }
    const Py p;
    const fl::SlotArgs<const Py> a = py_slot_args(live_only, estimator, erase, round_b, dets, maxc, budget, hints);
    if (joint_entry) return list_of(fl::receive_slot_joint(p, cap, spec, a));
    return list_of(fl::receive_slot(p, cap, spec, a));
}


// FormatError as the exception the reference raises (scipy's, numpy's,
// struct's or Python's own)
[[noreturn]] void raise_format_error(const wav::FormatError& e) {
    using K = wav::FormatError::Kind;
    py::object cls;
    if (e.kind == K::struct_error) {
        cls = py::module_::import("struct").attr("error");
    } else {
        PyObject* c = e.kind == K::zero_division   ? PyExc_ZeroDivisionError
                      : e.kind == K::type          ? PyExc_TypeError
                      : e.kind == K::unbound_local ? PyExc_UnboundLocalError
                                                   : PyExc_ValueError;
        cls = py::reinterpret_borrow<py::object>(c);
    }
    PyErr_SetString(cls.ptr(), e.what());
    throw py::error_already_set();
}

// the reference's warnings, in its order: scipy's WavFileWarnings and
// numpy's RuntimeWarnings (a warning filter that raises raises the first,
// as the reference's would)
void warn_wav(const std::vector<wav::Warning>& warnings) {
    for (const wav::Warning& w : warnings) {
        const py::object cat = w.kind == wav::Warning::Kind::runtime
                                   ? py::reinterpret_borrow<py::object>(PyExc_RuntimeWarning)
                                   : py::module_::import("scipy.io.wavfile").attr("WavFileWarning");
        if (PyErr_WarnEx(cat.ptr(), w.text.c_str(), 1) < 0) throw py::error_already_set();
    }
}

// wavio.read_wav(path) in C++, the GIL released while the file is read:
// its refusals and warnings as the reference's (a ReadError is open()'s
// OSError, naming os.fspath(path) as that does)
std::vector<double> read_wav_py(const fspath::FsPath& path) {
    std::vector<wav::Warning> warnings;
    std::vector<double> x;
    try {
        py::gil_scoped_release nogil;
        x = wav::read_wav(path.bytes, &warnings);
    } catch (const wav::ReadError& e) {
        fspath::raise_os_error(e.err, path.name);
    } catch (const wav::FormatError& e) {
        warn_wav(warnings);
        raise_format_error(e);
    }
    warn_wav(warnings);
    return x;
}

py::object capture_py(const std::vector<std::complex<float>>& fev, const py::object& q, const py::object& start_unix) {
    // q=int(q), t0 = 900.0 * int(q) + 1.0, lead = 12.0 or t0 - float(start_unix)
    const py::object qi = steal(PyNumber_Long(q.ptr()));
    const double t0 = 900.0 * rx::as_double(qi) + 1.0;
    const double lead = start_unix.is_none() ? 12.0 : t0 - rx::as_double(start_unix);
    const py::module_ R = py::module_::import("sstvae.qrss.receiver");
    return R.attr("frontend").attr("Capture")("fe"_a = to_numpy(fev), "q"_a = qi,
                                              "t0_index"_a = lead * static_cast<double>(cfg::FE_FS),
                                              "fs"_a = cfg::FE_FS);
}

py::object capture_from_wav_py(const py::object& path, const py::object& q, const py::object& start_unix,
                               bool typed) {
    if (typed) {
        // wavio.read_wav(str(path)): the reference's str() first
        const std::vector<double> x = read_wav_py(fspath::str_path(path));
        std::vector<std::complex<float>> v;
        {
            py::gil_scoped_release nogil;
            v = fe::audio_to_fe(x);
        }
        return capture_py(v, q, start_unix);
    }
    // the reference's own reading and front end, resolved now
    const py::object x = py::module_::import("sstvae.wavio").attr("read_wav")(py::str(path));
    const py::module_ R = py::module_::import("sstvae.qrss.receiver");
    const py::object fev = R.attr("frontend").attr("audio_to_fe")(x);
    const py::object qi = steal(PyNumber_Long(q.ptr()));
    const double t0 = 900.0 * rx::as_double(qi) + 1.0;
    const double lead = start_unix.is_none() ? 12.0 : t0 - rx::as_double(start_unix);
    return R.attr("frontend").attr("Capture")("fe"_a = fev, "q"_a = qi,
                                              "t0_index"_a = lead * static_cast<double>(cfg::FE_FS),
                                              "fs"_a = cfg::FE_FS);
}

void bind(py::module_& qrss) {
    rx::register_translators();
    py::module_ m = qrss.def_submodule("receiver_slot", "native/core/qrss/receiver_slot (receiver.py: the slot)");
    m.attr("GHOST_HZ") = rc::GHOST_HZ;
    m.attr("GHOST_T") = rc::GHOST_T;
    m.attr("GHOST_Z_RATIO") = rc::GHOST_Z_RATIO;
    m.attr("ALIAS_HZ") = rc::ALIAS_HZ;
    m.attr("ALIAS_RATIO") = rc::ALIAS_RATIO;
    m.attr("L_OWN_BELOW_HZ") = rc::L_OWN_BELOW_HZ;
    m.attr("L_OWN_ABOVE_HZ") = rc::L_OWN_ABOVE_HZ;
    m.attr("CE_RESID_MIN") = rc::CE_RESID_MIN;

    m.def("_ghost_of", [](const py::object& f, const py::object& tau0, const py::object& gf, const py::object& gt,
                          const py::object& t_sym) {
        // abs(f - g_f) < GHOST_HZ and abs(tau0 - g_tau0) < GHOST_T * t_sym * CH_FS
        const double a = rx::as_double(f), b = rx::as_double(gf);
        if (!(std::abs(a - b) < rc::GHOST_HZ)) return false;
        return rc::ghost_of(a, rx::as_double(tau0), b, rx::as_double(gt), rx::as_double(t_sym));
    }, py::arg("f_hz"), py::arg("tau0"), py::arg("g_f_hz"), py::arg("g_tau0"), py::arg("t_sym"));
    m.def("_aliases", [](const py::object& dets, const py::object& t_sym) {
        const Py p;
        return list_of(fl::aliases(p, objects(dets), rx::as_double(t_sym)));
    }, py::arg("dets"), py::arg("t_sym"));
    m.def("_ghosts", [](const py::object& dets, const py::object& t_sym) {
        const Py p;
        return list_of(fl::ghosts(p, objects(dets), rx::as_double(t_sym)));
    }, py::arg("dets"), py::arg("t_sym"));
    m.def("_suspect_neighbour", [](const py::object& x, const py::object& out) {
        const Py p;
        return fl::suspect_neighbour(p, x, objects(out));
    }, py::arg("p"), py::arg("out"));
    m.def("_suspect_beside", [](const py::object& x, const py::object& out) {
        const Py p;
        return fl::suspect_beside(p, x, objects(out));
    }, py::arg("p"), py::arg("out"));
    m.def("_a3_lines", [](const py::object& cands) {
        const Py p;
        py::list out;
        for (const fl::A3Line& l : fl::a3_lines(p, objects(cands))) out.append(py::make_tuple(l.line, l.centre, l.z));
        return out;
    }, py::arg("cands"));
    m.def("_owned", [](const py::object& d, const py::object& found, const py::object& a3) {
        // a3 walked (and its entries unpacked) only for a signal d falls
        // inside, once for each, as the reference's comprehension walks it
        const Py p;
        return fl::owned_by(p, d, objects(found), [&a3]() {
            std::vector<fl::A3Line> lines;
            for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(a3))) {
                const std::array<py::object, 3> t = unpack3(h);
                lines.push_back({rx::as_double(t[0]), rx::as_double(t[1]), rx::as_double(t[2])});
            }
            return lines;
        });
    }, py::arg("d"), py::arg("found"), py::arg("a3"));
    m.def("_ce_resid_ratio", [](const py::object& tr) {
        const tk::TrackResult t = rx::track_result(tr);
        double r;
        {
            py::gil_scoped_release nogil;
            r = rc::ce_resid_ratio(t);
        }
        return r;
    }, py::arg("tr"));

    m.def("candidates", &candidates_py, py::arg("prep"), py::arg("spec"), py::arg("live_only"), py::arg("hints"),
          py::arg("typed"));
    m.def("detect", [](const py::object& prep, const py::object& spec, const py::object& live_only,
                       const py::object& maxc, const py::object& budget, const py::object& erase,
                       const py::object& cands, const py::object& skip, const py::object& hints, bool typed) {
        return detect_py(prep, spec, live_only, maxc, budget, erase, cands, skip, hints, typed, false);
    }, py::arg("prep"), py::arg("spec"), py::arg("live_only"), py::arg("max_candidates"), py::arg("budget_s"),
          py::arg("erase_s"), py::arg("cands"), py::arg("skip_hz"), py::arg("hints"), py::arg("typed"));
    m.def("_detect_joint", [](const py::object& prep, const py::object& specs, const py::object& live_only,
                              const py::object& maxc, const py::object& budget, const py::object& erase,
                              const py::object& cands, const py::object& skip, const py::object& hints, bool typed) {
        return detect_py(prep, specs, live_only, maxc, budget, erase, cands, skip, hints, typed, true);
    }, py::arg("prep"), py::arg("specs"), py::arg("live_only"), py::arg("max_candidates"), py::arg("budget_s"),
          py::arg("erase_s"), py::arg("cands"), py::arg("skip_hz"), py::arg("hints"), py::arg("typed"));
    m.def("_gate", &gate_py, py::arg("prep"), py::arg("spec"), py::arg("d"), py::arg("others"), py::arg("erase_s"),
          py::arg("typed"));
    m.def("receive_slot", [](const py::object& cap, const py::object& spec, const py::object& live_only,
                             const py::object& estimator, const py::object& erase, const py::object& round_b,
                             const py::object& dets, const py::object& maxc, const py::object& budget,
                             const py::object& hints, bool typed) {
        return receive_slot_py(cap, spec, live_only, estimator, erase, round_b, dets, maxc, budget, hints, typed,
                               false);
    }, py::arg("cap"), py::arg("spec"), py::arg("live_only"), py::arg("estimator"), py::arg("erase_s"),
          py::arg("round_b"), py::arg("dets"), py::arg("max_candidates"), py::arg("budget_s"), py::arg("hints"),
          py::arg("typed"));
    m.def("_receive_slot_joint", [](const py::object& prep, const py::object& specs, const py::object& live_only,
                                    const py::object& estimator, const py::object& erase, const py::object& round_b,
                                    const py::object& dets, const py::object& maxc, const py::object& budget,
                                    const py::object& hints, bool typed) {
        return receive_slot_py(prep, specs, live_only, estimator, erase, round_b, dets, maxc, budget, hints, typed,
                               true);
    }, py::arg("prep"), py::arg("specs"), py::arg("live_only"), py::arg("estimator"), py::arg("erase_s"),
          py::arg("round_b"), py::arg("dets"), py::arg("max_candidates"), py::arg("budget_s"), py::arg("hints"),
          py::arg("typed"));
    m.def("capture_from_wav", &capture_from_wav_py, py::arg("path"), py::arg("q"), py::arg("start_unix"),
          py::arg("typed"));
    // wav::read_wav itself, float64 (no row: the tests' window on the
    // reader, compared with wavio.read_wav bit for bit)
    m.def("_read_wav", [](const py::object& path) { return to_numpy(read_wav_py(fspath::fs_path(path))); },
          py::arg("path"));
    m.def("receive_wav", [](const py::object& path, const py::object& q, const py::object& start_unix,
                            const py::object& spec, const py::object& estimator) {
        // the reference's order: the file read, then int(q) and
        // float(start_unix), then receive_slot's spec and estimator
        const std::vector<double> x = read_wav_py(fspath::str_path(path));   // capture_from_wav's str(path)
        const std::int64_t qi = rx::as_int(q);
        std::optional<double> su;
        if (!start_unix.is_none()) su = rx::as_double(start_unix);
        const std::vector<fr::FrameSpec> sel = selection(spec);
        const std::string est = rx::as_string(estimator);
        std::vector<types::PassResult> r;
        {
            py::gil_scoped_release nogil;
            const fe::Capture cap = rc::capture_from_audio(x, qi, su);
            rc::SlotOptions opt;
            opt.estimator = est;
            r = rc::receive_slot(rc::CaptureRef(&cap), sel, opt);
        }
        return passes_py(r);
    }, py::arg("path"), py::arg("q"), py::arg("start_unix"), py::arg("spec"), py::arg("estimator"));
}

const Registrar reg("receiver_slot", 86, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
