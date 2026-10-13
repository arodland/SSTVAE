// sstvae_native.qrss.acquire: native/core/qrss/acquire, the arbitration and
// acquire() of sstvae/qrss/acquire.py, under the reference's names and
// signatures (tests/qrss_native/acquire.py assigns them over acquire's
// attributes):
//
//   acquire(cap, spec, *, live_only=False, preprocessed=False, hints=(),
//           _seams=None)                      -> {"__type__": "Acquisition", ...}
//   _acquire_joint(cap, specs, *, live_only, preprocessed, hints, _seams=None)
//   _hint_items(hints, specs)                -> [(f, [waveform, ...])]
//   drop_symbol_rate_aliases(hits, t_sym=T_SYM), drop_band_owned(hits),
//   merge_hits(hits)                          -> the caller's own hit objects
//   live_end_s()
//   _detections(acq, spec), _joint_detections(acq, specs)
//                                             -> [Detection as a dict]
//
// Records come back as rx_types.hpp's "__type__" dicts. `cap` is anything
// with fe, t0_index and fs (frontend.Capture); `spec` a frame or a tuple
// (list) of frames; `specs` the {waveform: frame} dict frame_map makes.
// Hints are read before the stages run (the reference reads them as it
// reaches them), so a malformed one raises the reference's error type
// without the stages before it having run.
//
// `_seams`: the stages a caller has replaced, by the reference's name --
// "blank" and "normalise" (frontend's, together), "carrier_line_stats",
// "carrier_lines", "pilot_lines", "hint_line", "a1_threshold", "_a2_input"
// and "preamble_search" (together, for the one channelisation per line;
// "preamble_search" alone also replaces every other search), "lead_in_span",
// "noise_psd" (the native lead_in_span's), "track_before_detect",
// "shape_frames" (the native track_before_detect's) -- each a callable
// taking the reference's arguments. A missing name is the C++ stage. A seam
// is the reference's world, so its arguments are the reference's objects:
// the caller's cap.fe / t0_index / fs and frames, sstvae.qrss.acquire's
// PREAMBLE_KINDS and L_PRE, and its own CarrierStats and CarrierCandidates
// (built from the C++ values when a C++ stage made them). Whatever a seam
// returned is handed back as itself: the lines, preamble hits (their
// lead_in_s set, as the reference sets it) and A3 hits a Python stage made
// are those objects in the Acquisition.
//
// The GIL is released while the acquisition runs and taken back for each
// seam call.

#include "qrss/rx_types.hpp"

#include <pybind11/eval.h>

#include <algorithm>
#include <bit>
#include <complex>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "qrss/acquire.hpp"
#include "qrss/acquire_stats.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace aq = sstvae::qrss::acquire;
namespace rxa = sstvae::qrss::rxconfig::acquire;
namespace cfg = sstvae::qrss::config;
namespace types = sstvae::qrss::types;
using cf = std::complex<float>;
using cd = std::complex<double>;

// --- arguments --------------------------------------------------------------------------

// The capture's FE as the stages take it: complex64 as it is, anything else
// as complex128 (the a12 binding's rule).
struct FeArg {
    py::array keep;
    std::span<const cf> f32;
    std::span<const cd> f64;
    bool is32 = false;
};

FeArg fe_arg(const py::handle& o) {
    py::array a = rx::np_array(o);
    if (a.ndim() != 1) rx::not_implemented("fe must be 1-D");
    FeArg f;
    if (py::str(a.attr("dtype")).cast<std::string>() == "complex64") {
        auto c = py::array_t<cf, py::array::c_style | py::array::forcecast>::ensure(a);
        if (!c) throw py::error_already_set();
        f.f32 = std::span<const cf>(c.data(), static_cast<std::size_t>(c.size()));
        f.keep = c;
        f.is32 = true;
    } else {
        auto c = py::array_t<cd, py::array::c_style | py::array::forcecast>::ensure(a);
        if (!c) throw py::error_already_set();
        f.f64 = std::span<const cd>(c.data(), static_cast<std::size_t>(c.size()));
        f.keep = c;
    }
    return f;
}

bool is_seq(const py::handle& o) { return py::isinstance<py::tuple>(o) || py::isinstance<py::list>(o); }

// `f, w = hnt` with the interpreter's own unpacking (and its error text).
py::tuple unpack2(const py::handle& h) {
    static py::object* fn = [] {
        py::dict ns;
        ns["__builtins__"] = py::module_::import("builtins");
        py::exec("def _unpack2(h):\n    f, w = h\n    return f, w\n", ns);
        return new py::object(ns["_unpack2"]);
    }();
    return (*fn)(h);
}

// One frame's hints (acquire's single path): a number each; a (f, w) pair
// is kept as a waveform hint, which the core refuses as the reference does.
std::vector<aq::Hint> single_hints(const py::handle& hints) {
    std::vector<aq::Hint> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(hints)) {
        aq::Hint x;
        if (is_seq(h))
            x.waveform = cfg::WAVEFORM_CE;
        else
            x.f_hz = rx::as_double(h);
        out.push_back(x);
    }
    return out;
}

// The waveform-aware path's hints, read as _hint_items reads them.
std::vector<aq::Hint> joint_hints(const py::handle& hints) {
    std::vector<aq::Hint> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(hints)) {
        aq::Hint x;
        if (is_seq(h)) {
            const py::tuple fw = unpack2(h);
            x.f_hz = rx::as_double(fw[0]);
            x.waveform = rx::as_int(fw[1]);
        } else {
            x.f_hz = rx::as_double(h);
        }
        out.push_back(x);
    }
    return out;
}

// A {waveform: frame} dict as a FrameMap, with the caller's frame objects.
struct Frames {
    aq::FrameMap map;
    std::vector<py::object> objs;   // map order
};

Frames frames_of_dict(const py::handle& specs) {
    Frames f;
    for (const auto& [k, v] : py::reinterpret_borrow<py::dict>(specs)) {
        f.map.emplace_back(rx::as_int(k), rx::spec(v));
        f.objs.push_back(py::reinterpret_borrow<py::object>(v));
    }
    return f;
}

// frame_map(spec tuple): ValueError texts the core's.
Frames frames_of_tuple(const py::handle& specs) {
    std::vector<py::object> objs;
    std::vector<rx::frame::FrameSpec> fs;
    for (const py::handle s : py::reinterpret_borrow<py::iterable>(specs)) {
        objs.push_back(py::reinterpret_borrow<py::object>(s));
        fs.push_back(rx::spec(s));
    }
    Frames f;
    for (const auto& [w, i] : aq::frame_map(fs)) {
        f.map.emplace_back(w, fs[i]);
        f.objs.push_back(objs[i]);
    }
    return f;
}

py::tuple tuple_of(const std::vector<py::object>& v) {
    py::tuple t(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) t[i] = v[i];
    return t;
}

std::vector<std::int64_t> keys_of(const py::handle& specs) {
    std::vector<std::int64_t> out;
    for (const py::handle k : py::reinterpret_borrow<py::iterable>(specs)) out.push_back(rx::as_int(k));
    return out;
}

// --- one acquire() call's Python side -----------------------------------------------------

bool same_bits(double a, double b) { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

bool same_line(const types::CarrierCandidate& a, const types::CarrierCandidate& b, bool waveform) {
    return same_bits(a.f_hz, b.f_hz) && same_bits(a.drift_hz_per_min, b.drift_hz_per_min) &&
           same_bits(a.metric, b.metric) && (!waveform || a.waveform == b.waveform);
}

bool same_hit(const aq::PreambleHit& a, const aq::PreambleHit& b) {
    return same_bits(a.f_hz, b.f_hz) && same_bits(a.drift_hz_per_min, b.drift_hz_per_min) &&
           same_bits(a.tau0_s, b.tau0_s) && same_bits(a.phase, b.phase) && same_bits(a.lam, b.lam) &&
           same_bits(a.cell_lam, b.cell_lam) && same_bits(a.p_value, b.p_value) && a.passed == b.passed &&
           a.n_groups == b.n_groups && same_bits(a.tau0_sd_s, b.tau0_sd_s) && a.waveform == b.waveform;
}

struct Ctx {
    // the caller's
    py::dict seams;
    py::object cap_fe = py::none(), t0 = py::none(), fs = py::none();
    std::vector<py::object> spec_objs;      // map order (one frame: that frame)
    py::object shape_specs = py::none();    // tuple(specs.values()) on the joint path
    bool joint = false;
    // the reference's modules, once a seam needs their objects
    py::object A = py::none(), T = py::none();
    // Python objects for the values the stages pass around (GIL only)
    py::object fe = py::none();             // FE as the reference holds it
    py::object cs = py::none();             // the CarrierStats a seam made, or one built for one
    py::object lines = py::none();          // carrier_lines' list, when a seam made it
    py::object tbd = py::none();            // track_before_detect's list, when a seam made it
    std::vector<std::pair<types::CarrierCandidate, py::object>> cands;
    std::vector<std::pair<aq::PreambleHit, py::object>> hits;
    // what the C++ stages made (no GIL): which lines are A1 cells
    std::vector<types::CarrierCandidate> cells;
    std::vector<std::pair<types::CarrierCandidate, bool>> hinted;   // hint_line, refined?

    bool has(const char* name) const { return seams.contains(name); }
    py::object seam(const char* name) const { return seams[name]; }

    py::object acq_module() {
        if (A.is_none()) A = py::module_::import("sstvae.qrss.acquire");
        return A;
    }
    py::object types_module() {
        if (T.is_none()) T = py::module_::import("sstvae.qrss.types");
        return T;
    }

    py::object fe_py(const aq::FeView& v) {
        if (fe.is_none())
            fe = std::visit([](auto s) { return py::object(to_numpy(s)); }, v);
        return fe;
    }

    py::object cs_py(const aq::CarrierStats& c) {
        if (cs.is_none()) {
            py::dict d = rx::to_py(c);
            cs = acq_module().attr("CarrierStats")(py::arg("stat") = d["stat"], py::arg("f_hz") = d["f_hz"],
                                                   py::arg("drifts") = d["drifts"], py::arg("M") = d["M"]);
        }
        return cs;
    }

    // f_hz as the reference would hold it: an np.float64 for an A1 cell
    // (and a refined hint line), else a float.
    py::object f_obj(const types::CarrierCandidate& c) const {
        bool np64 = false;
        for (const auto& x : cells)
            if (same_line(x, c, false)) np64 = true;
        for (const auto& [x, refined] : hinted)
            if (same_line(x, c, false)) np64 = refined;
        if (np64) return rx::np_module().attr("float64")(c.f_hz);
        return py::float_(c.f_hz);
    }

    const py::object* find_cand(const types::CarrierCandidate& c, bool waveform) const {
        for (const auto& [x, o] : cands)
            if (same_line(x, c, waveform)) return &o;
        return nullptr;
    }

    // The reference's CarrierCandidate for a line a seam is handed: the
    // object a seam made it as, else one built (and kept, so the same line
    // is the same object).
    py::object cand_py(const types::CarrierCandidate& c) {
        if (const py::object* o = find_cand(c, true)) return *o;
        py::object w = c.waveform ? py::object(py::int_(*c.waveform)) : py::object(py::none());
        py::object o = types_module().attr("CarrierCandidate")(f_obj(c), c.drift_hz_per_min, c.metric,
                                                               py::arg("waveform") = w);
        cands.emplace_back(c, o);
        return o;
    }

    py::list lines_py(std::span<const types::CarrierCandidate> ls) {
        py::list out;
        for (const auto& c : ls) out.append(cand_py(c));
        return out;
    }

    types::CarrierCandidate keep_cand(const py::handle& o) {
        types::CarrierCandidate c = rx::carrier_candidate(o);
        cands.emplace_back(c, py::reinterpret_borrow<py::object>(o));
        return c;
    }

    aq::PreambleHit keep_hit(const py::handle& o) {
        aq::PreambleHit h = rx::preamble_hit(o);
        hits.emplace_back(h, py::reinterpret_borrow<py::object>(o));
        return h;
    }

    py::object kind_py(const aq::PreambleKind* k) {
        if (k->waveform == cfg::WAVEFORM_L) return acq_module().attr("L_PRE");
        return acq_module().attr("PREAMBLE_KINDS")[py::int_(k->waveform)];
    }

    py::object spec_py(const rx::frame::FrameSpec& s) const {
        if (spec_objs.size() == 1) return spec_objs.front();
        for (const py::object& o : spec_objs)
            if (rx::as_int(o.attr("waveform")) == static_cast<std::int64_t>(s.waveform())) return o;
        return py::cast(s);
    }
};

// The seams acquire() runs with: the caller's Python stages, and the C++
// ones wrapped where the result needs to know which stage made a line.
aq::AcquireSeams make_seams(Ctx& ctx) {
    aq::AcquireSeams s;
    if (ctx.has("blank") || ctx.has("normalise")) {
        if (!(ctx.has("blank") && ctx.has("normalise")))
            throw py::value_error("_seams: 'blank' and 'normalise' go together");
        s.prepare = [&ctx](const aq::FeView&, double) {
            py::gil_scoped_acquire gil;
            py::tuple b = ctx.seam("blank")(ctx.cap_fe, py::arg("fs") = ctx.fs);
            py::tuple n = ctx.seam("normalise")(b[0], b[1], py::arg("fs") = ctx.fs);
            ctx.fe = n[0];
            return rx::vec<cf>(n[0]);
        };
    }
    if (ctx.has("carrier_line_stats")) {
        s.carrier_line_stats = [&ctx](const aq::FeView& fe, double, double) {
            py::gil_scoped_acquire gil;
            ctx.cs = ctx.seam("carrier_line_stats")(ctx.fe_py(fe), ctx.t0, ctx.fs);
            return rx::carrier_stats(ctx.cs);
        };
    }
    if (ctx.has("carrier_lines")) {
        s.carrier_lines = [&ctx](const aq::FeView& fe, double, double, const aq::CarrierStats& cs) {
            py::gil_scoped_acquire gil;
            py::object ls = ctx.seam("carrier_lines")(ctx.fe_py(fe), ctx.t0, ctx.fs, py::arg("stats") = ctx.cs_py(cs));
            ctx.lines = ls;
            std::vector<types::CarrierCandidate> out;
            for (const py::handle c : py::reinterpret_borrow<py::iterable>(ls)) out.push_back(ctx.keep_cand(c));
            return out;
        };
    } else {
        s.carrier_lines = [&ctx](const aq::FeView&, double, double fs, const aq::CarrierStats& cs) {
            std::vector<types::CarrierCandidate> out = aq::carrier_lines(cs, fs);
            ctx.cells.insert(ctx.cells.end(), out.begin(), out.end());
            return out;
        };
    }
    if (ctx.has("pilot_lines")) {
        s.pilot_lines = [&ctx](const aq::CarrierStats& cs, std::span<const types::CarrierCandidate> ls) {
            py::gil_scoped_acquire gil;
            py::object lines = ctx.lines.is_none() ? py::object(ctx.lines_py(ls)) : ctx.lines;
            py::object ps = ctx.seam("pilot_lines")(ctx.cs_py(cs), lines);
            std::vector<types::CarrierCandidate> out;
            for (const py::handle c : py::reinterpret_borrow<py::iterable>(ps)) out.push_back(ctx.keep_cand(c));
            return out;
        };
    } else {
        s.pilot_lines = [&ctx](const aq::CarrierStats& cs, std::span<const types::CarrierCandidate> ls) {
            std::vector<types::CarrierCandidate> out = aq::pilot_lines(cs, ls);
            ctx.cells.insert(ctx.cells.end(), out.begin(), out.end());
            return out;
        };
    }
    if (ctx.has("hint_line")) {
        s.hint_line = [&ctx](const aq::CarrierStats& cs, double f, double) {
            py::gil_scoped_acquire gil;
            return ctx.keep_cand(ctx.seam("hint_line")(ctx.cs_py(cs), py::float_(f), ctx.fs));
        };
    } else {
        s.hint_line = [&ctx](const aq::CarrierStats& cs, double f, double fs) {
            bool refined = false;
            const types::CarrierCandidate c = aq::hint_line(cs, f, fs, rxa::A1_PFA, rxa::HINT_SPAN_HZ, &refined);
            ctx.hinted.emplace_back(c, refined);
            return c;
        };
    }
    if (ctx.has("a1_threshold")) {
        s.a1_threshold = [&ctx](std::int64_t M, double, double) {
            py::gil_scoped_acquire gil;
            py::object m = ctx.cs.is_none() ? py::object(py::int_(M)) : py::object(ctx.cs.attr("M"));
            return rx::as_double(ctx.seam("a1_threshold")(m, ctx.acq_module().attr("A1_PFA"), ctx.fs));
        };
    }
    if (ctx.has("_a2_input") || ctx.has("preamble_search")) {
        if (!(ctx.has("_a2_input") && ctx.has("preamble_search")))
            throw py::value_error("_seams: '_a2_input' and 'preamble_search' go together");
        s.line_search = [&ctx](const aq::FeView& fe, double, const types::CarrierCandidate& c, double,
                               std::span<const aq::PreambleKind* const> kinds) {
            py::gil_scoped_acquire gil;
            py::object fe_o = ctx.fe_py(fe);
            py::object line = ctx.cand_py(c);
            py::object inp = ctx.seam("_a2_input")(fe_o, ctx.t0, line.attr("f_hz"), line.attr("drift_hz_per_min"), ctx.fs);
            std::vector<aq::PreambleHit> out;
            for (const aq::PreambleKind* k : kinds)
                out.push_back(ctx.keep_hit(ctx.seam("preamble_search")(fe_o, ctx.t0, line, ctx.fs,
                                                                      py::arg("kind") = ctx.kind_py(k),
                                                                      py::arg("inp") = inp)));
            return out;
        };
        s.search = [&ctx](const aq::FeView& fe, double, const types::CarrierCandidate& c, double,
                          const aq::PreambleKind* kind) {
            py::gil_scoped_acquire gil;
            py::object fn = ctx.seam("preamble_search");
            py::object r = kind ? fn(ctx.fe_py(fe), ctx.t0, ctx.cand_py(c), ctx.fs, py::arg("kind") = ctx.kind_py(kind))
                                : fn(ctx.fe_py(fe), ctx.t0, ctx.cand_py(c), ctx.fs);
            return ctx.keep_hit(r);
        };
    }
    if (ctx.has("lead_in_span")) {
        s.lead_in_span = [&ctx](const aq::FeView& fe, double, double f, double drift, double tau0, double,
                                std::optional<double> t_sym) {
            py::gil_scoped_acquire gil;
            py::object fn = ctx.seam("lead_in_span");
            py::object r = t_sym ? fn(ctx.fe_py(fe), ctx.t0, f, drift, tau0, ctx.fs, py::arg("t_sym") = *t_sym)
                                 : fn(ctx.fe_py(fe), ctx.t0, f, drift, tau0, ctx.fs);
            return rx::as_double(r);
        };
    }
    if (ctx.has("noise_psd")) {
        s.noise_psd = [&ctx](std::span<const cf> ch, std::int64_t fs_ch, std::int64_t fs_in, double audio_hz) {
            py::gil_scoped_acquire gil;
            py::object r = ctx.seam("noise_psd")(to_numpy(ch), fs_ch, py::arg("fs_in") = fs_in,
                                                 py::arg("audio_hz") = audio_hz);
            return rx::vec<float>(r);
        };
    }
    if (ctx.has("track_before_detect")) {
        s.track_before_detect = [&ctx](const aq::FeView& fe, double, const rx::frame::FrameSpec& spec,
                                       std::span<const types::CarrierCandidate> ls, double,
                                       const std::vector<rx::frame::FrameSpec>* shape) {
            py::gil_scoped_acquire gil;
            py::object fn = ctx.seam("track_before_detect");
            py::object lines = ctx.lines.is_none() ? py::object(ctx.lines_py(ls)) : ctx.lines;
            py::object r = shape ? fn(ctx.fe_py(fe), ctx.t0, ctx.spec_py(spec), lines, ctx.fs,
                                      py::arg("shape_specs") = ctx.shape_specs)
                                 : fn(ctx.fe_py(fe), ctx.t0, ctx.spec_py(spec), lines, ctx.fs);
            py::list objs(r);
            ctx.tbd = objs;
            return rx::tbd_hits(objs);
        };
    }
    if (ctx.has("shape_frames")) {
        s.shape_frames = [&ctx](std::span<const double> t, std::span<const rx::frame::FrameSpec>) {
            py::gil_scoped_acquire gil;
            return rx::vec_bool(ctx.seam("shape_frames")(to_numpy(t), ctx.shape_specs));
        };
    }
    return s;
}

// --- the result ---------------------------------------------------------------------------

// A line in the Acquisition: the object a seam made it as, else a record
// (an A1 cell's f_hz an np.float64, a hint's as hint_line made it).
py::object line_out(const Ctx& ctx, const types::CarrierCandidate& c) {
    if (const py::object* o = ctx.find_cand(c, true)) return *o;
    py::dict d = rx::to_py(c);
    d["f_hz"] = ctx.f_obj(c);
    return d;
}

// A hint A2 did not pass: as hint_line made it, with the waveform the
// joint path gives it (dataclasses.replace).
py::object hinted_out(const Ctx& ctx, const types::CarrierCandidate& c) {
    if (const py::object* o = ctx.find_cand(c, true)) return *o;
    if (const py::object* o = ctx.find_cand(c, false)) {
        py::object w = c.waveform ? py::object(py::int_(*c.waveform)) : py::object(py::none());
        return py::module_::import("dataclasses").attr("replace")(*o, py::arg("waveform") = w);
    }
    return line_out(ctx, c);
}

py::dict result(Ctx& ctx, const aq::Acquisition& a) {
    py::dict d = rx::typed("Acquisition");
    py::list lines, pre, tbd, hinted, pilots;
    for (const auto& c : a.lines) lines.append(line_out(ctx, c));
    std::vector<bool> used(ctx.hits.size(), false);
    for (const auto& h : a.preamble) {
        py::object o;
        for (std::size_t i = 0; i < ctx.hits.size(); ++i)
            if (!used[i] && same_hit(ctx.hits[i].first, h)) {
                used[i] = true;
                o = ctx.hits[i].second;
                o.attr("lead_in_s") = h.lead_in_s;   // the reference sets it on its hit
                break;
            }
        pre.append(o ? o : py::object(rx::to_py(h)));
    }
    if (!ctx.tbd.is_none())
        tbd = py::list(ctx.tbd);
    else
        for (const auto& h : a.tbd) tbd.append(rx::to_py(h));
    for (const auto& c : a.hinted) hinted.append(hinted_out(ctx, c));
    for (const auto& c : a.pilots) pilots.append(line_out(ctx, c));
    d["lines"] = lines;
    d["preamble"] = pre;
    d["tbd"] = tbd;
    d["hinted"] = hinted;
    d["pilots"] = pilots;
    return d;
}

aq::AcquireOptions options(bool live_only, bool preprocessed, std::vector<aq::Hint> hints) {
    aq::AcquireOptions o;
    o.live_only = live_only;
    o.preprocessed = preprocessed;
    o.hints = std::move(hints);
    return o;
}

Ctx context(const py::object& cap, const py::object& seams) {
    Ctx ctx;
    if (!seams.is_none()) ctx.seams = py::dict(seams);
    ctx.cap_fe = cap.attr("fe");
    ctx.t0 = cap.attr("t0_index");
    ctx.fs = cap.attr("fs");
    return ctx;
}

// Runs the C++ acquisition on the capture, the GIL released.
template <typename Run>
aq::Acquisition run(Ctx& ctx, bool preprocessed, Run&& go) {
    const FeArg fe = fe_arg(ctx.cap_fe);
    if (preprocessed) ctx.fe = ctx.cap_fe;
    const double t0 = rx::as_double(ctx.t0), fs = rx::as_double(ctx.fs);
    const aq::AcquireSeams seams = make_seams(ctx);
    py::gil_scoped_release nogil;
    return fe.is32 ? go(fe.f32, t0, fs, seams) : go(fe.f64, t0, fs, seams);
}

py::dict acquire(const py::object& cap, const py::object& spec, bool live_only, bool preprocessed,
                 const py::object& hints, const py::object& seams) {
    Ctx ctx = context(cap, seams);
    aq::Acquisition a;
    if (is_seq(spec)) {
        const Frames f = frames_of_tuple(spec);
        ctx.joint = true;
        ctx.spec_objs = f.objs;
        ctx.shape_specs = tuple_of(f.objs);
        const aq::AcquireOptions opt = options(live_only, preprocessed, joint_hints(hints));
        a = run(ctx, preprocessed, [&](auto fe, double t0, double fs, const aq::AcquireSeams& s) {
            return aq::acquire_joint(fe, t0, fs, f.map, opt, s);
        });
    } else {
        const rx::frame::FrameSpec sp = rx::spec(spec);
        ctx.spec_objs = {spec};
        const aq::AcquireOptions opt = options(live_only, preprocessed, single_hints(hints));
        a = run(ctx, preprocessed, [&](auto fe, double t0, double fs, const aq::AcquireSeams& s) {
            return aq::acquire(fe, t0, fs, sp, opt, s);
        });
    }
    return result(ctx, a);
}

py::dict acquire_joint(const py::object& cap, const py::object& specs, bool live_only, bool preprocessed,
                       const py::object& hints, const py::object& seams) {
    Ctx ctx = context(cap, seams);
    const Frames f = frames_of_dict(specs);
    ctx.joint = true;
    ctx.spec_objs = f.objs;
    ctx.shape_specs = tuple_of(f.objs);
    const aq::AcquireOptions opt = options(live_only, preprocessed, joint_hints(hints));
    const aq::Acquisition a = run(ctx, preprocessed, [&](auto fe, double t0, double fs, const aq::AcquireSeams& s) {
        return aq::acquire_joint(fe, t0, fs, f.map, opt, s);
    });
    return result(ctx, a);
}

// --- arbitration over the caller's hits -------------------------------------------------------

template <typename Idx>
py::list kept(const py::object& hits, Idx&& idx) {
    const py::list objs(hits);
    std::vector<aq::PreambleHit> hv;
    hv.reserve(objs.size());
    for (const py::handle h : objs) hv.push_back(rx::preamble_hit(h));
    py::list out;
    for (const std::size_t i : idx(std::span<const aq::PreambleHit>(hv))) out.append(objs[i]);
    return out;
}

// --- detections -----------------------------------------------------------------------------

// The fields detections() reads (lines and pilots are not).
aq::Acquisition detection_fields(const py::handle& o) {
    aq::Acquisition a;
    a.preamble = rx::preamble_hits(rx::field(o, "preamble"));
    a.tbd = rx::tbd_hits(rx::field(o, "tbd"));
    a.hinted = rx::carrier_candidates(rx::field(o, "hinted"));
    return a;
}

py::list detections_py(const std::vector<types::Detection>& v) {
    py::list out;
    for (const auto& d : v) out.append(rx::to_py(d));
    return out;
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("acquire", "acquire.py's arbitration and acquire() (native/core/qrss/acquire)");

    m.def("acquire", &acquire, py::arg("cap"), py::arg("spec"), py::kw_only(), py::arg("live_only") = false,
          py::arg("preprocessed") = false, py::arg("hints") = py::tuple(), py::arg("_seams") = py::none());
    m.def("_acquire_joint", &acquire_joint, py::arg("cap"), py::arg("specs"), py::kw_only(), py::arg("live_only"),
          py::arg("preprocessed"), py::arg("hints"), py::arg("_seams") = py::none());
    m.def("_hint_items", [](const py::object& hints, const py::object& specs) {
        const std::vector<aq::Hint> hv = joint_hints(hints);
        const std::vector<std::int64_t> ws = keys_of(specs);
        py::list out;
        for (const auto& [f, w] : aq::hint_items(hv, ws)) {
            py::list wl;
            for (const std::int64_t x : w) wl.append(x);
            out.append(py::make_tuple(f, wl));
        }
        return out;
    }, py::arg("hints"), py::arg("specs"));

    m.def("drop_symbol_rate_aliases", [](const py::object& hits, double t_sym) {
        return kept(hits, [t_sym](std::span<const aq::PreambleHit> h) { return aq::drop_symbol_rate_aliases_idx(h, t_sym); });
    }, py::arg("hits"), py::arg("t_sym") = cfg::T_SYM);
    m.def("drop_band_owned", [](const py::object& hits) {
        return kept(hits, [](std::span<const aq::PreambleHit> h) { return aq::drop_band_owned_idx(h); });
    }, py::arg("hits"));
    m.def("merge_hits", [](const py::object& hits) {
        return kept(hits, [](std::span<const aq::PreambleHit> h) { return aq::merge_hits_idx(h); });
    }, py::arg("hits"));
    m.def("live_end_s", &aq::live_end_s);

    m.def("_detections", [](const py::object& acq, const py::object& spec) {
        const aq::Acquisition a = detection_fields(acq);
        if (is_seq(spec)) return detections_py(aq::joint_detections(a, frames_of_tuple(spec).map));
        return detections_py(a.detections(rx::spec(spec)));
    }, py::arg("self"), py::arg("spec"));
    m.def("_joint_detections", [](const py::object& acq, const py::object& specs) {
        return detections_py(aq::joint_detections(detection_fields(acq), frames_of_dict(specs).map));
    }, py::arg("self"), py::arg("specs"));
}

const Registrar reg("acquire", 80, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
