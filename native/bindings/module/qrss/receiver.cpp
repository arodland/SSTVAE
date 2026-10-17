// sstvae_native.qrss.receiver: bindings of native/core/qrss/receiver (the
// per-pass half of sstvae/qrss/receiver.py). Names, arguments and defaults
// follow the reference's, so tests/qrss_native/receiver.py can assign them
// over it:
//
//   CONF_BINS, CONF_FLOOR_DB, L_EXCLUDE_OFFSET_HZ, MARGIN_S
//   prepare(cap)                          cap itself for a Prepared, else a
//                                         "Prepared" dict
//   frames(spec), is_joint(spec), _single(spec)   the caller's frame objects
//   centre_hz(obj), _centres(skip), _ce_hints(hints)
//   channel_for(prep, spec, det, exclude_hz=(), f_mix_hz=None)
//       -> "ChanCapture" dict (at the CH rate holding prep.fe itself)
//   _psi_frame(tr) -> float32[n_pos]
//   _cw_result(tr, spec, hdr) -> ("CwIdResult" dict or None,
//                                 (uint8 keying units, ook) or None)
//   receive_pass(cap, spec, det, prior=None, estimator="joint", *,
//                exclude_hz=(), f_mix_hz=None, round_b=True,
//                return_tracks=False, erase_s=(), _seams=None)
//       -> "PassResult" dict, or None (gate V refused); with return_tracks
//          (pass, [track dicts]): each track's "spec" is the frame object it
//          ran as (the caller's, or a native frame.plain of it) and its
//          "chan" the one channel object (a seam's, else one dict)
//   decode_header(llr, diag, spec) -> (hdr, llr, spec): the caller's llr and
//          spec objects, or np.asarray(diag["llr_block"], llr.dtype) and a
//          native frame.plain(spec) after a version 1 block
//   _decode_header_l(llr, diag, spec) -> "HeaderFields" dict or None
//   detection_from_pass(p, lead_in_s=0.0) -> "Detection" dict holding p's
//          own f_hz, timing and report.z_ref objects
//   capture_from_pass(p) -> "Prepared" dict of p's own objects
//   mean_w_db(p), confidence_db(p, spec=None, elapsed_s=None, n=CONF_BINS)
//
// `_seams`: what receive_pass is to call in Python, by the reference's name
// -- "prepare", "channel_for", "verify" (track's), "verify_l" (ltrack's),
// "noise_psd" (frontend's) -- each called with the reference's arguments.
// A missing name is the C++. A seam is the reference's world: it is handed
// the caller's cap, spec, det, exclude_hz, f_mix_hz and erase_s objects,
// and the Prepared and ChanCapture as a seam made them (else as "Prepared"
// / "ChanCapture" dicts, which the row turns into the reference's). A
// channel a seam made keeps the noise table the C++ computed for it in its
// own `_psd`, as the reference's object would.
//
// The GIL is released while the pass is received and taken back for each
// seam call.

#include "qrss/rx_types.hpp"

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/frontend.hpp"
#include "qrss/receiver.hpp"
#include "qrss/trackbase.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace rc = sstvae::qrss::receiver;
namespace fr = sstvae::qrss::frame;
namespace tk = sstvae::qrss::track;
namespace fe = sstvae::qrss::frontend;
namespace cfg = sstvae::qrss::config;
namespace types = sstvae::qrss::types;
using cf = std::complex<float>;
using sstvae::qrss::Rational;

[[noreturn]] void raise(PyObject* type, const std::string& what) {
    PyErr_SetString(type, what.c_str());
    throw py::error_already_set();
}

// getattr(o, name, default) (a dict's key for the C++'s own records)
py::object attr_or(const py::handle& o, const char* name, py::object dflt) {
    if (py::isinstance<py::dict>(o)) {
        py::dict d = py::reinterpret_borrow<py::dict>(o);
        return d.contains(name) ? py::object(d[name]) : dflt;
    }
    return py::getattr(o, name, dflt);
}

bool is_seq(const py::handle& o) { return py::isinstance<py::tuple>(o) || py::isinstance<py::list>(o); }

// WAVEFORM_NAMES.get(wf, wf), as an f-string prints it
std::string waveform_name(const py::handle& wf) {
    if (wf.equal(py::int_(cfg::WAVEFORM_CE))) return "CE";
    if (wf.equal(py::int_(cfg::WAVEFORM_L))) return "L";
    return py::str(wf).cast<std::string>();
}

// erase_s as the reference unpacks it: `for a, b in spans`, float(a), float(b)
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

std::vector<double> doubles_of(const py::object& o) {
    std::vector<double> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(o))) out.push_back(rx::as_double(h));
    return out;
}

// Fraction(x), as the reference converts f_mix_hz
Rational fraction_of(const py::object& o) { return rx::rational(rx::fraction_class()(o)); }

void set_psd(const py::object& chan, const tk::NoisePsd& t) {
    py::object v = py::make_tuple(to_numpy(t.t_s), to_numpy(t.psd));
    if (py::isinstance<py::dict>(chan))
        py::reinterpret_borrow<py::dict>(chan)["_psd"] = v;
    else
        py::setattr(chan, "_psd", v);
}

// --- frames --------------------------------------------------------------------------------

// frames(spec): the caller's frame objects, CE's first.
py::tuple frames_py(const py::object& spec) {
    if (!is_seq(spec)) return py::make_tuple(spec);
    std::vector<py::object> objs;
    std::vector<fr::FrameSpec> specs;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(spec)) {
        objs.push_back(py::reinterpret_borrow<py::object>(h));
        specs.push_back(rx::spec(h));
    }
    const std::vector<fr::FrameSpec> order = rc::frames(specs);   // frame_map's refusals
    py::tuple out(order.size());
    for (std::size_t i = 0; i < order.size(); ++i)
        for (std::size_t j = 0; j < specs.size(); ++j)
            if (specs[j].waveform() == order[i].waveform()) out[i] = objs[j];
    return out;
}

double centre_of(const py::handle& obj) {
    const py::object wf = attr_or(obj, "waveform", py::int_(cfg::WAVEFORM_CE));
    const double f = rx::as_double(rx::field(obj, "f_hz"));
    return rc::centre_hz(f, wf.equal(py::int_(cfg::WAVEFORM_L)) ? cfg::WAVEFORM_L : cfg::WAVEFORM_CE);
}

// --- the detection's channel ---------------------------------------------------------------

// What channel_for reads of a detection: f_hz, path, timing.
types::Detection det_for_channel(const py::handle& o) {
    types::Detection d;
    d.f_hz = rx::as_double(rx::field(o, "f_hz"));
    d.path = rx::freq_path(rx::field(o, "path"));
    d.timing = rx::opt_timing(rx::field(o, "timing"));
    return d;
}

py::object channel_for_py(const py::object& prep, const py::object& spec, const py::object& det,
                          const py::object& exclude_hz, const py::object& f_mix_hz) {
    const std::int64_t fs = rx::as_int(rx::field(prep, "fs"));
    if (fs == cfg::CH_FS) {
        // the capture as it is, mixed at f_mix_hz
        const py::object fm = f_mix_hz.is_none()
                                  ? rx::to_py(fe::quantise_mix(fe::fe_hz(rx::as_double(rx::field(det, "f_hz")))))
                                  : rx::fraction_class()(f_mix_hz);
        py::dict d = rx::typed("ChanCapture");
        d["ch"] = rx::np_module().attr("asarray")(rx::field(prep, "fe"));
        d["fs"] = cfg::CH_FS;
        d["t0_index"] = rx::field(prep, "t0_index");
        d["f_mix"] = fm;
        d["keep"] = py::none();
        d["exclude_hz"] = py::tuple(exclude_hz);
        d["_psd"] = py::none();
        return d;
    }
    const rc::Prepared p = rx::prepared(prep);
    const fr::FrameSpec s = rx::spec(spec);
    const types::Detection d = det_for_channel(det);
    const std::vector<double> ex = doubles_of(exclude_hz);
    std::optional<tk::ChanCapture> c;
    {
        py::gil_scoped_release nogil;
        c.emplace(rc::channel_for(p, s, d, ex));
    }
    py::dict out = rx::to_py(*c);
    out["exclude_hz"] = py::tuple(exclude_hz);   // make_chan's tuple(exclude_hz): the caller's values
    return out;
}

// --- one pass ------------------------------------------------------------------------------

// One receive_pass call's Python side.
struct Ctx {
    py::dict seams;
    py::object cap, spec, det, exclude_hz, f_mix_hz, erase_s;
    py::object prep = py::none();   // the Prepared as a seam made it, or cap when it is one
    py::object chan = py::none();   // the ChanCapture a seam made, or the dict handed to one
    bool chan_from_seam = false;

    bool has(const char* name) const { return seams.contains(name); }
    py::object seam(const char* name) const { return seams[name]; }

    py::object chan_py(const tk::ChanCapture& c) {
        if (chan.is_none()) chan = rx::to_py(c);
        return chan;
    }
};

rc::ReceiverSeams make_seams(Ctx& ctx) {
    rc::ReceiverSeams s;
    if (ctx.has("prepare")) {
        s.prepare = [&ctx](rc::CaptureRef) {
            py::gil_scoped_acquire gil;
            py::object p = ctx.seam("prepare")(ctx.cap);
            ctx.prep = p;
            return std::make_shared<const rc::Prepared>(rx::prepared(p));
        };
    }
    if (ctx.has("channel_for")) {
        s.channel_for = [&ctx](const rc::Prepared& prep, const fr::FrameSpec&, const types::Detection&,
                               std::span<const double>, const std::optional<Rational>&) {
            py::gil_scoped_acquire gil;
            py::object p = ctx.prep.is_none() ? py::object(rx::to_py(prep)) : ctx.prep;
            py::object c = ctx.seam("channel_for")(p, ctx.spec, ctx.det, ctx.exclude_hz, ctx.f_mix_hz);
            ctx.chan = c;
            ctx.chan_from_seam = true;
            tk::ChanCapture out = rx::chan_capture(c);
            out.ch = rx::capture_fe(rx::field(c, "ch"), out.fs, "a channel");
            return std::make_shared<const tk::ChanCapture>(std::move(out));
        };
    }
    for (const char* name : {"verify", "verify_l"}) {
        if (!ctx.has(name)) continue;
        auto fn = [&ctx, name](std::shared_ptr<const tk::ChanCapture> chan, const fr::FrameSpec&, const types::Detection&,
                               std::span<const std::pair<double, double>>) -> std::optional<types::Detection> {
            py::gil_scoped_acquire gil;
            py::object r = ctx.seam(name)(ctx.chan_py(*chan), cfg::CH_FS, ctx.spec, ctx.det,
                                          py::arg("erase_s") = ctx.erase_s);
            if (r.is_none()) return std::nullopt;
            return rx::detection(r);
        };
        if (std::string(name) == "verify")
            s.verify = fn;
        else
            s.verify_l = fn;
    }
    if (ctx.has("noise_psd")) {
        s.noise_psd = [&ctx](std::span<const cf> ch, std::int64_t fs, std::span<const double> ex, double win_s,
                             std::optional<double> audio_hz) {
            py::gil_scoped_acquire gil;
            py::object ch_o = ctx.chan.is_none() ? py::object(to_numpy(ch)) : rx::field(ctx.chan, "ch");
            py::list exl;
            for (const double x : ex) exl.append(x);
            py::object r = ctx.seam("noise_psd")(ch_o, fs, py::arg("exclude_hz") = exl, py::arg("win_s") = win_s,
                                                 py::arg("audio_hz") = audio_hz ? py::object(py::float_(*audio_hz))
                                                                                : py::object(py::none()));
            return rx::vec<float>(r);
        };
    }
    return s;
}

py::object receive_pass_py(const py::object& cap, const py::object& spec, const py::object& det,
                           const py::object& prior, const py::object& estimator, const py::object& exclude_hz,
                           const py::object& f_mix_hz, const py::object& round_b, const py::object& return_tracks,
                           const py::object& erase_s, const py::object& seams) {
    // the waveform check first, before cap is touched
    const py::object wf = attr_or(det, "waveform", py::int_(cfg::WAVEFORM_CE));
    if (wf.not_equal(rx::field(spec, "waveform")))
        throw std::invalid_argument(rc::waveform_refusal(waveform_name(wf), rx::spec(spec)));

    Ctx ctx;
    if (!seams.is_none()) ctx.seams = py::dict(seams);
    ctx.cap = cap;
    ctx.spec = spec;
    ctx.det = det;
    ctx.exclude_hz = exclude_hz;
    ctx.f_mix_hz = f_mix_hz;
    ctx.erase_s = erase_s;

    const fr::FrameSpec s = rx::spec(spec);
    const types::Detection d = rx::detection(det);
    rc::PassArgs a;
    if (!prior.is_none()) a.prior = rx::em_prior(prior);
    a.estimator = rx::as_string(estimator);
    a.exclude_hz = doubles_of(exclude_hz);
    if (!f_mix_hz.is_none()) a.f_mix_hz = fraction_of(f_mix_hz);
    a.round_b = rx::as_bool(round_b);
    a.return_tracks = rx::as_bool(return_tracks);
    a.erase_s = spans_of(erase_s);

    // the capture, read only when the C++ prepares it
    std::optional<rc::Prepared> prep;
    std::optional<fe::Capture> raw;
    rc::CaptureRef ref = static_cast<const rc::Prepared*>(nullptr);
    if (rx::is_prepared(cap)) {
        ctx.prep = cap;
        if (!ctx.has("prepare")) {
            prep.emplace(rx::prepared(cap));
            ref = &*prep;
        }
    } else if (!ctx.has("prepare")) {
        raw.emplace(rx::capture(cap));
        ref = &*raw;
    }

    const rc::ReceiverSeams sm = make_seams(ctx);
    rc::PassOutcome out;
    {
        py::gil_scoped_release nogil;
        out = rc::receive_pass(ref, s, d, a, sm);
    }
    // a seam's channel keeps the noise table, as the reference's object does
    if (ctx.chan_from_seam && out.chan && rx::has_field(ctx.chan, "_psd") && rx::field(ctx.chan, "_psd").is_none())
        if (const auto t = out.chan->psd.get()) set_psd(ctx.chan, *t);
    if (!out.pass) return py::none();
    py::dict pass = rx::to_py(*out.pass);
    if (!a.return_tracks) return pass;
    py::object chan_o = ctx.chan_from_seam ? ctx.chan : py::object(rx::to_py(*out.chan));
    py::list tracks;
    for (std::size_t i = 0; i < out.tracks.size(); ++i) {
        out.tracks[i]->chan = nullptr;   // the one channel object, below
        py::dict t = rx::to_py(*out.tracks[i]);
        t["chan"] = chan_o;
        // round A ran as the caller's frame, round B as decode_header left it
        t["spec"] = (i == 0 || !out.plain) ? spec : py::object(py::cast(fr::plain(s)));
        tracks.append(t);
    }
    return py::make_tuple(pass, tracks);
}

// --- the header ----------------------------------------------------------------------------

// diag.get(key) as float64, or nullopt for None
std::optional<std::vector<double>> part_of(const py::object& diag, const char* key) {
    const py::object v = diag.attr("get")(key);
    if (v.is_none()) return std::nullopt;
    return rx::vec<double>(v);
}

// Whether decode_header goes past the sum for this frame when it fails.
bool falls_back(const fr::FrameSpec& s) {
    if (!s.has_header() || s.n_hdr() == 0 || s.hdr_rho() == 0.0) return false;
    return true;
}

py::object hdr_py(const std::optional<sstvae::qrss::header::HeaderFields>& h) {
    return h ? py::object(rx::to_py(*h)) : py::object(py::none());
}

py::tuple decode_header_py(const py::object& llr, const py::object& diag, const py::object& spec) {
    const fr::FrameSpec s = rx::spec(spec);
    if (!s.has_header()) return py::make_tuple(py::none(), llr, spec);
    const std::vector<double> l = rx::vec<double>(llr);
    rc::HeaderDecode h = rc::decode_header(l, nullptr, nullptr, s);
    if (!h.header && falls_back(s)) {
        // the sum failed: each copy alone, as diag holds them
        const auto spread = part_of(diag, "llr_spread");
        const auto block = part_of(diag, "llr_block");
        h = rc::decode_header(l, spread ? &*spread : nullptr, block ? &*block : nullptr, s);
    }
    py::object llr_out = llr;
    if (h.block) llr_out = rx::np_module().attr("asarray")(diag.attr("get")("llr_block"), llr.attr("dtype"));
    py::object spec_out = h.plain ? py::object(py::cast(fr::plain(s))) : spec;
    return py::make_tuple(hdr_py(h.header), llr_out, spec_out);
}

py::object decode_header_l_py(const py::object& llr, const py::object& diag, const py::object& spec) {
    const fr::FrameSpec s = rx::spec(spec);
    const std::vector<double> l = rx::vec<double>(llr);
    std::optional<sstvae::qrss::header::HeaderFields> h = rc::decode_header_l(l, nullptr, nullptr, s);
    if (!h && falls_back(s)) {
        // the spread copy, then the block: read as the reference reaches them
        const auto spread = part_of(diag, "llr_spread");
        if (spread) h = rc::decode_header_l(*spread, nullptr, nullptr, s);
        if (!h) {
            const auto block = part_of(diag, "llr_block");
            if (block) h = rc::decode_header_l(*block, nullptr, nullptr, s);
        }
    }
    return hdr_py(h);
}

// --- a stored pass -------------------------------------------------------------------------

py::object detection_from_pass_py(const py::object& p, const py::object& lead_in_s) {
    types::PassResult r;
    r.f_hz = rx::as_double(rx::field(p, "f_hz"));
    r.report = rx::track_report(rx::field(p, "report"));
    r.waveform = rx::as_int(attr_or(p, "waveform", py::int_(cfg::WAVEFORM_CE)));
    const types::Detection d = rc::detection_from_pass(r, rx::as_double(lead_in_s));
    py::dict out = rx::to_py(d);
    out["f_hz"] = rx::field(p, "f_hz");
    out["timing"] = rx::field(p, "timing");
    out["z_ref"] = rx::field(rx::field(p, "report"), "z_ref");
    out["lead_in_s"] = lead_in_s;
    return out;
}

py::object capture_from_pass_py(const py::object& p) {
    py::dict d = rx::typed("Prepared");
    d["fe"] = rx::np_module().attr("asarray")(rx::field(p, "ch"));
    d["keep"] = py::none();
    d["q"] = rx::field(p, "q");
    d["t0_index"] = rx::field(p, "ch_t0_index");
    d["fs"] = rx::field(p, "ch_fs");
    return d;
}

py::list confidence_db_py(const py::object& p, const py::object& spec, const py::object& elapsed_s,
                          const py::object& n) {
    const py::object pw = attr_or(p, "waveform", py::none());
    std::optional<fr::FrameSpec> sp;
    if (!spec.is_none()) sp = rx::spec(spec);
    if (!sp || (!pw.is_none() && rx::as_int(pw) != static_cast<std::int64_t>(sp->waveform())))
        sp = fr::get_any(rx::as_string(rx::field(p, "frame")));
    const std::vector<double> w = rx::vec<double>(rx::field(p, "w"));
    std::optional<std::vector<double>> llr, psi;
    const py::object l = attr_or(p, "hdr_llr", py::none());
    if (!l.is_none()) llr = rx::vec<double>(l);
    const py::object ps = attr_or(p, "psi", py::none());
    if (!ps.is_none() && static_cast<std::int64_t>(py::len(ps)) == sp->n_pos()) psi = rx::vec<double>(ps);
    std::optional<double> el;
    if (!elapsed_s.is_none()) el = rx::as_double(elapsed_s);
    const std::int64_t nb = rx::as_int(n);
    std::vector<std::optional<double>> c;
    {
        py::gil_scoped_release nogil;
        c = rc::confidence_db(*sp, w, llr ? &*llr : nullptr, psi ? &*psi : nullptr, el, nb);
    }
    py::list out;
    for (const auto& v : c) out.append(v ? py::object(py::float_(*v)) : py::object(py::none()));
    return out;
}

void bind(py::module_& qrss) {
    rx::register_translators();
    py::module_ m = qrss.def_submodule("receiver", "native/core/qrss/receiver (receiver.py: one pass)");
    m.attr("CONF_BINS") = rc::CONF_BINS;
    m.attr("CONF_FLOOR_DB") = rc::CONF_FLOOR_DB;
    m.attr("L_EXCLUDE_OFFSET_HZ") = rc::L_EXCLUDE_OFFSET_HZ;
    m.attr("MARGIN_S") = rc::MARGIN_S;

    m.def("prepare", [](const py::object& cap) -> py::object {
        if (rx::is_prepared(cap)) return cap;
        const std::int64_t fs = rx::as_int(rx::field(cap, "fs"));
        if (fs == cfg::CH_FS) {
            py::dict d = rx::typed("Prepared");
            d["fe"] = rx::np_module().attr("asarray")(rx::field(cap, "fe"));
            d["keep"] = py::none();
            d["q"] = rx::field(cap, "q");
            d["t0_index"] = rx::field(cap, "t0_index");
            d["fs"] = cfg::CH_FS;
            return d;
        }
        const fe::Capture c = rx::capture(cap);
        std::optional<rc::Prepared> p;
        {
            py::gil_scoped_release nogil;
            p.emplace(rc::prepare(c));
        }
        py::dict d = rx::to_py(*p);
        d["q"] = rx::field(cap, "q");
        d["t0_index"] = rx::field(cap, "t0_index");
        d["fs"] = rx::field(cap, "fs");
        return d;
    }, py::arg("cap"));

    m.def("frames", &frames_py, py::arg("spec"));
    m.def("is_joint", [](const py::object& spec) {
        const py::tuple f = frames_py(spec);
        for (const py::handle s : f)
            if (rx::field(s, "waveform").equal(py::int_(cfg::WAVEFORM_L))) return true;
        return false;
    }, py::arg("spec"));
    m.def("_single", [](const py::object& spec) -> py::object {
        if (is_seq(spec)) return frames_py(spec)[0];
        return spec;
    }, py::arg("spec"));
    m.def("centre_hz", &centre_of, py::arg("obj"));
    m.def("_centres", [](const py::object& skip) {
        const py::module_ np = rx::np_module();
        const py::object number = py::make_tuple(py::module_::import("builtins").attr("int"),
                                                 py::module_::import("builtins").attr("float"), np.attr("floating"),
                                                 np.attr("integer"));
        py::list out;
        for (const py::handle f : py::reinterpret_borrow<py::iterable>(py::iter(skip))) {
            if (PyObject_IsInstance(f.ptr(), number.ptr()) == 1)
                out.append(rx::as_double(f));
            else
                out.append(centre_of(f));
        }
        return out;
    }, py::arg("skip"));
    m.def("_ce_hints", [](const py::object& hints) -> py::object {
        // the C++ decides which to keep; the bare ones are handed back as they are
        std::vector<py::object> objs;
        std::vector<sstvae::qrss::acquire::Hint> hs;
        bool any_pair = false;
        for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(hints))) {
            objs.push_back(py::reinterpret_borrow<py::object>(h));
            if (is_seq(h)) any_pair = true;
        }
        if (!any_pair) return hints;
        py::list out;
        for (const py::object& h : objs) {
            if (!is_seq(h)) {
                out.append(h);
                continue;
            }
            const sstvae::qrss::acquire::Hint hint{0.0, rx::as_int(h[py::int_(1)])};
            if (rc::ce_hints(std::span<const sstvae::qrss::acquire::Hint>(&hint, 1)).empty()) continue;
            out.append(rx::as_double(h[py::int_(0)]));
        }
        return out;
    }, py::arg("hints"));

    m.def("channel_for", &channel_for_py, py::arg("prep"), py::arg("spec"), py::arg("det"),
          py::arg("exclude_hz") = py::tuple(), py::arg("f_mix_hz") = py::none());

    m.def("_psi_frame", [](const py::object& tr) {
        const std::unique_ptr<tk::TrackResult> t = rx::any_track_result(tr);
        return to_numpy(rc::psi_frame(*t));
    }, py::arg("tr"));

    m.def("_cw_result", [](const py::object& tr, const py::object& spec, const py::object& hdr) -> py::tuple {
        const fr::FrameSpec s = rx::spec(spec);
        if (s.n_win() == 0) return py::make_tuple(py::none(), py::none());
        // the capture's noise table as the reference's chan.noise_psd_at fills it
        const py::object chan = rx::field(tr, "chan");
        if (!py::isinstance<py::dict>(chan) && py::hasattr(chan, "_psd") && chan.attr("_psd").is_none())
            chan.attr("noise_psd_at")(0.0);
        const std::unique_ptr<tk::TrackResult> t = rx::any_track_result(tr);
        std::optional<sstvae::qrss::header::HeaderFields> h;
        if (!hdr.is_none()) h = rx::header_fields(hdr);
        rc::CwOutcome r;
        {
            py::gil_scoped_release nogil;
            r = rc::cw_result(*t, s, h);
        }
        py::object cw = r.cw ? py::object(rx::to_py(*r.cw)) : py::object(py::none());
        py::object known = r.known ? py::object(py::make_tuple(to_numpy(r.known->units), r.known->ook))
                                   : py::object(py::none());
        return py::make_tuple(cw, known);
    }, py::arg("tr"), py::arg("spec"), py::arg("hdr"));

    m.def("receive_pass", &receive_pass_py, py::arg("cap"), py::arg("spec"), py::arg("det"),
          py::arg("prior") = py::none(), py::arg("estimator") = "joint", py::kw_only(),
          py::arg("exclude_hz") = py::tuple(), py::arg("f_mix_hz") = py::none(), py::arg("round_b") = true,
          py::arg("return_tracks") = false, py::arg("erase_s") = py::tuple(), py::arg("_seams") = py::none());

    m.def("decode_header", &decode_header_py, py::arg("llr"), py::arg("diag"), py::arg("spec"));
    m.def("_decode_header_l", &decode_header_l_py, py::arg("llr"), py::arg("diag"), py::arg("spec"));

    m.def("detection_from_pass", &detection_from_pass_py, py::arg("p"), py::arg("lead_in_s") = 0.0);
    m.def("capture_from_pass", &capture_from_pass_py, py::arg("p"));
    m.def("mean_w_db", [](const py::object& p) {
        const std::vector<double> w = rx::vec<double>(rx::field(p, "w"));
        return rc::mean_w_db(w);
    }, py::arg("p"));
    m.def("confidence_db", &confidence_db_py, py::arg("p"), py::arg("spec") = py::none(),
          py::arg("elapsed_s") = py::none(), py::arg("n") = rc::CONF_BINS);
}

const Registrar reg("receiver", 85, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
