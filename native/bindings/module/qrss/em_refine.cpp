// sstvae_native.qrss.em_refine: the EM loop of sstvae/qrss/em.py
// (native/core/qrss/em_refine), with the reference's names, arguments and
// defaults:
//
//   pass_path(p, chan=None) -> "FreqPath" dict
//   _pass_path_trace(p, chan=None) -> {"line", "refined": "FreqPath" dicts,
//       "z_line", "z_refined": float, "choice": 0 (the line) or 1}
//       what pass_path measured, for the parity tests
//   rereceive(p, prior, hdr=None, cw_known=None, estimator=None,
//             em_round=None, path=None, return_path=False, split=True, *,
//             cancel=None)
//       -> "PassResult" dict or None; with return_path (pass, "FreqPath"
//          dict), (None, None) for an L pass or one without a capture
//   em_refine(store, key, rounds=EM_ROUNDS, tol_db=EM_TOL_DB, *,
//             estimator=None, use_header=True, log=None, progress=None,
//             cancel=None, _seams=None) -> [float]
//       log, progress and cancel are em.cpp's hooks (em_hooks.hpp): log(str)
//       with the reference's lines, progress(done, total, what), cancel a
//       sstvae_native.qrss.em.CancelFlag (em.Cancelled when it is seen).
//
// `_seams`: what em_refine is to call in Python, by the reference's name --
// "rereceive" only -- called as the reference calls it:
// rereceive(p, prior, hdr, cw_known, estimator, r, path=path,
// return_path=True), with p, prior, hdr and path as "PassResult" /
// "EmPrior" / "HeaderFields" / "FreqPath" dicts (or None) for the row to turn
// into the reference's objects, cw_known as known_keying's (units, ook)
// tuple or None, estimator None or a str and r an int. It must return
// (pass or None, path or None), the pass anything with a PassResult's
// fields. A missing name is the C++.
//
// `p` is a PassResult (anything with its fields; ch complex64, as every pass
// the receiver makes holds it: another dtype is NotImplementedError), `prior`
// an EmPrior or None, `hdr` a HeaderFields or None, `cw_known` None or
// (keying units, ook) with the units a (192,) array (None units are
// NotImplementedError: nothing in the reference makes them), `path` a
// FreqPath or None, `chan` a ChanCapture (the reference's or a dict) or
// None. `store` is sstvae_native.qrss.store.Store (or the rows' facade).
// The records come back as rx_types.hpp's "__type__" dicts, which
// tests/qrss_native/em_refine.py turns into the reference's objects.
//
// The GIL is released while a pass is re-received and around the whole of
// em_refine; the hooks and the seam take it back for each call. A chan the
// caller hands pass_path keeps the noise table the C++ computed for it in
// its own `_psd`, as the reference's object would.

#include "qrss/em_hooks.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "qrss/qrss_config.hpp"
#include "qrss/em.hpp"
#include "qrss/em_refine.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/receiver.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace em = sstvae::qrss::em;
namespace store = sstvae::qrss::store;
namespace types = sstvae::qrss::types;
namespace header = sstvae::qrss::header;
namespace frame = sstvae::qrss::frame;
namespace tk = sstvae::qrss::track;
namespace rc = sstvae::qrss::receiver;
namespace cfg = sstvae::qrss::config;
using st::nogil;
using st::PyStore;

[[noreturn]] void raise(PyObject* type, const std::string& what) {
    PyErr_SetString(type, what.c_str());
    throw py::error_already_set();
}

// `a, b = obj` as CPython 3.13 unpacks it, refusals and messages included.
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

// A pass's frame, from p.frame and p.header.version alone (pass_spec's reading).
types::PassResult frame_of(const py::handle& p) {
    types::PassResult pr;
    pr.frame = rx::as_string(rx::field(p, "frame"));
    const py::object h = rx::field(p, "header");
    if (!h.is_none()) {
        header::HeaderFields hf;
        hf.version = rx::as_int(rx::field(h, "version"));
        pr.header = hf;
    }
    return pr;
}

// p.ch is None or len(p.ch) == 0
bool no_capture(const py::handle& p) {
    const py::object ch = rx::field(p, "ch");
    return ch.is_none() || py::len(ch) == 0;
}

// The whole pass, its capture complex64 as the receiver keeps it.
types::PassResult pass_of(const py::handle& p) {
    const py::object ch = rx::field(p, "ch");
    if (!ch.is_none()) (void)rx::capture_fe(ch, rx::as_int(rx::field(p, "ch_fs")), "a pass's capture");
    return rx::pass_result(p);
}

// cw_known: None, or (units, ook) as known_keying gives it.
std::optional<rc::CwKnown> cw_known_of(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    const auto [k, ook] = unpack2(o);
    if (k.is_none()) rx::not_implemented("cw_known with no keying units (None, ook): nothing in em.py makes one");
    py::array a = rx::np_module().attr("asarray")(k, py::arg("dtype") = "uint8");
    if (a.ndim() != 1 || a.shape(0) != cfg::CW_UNITS)
        rx::not_implemented("cw_known units of another shape than (" + std::to_string(cfg::CW_UNITS) +
                            ",), as known_keying makes them");
    rc::CwKnown c;
    c.units = rx::vec<std::uint8_t>(a);
    c.ook = rx::as_bool(ook);
    return c;
}

std::optional<types::EmPrior> prior_of(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    return rx::em_prior(o);
}

std::optional<header::HeaderFields> hdr_of(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    return rx::header_fields(o);
}

// `estimator or p.estimator`: a falsy estimator is the pass's own.
std::optional<std::string> estimator_of(const py::object& o) {
    if (!rx::as_bool(o)) return std::nullopt;
    return rx::as_string(o);
}

py::object opt_py(const std::optional<types::FreqPath>& p) { return p ? py::object(rx::to_py(*p)) : py::object(py::none()); }

py::object opt_py(const std::optional<types::PassResult>& p) {
    return p ? py::object(rx::to_py(*p)) : py::object(py::none());
}

void set_psd(const py::object& chan, const tk::NoisePsd& t) {
    py::object v = py::make_tuple(to_numpy(t.t_s), to_numpy(t.psd));
    if (py::isinstance<py::dict>(chan))
        py::reinterpret_borrow<py::dict>(chan)["_psd"] = v;
    else
        py::setattr(chan, "_psd", v);
}

// --- pass_path -------------------------------------------------------------------------------

em::PassPathTrace pass_path_run(const py::object& p, const py::object& chan) {
    // pass_spec and _ce_only first, as the reference reads them
    em::ce_only(em::pass_spec(frame_of(p)), "em.pass_path");
    const types::PassResult pr = pass_of(p);
    std::shared_ptr<tk::ChanCapture> c;
    if (!chan.is_none()) {
        tk::ChanCapture cc = rx::chan_capture(chan);
        cc.ch = rx::capture_fe(rx::field(chan, "ch"), cc.fs, "a channel");
        c = std::make_shared<tk::ChanCapture>(std::move(cc));
    }
    em::PassPathTrace t;
    nogil([&] { (void)em::pass_path(pr, c, &t); });
    // the caller's channel keeps the noise table, as the reference's object does
    if (c && rx::has_field(chan, "_psd") && rx::field(chan, "_psd").is_none())
        if (const auto psd = c->psd.get()) set_psd(chan, *psd);
    return t;
}

py::object trace_py(const em::PassPathTrace& t) {
    py::dict d;
    d["line"] = rx::to_py(t.line);
    d["refined"] = rx::to_py(t.refined);
    d["z_line"] = t.z_line;
    d["z_refined"] = t.z_refined;
    d["choice"] = t.choice;
    return d;
}

// --- rereceive -------------------------------------------------------------------------------

py::object rereceive_py(const py::object& p, const py::object& prior, const py::object& hdr,
                        const py::object& cw_known, const py::object& estimator, const py::object& em_round,
                        const py::object& path, const py::object& return_path, const py::object& split,
                        const py::object& cancel) {
    const bool rp = rx::as_bool(return_path);
    auto nothing = [rp]() -> py::object { return rp ? py::object(py::make_tuple(py::none(), py::none())) : py::none(); };
    // an L pass, or one without a capture: None, before anything else is read
    if (em::pass_spec(frame_of(p)).waveform() != frame::Waveform::CE) return nothing();
    if (no_capture(p)) return nothing();
    const types::PassResult pr = pass_of(p);
    const std::optional<types::EmPrior> pri = prior_of(prior);
    em::RereceiveArgs a;
    a.hdr = hdr_of(hdr);
    a.cw_known = cw_known_of(cw_known);
    a.estimator = estimator_of(estimator);
    if (!em_round.is_none()) a.em_round = rx::as_int(em_round);
    if (!path.is_none()) a.path = rx::freq_path(path);
    a.split = rx::as_bool(split);
    const emh::PyHooks h = emh::hooks_from_py(py::none(), py::none(), cancel);
    const em::Rereceived r = nogil([&] { return em::rereceive(pr, pri, a, h.hooks); });
    if (!rp) return opt_py(r.pass);
    return py::make_tuple(opt_py(r.pass), opt_py(r.path));
}

// --- em_refine -------------------------------------------------------------------------------

// The seams, as the module docstring says.
em::EmSeams make_seams(const py::object& seams) {
    em::EmSeams s;
    if (seams.is_none()) return s;
    const py::dict d(seams);
    if (!d.contains("rereceive")) return s;
    auto fn = emh::held_object(d["rereceive"]);
    s.rereceive = [fn](const types::PassResult& p, const types::EmPrior& prior,
                       const std::optional<header::HeaderFields>& hdr, const std::optional<rc::CwKnown>& cw,
                       const std::optional<std::string>& estimator, std::int64_t r,
                       const std::optional<types::FreqPath>& path) {
        py::gil_scoped_acquire gil;
        py::object hdr_o = hdr ? py::object(rx::to_py(*hdr)) : py::object(py::none());
        py::object cw_o = cw ? py::object(py::make_tuple(to_numpy(cw->units), py::bool_(cw->ook)))
                             : py::object(py::none());
        py::object est_o = estimator ? py::object(rx::to_str(*estimator)) : py::object(py::none());
        py::object got = (*fn)(rx::to_py(p), rx::to_py(prior), hdr_o, cw_o, est_o, r, py::arg("path") = opt_py(path),
                               py::arg("return_path") = true);
        const auto [np_, pth] = unpack2(got);
        em::Rereceived out;
        if (!np_.is_none()) out.pass = pass_of(np_);
        if (!pth.is_none()) out.path = rx::freq_path(pth);
        return out;
    };
    return s;
}

py::list em_refine_py(PyStore& s, const py::object& key, const py::object& rounds,
                     const py::object& tol_db, const py::object& estimator, const py::object& use_header,
                     const py::object& log, const py::object& progress, const py::object& cancel,
                     const py::object& seams) {
    // key = (key[0], int(key[1]))
    const py::object k0 = key[py::int_(0)];
    const py::object k1 = key[py::int_(1)];
    const store::Key k = st::key_of(py::make_tuple(k0, py::int_(rx::as_int(k1))));
    em::EmRefineArgs a;
    a.rounds = rx::as_int(rounds);
    a.tol_db = rx::as_double(tol_db);
    if (!estimator.is_none()) a.estimator = rx::as_string(estimator);
    a.use_header = rx::as_bool(use_header);
    const emh::PyHooks h = emh::hooks_from_py(log, progress, cancel);
    const em::EmSeams sm = make_seams(seams);
    store::Store& core = *s.core;
    const std::vector<double> hist = nogil([&] { return em::em_refine(core, k, a, h.hooks, sm); });
    py::list out;
    for (const double x : hist) out.append(x);
    return out;
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("em_refine", "Port of sstvae/qrss/em.py, the EM loop (native/core/qrss/em_refine)");
    m.attr("EM_ROUNDS") = em::EM_ROUNDS;
    m.attr("EM_TOL_DB") = em::EM_TOL_DB;

    m.def("pass_path", [](const py::object& p, const py::object& chan) {
        const em::PassPathTrace t = pass_path_run(p, chan);
        return rx::to_py(t.choice == 0 ? t.line : t.refined);
    }, py::arg("p"), py::arg("chan") = py::none());
    m.def("_pass_path_trace", [](const py::object& p, const py::object& chan) { return trace_py(pass_path_run(p, chan)); },
          py::arg("p"), py::arg("chan") = py::none());
    m.def("rereceive", &rereceive_py, py::arg("p"), py::arg("prior"), py::arg("hdr") = py::none(),
          py::arg("cw_known") = py::none(), py::arg("estimator") = py::none(), py::arg("em_round") = py::none(),
          py::arg("path") = py::none(), py::arg("return_path") = false, py::arg("split") = true, py::kw_only(),
          py::arg("cancel") = py::none());
    m.def("em_refine", &em_refine_py, py::arg("store"), py::arg("key"), py::arg("rounds") = em::EM_ROUNDS,
          py::arg("tol_db") = em::EM_TOL_DB, py::kw_only(), py::arg("estimator") = py::none(),
          py::arg("use_header") = true, py::arg("log") = py::none(), py::arg("progress") = py::none(),
          py::arg("cancel") = py::none(), py::arg("_seams") = py::none());
}

const Registrar reg("em_refine", 99, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
