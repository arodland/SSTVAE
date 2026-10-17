// sstvae_native.qrss.em_search: the template search and retroactive
// detection of sstvae/qrss/em.py (native/core/qrss/em_search), with the
// reference's names, arguments and defaults (tests/qrss_native/
// em_search.py assigns them over sstvae.qrss.em):
//
//   _candidates(store, acc, extra_hz=()) -> [(f_hz, drift, ppm), ...] floats
//   _chunk_template(spec, mu, nu, keying, ook, t0_index, ppm, n_ch, chunk)
//       -> (starts int64[K], E complex128[K, chunk], en float64[K])
//   _search_chan(z, N0, starts, E, en, lag_max, m_max, drift_bins)
//       -> (lam float, lag int, m int)
//   _search_input(cap, f_mix_hz) -> (Prepared, Fraction | None): a Prepared
//       argument itself, else a "Prepared" dict
//   _from_passband(cap, q, spec) -> a frontend.Capture of the slot for a
//       PassbandStore (the reference's class); anything else as it is
//   template_search(store, key, cap, spec, *, freqs=(), segments=None,
//       tau_s=TS_TAU_S, df_hz=TS_DF_HZ, ppm_steps=TS_PPM_STEPS,
//       z_accept=Z_ACCEPT, f_mix_hz=None, q=None, return_stats=False,
//       cancel=None) -> "Detection" dict | None; with return_stats
//       (Detection | None, Z, info dict)
//   receive_template(store, key, cap, spec, det=None, estimator="joint",
//       segment=None, **search) -> "PassResult" dict | None  (search: the
//       template_search keywords and cancel)
//   retro_detect(store, key, pb, spec, *, slots=None, estimator="joint",
//       log=None, progress=None, cancel=None, _seams=None, **search)
//       -> [(PassResult, key | None, rule), ...]
//
// `store` is sstvae_native.qrss.store.Store (or the rows' facade); `acc` its
// Accumulator; `cap` a frontend.Capture, a receiver.Prepared, a PassResult
// (anything with their fields, or their "__type__" dicts) or a
// PassbandStore (the reference's or the native one: anything of that class
// name with a `root`, read here by the C++ PassbandStore opened on that
// directory, which reads the same files to the bit); `pb` the same kind of
// PassbandStore, or any object with slots(dur) and capture(q, dur) (called
// in Python then). `det` a Detection, `spec` a FrameSpec (either kind).
// log / progress / cancel are em_hooks.hpp's (progress (done, total,
// "slot q=<q>") per slot of the list, cancel a qrss.em.CancelFlag polled
// before every slot and between hypotheses).
//
// `_seams` (retro_detect): what the loop is to call in Python, by the
// reference's name, each with the reference's arguments --
// "template_search" (store, key, cap, spec, return_stats=True, **search),
// "receive_template" (store, key, cap, spec, det=, estimator=, segment=)
// and "associate" (p, store) -- where a test replaced em.template_search,
// em.receive_template or associate.associate (the rows decide). A missing
// name is the C++. A seam is handed the caller's store, (key[0],
// int(key[1])), spec and search keywords, the slot's capture as pb.capture
// gave it (or the reference's frontend.Capture of the C++ read), and the
// records the C++ made as "__type__" dicts (which the rows' wrappers turn
// into the reference's); what a seam returns is read back by field. The
// result tuple carries the receive seam's own pass object and the
// associate seam's own key and rule (a None rule stays None).
//
// THREADS and the GIL: every core call runs with the GIL released; hooks
// and seams take it back for their Python calls, and every Python object a
// step keeps across them is held through em_hooks.hpp's held_object
// (released with the GIL taken). Nothing waits on the store's mutex with
// the GIL held.

#include "qrss/em_hooks.hpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/em.hpp"
#include "qrss/em_search.hpp"
#include "qrss/fspath.hpp"
#include "qrss/passband.hpp"
#include "qrss/receiver.hpp"
#include "qrss/types.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace em = sstvae::qrss::em;
namespace store = sstvae::qrss::store;
namespace types = sstvae::qrss::types;
namespace frame = sstvae::qrss::frame;
namespace pbs = sstvae::qrss::passband;
namespace fe = sstvae::qrss::frontend;
namespace rcv = sstvae::qrss::receiver;
namespace as = sstvae::qrss::associate;
namespace ce = sstvae::qrss::ce;
namespace cfg = sstvae::qrss::config;
using sstvae::qrss::Rational;
using st::nogil;
using st::PyStore;

using emh::held_object;
using Held = std::shared_ptr<py::object>;

// (key[0], int(key[1])): the reference's first line after _ce_only.
py::tuple key_tuple(const py::object& key) {
    py::object k0 = key[py::int_(0)];
    PyObject* k1 = PyNumber_Long(py::object(key[py::int_(1)]).ptr());
    if (!k1) throw py::error_already_set();
    return py::make_tuple(k0, py::reinterpret_steal<py::object>(k1));
}

PyStore& store_of(const py::object& o) { return o.cast<PyStore&>(); }

// Fraction(x) as the reference converts f_mix_hz.
Rational fraction_of(const py::object& o) { return rx::rational(rx::fraction_class()(o)); }

std::optional<Rational> opt_fraction(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    return fraction_of(o);
}

bool is_passband(const py::handle& o) { return rx::type_name(o) == "PassbandStore" && py::hasattr(o, "root"); }

// The C++ PassbandStore on a PassbandStore's directory (reads only).
std::unique_ptr<pbs::PassbandStore> open_passband(const py::handle& o) {
    const fspath::FsPath p = fspath::fs_path(rx::field(o, "root"));
    return nogil([&] { return std::make_unique<pbs::PassbandStore>(p.bytes); });
}

// The reference's frontend.Capture of a C++ capture.
py::object capture_py(const fe::Capture& c) {
    static py::object* cls = new py::object(py::module_::import("sstvae.qrss.frontend").attr("Capture"));
    return (*cls)(py::arg("fe") = to_numpy(c.fe), py::arg("q") = c.q, py::arg("t0_index") = c.t0_index,
                  py::arg("fs") = c.fs);
}

// A stored pass as _search_input reads it: uid, its capture (ch: none or
// empty is the reference's ValueError, raised by the core), q, ch_fs,
// ch_t0_index and f_mix_hz.
types::PassResult pass_capture(const py::handle& o) {
    types::PassResult p;
    p.uid = py::str(rx::field(o, "uid")).cast<std::string>();
    const py::object ch = rx::field(o, "ch");
    if (ch.is_none() || py::len(ch) == 0) return p;
    p.ch_fs = rx::as_int(rx::field(o, "ch_fs"));
    p.ch = rx::capture_fe(ch, p.ch_fs, "a pass's capture");
    p.q = rx::as_int(rx::field(o, "q"));
    p.ch_t0_index = rx::as_double(rx::field(o, "ch_t0_index"));
    p.f_mix_hz = fraction_of(rx::field(o, "f_mix_hz"));
    return p;
}

// A `cap` argument converted, owning what the SearchCap points at.
struct Cap {
    std::optional<fe::Capture> capture;
    std::optional<rcv::Prepared> prep;
    std::optional<types::PassResult> pass;
    std::unique_ptr<pbs::PassbandStore> pb;

    em::SearchCap ref() const {
        if (pb) return pb.get();
        if (pass) return &*pass;
        if (prep) return &*prep;
        return &*capture;
    }
};

Cap cap_of(const py::object& o) {
    Cap c;
    const std::string t = rx::type_name(o);
    if (is_passband(o))
        c.pb = open_passband(o);
    else if (t == "PassResult")
        c.pass = pass_capture(o);
    else if (rx::is_prepared(o))
        c.prep = rx::prepared(o);
    else
        c.capture = rx::capture(o);
    return c;
}

// template_search's keywords as SearchOptions, and the names it does not take.
struct Search {
    em::SearchOptions opt;
    std::vector<std::string> unknown;
};

Search search_of(const py::dict& kw) {
    Search s;
    for (const auto& [k_, v_] : kw) {
        const std::string k = py::str(k_).cast<std::string>();
        const py::object v = py::reinterpret_borrow<py::object>(v_);
        if (k == "freqs") {
            for (const py::handle f : py::reinterpret_borrow<py::iterable>(py::iter(v))) s.opt.freqs.push_back(rx::as_double(f));
        } else if (k == "segments") {
            if (!v.is_none()) {
                std::vector<std::int64_t> g;
                for (const py::handle x : py::reinterpret_borrow<py::iterable>(py::iter(v))) g.push_back(rx::as_int(x));
                s.opt.segments = std::move(g);
            }
        } else if (k == "tau_s") {
            s.opt.tau_s = rx::as_double(v);
        } else if (k == "df_hz") {
            s.opt.df_hz = rx::as_double(v);
        } else if (k == "ppm_steps") {
            s.opt.ppm_steps = rx::as_int(v);
        } else if (k == "z_accept") {
            s.opt.z_accept = rx::as_double(v);
        } else if (k == "f_mix_hz") {
            s.opt.f_mix_hz = opt_fraction(v);
        } else if (k == "q") {
            if (!v.is_none()) s.opt.q = rx::as_int(v);
        } else {
            s.unknown.push_back(k);
        }
    }
    return s;
}

[[noreturn]] void unexpected_keyword(const std::string& fn, const std::string& name) {
    throw py::type_error(fn + "() got an unexpected keyword argument '" + name + "'");
}

py::dict info_py(const em::SearchInfo& i) {
    py::dict d;
    d["lam"] = i.lam;
    d["chunks"] = i.chunks;
    d["cells"] = i.cells;
    d["segment"] = i.segment;
    d["f_hz"] = i.f_hz;
    d["ppm"] = i.ppm;
    d["tau0"] = i.tau0;
    d["drift"] = i.drift;
    return d;
}

py::object det_py(const std::optional<types::Detection>& d) { return d ? py::object(rx::to_py(*d)) : py::object(py::none()); }

py::object result_py(const em::SearchResult& r, bool return_stats) {
    if (!return_stats) return det_py(r.det);
    return py::make_tuple(det_py(r.det), r.z, r.info ? py::object(info_py(*r.info)) : py::object(py::dict()));
}

// (a, b) = seq: Python's unpacking errors.
std::pair<py::object, py::object> unpack2(const py::object& r) {
    const py::sequence s = py::reinterpret_borrow<py::sequence>(py::tuple(r));
    if (s.size() != 2)
        throw py::value_error(s.size() < 2 ? "not enough values to unpack (expected 2, got " + std::to_string(s.size()) + ")"
                                           : "too many values to unpack (expected 2)");
    return {s[0], s[1]};
}

std::tuple<py::object, py::object, py::object> unpack3(const py::object& r) {
    const py::sequence s = py::reinterpret_borrow<py::sequence>(py::tuple(r));
    if (s.size() != 3)
        throw py::value_error(s.size() < 3 ? "not enough values to unpack (expected 3, got " + std::to_string(s.size()) + ")"
                                           : "too many values to unpack (expected 3)");
    return {s[0], s[1], s[2]};
}

// --- retro_detect over the C++ and Python steps ---------------------------------------

py::list retro_detect(const py::object& store_obj, const py::object& key, const py::object& pb, const py::object& spec,
                      const py::object& slots, const py::object& estimator, const py::object& log,
                      const py::object& progress, const py::object& cancel, const py::object& seams_obj,
                      const py::kwargs& search) {
    const frame::FrameSpec sp = rx::spec(spec);
    em::ce_only(sp, "em.retro_detect");
    const py::tuple kt = key_tuple(key);
    const store::Key k = st::key_of(kt);
    PyStore& s = store_of(store_obj);
    const emh::PyHooks h = emh::hooks_from_py(log, progress, cancel);
    const py::dict kw = search ? py::dict(search) : py::dict();
    const Search srch = search_of(kw);
    std::optional<std::vector<std::int64_t>> slot_list;
    if (!slots.is_none()) {
        std::vector<std::int64_t> v;
        for (const py::handle x : py::reinterpret_borrow<py::iterable>(py::iter(slots))) v.push_back(rx::as_int(x));
        slot_list = std::move(v);
    }
    const py::dict seams = seams_obj.is_none() ? py::dict() : py::dict(seams_obj);
    auto seam = [&](const char* name) -> Held {
        if (!seams.contains(name)) return nullptr;
        return held_object(seams[name]);
    };
    const Held ts_seam = seam("template_search"), rt_seam = seam("receive_template"), as_seam = seam("associate");

    std::unique_ptr<pbs::PassbandStore> core_pb;
    if (is_passband(pb)) core_pb = open_passband(pb);

    em::RetroOptions ro;
    ro.slots = slot_list;
    ro.estimator = rx::as_string(estimator);
    ro.search = srch.opt;
    em::RetroSteps steps = em::retro_steps(*s.core, k, core_pb.get(), sp, ro, h.hooks);

    const Held py_store = held_object(store_obj), py_key = held_object(kt), py_spec = held_object(spec),
               py_pb = held_object(pb), py_kw = held_object(kw), py_est = held_object(estimator);
    // the Python det / pass of the slot in hand (a seam's own objects), and
    // the pass object of each find, in order
    auto last_det = std::make_shared<Held>();
    auto last_pass = std::make_shared<Held>();
    auto pass_objs = std::make_shared<std::vector<Held>>();
    // an associate seam's own (key, rule) of each find, in order (a None rule stays None)
    auto assoc_objs = std::make_shared<std::vector<std::pair<Held, Held>>>();

    // the slot's capture as Python sees it (needs the GIL)
    auto cap_obj = [](const em::SlotInputPtr& in) -> py::object {
        if (in->other) return *std::static_pointer_cast<const py::object>(in->other);
        return capture_py(*in->capture);
    };
    // a C++ step's prepared capture of a slot only Python holds
    auto ensure_prep = [](const em::SlotInputPtr& in) {
        if (in->prep || in->capture || !in->other) return;
        std::optional<rcv::Prepared> prepd;
        std::optional<fe::Capture> raw;
        {
            py::gil_scoped_acquire gil;
            const py::object& o = *std::static_pointer_cast<const py::object>(in->other);
            if (rx::is_prepared(o))
                prepd = rx::prepared(o);
            else
                raw = rx::capture(o);
        }
        in->prep = std::make_shared<const rcv::Prepared>(prepd ? std::move(*prepd) : rcv::prepare(*raw));
    };

    if (!core_pb) {
        steps.slots = [py_pb](double dur) {
            py::gil_scoped_acquire gil;
            std::vector<std::int64_t> out;
            const py::object r = py_pb->attr("slots")(dur);
            for (const py::handle x : py::reinterpret_borrow<py::iterable>(py::iter(r))) out.push_back(rx::as_int(x));
            return out;
        };
        steps.capture = [py_pb](std::int64_t q, double dur) -> std::pair<em::SlotInputPtr, double> {
            py::gil_scoped_acquire gil;
            const auto [cap, cov] = unpack2(py_pb->attr("capture")(q, dur));
            if (cap.is_none()) return {nullptr, 0.0};
            auto in = std::make_shared<em::SlotInput>();
            in->other = std::shared_ptr<const void>(held_object(cap));
            return {std::move(in), rx::as_double(cov)};
        };
    }
    const auto c_search = steps.search;
    const std::vector<std::string> unknown = srch.unknown;
    steps.search = [=](const em::SlotInputPtr& in) -> em::SlotSearch {
        if (ts_seam) {
            py::gil_scoped_acquire gil;
            py::dict kwargs(*py_kw);
            kwargs["return_stats"] = true;
            const py::object r = (*ts_seam)(*py_store, *py_key, cap_obj(in), *py_spec, **kwargs);
            const auto [det, z, info] = unpack3(r);
            em::SlotSearch out;
            *last_det = det.is_none() ? nullptr : held_object(det);
            if (!det.is_none()) out.det = rx::detection(det);
            out.z = rx::as_double(z);
            PyObject* g = PyObject_GetItem(info.ptr(), py::str("segment").ptr());
            if (g) {
                out.segment = rx::as_int(py::reinterpret_steal<py::object>(g));
            } else if (PyErr_ExceptionMatches(PyExc_KeyError)) {
                PyErr_Clear();   // read (and raised) only where the reference reads it
            } else {
                throw py::error_already_set();
            }
            return out;
        }
        if (!unknown.empty()) unexpected_keyword("template_search", unknown.front());
        ensure_prep(in);
        *last_det = nullptr;
        return c_search(in);
    };
    const auto c_receive = steps.receive;
    steps.receive = [=](const em::SlotInputPtr& in, const types::Detection& det,
                        std::int64_t segment) -> std::optional<types::PassResult> {
        if (rt_seam) {
            py::gil_scoped_acquire gil;
            const py::object d = *last_det ? **last_det : py::object(rx::to_py(det));
            const py::object r = (*rt_seam)(*py_store, *py_key, cap_obj(in), *py_spec, py::arg("det") = d,
                                            py::arg("estimator") = *py_est, py::arg("segment") = segment);
            if (r.is_none()) {
                *last_pass = nullptr;
                return std::nullopt;
            }
            *last_pass = held_object(r);
            return rx::pass_result(r);
        }
        ensure_prep(in);
        *last_pass = nullptr;
        return c_receive(in, det, segment);
    };
    const auto c_associate = steps.associate;
    steps.associate = [=](const types::PassResult& p) -> as::Result {
        const Held po = *last_pass;
        pass_objs->push_back(po);
        if (!as_seam) {
            assoc_objs->emplace_back(nullptr, nullptr);
            return c_associate(p);
        }
        py::gil_scoped_acquire gil;
        const py::object pobj = po ? *po : py::object(rx::to_py(p));
        const auto [kk, rule] = unpack2((*as_seam)(pobj, *py_store));
        as::Result out;
        if (!kk.is_none()) out.key = st::key_of(kk);
        if (!rule.is_none()) out.rule = rx::as_string(rule);
        assoc_objs->emplace_back(held_object(kk), held_object(rule));
        return out;
    };

    std::vector<em::RetroFind> finds =
        nogil([&] { return em::retro_detect_steps(*s.core, k, sp, steps, slot_list, h.hooks); });
    py::list out;
    for (std::size_t i = 0; i < finds.size(); ++i) {
        const Held& po = (*pass_objs)[i];
        py::object p = po ? *po : py::object(rx::to_py(finds[i].pass));
        const auto& [ko, ro] = (*assoc_objs)[i];
        if (ko)   // the seam's own key and rule, as the reference returns them
            out.append(py::make_tuple(p, *ko, *ro));
        else
            out.append(py::make_tuple(p, st::opt_key(finds[i].key), rx::to_str(finds[i].rule)));
    }
    return out;
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule(
        "em_search", "Port of sstvae/qrss/em.py's template search and retroactive detection (native/core/qrss/em_search)");

    m.def("_candidates", [](PyStore& s, const store::Accumulator& acc, const py::object& extra_hz) {
        std::vector<double> extra;
        for (const py::handle f : py::reinterpret_borrow<py::iterable>(py::iter(extra_hz))) extra.push_back(rx::as_double(f));
        const auto c = nogil([&] { return em::candidates(*s.core, acc, extra); });
        py::list out;
        for (const em::Hypothesis& h : c) out.append(py::make_tuple(h.f_hz, h.drift_hz_per_min, h.ppm));
        return out;
    }, py::arg("store"), py::arg("acc"), py::arg("extra_hz") = py::tuple());

    m.def("_chunk_template", [](const py::object& spec, const py::object& mu, const py::object& nu,
                                const py::object& keying, const py::object& ook, const py::object& t0_index,
                                const py::object& ppm, const py::object& n_ch, const py::object& chunk) {
        const frame::FrameSpec sp = rx::spec(spec);
        const std::vector<double> m = from_numpy<double>(mu), n = from_numpy<double>(nu);
        std::optional<ce::Keying> kg;
        if (!keying.is_none()) {
            const py::array a = rx::np_module().attr("asarray")(keying);
            const std::vector<double> v = from_numpy<double>(a);
            const std::vector<std::int64_t> shape(a.shape(), a.shape() + a.ndim());
            kg = ce::window_keying(v, shape, sp);
        }
        const bool o = rx::as_bool(ook);
        const double t0 = rx::as_double(t0_index), pp = rx::as_double(ppm);
        const std::int64_t nc = rx::as_int(n_ch), ck = rx::as_int(chunk);
        const em::ChunkTemplate t = nogil([&] {
            return em::chunk_template(sp, m, n, kg ? &*kg : nullptr, o, t0, pp, nc, ck);
        });
        py::array_t<std::complex<double>> E({static_cast<py::ssize_t>(t.size()), static_cast<py::ssize_t>(t.chunk)});
        std::copy(t.E.begin(), t.E.end(), E.mutable_data());
        return py::make_tuple(to_numpy(t.starts), E, to_numpy(t.en));
    }, py::arg("spec"), py::arg("mu"), py::arg("nu"), py::arg("keying"), py::arg("ook"), py::arg("t0_index"),
          py::arg("ppm"), py::arg("n_ch"), py::arg("chunk"));

    m.def("_search_chan", [](const py::object& z, const py::object& N0, const py::object& starts, const py::object& E,
                             const py::object& en, const py::object& lag_max, const py::object& m_max,
                             const py::object& drift_bins) {
        const std::vector<std::complex<double>> zz = from_numpy<std::complex<double>>(z);
        const py::array Ea = rx::np_module().attr("asarray")(E, py::arg("dtype") = "complex128");
        if (Ea.ndim() != 2) throw py::value_error("E must be (chunks, chunk)");
        em::ChunkTemplate t;
        t.chunk = static_cast<std::int64_t>(Ea.shape(1));
        t.E = from_numpy<std::complex<double>>(Ea);
        t.starts = from_numpy<std::int64_t>(starts);
        t.en = from_numpy<double>(en);
        if (t.starts.size() != static_cast<std::size_t>(Ea.shape(0)) || t.en.size() != t.starts.size())
            throw py::value_error("starts, E and en must hold one row per chunk");
        const std::vector<std::int64_t> db = from_numpy<std::int64_t>(drift_bins);
        const double n0 = rx::as_double(N0);
        const std::int64_t lm = rx::as_int(lag_max), mm = rx::as_int(m_max);
        const em::ChanPeak pk = nogil([&] { return em::search_chan(zz, n0, t, lm, mm, db); });
        return py::make_tuple(pk.lam, pk.lag, pk.m);
    }, py::arg("z"), py::arg("N0"), py::arg("starts"), py::arg("E"), py::arg("en"), py::arg("lag_max"),
          py::arg("m_max"), py::arg("drift_bins"));

    m.def("_search_input", [](const py::object& cap, const py::object& f_mix_hz) -> py::object {
        const std::optional<Rational> fm = opt_fraction(f_mix_hz);
        const std::string t = rx::type_name(cap);
        em::SearchInput in;
        if (t == "PassResult") {
            const types::PassResult p = pass_capture(cap);
            in = nogil([&] { return em::search_input(p); });
        } else if (rx::is_prepared(cap)) {
            const rcv::Prepared p = rx::prepared(cap);
            in = em::search_input(p, fm);
            // receiver.prepare hands a Prepared back as it is
            return py::make_tuple(cap, in.f_mix ? rx::to_py(*in.f_mix) : py::object(py::none()));
        } else {
            const fe::Capture c = rx::capture(cap);
            in = nogil([&] { return em::search_input(c, fm); });
        }
        return py::make_tuple(rx::to_py(*in.prep), in.f_mix ? rx::to_py(*in.f_mix) : py::object(py::none()));
    }, py::arg("cap"), py::arg("f_mix_hz"));

    m.def("_from_passband", [](const py::object& cap, const py::object& q, const py::object& spec) -> py::object {
        if (!is_passband(cap)) return cap;
        const frame::FrameSpec sp = rx::spec(spec);
        std::optional<std::int64_t> qq;
        if (!q.is_none()) qq = rx::as_int(q);
        const std::unique_ptr<pbs::PassbandStore> pb = open_passband(cap);
        const fe::Capture c = nogil([&] { return em::from_passband(*pb, qq, sp); });
        return capture_py(c);
    }, py::arg("cap"), py::arg("q"), py::arg("spec"));

    m.def("template_search", [](const py::object& store_obj, const py::object& key, const py::object& cap,
                                const py::object& spec, const py::object& freqs, const py::object& segments,
                                const py::object& tau_s, const py::object& df_hz, const py::object& ppm_steps,
                                const py::object& z_accept, const py::object& f_mix_hz, const py::object& q,
                                bool return_stats, const py::object& cancel) {
        const frame::FrameSpec sp = rx::spec(spec);
        em::ce_only(sp, "em.template_search");
        const store::Key k = st::key_of(key_tuple(key));
        PyStore& s = store_of(store_obj);
        py::dict kw;
        kw["freqs"] = freqs;
        kw["segments"] = segments;
        kw["tau_s"] = tau_s;
        kw["df_hz"] = df_hz;
        kw["ppm_steps"] = ppm_steps;
        kw["z_accept"] = z_accept;
        kw["f_mix_hz"] = f_mix_hz;
        kw["q"] = q;
        const Search srch = search_of(kw);
        const emh::PyHooks h = emh::hooks_from_py(py::none(), py::none(), cancel);
        const Cap c = cap_of(cap);
        const em::SearchResult r = nogil([&] { return em::template_search(*s.core, k, c.ref(), sp, srch.opt, h.hooks); });
        return result_py(r, return_stats);
    }, py::arg("store"), py::arg("key"), py::arg("cap"), py::arg("spec"), py::kw_only(), py::arg("freqs") = py::tuple(),
          py::arg("segments") = py::none(), py::arg("tau_s") = em::TS_TAU_S, py::arg("df_hz") = em::TS_DF_HZ,
          py::arg("ppm_steps") = em::TS_PPM_STEPS, py::arg("z_accept") = cfg::Z_ACCEPT,
          py::arg("f_mix_hz") = py::none(), py::arg("q") = py::none(), py::arg("return_stats") = false,
          py::arg("cancel") = py::none());

    m.def("receive_template", [](const py::object& store_obj, const py::object& key, const py::object& cap,
                                 const py::object& spec, const py::object& det, const py::object& estimator,
                                 const py::object& segment, const py::kwargs& search) -> py::object {
        const frame::FrameSpec sp = rx::spec(spec);
        em::ce_only(sp, "em.receive_template");
        const store::Key k = st::key_of(key_tuple(key));
        PyStore& s = store_of(store_obj);
        py::dict kw = search ? py::dict(search) : py::dict();
        py::object cancel = py::none();
        if (kw.contains("cancel")) cancel = kw.attr("pop")("cancel");
        const Search srch = search_of(kw);
        if (det.is_none() && !srch.unknown.empty()) {
            // the reference raises it calling template_search, after reading the capture
            const Cap c0 = cap_of(cap);
            (void)nogil([&] {
                const std::optional<Rational>& fm = srch.opt.f_mix_hz;
                if (c0.pb)
                    (void)em::search_input(em::from_passband(*c0.pb, srch.opt.q, sp), fm);
                else if (c0.pass)
                    (void)em::search_input(*c0.pass);
                else if (c0.prep)
                    (void)em::search_input(*c0.prep, fm);
                else
                    (void)em::search_input(*c0.capture, fm);
                return 0;
            });
            unexpected_keyword("template_search", srch.unknown.front());
        }
        std::optional<types::Detection> d;
        if (!det.is_none()) d = rx::detection(det);
        std::optional<std::int64_t> seg;
        if (!segment.is_none()) seg = rx::as_int(segment);
        const std::string est = rx::as_string(estimator);
        const emh::PyHooks h = emh::hooks_from_py(py::none(), py::none(), cancel);
        const Cap c = cap_of(cap);
        const auto p = nogil([&] { return em::receive_template(*s.core, k, c.ref(), sp, d, est, seg, srch.opt, h.hooks); });
        if (!p) return py::none();
        return rx::to_py(*p);
    }, py::arg("store"), py::arg("key"), py::arg("cap"), py::arg("spec"), py::arg("det") = py::none(),
          py::arg("estimator") = "joint", py::arg("segment") = py::none());

    m.def("retro_detect", &retro_detect, py::arg("store"), py::arg("key"), py::arg("pb"), py::arg("spec"),
          py::kw_only(), py::arg("slots") = py::none(), py::arg("estimator") = "joint", py::arg("log") = py::none(),
          py::arg("progress") = py::none(), py::arg("cancel") = py::none(), py::arg("_seams") = py::none());
}

const Registrar reg("em_search", 99, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
