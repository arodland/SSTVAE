// sstvae_native.qrss.em: the shared half of sstvae/qrss/em.py
// (native/core/qrss/em), with the reference's names, arguments and
// defaults, plus the hooks every long EM / retro call takes.
//
//   EM_ROUNDS EM_TOL_DB TS_CHUNK_S TS_NFFT TS_DF_HZ TS_TAU_S TS_PPM_STEPS
//   TS_SLIP_SAMPLES TS_CH_REACH_HZ RETRO_MIN_COVERAGE RETRO_SAME_HZ
//   lmmse_latents(S, W) -> (a, r)       float64 arrays (a of S and W's
//                                       broadcast shape, r of W's), numpy
//                                       scalars for 0-d
//   prior_from(S, W, q) -> "EmPrior" dict    (S, W flattened)
//   em_prior(acc, uid, store, q, spec) -> "EmPrior" dict
//   acc_mean_w_db(acc) -> float
//   known_header(store, acc, segment) -> "HeaderFields" dict | None
//   known_keying(store, acc, call, waveform=WAVEFORM_CE)
//       -> (uint8[192] keying units, bool ook) | None
//   pass_spec(p) -> FrameSpec           (reads p.frame and p.header.version only)
//   is_l(p) -> bool
//   _ce_only(spec, what) -> None
//   _psi_frame(tr) -> float32[n_pos]
//   _path_of(tr) -> "FreqPath" dict
//   _flat(f_hz) -> "FreqPath" dict
//   _gauss_z(log_p) -> float
//   template_z(lam, n_chunks, n_cells) -> float   (n_chunks as int(n_chunks))
//
//   CancelFlag()  .set() .clear() .is_set()   the `cancel` of the hooks
//   Cancelled     (RuntimeError) what a cancelled call raises
//   _run_hooks(n, log=None, progress=None, cancel=None) -> None
//       a test vehicle for the hooks (em_hooks.hpp): with the GIL released,
//       n steps of check_cancel, log "step i of n", progress (i + 1, n,
//       "steps"); a CancelFlag set during step k stops it at step k + 1.
//
// `store` is sstvae_native.qrss.store.Store (or the rows' facade, a
// subclass), `acc` its Accumulator, `p` a PassResult (anything with its
// fields), `tr` a TrackResult or LTrackResult (or its dict), `spec` a
// FrameSpec (the reference's or the native one). The records come back as
// rx_types.hpp's "__type__" dicts and the native FrameSpec, which
// tests/qrss_native/em.py turns into the reference's objects (_rx.to_ref).
//
// The GIL is released around every core call that reads the store or does
// more than a few microseconds of arithmetic; no Python object is built
// while it is released. The store's own mutex is taken and released by each
// store call inside (store.hpp); nothing here holds the GIL while waiting
// on it.

#include "qrss/em_hooks.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "qrss/config.hpp"
#include "qrss/em.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/types.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace em = sstvae::qrss::em;
namespace store = sstvae::qrss::store;
namespace types = sstvae::qrss::types;
namespace header = sstvae::qrss::header;
namespace frame = sstvae::qrss::frame;
using st::nogil;
using st::PyStore;

store::Store& core(PyStore& s) { return *s.core; }

// --- lmmse_latents' broadcasting ----------------------------------------------------------

using Shape = std::vector<py::ssize_t>;

py::array f64(const py::object& x) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(x, py::arg("dtype") = "float64"));
    if (!a) throw py::error_already_set();
    return a;
}

Shape shape_of(const py::array& a) { return Shape(a.shape(), a.shape() + a.ndim()); }

// numpy's spelling of a shape in its broadcast error: "(3,)", "(2,3)", "()".
std::string shape_text(const Shape& s) {
    std::string out = "(";
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(s[i]);
    }
    if (s.size() == 1) out += ",";
    return out + ")";
}

Shape broadcast(const Shape& a, const Shape& b) {
    const std::size_t n = std::max(a.size(), b.size());
    Shape out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const py::ssize_t x = i < n - a.size() ? 1 : a[i - (n - a.size())];
        const py::ssize_t y = i < n - b.size() ? 1 : b[i - (n - b.size())];
        if (x != y && x != 1 && y != 1)
            throw py::value_error("operands could not be broadcast together with shapes " + shape_text(a) + " " +
                                  shape_text(b) + " ");
        out[i] = x == 1 ? y : x;
    }
    return out;
}

std::vector<double> flat_as(const py::array& a, const Shape& to) {
    py::tuple shape(to.size());
    for (std::size_t i = 0; i < to.size(); ++i) shape[i] = py::int_(to[i]);
    const py::object b = rx::np_module().attr("broadcast_to")(a, shape);
    return rx::vec<double>(b);
}

py::object shaped(const std::vector<double>& v, const Shape& s) {
    py::array_t<double> out(s);
    std::copy(v.begin(), v.end(), out.mutable_data());
    if (s.empty()) return rx::np_module().attr("float64")(v.empty() ? 0.0 : v[0]);
    return out;
}

// --- a pass's frame, from p.frame and p.header.version alone -----------------------------

types::PassResult frame_of(const py::object& p) {
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

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("em", "Port of sstvae/qrss/em.py, the shared half (native/core/qrss/em)");
    m.attr("EM_ROUNDS") = em::EM_ROUNDS;
    m.attr("EM_TOL_DB") = em::EM_TOL_DB;
    m.attr("TS_CHUNK_S") = em::TS_CHUNK_S;
    m.attr("TS_NFFT") = em::TS_NFFT;
    m.attr("TS_DF_HZ") = em::TS_DF_HZ;
    m.attr("TS_TAU_S") = em::TS_TAU_S;
    m.attr("TS_PPM_STEPS") = em::TS_PPM_STEPS;
    m.attr("TS_SLIP_SAMPLES") = em::TS_SLIP_SAMPLES;
    m.attr("TS_CH_REACH_HZ") = em::TS_CH_REACH_HZ;
    m.attr("RETRO_MIN_COVERAGE") = em::RETRO_MIN_COVERAGE;
    m.attr("RETRO_SAME_HZ") = em::RETRO_SAME_HZ;

    // --- the hooks ---
    py::class_<emh::CancelFlag>(m, "CancelFlag",
                                "The cancel of a long EM / retro call: set() from any thread stops the "
                                "call at its next poll, which raises Cancelled.")
        .def(py::init<>())
        .def("set", [](emh::CancelFlag& f) { f.flag.store(true, std::memory_order_relaxed); })
        .def("clear", [](emh::CancelFlag& f) { f.flag.store(false, std::memory_order_relaxed); })
        .def("is_set", [](const emh::CancelFlag& f) { return f.flag.load(std::memory_order_relaxed); });
    py::register_local_exception<em::Cancelled>(m, "Cancelled", PyExc_RuntimeError);
    m.def("_run_hooks", [](std::int64_t n, const py::object& log, const py::object& progress,
                           const py::object& cancel) {
        const emh::PyHooks h = emh::hooks_from_py(log, progress, cancel);
        nogil([&] {
            for (std::int64_t i = 0; i < n; ++i) {
                em::check_cancel(h.hooks);
                em::log_line(h.hooks, "step " + std::to_string(i) + " of " + std::to_string(n));
                em::report_progress(h.hooks, i + 1, n, "steps");
            }
        });
    }, py::arg("n"), py::arg("log") = py::none(), py::arg("progress") = py::none(), py::arg("cancel") = py::none());

    // --- references ---
    m.def("lmmse_latents", [](const py::object& S, const py::object& W) {
        const py::array s = f64(S), w = f64(W);
        const Shape ss = shape_of(s), ws = shape_of(w);
        const Shape bs = broadcast(ss, ws);
        const std::vector<double> sv = flat_as(s, bs), wb = flat_as(w, bs), wv = rx::vec<double>(w);
        const auto [a, r] = nogil([&] {
            em::Lmmse ab = em::lmmse_latents(sv, wb);
            em::Lmmse own = em::lmmse_latents(wv, wv);
            return std::pair{std::move(ab.a), std::move(own.r)};
        });
        return py::make_tuple(shaped(a, bs), shaped(r, ws));
    }, py::arg("S"), py::arg("W"));
    m.def("prior_from", [](const py::object& S, const py::object& W, const py::object& q_) {
        const std::vector<double> s = rx::vec<double>(f64(S)), w = rx::vec<double>(f64(W));
        // scrambler's q = int(q) and its range check, after lmmse_latents'
        // refusal (the reference's order)
        PyObject* qo = PyNumber_Long(q_.ptr());
        if (!qo) throw py::error_already_set();
        const py::int_ qi = py::reinterpret_steal<py::int_>(qo);
        const bool fits = qi >= py::int_(0) && qi < py::int_(1).attr("__lshift__")(64);
        const std::string refusal = "q must fit in uint64, not " + py::str(qi).cast<std::string>();
        const std::uint64_t qq = fits ? qi.cast<std::uint64_t>() : 0;
        return rx::to_py(nogil([&] {
            if (!fits) {
                (void)em::lmmse_latents(s, w);
                throw std::invalid_argument(refusal);
            }
            return em::prior_from_u64(s, w, qq);
        }));
    }, py::arg("S"), py::arg("W"), py::arg("q"));
    m.def("em_prior", [](const store::Accumulator& acc, const py::object& uid, PyStore& s, const py::object& q_,
                         const py::object& spec) {
        const std::string u = rx::as_string(uid);
        const std::int64_t qq = rx::as_int(q_);
        const frame::FrameSpec sp = rx::spec(spec);
        return rx::to_py(nogil([&] { return em::em_prior(acc, u, core(s), qq, sp); }));
    }, py::arg("acc"), py::arg("uid"), py::arg("store"), py::arg("q"), py::arg("spec"));
    m.def("acc_mean_w_db", [](const store::Accumulator& acc) {
        return nogil([&] { return em::acc_mean_w_db(acc); });
    }, py::arg("acc"));

    // --- what every member pass is known to have sent ---
    m.def("known_header", [](PyStore& s, const store::Accumulator& acc, const py::object& segment) -> py::object {
        const std::int64_t g = rx::as_int(segment);
        const auto h = nogil([&] { return em::known_header(core(s), acc, g); });
        if (!h) return py::none();
        return rx::to_py(*h);
    }, py::arg("store"), py::arg("acc"), py::arg("segment"));
    m.def("known_keying", [](PyStore& s, const store::Accumulator& acc, const py::object& call,
                             const py::object& waveform) -> py::object {
        // check_callsign refuses anything but a str (ValueError), which the
        // reference swallows: None
        if (!call.is_none() && !PyUnicode_Check(call.ptr())) return py::none();
        const std::optional<std::string> c = rx::opt_string(call);
        const std::int64_t wf = rx::as_int(waveform);
        const auto k = nogil([&] { return em::known_keying(core(s), acc, c, wf); });
        if (!k) return py::none();
        return py::make_tuple(to_numpy(k->units), py::bool_(k->ook));
    }, py::arg("store"), py::arg("acc"), py::arg("call"),
          py::arg("waveform") = sstvae::qrss::config::WAVEFORM_CE);

    // --- a stored pass ---
    m.def("pass_spec", [](const py::object& p) { return rx::to_py(em::pass_spec(frame_of(p))); }, py::arg("p"));
    m.def("is_l", [](const py::object& p) { return em::is_l(frame_of(p)); }, py::arg("p"));
    m.def("_ce_only", [](const py::object& spec, const py::object& what) {
        em::ce_only(rx::spec(spec), rx::as_string(what));
    }, py::arg("spec"), py::arg("what"));
    m.def("_psi_frame", [](const py::object& tr) {
        const std::unique_ptr<sstvae::qrss::track::TrackResult> t = rx::any_track_result(tr);
        return to_numpy(nogil([&] { return em::psi_frame(*t); }));
    }, py::arg("tr"));
    m.def("_path_of", [](const py::object& tr) {
        const std::unique_ptr<sstvae::qrss::track::TrackResult> t = rx::any_track_result(tr);
        return rx::to_py(nogil([&] { return em::path_of(*t); }));
    }, py::arg("tr"));
    m.def("_flat", [](const py::object& f_hz) { return rx::to_py(em::flat(rx::as_double(f_hz))); },
          py::arg("f_hz"));

    // --- the template search's statistic ---
    m.def("_gauss_z", [](const py::object& log_p) { return em::gauss_z(rx::as_double(log_p)); }, py::arg("log_p"));
    m.def("template_z", [](const py::object& lam, const py::object& n_chunks, const py::object& n_cells) {
        return em::template_z(rx::as_double(lam), rx::as_int(n_chunks), rx::as_double(n_cells));
    }, py::arg("lam"), py::arg("n_chunks"), py::arg("n_cells"));
}

const Registrar reg("em", 98, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
