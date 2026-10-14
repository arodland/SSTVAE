// sstvae_native.qrss.rx_api: the in-process receive API
// (native/core/qrss/rx_api.hpp), the entry point an application calls, for
// the statistical acceptance test (tests/test_qrss_native_acceptance.py).
// Not a port of a Python function, so there is no substitution row: what
// it is compared with is the Python receiver's committed reference
// summaries (tools/gen_qrss_rx_reference.py).
//
//   receive_slot(cap, frames=("full",), live_only=False, estimator="joint",
//                erase_s=(), round_b=True, max_candidates=None,
//                budget_s=None, hints=(), old_edge_noise=False,
//                old_gating=False) -> [PassResult]
//
// `cap` is a frontend.Capture (or anything with fe, q, t0_index and fs);
// `frames` a frame name, a FrameSpec (native or the reference's), or a
// sequence of them, at most one per waveform; `hints` frequencies or
// (f, waveform) pairs. The passes come back as "__type__" dicts
// (tests/qrss_native/_rx.py's to_ref makes them the reference's).
//
// Unlike receiver_slot's binding, which the rows hand Python objects and
// which runs over them when an argument does not convert, this one has no
// second way: the arguments become api::ReceiveConfig, or the call fails,
// and the receive is the C++ from the capture to the passes with the GIL
// released throughout. The two QRSS_OLD_* rules are arguments, never read
// from the environment (rx_api.hpp).

#include "qrss/rx_types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "qrss/acquire.hpp"
#include "qrss/frame.hpp"
#include "qrss/rx_api.hpp"
#include "qrss/types.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace api = sstvae::qrss::api;
namespace fr = sstvae::qrss::frame;
namespace aq = sstvae::qrss::acquire;

bool is_seq(const py::handle& o) { return py::isinstance<py::tuple>(o) || py::isinstance<py::list>(o); }

fr::FrameSpec frame_of(const py::handle& o) {
    if (py::isinstance<py::str>(o)) return fr::get_any(rx::as_string(o));
    return rx::spec(o);
}

std::vector<fr::FrameSpec> frames_of(const py::object& o) {
    std::vector<fr::FrameSpec> out;
    if (!is_seq(o)) {
        out.push_back(frame_of(o));
        return out;
    }
    for (const py::handle h : o) out.push_back(frame_of(h));
    return out;
}

std::vector<std::pair<double, double>> spans_of(const py::object& o) {
    std::vector<std::pair<double, double>> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(o))) {
        const py::sequence ab = py::reinterpret_borrow<py::sequence>(h);
        if (ab.size() != 2) throw py::value_error("an erase span is (a, b)");
        out.emplace_back(rx::as_double(ab[0]), rx::as_double(ab[1]));
    }
    return out;
}

std::vector<aq::Hint> hints_of(const py::object& o) {
    std::vector<aq::Hint> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(o))) {
        aq::Hint x;
        if (is_seq(h)) {
            const py::sequence fw = py::reinterpret_borrow<py::sequence>(h);
            if (fw.size() != 2) throw py::value_error("a hint is f or (f, waveform)");
            x.f_hz = rx::as_double(fw[0]);
            x.waveform = rx::as_int(fw[1]);
        } else {
            x.f_hz = rx::as_double(h);
        }
        out.push_back(x);
    }
    return out;
}

py::list receive_slot_py(const py::object& cap, const py::object& frames, bool live_only,
                         const std::string& estimator, const py::object& erase_s, bool round_b,
                         const py::object& max_candidates, const py::object& budget_s, const py::object& hints,
                         bool old_edge_noise, bool old_gating) {
    const api::Capture c = rx::capture(cap);
    api::ReceiveConfig cfg;
    cfg.frames = frames_of(frames);
    cfg.live_only = live_only;
    cfg.estimator = estimator;
    cfg.erase_s = spans_of(erase_s);
    cfg.round_b = round_b;
    if (!max_candidates.is_none()) cfg.max_candidates = rx::as_int(max_candidates);
    if (!budget_s.is_none()) cfg.budget_s = rx::as_double(budget_s);
    cfg.hints = hints_of(hints);
    cfg.old_edge_noise = old_edge_noise;
    cfg.old_gating = old_gating;
    std::vector<sstvae::qrss::types::PassResult> passes;
    {
        py::gil_scoped_release nogil;
        passes = api::receive_slot(c, cfg);
    }
    py::list out;
    for (const auto& p : passes) out.append(rx::to_py(p));
    return out;
}

void bind(py::module_& qrss) {
    rx::register_translators();
    py::module_ m = qrss.def_submodule("rx_api", "native/core/qrss/rx_api (the in-process receive API)");
    m.def("receive_slot", &receive_slot_py, py::arg("cap"), py::arg("frames") = py::make_tuple("full"),
          py::arg("live_only") = false, py::arg("estimator") = "joint", py::arg("erase_s") = py::tuple(),
          py::arg("round_b") = true, py::arg("max_candidates") = py::none(), py::arg("budget_s") = py::none(),
          py::arg("hints") = py::tuple(), py::arg("old_edge_noise") = false, py::arg("old_gating") = false);
}

const Registrar reg("rx_api", 91, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
