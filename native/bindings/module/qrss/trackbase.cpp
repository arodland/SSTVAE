// sstvae_native.qrss.trackbase: bindings of native/core/qrss/trackbase (the
// tracker's shared vocabulary from sstvae/qrss/track.py and ltrack.py).
//
//   q_of_spread(spread_hz, t_sym=T_SYM), spread_of_q(q, t_sym=T_SYM),
//   q_grid(t_sym=T_SYM) -> tuple, Q_GRID, Q_SPREADS_HZ, KNOTS_S, MF_RES
//   rx_index(timing, tau, n_pos, fs=CH_FS, t_sym=T_SYM)     track.rx_index
//   _tau_of_rx(timing, rel, n_pos, t_sym=T_SYM)             track._tau_of_rx
//   PathFn(t, f, w=None, knots_s=KNOTS_S)                   track.PathFn (the
//       native object: __call__, lo, hi, line, spline -> (t, c, k) or None,
//       from_parts(lo, hi, line, t=None, c=None, k=3), picklable)
//   freq_path_spline(path, knots_s=KNOTS_S) -> PathFn       track.freq_path_spline
//   chan_t_s(chan, n), chan_mix_audio_hz(chan)              ChanCapture's methods
//   mf_table_check(t_sym, sps, half, tab, g0) -> dict       MfTable(...) and its ValueError
//   track_result_info(tr, p=None) -> dict                   p_lo, gi(p), doppler_hz, t_pos_s(p)
//   roundtrip(obj, kind=None) -> dict                       Python -> C++ -> Python
//
// Array arguments are anything np.asarray(x, float64) accepts, of any
// shape; array results keep the argument's shape, and a 0-d argument gives
// a numpy float64 scalar, as the reference's arithmetic does.

#include "qrss/rx_types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sstvae_bindings::qrss {
namespace {

namespace tr = sstvae::qrss::track;
namespace cfg = sstvae::qrss::config;
namespace rxc = sstvae::qrss::rxconfig;

// np.asarray(x, float64): its values (flat) and shape.
struct Doubles {
    std::vector<double> v;
    py::tuple shape;
};

Doubles doubles(const py::handle& o) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o, py::arg("dtype") = "float64"));
    if (!a) throw py::error_already_set();
    return {from_numpy<double>(a), py::tuple(a.attr("shape"))};
}

// v in `shape`; a 0-d shape gives the float64 scalar.
py::object shaped(const std::vector<double>& v, const py::tuple& shape) {
    py::object a = to_numpy(v).attr("reshape")(shape);
    if (shape.empty()) return a[py::tuple()];
    return a;
}

template <typename T>
py::tuple tuple_of(const T& xs) {
    py::tuple out(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) out[i] = py::float_(xs[i]);
    return out;
}

// What the timing maps read of a Timing: tau0, ppm and gamma (not z or cov,
// so a stand-in with only those attributes works as it does in Python).
sstvae::qrss::types::Timing map_timing(const py::handle& o) {
    sstvae::qrss::types::Timing t;
    t.tau0 = rx::as_double(rx::field(o, "tau0"));
    t.ppm = rx::as_double(rx::field(o, "ppm"));
    t.gamma = rx::as_double(rx::field(o, "gamma"));
    return t;
}

tr::PathFn make_path(const py::object& t, const py::object& f, const py::object& w, double knots_s) {
    const std::vector<double> tv = rx::vec<double>(t);
    const std::vector<double> fv = rx::vec<double>(f);
    if (w.is_none()) return tr::PathFn(tv, fv, std::nullopt, knots_s);
    const std::vector<double> wv = rx::vec<double>(w);
    return tr::PathFn(tv, fv, std::span<const double>(wv), knots_s);
}

tr::PathFn path_from_parts(double lo, double hi, const py::object& line, const py::object& t, const py::object& c,
                           int k) {
    std::optional<sstvae::math::BSpline> s;
    if (!t.is_none()) {
        if (c.is_none()) throw py::value_error("PathFn.from_parts: a spline needs both t and c");
        s = sstvae::math::BSpline{rx::vec<double>(t), rx::vec<double>(c), k};
    }
    return tr::PathFn::from_parts(lo, hi, rx::vec<double>(line), std::move(s));
}

py::object spline_of(const tr::PathFn& p) {
    if (!p.spline()) return py::none();
    return py::make_tuple(to_numpy(p.spline()->t), to_numpy(p.spline()->c), p.spline()->k);
}

py::tuple state_of(const tr::PathFn& p) {
    const auto& s = p.spline();
    return py::make_tuple(p.lo(), p.hi(), to_numpy(p.line()), s ? py::object(to_numpy(s->t)) : py::object(py::none()),
                          s ? py::object(to_numpy(s->c)) : py::object(py::none()), s ? s->k : 3);
}

py::object roundtrip(const py::object& obj, const py::object& kind_o) {
    const std::string kind = kind_o.is_none() ? rx::type_name(obj) : py::str(kind_o).cast<std::string>();
    if (kind == "ChanCapture") return rx::to_py(rx::chan_capture(obj));
    if (kind == "MfTable") return rx::to_py(rx::mf_table(obj));
    if (kind == "Known" || kind == "_Known") return rx::to_py(rx::known(obj));
    if (kind == "Classes") return rx::to_py(rx::classes(obj));
    if (kind == "LClasses") return rx::to_py(rx::lclasses(obj));
    if (kind == "TrackResult" || kind == "LTrackResult") return rx::to_py(*rx::any_track_result(obj));
    if (kind == "PathFn" || kind == "NativePathFn") return rx::to_py(rx::path_fn(obj));
    throw py::value_error("trackbase.roundtrip: no conversion for " + kind);
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("trackbase",
                                    "Port of the tracker's shared records and maps (native/core/qrss/trackbase)");

    m.def("q_of_spread", [](double spread_hz, double t_sym) { return tr::q_of_spread(spread_hz, t_sym); },
          py::arg("spread_hz"), py::arg("t_sym") = cfg::T_SYM);
    m.def("spread_of_q", [](double q_, double t_sym) { return tr::spread_of_q(q_, t_sym); }, py::arg("q"),
          py::arg("t_sym") = cfg::T_SYM);
    m.def("q_grid", [](double t_sym) { return tuple_of(tr::q_grid(t_sym)); }, py::arg("t_sym") = cfg::T_SYM);
    m.attr("Q_GRID") = tuple_of(tr::Q_GRID);
    m.attr("Q_SPREADS_HZ") = tuple_of(rxc::track::Q_SPREADS_HZ);
    m.attr("KNOTS_S") = rxc::track::KNOTS_S;
    m.attr("MF_RES") = rxc::track::MF_RES;

    m.def("rx_index", [](const py::object& timing, const py::object& tau, const py::object& n_pos, int fs,
                         double t_sym) {
        const auto tm = map_timing(timing);
        const Doubles d = doubles(tau);
        return shaped(tr::rx_index(tm, d.v, rx::as_int(n_pos), fs, t_sym), d.shape);
    }, py::arg("timing"), py::arg("tau"), py::arg("n_pos"), py::arg("fs") = cfg::CH_FS, py::arg("t_sym") = cfg::T_SYM);
    m.def("_tau_of_rx", [](const py::object& timing, const py::object& rel, const py::object& n_pos, double t_sym) {
        const auto tm = map_timing(timing);
        const Doubles d = doubles(rel);
        return shaped(tr::tau_of_rx(tm, d.v, rx::as_int(n_pos), t_sym), d.shape);
    }, py::arg("timing"), py::arg("rel"), py::arg("n_pos"), py::arg("t_sym") = cfg::T_SYM);

    py::class_<tr::PathFn>(m, "PathFn", "track.PathFn: a frequency trajectory f(t), audio Hz at receiver time t")
        .def(py::init(&make_path), py::arg("t"), py::arg("f"), py::arg("w") = py::none(),
             py::arg("knots_s") = rxc::track::KNOTS_S)
        .def("__call__", [](const tr::PathFn& p, const py::object& t) {
            const Doubles d = doubles(t);
            return shaped(p(std::span<const double>(d.v)), d.shape);
        }, py::arg("t"))
        .def_property_readonly("lo", &tr::PathFn::lo)
        .def_property_readonly("hi", &tr::PathFn::hi)
        .def_property_readonly("line", [](const tr::PathFn& p) { return to_numpy(p.line()); })
        .def_property_readonly("spline", &spline_of, "(t, c, k) of the cubic spline, or None: the line")
        .def_static("from_parts", &path_from_parts, py::arg("lo"), py::arg("hi"), py::arg("line"),
                    py::arg("t") = py::none(), py::arg("c") = py::none(), py::arg("k") = 3)
        .def("__copy__", [](const tr::PathFn& p) { return tr::PathFn(p); })
        .def("__deepcopy__", [](const tr::PathFn& p, const py::object&) { return tr::PathFn(p); }, py::arg("memo"))
        .def(py::pickle(&state_of, [](const py::tuple& s) {
            if (s.size() != 6) throw py::value_error("PathFn: bad pickle state");
            return path_from_parts(s[0].cast<double>(), s[1].cast<double>(), s[2], s[3], s[4], s[5].cast<int>());
        }));

    m.def("freq_path_spline", [](const py::object& path, double knots_s) {
        return tr::freq_path_spline(rx::freq_path(path), knots_s);
    }, py::arg("path"), py::arg("knots_s") = rxc::track::KNOTS_S);

    m.def("chan_t_s", [](const py::object& chan, const py::object& n) {
        const auto c = rx::chan_capture(chan);
        const Doubles d = doubles(n);
        return shaped(c.t_s(std::span<const double>(d.v)), d.shape);
    }, py::arg("chan"), py::arg("n"));
    m.def("chan_mix_audio_hz", [](const py::object& chan) { return rx::chan_capture(chan).mix_audio_hz(); },
          py::arg("chan"));

    m.def("mf_table_check", [](const py::object& t_sym, const py::object& sps, const py::object& half,
                               const py::object& tab, const py::object& g0) {
        py::dict d;
        d["t_sym"] = t_sym;
        d["sps"] = sps;
        d["half"] = half;
        d["tab"] = tab;
        d["g0"] = g0;
        return rx::to_py(rx::mf_table(d));
    }, py::arg("t_sym"), py::arg("sps"), py::arg("half"), py::arg("tab"), py::arg("g0"));

    m.def("track_result_info", [](const py::object& obj, const py::object& p) {
        const auto r = rx::any_track_result(obj);
        py::dict d;
        d["p_lo"] = r->p_lo();
        d["doppler_hz"] = r->doppler_hz();
        if (p.is_none()) {
            d["gi"] = to_numpy(r->gi(std::span<const std::int64_t>(r->pos)));
            d["t_pos_s"] = to_numpy(r->t_pos_s());
        } else {
            const std::vector<std::int64_t> pi = rx::vec<std::int64_t>(p);
            const std::vector<double> pd = rx::vec<double>(p);
            d["gi"] = to_numpy(r->gi(std::span<const std::int64_t>(pi)));
            d["t_pos_s"] = to_numpy(r->t_pos_s(std::span<const double>(pd)));
        }
        d["classes"] = r->ce_classes() ? "Classes" : r->l_classes() ? "LClasses" : "None";
        d["is_l"] = dynamic_cast<const tr::LTrackResult*>(r.get()) != nullptr;
        return d;
    }, py::arg("tr"), py::arg("p") = py::none());

    m.def("roundtrip", &roundtrip, py::arg("obj"), py::arg("kind") = py::none());
}

const Registrar reg("trackbase", 66, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
