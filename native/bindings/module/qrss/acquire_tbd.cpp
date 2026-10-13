// sstvae_native.qrss.acquire_tbd: native/core/qrss/acquire_tbd, A3 of
// sstvae/qrss/acquire.py, under the reference's names and signatures
// (tests/qrss_native/acquire_tbd.py assigns them over acquire's
// attributes):
//
//   tbd_frame_starts(spec, n_fe=None, t0_index=0.0, fs=FE_FS)
//   tbd_n_frames(spec)
//   tbd_viterbi(E, max_step=TBD_MAX_STEP, cost=TBD_COST) -> (path, Z)
//   _path_line(path)
//   tbd_channels(cands=(), band_hz=BAND_HZ)
//   tbd_emissions(fe, t0_index, spec, fs=FE_FS, *, band_hz=BAND_HZ) -> (E, f, t)
//   shape_frames(t_s, specs, frame_s=STFT_FRAME_S)
//   shape_z(E, bins, keep=None, df_hz=1/STFT_FRAME_S, fs=FE_FS)
//   shape_waveform(z1, z_above), _shape_verdict(v, w)
//   track_before_detect(fe, t0_index, spec, cands=(), fs=FE_FS, *, pfa=TBD_PFA,
//                       band_hz=BAND_HZ, all_hits=False, shape_specs=None,
//                       _shape_frames=None) -> [TbdHit as a dict]
//   _bins_path(E, f, t, bins) -> FreqPath as a dict
//   l_pilot_bins(E, f, t, line, keep=None, fs=FE_FS, span_hz=L_ANCHOR_HZ)
//   frame_map(specs) -> {waveform: the caller's spec}
//   _path_of(f_hz, drift_hz_per_min, dur) -> FreqPath as a dict
//   TbdHit's methods, over any object with its fields:
//   _hit_f_hz(hit), _hit_line(hit), _hit_passed(hit), _hit_to_detection(hit, waveform=None)
//   _np_log_f32(x)   numpy's float32 log (math/nplog.hpp), for the parity test
//
// Records come back as rx_types.hpp's "__type__" dicts ("TbdHit" with a
// FreqPath "path" and an optional "l_path"); the rows turn them into the
// reference's classes. `_shape_frames`: the function track_before_detect
// finds the shape-test frames with (the reference calls its module
// attribute, which a test may replace); None is the native shape_frames.
//
// Emissions are float32 or float64 (other dtypes NotImplementedError,
// except tbd_viterbi, which widens anything to float64 as the reference
// does). The capture is used in place when it is a contiguous complex64,
// complex128 or float64 array; the GIL is released while A3 runs.

#include "qrss/rx_types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "math/nplog.hpp"
#include "qrss/acquire_tbd.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace aq = sstvae::qrss::acquire;
namespace rxa = sstvae::qrss::rxconfig::acquire;
namespace cfg = sstvae::qrss::config;

py::tuple shape_tuple(const std::vector<py::ssize_t>& s) {
    py::tuple t(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) t[i] = py::int_(s[i]);
    return t;
}

[[noreturn]] void not_implemented(const std::string& what) {
    PyErr_SetString(PyExc_NotImplementedError, ("native acquire_tbd: " + what).c_str());
    throw py::error_already_set();
}

void register_errors() {
    static const bool done = [] {
        py::register_local_exception_translator([](std::exception_ptr p) {
            try {
                if (p) std::rethrow_exception(p);
            } catch (const aq::UnboundLocal& e) {
                PyErr_SetString(PyExc_UnboundLocalError, e.what());
            } catch (const aq::NotPorted& e) {
                PyErr_SetString(PyExc_NotImplementedError, (std::string("native acquire_tbd: ") + e.what()).c_str());
            }
        });
        return true;
    }();
    (void)done;
}

py::array as_array(const py::handle& o) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o));
    if (!a) throw py::error_already_set();
    return a;
}

std::string dtype_of(const py::array& a) { return py::str(a.attr("dtype")).cast<std::string>(); }

std::array<double, 2> pair_of(const py::handle& o, const char* what) {
    const std::vector<double> b = rx::vec<double>(o);
    if (b.size() != 2) throw py::value_error(std::string(what) + " must be (lo, hi)");
    return {b[0], b[1]};
}

py::tuple default_band() { return py::make_tuple(rxa::BAND_HZ[0], rxa::BAND_HZ[1]); }

// A 2-D float32 or float64 E, as the reference keeps it.
struct AnyGrid {
    std::optional<aq::Grid<float>> f32;
    std::optional<aq::Grid<double>> f64;
};

template <typename T>
aq::Grid<T> grid_from(const py::array& a) {
    aq::Grid<T> g;
    g.rows = static_cast<std::size_t>(a.shape(0));
    g.cols = static_cast<std::size_t>(a.shape(1));
    g.data = rx::vec<T>(a);
    return g;
}

AnyGrid emissions_arg(const py::handle& E) {
    py::array a = as_array(E);
    if (a.ndim() != 2) not_implemented("E must be 2-D");
    const std::string dt = dtype_of(a);
    AnyGrid g;
    if (dt == "float32") g.f32 = grid_from<float>(a);
    else if (dt == "float64") g.f64 = grid_from<double>(a);
    else not_implemented("E of dtype " + dt + " (float32 or float64 only)");
    return g;
}

template <typename F>
auto on_grid(const AnyGrid& g, F&& f) {
    return g.f32 ? f(*g.f32) : f(*g.f64);
}

std::vector<std::int64_t> bins_arg(const py::handle& o) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o, py::arg("dtype") = "int64"));
    if (!a) throw py::error_already_set();
    if (a.ndim() != 1) not_implemented("bins must be 1-D");
    return rx::vec<std::int64_t>(a);
}

std::optional<std::vector<std::uint8_t>> keep_arg(const py::handle& o) {
    if (o.is_none()) return std::nullopt;
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o, py::arg("dtype") = "bool"));
    if (!a) throw py::error_already_set();
    if (a.ndim() != 1) not_implemented("keep must be 1-D");
    return rx::vec_bool(a);
}

std::optional<std::span<const std::uint8_t>> span_of(const std::optional<std::vector<std::uint8_t>>& v) {
    if (!v) return std::nullopt;
    return std::span<const std::uint8_t>(*v);
}

std::vector<double> f64_1d(const py::handle& o, const char* what) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o, py::arg("dtype") = "float64"));
    if (!a) throw py::error_already_set();
    if (a.ndim() != 1) not_implemented(std::string(what) + " must be 1-D");
    return rx::vec<double>(a);
}

std::vector<rx::frame::FrameSpec> specs_of(const py::handle& o) {
    std::vector<rx::frame::FrameSpec> out;
    for (const py::handle s : py::reinterpret_borrow<py::iterable>(o)) out.push_back(rx::spec(s));
    return out;
}

std::vector<double> cand_freqs(const py::handle& cands) {
    std::vector<double> out;
    for (const py::handle c : py::reinterpret_borrow<py::iterable>(cands)) out.push_back(rx::as_double(c.attr("f_hz")));
    return out;
}

// TbdHit's line and f_hz read the path alone (to_detection reads the whole
// hit, rx_types.hpp's tbd_hit).
aq::TbdHit path_hit(const py::handle& o) {
    aq::TbdHit h;
    h.path = rx::freq_path(rx::field(o, "path"));
    return h;
}

// --- tbd_viterbi --------------------------------------------------------------

py::tuple tbd_viterbi(const py::object& E_o, int max_step, double cost) {
    py::array a = as_array(E_o);
    std::string dt = dtype_of(a);
    if (dt != "float32" && dt != "float64") {
        a = py::array::ensure(rx::np_module().attr("asarray")(E_o, py::arg("dtype") = "float64"));
        if (!a) throw py::error_already_set();
        dt = "float64";
    }
    if (a.ndim() < 2) not_implemented("E must be at least 2-D");
    const int nd = static_cast<int>(a.ndim());
    const auto T = static_cast<std::size_t>(a.shape(nd - 2));
    const auto B = static_cast<std::size_t>(a.shape(nd - 1));
    std::size_t lead = 1;
    std::vector<py::ssize_t> lead_shape;
    for (int i = 0; i < nd - 2; ++i) {
        lead *= static_cast<std::size_t>(a.shape(i));
        lead_shape.push_back(a.shape(i));
    }
    if (T == 0)
        throw py::index_error("index 0 is out of bounds for axis " + std::to_string(nd - 2) + " with size 0");
    std::vector<std::int64_t> paths;
    std::vector<double> zs;
    auto run = [&](auto tag) {
        using Tp = decltype(tag);
        const std::vector<Tp> v = rx::vec<Tp>(a);
        for (std::size_t k = 0; k < lead; ++k) {
            const aq::TbdPath p = aq::tbd_viterbi<Tp>(v.data() + k * T * B, T, B, 0, B, max_step, cost);
            paths.insert(paths.end(), p.path.begin(), p.path.end());
            zs.push_back(p.z);
        }
        if (lead == 0) {
            // the reference's argmax of an empty sequence, or nothing at all
            if (B == 0) throw py::value_error("attempt to get argmax of an empty sequence");
        }
    };
    if (dt == "float32") run(float{});
    else run(double{});
    std::vector<py::ssize_t> pshape = lead_shape;
    pshape.push_back(static_cast<py::ssize_t>(T));
    py::array path = to_numpy(paths).attr("reshape")(shape_tuple(pshape));
    py::object z;
    if (nd == 2) z = rx::np_module().attr("float64")(zs[0]);
    else z = to_numpy(zs).attr("reshape")(shape_tuple(lead_shape));
    return py::make_tuple(path, z);
}

// --- the capture --------------------------------------------------------------

// The capture as the core takes it, viewed in place where its layout allows.
struct Capture {
    py::array arr;   // keeps the data alive
    enum Kind { C64, C128, F64 } kind;
};

Capture capture_of(const py::object& fe) {
    py::module_ np = rx::np_module();
    py::array a = as_array(fe);
    if (a.ndim() != 1) not_implemented("fe must be 1-D");
    const std::string dt = dtype_of(a);
    Capture c;
    if (dt == "complex64") c.kind = Capture::C64;
    else if (dt == "complex128") c.kind = Capture::C128;
    else if (a.dtype().kind() == 'c') {
        c.kind = Capture::C128;
        a = py::array::ensure(np.attr("asarray")(a, py::arg("dtype") = "complex128"));
    } else {
        c.kind = Capture::F64;
        if (dt != "float64") a = py::array::ensure(np.attr("asarray")(a, py::arg("dtype") = "float64"));
    }
    if (!a) throw py::error_already_set();
    c.arr = py::array::ensure(np.attr("ascontiguousarray")(a));
    if (!c.arr) throw py::error_already_set();
    return c;
}

template <typename F>
auto on_capture(const Capture& c, F&& f) {
    const auto n = static_cast<std::size_t>(c.arr.size());
    switch (c.kind) {
    case Capture::C64:
        return f(std::span<const std::complex<float>>(static_cast<const std::complex<float>*>(c.arr.data()), n));
    case Capture::C128:
        return f(std::span<const std::complex<double>>(static_cast<const std::complex<double>*>(c.arr.data()), n));
    default:
        return f(std::span<const double>(static_cast<const double*>(c.arr.data()), n));
    }
}

py::tuple emissions_to_py(const aq::Emissions& em) {
    py::array_t<float> E(std::vector<py::ssize_t>{static_cast<py::ssize_t>(em.E.rows),
                                                  static_cast<py::ssize_t>(em.E.cols)});
    std::copy(em.E.data.begin(), em.E.data.end(), E.mutable_data());
    return py::make_tuple(E, to_numpy(em.f), to_numpy(em.t));
}

py::tuple tbd_emissions(const py::object& fe, double t0_index, const py::object& spec_o, double fs,
                        const py::object& band_o) {
    const rx::frame::FrameSpec spec = rx::spec(spec_o);
    const std::array<double, 2> band = pair_of(band_o, "band_hz");
    const Capture cap = capture_of(fe);
    aq::Emissions em;
    {
        py::gil_scoped_release nogil;
        em = on_capture(cap, [&](auto x) { return aq::tbd_emissions(x, t0_index, spec, fs, band); });
    }
    return emissions_to_py(em);
}

py::list track_before_detect(const py::object& fe, double t0_index, const py::object& spec_o, const py::object& cands,
                             double fs, double pfa, const py::object& band_o, bool all_hits,
                             const py::object& shape_specs, const py::object& shape_frames_fn) {
    const rx::frame::FrameSpec spec = rx::spec(spec_o);
    aq::TbdOptions opt;
    opt.pfa = pfa;
    opt.band_hz = pair_of(band_o, "band_hz");
    opt.all_hits = all_hits;
    py::tuple specs_tuple;
    if (!shape_specs.is_none()) {
        specs_tuple = py::tuple(shape_specs);
        opt.shape_specs = specs_of(specs_tuple);
        if (!shape_frames_fn.is_none()) {
            opt.keep_fn = [&](std::span<const double> t) {
                py::gil_scoped_acquire gil;
                py::object k = shape_frames_fn(to_numpy(std::vector<double>(t.begin(), t.end())), specs_tuple);
                return rx::vec_bool(k);
            };
        }
    }
    const std::vector<double> cf = cand_freqs(cands);
    const Capture cap = capture_of(fe);
    std::vector<aq::TbdHit> hits;
    {
        py::gil_scoped_release nogil;
        hits = on_capture(cap, [&](auto x) -> std::vector<aq::TbdHit> {
            if constexpr (std::is_same_v<decltype(x), std::span<const double>>) {
                return aq::track_before_detect(aq::tbd_emissions(x, t0_index, spec, fs, opt.band_hz), cf, fs, opt);
            } else {
                return aq::track_before_detect(x, t0_index, spec, cf, fs, opt);
            }
        });
    }
    py::list out;
    for (const aq::TbdHit& h : hits) out.append(rx::to_py(h));
    return out;
}

void bind(py::module_& q) {
    rx::register_translators();
    register_errors();
    py::module_ m = q.def_submodule("acquire_tbd", "acquire.py's A3 (native/core/qrss/acquire_tbd)");
    const double FS = static_cast<double>(cfg::FE_FS);

    m.def("tbd_frame_starts", [](const py::object& spec, const py::object& n_fe, double t0_index, double fs) {
        std::optional<std::int64_t> n;
        if (!n_fe.is_none()) n = rx::as_int(n_fe);
        return to_numpy(aq::tbd_frame_starts(rx::spec(spec), n, t0_index, fs));
    }, py::arg("spec"), py::arg("n_fe") = py::none(), py::arg("t0_index") = 0.0, py::arg("fs") = FS);
    m.def("tbd_n_frames", [](const py::object& spec) { return aq::tbd_n_frames(rx::spec(spec)); }, py::arg("spec"));
    m.def("tbd_viterbi", &tbd_viterbi, py::arg("E"), py::arg("max_step") = rxa::TBD_MAX_STEP,
          py::arg("cost") = rxa::TBD_COST);
    m.def("_path_line", [](const py::object& path) { return to_numpy(aq::path_line(rx::freq_path(path))); },
          py::arg("path"));
    m.def("tbd_channels", [](const py::object& cands, const py::object& band) {
        return to_numpy(aq::tbd_channels(cand_freqs(cands), pair_of(band, "band_hz")));
    }, py::arg("cands") = py::tuple(), py::arg("band_hz") = default_band());
    m.def("tbd_emissions", &tbd_emissions, py::arg("fe"), py::arg("t0_index"), py::arg("spec"), py::arg("fs") = FS,
          py::kw_only(), py::arg("band_hz") = default_band());
    m.def("shape_frames", [](const py::object& t_s, const py::object& specs, double frame_s) {
        return rx::arr_bool(aq::shape_frames(f64_1d(t_s, "t_s"), specs_of(specs), frame_s));
    }, py::arg("t_s"), py::arg("specs"), py::arg("frame_s") = rxa::STFT_FRAME_S);
    m.def("shape_z", [](const py::object& E, const py::object& bins, const py::object& keep, double df_hz, double fs) {
        const AnyGrid g = emissions_arg(E);
        const std::vector<std::int64_t> b = bins_arg(bins);
        const auto k = keep_arg(keep);
        const auto r = on_grid(g, [&](const auto& G) { return aq::shape_z(G, b, span_of(k), df_hz, fs); });
        return py::make_tuple(r.first, r.second);
    }, py::arg("E"), py::arg("bins"), py::arg("keep") = py::none(), py::arg("df_hz") = 1.0 / rxa::STFT_FRAME_S,
       py::arg("fs") = FS);
    m.def("shape_waveform", [](double z1, double za) { return rx::opt_int_py(aq::shape_waveform(z1, za)); },
          py::arg("z1"), py::arg("z_above"));
    m.def("_shape_verdict", [](const py::object& v, const py::object& w) {
        return rx::opt_int_py(aq::shape_verdict(rx::opt_int(v), rx::opt_int(w)));
    }, py::arg("v"), py::arg("w"));
    m.def("track_before_detect", &track_before_detect, py::arg("fe"), py::arg("t0_index"), py::arg("spec"),
          py::arg("cands") = py::tuple(), py::arg("fs") = FS, py::kw_only(), py::arg("pfa") = rxa::TBD_PFA,
          py::arg("band_hz") = default_band(), py::arg("all_hits") = false, py::arg("shape_specs") = py::none(),
          py::arg("_shape_frames") = py::none());
    m.def("_bins_path", [](const py::object& E, const py::object& f, const py::object& t, const py::object& bins) {
        const AnyGrid g = emissions_arg(E);
        const std::vector<double> fv = f64_1d(f, "f"), tv = f64_1d(t, "t");
        const std::vector<std::int64_t> b = bins_arg(bins);
        return rx::to_py(on_grid(g, [&](const auto& G) { return aq::bins_path(G, fv, tv, b); }));
    }, py::arg("E"), py::arg("f"), py::arg("t"), py::arg("bins"));
    m.def("l_pilot_bins", [](const py::object& E, const py::object& f, const py::object& t, const py::object& line,
                             const py::object& keep, double fs, const py::object& span) {
        const AnyGrid g = emissions_arg(E);
        const std::vector<double> fv = f64_1d(f, "f"), tv = f64_1d(t, "t"), lv = f64_1d(line, "line");
        const auto k = keep_arg(keep);
        const std::array<double, 2> sp = pair_of(span, "span_hz");
        return to_numpy(on_grid(g, [&](const auto& G) { return aq::l_pilot_bins(G, fv, tv, lv, span_of(k), fs, sp); }));
    }, py::arg("E"), py::arg("f"), py::arg("t"), py::arg("line"), py::arg("keep") = py::none(), py::arg("fs") = FS,
       py::arg("span_hz") = py::make_tuple(rxa::L_ANCHOR_HZ[0], rxa::L_ANCHOR_HZ[1]));
    m.def("frame_map", [](const py::object& specs) {
        std::vector<py::object> objs;
        for (const py::handle s : py::reinterpret_borrow<py::iterable>(specs)) objs.push_back(py::reinterpret_borrow<py::object>(s));
        std::vector<rx::frame::FrameSpec> fs;
        for (const py::object& s : objs) fs.push_back(rx::spec(s));
        py::dict out;
        for (const auto& [w, i] : aq::frame_map(fs)) out[py::int_(w)] = objs[i];
        return out;
    }, py::arg("specs"));
    m.def("_path_of", [](double f_hz, double drift, double dur) { return rx::to_py(aq::path_of(f_hz, drift, dur)); },
          py::arg("f_hz"), py::arg("drift_hz_per_min"), py::arg("dur"));

    m.def("_hit_line", [](const py::object& hit) { return to_numpy(path_hit(hit).line()); }, py::arg("self"));
    m.def("_hit_f_hz", [](const py::object& hit) { return path_hit(hit).f_hz(); }, py::arg("self"));
    m.def("_hit_passed", [](const py::object& hit) {
        return rx::as_double(rx::field(hit, "p_value")) < rxa::TBD_PFA;
    }, py::arg("self"));
    m.def("_hit_to_detection", [](const py::object& hit, const py::object& waveform) {
        return rx::to_py(rx::tbd_hit(hit).to_detection(rx::opt_int(waveform)));
    }, py::arg("self"), py::arg("waveform") = py::none());

    m.def("_np_log_f32", [](const py::object& x) {
        std::vector<float> v = rx::vec<float>(x);
        for (float& e : v) e = sstvae::math::np::log_f32(e);
        return to_numpy(v);
    }, py::arg("x"));
}

const Registrar reg("acquire_tbd", 77, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
