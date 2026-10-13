// sstvae_native.qrss.acquire_a12: native/core/qrss/acquire_a12, A1 (carrier
// lines), A2 (preamble search) and lead_in_span of sstvae/qrss/acquire.py,
// under the reference's names and signatures. The rows
// (tests/qrss_native/acquire_a12.py) assign them over acquire's attributes
// through thin wrappers that turn the records handed back here into the
// reference's own dataclasses.
//
//   carrier_line_stats(fe, t0_index, fs=FE_FS, *, span_s, drifts, band_hz)
//                                        -> {"__type__": "CarrierStats", ...}
//   carrier_lines(fe, t0_index, fs=FE_FS, *, pfa, max_cands, min_sep_hz,
//                 band_hz, stats=None)   -> [CarrierCandidate]
//   hint_line(cs, f_hz, fs=FE_FS, *, pfa, span_hz), _cell_line(cs, b, d, v),
//   pilot_lines(cs, lines, *, sep_hz)
//   _notch_taps(), _mf_taps(fs=CH_FS), _l_mf_taps(fs=CH_FS)
//   PreambleKind (read-only), CE_PRE, L_PRE, PREAMBLE_KINDS
//   preamble_channel, _a2_input, _a2_mf -> {"__type__": "_A2Channel" | "_A2Input"}
//   _chunks, _norm, _templates, preamble_lambda, _preamble_sums,
//   _fine_sums, _coherent, _phase_line
//   preamble_search(fe, t0_index, cand, fs=FE_FS, *, pfa, kind=None, inp=None)
//                                        -> {"__type__": "PreambleHit", ...}
//   lead_in_span(fe, t0_index, f_hz, drift, tau0_s, fs, *, z_min, t_sym,
//                _noise_psd=None)
//   to_detection(hit, duration_s=1800.0) -> Detection
//   _log_f32(x): numpy's float32 log (math/nplog.hpp), for the parity test
//
// Arguments are read the way the reference reads them: a CarrierStats, an
// _A2Input, an _A2Channel, a PreambleKind and a PreambleHit may each be
// the reference's dataclass or the dict (or class) this module returns; a
// kind of None is CE's. _preamble_sums hands S and Lambda back
// Fortran-ordered, as the reference's einsum builds them, so numpy -- and
// the native preamble_p -- add them up in the same order the reference
// does. lead_in_span's `_noise_psd` is the frontend.noise_psd to call when
// it is not the native one (a test spying on its arguments).

#include "qrss/rx_types.hpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "math/cubic_spline.hpp"
#include "math/nplog.hpp"
#include "qrss/acquire_a12.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace aq = sstvae::qrss::acquire;
namespace rxa = sstvae::qrss::rxconfig::acquire;
namespace cfg = sstvae::qrss::config;
namespace sm = sstvae::math;
using cf = std::complex<float>;
using cd = std::complex<double>;

[[noreturn]] void not_implemented(const std::string& what) {
    PyErr_SetString(PyExc_NotImplementedError, ("native acquire_a12: " + what).c_str());
    throw py::error_already_set();
}

py::array as_array(const py::handle& o) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o));
    if (!a) throw py::error_already_set();
    return a;
}

std::string dtype_of(const py::array& a) { return py::str(a.attr("dtype")).cast<std::string>(); }

// fe as a view: complex64 stays complex64 (no copy when contiguous), any
// other dtype is read as complex128 (numpy's own cast when it builds `seg`).
struct Fe {
    py::array keep;
    std::span<const cf> f32;
    std::span<const cd> f64;
    bool is32 = false;
};

Fe fe_of(const py::handle& o) {
    py::array a = as_array(o);
    if (a.ndim() != 1) not_implemented("fe must be 1-D");
    Fe f;
    if (dtype_of(a) == "complex64") {
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

std::array<double, 2> pair_of(const py::handle& o) {
    const std::vector<double> v = rx::vec<double>(o);
    if (v.size() != 2) throw py::value_error("expected a pair (lo, hi)");
    return {v[0], v[1]};
}

py::array read_only(py::array a) {
    a.attr("setflags")(py::arg("write") = false);
    return a;
}

template <typename T>
py::array grid(const std::vector<T>& v, std::vector<py::ssize_t> shape, bool fortran = false) {
    py::array_t<T> out(shape);
    std::copy(v.begin(), v.end(), out.mutable_data());
    if (fortran) return py::array::ensure(rx::np_module().attr("asfortranarray")(out));
    return out;
}

// --- PreambleKind -----------------------------------------------------------------

aq::PreambleKind kind_of(const py::handle& o) {
    if (o.is_none()) return aq::ce_pre();
    if (py::isinstance<aq::PreambleKind>(o)) return o.cast<const aq::PreambleKind&>();
    aq::PreambleKind k;
    k.waveform = rx::as_int(rx::field(o, "waveform"));
    py::array seq = as_array(rx::field(o, "seq"));
    k.real = seq.dtype().kind() != 'c';
    k.seq = rx::vec<cd>(seq);
    k.t_sym = rx::as_double(rx::field(o, "t_sym"));
    k.chunk = rx::as_int(rx::field(o, "chunk"));
    k.data_offset_hz = rx::as_double(rx::field(o, "data_offset_hz"));
    k.phase_offset = rx::as_double(rx::field(o, "phase_offset"));
    k.mf_taps = rx::vec<double>(rx::field(o, "mf_taps"));
    return k;
}

py::array seq_array(const aq::PreambleKind& k) {
    if (k.real) {
        std::vector<double> re(k.seq.size());
        for (std::size_t i = 0; i < re.size(); ++i) re[i] = k.seq[i].real();
        return read_only(to_numpy(re));
    }
    return read_only(to_numpy(k.seq));
}

// --- records -------------------------------------------------------------------------

aq::CarrierStats carrier_stats(const py::handle& o) {
    aq::CarrierStats cs;
    py::array st = as_array(rx::field(o, "stat"));
    if (st.ndim() != 2) not_implemented("stat must be 2-D (n_drift, n_bin)");
    if (dtype_of(st) != "float32") not_implemented("stat must be float32, as carrier_line_stats makes it");
    cs.stat.rows = static_cast<std::size_t>(st.shape(0));
    cs.stat.cols = static_cast<std::size_t>(st.shape(1));
    cs.stat.data = rx::vec<float>(st);
    cs.f_hz = rx::vec<double>(rx::field(o, "f_hz"));
    cs.drifts = rx::vec<double>(rx::field(o, "drifts"));
    cs.M = rx::as_int(rx::field(o, "M"));
    return cs;
}

py::dict to_py(const aq::CarrierStats& cs) {
    py::dict d = rx::typed("CarrierStats");
    d["stat"] = grid(cs.stat.data, {static_cast<py::ssize_t>(cs.stat.rows), static_cast<py::ssize_t>(cs.stat.cols)});
    d["f_hz"] = to_numpy(cs.f_hz);
    d["drifts"] = to_numpy(cs.drifts);
    d["M"] = cs.M;
    return d;
}

// A line refined from an A1 cell: the reference's f_hz is float(f_hz[b]) +
// off * df, df an element difference of the f_hz array, so an np.float64.
py::object cell_to_py(const sstvae::qrss::types::CarrierCandidate& c) {
    py::object d = rx::to_py(c);
    d["f_hz"] = rx::np_module().attr("float64")(c.f_hz);
    return d;
}

py::list to_py(const std::vector<sstvae::qrss::types::CarrierCandidate>& v) {
    py::list out;
    for (const auto& c : v) out.append(cell_to_py(c));
    return out;
}

aq::A2Input a2_input_of(const py::handle& o) {
    aq::A2Input in;
    in.y = rx::vec<cd>(rx::field(o, "y"));
    in.t0_ch = rx::as_double(rx::field(o, "t0_ch"));
    const py::object fm = rx::field(o, "f_mix");
    if (!fm.is_none()) in.f_mix = rx::rational(fm);
    return in;
}

py::dict to_py(const aq::A2Input& in) {
    py::dict d = rx::typed("_A2Input");
    d["y"] = to_numpy(in.y);
    d["t0_ch"] = in.t0_ch;
    d["f_mix"] = rx::to_py(in.f_mix);
    return d;
}

aq::A2Channel chan_of(const py::handle& o) {
    aq::A2Channel c;
    c.m = rx::vec<cd>(rx::field(o, "m"));
    c.t0_ch = rx::as_double(rx::field(o, "t0_ch"));
    c.sigma2 = rx::as_double(rx::field(o, "sigma2"));
    const py::object fm = rx::field(o, "f_mix");
    if (!fm.is_none()) c.f_mix = rx::rational(fm);
    const py::object loc = rx::field(o, "local");
    if (!loc.is_none()) c.local = rx::vec<double>(loc);
    c.kind = kind_of(rx::field(o, "kind"));
    return c;
}

// The channel as a record; `kind` is the caller's own object (the
// reference's _A2Channel holds the PreambleKind it was made for).
py::dict to_py(const aq::A2Channel& c, const py::object& kind) {
    py::dict d = rx::typed("_A2Channel");
    d["m"] = to_numpy(c.m);
    d["t0_ch"] = c.t0_ch;
    d["sigma2"] = c.sigma2;
    d["f_mix"] = rx::to_py(c.f_mix);
    d["local"] = c.local ? py::object(to_numpy(*c.local)) : py::object(py::none());
    d["kind"] = kind;
    return d;
}

py::dict to_py(const aq::PreambleHit& h) {
    py::dict d = rx::typed("PreambleHit");
    d["f_hz"] = h.f_hz;
    d["drift_hz_per_min"] = h.drift_hz_per_min;
    d["tau0_s"] = h.tau0_s;
    d["phase"] = h.phase;
    d["lam"] = h.lam;
    d["cell_lam"] = h.cell_lam;
    d["p_value"] = h.p_value;
    d["passed"] = h.passed;
    d["n_groups"] = h.n_groups;
    d["f_mix"] = h.f_mix ? rx::to_py(*h.f_mix) : py::object(py::none());
    d["tau0_sd_s"] = h.tau0_sd_s;
    d["chunk_sums"] = h.chunk_sums ? py::object(to_numpy(*h.chunk_sums)) : py::object(py::none());
    d["lead_in_s"] = h.lead_in_s;
    d["waveform"] = h.waveform;
    return d;
}

aq::PreambleHit hit_of(const py::handle& o) {
    aq::PreambleHit h;
    h.f_hz = rx::as_double(rx::field(o, "f_hz"));
    h.drift_hz_per_min = rx::as_double(rx::field(o, "drift_hz_per_min"));
    h.tau0_s = rx::as_double(rx::field(o, "tau0_s"));
    h.lam = rx::as_double(rx::field(o, "lam"));
    h.tau0_sd_s = rx::as_double(rx::field(o, "tau0_sd_s"));
    h.lead_in_s = rx::as_double(rx::field(o, "lead_in_s"));
    h.waveform = rx::as_int(rx::field(o, "waveform"));
    return h;
}

// --- splines and callbacks ----------------------------------------------------------

// A spline argument: the native CubicSpline directly, anything else (scipy's)
// called back with a float64 array.
struct Spline {
    std::shared_ptr<sm::CubicSpline> native;
    aq::SplineFn fn;
};

Spline spline_of(const py::object& o) {
    Spline s;
    if (py::isinstance<sm::CubicSpline>(o)) {
        s.native = o.cast<std::shared_ptr<sm::CubicSpline>>();
        s.fn = aq::spline_fn(*s.native);
        return s;
    }
    s.fn = [o](std::span<const double> x) {
        py::gil_scoped_acquire gil;
        py::object r = o(to_numpy(std::vector<double>(x.begin(), x.end())));
        return rx::vec<cd>(r);
    };
    return s;
}

// --- functions --------------------------------------------------------------------------

py::dict carrier_line_stats(const py::object& fe, double t0_index, double fs, const py::object& span_s,
                            const py::object& drifts, const py::object& band_hz) {
    const Fe f = fe_of(fe);
    const std::array<double, 2> sp = pair_of(span_s), bh = pair_of(band_hz);
    const std::vector<double> dv = rx::vec<double>(drifts);
    aq::CarrierStats cs;
    {
        py::gil_scoped_release nogil;
        cs = f.is32 ? aq::carrier_line_stats(f.f32, t0_index, fs, sp, dv, bh)
                    : aq::carrier_line_stats(f.f64, t0_index, fs, sp, dv, bh);
    }
    return to_py(cs);
}

py::list carrier_lines(const py::object& fe, double t0_index, double fs, double pfa, std::int64_t max_cands,
                       double min_sep_hz, const py::object& band_hz, const py::object& stats) {
    aq::CarrierStats cs;
    if (stats.is_none()) {
        const Fe f = fe_of(fe);
        const std::array<double, 2> bh = pair_of(band_hz);
        py::gil_scoped_release nogil;
        cs = f.is32 ? aq::carrier_line_stats(f.f32, t0_index, fs, rxa::A1_SPAN_S, rxa::A1_DRIFTS, bh)
                    : aq::carrier_line_stats(f.f64, t0_index, fs, rxa::A1_SPAN_S, rxa::A1_DRIFTS, bh);
    } else {
        cs = carrier_stats(stats);
    }
    return to_py(aq::carrier_lines(cs, fs, pfa, max_cands, min_sep_hz));
}

std::vector<sstvae::qrss::types::CarrierCandidate> lines_of(const py::object& lines) {
    std::vector<sstvae::qrss::types::CarrierCandidate> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(lines)) {
        sstvae::qrss::types::CarrierCandidate c;
        c.f_hz = rx::as_double(rx::field(h, "f_hz"));
        out.push_back(c);
    }
    return out;
}

py::dict a2_input(const py::object& fe, double t0_index, double f_audio_hz, double drift, double fs) {
    const Fe f = fe_of(fe);
    aq::A2Input in;
    {
        py::gil_scoped_release nogil;
        in = f.is32 ? aq::a2_input(f.f32, t0_index, f_audio_hz, drift, fs)
                    : aq::a2_input(f.f64, t0_index, f_audio_hz, drift, fs);
    }
    return to_py(in);
}

py::object kind_or_ce(const py::object& kind, const py::object& ce) { return kind.is_none() ? ce : kind; }

py::tuple sums_to_py(const aq::Sums& s, bool with_var) {
    const auto nd = static_cast<py::ssize_t>(s.n_df), nt = static_cast<py::ssize_t>(s.n_tau),
               nc = static_cast<py::ssize_t>(s.nc);
    py::array lam = grid(s.lam, {nd, nt}, true);
    py::array S = grid(s.S, {nd, nt, nc}, true);
    if (!with_var) return py::make_tuple(lam, to_numpy(s.taus), S);
    return py::make_tuple(lam, to_numpy(s.taus), S, grid(s.var, {nt, nc}));
}

py::array chunks_to_py(const aq::Chunks& b) {
    const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(b.nc), static_cast<py::ssize_t>(b.ck)};
    if (b.real) {
        std::vector<double> re(b.b.size());
        for (std::size_t i = 0; i < re.size(); ++i) re[i] = b.b[i].real();
        return grid(re, shape);
    }
    return grid(b.b, shape);
}

aq::Chunks chunks_of(const py::object& b) {
    py::array a = as_array(b);
    if (a.ndim() != 2) not_implemented("b must be 2-D (n_chunk, chunk)");
    aq::Chunks c;
    c.nc = static_cast<std::size_t>(a.shape(0));
    c.ck = static_cast<std::size_t>(a.shape(1));
    c.real = a.dtype().kind() != 'c';
    c.b = rx::vec<cd>(a);
    return c;
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("acquire_a12", "acquire.py's A1, A2 and lead-in (native/core/qrss/acquire_a12)");

    py::class_<aq::PreambleKind, std::shared_ptr<aq::PreambleKind>>(m, "PreambleKind")
        .def_property_readonly("waveform", [](const aq::PreambleKind& k) { return k.waveform; })
        .def_property_readonly("seq", &seq_array)
        .def_property_readonly("t_sym", [](const aq::PreambleKind& k) { return k.t_sym; })
        .def_property_readonly("chunk", [](const aq::PreambleKind& k) { return k.chunk; })
        .def_property_readonly("data_offset_hz", [](const aq::PreambleKind& k) { return k.data_offset_hz; })
        .def_property_readonly("phase_offset", [](const aq::PreambleKind& k) { return k.phase_offset; })
        .def_property_readonly("mf_taps", [](const aq::PreambleKind& k) { return read_only(to_numpy(k.mf_taps)); })
        .def_property_readonly("n_chunks", &aq::PreambleKind::n_chunks)
        .def_property_readonly("name", &aq::PreambleKind::name);
    const py::object ce = py::cast(std::make_shared<aq::PreambleKind>(aq::ce_pre()));
    const py::object lk = py::cast(std::make_shared<aq::PreambleKind>(aq::l_pre()));
    m.attr("CE_PRE") = ce;
    m.attr("L_PRE") = lk;
    py::dict kinds;
    kinds[py::int_(cfg::WAVEFORM_CE)] = ce;
    kinds[py::int_(cfg::WAVEFORM_L)] = lk;
    m.attr("PREAMBLE_KINDS") = kinds;
    m.attr("LEAD_WIN_S") = rxa::LEAD_WIN_S;
    m.attr("LEAD_Z") = rxa::LEAD_Z;

    const py::tuple span_default = py::make_tuple(rxa::A1_SPAN_S[0], rxa::A1_SPAN_S[1]);
    const py::tuple band_default = py::make_tuple(rxa::BAND_HZ[0], rxa::BAND_HZ[1]);
    py::list drift_list;
    for (const double d : rxa::A1_DRIFTS) drift_list.append(d);
    const py::tuple drifts_default(drift_list);
    py::list df_list;
    for (const double d : rxa::A2_DF_HZ) df_list.append(d);
    const py::tuple df_default(df_list);
    const py::int_ fe_fs(cfg::FE_FS), ch_fs(cfg::CH_FS);

    // --- A1
    m.def("carrier_line_stats", &carrier_line_stats, py::arg("fe"), py::arg("t0_index"), py::arg("fs") = fe_fs,
          py::kw_only(), py::arg("span_s") = span_default, py::arg("drifts") = drifts_default,
          py::arg("band_hz") = band_default);
    m.def("carrier_lines", &carrier_lines, py::arg("fe"), py::arg("t0_index"), py::arg("fs") = fe_fs, py::kw_only(),
          py::arg("pfa") = rxa::A1_PFA, py::arg("max_cands") = rxa::A1_MAX_CANDS,
          py::arg("min_sep_hz") = rxa::A1_MIN_SEP_HZ, py::arg("band_hz") = band_default,
          py::arg("stats") = py::none());
    m.def("hint_line", [](const py::object& cs, double f_hz, double fs, double pfa, double span_hz) {
        bool refined = false;
        const auto c = aq::hint_line(carrier_stats(cs), f_hz, fs, pfa, span_hz, &refined);
        return refined ? cell_to_py(c) : py::object(rx::to_py(c));
    }, py::arg("cs"), py::arg("f_hz"), py::arg("fs") = fe_fs, py::kw_only(), py::arg("pfa") = rxa::A1_PFA,
       py::arg("span_hz") = rxa::HINT_SPAN_HZ);
    m.def("_cell_line", [](const py::object& cs, std::int64_t b, std::int64_t d, double v) {
        return cell_to_py(aq::cell_line(carrier_stats(cs), b, d, v));
    }, py::arg("cs"), py::arg("b"), py::arg("d"), py::arg("v"));
    m.def("pilot_lines", [](const py::object& cs, const py::object& lines, double sep_hz) {
        const auto lv = lines_of(lines);
        return to_py(aq::pilot_lines(carrier_stats(cs), lv, sep_hz));
    }, py::arg("cs"), py::arg("lines"), py::kw_only(), py::arg("sep_hz") = rxa::A2_ALIAS_HZ);

    // --- A2 filters
    m.def("_notch_taps", [] { return to_numpy(aq::notch_taps()); });
    m.def("_mf_taps", [](double fs) { return to_numpy(aq::mf_taps(fs)); }, py::arg("fs") = ch_fs);
    m.def("_l_mf_taps", [](double fs) { return to_numpy(aq::l_mf_taps(fs)); }, py::arg("fs") = ch_fs);

    // --- A2 channel
    m.def("_a2_input", &a2_input, py::arg("fe"), py::arg("t0_index"), py::arg("f_audio_hz"),
          py::arg("drift_hz_per_min") = 0.0, py::arg("fs") = fe_fs);
    m.def("_a2_mf", [ce](const py::object& inp, const py::object& kind) {
        const aq::A2Input in = a2_input_of(inp);
        const aq::PreambleKind k = kind_of(kind);
        aq::A2Channel c;
        {
            py::gil_scoped_release nogil;
            c = aq::a2_mf(in, k);
        }
        return to_py(c, kind_or_ce(kind, ce));
    }, py::arg("inp"), py::arg("kind") = py::none());
    m.def("preamble_channel", [ce](const py::object& fe, double t0_index, double f_audio_hz, double drift, double fs,
                                   const py::object& kind) {
        const Fe f = fe_of(fe);
        const aq::PreambleKind k = kind_of(kind);
        aq::A2Channel c;
        {
            py::gil_scoped_release nogil;
            c = f.is32 ? aq::preamble_channel(f.f32, t0_index, f_audio_hz, drift, fs, k)
                       : aq::preamble_channel(f.f64, t0_index, f_audio_hz, drift, fs, k);
        }
        return to_py(c, kind_or_ce(kind, ce));
    }, py::arg("fe"), py::arg("t0_index"), py::arg("f_audio_hz"), py::arg("drift_hz_per_min") = 0.0,
       py::arg("fs") = fe_fs, py::kw_only(), py::arg("kind") = py::none());

    // --- A2 templates and sums
    m.def("_chunks", [](const py::object& kind) { return chunks_to_py(aq::chunks(kind_of(kind))); }, py::arg("kind"));
    m.def("_norm", [](const py::object& b) { return to_numpy(aq::norm(chunks_of(b))); }, py::arg("b"));
    m.def("_templates", [](const py::object& df_hz, const py::object& kind) {
        const std::vector<double> df = rx::vec<double>(df_hz);
        const aq::Templates T = aq::templates(df, kind_of(kind));
        py::array W = grid(T.W, {static_cast<py::ssize_t>(df.size()), static_cast<py::ssize_t>(T.b.nc),
                                 static_cast<py::ssize_t>(T.b.ck)});
        return py::make_tuple(chunks_to_py(T.b), W, to_numpy(T.norm));
    }, py::arg("df_hz"), py::arg("kind") = py::none());
    m.def("preamble_lambda", [](const py::object& chan, const py::object& df_hz, double tau_s) {
        const aq::A2Channel c = chan_of(chan);
        const std::vector<double> df = rx::vec<double>(df_hz);
        aq::Sums s;
        {
            py::gil_scoped_release nogil;
            s = aq::preamble_lambda(c, df, tau_s);
        }
        return sums_to_py(s, false);
    }, py::arg("chan"), py::arg("df_hz") = df_default, py::arg("tau_s") = rxa::A2_TAU_S);
    m.def("_preamble_sums", [](const py::object& chan, const py::object& df_hz, double tau_s) {
        const aq::A2Channel c = chan_of(chan);
        const std::vector<double> df = rx::vec<double>(df_hz);
        aq::Sums s;
        {
            py::gil_scoped_release nogil;
            s = aq::preamble_sums(c, df, tau_s);
        }
        return sums_to_py(s, true);
    }, py::arg("chan"), py::arg("df_hz") = df_default, py::arg("tau_s") = rxa::A2_TAU_S);
    m.def("_fine_sums", [](const py::object& spline, const py::object& chan, double tau, double df, double rate) {
        const Spline s = spline_of(spline);
        const auto [S, n] = aq::fine_sums(s.fn, chan_of(chan), tau, df, rate);
        return py::make_tuple(to_numpy(S), to_numpy(n));
    }, py::arg("spline"), py::arg("chan"), py::arg("tau"), py::arg("df"), py::arg("rate") = 0.0);
    m.def("_coherent", [](const py::object& spline, const py::object& chan, double tau, double df, double rate) {
        const Spline s = spline_of(spline);
        return aq::coherent(s.fn, chan_of(chan), tau, df, rate);
    }, py::arg("spline"), py::arg("chan"), py::arg("tau"), py::arg("df"), py::arg("rate"));
    m.def("_phase_line", [](const py::object& S, double dt) {
        bool fitted = false;
        const auto [a, b, sd] = aq::phase_line(rx::vec<cd>(S), dt, &fitted);
        // a - b * tb, tb an np.average: an np.float64 once a line is fitted
        const py::object first = fitted ? rx::np_module().attr("float64")(a) : py::object(py::float_(a));
        return py::make_tuple(first, b, sd);
    }, py::arg("S"), py::arg("dt"));

    // --- A2 search
    m.def("preamble_search", [](const py::object& fe, double t0_index, const py::object& cand, double fs, double pfa,
                                const py::object& kind, const py::object& inp) {
        const auto c = rx::carrier_candidate(cand);
        const aq::PreambleKind k = kind_of(kind);
        aq::PreambleHit h;
        if (!inp.is_none()) {
            const aq::A2Input in = a2_input_of(inp);
            py::gil_scoped_release nogil;
            h = aq::preamble_search(in, c, pfa, k);
        } else {
            const Fe f = fe_of(fe);
            py::gil_scoped_release nogil;
            h = f.is32 ? aq::preamble_search(f.f32, t0_index, c, fs, pfa, k)
                       : aq::preamble_search(f.f64, t0_index, c, fs, pfa, k);
        }
        return to_py(h);
    }, py::arg("fe"), py::arg("t0_index"), py::arg("cand"), py::arg("fs") = fe_fs, py::kw_only(),
       py::arg("pfa") = rxa::A2_PFA, py::arg("kind") = py::none(), py::arg("inp") = py::none());

    // --- lead-in
    m.def("lead_in_span", [](const py::object& fe, double t0_index, double f_hz, double drift, double tau0_s,
                             double fs, double z_min, double t_sym, const py::object& noise_psd) {
        const Fe f = fe_of(fe);
        aq::NoisePsdFn fn;
        if (!noise_psd.is_none()) {
            fn = [noise_psd](std::span<const cf> ch, std::int64_t fs_ch, std::int64_t fs_in, double audio_hz) {
                py::gil_scoped_acquire gil;
                py::object r = noise_psd(to_numpy(std::vector<cf>(ch.begin(), ch.end())), fs_ch,
                                         py::arg("fs_in") = fs_in, py::arg("audio_hz") = audio_hz);
                return rx::vec<float>(r);
            };
        }
        py::gil_scoped_release nogil;
        return f.is32 ? aq::lead_in_span(f.f32, t0_index, f_hz, drift, tau0_s, fs, z_min, t_sym, fn)
                      : aq::lead_in_span(f.f64, t0_index, f_hz, drift, tau0_s, fs, z_min, t_sym, fn);
    }, py::arg("fe"), py::arg("t0_index"), py::arg("f_hz"), py::arg("drift_hz_per_min") = 0.0,
       py::arg("tau0_s") = 0.0, py::arg("fs") = fe_fs, py::kw_only(), py::arg("z_min") = rxa::LEAD_Z,
       py::arg("t_sym") = cfg::T_SYM, py::arg("_noise_psd") = py::none());

    m.def("to_detection", [](const py::object& hit, double duration_s) {
        return rx::to_py(hit_of(hit).to_detection(duration_s));
    }, py::arg("hit"), py::arg("duration_s") = 1800.0);

    m.def("_log_f32", [](const py::object& x) {
        py::array a = py::array::ensure(rx::np_module().attr("asarray")(x, py::arg("dtype") = "float32"));
        if (!a) throw py::error_already_set();
        std::vector<float> v = rx::vec<float>(a);
        for (float& e : v) e = sm::np::log_f32(e);
        return to_numpy(v).attr("reshape")(a.attr("shape"));
    }, py::arg("x"));
}

const Registrar reg("acquire_a12", 50, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
