// sstvae_native.qrss.track: bindings of native/core/qrss/track (the tracker
// core of sstvae/qrss/track.py). Names, arguments and defaults follow the
// reference's, so tests/qrss_native/track.py can assign them over it:
//
//   derotate(chan, fn)                    fn a PathFn (native, NativePathFn or
//                                         the reference's) or any callable
//   _pulse_at(x, t_sym), _pulse_g0(t_sym)
//   mf_table(t_sym=T_SYM) -> "MfTable" dict, its tab a read-only view of the
//                            C++ table (one per waveform, built once)
//   symbol_mf(z, ch_fs, timing, pos, *, t0_index, n_pos, chunk=4096, table=None)
//   psi_at(keep_ch, timing, pos, *, t0_index, n_pos, table=None)
//   _lowpass(x, hz, fs=CH_FS), _mf_taps(fs=CH_FS, t_sym=T_SYM)
//   known_symbols(spec, hdr_bits=None) -> "Known" dict
//   make_classes(spec, hdr_bits=None, prior=None) -> "Classes" dict
//   remove_spread(z, chan, spec, timing, spread)   z itself for spread None
//   templates(spec, grid_pos, mu, nu, keying=None, ook=False, keyed_from_pos=None)
//   _kf_forward(mu, R, q, Pu, store), kalman_rts(mu, R, q, Pu, causal=False)
//   select_q(mu, R, Pu, grid=Q_GRID) -> (grid's own element, loglik) or None
//   _window_unknown(spec, grid_pos, ook_possible=True, reach=SPAN)
//   _smooth(x, n)
//   _measurements(tr, kappa, sf, Pu_loc, c=None, meas=None), _pu_initial(tr)
//   _recenter(tr) -> (PathFn, df_max); tr.freq itself when nothing qualified
//   _keying_tuple(cw_known)
//   _lead_supported(chan, z_d, tm, spec, lead)
//   track(chan, spec, det, classes=None, cw_known=None, prior=None, *,
//         timing=None, freq=None, withhold_known=False, outer=2, refine=True,
//         lead_in_s=None, z_d=None)
//       -> ("TrackResult" dict with chan None, same_freq, same_timing):
//          same_* say the result's path / timing are the ones passed in (or
//          det's), so the row can hand back those very objects
//   refine_timing(tr, iters=3, u_ref=None, P_ref=None) -> "Timing" dict or None
//   z_ref_stat(tr, c_known, d_known=None)
//   erase(tr, spans)                      sets tr.psi (a new array), as the reference
//   report(tr, kappa=1.0, suspect=False) -> "TrackReport" dict
//
// A track is read as the reference reads it: by attribute (or a dict's key),
// and only the fields the function uses -- test_qrss_lunits' fakes carry
// chan=None and classes=SimpleNamespace(known=...), and a CE-only function
// given an L frame refuses it before reading anything else. The capture's
// noise table is its `_psd`: when that is still None, `track` and
// `_lead_supported` call chan.noise_psd_at(0.0) first (whatever implements
// it fills it, the C++ row or the reference), so the C++ interpolates the
// very table the reference would; `report` asks tr.chan.noise_floor().

#include "qrss/rx_types.hpp"

#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/config.hpp"
#include "qrss/demod_rx.hpp"
#include "qrss/frame.hpp"
#include "qrss/mf_rx.hpp"
#include "qrss/rx_config.hpp"
#include "qrss/track.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace tk = sstvae::qrss::track;
namespace ce = sstvae::qrss::ce;
namespace fr = sstvae::qrss::frame;
namespace cfg = sstvae::qrss::config;
namespace types = sstvae::qrss::types;
using cd = std::complex<double>;

[[noreturn]] void raise(PyObject* type, const std::string& what) {
    PyErr_SetString(type, what.c_str());
    throw py::error_already_set();
}

py::object numpy() { return rx::np_module(); }

// np.asarray(o) and its dtype kind ('c' complex, 'f' float, 'i'/'u'/'b').
char kind_of(const py::handle& o) {
    py::array a = py::array::ensure(numpy().attr("asarray")(o));
    if (!a) throw py::error_already_set();
    return a.dtype().kind();
}

std::vector<std::int64_t> shape_of(const py::handle& o) {
    py::array a = py::array::ensure(numpy().attr("asarray")(o));
    if (!a) throw py::error_already_set();
    return std::vector<std::int64_t>(a.shape(), a.shape() + a.ndim());
}

// --- what the reference reads -------------------------------------------------

// A Timing as rx_index reads it: tau0, ppm and gamma.
types::Timing timing_lite(const py::handle& o) {
    if (o.is_none()) raise(PyExc_AttributeError, "'NoneType' object has no attribute 'tau0'");
    types::Timing t;
    t.tau0 = rx::as_double(rx::field(o, "tau0"));
    t.ppm = rx::as_double(rx::field(o, "ppm"));
    t.gamma = rx::as_double(rx::field(o, "gamma"));
    return t;
}

// Whether o is a PathFn the C++ can hold (rx::path_fn reads it).
bool is_pathfn(const py::handle& o) {
    if (py::isinstance<tk::PathFn>(o)) return true;
    if (py::hasattr(o, "_native") && py::isinstance<tk::PathFn>(o.attr("_native"))) return true;
    return py::hasattr(o, "_spl") && py::hasattr(o, "_line") && py::hasattr(o, "lo") && py::hasattr(o, "hi");
}

tk::PathFn path_of(const py::handle& o) {
    if (!is_pathfn(o))
        raise(PyExc_TypeError, "the C++ tracker needs a track.PathFn as its frequency path, not " +
                                   std::string(py::str(py::type::of(o).attr("__name__"))));
    return rx::path_fn(o);
}

tk::PathFn no_path() { return tk::PathFn::from_parts(0.0, 0.0, {0.0, 0.0}, std::nullopt); }

// A capture's grid (fs, t0_index, f_mix) without its samples.
tk::ChanCapture chan_grid(const py::handle& o) {
    tk::ChanCapture c;
    c.fs = static_cast<int>(rx::as_int(rx::field(o, "fs")));
    c.t0_index = rx::as_double(rx::field(o, "t0_index"));
    c.f_mix = rx::rational(rx::field(o, "f_mix"));
    return c;
}

// A capture holding only its t0_index (what refine_timing reads of tr.chan).
std::shared_ptr<const tk::ChanCapture> chan_t0(const py::handle& o) {
    auto c = std::make_shared<tk::ChanCapture>();
    c->t0_index = rx::as_double(rx::field(o, "t0_index"));
    return c;
}

// chan.noise_psd_at(0.0) when chan._psd is still None: fills the table by
// whatever implements the method (the reference caches it there too).
void ensure_psd(const py::handle& chan) {
    if (py::isinstance<py::dict>(chan) || !py::hasattr(chan, "_psd")) return;
    if (chan.attr("_psd").is_none()) chan.attr("noise_psd_at")(0.0);
}

// The full capture for the tracker, its noise table filled.
std::shared_ptr<const tk::ChanCapture> chan_full(const py::handle& chan) {
    ensure_psd(chan);
    return std::make_shared<const tk::ChanCapture>(rx::chan_capture(chan));
}

std::shared_ptr<const std::vector<cd>> zd_of(const py::handle& o) {
    return std::make_shared<const std::vector<cd>>(rx::vec<cd>(o));
}

// A track with the fields `names` read (a per-position array None is empty).
struct TrackRead {
    bool timing = false, freq = false, z_d = false, chan_t0 = false;
    bool m = false, c = false, u = false, P = false, N = false, psi = false, meas = false, pu = false;
    bool known = false;   // tr.classes.known (classes None: none)
    bool q = false, z_ref = false;
};

tk::TrackResult read_track(const py::handle& o, const fr::FrameSpec& spec, const TrackRead& r) {
    types::Timing tm;
    if (r.timing) tm = timing_lite(rx::field(o, "timing"));
    tk::TrackResult t(spec, r.chan_t0 ? chan_t0(rx::field(o, "chan")) : nullptr, tm,
                      r.freq ? path_of(rx::field(o, "freq")) : no_path());
    t.pos = rx::vec<std::int64_t>(rx::field(o, "pos"));
    if (r.z_d) {
        const py::object zd = rx::field(o, "z_d");
        if (!zd.is_none()) t.z_d = zd_of(zd);
    }
    if (r.m) t.m = rx::vec_or_empty<cd>(rx::field(o, "m"));
    if (r.c) t.c = rx::vec_or_empty<cd>(rx::field(o, "c"));
    if (r.u) t.u = rx::vec_or_empty<cd>(rx::field(o, "u"));
    if (r.P) t.P = rx::vec_or_empty<double>(rx::field(o, "P"));
    if (r.N) t.N = rx::vec_or_empty<double>(rx::field(o, "N"));
    if (r.psi) t.psi = rx::vec_or_empty<double>(rx::field(o, "psi"));
    if (r.meas) {
        const py::object m = rx::field(o, "meas");
        if (!m.is_none()) t.meas = rx::vec_bool(m);
    }
    if (r.pu) t.pu = rx::vec_or_empty<double>(rx::field(o, "pu"));
    if (r.known) {
        const py::object cl = rx::field(o, "classes");
        if (!cl.is_none()) {
            tk::Classes c;
            c.known = rx::vec_bool(rx::field(cl, "known"));
            t.classes = std::move(c);
        }
    }
    if (r.q) t.q = rx::as_double(rx::field(o, "q"));
    if (r.z_ref) t.z_ref = rx::as_double(rx::field(o, "z_ref"));
    return t;
}

fr::FrameSpec spec_of_track(const py::handle& o) { return rx::spec(rx::field(o, "spec")); }

// For the functions that never read tr.spec (_measurements, _pu_initial):
// the track's own when it has one, else a placeholder the C++ never reads.
fr::FrameSpec no_spec(const py::handle& o) {
    const py::object s = demod_rx::opt_field(o, "spec");
    if (!s.is_none()) return rx::spec(s);
    return fr::presets(cfg::PresetGroup::Presets).front();
}

// --- hdr_bits, classes, keying --------------------------------------------------

// np.asarray(hdr_bits, dtype=np.float64) as frame's binding reads it.
std::optional<fr::HdrBits> bits_f64(const py::object& obj) {
    if (obj.is_none()) return std::nullopt;
    auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
        numpy().attr("asarray")(obj, py::arg("dtype") = "float64"));
    if (!a) throw py::error_already_set();
    fr::HdrBits b;
    b.one_d = a.ndim() == 1;
    b.len = a.ndim() >= 1 ? static_cast<std::size_t>(a.shape(0)) : 1;
    if (b.one_d) b.values.assign(a.data(), a.data() + a.size());
    return b;
}

const fr::HdrBits* ptr(const std::optional<fr::HdrBits>& b) { return b ? &*b : nullptr; }

std::string shape_str(const py::array& a) {        // numpy's "(2,2474)", "(5,)", "()"
    std::string s = "(";
    for (py::ssize_t i = 0; i < a.ndim(); ++i) s += (i ? "," : "") + std::to_string(a.shape(i));
    return s + (a.ndim() == 1 ? ",)" : ")");
}

// make_classes' mu[lay.hdr_bits] = 1 - 2 bits for header bits that are not 1-D: numpy
// broadcasts a shape whose only obstacle is leading unit axes, and refuses any other with
// its own message. A broadcastable one is handed on flattened (one_d stays false, so the
// spread copy refuses it as the reference's spread_signs does).
std::optional<fr::HdrBits> classes_bits(const fr::FrameSpec& spec, const py::object& obj) {
    std::optional<fr::HdrBits> b = bits_f64(obj);
    if (!b || b->one_d || !spec.n_hdr()) return b;
    const py::array a = numpy().attr("asarray")(obj, py::arg("dtype") = "float64");
    py::ssize_t lead = 0;
    while (lead < a.ndim() && a.shape(lead) == 1) ++lead;
    const std::size_t n = fr::layout(spec)->hdr_bits.size();
    const bool ok = lead == a.ndim() ||
                    (lead + 1 == a.ndim() && static_cast<std::size_t>(a.shape(lead)) == n);
    if (!ok)
        raise(PyExc_ValueError, "shape mismatch: value array of shape " + shape_str(a) +
                                    " could not be broadcast to indexing result of shape (" + std::to_string(n) + ",)");
    b->values = rx::vec<double>(numpy().attr("ravel")(a));
    return b;
}

// known_symbols' np.concatenate([val, 1 - 2 bits]) refuses bits that are not 1-D.
void check_known_bits(const fr::FrameSpec& spec, const py::object& obj) {
    if (obj.is_none() || !spec.n_hdr()) return;
    const py::array a = numpy().attr("asarray")(obj, py::arg("dtype") = "float64");
    if (a.ndim() == 1) return;
    raise(PyExc_ValueError, "all the input arrays must have same number of dimensions, but the array at index 0 has 1 "
                            "dimension(s) and the array at index 1 has " + std::to_string(a.ndim()) + " dimension(s)");
}

std::optional<types::EmPrior> prior_of(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    return types::EmPrior{rx::vec<double>(rx::field(o, "x_hat")), rx::vec<double>(rx::field(o, "v"))};
}

// track's classes as it reads them: mu, nu, known, spread.
tk::Classes classes_of(const py::handle& o) {
    tk::Classes c;
    c.mu = rx::vec<double>(rx::field(o, "mu"));
    c.nu = rx::vec<double>(rx::field(o, "nu"));
    c.known = rx::vec_bool(rx::field(o, "known"));
    c.spread = rx::opt_vec<double>(rx::field(o, "spread"));
    const py::object hk = demod_rx::opt_field(o, "hdr_known");
    c.hdr_known = !hk.is_none() && rx::as_bool(hk);
    const py::object rho = demod_rx::opt_field(o, "spread_rho");
    c.spread_rho = rho.is_none() ? 0.0 : rx::as_double(rho);
    return c;
}

// window_keying(keying, spec) as ce's binding converts it: None, else the
// validated (n_win, 192) keying.
std::optional<ce::Keying> keying_of(const py::handle& keying, const fr::FrameSpec& spec) {
    if (keying.is_none()) return std::nullopt;
    const py::object np = numpy();
    py::array k = np.attr("asarray")(keying);
    std::vector<std::int64_t> shape(k.shape(), k.shape() + k.ndim());
    const char kind = k.dtype().kind();
    if (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f') {
        const std::vector<double> v = from_numpy<double>(np.attr("asarray")(k, py::arg("dtype") = "float64"));
        return ce::window_keying(std::span<const double>(v), shape, spec);
    }
    std::vector<std::int64_t> sh = shape;
    if (sh.size() == 1) sh = {spec.n_win(), sh[0]};
    bool ok = sh.size() == 2 && sh[0] == spec.n_win() && sh[1] == cfg::CW_UNITS;
    if (ok) {
        py::object b = np.attr("broadcast_to")(k, py::make_tuple(sh[0], sh[1]));
        auto ne = [&](int v) {
            py::object r = py::reinterpret_steal<py::object>(PyObject_RichCompare(b.ptr(), py::int_(v).ptr(), Py_NE));
            if (!r) throw py::error_already_set();
            return r;
        };
        ok = !py::bool_(np.attr("any")(ne(0) & ne(1))).cast<bool>();
    }
    if (!ok) raise(PyExc_ValueError, ce::keying_refusal(spec, shape));
    const std::vector<double> v =
        from_numpy<double>(np.attr("asarray")(k.attr("astype")("uint8"), py::arg("dtype") = "float64"));
    return ce::window_keying(std::span<const double>(v), shape, spec);
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

// _keying_tuple(cw_known): (None or np.asarray(k, uint8), bool(ook)).
std::pair<py::object, bool> keying_tuple(const py::object& cw_known) {
    if (cw_known.is_none()) return {py::none(), false};
    const auto [k, ook] = unpack2(cw_known);
    py::object arr = k.is_none() ? py::object(py::none()) : numpy().attr("asarray")(k, py::arg("dtype") = "uint8");
    return {arr, rx::as_bool(ook)};
}

// --- tables -----------------------------------------------------------------------

// A read-only numpy view of storage that lives as long as the process (the
// two function-local tables of tk::mf_table).
py::array view(std::span<const double> v) {
    py::array_t<double> a(std::vector<py::ssize_t>{static_cast<py::ssize_t>(v.size())},
                          std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(double))}, v.data(),
                          py::capsule(v.data(), [](void*) {}));
    a.attr("setflags")(py::arg("write") = false);
    return a;
}

py::dict table_dict(const tk::MfTable& t, const py::object& t_sym) {
    py::dict d = rx::typed("MfTable");
    d["t_sym"] = t_sym;
    d["sps"] = t.sps();
    d["half"] = t.half();
    d["tab"] = view(t.tab());
    d["g0"] = t.g0();
    return d;
}

// The C++ table `table` is when its tab is a view of one this module handed
// out (and its other fields are that table's), else nullptr.
const tk::MfTable* builtin_table(const py::handle& table) {
    const py::object tab = rx::field(table, "tab");
    if (!py::isinstance<py::array>(tab)) return nullptr;
    const py::array a = py::reinterpret_borrow<py::array>(tab);
    if (a.ndim() != 1 || a.dtype().kind() != 'f' || a.itemsize() != 8 || a.strides(0) != 8) return nullptr;
    const double t_sym = rx::as_double(rx::field(table, "t_sym"));
    if (t_sym != cfg::T_SYM && t_sym != cfg::L_T_SYM) return nullptr;
    const tk::MfTable& s = tk::mf_table(t_sym);
    if (a.data() != static_cast<const void*>(s.tab().data()) || static_cast<std::size_t>(a.size()) != s.size())
        return nullptr;
    if (rx::as_double(rx::field(table, "sps")) != s.sps() || rx::as_int(rx::field(table, "half")) != s.half() ||
        rx::as_double(rx::field(table, "g0")) != s.g0())
        return nullptr;
    return &s;
}

// `table` as symbol_mf reads it: None is CE's, a module table is used as it
// is, anything else converted.
struct TableArg {
    const tk::MfTable* table = nullptr;
    std::optional<tk::MfTable> owned;
};
TableArg table_of(const py::object& table) {
    TableArg t;
    if (table.is_none()) {
        t.table = &tk::mf_table();
        return t;
    }
    t.table = builtin_table(table);
    if (!t.table) {
        t.owned.emplace(rx::mf_table(table));
        t.table = &*t.owned;
    }
    return t;
}

// --- results ----------------------------------------------------------------------

py::array complex_or_float_empty(const std::vector<cd>& v) {
    if (v.empty()) return to_numpy(std::vector<double>{});   // np.array([]) is float64
    return to_numpy(v);
}

bool same_bits(double a, double b) { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

bool same_vec(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!same_bits(a[i], b[i])) return false;
    return true;
}

bool same_path(const tk::PathFn& a, const tk::PathFn& b) {
    if (!same_bits(a.lo(), b.lo()) || !same_bits(a.hi(), b.hi()) || !same_vec(a.line(), b.line())) return false;
    if (a.spline().has_value() != b.spline().has_value()) return false;
    if (!a.spline()) return true;
    return same_vec(a.spline()->t, b.spline()->t) && same_vec(a.spline()->c, b.spline()->c) &&
           a.spline()->k == b.spline()->k;
}

bool same_timing(const types::Timing& a, const types::Timing& b) {
    return same_bits(a.tau0, b.tau0) && same_bits(a.ppm, b.ppm) && same_bits(a.gamma, b.gamma) &&
           same_bits(a.z, b.z) && a.cov.rows == b.cov.rows && a.cov.cols == b.cov.cols && same_vec(a.cov.data, b.cov.data);
}

// --- the bindings -----------------------------------------------------------------

py::object derotate_py(const py::object& chan, const py::object& fn) {
    const tk::ChanCapture grid = chan_grid(chan);
    const py::object ch = rx::field(chan, "ch");
    const bool single = py::isinstance<py::array>(ch) &&
                        py::reinterpret_borrow<py::array>(ch).dtype().is(py::dtype::of<std::complex<float>>());
    std::vector<cd> out;
    if (is_pathfn(fn)) {
        const tk::PathFn p = rx::path_fn(fn);
        if (single) {
            tk::ChanCapture c = grid;
            c.ch = rx::vec<std::complex<float>>(ch);
            py::gil_scoped_release nogil;
            out = tk::derotate(c, p);
        } else {
            const std::vector<cd> v = rx::vec<cd>(ch);
            py::gil_scoped_release nogil;
            out = tk::derotate(std::span<const cd>(v), grid, p);
        }
    } else {
        // any other callable, evaluated by Python (f(t) - mix broadcasts)
        // -- like the reference, a result of another length is NOT broadcast: its cumsum
        // has that length and np.interp over the n capture indices refuses it (with no
        // capture at all, the product with the empty CH broadcasts a single value away).
        const tk::PathEval eval = [&fn](std::span<const double> t) {
            std::vector<double> f = rx::vec<double>(numpy().attr("ravel")(fn(to_numpy(t))));
            if (f.size() == t.size()) return f;
            if (!t.empty()) raise(PyExc_ValueError, "fp and xp are not of the same length.");
            if (f.size() == 1) return std::vector<double>{};
            raise(PyExc_ValueError,
                  "operands could not be broadcast together with shapes (0,) (" + std::to_string(f.size()) + ",) ");
        };
        const std::vector<cd> v = rx::vec<cd>(ch);
        out = tk::derotate(std::span<const cd>(v), grid, eval);
    }
    return to_numpy(out);
}

py::object symbol_mf_py(const py::object& z, const py::object& ch_fs, const py::object& timing, const py::object& pos,
                        const py::object& t0_index, const py::object& n_pos, const py::object& table) {
    if (rx::as_int(ch_fs) != cfg::CH_FS) raise(PyExc_AssertionError, "");
    const TableArg tb = table_of(table);
    const types::Timing tm = timing_lite(timing);
    const std::vector<double> p = rx::vec<double>(pos);
    const double t0 = rx::as_double(t0_index);
    const std::int64_t np_ = rx::as_int(n_pos);
    std::vector<cd> out;
    if (kind_of(z) == 'c') {
        const std::vector<cd> v = rx::vec<cd>(z);
        py::gil_scoped_release nogil;
        out = tk::symbol_mf(std::span<const cd>(v), tm, p, t0, np_, tb.table);
    } else {
        const std::vector<double> v = rx::vec<double>(z);
        py::gil_scoped_release nogil;
        out = tk::symbol_mf(std::span<const double>(v), tm, p, t0, np_, tb.table);
    }
    return to_numpy(out);
}

py::object psi_at_py(const py::object& keep, const py::object& timing, const py::object& pos,
                     const py::object& t0_index, const py::object& n_pos, const py::object& table) {
    if (keep.is_none()) return to_numpy(std::vector<double>(static_cast<std::size_t>(py::len(numpy().attr("asarray")(pos))), 1.0));
    const TableArg tb = table_of(table);
    const types::Timing tm = timing_lite(timing);
    const std::vector<double> p = rx::vec<double>(pos);
    const std::vector<double> k = rx::vec<double>(keep);
    const double t0 = rx::as_double(t0_index);
    const std::int64_t np_ = rx::as_int(n_pos);
    std::vector<double> out;
    {
        py::gil_scoped_release nogil;
        out = tk::psi_at(std::span<const double>(k), tm, p, t0, np_, tb.table);
    }
    return to_numpy(out);
}

py::object templates_py(const py::object& spec, const py::object& grid_pos, const py::object& mu, const py::object& nu,
                        const py::object& keying, const py::object& ook, const py::object& keyed_from_pos) {
    const fr::FrameSpec s = rx::spec(spec);
    tk::ce_only(s, "track.templates", "ltrack's L templates");
    const std::vector<std::int64_t> g = rx::vec<std::int64_t>(grid_pos);
    if (g.empty()) raise(PyExc_IndexError, "index 0 is out of bounds for axis 0 with size 0");
    ce::check_sym(s, shape_of(numpy().attr("asarray")(mu, py::arg("dtype") = "float64")), "sym");
    const std::vector<double> m = rx::vec<double>(mu);
    ce::check_sym(s, shape_of(numpy().attr("asarray")(nu, py::arg("dtype") = "float64")), "var");
    const std::vector<double> v = rx::vec<double>(nu);
    std::optional<double> lo;
    if (!keyed_from_pos.is_none()) lo = rx::as_double(keyed_from_pos);
    const auto k = keying_of(keying, s);
    const bool o = rx::as_bool(ook);
    tk::Templates t;
    {
        py::gil_scoped_release nogil;
        t = tk::templates(s, g, m, v, k ? &*k : nullptr, o, lo);
    }
    return py::make_tuple(to_numpy(t.c), to_numpy(t.sf));
}

py::object kf_forward_py(const py::object& mu, const py::object& R, const py::object& q, const py::object& Pu,
                         const py::object& store) {
    const std::vector<cd> m = rx::vec<cd>(mu);
    const std::vector<double> r = rx::vec<double>(R);
    const double qq = rx::as_double(q), pu = rx::as_double(Pu);
    if (!rx::as_bool(store)) {
        double ll = 0.0;
        {
            py::gil_scoped_release nogil;     // (the float is built with the GIL held again)
            ll = tk::kf_forward(m, r, qq, pu);
        }
        return py::float_(ll);
    }
    tk::KfPass p;
    {
        py::gil_scoped_release nogil;
        p = tk::kf_forward_pass(m, r, qq, pu);
    }
    auto cl = [](const std::vector<cd>& v) {
        py::list l(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) l[i] = py::cast(v[i]);
        return l;
    };
    auto fl = [](const std::vector<double>& v) {
        py::list l(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) l[i] = py::float_(v[i]);
        return l;
    };
    return py::make_tuple(p.loglik, py::make_tuple(cl(p.xa), cl(p.xb), fl(p.pa), fl(p.pb), fl(p.pc), cl(p.fx1),
                                                   cl(p.fx2), fl(p.fa), fl(p.fb), fl(p.fc)));
}

py::object select_q_py(const py::object& mu, const py::object& R, const py::object& Pu, const py::object& grid) {
    const std::vector<cd> m = rx::vec<cd>(mu);
    const std::vector<double> r = rx::vec<double>(R);
    const double pu = rx::as_double(Pu);
    std::vector<py::object> items;
    std::vector<double> qs;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(grid))) {
        items.push_back(py::reinterpret_borrow<py::object>(h));
        qs.push_back(rx::as_double(h));
    }
    std::vector<double> ll(qs.size());
    {
        py::gil_scoped_release nogil;
        for (std::size_t i = 0; i < qs.size(); ++i) ll[i] = tk::kf_forward(m, r, qs[i], pu);
    }
    if (qs.empty()) return py::none();
    std::size_t best = 0;
    for (std::size_t i = 1; i < qs.size(); ++i)
        if (ll[i] > ll[best]) best = i;
    return py::make_tuple(items[best], ll[best]);
}

py::object window_unknown_py(const py::object& spec, const py::object& grid_pos, const py::object& ook_possible,
                             const py::object& reach) {
    const fr::FrameSpec s = rx::spec(spec);
    const char k = kind_of(grid_pos);
    const bool ook = rx::as_bool(ook_possible);
    const bool int_reach = py::isinstance<py::int_>(reach) || py::isinstance(reach, numpy().attr("integer"));
    std::vector<std::uint8_t> out;
    if ((k == 'i' || k == 'u' || k == 'b') && int_reach) {
        const std::vector<std::int64_t> g = rx::vec<std::int64_t>(grid_pos);
        out = tk::window_unknown(s, std::span<const std::int64_t>(g), ook, rx::as_int(reach));
    } else {
        const std::vector<double> g = rx::vec<double>(grid_pos);
        out = tk::window_unknown(s, std::span<const double>(g), ook, rx::as_double(reach));
    }
    return rx::arr_bool(out);
}

py::object track_py(const py::object& chan, const py::object& spec, const py::object& det, const py::object& classes,
                    const py::object& cw_known, const py::object& prior, const py::object& timing,
                    const py::object& freq, const py::object& withhold_known, const py::object& outer,
                    const py::object& refine, const py::object& lead_in_s, const py::object& z_d) {
    const fr::FrameSpec s = rx::spec(spec);
    tk::ce_only(s, "track.track", "ltrack.track_l");
    tk::TrackArgs a;
    if (classes.is_none())
        a.prior = prior_of(prior);
    else
        a.classes = classes_of(classes);
    const auto [k, ook] = keying_tuple(cw_known);
    a.keying = keying_of(k, s);
    a.ook = ook;
    types::Detection d;
    if (freq.is_none())
        d.path = rx::freq_path(rx::field(det, "path"));
    else
        a.freq = path_of(freq);
    if (timing.is_none()) {
        const py::object dt = rx::field(det, "timing");
        if (dt.is_none()) raise(PyExc_AttributeError, "'NoneType' object has no attribute 'tau0'");
        a.timing = rx::timing(dt);
    } else {
        a.timing = rx::timing(timing);
    }
    if (lead_in_s.is_none())
        a.lead_in_s = rx::as_double(rx::field(det, "lead_in_s"));
    else
        a.lead_in_s = rx::as_double(lead_in_s);
    a.withhold_known = rx::as_bool(withhold_known);
    a.outer = static_cast<int>(rx::as_int(outer));
    a.refine = rx::as_bool(refine);
    if (!z_d.is_none()) a.z_d = zd_of(z_d);
    const auto c = chan_full(chan);
    std::optional<tk::TrackResult> r;
    {
        py::gil_scoped_release nogil;
        r.emplace(tk::track(c, s, d, a));
    }
    const bool same_f = a.freq && same_path(r->freq, *a.freq);
    const bool same_t = same_timing(r->timing, *a.timing);
    r->chan = nullptr;   // the row puts the caller's object back
    return py::make_tuple(rx::to_py(*r), same_f, same_t);
}

py::object refine_timing_py(const py::object& tr, const py::object& iters, const py::object& u_ref,
                            const py::object& P_ref) {
    const fr::FrameSpec s = spec_of_track(tr);
    tk::ce_only(s, "track.refine_timing", "ltrack.refine_timing_l");
    TrackRead want;
    want.timing = want.z_d = want.chan_t0 = true;
    want.c = want.psi = want.N = want.meas = want.pu = want.known = true;
    want.u = u_ref.is_none();
    want.P = P_ref.is_none();
    const tk::TrackResult t = read_track(tr, s, want);
    std::optional<std::vector<cd>> u;
    std::optional<std::vector<double>> P;
    if (!u_ref.is_none()) u = rx::vec<cd>(u_ref);
    if (!P_ref.is_none()) P = rx::vec<double>(P_ref);
    const int it = static_cast<int>(rx::as_int(iters));
    std::optional<types::Timing> out;
    {
        py::gil_scoped_release nogil;
        out = tk::refine_timing(t, it, u ? &*u : nullptr, P ? &*P : nullptr);
    }
    if (!out) return py::none();
    return rx::to_py(*out);
}

void bind(py::module_& q) {
    rx::register_translators();
    demod_rx::register_wrong_waveform();   // an L frame's AssertionError
    py::module_ m = q.def_submodule("track", "native/core/qrss/track (sstvae.qrss.track's tracker core)");

    m.def("derotate", &derotate_py, py::arg("chan"), py::arg("fn"));

    m.def("_pulse_at", [](const py::object& x, double t_sym) -> py::object {
        py::array a = py::array::ensure(numpy().attr("asarray")(x, py::arg("dtype") = "float64"));
        if (!a) throw py::error_already_set();
        const std::vector<double> v = from_numpy<double>(a);
        const std::vector<double> out = tk::pulse_at(std::span<const double>(v), t_sym);
        return to_numpy(out).attr("reshape")(a.attr("shape"));
    }, py::arg("x"), py::arg("t_sym"));
    m.def("_pulse_g0", [](double t_sym) { return tk::pulse_g0(t_sym); }, py::arg("t_sym"));

    m.def("mf_table", [](const py::object& t_sym) {
        const double t = rx::as_double(t_sym);
        const tk::MfTable& tb = tk::mf_table(t);
        return table_dict(tb, t_sym);
    }, py::arg("t_sym") = cfg::T_SYM);

    m.def("symbol_mf", [](const py::object& z, const py::object& ch_fs, const py::object& timing,
                          const py::object& pos, const py::object& t0_index, const py::object& n_pos,
                          const py::object& chunk, const py::object& table) {
        (void)chunk;   // the chunking changes nothing in the result
        return symbol_mf_py(z, ch_fs, timing, pos, t0_index, n_pos, table);
    }, py::arg("z"), py::arg("ch_fs"), py::arg("timing"), py::arg("pos"), py::kw_only(), py::arg("t0_index"),
       py::arg("n_pos"), py::arg("chunk") = 4096, py::arg("table") = py::none());

    m.def("psi_at", &psi_at_py, py::arg("keep_ch"), py::arg("timing"), py::arg("pos"), py::kw_only(),
          py::arg("t0_index"), py::arg("n_pos"), py::arg("table") = py::none());

    m.def("_lowpass", [](const py::object& x, const py::object& hz, const py::object& fs) -> py::object {
        const double h = rx::as_double(hz), f = rx::as_double(fs);
        if (kind_of(x) == 'c') {
            const std::vector<cd> v = rx::vec<cd>(x);
            std::vector<cd> out;
            {
                py::gil_scoped_release nogil;
                out = tk::lowpass(std::span<const cd>(v), h, f);
            }
            return to_numpy(out);
        }
        const std::vector<double> v = rx::vec<double>(x);
        std::vector<double> out;
        {
            py::gil_scoped_release nogil;
            out = tk::lowpass(std::span<const double>(v), h, f);
        }
        return to_numpy(out);
    }, py::arg("x"), py::arg("hz"), py::arg("fs") = cfg::CH_FS);

    m.def("_mf_taps", [](const py::object& fs, const py::object& t_sym) {
        return to_numpy(tk::mf_taps(rx::as_double(fs), rx::as_double(t_sym)));
    }, py::arg("fs") = cfg::CH_FS, py::arg("t_sym") = cfg::T_SYM);

    m.def("known_symbols", [](const py::object& spec, const py::object& hdr_bits) {
        const fr::FrameSpec s = rx::spec(spec);
        tk::ce_only(s, "track.known_symbols", "lframe.layout's known_idx/known_val");
        const auto b = bits_f64(hdr_bits);
        check_known_bits(s, hdr_bits);
        return rx::to_py(tk::known_symbols(s, ptr(b)));
    }, py::arg("spec"), py::arg("hdr_bits") = py::none());

    m.def("make_classes", [](const py::object& spec, const py::object& hdr_bits, const py::object& prior) {
        const fr::FrameSpec s = rx::spec(spec);
        tk::ce_only(s, "track.make_classes", "ltrack.make_classes_l");
        const auto b = classes_bits(s, hdr_bits);
        const auto p = prior_of(prior);
        return rx::to_py(tk::make_classes(s, ptr(b), p ? &*p : nullptr));
    }, py::arg("spec"), py::arg("hdr_bits") = py::none(), py::arg("prior") = py::none());

    m.def("remove_spread", [](const py::object& z, const py::object& chan, const py::object& spec,
                              const py::object& timing, const py::object& spread) -> py::object {
        const fr::FrameSpec s = rx::spec(spec);
        tk::ce_only(s, "track.remove_spread", "nothing: L's spread copy is linear, in ltrack.make_classes_l's mu");
        if (spread.is_none()) return z;
        const std::vector<cd> v = rx::vec<cd>(z);
        const double t0 = rx::as_double(rx::field(chan, "t0_index"));
        const types::Timing tm = timing_lite(timing);
        ce::check_sym(s, shape_of(numpy().attr("asarray")(spread, py::arg("dtype") = "float64")), "sym");
        const std::vector<double> sp = rx::vec<double>(spread);
        std::vector<cd> out;
        {
            py::gil_scoped_release nogil;
            out = tk::remove_spread(v, t0, s, tm, &sp);
        }
        return to_numpy(out);
    }, py::arg("z"), py::arg("chan"), py::arg("spec"), py::arg("timing"), py::arg("spread"));

    m.def("templates", &templates_py, py::arg("spec"), py::arg("grid_pos"), py::arg("mu"), py::arg("nu"),
          py::arg("keying") = py::none(), py::arg("ook") = false, py::arg("keyed_from_pos") = py::none());

    m.def("_kf_forward", &kf_forward_py, py::arg("mu"), py::arg("R"), py::arg("q"), py::arg("Pu"), py::arg("store"));

    m.def("kalman_rts", [](const py::object& mu, const py::object& R, const py::object& q, const py::object& Pu,
                           const py::object& causal) {
        const std::vector<cd> mv = rx::vec<cd>(mu);
        const std::vector<double> r = rx::vec<double>(R);
        const double qq = rx::as_double(q), pu = rx::as_double(Pu);
        const bool c = rx::as_bool(causal);
        tk::KalmanResult k;
        {
            py::gil_scoped_release nogil;
            k = tk::kalman_rts(mv, r, qq, pu, c);
        }
        return py::make_tuple(complex_or_float_empty(k.u), to_numpy(k.P), k.loglik);
    }, py::arg("mu"), py::arg("R"), py::arg("q"), py::arg("Pu"), py::arg("causal") = false);

    py::tuple q_grid(tk::Q_GRID.size());
    for (std::size_t i = 0; i < tk::Q_GRID.size(); ++i) q_grid[i] = py::float_(tk::Q_GRID[i]);
    m.attr("Q_GRID") = q_grid;
    m.def("select_q", &select_q_py, py::arg("mu"), py::arg("R"), py::arg("Pu"), py::arg("grid") = q_grid);

    m.def("_window_unknown", &window_unknown_py, py::arg("spec"), py::arg("grid_pos"), py::arg("ook_possible") = true,
          py::arg("reach") = cfg::SPAN);

    m.def("_smooth", [](const py::object& x, const py::object& n) {
        const std::vector<double> v = rx::vec<double>(x);
        return to_numpy(tk::smooth(v, rx::as_int(n)));
    }, py::arg("x"), py::arg("n"));

    m.def("_measurements", [](const py::object& tr, const py::object& kappa, const py::object& sf,
                              const py::object& pu_loc, const py::object& c, const py::object& meas) {
        TrackRead want;
        want.m = want.psi = want.N = true;
        want.c = c.is_none();
        want.meas = meas.is_none();
        const tk::TrackResult t = read_track(tr, no_spec(tr), want);
        std::optional<std::vector<cd>> cv;
        std::optional<std::vector<std::uint8_t>> mv;
        if (!c.is_none()) cv = rx::vec<cd>(c);
        if (!meas.is_none()) mv = rx::vec_bool(meas);
        const std::vector<double> kv = rx::vec<double>(kappa), sv = rx::vec<double>(sf), pv = rx::vec<double>(pu_loc);
        const tk::Measurements ms = tk::measurements(t, kv, sv, pv, cv ? &*cv : nullptr, mv ? &*mv : nullptr);
        return py::make_tuple(to_numpy(ms.mu), to_numpy(ms.R));
    }, py::arg("tr"), py::arg("kappa"), py::arg("sf"), py::arg("Pu_loc"), py::arg("c") = py::none(),
       py::arg("meas") = py::none());

    m.def("_pu_initial", [](const py::object& tr) {
        TrackRead want;
        want.c = want.psi = want.meas = want.m = want.N = true;
        return tk::pu_initial(read_track(tr, no_spec(tr), want));
    }, py::arg("tr"));

    m.def("_recenter", [](const py::object& tr) {
        TrackRead want;
        want.timing = want.freq = want.u = true;
        const tk::TrackResult t = read_track(tr, spec_of_track(tr), want);
        const tk::Recenter r = tk::recenter(t);
        if (!r.fn) return py::make_tuple(rx::field(tr, "freq"), r.df_max);
        return py::make_tuple(rx::to_py(*r.fn), r.df_max);
    }, py::arg("tr"));

    m.def("_keying_tuple", [](const py::object& cw_known) {
        const auto [k, ook] = keying_tuple(cw_known);
        return py::make_tuple(k, ook);
    }, py::arg("cw_known"));

    m.def("_lead_supported", [](const py::object& chan, const py::object& z_d, const py::object& tm,
                                const py::object& spec, const py::object& lead) {
        ensure_psd(chan);
        tk::ChanCapture c;
        c.t0_index = rx::as_double(rx::field(chan, "t0_index"));
        const py::object psd = rx::field(chan, "_psd");
        py::tuple pt = py::tuple(psd);
        if (pt.size() != 2) raise(PyExc_ValueError, "ChanCapture._psd must be (t_s, psd)");
        c.psd.set(std::make_shared<const tk::NoisePsd>(tk::NoisePsd{rx::vec<double>(pt[0]), rx::vec<double>(pt[1])}));
        const std::vector<cd> z = rx::vec<cd>(z_d);
        const types::Timing t = timing_lite(tm);
        const fr::FrameSpec s = rx::spec(spec);
        const double l = rx::as_double(lead);
        py::gil_scoped_release nogil;
        return tk::lead_supported(c, z, t, s, l);
    }, py::arg("chan"), py::arg("z_d"), py::arg("tm"), py::arg("spec"), py::arg("lead"));

    m.def("track", &track_py, py::arg("chan"), py::arg("spec"), py::arg("det"), py::arg("classes") = py::none(),
          py::arg("cw_known") = py::none(), py::arg("prior") = py::none(), py::kw_only(),
          py::arg("timing") = py::none(), py::arg("freq") = py::none(), py::arg("withhold_known") = false,
          py::arg("outer") = 2, py::arg("refine") = true, py::arg("lead_in_s") = py::none(),
          py::arg("z_d") = py::none());

    m.def("refine_timing", &refine_timing_py, py::arg("tr"), py::arg("iters") = 3, py::arg("u_ref") = py::none(),
          py::arg("P_ref") = py::none());

    m.def("z_ref_stat", [](const py::object& tr, const py::object& c_known, const py::object& d_known) {
        TrackRead want;
        want.known = want.psi = want.m = want.u = want.N = want.P = true;
        const tk::TrackResult t = read_track(tr, spec_of_track(tr), want);
        const std::vector<cd> ck = rx::vec<cd>(c_known);
        std::optional<std::vector<cd>> dk;
        if (!d_known.is_none()) dk = rx::vec<cd>(d_known);
        std::optional<std::span<const cd>> ds;
        if (dk) ds = std::span<const cd>(*dk);
        return tk::z_ref_stat(t, ck, ds);
    }, py::arg("tr"), py::arg("c_known"), py::arg("d_known") = py::none());

    m.def("erase", [](const py::object& tr, const py::object& spans) {
        if (!rx::as_bool(spans)) return;
        TrackRead want;
        want.timing = want.psi = true;
        tk::TrackResult t = read_track(tr, spec_of_track(tr), want);
        std::vector<std::pair<double, double>> sp;
        for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(spans))) {
            const auto [a, b] = unpack2(h);
            sp.emplace_back(rx::as_double(a), rx::as_double(b));
        }
        tk::erase(t, sp);
        py::object psi = to_numpy(t.psi);
        if (py::isinstance<py::dict>(tr))
            py::reinterpret_borrow<py::dict>(tr)["psi"] = psi;
        else
            py::setattr(tr, "psi", psi);
    }, py::arg("tr"), py::arg("spans"));

    m.def("report", [](const py::object& tr, const py::object& kappa, const py::object& suspect) {
        TrackRead want;
        want.timing = want.freq = want.u = want.pu = want.q = want.z_ref = true;
        const tk::TrackResult t = read_track(tr, spec_of_track(tr), want);
        const double k = rx::as_double(kappa);
        const bool s = rx::as_bool(suspect);
        const double nf = rx::as_double(rx::field(tr, "chan").attr("noise_floor")());
        return rx::to_py(tk::report(t, k, s, nf));
    }, py::arg("tr"), py::arg("kappa") = 1.0, py::arg("suspect") = false);
}

const Registrar reg("track", 75, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
