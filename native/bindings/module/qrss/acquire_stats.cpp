// sstvae_native.qrss.acquire_stats: native/core/qrss/acquire_stats, the
// pure-array statistics of sstvae/qrss/acquire.py, under the reference's
// names and signatures (tests/qrss_native/acquire_stats.py assigns them
// over acquire's attributes):
//
//   _frame_len(fs), frame_window(fs)
//   spectrogram(x, fs, starts, bins=None, chunk=32)  float32[n_frames, n_bins]
//   band_bins(fs, band_hz=BAND_HZ, centre_hz=FE_CENTER_HZ) -> (sel, f)
//   cfar_normalise(P, local_bins=CFAR_LOCAL_BINS)    float32
//   spectral_shape(Pn, local_bins=CFAR_LOCAL_BINS, rows=None)
//   _parabola(ym, y0, yp)        float32 arithmetic when numpy's would be
//   a1_gamma(M, fs=FE_FS), a1_threshold(M, pfa=A1_PFA, fs=FE_FS)
//   a2_threshold(pfa=A2_PFA, n_chunks=N_PRE // A2_CHUNK, n_scales=4)
//   preamble_scales(S, var, scales=A2_SCALES) -> [(k, Lambda_k)]
//   preamble_p(S, var, scales=A2_SCALES) -> (p, best, [(k, Lambda_k)])
//   tbd_gumbel(n_frames), gumbel_p(z, n_frames)
//   _band_cov(n_bins, fs=FE_FS)
//
// Only the shapes the receiver uses are ported (a 1-D x and starts, a
// 2-D P, S of [n_df, n_tau, n_chunk] with var [n_tau, n_chunk]); others
// raise NotImplementedError rather than computing something else.

#include "qrss/rx_types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "qrss/acquire_stats.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace aq = sstvae::qrss::acquire;
namespace rxa = sstvae::qrss::rxconfig::acquire;
namespace cfg = sstvae::qrss::config;

[[noreturn]] void not_implemented(const std::string& what) {
    PyErr_SetString(PyExc_NotImplementedError, ("native acquire_stats: " + what).c_str());
    throw py::error_already_set();
}

py::array as_array(const py::object& o) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o));
    if (!a) throw py::error_already_set();
    return a;
}

std::string dtype_of(const py::array& a) { return py::str(a.attr("dtype")).cast<std::string>(); }

template <typename T>
py::array grid_to_py(const aq::Grid<T>& g) {
    py::array_t<T> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(g.rows), static_cast<py::ssize_t>(g.cols)});
    std::copy(g.data.begin(), g.data.end(), out.mutable_data());
    return out;
}

template <typename T>
aq::Grid<T> grid_from_py(const py::array& a) {
    aq::Grid<T> g;
    g.rows = static_cast<std::size_t>(a.shape(0));
    g.cols = static_cast<std::size_t>(a.shape(1));
    g.data = rx::vec<T>(a);
    return g;
}

py::array matrix_2d(const py::object& o, const char* what) {
    py::array a = as_array(o);
    if (a.ndim() != 2) not_implemented(std::string(what) + " must be 2-D");
    return a;
}

py::object spectrogram(const py::object& x, double fs, const py::object& starts, const py::object& bins,
                       const py::object& chunk_o) {
    py::array xa = as_array(x);
    if (xa.ndim() != 1) not_implemented("x must be 1-D");
    (void)aq::frame_window(fs);   // a NaN/inf/negative fs fails first, as _frame_len/frame_window do
    py::array sa = py::array::ensure(rx::np_module().attr("asarray")(starts, py::arg("dtype") = "float64"));
    if (!sa) throw py::error_already_set();
    std::optional<std::vector<std::int64_t>> bv;
    if (!bins.is_none()) {
        (void)py::len(bins);                     // nb = len(bins): its TypeError, in its words
        py::array ba = as_array(bins);
        if (ba.ndim() != 1) not_implemented("bins must be 1-D");
        if (dtype_of(ba) == "bool") not_implemented("a boolean bins mask");
        bv = rx::vec<std::int64_t>(ba);
    }
    // np.round of a 0-d array is a numpy scalar, so len(starts) is asked
    // of an np.int64
    if (sa.ndim() == 0) throw py::type_error("object of type 'numpy.int64' has no len()");
    if (sa.ndim() != 1) not_implemented("starts must be 1-D");
    const std::vector<double> st = rx::vec<double>(sa);
    // the reference's chunk loop, range(0, len(starts), chunk): 0 is
    // range()'s refusal; a negative step would leave its np.empty output
    // unwritten, which there is no defined answer to copy.
    const std::int64_t chunk = rx::as_int(chunk_o);
    if (chunk == 0) throw py::value_error("range() arg 3 must not be zero");
    if (chunk < 0 && !st.empty()) not_implemented("a negative chunk");
    std::optional<std::span<const std::int64_t>> bs;
    if (bv) bs = std::span<const std::int64_t>(*bv);
    if (dtype_of(xa) == "complex64") {
        const std::vector<std::complex<float>> xv = rx::vec<std::complex<float>>(xa);
        return grid_to_py(aq::spectrogram(std::span<const std::complex<float>>(xv), fs, st, bs));
    }
    if (xa.dtype().kind() == 'c') {
        const std::vector<std::complex<double>> xv = rx::vec<std::complex<double>>(xa);
        return grid_to_py(aq::spectrogram(std::span<const std::complex<double>>(xv), fs, st, bs));
    }
    // Real input: numpy windows it as a real product and fft promotes the
    // result to complex (v*w + 0j), which is not the complex product an
    // inf or NaN sample would get through the complex overload.
    const std::vector<double> xv = rx::vec<double>(xa);
    return grid_to_py(aq::spectrogram(std::span<const double>(xv), fs, st, bs));
}

// local_bins as an int: the reference only ever uses it as int(lb) | 1
// against a window of at most n_bins, so a value past int's range clamps
// to INT_MAX (odd, so the min picks the same window) or INT_MIN (still
// negative: the same refusal) rather than wrapping.
int bins_int(const py::object& o) {
    const std::int64_t v = rx::as_int(o);
    return static_cast<int>(std::clamp<std::int64_t>(v, std::numeric_limits<int>::min(),
                                                     std::numeric_limits<int>::max()));
}

// cfar_normalise's `if local_bins:` is Python truth, then int(lb) | 1: a
// true value whose int is 0 (0.5) is the window of 1.
std::optional<int> opt_bins(const py::object& o) {
    if (!rx::as_bool(o)) return std::nullopt;
    const int v = bins_int(o);
    return v == 0 ? 1 : v;
}

py::object cfar_normalise(const py::object& P, const py::object& local_bins) {
    py::array a = matrix_2d(P, "P");
    if (dtype_of(a) == "float32") return grid_to_py(aq::cfar_normalise(grid_from_py<float>(a), opt_bins(local_bins)));
    return grid_to_py(aq::cfar_normalise(grid_from_py<double>(a), opt_bins(local_bins)));
}

py::object spectral_shape(const py::object& Pn, const py::object& local_bins_o, const py::object& rows) {
    const int local_bins = bins_int(local_bins_o);
    py::array a = matrix_2d(Pn, "Pn");
    std::optional<std::vector<std::uint8_t>> rv;
    if (!rows.is_none()) rv = rx::vec_bool(rows);
    std::optional<std::span<const std::uint8_t>> rs;
    if (rv) rs = std::span<const std::uint8_t>(*rv);
    if (dtype_of(a) == "float32") return to_numpy(aq::spectral_shape(grid_from_py<float>(a), local_bins, rs));
    return to_numpy(aq::spectral_shape(grid_from_py<double>(a), local_bins, rs));
}

// _parabola's arguments as numpy types them (NEP 50): a Python float or
// int is "weak" and takes the other operand's type; np.float32 against a
// weak value or another float32 stays float32; anything with np.float64
// is float64. The receiver passes three float32s (carrier_lines) or three
// float64s, which are aq::parabola's two overloads; a mixed call is
// evaluated here operation by operation, each in the type numpy gives it.
struct Typed {
    double v;
    enum Kind { Weak, F32, F64 } kind;
};

Typed typed(const py::object& o) {
    py::module_ np = rx::np_module();
    if (py::isinstance(o, np.attr("ndarray")) || py::hasattr(o, "dtype")) {
        const std::string dt = py::str(o.attr("dtype")).cast<std::string>();
        if (dt == "float32") return {rx::as_double(o), Typed::F32};
        return {rx::as_double(o), Typed::F64};
    }
    return {rx::as_double(o), Typed::Weak};
}

Typed::Kind join(Typed::Kind a, Typed::Kind b) {
    if (a == Typed::F64 || b == Typed::F64) return Typed::F64;
    if (a == Typed::F32 || b == Typed::F32) return Typed::F32;
    return Typed::Weak;
}

template <typename Op>
Typed apply(Typed a, Typed b, Op op) {
    const Typed::Kind k = join(a.kind, b.kind);
    if (k == Typed::F32)
        return {static_cast<double>(op(static_cast<float>(a.v), static_cast<float>(b.v))), k};
    return {op(a.v, b.v), k};
}

double parabola(const py::object& ym_o, const py::object& y0_o, const py::object& yp_o) {
    const Typed ym = typed(ym_o), y0 = typed(y0_o), yp = typed(yp_o);
    if (ym.kind == Typed::F32 && y0.kind == Typed::F32 && yp.kind == Typed::F32)
        return aq::parabola(static_cast<float>(ym.v), static_cast<float>(y0.v), static_cast<float>(yp.v));
    if (ym.kind != Typed::F32 && y0.kind != Typed::F32 && yp.kind != Typed::F32)
        return aq::parabola(ym.v, y0.v, yp.v);
    auto add = [](auto x, auto y) { return x + y; };
    auto sub = [](auto x, auto y) { return x - y; };
    auto mul = [](auto x, auto y) { return x * y; };
    auto div = [](auto x, auto y) { return x / y; };
    const Typed two{2.0, Typed::Weak}, half{0.5, Typed::Weak};
    const Typed den = apply(apply(ym, apply(two, y0, mul), sub), yp, add);
    if (!std::isfinite(den.v) || den.v >= 0) return 0.0;
    const Typed v = apply(apply(half, apply(ym, yp, sub), mul), den, div);
    if (std::isnan(v.v)) return v.v;
    return std::min(std::max(v.v, -0.5), 0.5);
}

struct SVar {
    std::vector<std::complex<double>> S;
    std::vector<double> var;
    std::size_t n_df, n_tau, nc;
    aq::SumOrder order;
};

// The order numpy adds preamble_scales' runs and groups in (aq::SumOrder):
// pairwise when Sn = S / np.sqrt(var)[None] is allocated with its chunk
// axis innermost, one term at a time otherwise. numpy's iterator chooses
// that layout from the operands' strides alone, so it is read off the
// same division on the 2x2x2 corner of S (and 2x2 of var), which keeps
// their strides, dtypes and broadcasting; an axis of length 1 has no say.
aq::SumOrder sum_order(const py::object& S, const py::object& var) {
    py::module_ np = rx::np_module();
    py::array s = py::array::ensure(np.attr("asarray")(S));
    py::array v = py::array::ensure(np.attr("asarray")(var));
    if (!s || !v) throw py::error_already_set();
    const py::slice c2(0, 2, 1);
    py::object d = np.attr("sqrt")(v[py::make_tuple(c2, c2)]);
    py::array sn = py::array::ensure(np.attr("true_divide")(s[py::make_tuple(c2, c2, c2)],
                                                            d[py::make_tuple(py::none(), py::ellipsis())]));
    if (!sn) throw py::error_already_set();
    if (s.shape(2) < 2) return aq::SumOrder::Pairwise;   // one chunk: nothing to add
    const auto inner = std::abs(sn.strides(2));
    for (py::ssize_t a = 0; a < 2; ++a)
        if (s.shape(a) > 1 && std::abs(sn.strides(a)) < inner) return aq::SumOrder::Sequential;
    return aq::SumOrder::Pairwise;
}

SVar s_var(const py::object& S, const py::object& var) {
    py::array sa = py::array::ensure(rx::np_module().attr("asarray")(S, py::arg("dtype") = "complex128"));
    py::array va = py::array::ensure(rx::np_module().attr("asarray")(var, py::arg("dtype") = "float64"));
    if (!sa || !va) throw py::error_already_set();
    if (sa.ndim() != 3 || va.ndim() != 2 || va.shape(0) != sa.shape(1) || va.shape(1) != sa.shape(2))
        not_implemented("S must be [n_df, n_tau, n_chunk] and var [n_tau, n_chunk]");
    return {rx::vec<std::complex<double>>(sa), rx::vec<double>(va), static_cast<std::size_t>(sa.shape(0)),
            static_cast<std::size_t>(sa.shape(1)), static_cast<std::size_t>(sa.shape(2)), sum_order(S, var)};
}

std::vector<int> scale_list(const py::object& scales) {
    std::vector<int> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(scales)) out.push_back(static_cast<int>(rx::as_int(h)));
    return out;
}

py::list scales_to_py(const std::vector<aq::Scale>& st, std::size_t n_df, std::size_t n_tau) {
    py::list out;
    for (const aq::Scale& s : st) {
        py::array lam = to_numpy(s.lam).attr("reshape")(py::make_tuple(n_df, n_tau));
        out.append(py::make_tuple(s.k, lam));
    }
    return out;
}

py::tuple pair(const std::pair<double, double>& p) { return py::make_tuple(p.first, p.second); }

// a1_gamma's (M * M / var, var / M) are np.float64 in Python (var is one);
// tbd_gumbel's and _band_cov's are float(...)'d there.
py::tuple np_pair(const std::pair<double, double>& p) {
    const py::object f64 = rx::np_module().attr("float64");
    return py::make_tuple(f64(p.first), f64(p.second));
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("acquire_stats", "acquire.py's shared statistics (native/core/qrss/acquire_stats)");

    m.def("_frame_len", [](double fs) { return aq::frame_len(fs); }, py::arg("fs"));
    m.def("frame_window", [](double fs) { return to_numpy(aq::frame_window(fs)); }, py::arg("fs"));
    m.def("spectrogram", &spectrogram, py::arg("x"), py::arg("fs"), py::arg("starts"), py::arg("bins") = py::none(),
          py::arg("chunk") = 32);
    m.def("band_bins", [](double fs, const py::object& band_hz, double centre_hz) {
        const std::vector<double> b = rx::vec<double>(band_hz);
        if (b.size() != 2) throw py::value_error("band_hz must be (lo, hi)");
        const aq::BandBins r = aq::band_bins(fs, {b[0], b[1]}, centre_hz);
        return py::make_tuple(to_numpy(r.sel), to_numpy(r.f_hz));
    }, py::arg("fs"), py::arg("band_hz") = py::make_tuple(rxa::BAND_HZ[0], rxa::BAND_HZ[1]),
       py::arg("centre_hz") = static_cast<double>(cfg::FE_CENTER_HZ));
    m.def("cfar_normalise", &cfar_normalise, py::arg("P"), py::arg("local_bins") = rxa::CFAR_LOCAL_BINS);
    m.def("spectral_shape", &spectral_shape, py::arg("Pn"), py::arg("local_bins") = rxa::CFAR_LOCAL_BINS,
          py::arg("rows") = py::none());
    m.def("_parabola", &parabola, py::arg("ym"), py::arg("y0"), py::arg("yp"));

    m.def("a1_gamma", [](std::int64_t M, double fs) { return np_pair(aq::a1_gamma(M, fs)); }, py::arg("M"),
          py::arg("fs") = static_cast<double>(cfg::FE_FS));
    m.def("a1_threshold", [](std::int64_t M, double pfa, double fs) { return aq::a1_threshold(M, pfa, fs); },
          py::arg("M"), py::arg("pfa") = rxa::A1_PFA, py::arg("fs") = static_cast<double>(cfg::FE_FS));
    m.def("a2_threshold", [](double pfa, std::int64_t n_chunks, std::int64_t n_scales) {
        return aq::a2_threshold(pfa, n_chunks, n_scales);
    }, py::arg("pfa") = rxa::A2_PFA, py::arg("n_chunks") = cfg::N_PRE / rxa::A2_CHUNK,
       py::arg("n_scales") = static_cast<std::int64_t>(rxa::A2_SCALES.size()));

    const py::tuple default_scales = py::make_tuple(rxa::A2_SCALES[0], rxa::A2_SCALES[1], rxa::A2_SCALES[2],
                                                    rxa::A2_SCALES[3]);
    m.def("preamble_scales", [](const py::object& S, const py::object& var, const py::object& scales) {
        const SVar sv = s_var(S, var);
        const std::vector<int> sc = scale_list(scales);
        return scales_to_py(aq::preamble_scales(sv.S, sv.n_df, sv.n_tau, sv.nc, sv.var, sc, sv.order), sv.n_df,
                            sv.n_tau);
    }, py::arg("S"), py::arg("var"), py::arg("scales") = default_scales);
    m.def("preamble_p", [](const py::object& S, const py::object& var, const py::object& scales) {
        const SVar sv = s_var(S, var);
        const std::vector<int> sc = scale_list(scales);
        const aq::PreambleP r = aq::preamble_p(sv.S, sv.n_df, sv.n_tau, sv.nc, sv.var, sc, sv.order);
        const py::tuple shape = py::make_tuple(sv.n_df, sv.n_tau);
        py::array p = to_numpy(r.p).attr("reshape")(shape);
        py::array best = to_numpy(r.best).attr("astype")(rx::np_module().attr("intp")).attr("reshape")(shape);
        return py::make_tuple(p, best, scales_to_py(r.st, sv.n_df, sv.n_tau));
    }, py::arg("S"), py::arg("var"), py::arg("scales") = default_scales);

    m.def("tbd_gumbel", [](std::int64_t n) { return pair(aq::tbd_gumbel(n)); }, py::arg("n_frames"));
    m.def("gumbel_p", [](double z, std::int64_t n) { return aq::gumbel_p(z, n); }, py::arg("z"), py::arg("n_frames"));
    m.def("_band_cov", [](std::int64_t n, double fs) { return pair(aq::band_cov(n, fs)); }, py::arg("n_bins"),
          py::arg("fs") = static_cast<double>(cfg::FE_FS));
}

const Registrar reg("acquire_stats", 40, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
