// sstvae_native.qrss.frontend: bindings of native/core/qrss/frontend (the
// receiver's front end, sstvae/qrss/frontend.py without the passband
// store, and live.StreamFE). Names, argument names and defaults (keyword-
// only where the reference's are), dtypes and error texts follow the
// reference, so tests/qrss_native/frontend.py can assign these over its
// functions:
//
//   audio_to_fe(x8k) -> complex64            fe_to_audio(fe) -> float64
//   fe_hz(audio_hz)                          FE_FLAT_DB FE_MIN_HZ FE_SKIRT_DB
//   running_median(p, fs, win_s=1.0, hop_s=0.05, chunk=256)   F_MIX_DEN
//   _interp_to_samples(n, centres, vals, chunk=1 << 20)       _FE_DECIM
//   blank(fe, k_sigma=5.0, guard_ms=2.0, win_s=1.0, hop_s=0.05, *,
//         fs=FE_FS, taper_ms=1.0)                             _FE_TAPS
//   normalise(fe, keep=None, win_ms=100.0, *, fs=FE_FS, hop_ms=10.0,
//             min_kept=0.25, chunk=512)                       ALIGN HALO
//   quantise_mix(f_mix_hz) -> Fraction
//   mix_phasor(n0, n, f_mix, fs_in) -> complex128
//   channelise(fe, fs_in, f_mix_hz, fs_out=CH_FS, *, n0=0) -> (complex64, Fraction)
//   channelise_keep(keep, fs_in=FE_FS, fs_out=CH_FS) -> float32
//   _noise_response(fs_in, fs_out, nfft)
//   _fold_power(fs_in, fs_out, nfft, band_hz) -> (idx, fk, P)
//   _fe_power(fs_in, fs_out, nfft, audio_hz, band_hz) -> (idx, num, den)
//   _fe_flat(..., band_hz=(85.0, 120.0)) -> bool, read-only
//   _fe_response(..., band_hz=(85.0, 120.0)) -> float64, read-only
//   noise_psd(ch, fs=CH_FS, exclude_hz=(), win_s=10.0, *, band_hz=(85.0, 120.0),
//             guard_hz=45.0, fs_in=FE_FS, audio_hz=None) -> float32
//   StreamFE(block=8000): .block .a0 .pend .next_index .push(a, x)
//
// Differences a caller can see: the `chunk` arguments only batch the
// reference's work, so they are accepted and ignored here (0 still raises
// the reference's ValueError where its range() would); the lru_caches have
// no cache_clear; integer rates (fs, fs_in, fs_out, nfft) must be ints.
// The long operations release the GIL; the rules (rules.EDGE_NOISE) are
// read while it is held.

#include "qrss/rx_types.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "qrss/config.hpp"
#include "qrss/frontend.hpp"
#include "qrss/rules.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace fe = sstvae::qrss::frontend;
namespace rules = sstvae::qrss::rules;
namespace cfg = sstvae::qrss::config;
using cf = std::complex<float>;
using cd = std::complex<double>;

[[noreturn]] void raise(PyObject* type, const std::string& what) {
    PyErr_SetString(type, what.c_str());
    throw py::error_already_set();
}

// range(0, n, chunk) refuses a zero step, whether or not it would run.
void check_chunk(std::int64_t chunk) {
    if (chunk == 0) raise(PyExc_ValueError, "range() arg 3 must not be zero");
}

// operator.index(x) as int64.
std::int64_t as_index(const py::handle& h) {
    py::object i = py::reinterpret_steal<py::object>(PyNumber_Index(h.ptr()));
    if (!i) throw py::error_already_set();
    const long long v = PyLong_AsLongLong(i.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

fe::Band band_of(const py::handle& o) {
    py::sequence s = py::reinterpret_borrow<py::sequence>(o);
    return {rx::as_double(s[0]), rx::as_double(s[1])};
}

py::array read_only(py::array a) {
    a.attr("setflags")(py::arg("write") = false);
    return a;
}

py::array_t<double> matrix(const std::vector<double>& v, std::size_t rows, std::size_t cols) {
    py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

std::string dtype_name(const py::handle& o) {
    return py::str(rx::np_module().attr("asarray")(o).attr("dtype")).cast<std::string>();
}

py::tuple running_median_py(const py::object& p, double fs, double win_s, double hop_s, std::int64_t chunk) {
    const auto pv = rx::vec<float>(p);
    fe::RunningMedian r;
    {
        py::gil_scoped_release nogil;
        r = fe::running_median(pv, fs, win_s, hop_s);
    }
    // the reference's chunked loop runs only for a stream longer than a window
    // (win_s * fs is finite here: the core has refused NaN and inf)
    const auto w = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::nearbyint(win_s * fs)));
    if (static_cast<std::int64_t>(pv.size()) > w) check_chunk(chunk);
    return py::make_tuple(to_numpy(r.centres), to_numpy(r.medians));
}

py::array interp_to_samples_py(const py::object& n, const py::object& centres, const py::object& vals,
                               std::int64_t chunk) {
    const std::int64_t nn = as_index(n);
    if (nn < 0) throw py::value_error("negative dimensions are not allowed");
    check_chunk(chunk);
    if (nn == 0) return to_numpy(std::vector<float>{});
    const auto c = rx::vec<double>(centres), v = rx::vec<double>(vals);
    return to_numpy(fe::interp_to_samples(nn, c, v));
}

py::tuple blank_py(const py::object& x, double k_sigma, double guard_ms, double win_s, double hop_s, double fs,
                   double taper_ms) {
    const auto v = rx::vec<cf>(x);
    fe::Blanked b;
    {
        py::gil_scoped_release nogil;
        b = fe::blank(v, k_sigma, guard_ms, win_s, hop_s, fs, taper_ms);
    }
    return py::make_tuple(to_numpy(b.fe), to_numpy(b.keep));
}

py::tuple normalise_py(const py::object& x, const py::object& keep, double win_ms, double fs, double hop_ms,
                       double min_kept, std::int64_t chunk) {
    const auto v = rx::vec<cf>(x);
    std::optional<std::vector<float>> k;
    if (!v.empty() && !keep.is_none()) {
        // the reference's `np.asarray(keep) < 1.0`, in keep's own dtype: kept
        // samples as 1, the rest as 0
        py::object lt = rx::np_module().attr("asarray")(keep).attr("__lt__")(1.0);
        const auto mask = rx::vec_bool(lt);
        k.emplace(mask.size());
        for (std::size_t i = 0; i < mask.size(); ++i) (*k)[i] = mask[i] ? 0.0f : 1.0f;
    }
    fe::Normalised r;
    {
        py::gil_scoped_release nogil;
        std::optional<std::span<const float>> ks;
        if (k) ks = std::span<const float>(*k);
        r = fe::normalise(v, ks, win_ms, fs, hop_ms, min_kept);
    }
    if (!v.empty()) check_chunk(chunk);
    if (v.empty()) r.level.clear();
    return py::make_tuple(to_numpy(r.fe), to_numpy(r.level));
}

py::object fraction_of(const py::handle& o) { return rx::fraction_class()(o); }

py::array mix_phasor_py(const py::object& n0, const py::object& n, const py::object& f_mix, const py::object& fs_in) {
    const sstvae::qrss::Rational fm = rx::rational(fraction_of(f_mix));
    const std::int64_t fs = rx::as_int(fs_in);
    const std::int64_t i0 = rx::as_int(n0);
    const std::int64_t nn = as_index(n);
    return to_numpy(fe::mix_phasor(i0, nn, fm, fs));
}

py::tuple channelise_py(const py::object& x, const py::object& fs_in, const py::object& f_mix_hz,
                        const py::object& fs_out, const py::object& n0) {
    const std::int64_t fi = rx::as_int(fs_in), fo = rx::as_int(fs_out);
    // the reference's order: the rates, then quantise_mix (float(f_mix_hz))
    if (fo == 0) throw sstvae::qrss::ZeroDivision("integer modulo by zero");
    if (fi % fo != 0)   // Python's fs_in % fs_out is nonzero exactly when C's is
        throw py::value_error("fs_out " + std::to_string(fo) + " must divide fs_in " + std::to_string(fi));
    const double f = rx::as_double(f_mix_hz);
    const std::int64_t i0 = rx::as_int(n0);
    fe::Channelised c;
    if (dtype_name(x) == "complex64") {
        const auto v = rx::vec<cf>(x);
        py::gil_scoped_release nogil;
        c = fe::channelise(std::span<const cf>(v), fi, f, fo, i0);
    } else {
        const auto v = rx::vec<cd>(x);
        py::gil_scoped_release nogil;
        c = fe::channelise(std::span<const cd>(v), fi, f, fo, i0);
    }
    return py::make_tuple(to_numpy(c.ch), rx::to_py(c.f_mix));
}

py::array channelise_keep_py(const py::object& keep, const py::object& fs_in, const py::object& fs_out) {
    const std::int64_t fi = rx::as_int(fs_in), fo = rx::as_int(fs_out);
    const auto k = rx::vec<double>(keep);
    std::vector<float> out;
    {
        py::gil_scoped_release nogil;
        out = fe::channelise_keep(std::span<const double>(k), fi, fo);
    }
    return to_numpy(out);
}

// A sample rate: an int, or a float holding one (the reference's arithmetic
// with an integral float is its arithmetic with the int). TypeError for a
// fractional rate, which the C++ does not take.
std::int64_t int_rate(const py::handle& o, const char* name) {
    if (py::isinstance<py::float_>(o)) {
        const double d = o.cast<double>();
        if (std::isfinite(d) && d == std::trunc(d) && std::fabs(d) < 9.0e15) return static_cast<std::int64_t>(d);
        throw py::type_error(std::string(name) + " must be a whole number of Hz, not " +
                             py::repr(o).cast<std::string>());
    }
    return as_index(o);
}

py::array noise_psd_py(const py::object& ch, const py::object& fs_obj, const py::object& exclude_hz, double win_s,
                       const py::object& band_hz, double guard_hz, const py::object& fs_in_obj,
                       const py::object& audio_hz) {
    const std::int64_t fs = int_rate(fs_obj, "fs"), fs_in = int_rate(fs_in_obj, "fs_in");
    const auto v = rx::vec<cf>(ch);
    std::vector<double> ex;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(exclude_hz)) ex.push_back(rx::as_double(h));
    fe::NoisePsdOptions opt;
    opt.band_hz = band_of(band_hz);
    opt.guard_hz = guard_hz;
    opt.fs_in = fs_in;
    if (!audio_hz.is_none()) opt.audio_hz = rx::as_double(audio_hz);
    opt.old_edge_noise = rules::old(rules::EDGE_NOISE);   // read under the GIL
    std::vector<float> out;
    {
        py::gil_scoped_release nogil;
        out = fe::noise_psd(v, fs, ex, win_s, opt);
    }
    return to_numpy(out);
}

void bind(py::module_& qrss) {
    rx::register_translators();
    py::module_ m = qrss.def_submodule("frontend", "native/core/qrss/frontend (sstvae/qrss/frontend.py, live.StreamFE)");

    m.attr("FE_FLAT_DB") = fe::FE_FLAT_DB;
    m.attr("FE_MIN_HZ") = fe::FE_MIN_HZ;
    m.attr("FE_SKIRT_DB") = fe::FE_SKIRT_DB;
    m.attr("F_MIX_DEN") = fe::F_MIX_DEN;
    m.attr("_FE_DECIM") = fe::FE_DECIM;
    m.attr("_FE_TAPS") = read_only(to_numpy(fe::fe_taps()));
    m.attr("ALIGN") = fe::ALIGN;
    m.attr("HALO") = fe::HALO;

    m.def("audio_to_fe", [](const py::object& x8k) {
        const auto x = rx::vec<double>(x8k);
        std::vector<cf> out;
        {
            py::gil_scoped_release nogil;
            out = fe::audio_to_fe(x);
        }
        return to_numpy(out);
    }, py::arg("x8k"));
    m.def("fe_to_audio", [](const py::object& x) {
        const auto v = rx::vec<cd>(x);
        std::vector<double> out;
        {
            py::gil_scoped_release nogil;
            out = fe::fe_to_audio(v);
        }
        return to_numpy(out);
    }, py::arg("fe"));
    m.def("fe_hz", [](const py::object& audio_hz) { return fe::fe_hz(rx::as_double(audio_hz)); }, py::arg("audio_hz"));

    m.def("running_median", &running_median_py, py::arg("p"), py::arg("fs"), py::arg("win_s") = 1.0,
          py::arg("hop_s") = 0.05, py::arg("chunk") = 256);
    m.def("_interp_to_samples", &interp_to_samples_py, py::arg("n"), py::arg("centres"), py::arg("vals"),
          py::arg("chunk") = std::int64_t{1} << 20);
    m.def("blank", &blank_py, py::arg("fe"), py::arg("k_sigma") = 5.0, py::arg("guard_ms") = 2.0,
          py::arg("win_s") = 1.0, py::arg("hop_s") = 0.05, py::kw_only(), py::arg("fs") = static_cast<double>(cfg::FE_FS),
          py::arg("taper_ms") = 1.0);
    m.def("normalise", &normalise_py, py::arg("fe"), py::arg("keep") = py::none(), py::arg("win_ms") = 100.0,
          py::kw_only(), py::arg("fs") = static_cast<double>(cfg::FE_FS), py::arg("hop_ms") = 10.0,
          py::arg("min_kept") = 0.25, py::arg("chunk") = 512);

    m.def("quantise_mix", [](const py::object& f) { return rx::to_py(fe::quantise_mix(rx::as_double(f))); },
          py::arg("f_mix_hz"));
    m.def("mix_phasor", &mix_phasor_py, py::arg("n0"), py::arg("n"), py::arg("f_mix"), py::arg("fs_in"));
    m.def("channelise", &channelise_py, py::arg("fe"), py::arg("fs_in"), py::arg("f_mix_hz"),
          py::arg("fs_out") = cfg::CH_FS, py::kw_only(), py::arg("n0") = 0);
    m.def("channelise_keep", &channelise_keep_py, py::arg("keep"), py::arg("fs_in") = cfg::FE_FS,
          py::arg("fs_out") = cfg::CH_FS);

    m.def("_noise_response", [](std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft) {
        return to_numpy(*fe::noise_response(fs_in, fs_out, nfft));
    }, py::arg("fs_in"), py::arg("fs_out"), py::arg("nfft"));
    m.def("_fold_power", [](std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, const py::object& band) {
        const auto r = fe::fold_power(fs_in, fs_out, nfft, band_of(band));
        return py::make_tuple(to_numpy(r->idx), matrix(r->fk, r->n_fold, r->idx.size()),
                              matrix(r->P, r->n_fold, r->idx.size()));
    }, py::arg("fs_in"), py::arg("fs_out"), py::arg("nfft"), py::arg("band_hz"));
    m.def("_fe_power", [](std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, double audio_hz,
                          const py::object& band) {
        const auto r = fe::fe_power(fs_in, fs_out, nfft, audio_hz, band_of(band));
        return py::make_tuple(to_numpy(r->idx), to_numpy(r->num), to_numpy(r->den));
    }, py::arg("fs_in"), py::arg("fs_out"), py::arg("nfft"), py::arg("audio_hz"), py::arg("band_hz"));
    m.def("_fe_flat", [](std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, double audio_hz,
                         const py::object& band) {
        return read_only(rx::arr_bool(fe::fe_flat(fs_in, fs_out, nfft, audio_hz, band_of(band))));
    }, py::arg("fs_in"), py::arg("fs_out"), py::arg("nfft"), py::arg("audio_hz"),
          py::arg("band_hz") = py::make_tuple(fe::NOISE_BAND.first, fe::NOISE_BAND.second));
    m.def("_fe_response", [](std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, double audio_hz,
                             const py::object& band) {
        return read_only(to_numpy(fe::fe_response(fs_in, fs_out, nfft, audio_hz, band_of(band))));
    }, py::arg("fs_in"), py::arg("fs_out"), py::arg("nfft"), py::arg("audio_hz"),
          py::arg("band_hz") = py::make_tuple(fe::NOISE_BAND.first, fe::NOISE_BAND.second));

    m.def("noise_psd", &noise_psd_py, py::arg("ch"), py::arg("fs") = cfg::CH_FS, py::arg("exclude_hz") = py::tuple(),
          py::arg("win_s") = 10.0, py::kw_only(),
          py::arg("band_hz") = py::make_tuple(fe::NOISE_BAND.first, fe::NOISE_BAND.second),
          py::arg("guard_hz") = 45.0, py::arg("fs_in") = cfg::FE_FS, py::arg("audio_hz") = py::none());

    py::class_<fe::StreamFE>(m, "StreamFE", "live.StreamFE: audio_to_fe over a stream of chunks")
        .def(py::init<std::int64_t>(), py::arg("block") = 8000)
        .def_property_readonly("block", &fe::StreamFE::block)
        .def_property_readonly("a0", [](const fe::StreamFE& s) {
            return s.a0() ? py::object(py::int_(*s.a0())) : py::object(py::none());
        })
        .def_property_readonly("pend", [](const fe::StreamFE& s) { return to_numpy(s.pending()); })
        .def_property_readonly("next_index", [](const fe::StreamFE& s) {
            const auto n = s.next_index();
            return n ? py::object(py::int_(*n)) : py::object(py::none());
        })
        .def("push", [](fe::StreamFE& s, std::int64_t a, const py::object& x) {
            const auto v = rx::vec<double>(x);
            const auto chunks = s.push(a, v);
            py::list out;
            for (const auto& c : chunks) out.append(py::make_tuple(c.index, to_numpy(c.fe)));
            return out;
        }, py::arg("a"), py::arg("x"));
}

const Registrar reg("frontend", 40, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
