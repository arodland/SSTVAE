// Conversions between the QRSSTVAE receiver's records in Python and in C++
// (native/core/qrss/types.hpp, trackbase.hpp), for every receive-side
// binding: header-only, so each lane's bindings/module/qrss/<module>.cpp
// includes it and nothing else needs editing.
//
// Python -> C++ (`timing(obj)`, `pass_result(obj)`, `track_result(obj)`,
// ...) reads any object with the reference dataclass's attributes, or a
// dict with its field names as keys (what the C++ -> Python direction
// produces, so a value round-trips without the reference installed).
// Arrays are anything numpy converts (np.asarray with the field's dtype),
// None stays None where the reference allows it, a Fraction (or anything
// with numerator and denominator, an int, or a float as Fraction(float))
// becomes a Rational. A PathFn is the native object itself, a wrapper
// carrying one as `_native` (tests/qrss_native/_rx.py's NativePathFn), or
// the reference's PathFn read without refitting: lo, hi, _line and _spl's
// (t, c, k).
//
// C++ -> Python (`to_py(...)`) gives plain nested dicts with a "__type__"
// key naming the reference class ("Timing", "PassResult", "TrackResult",
// "LTrackResult", "HeaderFields", ...); numpy arrays of the reference's
// dtypes; fractions.Fraction; the native FrameSpec and PathFn objects.
// tests/qrss_native/_rx.py's `to_ref` turns such a tree back into the
// reference's dataclasses.
//
// Errors raise the reference's Python types: register_translators()
// (called by every binding that uses this header; idempotent) maps
// math::LinAlgError to numpy.linalg.LinAlgError, math::TypeError to
// TypeError and qrss::ZeroDivision to ZeroDivisionError; pybind11 already
// maps std::invalid_argument, std::out_of_range and std::overflow_error to
// ValueError, IndexError and OverflowError.

#pragma once

#include "qrss/registry.hpp"

#include <pybind11/complex.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "math/bspline.hpp"
#include "math/errors.hpp"
#include "math/linalg.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/rational.hpp"
#include "qrss/trackbase.hpp"
#include "qrss/types.hpp"

namespace sstvae_bindings::qrss::rx {

namespace types = sstvae::qrss::types;
namespace track = sstvae::qrss::track;
namespace frame = sstvae::qrss::frame;
namespace header = sstvae::qrss::header;
namespace smath = sstvae::math;
using sstvae::qrss::Rational;

// --- modules (imported once, never released: see frame.cpp's numpy()) -------

inline py::module_ np_module() {
    static py::module_* m = new py::module_(py::module_::import("numpy"));
    return *m;
}

inline py::object fraction_class() {
    static py::object* c = new py::object(py::module_::import("fractions").attr("Fraction"));
    return *c;
}

inline void register_translators() {
    static const bool done = [] {
        py::register_local_exception_translator([](std::exception_ptr p) {
            try {
                if (p) std::rethrow_exception(p);
            } catch (const smath::LinAlgError& e) {
                static py::object* cls =
                    new py::object(py::module_::import("numpy.linalg").attr("LinAlgError"));
                PyErr_SetString(cls->ptr(), e.what());
            } catch (const smath::TypeError& e) {
                PyErr_SetString(PyExc_TypeError, e.what());
            } catch (const sstvae::qrss::ZeroDivision& e) {
                PyErr_SetString(PyExc_ZeroDivisionError, e.what());
            }
        });
        return true;
    }();
    (void)done;
}

// --- scalars ------------------------------------------------------------------

// obj[name] for a dict, else getattr(obj, name).
inline py::object field(const py::handle& obj, const char* name) {
    if (py::isinstance<py::dict>(obj)) {
        py::dict d = py::reinterpret_borrow<py::dict>(obj);
        if (!d.contains(name)) throw py::key_error(name);
        return d[name];
    }
    return obj.attr(name);
}

inline bool has_field(const py::handle& obj, const char* name) {
    if (py::isinstance<py::dict>(obj)) return py::reinterpret_borrow<py::dict>(obj).contains(name);
    return py::hasattr(obj, name);
}

// float(o)
inline double as_double(const py::handle& o) {
    py::object f = py::reinterpret_steal<py::object>(PyNumber_Float(o.ptr()));
    if (!f) throw py::error_already_set();
    return PyFloat_AsDouble(f.ptr());
}

// int(o), int64 (OverflowError past it)
inline std::int64_t as_int(const py::handle& o) {
    py::object i = py::reinterpret_steal<py::object>(PyNumber_Long(o.ptr()));
    if (!i) throw py::error_already_set();
    const long long v = PyLong_AsLongLong(i.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

// bool(o)
inline bool as_bool(const py::handle& o) {
    const int t = PyObject_IsTrue(o.ptr());
    if (t < 0) throw py::error_already_set();
    return t != 0;
}

// A str as UTF-8, lone surrogates passed through.
inline std::string as_string(const py::handle& o) {
    if (!py::isinstance<py::str>(o)) throw py::type_error("expected a str, got " + std::string(py::str(py::type::of(o))));
    return std::string(py::bytes(o.attr("encode")("utf-8", "surrogatepass")));
}

inline py::str to_str(std::string_view s) {
    PyObject* o = PyUnicode_DecodeUTF8(s.data(), static_cast<py::ssize_t>(s.size()), "surrogatepass");
    if (!o) throw py::error_already_set();
    return py::reinterpret_steal<py::str>(o);
}

inline std::optional<std::string> opt_string(const py::handle& o) {
    if (o.is_none()) return std::nullopt;
    return as_string(o);
}

inline py::object to_py(const std::optional<std::string>& s) {
    return s ? py::object(to_str(*s)) : py::object(py::none());
}

// --- arrays -------------------------------------------------------------------

// np.asarray(o, dtype).ravel() as a vector.
template <typename T>
std::vector<T> vec(const py::handle& o) {
    py::object a = np_module().attr("asarray")(o, py::arg("dtype") = py::dtype::of<T>());
    return from_numpy<T>(a);
}

// np.asarray(o, dtype=bool) as 0/1 bytes.
inline std::vector<std::uint8_t> vec_bool(const py::handle& o) {
    py::object a = np_module().attr("asarray")(o, py::arg("dtype") = "bool");
    return from_numpy<std::uint8_t>(a.attr("view")("uint8"));
}

template <typename T>
std::optional<std::vector<T>> opt_vec(const py::handle& o) {
    if (o.is_none()) return std::nullopt;
    return vec<T>(o);
}

// None -> empty (the tracker's not-yet-filled arrays).
template <typename T>
std::vector<T> vec_or_empty(const py::handle& o) {
    if (o.is_none()) return {};
    return vec<T>(o);
}

template <typename T>
py::array arr(const std::vector<T>& v) {
    return to_numpy(v);
}

inline py::array arr_bool(const std::vector<std::uint8_t>& v) {
    py::array_t<bool> out(static_cast<py::ssize_t>(v.size()));
    bool* d = out.mutable_data();
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i] != 0;
    return out;
}

template <typename T>
py::object opt_arr(const std::optional<std::vector<T>>& v) {
    return v ? py::object(arr(*v)) : py::object(py::none());
}

template <typename T>
py::object arr_or_none(const std::vector<T>& v) {
    return v.empty() ? py::object(py::none()) : py::object(arr(v));
}

// A 2-D float64 matrix; anything with no elements is 0x0.
inline smath::Matrix matrix(const py::handle& o) {
    py::array a = py::array::ensure(np_module().attr("asarray")(o, py::arg("dtype") = "float64"));
    if (!a) throw py::error_already_set();
    if (a.size() == 0) return {};
    if (a.ndim() != 2)
        throw py::value_error("expected a 2-D matrix, got " + std::to_string(a.ndim()) + " dimensions");
    return smath::Matrix(static_cast<std::size_t>(a.shape(0)), static_cast<std::size_t>(a.shape(1)),
                         from_numpy<double>(a));
}

inline py::array to_py(const smath::Matrix& m) {
    py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(m.rows),
                                                     static_cast<py::ssize_t>(m.cols)});
    std::copy(m.data.begin(), m.data.end(), out.mutable_data());
    return out;
}

// --- Rational, FrameSpec, PathFn --------------------------------------------

inline Rational rational(const py::handle& o) {
    if (py::isinstance<py::float_>(o)) return Rational::from_double(o.cast<double>());
    if (py::hasattr(o, "numerator") && py::hasattr(o, "denominator"))
        return Rational(as_int(o.attr("numerator")), as_int(o.attr("denominator")));
    return Rational::from_double(as_double(o));
}

inline py::object to_py(const Rational& r) { return fraction_class()(r.num(), r.den()); }

// A native FrameSpec, or the reference's (any object with its fields).
inline frame::FrameSpec spec(const py::handle& o) {
    if (py::isinstance<frame::FrameSpec>(o)) return o.cast<frame::FrameSpec>();
    std::vector<std::int64_t> cw;
    for (const py::handle c : py::reinterpret_borrow<py::iterable>(field(o, "cw_after"))) cw.push_back(as_int(c));
    const py::object rho_o = field(o, "hdr_rho");
    std::optional<double> rho;
    if (!rho_o.is_none()) rho = as_double(rho_o);
    const std::int64_t w = as_int(field(o, "waveform"));
    if (w != sstvae::qrss::config::WAVEFORM_CE && w != sstvae::qrss::config::WAVEFORM_L)
        throw py::value_error("waveform must be 0 (CE) or 1 (L), not " + std::to_string(w));
    return frame::FrameSpec::make(static_cast<frame::Waveform>(w), as_string(py::str(field(o, "name"))),
                                  as_int(field(o, "n_pre")), as_int(field(o, "n_hdr")), as_int(field(o, "n_data")),
                                  std::move(cw), rho);
}

inline py::object to_py(const frame::FrameSpec& s) { return py::cast(s); }

// The native PathFn, a wrapper carrying one as `_native`, or the
// reference's (lo, hi, _line, _spl read as they are: no refit).
inline track::PathFn path_fn(const py::handle& o) {
    if (py::isinstance<track::PathFn>(o)) return o.cast<track::PathFn>();
    if (py::hasattr(o, "_native")) {
        py::object n = o.attr("_native");
        if (py::isinstance<track::PathFn>(n)) return n.cast<track::PathFn>();
    }
    const py::object spl = o.attr("_spl");
    std::optional<smath::BSpline> s;
    if (!spl.is_none())
        s = smath::BSpline{vec<double>(spl.attr("t")), vec<double>(spl.attr("c")),
                           static_cast<int>(as_int(spl.attr("k")))};
    return track::PathFn::from_parts(as_double(o.attr("lo")), as_double(o.attr("hi")), vec<double>(o.attr("_line")),
                                     std::move(s));
}

inline py::object to_py(const track::PathFn& p) { return py::cast(p); }

// --- records (types.py) -------------------------------------------------------

inline py::dict typed(const char* name) {
    py::dict d;
    d["__type__"] = name;
    return d;
}

inline types::Timing timing(const py::handle& o) {
    types::Timing t;
    t.tau0 = as_double(field(o, "tau0"));
    t.ppm = as_double(field(o, "ppm"));
    t.gamma = as_double(field(o, "gamma"));
    t.z = as_double(field(o, "z"));
    t.cov = matrix(field(o, "cov"));
    return t;
}

inline std::optional<types::Timing> opt_timing(const py::handle& o) {
    if (o.is_none()) return std::nullopt;
    return timing(o);
}

inline py::dict to_py(const types::Timing& t) {
    py::dict d = typed("Timing");
    d["tau0"] = t.tau0;
    d["ppm"] = t.ppm;
    d["gamma"] = t.gamma;
    d["z"] = t.z;
    d["cov"] = to_py(t.cov);
    return d;
}

inline types::FreqPath freq_path(const py::handle& o) {
    return {vec<double>(field(o, "t_s")), vec<double>(field(o, "f_hz")), vec<double>(field(o, "weight"))};
}

inline py::dict to_py(const types::FreqPath& p) {
    py::dict d = typed("FreqPath");
    d["t_s"] = arr(p.t_s);
    d["f_hz"] = arr(p.f_hz);
    d["weight"] = arr(p.weight);
    return d;
}

inline types::CarrierCandidate carrier_candidate(const py::handle& o) {
    types::CarrierCandidate c;
    c.f_hz = as_double(field(o, "f_hz"));
    c.drift_hz_per_min = as_double(field(o, "drift_hz_per_min"));
    c.metric = as_double(field(o, "metric"));
    const py::object w = field(o, "waveform");
    if (!w.is_none()) c.waveform = as_int(w);
    return c;
}

inline py::dict to_py(const types::CarrierCandidate& c) {
    py::dict d = typed("CarrierCandidate");
    d["f_hz"] = c.f_hz;
    d["drift_hz_per_min"] = c.drift_hz_per_min;
    d["metric"] = c.metric;
    d["waveform"] = c.waveform ? py::object(py::int_(*c.waveform)) : py::object(py::none());
    return d;
}

inline types::Detection detection(const py::handle& o) {
    types::Detection x;
    x.f_hz = as_double(field(o, "f_hz"));
    x.path = freq_path(field(o, "path"));
    x.timing = opt_timing(field(o, "timing"));
    x.z_ref = as_double(field(o, "z_ref"));
    x.method = as_string(field(o, "method"));
    x.lead_in_s = as_double(field(o, "lead_in_s"));
    x.waveform = as_int(field(o, "waveform"));
    x.a3_z = as_double(field(o, "a3_z"));
    return x;
}

inline py::dict to_py(const types::Detection& x) {
    py::dict d = typed("Detection");
    d["f_hz"] = x.f_hz;
    d["path"] = to_py(x.path);
    d["timing"] = x.timing ? py::object(to_py(*x.timing)) : py::object(py::none());
    d["z_ref"] = x.z_ref;
    d["method"] = to_str(x.method);
    d["lead_in_s"] = x.lead_in_s;
    d["waveform"] = x.waveform;
    d["a3_z"] = x.a3_z;
    return d;
}

inline types::TrackReport track_report(const py::handle& o) {
    types::TrackReport r;
    r.offset_hz = as_double(field(o, "offset_hz"));
    r.drift_hz_per_min = as_double(field(o, "drift_hz_per_min"));
    r.wander_hz_rms = as_double(field(o, "wander_hz_rms"));
    r.doppler_hz = as_double(field(o, "doppler_hz"));
    r.ppm = as_double(field(o, "ppm"));
    r.snr2500_db = as_double(field(o, "snr2500_db"));
    r.z_ref = as_double(field(o, "z_ref"));
    r.kappa = as_double(field(o, "kappa"));
    r.slip_free = as_bool(field(o, "slip_free"));
    r.suspect = as_bool(field(o, "suspect"));
    return r;
}

inline py::dict to_py(const types::TrackReport& r) {
    py::dict d = typed("TrackReport");
    d["offset_hz"] = r.offset_hz;
    d["drift_hz_per_min"] = r.drift_hz_per_min;
    d["wander_hz_rms"] = r.wander_hz_rms;
    d["doppler_hz"] = r.doppler_hz;
    d["ppm"] = r.ppm;
    d["snr2500_db"] = r.snr2500_db;
    d["z_ref"] = r.z_ref;
    d["kappa"] = r.kappa;
    d["slip_free"] = r.slip_free;
    d["suspect"] = r.suspect;
    return d;
}

inline types::CwIdResult cw_result(const py::handle& o) {
    types::CwIdResult c;
    c.text = opt_string(field(o, "text"));
    c.z_match = as_double(field(o, "z_match"));
    c.keying = as_string(field(o, "keying"));
    const py::object a = field(o, "agrees");
    if (!a.is_none()) c.agrees = as_bool(a);
    return c;
}

inline py::dict to_py(const types::CwIdResult& c) {
    py::dict d = typed("CwIdResult");
    d["text"] = to_py(c.text);
    d["z_match"] = c.z_match;
    d["keying"] = to_str(c.keying);
    d["agrees"] = c.agrees ? py::object(py::bool_(*c.agrees)) : py::object(py::none());
    return d;
}

inline types::EmPrior em_prior(const py::handle& o) { return {vec<double>(field(o, "x_hat")), vec<double>(field(o, "v"))}; }

inline py::dict to_py(const types::EmPrior& p) {
    py::dict d = typed("EmPrior");
    d["x_hat"] = arr(p.x_hat);
    d["v"] = arr(p.v);
    return d;
}

inline header::HeaderFields header_fields(const py::handle& o) {
    header::HeaderFields h;
    h.callsign = as_string(field(o, "callsign"));
    h.grid = opt_string(field(o, "grid"));
    h.picture_id = as_int(field(o, "picture_id"));
    h.mode = as_int(field(o, "mode"));
    h.segment = as_int(field(o, "segment"));
    h.codec_id = as_int(field(o, "codec_id"));
    h.version = as_int(field(o, "version"));
    h.reserved = as_int(field(o, "reserved"));
    return h;
}

inline py::dict to_py(const header::HeaderFields& h) {
    py::dict d = typed("HeaderFields");
    d["callsign"] = to_str(h.callsign);
    d["grid"] = to_py(h.grid);
    d["picture_id"] = h.picture_id;
    d["mode"] = h.mode;
    d["segment"] = h.segment;
    d["codec_id"] = h.codec_id;
    d["version"] = h.version;
    d["reserved"] = h.reserved;
    return d;
}

inline types::PassResult pass_result(const py::handle& o) {
    types::PassResult p;
    p.uid = as_string(field(o, "uid"));
    p.q = as_int(field(o, "q"));
    p.frame = as_string(field(o, "frame"));
    p.waveform = as_int(field(o, "waveform"));
    p.f_hz = as_double(field(o, "f_hz"));
    p.timing = timing(field(o, "timing"));
    p.report = track_report(field(o, "report"));
    p.z = vec<float>(field(o, "z"));
    p.w = vec<float>(field(o, "w"));
    p.hdr_llr = vec<float>(field(o, "hdr_llr"));
    const py::object h = field(o, "header");
    if (!h.is_none()) p.header = header_fields(h);
    const py::object c = field(o, "cw");
    if (!c.is_none()) p.cw = cw_result(c);
    p.ch = vec<std::complex<float>>(field(o, "ch"));
    p.ch_fs = as_int(field(o, "ch_fs"));
    p.ch_t0_index = as_double(field(o, "ch_t0_index"));
    p.f_mix_hz = rational(field(o, "f_mix_hz"));
    p.psi = vec<float>(field(o, "psi"));
    p.estimator = as_string(field(o, "estimator"));
    p.em_round = as_int(field(o, "em_round"));
    return p;
}

inline py::dict to_py(const types::PassResult& p) {
    py::dict d = typed("PassResult");
    d["uid"] = to_str(p.uid);
    d["q"] = p.q;
    d["frame"] = to_str(p.frame);
    d["waveform"] = p.waveform;
    d["f_hz"] = p.f_hz;
    d["timing"] = to_py(p.timing);
    d["report"] = to_py(p.report);
    d["z"] = arr(p.z);
    d["w"] = arr(p.w);
    d["hdr_llr"] = arr(p.hdr_llr);
    d["header"] = p.header ? py::object(to_py(*p.header)) : py::object(py::none());
    d["cw"] = p.cw ? py::object(to_py(*p.cw)) : py::object(py::none());
    d["ch"] = arr(p.ch);
    d["ch_fs"] = p.ch_fs;
    d["ch_t0_index"] = p.ch_t0_index;
    d["f_mix_hz"] = to_py(p.f_mix_hz);
    d["psi"] = arr(p.psi);
    d["estimator"] = to_str(p.estimator);
    d["em_round"] = p.em_round;
    return d;
}

// --- the tracker's records (trackbase.hpp) --------------------------------------

inline track::ChanCapture chan_capture(const py::handle& o) {
    track::ChanCapture c;
    c.ch = vec<std::complex<float>>(field(o, "ch"));
    c.fs = static_cast<int>(as_int(field(o, "fs")));
    c.t0_index = as_double(field(o, "t0_index"));
    c.f_mix = rational(field(o, "f_mix"));
    c.keep = opt_vec<float>(field(o, "keep"));
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(field(o, "exclude_hz")))
        c.exclude_hz.push_back(as_double(h));
    if (has_field(o, "_psd")) {
        const py::object psd = field(o, "_psd");
        if (!psd.is_none()) {
            py::tuple t = py::reinterpret_borrow<py::tuple>(py::tuple(psd));
            if (t.size() != 2) throw py::value_error("ChanCapture._psd must be (t_s, psd)");
            c.psd.set(std::make_shared<const track::NoisePsd>(track::NoisePsd{vec<double>(t[0]), vec<double>(t[1])}));
        }
    }
    return c;
}

inline py::dict to_py(const track::ChanCapture& c) {
    py::dict d = typed("ChanCapture");
    d["ch"] = arr(c.ch);
    d["fs"] = c.fs;
    d["t0_index"] = c.t0_index;
    d["f_mix"] = to_py(c.f_mix);
    d["keep"] = opt_arr(c.keep);
    py::tuple ex(c.exclude_hz.size());
    for (std::size_t i = 0; i < c.exclude_hz.size(); ++i) ex[i] = c.exclude_hz[i];
    d["exclude_hz"] = ex;
    const auto psd = c.psd.get();
    d["_psd"] = psd ? py::object(py::make_tuple(arr(psd->t_s), arr(psd->psd))) : py::object(py::none());
    return d;
}

inline bool is_complex_array(const py::handle& o) {
    py::object a = np_module().attr("asarray")(o);
    return py::str(a.attr("dtype").attr("kind")).cast<std::string>() == "c";
}

inline track::MfTable mf_table(const py::handle& o) {
    const double t_sym = as_double(field(o, "t_sym"));
    const double sps = as_double(field(o, "sps"));
    const int half = static_cast<int>(as_int(field(o, "half")));
    const double g0 = as_double(field(o, "g0"));
    const py::object tab = field(o, "tab");
    if (is_complex_array(tab)) return track::MfTable(t_sym, sps, half, vec<std::complex<double>>(tab), g0);
    return track::MfTable(t_sym, sps, half, vec<double>(tab), g0);
}

inline py::dict to_py(const track::MfTable& t) {
    py::dict d = typed("MfTable");
    d["t_sym"] = t.t_sym();
    d["sps"] = t.sps();
    d["half"] = t.half();
    if (t.is_complex()) {
        const auto v = t.ctab();
        d["tab"] = to_numpy(v, false);
    } else {
        const auto v = t.tab();
        d["tab"] = to_numpy(v, false);
    }
    d["g0"] = t.g0();
    return d;
}

inline track::Known known(const py::handle& o) {
    track::Known k;
    k.pos = vec<std::int64_t>(field(o, "pos"));
    const py::object v = field(o, "val");
    if (is_complex_array(v))
        k.val = vec<std::complex<double>>(v);
    else
        k.val = vec<double>(v);
    return k;
}

inline py::dict to_py(const track::Known& k) {
    py::dict d = typed("Known");
    d["pos"] = arr(k.pos);
    d["val"] = k.is_complex() ? arr(k.cplx()) : arr(k.real());
    return d;
}

inline track::Classes classes(const py::handle& o) {
    track::Classes c;
    c.mu = vec<double>(field(o, "mu"));
    c.nu = vec<double>(field(o, "nu"));
    c.known = vec_bool(field(o, "known"));
    c.hdr_known = as_bool(field(o, "hdr_known"));
    c.spread = opt_vec<double>(field(o, "spread"));
    c.spread_rho = as_double(field(o, "spread_rho"));
    return c;
}

inline py::dict to_py(const track::Classes& c) {
    py::dict d = typed("Classes");
    d["mu"] = arr(c.mu);
    d["nu"] = arr(c.nu);
    d["known"] = arr_bool(c.known);
    d["hdr_known"] = c.hdr_known;
    d["spread"] = opt_arr(c.spread);
    d["spread_rho"] = c.spread_rho;
    return d;
}

inline track::LClasses lclasses(const py::handle& o) {
    track::LClasses c;
    c.mu = vec<std::complex<double>>(field(o, "mu"));
    c.nu = vec<double>(field(o, "nu"));
    c.known = vec_bool(field(o, "known"));
    c.gain = vec<double>(field(o, "gain"));
    c.hdr_bits = opt_vec<std::uint8_t>(field(o, "hdr_bits"));
    c.hdr_known = as_bool(field(o, "hdr_known"));
    c.spread_rho = as_double(field(o, "spread_rho"));
    return c;
}

inline py::dict to_py(const track::LClasses& c) {
    py::dict d = typed("LClasses");
    d["mu"] = arr(c.mu);
    d["nu"] = arr(c.nu);
    d["known"] = arr_bool(c.known);
    d["gain"] = arr(c.gain);
    d["hdr_bits"] = opt_arr(c.hdr_bits);
    d["hdr_known"] = c.hdr_known;
    d["spread"] = py::none();
    d["spread_rho"] = c.spread_rho;
    return d;
}

// The type a "__type__" key names, else what the object's class is called.
inline std::string type_name(const py::handle& o) {
    if (py::isinstance<py::dict>(o)) {
        py::dict d = py::reinterpret_borrow<py::dict>(o);
        if (d.contains("__type__")) return d["__type__"].cast<std::string>();
        return "";
    }
    return py::str(py::type::of(o).attr("__name__")).cast<std::string>();
}

inline std::variant<std::monostate, track::Classes, track::LClasses> any_classes(const py::handle& o) {
    if (o.is_none()) return std::monostate{};
    if (has_field(o, "gain")) return lclasses(o);
    return classes(o);
}

inline void fill_track_result(track::TrackResult& tr, const py::handle& o) {
    tr.pos = vec<std::int64_t>(field(o, "pos"));
    const py::object zd = field(o, "z_d");
    if (!zd.is_none())
        tr.z_d = std::make_shared<const std::vector<std::complex<double>>>(vec<std::complex<double>>(zd));
    tr.m = vec_or_empty<std::complex<double>>(field(o, "m"));
    tr.c = vec_or_empty<std::complex<double>>(field(o, "c"));
    tr.u = vec_or_empty<std::complex<double>>(field(o, "u"));
    tr.P = vec_or_empty<double>(field(o, "P"));
    tr.N = vec_or_empty<double>(field(o, "N"));
    tr.psi = vec_or_empty<double>(field(o, "psi"));
    const py::object meas = field(o, "meas");
    if (!meas.is_none()) tr.meas = vec_bool(meas);
    tr.pu = vec_or_empty<double>(field(o, "pu"));
    tr.q = as_double(field(o, "q"));
    tr.loglik = as_double(field(o, "loglik"));
    tr.classes = any_classes(field(o, "classes"));
    tr.keying = opt_vec<std::uint8_t>(field(o, "keying"));
    tr.ook = as_bool(field(o, "ook"));
    tr.lead_in_s = as_double(field(o, "lead_in_s"));
    tr.z_ref = as_double(field(o, "z_ref"));
    tr.df_max = as_double(field(o, "df_max"));
    const py::object rep = field(o, "report");
    if (!rep.is_none()) tr.report = track_report(rep);
}

inline track::TrackResult track_result(const py::handle& o) {
    track::TrackResult tr(spec(field(o, "spec")), std::make_shared<const track::ChanCapture>(chan_capture(field(o, "chan"))),
                          timing(field(o, "timing")), path_fn(field(o, "freq")));
    fill_track_result(tr, o);
    return tr;
}

inline track::LTrackResult ltrack_result(const py::handle& o) {
    track::LTrackResult tr(spec(field(o, "spec")), std::make_shared<const track::ChanCapture>(chan_capture(field(o, "chan"))),
                           timing(field(o, "timing")), path_fn(field(o, "freq")));
    fill_track_result(tr, o);
    tr.m_pl = opt_vec<std::complex<double>>(field(o, "m_pl"));
    tr.c_pl = opt_vec<std::complex<double>>(field(o, "c_pl"));
    tr.N_pl = opt_vec<double>(field(o, "N_pl"));
    tr.psi_pl = opt_vec<double>(field(o, "psi_pl"));
    const py::object mp = field(o, "meas_pl");
    if (!mp.is_none()) tr.meas_pl = vec_bool(mp);
    {
        const py::object dl = field(o, "delta");
        py::object c = py::reinterpret_steal<py::object>(PyNumber_Float(dl.ptr()));
        if (c) {
            tr.delta = {PyFloat_AsDouble(c.ptr()), 0.0};
        } else {
            PyErr_Clear();
            tr.delta = py::cast<std::complex<double>>(py::module_::import("builtins").attr("complex")(dl));
        }
    }
    tr.u_pl = opt_vec<std::complex<double>>(field(o, "u_pl"));
    tr.P_pl = opt_vec<double>(field(o, "P_pl"));
    tr.resid_ratio = as_double(field(o, "resid_ratio"));
    return tr;
}

inline py::dict to_py(const track::TrackResult& tr) {
    const auto* l = dynamic_cast<const track::LTrackResult*>(&tr);
    py::dict d = typed(l ? "LTrackResult" : "TrackResult");
    d["spec"] = to_py(tr.spec);
    d["chan"] = tr.chan ? py::object(to_py(*tr.chan)) : py::object(py::none());
    d["timing"] = to_py(tr.timing);
    d["freq"] = to_py(tr.freq);
    d["pos"] = arr(tr.pos);
    d["z_d"] = tr.z_d ? py::object(arr(*tr.z_d)) : py::object(py::none());
    d["m"] = arr_or_none(tr.m);
    d["c"] = arr_or_none(tr.c);
    d["u"] = arr_or_none(tr.u);
    d["P"] = arr_or_none(tr.P);
    d["N"] = arr_or_none(tr.N);
    d["psi"] = arr_or_none(tr.psi);
    d["meas"] = tr.meas.empty() ? py::object(py::none()) : py::object(arr_bool(tr.meas));
    d["pu"] = arr_or_none(tr.pu);
    d["q"] = tr.q;
    d["loglik"] = tr.loglik;
    if (const auto* c = tr.ce_classes())
        d["classes"] = to_py(*c);
    else if (const auto* lc = tr.l_classes())
        d["classes"] = to_py(*lc);
    else
        d["classes"] = py::none();
    d["keying"] = opt_arr(tr.keying);
    d["ook"] = tr.ook;
    d["lead_in_s"] = tr.lead_in_s;
    d["z_ref"] = tr.z_ref;
    d["df_max"] = tr.df_max;
    d["report"] = tr.report ? py::object(to_py(*tr.report)) : py::object(py::none());
    if (l) {
        d["m_pl"] = opt_arr(l->m_pl);
        d["c_pl"] = opt_arr(l->c_pl);
        d["N_pl"] = opt_arr(l->N_pl);
        d["psi_pl"] = opt_arr(l->psi_pl);
        d["meas_pl"] = l->meas_pl ? py::object(arr_bool(*l->meas_pl)) : py::object(py::none());
        d["delta"] = l->delta;
        d["u_pl"] = opt_arr(l->u_pl);
        d["P_pl"] = opt_arr(l->P_pl);
        d["resid_ratio"] = l->resid_ratio;
    }
    return d;
}

// A TrackResult or LTrackResult, by the object's type (an LTrackResult has
// m_pl).
inline std::unique_ptr<track::TrackResult> any_track_result(const py::handle& o) {
    if (has_field(o, "m_pl")) return std::make_unique<track::LTrackResult>(ltrack_result(o));
    return std::make_unique<track::TrackResult>(track_result(o));
}

}  // namespace sstvae_bindings::qrss::rx
