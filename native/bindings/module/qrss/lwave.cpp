// sstvae_native.qrss.lwave: bindings of native/core/qrss/lwave, waveform L's
// transmitter. Function names, argument names (keyword-only where the
// reference has them), defaults, dtypes and ValueError texts follow
// sstvae/qrss/lwave.py, so tests/qrss_native/lwave.py can assign these over
// the reference's.
//
// A `spec` is the reference's LFrameSpec (or a native FrameSpec), converted
// through sstvae_native.qrss.frame.of. A `bb` is anything with z, n0 and fs
// (the reference's Baseband dataclass, or this module's Baseband).
// frame_baseband returns this module's Baseband; the substitution rows wrap
// it in the reference's dataclass.

#include "qrss/registry.hpp"

#include <pybind11/complex.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/lwave.hpp"
#include "qrss/rational.hpp"
#include "qrss/tables.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace lw = sstvae::qrss::lwave;
namespace fr = sstvae::qrss::frame;
namespace cfg = sstvae::qrss::config;
using cplx = std::complex<double>;

py::module_ numpy() {
    static py::module_* np = new py::module_(py::module_::import("numpy"));
    return *np;
}

// sstvae_native.qrss.frame.of, set when this module binds (after frame's).
py::object* g_of = nullptr;

fr::FrameSpec spec_of(const py::handle& spec) {
    if (py::isinstance<fr::FrameSpec>(spec)) return spec.cast<fr::FrameSpec>();
    return (*g_of)(spec).cast<fr::FrameSpec>();
}

// int(h) as int64 (OverflowError past int64, as Python's int -> C long).
std::int64_t as_int(const py::handle& h) {
    py::object i = py::reinterpret_steal<py::object>(PyNumber_Long(h.ptr()));
    if (!i) throw py::error_already_set();
    const long long v = PyLong_AsLongLong(i.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

// float(h).
double as_float(const py::handle& h) {
    py::object f = py::reinterpret_steal<py::object>(PyNumber_Float(h.ptr()));
    if (!f) throw py::error_already_set();
    return PyFloat_AsDouble(f.ptr());
}

std::vector<std::int64_t> shape_of(const py::array& a) {
    return std::vector<std::int64_t>(a.shape(), a.shape() + a.ndim());
}

std::vector<py::ssize_t> pyshape(const py::array& a) { return std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim()); }

// np.asarray(obj, dtype=float64) / complex128, C-contiguous.
py::array_t<double> f64(const py::handle& obj) {
    py::object a = numpy().attr("asarray")(obj, py::arg("dtype") = "float64");
    auto v = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
    if (!v) throw py::error_already_set();
    return v;
}

py::array_t<cplx> c128(const py::handle& obj) {
    py::object a = numpy().attr("asarray")(obj, py::arg("dtype") = "complex128");
    auto v = py::array_t<cplx, py::array::c_style | py::array::forcecast>::ensure(a);
    if (!v) throw py::error_already_set();
    return v;
}

template <typename T>
std::span<const T> span_of(const py::array_t<T>& a) {
    return {a.data(), static_cast<std::size_t>(a.size())};
}

template <typename T>
py::array_t<T> shaped(const std::vector<T>& v, const std::vector<py::ssize_t>& shape, bool writeable = true) {
    py::array_t<T> out(shape);
    std::copy(v.begin(), v.end(), out.mutable_data());
    if (!writeable) out.attr("setflags")(py::arg("write") = false);
    return out;
}

// The lead-in refusal with the value shown as str() of what was given, as
// the reference's f-string shows it ("11" for an int).
template <typename F>
auto with_lead_text(const py::handle& lead_obj, F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const lw::LeadInRefusal&) {
        throw py::value_error(lw::lead_in_refusal(py::str(lead_obj).cast<std::string>()));
    }
}

// The keying argument as the reference sees it: np.asarray(keying).
std::optional<lw::Keying> keying_of(const py::object& obj) {
    if (obj.is_none()) return std::nullopt;
    py::array a = numpy().attr("asarray")(obj);
    lw::Keying k;
    k.shape = shape_of(a);
    const char kind = a.dtype().kind();
    if (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f') {
        const auto v = f64(a);
        k.values.assign(v.data(), v.data() + v.size());
    } else if (kind == 'c') {
        // (k != 0) & (k != 1) holds for any value with an imaginary part.
        const auto re = f64(a.attr("real")), im = f64(a.attr("imag"));
        k.values.resize(static_cast<std::size_t>(re.size()));
        for (std::size_t i = 0; i < k.values.size(); ++i)
            k.values[i] = im.data()[i] != 0 ? std::numeric_limits<double>::quiet_NaN() : re.data()[i];
    } else {
        // Not a number: numpy's elementwise != is true, so the reference refuses it.
        k.values.assign(static_cast<std::size_t>(a.size()), std::numeric_limits<double>::quiet_NaN());
    }
    return k;
}

const lw::Keying* ptr(const std::optional<lw::Keying>& k) { return k ? &*k : nullptr; }

// sym as the reference converts it, its shape checked against the frame.
py::array_t<cplx> sym_of(const py::object& sym, const fr::FrameSpec& spec) {
    auto a = c128(sym);
    lw::check_sym_shape(spec, shape_of(a));
    return a;
}

py::object keyed_end(const fr::FrameSpec& s) {
    if (s.last_window_ends_frame()) return py::float_(s.keyed_end_pos());
    return py::int_(static_cast<std::int64_t>(s.keyed_end_pos()));
}

// The reference's Fraction(f_hz), as an int64 Rational.
sstvae::qrss::Rational fraction_of(const py::handle& f_hz) {
    static py::object* Fraction = new py::object(py::module_::import("fractions").attr("Fraction"));
    py::object f = (*Fraction)(f_hz);
    const std::int64_t num = as_int(f.attr("numerator")), den = as_int(f.attr("denominator"));
    return sstvae::qrss::Rational(num, den);
}

// What `bb` holds: z as complex128, n0 and fs as ints.
struct BbView {
    py::array_t<cplx> z;
    std::int64_t n0;
    std::int64_t fs;
};

BbView bb_of(const py::handle& bb) {
    return {c128(bb.attr("z")), as_int(bb.attr("n0")), as_int(bb.attr("fs"))};
}

// This module's Baseband: z read-only, as the reference's frame_baseband
// leaves it.
struct PyBaseband {
    py::array z;
    std::int64_t n0;
    std::int64_t fs;
};

// A real or a 0-d result as numpy gives it: a numpy float64 scalar for 0-d.
py::object real_out(const std::vector<double>& v, const py::array& like) {
    py::array_t<double> r = shaped(v, pyshape(like));
    if (like.ndim() == 0) return r[py::tuple()];
    return r;
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("lwave", "Port of sstvae.qrss.lwave (waveform L, transmit side)");
    g_of = new py::object(q.attr("frame").attr("of"));

    // --- constants
    m.attr("GRID_FS") = lw::GRID_FS;
    m.attr("HALF") = lw::HALF;
    m.attr("CLIP_FS") = lw::CLIP_FS;
    m.attr("PULSE_TABLE") = to_numpy(lw::pulse_table(), false);
    m.attr("G0") = lw::g0();
    m.attr("A_PILOT") = lw::a_pilot();
    m.attr("A_CLIP") = lw::a_clip();
    m.attr("L_AVG_POWER") = lw::L_AVG_POWER;
    m.attr("A_ID") = lw::a_id();
    m.attr("ID_RAMP_UNITS") = lw::ID_RAMP_UNITS;
    m.attr("KEY_RAMP_S") = lw::KEY_RAMP_S;
    m.attr("INTERP_HALF") = lw::INTERP_HALF;

    py::class_<PyBaseband>(m, "Baseband", "The clipped frame (lwave.Baseband): z, n0, fs")
        .def(py::init([](const py::object& z, const py::object& n0, const py::object& fs) {
                 return PyBaseband{numpy().attr("asarray")(z), as_int(n0), as_int(fs)};
             }),
             py::arg("z"), py::arg("n0"), py::arg("fs"))
        .def_readonly("z", &PyBaseband::z)
        .def_readonly("n0", &PyBaseband::n0)
        .def_readonly("fs", &PyBaseband::fs)
        .def("__repr__", [](const PyBaseband& b) {
            return "Baseband(z=<" + std::to_string(b.z.size()) + " samples>, n0=" + std::to_string(b.n0) +
                   ", fs=" + std::to_string(b.fs) + ")";
        });

    // --- the data on a sample grid
    m.def("_grid_step", [](const py::object& fs) { return lw::grid_step(as_int(fs)); }, py::arg("fs"));

    m.def("_grid_sum", [](const py::object& fs, const py::object& n0, const py::object& n, const py::object& vals,
                          const py::object& table) -> py::object {
        const std::int64_t f = as_int(fs);
        lw::grid_step(f);
        const std::int64_t a = as_int(n0), b = as_int(n);
        py::array v = numpy().attr("asarray")(vals);
        const auto t = f64(table);
        if (v.dtype().kind() == 'c') {
            const auto c = c128(v);
            return to_numpy(lw::grid_sum(f, a, b, span_of(c), span_of(t)));
        }
        const auto r = f64(v);
        return to_numpy(lw::grid_sum(f, a, b, span_of(r), span_of(t)));
    }, py::arg("fs"), py::arg("n0"), py::arg("n"), py::arg("vals"), py::arg("table"));

    m.def("at_positions", [](const py::object& sym, const py::object& spec) {
        const fr::FrameSpec s = spec_of(spec);
        const auto a = sym_of(sym, s);
        return to_numpy(lw::at_positions(span_of(a), s));
    }, py::arg("sym"), py::arg("spec"));

    m.def("data_grid", [](const py::object& fs, const py::object& n0, const py::object& n, const py::object& sym,
                          const py::object& spec) {
        const fr::FrameSpec s = spec_of(spec);
        const auto a = sym_of(sym, s);
        return to_numpy(lw::data_grid(as_int(fs), as_int(n0), as_int(n), span_of(a), s));
    }, py::arg("fs"), py::arg("n0"), py::arg("n"), py::arg("sym"), py::arg("spec"));

    m.def("mix_turns", [](const py::object& fs, const py::object& n0, const py::object& n, const py::object& f_hz) {
        const auto f = fraction_of(f_hz);
        const std::int64_t fsi = as_int(fs);
        if (fsi > 0 && f.den() > std::numeric_limits<std::int64_t>::max() / fsi) {
            PyErr_SetString(PyExc_OverflowError, "Python int too large to convert to C long");
            throw py::error_already_set();
        }
        return to_numpy(lw::mix_turns(fsi, as_int(n0), as_int(n), f));
    }, py::arg("fs"), py::arg("n0"), py::arg("n"), py::arg("f_hz"));

    m.def("tau_grid", [](const py::object& fs, const py::object& n0, const py::object& n) {
        return to_numpy(lw::tau_grid(as_int(fs), as_int(n0), as_int(n)));
    }, py::arg("fs"), py::arg("n0"), py::arg("n"));

    // --- pilot
    m.def("keyed_span_pos", [](const py::object& spec, const py::object& lead_in_s) {
        const fr::FrameSpec s = spec_of(spec);
        const double lead = as_float(lead_in_s);
        const auto lo_hi = with_lead_text(lead_in_s, [&] { return lw::keyed_span_pos(s, lead); });
        return py::make_tuple(lo_hi.first, keyed_end(s));
    }, py::arg("spec"), py::arg("lead_in_s") = 0.0);

    m.def("window_keying", [](const py::object& keying, const py::object& spec) -> py::object {
        const fr::FrameSpec s = spec_of(spec);
        const auto k = keying_of(keying);
        const auto w = lw::window_keying(ptr(k), s);
        if (!w) return py::none();
        return shaped(*w, {static_cast<py::ssize_t>(s.n_win()), static_cast<py::ssize_t>(cfg::L_CW_WINDOW)});
    }, py::arg("keying"), py::arg("spec"));

    m.def("_raised", [](const py::object& y) {
        const auto a = f64(y);
        std::vector<double> out(static_cast<std::size_t>(a.size()));
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = lw::raised(a.data()[i]);
        return real_out(out, a);
    }, py::arg("y"));

    m.def("cw_turns", [](const py::object& tau, const py::object& spec, const py::object& keying) {
        const auto t = f64(tau);
        const fr::FrameSpec s = spec_of(spec);
        const auto k = keying_of(keying);
        return shaped(lw::cw_turns(span_of(t), s, ptr(k)), pyshape(t));
    }, py::arg("tau"), py::arg("spec"), py::arg("keying"));

    m.def("id_weight", [](const py::object& tau, const py::object& spec, const py::object& lead_in_s) {
        const auto t = f64(tau);
        const fr::FrameSpec s = spec_of(spec);
        return shaped(lw::id_weight(span_of(t), s, as_float(lead_in_s)), pyshape(t));
    }, py::arg("tau"), py::arg("spec"), py::arg("lead_in_s") = 0.0);

    m.def("pilot", [](const py::object& tau, const py::object& spec, const py::object& keying,
                      const py::object& lead_in_s, const py::object& ramp_s) {
        const auto t = f64(tau);
        const fr::FrameSpec s = spec_of(spec);
        const auto k = keying_of(keying);
        const double lead = as_float(lead_in_s), ramp = as_float(ramp_s);
        return shaped(with_lead_text(lead_in_s, [&] { return lw::pilot(span_of(t), s, ptr(k), lead, ramp); }),
                      pyshape(t));
    }, py::arg("tau"), py::arg("spec"), py::arg("keying") = py::none(), py::arg("lead_in_s") = 0.0,
       py::arg("ramp_s") = lw::KEY_RAMP_S);

    // --- composite and clipper
    m.def("composite", [](const py::object& sym, const py::object& spec, const py::object& fs,
                          const py::object& n0, const py::object& n, const py::object& keying,
                          const py::object& lead_in_s, const py::object& ramp_s) {
        const fr::FrameSpec s = spec_of(spec);
        const auto a = sym_of(sym, s);
        const auto k = keying_of(keying);
        const double lead = as_float(lead_in_s), ramp = as_float(ramp_s);
        return to_numpy(with_lead_text(lead_in_s, [&] {
            return lw::composite(span_of(a), s, as_int(fs), as_int(n0), as_int(n), ptr(k), lead, ramp);
        }));
    }, py::arg("sym"), py::arg("spec"), py::arg("fs"), py::arg("n0"), py::arg("n"), py::kw_only(),
       py::arg("keying") = py::none(), py::arg("lead_in_s") = 0.0, py::arg("ramp_s") = lw::KEY_RAMP_S);

    m.def("band_mask", [](const py::object& freqs, const py::object& guard_hz, const py::object& taper_hz) {
        const auto f = f64(freqs);
        return real_out(lw::band_mask(span_of(f), as_float(guard_hz), as_float(taper_hz)), f);
    }, py::arg("freqs"), py::arg("guard_hz") = cfg::L_CLIP_GUARD_HZ, py::arg("taper_hz") = 1.0);

    m.def("clip", [](const py::object& z, const py::object& fs, const py::object& threshold,
                     const py::object& overshoot, const py::object& guard_hz) {
        const auto a = c128(z);
        if (a.ndim() != 1) throw py::value_error("clip takes a 1-D signal");
        std::vector<double> ks;
        for (const py::handle k : py::reinterpret_borrow<py::iterable>(overshoot)) ks.push_back(as_float(k));
        return to_numpy(lw::clip(span_of(a), as_int(fs), as_float(threshold), ks, as_float(guard_hz)));
    }, py::arg("z"), py::arg("fs") = lw::CLIP_FS, py::arg("threshold") = lw::a_clip(),
       py::arg("overshoot") = py::make_tuple(cfg::L_CLIP_OVERSHOOT[0], cfg::L_CLIP_OVERSHOOT[1],
                                             cfg::L_CLIP_OVERSHOOT[2]),
       py::arg("guard_hz") = cfg::L_CLIP_GUARD_HZ);

    m.def("frame_baseband", [](const py::object& sym, const py::object& spec, const py::object& keying,
                               const py::object& lead_in_s, const py::object& fs, const py::object& ramp_s,
                               const py::object& margin_s) {
        const fr::FrameSpec s = spec_of(spec);
        const double lead = as_float(lead_in_s);
        // The reference's order: the lead-in first (keyed_span_pos), then sym.
        with_lead_text(lead_in_s, [&] { return lw::keyed_span_pos(s, lead); });
        const auto a = sym_of(sym, s);
        const auto k = keying_of(keying);
        const lw::Baseband bb = lw::frame_baseband(span_of(a), s, ptr(k), lead, as_int(fs), as_float(ramp_s),
                                                   as_float(margin_s));
        return PyBaseband{to_numpy(bb.z, false), bb.n0, bb.fs};
    }, py::arg("sym"), py::arg("spec"), py::kw_only(), py::arg("keying") = py::none(), py::arg("lead_in_s") = 0.0,
       py::arg("fs") = lw::CLIP_FS, py::arg("ramp_s") = lw::KEY_RAMP_S, py::arg("margin_s") = lw::MARGIN_S);

    // --- to 8 kHz
    m.def("_interp_taps", [](const py::object& up) { return to_numpy(lw::interp_taps(as_int(up))); },
          py::arg("up"));

    m.def("resample", [](const py::object& bb, const py::object& fs_out, const py::object& i0, const py::object& m) {
        const BbView b = bb_of(bb);
        const std::int64_t f_out = as_int(fs_out);
        // The reference's int(fs_out) // fs, checked against fs_out itself:
        // a fractional rate is refused, not truncated.
        if (!PyLong_Check(fs_out.ptr()) && as_float(fs_out) != static_cast<double>(f_out))
            throw py::value_error("fs_out must be a multiple of " + std::to_string(b.fs));
        const std::int64_t mm = as_int(m);
        return to_numpy(lw::resample(span_of(b.z), b.n0, b.fs, f_out, as_int(i0), mm));
    }, py::arg("bb"), py::arg("fs_out"), py::arg("i0"), py::arg("m"));

    m.def("to_data_centre", [](const py::object& bb) {
        const BbView b = bb_of(bb);
        return to_numpy(lw::to_data_centre(span_of(b.z), b.n0, b.fs));
    }, py::arg("bb"));

    // --- numpy's pieces, exposed for the parity test
    m.def("_next_fast_len", [](std::int64_t target) { return lw::next_fast_len(target); }, py::arg("target"));
    m.def("_fftfreq", [](std::int64_t n, double d) { return to_numpy(lw::fftfreq(n, d)); }, py::arg("n"),
          py::arg("d"));
    m.def("_abs", [](const py::object& z) {
        const auto a = c128(z);
        std::vector<double> out(static_cast<std::size_t>(a.size()));
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = lw::np_abs(a.data()[i]);
        return shaped(out, pyshape(a));
    }, py::arg("z"), "np.abs on complex128, as numpy computes it");
    m.def("_mul", [](const py::object& x, const py::object& y) {
        const auto a = c128(x), b = c128(y);
        if (a.size() != b.size()) throw py::value_error("_mul takes two arrays of one size");
        std::vector<cplx> out(static_cast<std::size_t>(a.size()));
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = lw::np_mul(a.data()[i], b.data()[i]);
        return shaped(out, pyshape(a));
    }, py::arg("x"), py::arg("y"), "x * y on complex128, as numpy's SIMD loop computes it");
}

const Registrar reg("lwave", 60, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
