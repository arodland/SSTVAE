// sstvae_native.qrss.ce: bindings of native/core/qrss/ce (waveform CE, the
// transmit side). Function names, argument names, defaults, dtypes, shapes
// and ValueError texts follow sstvae/qrss/ce.py, so tests/qrss_native/ce.py
// can assign these over the reference's functions.
//
// A `spec` argument is a native FrameSpec or the reference's own FrameSpec
// (anything sstvae_native.qrss.frame.of takes). Arrays come in through
// np.asarray with the reference's dtype (float64, complex128), any shape
// where the reference is element-wise, and go back in the same shape: a
// 0-d array for a scalar where the reference returns one, a numpy scalar
// where it returns that (window_units).
//
// The texts that quote the caller's value (the lead-in, the carrier) show
// the caller's own object as the reference's f-string does, format(x, '').

#include "qrss/registry.hpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/rational.hpp"
#include "qrss/tables.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace ce = sstvae::qrss::ce;
namespace fr = sstvae::qrss::frame;
namespace cfg = sstvae::qrss::config;
using cd = std::complex<double>;

// The Python objects this module keeps, each looked up once and held for
// the life of the process (a static py::object is released after the
// interpreter has gone, and that decref aborts it: "the GIL is released"),
// under pybind11's GIL-aware once: an import can release the GIL, which a
// plain function-local static would turn into a deadlock between threads.
template <typename F>
const py::object& held(py::gil_safe_call_once_and_store<py::object>& store, F&& make) {
    return store.call_once_and_store_result(std::forward<F>(make)).get_stored();
}

const py::object& numpy() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> store;
    return held(store, [] { return py::object(py::module_::import("numpy")); });
}

const py::object& fraction_type() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> store;
    return held(store, [] { return py::module_::import("fractions").attr("Fraction"); });
}

// sstvae_native.qrss.frame.of (bound before this module), stored at bind time.
PYBIND11_CONSTINIT py::gil_safe_call_once_and_store<py::object> frame_of_store;

const py::object& frame_of() { return frame_of_store.get_stored(); }

// A native FrameSpec, or the reference's (frame.of converts it).
fr::FrameSpec spec_of(const py::handle& obj) {
    if (py::isinstance<fr::FrameSpec>(obj)) return obj.cast<fr::FrameSpec>();
    return frame_of()(obj).cast<fr::FrameSpec>();
}

[[noreturn]] void raise(PyObject* type, const std::string& what) {
    PyErr_SetString(type, what.c_str());
    throw py::error_already_set();
}

// int(x) as int64 (OverflowError past int64).
std::int64_t as_int(const py::handle& h) {
    py::object i = py::reinterpret_steal<py::object>(PyNumber_Long(h.ptr()));
    if (!i) throw py::error_already_set();
    const long long v = PyLong_AsLongLong(i.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

// operator.index(x) as int64.
std::int64_t as_index(const py::handle& h) {
    py::object i = py::reinterpret_steal<py::object>(PyNumber_Index(h.ptr()));
    if (!i) throw py::error_already_set();
    const long long v = PyLong_AsLongLong(i.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

// float(x).
double as_float(const py::handle& h) {
    py::object f = py::reinterpret_steal<py::object>(PyNumber_Float(h.ptr()));
    if (!f) throw py::error_already_set();
    return PyFloat_AsDouble(f.ptr());
}

// format(x, ''): how an f-string shows x.
std::string shown(const py::handle& h) {
    py::object s = py::reinterpret_steal<py::object>(PyObject_Format(h.ptr(), nullptr));
    if (!s) throw py::error_already_set();
    return s.cast<std::string>();
}

// np.asarray(x, dtype=T), C-contiguous, any shape.
template <typename T>
py::array_t<T> as_array(const py::handle& obj) {
    const char* dt = std::is_same_v<T, double> ? "float64" : "complex128";
    py::object a = numpy().attr("asarray")(obj, py::arg("dtype") = dt);
    auto v = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(a);
    if (!v) throw py::error_already_set();
    return v;
}

template <typename T>
std::span<const T> span_of(const py::array_t<T>& a) {
    return std::span<const T>(a.data(), static_cast<std::size_t>(a.size()));
}

std::vector<std::int64_t> shape_of(const py::array& a) {
    return std::vector<std::int64_t>(a.shape(), a.shape() + a.ndim());
}

// A new array of `shape` holding v (0-d for an empty shape).
template <typename T>
py::array_t<T> shaped(const std::vector<T>& v, const std::vector<std::int64_t>& shape) {
    py::array_t<T> out(std::vector<py::ssize_t>(shape.begin(), shape.end()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

// x elementwise through f, in x's shape.
template <typename R, typename F>
py::array_t<R> elementwise(const py::handle& x, F&& f) {
    const auto a = as_array<double>(x);
    std::vector<R> out(static_cast<std::size_t>(a.size()));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = f(a.data()[i]);
    return shaped(out, shape_of(a));
}

// _check_sym(sym, spec, name): np.asarray(sym, float64) of shape (n_sym,).
py::array_t<double> checked_sym(const py::handle& sym, const fr::FrameSpec& spec, const std::string& name = "sym") {
    auto a = as_array<double>(sym);
    ce::check_sym(spec, shape_of(a), name);
    return a;
}

// window_keying(keying, spec): None for None, else the validated (n_win,
// 192) keying. Numeric dtypes are checked by the C++ on their float64
// values (exactly 0 and 1 survive that, and nothing else becomes them);
// any other dtype gets the reference's own numpy comparisons and cast.
std::optional<ce::Keying> keying_of(const py::handle& keying, const fr::FrameSpec& spec) {
    if (keying.is_none()) return std::nullopt;
    py::array k = numpy().attr("asarray")(keying);
    const char kind = k.dtype().kind();
    if (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f') {
        const auto v = as_array<double>(k);
        return ce::window_keying(span_of(v), shape_of(k), spec);
    }
    // Python's comparisons for the rest: shape first, then values.
    std::vector<std::int64_t> sh = shape_of(k);
    if (sh.size() == 1) sh = {spec.n_win(), sh[0]};
    bool ok = sh.size() == 2 && sh[0] == spec.n_win() && sh[1] == cfg::CW_UNITS;
    if (ok) {
        py::object b = numpy().attr("broadcast_to")(k, py::make_tuple(sh[0], sh[1]));
        auto ne = [&](int v) {
            py::object r = py::reinterpret_steal<py::object>(PyObject_RichCompare(b.ptr(), py::int_(v).ptr(), Py_NE));
            if (!r) throw py::error_already_set();
            return r;
        };
        ok = !py::bool_(numpy().attr("any")(ne(0) & ne(1))).cast<bool>();
    }
    if (!ok) raise(PyExc_ValueError, ce::keying_refusal(spec, shape_of(k)));
    const auto v = as_array<double>(k.attr("astype")("uint8"));
    return ce::window_keying(span_of(v), shape_of(k), spec);
}

const ce::Keying* ptr(const std::optional<ce::Keying>& k) { return k ? &*k : nullptr; }

py::object keyed_end(const fr::FrameSpec& s) {
    if (s.last_window_ends_frame()) return py::float_(s.keyed_end_pos());
    return py::int_(static_cast<std::int64_t>(s.keyed_end_pos()));
}

py::tuple keyed_span(const fr::FrameSpec& spec, const py::handle& lead_in_s) {
    try {
        const auto [lo, hi] = ce::keyed_span_pos(spec, as_float(lead_in_s));
        (void)hi;
        return py::make_tuple(lo, keyed_end(spec));
    } catch (const ce::LeadInRefusal&) {
        raise(PyExc_ValueError, ce::lead_in_refusal(shown(lead_in_s)));
    }
}

// carrier_fraction's argument as the C++ takes it: a float (numpy's float64
// included), a str or a Python int in int64. Anything else -- Fraction keeps
// a numpy integer as its own type and refuses a float32 -- is nullopt, for
// the caller to hand to Python's Fraction, and so is anything the C++
// refuses: a numerator or denominator past int64 ("2650.123456789123456789"
// has one before it is limited), a Unicode digit, NaN. The reference's
// Fraction then decides, with its own result or its own text.
std::optional<sstvae::qrss::Rational> fraction_of(const py::handle& x) {
    try {
        if (PyFloat_Check(x.ptr())) return ce::carrier_fraction(PyFloat_AsDouble(x.ptr()));
        if (py::isinstance<py::str>(x)) return ce::carrier_fraction(std::string_view(x.cast<std::string>()));
    } catch (const py::error_already_set&) {
        throw;
    } catch (const std::exception&) {
        return std::nullopt;
    }
    if (PyLong_Check(x.ptr())) {
        py::object i = py::reinterpret_steal<py::object>(PyNumber_Index(x.ptr()));
        if (!i) throw py::error_already_set();
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(i.ptr(), &overflow);
        if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
        if (overflow != 0) return std::nullopt;
        return ce::carrier_fraction(static_cast<std::int64_t>(v));
    }
    return std::nullopt;
}

void bind(py::module_& q) {
    py::module_ m = q.def_submodule("ce", "Port of sstvae.qrss.ce (waveform CE), the transmit side");
    held(frame_of_store, [&] { return py::object(q.attr("frame").attr("of")); });

    // --- constants
    m.attr("GRID_FS") = ce::GRID_FS;
    m.attr("HALF") = ce::HALF;
    m.attr("OOK_FIRST_KEYED_UNIT") = ce::OOK_FIRST_KEYED_UNIT;
    m.attr("OOK_LAST_KEYED_UNIT") = ce::OOK_LAST_KEYED_UNIT;
    m.attr("FSK_SHIFT_HZ") = ce::FSK_SHIFT_HZ;
    m.attr("_ALPHA") = ce::ALPHA_F;
    m.attr("_BETA") = ce::BETA_F;
    m.attr("PULSE_NORM") = ce::pulse_norm();
    m.attr("G0") = ce::g0();
    {
        const auto t = ce::pulse_table();
        m.attr("PULSE_TABLE") = to_numpy(t, false);
    }

    // --- pulse
    m.def("_rrc", [](const py::object& tau) { return elementwise<double>(tau, ce::rrc); }, py::arg("tau"));
    m.def("pulse", [](const py::object& tau) { return elementwise<double>(tau, ce::pulse); }, py::arg("tau"));
    m.def("_grid_step", [](const py::object& fs) { return ce::grid_step(as_int(fs)); }, py::arg("fs"));

    // --- symbols on the position axis
    m.def("_check_sym", [](const py::object& sym, const py::object& spec, const py::object& name) {
        return checked_sym(sym, spec_of(spec), py::str(name).cast<std::string>());
    }, py::arg("sym"), py::arg("spec"), py::arg("name") = "sym");
    m.def("at_positions", [](const py::object& sym, const py::object& spec) {
        const fr::FrameSpec s = spec_of(spec);
        return to_numpy(ce::at_positions(span_of(checked_sym(sym, s)), s));
    }, py::arg("sym"), py::arg("spec"));

    // --- phase
    m.def("_pulse_offsets", [](const py::object& d, bool square) {
        const auto a = as_array<double>(d);
        const auto shape = shape_of(a);
        std::vector<std::vector<double>> p(ce::N_OFFSETS, std::vector<double>(static_cast<std::size_t>(a.size())));
        for (std::size_t i = 0; i < static_cast<std::size_t>(a.size()); ++i) {
            const auto o = ce::pulse_offsets(a.data()[i], square);
            for (std::size_t j = 0; j < ce::N_OFFSETS; ++j) p[j][i] = o[j];
        }
        py::list out;
        for (std::size_t j = 0; j < ce::N_OFFSETS; ++j)
            out.append(py::make_tuple(static_cast<int>(j) - cfg::SPAN, shaped(p[j], shape)));
        return out;
    }, py::arg("d"), py::arg("square"), "the (j, p(d - j)) pairs, j = -8..8, as a list");

    m.def("phase_at", [](const py::object& tau_pos, const py::object& sym, const py::object& spec,
                         const py::object& var, const py::object& chunk) -> py::tuple {
        const auto tau = as_array<double>(tau_pos);
        const fr::FrameSpec s = spec_of(spec);
        const auto sy = checked_sym(sym, s);
        std::optional<py::array_t<double>> va;
        if (!var.is_none()) va = checked_sym(var, s, "var");
        const std::int64_t ch = as_index(chunk);
        if (ch == 0) raise(PyExc_ValueError, "range() arg 3 must not be zero");
        const auto shape = shape_of(tau);
        const auto nt = static_cast<std::size_t>(tau.size());
        if (ch < 0) {   // range(0, len, chunk) is empty: nothing is evaluated
            const std::vector<double> zero(nt, 0.0);
            return py::make_tuple(shaped(zero, shape), va ? py::object(shaped(zero, shape)) : py::none());
        }
        if (!va) return py::make_tuple(shaped(ce::phase_at(span_of(tau), span_of(sy), s), shape), py::none());
        auto [phi, v] = ce::phase_at(span_of(tau), span_of(sy), span_of(*va), s);
        return py::make_tuple(shaped(phi, shape), shaped(v, shape));
    }, py::arg("tau_pos"), py::arg("sym"), py::arg("spec"), py::arg("var") = py::none(),
       py::arg("chunk") = 1 << 19);

    m.def("_grid_sum", [](const py::object& fs, const py::object& n0, const py::object& n, const py::object& vals,
                          const py::object& table) {
        const std::int64_t f = as_int(fs);
        ce::grid_step(f);   // _grid_step(fs) first, then int(n0), int(n)
        const std::int64_t a = as_int(n0), b = as_int(n);
        const auto v = as_array<double>(vals), t = as_array<double>(table);
        return to_numpy(ce::grid_sum(f, a, b, span_of(v), span_of(t)));
    }, py::arg("fs"), py::arg("n0"), py::arg("n"), py::arg("vals"), py::arg("table"));

    m.def("phase_grid", [](const py::object& fs, const py::object& n0, const py::object& n, const py::object& sym,
                           const py::object& spec) {
        const fr::FrameSpec s = spec_of(spec);
        const auto sy = checked_sym(sym, s);
        const std::int64_t f = as_int(fs);
        ce::grid_step(f);
        return to_numpy(ce::phase_grid(f, as_int(n0), as_int(n), span_of(sy), s));
    }, py::arg("fs"), py::arg("n0"), py::arg("n"), py::arg("sym"), py::arg("spec"));

    m.def("var_grid", [](const py::object& fs, const py::object& n0, const py::object& n, const py::object& var,
                         const py::object& spec) {
        const fr::FrameSpec s = spec_of(spec);
        const auto va = checked_sym(var, s, "var");
        const std::int64_t f = as_int(fs);
        ce::grid_step(f);
        return to_numpy(ce::var_grid(f, as_int(n0), as_int(n), span_of(va), s));
    }, py::arg("fs"), py::arg("n0"), py::arg("n"), py::arg("var"), py::arg("spec"));

    // --- callsign windows
    m.def("_hann_int", [](const py::object& y) { return elementwise<double>(y, ce::hann_int); }, py::arg("y"));

    m.def("window_keying", [](const py::object& keying, const py::object& spec) -> py::object {
        const fr::FrameSpec s = spec_of(spec);
        const auto k = keying_of(keying, s);
        if (!k) return py::none();
        py::array_t<std::uint8_t> out({static_cast<py::ssize_t>(s.n_win()), static_cast<py::ssize_t>(cfg::CW_UNITS)});
        std::copy(k->begin(), k->end(), out.mutable_data());
        return out;
    }, py::arg("keying"), py::arg("spec"));

    m.def("window_units", [](const py::object& tau_pos, const py::object& spec, const py::object& w) -> py::object {
        const fr::FrameSpec s = spec_of(spec);
        const std::int64_t wi = as_index(w);
        const auto tau = as_array<double>(tau_pos);
        std::vector<double> out(static_cast<std::size_t>(tau.size()));
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = ce::window_units(tau.data()[i], s, wi);
        if (out.empty()) ce::window_units(0.0, s, wi);   // the index is checked with no samples too
        py::array r = shaped(out, shape_of(tau));
        if (tau.ndim() == 0) return r[py::tuple()];   // a numpy scalar, as 0-d arithmetic gives
        return r;
    }, py::arg("tau_pos"), py::arg("spec"), py::arg("w"));

    m.def("cw_phase_turns", [](const py::object& tau_pos, const py::object& spec, const py::object& keying,
                               const py::object& ook) {
        const auto tau = as_array<double>(tau_pos);
        const fr::FrameSpec s = spec_of(spec);
        const auto k = keying_of(keying, s);
        const bool o = py::bool_(ook).cast<bool>();
        const auto r = ce::cw_phase_turns(span_of(tau), s, ptr(k), o);
        const auto shape = shape_of(tau);
        return py::make_tuple(shaped(r.theta, shape), shaped(r.a, shape));
    }, py::arg("tau_pos"), py::arg("spec"), py::arg("keying"), py::arg("ook") = false);

    // --- keyed span
    m.def("keyed_span_pos", [](const py::object& spec, const py::object& lead_in_s) {
        return keyed_span(spec_of(spec), lead_in_s);
    }, py::arg("spec"), py::arg("lead_in_s") = 0.0);

    m.def("_ramp_envelope", [](const py::object& tau, const py::object& lo, const py::object& hi,
                               const py::object& ramp_sym) {
        const double l = as_float(lo), h = as_float(hi), r = as_float(ramp_sym);
        return elementwise<double>(tau, [&](double t) { return ce::ramp_envelope(t, l, h, r); });
    }, py::arg("tau"), py::arg("lo"), py::arg("hi"), py::arg("ramp_sym"));

    m.def("_turns_to_phasor", [](const py::object& turns) {
        return elementwise<cd>(turns, ce::turns_to_phasor);
    }, py::arg("turns"));

    // --- baseband and audio
    m.def("baseband", [](const py::object& sym, const py::object& spec, const py::object& fs,
                         const py::object& start_s, const py::object& n, const py::object& keying,
                         const py::object& ook, const py::object& lead_in_s, const py::object& ramp_s,
                         const py::object& time_scale, const py::object& chunk) {
        const fr::FrameSpec s = spec_of(spec);
        const auto sy = checked_sym(sym, s);
        const std::int64_t nn = as_int(n);
        const double f = as_float(fs), st = as_float(start_s);
        ce::BasebandOptions opt;
        opt.time_scale = as_float(time_scale);
        opt.lead_in_s = as_float(lead_in_s);
        try {
            ce::baseband_plan(s, f, st, opt);   // round(start_s*fs), the exact test, the keyed span
        } catch (const ce::LeadInRefusal&) {
            raise(PyExc_ValueError, ce::lead_in_refusal(shown(lead_in_s)));
        }
        opt.ramp_s = as_float(ramp_s);
        const auto k = keying_of(keying, s);
        opt.keying = ptr(k);
        opt.ook = py::bool_(ook).cast<bool>();
        if (nn < 0) raise(PyExc_ValueError, "negative dimensions are not allowed");
        opt.chunk = as_index(chunk);
        if (opt.chunk == 0) raise(PyExc_ValueError, "range() arg 3 must not be zero");
        if (opt.chunk < 0) return to_numpy(std::vector<cd>(static_cast<std::size_t>(nn)));   // the loop never runs
        std::vector<cd> z;
        {
            py::gil_scoped_release nogil;
            z = ce::baseband(span_of(sy), s, f, st, nn, opt);
        }
        return to_numpy(z);
    }, py::arg("sym"), py::arg("spec"), py::arg("fs"), py::arg("start_s"), py::arg("n"), py::kw_only(),
       py::arg("keying") = py::none(), py::arg("ook") = false, py::arg("lead_in_s") = 0.0,
       py::arg("ramp_s") = 0.05, py::arg("time_scale") = 1.0, py::arg("chunk") = 1 << 18);

    m.def("to_audio", [](const py::object& z, const py::object& fs, const py::object& num, const py::object& den,
                         const py::object& amplitude, const py::object& n0) {
        const auto za = as_array<cd>(z);
        if (za.ndim() != 1) raise(PyExc_ValueError, "to_audio takes a 1-D z, not shape " + fr::shape_repr(shape_of(za)));
        const std::int64_t f = as_int(fs);
        const std::int64_t nu = as_int(num), de = as_int(den);
        const double amp = as_float(amplitude);
        const std::int64_t start = as_int(n0);
        std::vector<double> out;
        {
            py::gil_scoped_release nogil;
            out = ce::to_audio(span_of(za), f, nu, de, amp, start);
        }
        return to_numpy(out);
    }, py::arg("z"), py::arg("fs"), py::arg("carrier_hz_num"), py::arg("carrier_hz_den") = 1,
       py::arg("amplitude") = 0.5, py::arg("n0") = 0);

    m.def("check_carrier", [](const py::object& carrier_hz) {
        return ce::check_carrier(as_float(carrier_hz), shown(carrier_hz));
    }, py::arg("carrier_hz"));

    m.def("carrier_fraction", [](const py::object& carrier_hz) -> py::tuple {
        if (py::isinstance(carrier_hz, fraction_type()))
            return py::make_tuple(py::object(carrier_hz.attr("numerator")), py::object(carrier_hz.attr("denominator")));
        if (const auto r = fraction_of(carrier_hz)) return py::make_tuple(py::int_(r->num()), py::int_(r->den()));
        // A Decimal, a long double, an int past int64 or a wrong type: the
        // reference's own Fraction(x).limit_denominator(2**20), so the result
        // and the refusal are the reference's whatever the Python version.
        py::object f = fraction_type()(carrier_hz).attr("limit_denominator")(ce::CARRIER_MAX_DEN);
        return py::make_tuple(py::object(f.attr("numerator")), py::object(f.attr("denominator")));
    }, py::arg("carrier_hz"));

    m.def("_mulmod", &ce::mulmod, py::arg("a"), py::arg("b"), py::arg("m"));
}

const Registrar reg("ce", 30, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
