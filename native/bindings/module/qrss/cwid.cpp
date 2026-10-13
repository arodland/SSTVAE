// sstvae_native.qrss.cwid: native/core/qrss/cwid, sstvae/qrss/cwid.py
// under the reference's names and signatures (tests/qrss_native/cwid.py
// assigns them over cwid's attributes):
//
//   _unit_sums(tr, spec, z=None) -> (X0, X1, s2)
//   window_soft(ch, ch_fs, track, spec, ook=False)     float32[n_win, 176]
//   _fsk_only(spec, ook)
//   _ln_i0(x)
//   window_llr(ch, ch_fs, track, spec, ook=False)      float32[n_win, 176]
//   _combine(soft) -> (s, n)
//   decode_units(hard)
//   _patterns() -> (chars, pats)
//   _dp(gains, lens, n, max_chars, forbid=None) -> (best, back)
//   _trace(best, back) -> (text, score, items)
//   read_ml(llr, max_chars=8, margin=False), read(llr, ook=False)
//   match(soft, callsign), keying_type(ch, track, spec, callsign=None)
//
// A track is read by attribute, as the reference reads it: z_d (or the
// `ch` given, complex128 -- anything else NotImplementedError), chan's
// t0_index and fs, timing, and chan.t_s / chan.noise_psd_at, which are
// called back in Python (so a reference ChanCapture, its native rows, or
// a stand-in all work). A frame without windows returns before anything
// of the track is read, and keying_type of an L frame too, as in the
// reference. Unit values are 1-D or 2-D (rows of 176); other ranks
// NotImplementedError (0-D the reference's IndexError).

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

#include "qrss/cwid.hpp"
#include "qrss/morse.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace cw = sstvae::qrss::cwid;
namespace mo = sstvae::qrss::morse;
namespace qc = sstvae::qrss::config;

py::tuple shape_tuple(const std::vector<py::ssize_t>& s) {
    py::tuple t(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) t[i] = py::int_(s[i]);
    return t;
}

[[noreturn]] void not_implemented(const std::string& what) {
    PyErr_SetString(PyExc_NotImplementedError, ("native cwid: " + what).c_str());
    throw py::error_already_set();
}

std::string dtype_of(const py::array& a) { return py::str(a.attr("dtype")).cast<std::string>(); }

// --- callsigns (morse.py's refusals, worded with the caller's repr) ------------

[[noreturn]] void value_error(const py::str& msg) {
    PyErr_SetObject(PyExc_ValueError, msg.ptr());
    throw py::error_already_set();
}

std::string utf8_of(const py::handle& s) {
    py::object b = py::reinterpret_steal<py::object>(PyUnicode_AsEncodedString(s.ptr(), "utf-8", "surrogatepass"));
    if (!b) throw py::error_already_set();
    return b.cast<std::string>();
}

std::string callsign_of(const py::object& callsign) {
    if (!py::isinstance<py::str>(callsign))
        value_error(py::str("callsign must be a str, not {}").format(py::type::of(callsign).attr("__name__")));
    return utf8_of(callsign);
}

py::str chr_of(char32_t cp) {
    const Py_UCS4 c = static_cast<Py_UCS4>(cp);
    py::object o = py::reinterpret_steal<py::object>(PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, &c, 1));
    if (!o) throw py::error_already_set();
    return py::reinterpret_borrow<py::str>(o);
}

template <typename F>
auto refusing_as_python(const py::object& callsign, F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const mo::CallsignError& e) {
        switch (e.why()) {
        case mo::CallsignError::Why::length:
            value_error(py::str("callsign {!r} must be 1 to {} characters").format(callsign, qc::MAX_CALL_CHARS));
        case mo::CallsignError::Why::characters: {
            py::list bad;
            for (char32_t c : e.bad()) bad.append(chr_of(c));
            value_error(py::str("callsign {!r} has characters outside [A-Z0-9/]: {}").format(callsign, bad));
        }
        case mo::CallsignError::Why::units:
            value_error(py::str("callsign {!r} needs {} Morse units, more than the window's {}")
                            .format(callsign, e.units(), qc::CW_ROOM_UNITS));
        }
        throw;
    }
}

// --- the track ------------------------------------------------------------------

// A UnitSource over a Python track, holding what it points into.
struct PySource {
    py::array z;
    py::object chan;
    cw::UnitSource src;
};

PySource source_of(const py::object& tr, const py::object& z_given) {
    PySource ps;
    py::module_ np = rx::np_module();
    const py::object z_o = z_given.is_none() ? rx::field(tr, "z_d") : z_given;
    py::array z = py::array::ensure(np.attr("asarray")(z_o));
    if (!z) throw py::error_already_set();
    if (dtype_of(z) != "complex128") not_implemented("z of dtype " + dtype_of(z) + " (complex128 only)");
    if (z.ndim() != 1) not_implemented("z must be 1-D");
    ps.z = py::array::ensure(np.attr("ascontiguousarray")(z));
    if (!ps.z) throw py::error_already_set();
    ps.chan = rx::field(tr, "chan");
    ps.src.z = std::span<const std::complex<double>>(static_cast<const std::complex<double>*>(ps.z.data()),
                                                     static_cast<std::size_t>(ps.z.size()));
    ps.src.t0_index = rx::as_double(rx::field(ps.chan, "t0_index"));
    ps.src.fs = rx::as_double(rx::field(ps.chan, "fs"));
    ps.src.timing = rx::timing(rx::field(tr, "timing"));
    py::object chan = ps.chan;
    ps.src.noise_at = [chan](std::span<const double> n) {
        py::object t = chan.attr("t_s")(to_numpy(std::vector<double>(n.begin(), n.end())));
        py::array p = py::array::ensure(rx::np_module().attr("asarray")(chan.attr("noise_psd_at")(t)));
        if (!p) throw py::error_already_set();
        if (dtype_of(p) != "float64") not_implemented("noise_psd_at of dtype " + dtype_of(p) + " (float64 only)");
        return rx::vec<double>(p);
    };
    return ps;
}

py::array units_to_py(const cw::Units32& u) {
    py::array_t<float> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(u.rows), cw::CALL_N});
    std::copy(u.data.begin(), u.data.end(), out.mutable_data());
    return out;
}

template <typename F>
py::array windows(const py::object& ch, const py::object& track, const py::object& spec_o, bool ook, F&& f) {
    const rx::frame::FrameSpec spec = rx::spec(spec_o);
    if (spec.n_win() == 0) return units_to_py({});
    cw::fsk_only(spec, ook);
    const PySource ps = source_of(track, ch);
    return units_to_py(f(ps.src, spec, ook));
}

// --- unit values ------------------------------------------------------------------

// _combine(soft): (s, n) from 1-D or 2-D values.
cw::Combined combined_of(const py::object& o) {
    py::array a = py::array::ensure(rx::np_module().attr("asarray")(o, py::arg("dtype") = "float64"));
    if (!a) throw py::error_already_set();
    // A 0-d value: numpy's sum(axis=0) accepts it, and the reference's
    // s.shape[0] is what refuses it.
    if (a.ndim() == 0) throw py::index_error("tuple index out of range");
    if (a.ndim() > 2) not_implemented("unit values of rank " + std::to_string(a.ndim()));
    const std::vector<double> v = rx::vec<double>(a);
    if (a.ndim() == 1) return cw::combine(v, 1, v.size(), true);
    return cw::combine(v, static_cast<std::size_t>(a.shape(0)), static_cast<std::size_t>(a.shape(1)), false);
}

py::object opt_str(const std::optional<std::string>& s) {
    return s ? py::object(py::str(*s)) : py::object(py::none());
}

py::tuple read_ml(const py::object& llr, std::int64_t max_chars, bool margin) {
    const cw::ReadMl r = cw::read_ml(combined_of(llr), max_chars, margin);
    if (margin) return py::make_tuple(opt_str(r.text), r.z, r.margin);
    return py::make_tuple(opt_str(r.text), r.z);
}

std::vector<std::int64_t> pattern_lens() {
    std::vector<std::int64_t> lens;
    for (const cw::Pattern& p : cw::patterns()) lens.push_back(static_cast<std::int64_t>(p.units.size()));
    return lens;
}

void bind(py::module_& q) {
    rx::register_translators();
    py::module_ m = q.def_submodule("cwid", "Port of sstvae.qrss.cwid (native/core/qrss/cwid)");

    m.def("_unit_sums", [](const py::object& tr, const py::object& spec_o, const py::object& z) {
        const rx::frame::FrameSpec spec = rx::spec(spec_o);
        const PySource ps = source_of(tr, z);
        const cw::UnitSums us = cw::unit_sums(ps.src, spec);
        const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(us.n_win), cw::UNITS};
        py::object x0 = to_numpy(us.x0).attr("reshape")(shape_tuple(shape));
        py::object x1 = to_numpy(us.x1).attr("reshape")(shape_tuple(shape));
        py::object s2 = to_numpy(us.s2).attr("reshape")(shape_tuple(shape));
        return py::make_tuple(x0, x1, s2);
    }, py::arg("tr"), py::arg("spec"), py::arg("z") = py::none());
    m.def("window_soft", [](const py::object& ch, const py::object&, const py::object& track, const py::object& spec,
                            bool ook) {
        return windows(ch, track, spec, ook, [](const cw::UnitSource& s, const rx::frame::FrameSpec& sp, bool o) {
            return cw::window_soft(s, sp, o);
        });
    }, py::arg("ch"), py::arg("ch_fs"), py::arg("track"), py::arg("spec"), py::arg("ook") = false);
    m.def("_fsk_only", [](const py::object& spec, bool ook) { cw::fsk_only(rx::spec(spec), ook); }, py::arg("spec"),
          py::arg("ook"));
    m.def("_ln_i0", [](const py::object& x) -> py::object {
        py::array a = py::array::ensure(rx::np_module().attr("asarray")(x, py::arg("dtype") = "float64"));
        if (!a) throw py::error_already_set();
        std::vector<double> v = rx::vec<double>(a);
        for (double& e : v) e = cw::ln_i0(e);
        if (a.ndim() == 0) return rx::np_module().attr("float64")(v[0]);
        std::vector<py::ssize_t> shape(a.shape(), a.shape() + a.ndim());
        return to_numpy(v).attr("reshape")(shape_tuple(shape));
    }, py::arg("x"));
    m.def("window_llr", [](const py::object& ch, const py::object&, const py::object& track, const py::object& spec,
                           bool ook) {
        return windows(ch, track, spec, ook, [](const cw::UnitSource& s, const rx::frame::FrameSpec& sp, bool o) {
            return cw::window_llr(s, sp, o);
        });
    }, py::arg("ch"), py::arg("ch_fs"), py::arg("track"), py::arg("spec"), py::arg("ook") = false);
    m.def("_combine", [](const py::object& soft) {
        const cw::Combined c = combined_of(soft);
        return py::make_tuple(to_numpy(c.s), c.nrow);
    }, py::arg("soft"));
    m.def("decode_units", [](const py::object& hard) {
        py::array a = py::array::ensure(rx::np_module().attr("asarray")(hard, py::arg("dtype") = "bool"));
        if (!a) throw py::error_already_set();
        if (a.ndim() != 1) not_implemented("hard must be 1-D");
        return opt_str(cw::decode_units(rx::vec_bool(a)));
    }, py::arg("hard"));
    m.def("_patterns", [] {
        py::list chars, pats;
        for (const cw::Pattern& p : cw::patterns()) {
            chars.append(py::str(std::string(1, p.ch)));
            pats.append(to_numpy(p.units));
        }
        return py::make_tuple(chars, pats);
    });
    m.def("_dp", [](const py::object& gains_o, const py::object& lens_o, std::int64_t n, std::int64_t max_chars,
                    const py::object& forbid_o) {
        std::vector<std::vector<double>> gains;
        for (const py::handle g : py::reinterpret_borrow<py::iterable>(gains_o)) {
            py::array a = py::array::ensure(rx::np_module().attr("asarray")(g, py::arg("dtype") = "float64"));
            if (!a) throw py::error_already_set();
            if (a.ndim() != 1 || a.shape(0) != n + 1) not_implemented("gains must be rows of n + 1");
            gains.push_back(rx::vec<double>(a));
        }
        const std::vector<std::int64_t> lens = rx::vec<std::int64_t>(lens_o);
        std::optional<std::pair<std::int64_t, std::int64_t>> forbid;
        if (!forbid_o.is_none()) {
            py::tuple t(forbid_o);
            if (t.size() != 2) not_implemented("forbid must be (character, start)");
            forbid = std::make_pair(rx::as_int(t[0]), rx::as_int(t[1]));
        }
        const cw::Dp d = cw::dp(gains, lens, n, max_chars, forbid);
        py::object best = to_numpy(d.best).attr("reshape")(py::make_tuple(d.kmax, d.n1));
        py::object back = to_numpy(d.back).attr("reshape")(py::make_tuple(d.kmax, d.n1, 2));
        return py::make_tuple(best, back);
    }, py::arg("gains"), py::arg("lens"), py::arg("n"), py::arg("max_chars"), py::arg("forbid") = py::none());
    m.def("_trace", [](const py::object& best_o, const py::object& back_o) {
        py::array best = py::array::ensure(rx::np_module().attr("asarray")(best_o, py::arg("dtype") = "float64"));
        py::array back = py::array::ensure(rx::np_module().attr("asarray")(back_o, py::arg("dtype") = "int64"));
        if (!best || !back) throw py::error_already_set();
        if (best.ndim() != 2 || back.ndim() != 3 || back.shape(0) != best.shape(0) || back.shape(1) != best.shape(1) ||
            back.shape(2) != 2)
            not_implemented("best must be [k, n + 1] and back [k, n + 1, 2]");
        cw::Dp d;
        d.kmax = static_cast<std::size_t>(best.shape(0));
        d.n1 = static_cast<std::size_t>(best.shape(1));
        d.best = rx::vec<double>(best);
        d.back = rx::vec<std::int64_t>(back);
        const cw::Trace t = cw::trace(d, pattern_lens());
        py::list items;
        for (const auto& [c, st] : t.items) items.append(py::make_tuple(c, st));
        return py::make_tuple(opt_str(t.text), t.score, items);
    }, py::arg("best"), py::arg("back"));
    m.def("read_ml", &read_ml, py::arg("llr"), py::arg("max_chars") = 8, py::arg("margin") = false);
    m.def("read", [](const py::object& llr, const py::object&) { return opt_str(cw::read(combined_of(llr))); },
          py::arg("llr"), py::arg("ook") = false);
    m.def("match", [](const py::object& soft, const py::object& callsign) {
        const cw::Combined c = combined_of(soft);
        const std::string call = callsign_of(callsign);
        return refusing_as_python(callsign, [&] { return cw::match(c, call); });
    }, py::arg("soft"), py::arg("callsign"));
    m.def("keying_type", [](const py::object& ch, const py::object& track, const py::object& spec_o,
                            const py::object& callsign) {
        const rx::frame::FrameSpec spec = rx::spec(spec_o);
        if (spec.n_win() == 0 || spec.waveform() == rx::frame::Waveform::L) return std::string("fsk");
        const PySource ps = source_of(track, ch);
        if (callsign.is_none()) return cw::keying_type(ps.src, spec);
        const std::string call = callsign_of(callsign);
        return refusing_as_python(callsign, [&] { return cw::keying_type(ps.src, spec, call); });
    }, py::arg("ch"), py::arg("track"), py::arg("spec"), py::arg("callsign") = py::none());
}

const Registrar reg("cwid", 78, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
