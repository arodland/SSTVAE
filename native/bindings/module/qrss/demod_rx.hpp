// What the demodulator bindings (demod.cpp, ldemod.cpp) share: reading a
// track as the reference's demodulators read it, and the conversions of
// their arguments and results.
//
// The reference reads a track by attribute and only what it uses, and the
// suites lean on that: test_qrss_ldemod's GenieTrack has spec, pos, m, u,
// P, N, psi and classes.hdr_bits and nothing else, test_qrss_receiver's
// stubs not even classes. So `read_track` takes the fields a function
// reads, by the reference's rules for each:
//
// - always tr.spec (checked for its waveform first where the reference
//   asserts it, so a stub with nothing else still gets the AssertionError),
//   tr.pos, tr.m, tr.u, tr.P, tr.N, tr.psi as np.asarray arrays;
// - CE's classes: tr.classes (an AttributeError if absent, as demod's
//   data_estimates), then float(getattr(classes, "spread_rho", 0.0) or 0.0);
// - CE's template tr.c only where calibrate reads it, and as it reads it
//   (tr.c[g].imag: an AttributeError when absent, a TypeError for None
//   or a list); extract without calibration never touches it;
// - L's getattr(tr, "meas", None) -- an empty mask is not None: it fails
//   its first index, as numpy's does -- and getattr(getattr(tr,
//   "classes", None), "hdr_bits", None), the bits as the reference reads
//   them (hdr_bits_of below).
//
// A dict (the C++ -> Python direction's records) is read by key. The grid
// index of a position is p - tr.pos[0], which is what TrackResult.gi and
// every stub's gi compute.

#pragma once

#include "qrss/rx_types.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "qrss/qrss_config.hpp"
#include "qrss/demod.hpp"
#include "qrss/frame.hpp"
#include "qrss/pytext.hpp"

namespace sstvae_bindings::qrss::demod_rx {

namespace dm = sstvae::qrss::demod;
namespace fr = sstvae::qrss::frame;
using cd = std::complex<double>;

// getattr(o, name, None) (o[name] / None for a dict).
inline py::object opt_field(const py::handle& o, const char* name) {
    if (o.is_none()) return py::none();
    if (py::isinstance<py::dict>(o)) {
        py::dict d = py::reinterpret_borrow<py::dict>(o);
        return d.contains(name) ? py::object(d[name]) : py::object(py::none());
    }
    return py::getattr(o, name, py::none());
}

enum Need : unsigned {
    BASE = 0,
    CE_CLASSES = 1,   // tr.classes, its spread_rho
    CE_TEMPLATE = 2,  // tr.c, as calibrate reads it (tr.c[g].imag)
    L_MEAS = 4,       // getattr(tr, "meas", None)
    L_HDR = 8,        // classes' hdr_bits, if any
};

// classes.hdr_bits as the reference reads them, lazily: data_estimates_l
// takes np.asarray(bits) only on a frame with the spread copy, and
// frame.spread_signs then np.asarray(b, float64), refusing any shape but
// (2474,) ("hdr_bits must be 2474 values of 0/1") and computing 1 - 2b
// from whatever values it holds. The core carries the bits as uint8 0/1
// (trackbase's LClasses), so 2,474 values of exactly 0/1 -- any dtype --
// go through exactly, and anything else (another shape, values that are
// not 0/1, NaN, what numpy cannot convert) goes as no bits at all: the
// core's spread_signs then refuses it with that text exactly where the
// reference reads it, and never where the reference ignores the bits (a
// frame without the spread copy). Values other than 0/1 of the right
// shape are thus refused where the reference computes with them -- a
// deviation, but a refusal, never a different number (a uint8 cast had
// turned -1 into 255 and 0.5 into 0).
inline std::vector<std::uint8_t> hdr_bits_of(const py::handle& bits) {
    std::vector<double> v;
    try {
        const py::object np = rx::np_module();
        const py::array b = np.attr("asarray")(np.attr("asarray")(bits), py::arg("dtype") = "float64");
        if (b.ndim() != 1 || b.shape(0) != sstvae::qrss::config::N_HDR_BITS) return {};
        v = from_numpy<double>(b);
    } catch (py::error_already_set& e) {
        if (e.matches(PyExc_ValueError) || e.matches(PyExc_TypeError)) return {};
        throw;
    }
    std::vector<std::uint8_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i] != 0.0 && v[i] != 1.0) return {};
        out[i] = v[i] == 1.0 ? 1 : 0;
    }
    return out;
}

// The arrays a TrackView points into, owned.
struct Track {
    fr::FrameSpec spec;
    std::vector<std::int64_t> pos;
    std::vector<cd> m, u, c;
    std::vector<double> P, N, psi;
    std::vector<std::uint8_t> meas;
    bool meas_given = false;
    double spread_rho = 0.0;
    std::optional<std::vector<std::uint8_t>> hdr_bits;

    dm::TrackView view() const {
        dm::TrackView v;
        v.spec = &spec;
        v.pos = pos;
        v.m = m;
        v.u = u;
        v.c = c;
        v.P = P;
        v.N = N;
        v.psi = psi;
        v.meas = meas;
        v.meas_given = meas_given;
        v.spread_rho = spread_rho;
        if (hdr_bits) v.hdr_bits = std::span<const std::uint8_t>(*hdr_bits);
        return v;
    }
};

// tr.spec, as a native FrameSpec.
inline fr::FrameSpec track_spec(const py::handle& tr) { return rx::spec(rx::field(tr, "spec")); }

// The rest of the track, after its spec (`spec`), per `need`.
inline Track read_track(const py::handle& tr, fr::FrameSpec spec, unsigned need) {
    Track t{std::move(spec), {}, {}, {}, {}, {}, {}, {}, {}, false, 0.0, std::nullopt};
    t.pos = rx::vec<std::int64_t>(rx::field(tr, "pos"));
    t.m = rx::vec<cd>(rx::field(tr, "m"));
    t.u = rx::vec<cd>(rx::field(tr, "u"));
    t.P = rx::vec<double>(rx::field(tr, "P"));
    t.N = rx::vec<double>(rx::field(tr, "N"));
    t.psi = rx::vec<double>(rx::field(tr, "psi"));
    if (need & CE_CLASSES) {
        const py::object cls = rx::field(tr, "classes");
        py::object rho = py::isinstance<py::dict>(cls) ? opt_field(cls, "spread_rho")
                                                         : py::getattr(cls, "spread_rho", py::float_(0.0));
        if (rho.is_none()) rho = py::float_(0.0);
        t.spread_rho = rx::as_bool(rho) ? rx::as_double(rho) : 0.0;
    }
    if (need & CE_TEMPLATE && t.spec.waveform() == fr::Waveform::CE && fr::layout(t.spec)->ref.size() >= 16) {
        // calibrate's `tr.c[g].imag` with g an int64 array: tr.c is a
        // plain attribute (AttributeError when absent), and only what an
        // integer array indexes will do (None and a list are TypeErrors).
        const py::object c = rx::field(tr, "c");
        const py::object g0 = rx::np_module().attr("zeros")(0, py::arg("dtype") = "int64");
        const py::object probe = py::reinterpret_steal<py::object>(PyObject_GetItem(c.ptr(), g0.ptr()));
        if (!probe) throw py::error_already_set();
        (void)probe.attr("imag");
        t.c = rx::vec<cd>(c);
    }
    if (need & L_MEAS) {
        const py::object meas = opt_field(tr, "meas");
        if (!meas.is_none()) {   // np.asarray(meas, dtype=bool)[gi], lazily
            const py::array a = rx::np_module().attr("asarray")(meas, py::arg("dtype") = "bool");
            // Only a 1-D mask indexes as the reference's does; any other
            // shape goes as an empty one, which fails its first index
            // (IndexError where numpy's message would name the 0-d array,
            // or 2-D rows -- never a value numpy would not give).
            if (a.ndim() == 1) t.meas = from_numpy<std::uint8_t>(a.attr("view")("uint8"));
            t.meas_given = true;
        }
    }
    if (need & L_HDR) {
        const py::object bits = opt_field(opt_field(tr, "classes"), "hdr_bits");
        if (!bits.is_none()) t.hdr_bits = hdr_bits_of(bits);
    }
    return t;
}

inline Track read_track(const py::handle& tr, unsigned need) { return read_track(tr, track_spec(tr), need); }

// The estimator argument: its text and how the reference's f-string shows
// it ({estimator!r}). A str is passed on as it is; anything else can match
// neither "joint" nor "plain", so it is passed as "" with its repr shown.
struct Estimator {
    std::string name;
    std::string shown;
};
inline Estimator estimator_of(const py::handle& e) {
    if (py::isinstance<py::str>(e)) return {rx::as_string(e), {}};
    return {std::string(), py::repr(e).cast<std::string>()};
}

// scrambler's q: int(q) in uint64, else its ValueError.
inline std::uint64_t q_of(const py::handle& q) {
    py::object i = py::reinterpret_steal<py::object>(PyNumber_Long(q.ptr()));
    if (!i) throw py::error_already_set();
    const py::int_ qi = py::reinterpret_borrow<py::int_>(i);
    return to_u64(qi, "q must fit in uint64, not " + py::str(qi).cast<std::string>());
}

// extract's m argument: `if m is not None and m is not track.m: track.m = np.asarray(m)`.
inline void replace_m(const py::handle& m, const py::handle& track) {
    if (m.is_none()) return;
    const py::object cur = track.attr("m");
    if (m.is(cur)) return;
    track.attr("m") = rx::np_module().attr("asarray")(m);
}

// (z, w, llr, diag) as the reference returns them; `estimator` is the
// caller's own object (diag keeps it).
inline py::tuple extraction_to_py(const dm::Extraction& ex, const py::handle& estimator) {
    py::dict diag;
    diag["kappa"] = ex.kappa;
    diag["suspect"] = ex.suspect;
    diag["y"] = to_numpy(ex.y);
    diag["s2"] = to_numpy(ex.s2);
    diag["y_hdr"] = to_numpy(ex.y_hdr);
    diag["estimator"] = estimator;
    diag["llr_block"] = ex.llr_block ? py::object(to_numpy(*ex.llr_block)) : py::object(py::none());
    diag["llr_spread"] = ex.llr_spread ? py::object(to_numpy(*ex.llr_spread)) : py::object(py::none());
    return py::make_tuple(to_numpy(ex.z), to_numpy(ex.w), to_numpy(ex.llr), diag);
}

// frame::WrongWaveform -> AssertionError (frame.cpp registers the same; a
// second copy keeps these modules right on their own).
inline void register_wrong_waveform() {
    static const bool done = [] {
        py::register_local_exception_translator([](std::exception_ptr p) {
            try {
                if (p) std::rethrow_exception(p);
            } catch (const fr::WrongWaveform& e) {
                PyErr_SetObject(PyExc_AssertionError, rx::to_str(e.what()).ptr());
            }
        });
        return true;
    }();
    (void)done;
}

}  // namespace sstvae_bindings::qrss::demod_rx
