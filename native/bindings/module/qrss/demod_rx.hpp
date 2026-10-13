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
// - CE's template tr.c where calibrate reads it (empty when absent);
// - L's getattr(tr, "meas", None) and getattr(getattr(tr, "classes",
//   None), "hdr_bits", None).
//
// A dict (the C++ -> Python direction's records) is read by key. The grid
// index of a position is p - tr.pos[0], which is what TrackResult.gi and
// every stub's gi compute.

#pragma once

#include "qrss/rx_types.hpp"

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
    CE_TEMPLATE = 2,  // tr.c, if present
    L_MEAS = 4,       // getattr(tr, "meas", None)
    L_HDR = 8,        // classes' hdr_bits, if any
};

// The arrays a TrackView points into, owned.
struct Track {
    fr::FrameSpec spec;
    std::vector<std::int64_t> pos;
    std::vector<cd> m, u, c;
    std::vector<double> P, N, psi;
    std::vector<std::uint8_t> meas;
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
        v.spread_rho = spread_rho;
        if (hdr_bits) v.hdr_bits = std::span<const std::uint8_t>(*hdr_bits);
        return v;
    }
};

// tr.spec, as a native FrameSpec.
inline fr::FrameSpec track_spec(const py::handle& tr) { return rx::spec(rx::field(tr, "spec")); }

// The rest of the track, after its spec (`spec`), per `need`.
inline Track read_track(const py::handle& tr, fr::FrameSpec spec, unsigned need) {
    Track t{std::move(spec), {}, {}, {}, {}, {}, {}, {}, {}, 0.0, std::nullopt};
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
    if (need & CE_TEMPLATE) {
        const py::object c = opt_field(tr, "c");
        if (!c.is_none()) t.c = rx::vec<cd>(c);
    }
    if (need & L_MEAS) {
        const py::object meas = opt_field(tr, "meas");
        if (!meas.is_none()) t.meas = rx::vec_bool(meas);
    }
    if (need & L_HDR) {
        const py::object bits = opt_field(opt_field(tr, "classes"), "hdr_bits");
        if (!bits.is_none()) t.hdr_bits = rx::vec<std::uint8_t>(bits);
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
