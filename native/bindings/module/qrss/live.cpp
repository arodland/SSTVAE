// sstvae_native.qrss.live: bindings of native/core/qrss/live (sstvae/qrss/live.py
// up to LiveListener). Names and arguments follow the reference's;
// tests/qrss_native/live.py assigns them over it (the functions that need
// the reference's defaults or a `typed` flag through small wrappers):
//
//   utc_iso(t), frame_seconds(spec), in_band(f_hz, waveform=0),
//   centre_of(f_hz, waveform), _waveform_id(w), _due_waveform(d)
//   FeRing(span_s=RING_S): n, end, write(n0, x), read(a, b), _clear(a, b),
//       x, ok (copies of the storage)
//   _spans(mask, fs, origin)
//   capture_read(ring, q, spec, lead_s, tail_s, typed)
//   capture_prepare(x, ok, origin, q, spec, seed, noise, typed)
//   slot_capture(ring, q, spec, seed, lead_s, tail_s, noise, typed)
//   finish_receive(x, ok, origin, q, spec, estimator, dets, hints, noise, typed)
//   DialLog(path): path, points, refresh(), spans(a, b), by_dial(a, b),
//       dial_at(t), span_at(t)
//   same_dial(a, b), restrict(ok, origin, q, spans)
//
// and, for the parity tests only: _tile_json, _tile_roundtrip,
// _live_config, _live_config_defaults, _default_noise, _parse_dial_text,
// _finish_gate, _finish_branch, _due_waveform_json, _waveform_id_json,
// _utc_iso_int. Tile, SlotState and LiveConfig stay the reference's
// dataclasses (the listener sets their fields freely and pickles none of
// them; a pybind class would not stand in for that), and their C++ twins
// are tested against them through those windows.
//
// `noise` is the fill for what was not heard: a callable noise(seed, n)
// returning n complex samples (the parity tests' numpy default_rng(seed),
// which makes a capture the reference's bit for bit), or None for the
// C++'s own generator (live.hpp: default_noise). It is called with the
// GIL held, before any C++ runs without it, always (n may be 0: the
// reference makes its generator from the seed whether or not it draws).
//
// `typed` (the row says every stage the call reaches is the native row or
// the reference's own: frontend.blank and .normalise, live.frame_seconds,
// live.slot_t0, live.capture_prepare, and for finish_receive the slot
// receiver's rows with their own `typed` true) runs the call in C++ with
// the GIL released. Otherwise every stage is called as the reference calls
// it, through the module attributes (live.frame_seconds,
// live.capture_prepare, live.receiver.receive_slot, ...: resolved at each
// call, with the reference's arguments), and the C++ does the rest -- the
// ring arithmetic, the masks and spans, the gates -- holding the GIL. An
// argument that does not convert (a stub prep, a non-bool mask, a stub
// detection) takes the second way too.

#include "qrss/rx_types.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "qrss/acquire.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/frontend.hpp"
#include "qrss/live.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/receiver.hpp"
#include "qrss/rx_api.hpp"
#include "qrss/types.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace lv = sstvae::qrss::live;
namespace fr = sstvae::qrss::frame;
namespace aq = sstvae::qrss::acquire;
namespace api = sstvae::qrss::api;
namespace rc = sstvae::qrss::receiver;
namespace fe = sstvae::qrss::frontend;
namespace cfg = sstvae::qrss::config;
namespace pyjson = sstvae::qrss::pyjson;
namespace types = sstvae::qrss::types;
using cf = std::complex<float>;
using namespace pybind11::literals;

// --- small things ----------------------------------------------------------------------------

py::object steal(PyObject* o) {
    if (!o) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(o);
}

// sstvae.qrss.live's attribute `name`, read at each call (a test may have
// replaced it)
py::object live_attr(const char* name) { return py::module_::import("sstvae.qrss.live").attr(name); }

bool is_seq(const py::handle& o) { return py::isinstance<py::tuple>(o) || py::isinstance<py::list>(o); }

// list(iterable)
std::vector<py::object> objects(const py::handle& o) {
    std::vector<py::object> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(o)))
        out.push_back(py::reinterpret_borrow<py::object>(h));
    return out;
}

// `a, b, ... = h` for n names, as Python unpacks
std::vector<py::object> unpack(const py::handle& h, std::size_t n) {
    PyObject* it = PyObject_GetIter(h.ptr());
    if (!it) {
        PyTypeObject* t = Py_TYPE(h.ptr());
        if (PyErr_ExceptionMatches(PyExc_TypeError) && t->tp_iter == nullptr && !PySequence_Check(h.ptr())) {
            PyErr_Clear();
            throw py::type_error(std::string("cannot unpack non-iterable ") + t->tp_name + " object");
        }
        throw py::error_already_set();
    }
    const py::object iter = py::reinterpret_steal<py::object>(it);
    std::vector<py::object> out;
    while (PyObject* x = PyIter_Next(iter.ptr())) {
        if (out.size() == n) {
            Py_DECREF(x);
            throw py::value_error("too many values to unpack (expected " + std::to_string(n) + ")");
        }
        out.push_back(py::reinterpret_steal<py::object>(x));
    }
    if (PyErr_Occurred()) throw py::error_already_set();
    if (out.size() < n)
        throw py::value_error("not enough values to unpack (expected " + std::to_string(n) + ", got " +
                              std::to_string(out.size()) + ")");
    return out;
}

// operator.index(o) as int64 (OverflowError past it)
std::int64_t index_of(const py::handle& o) {
    const py::object i = steal(PyNumber_Index(o.ptr()));
    const long long v = PyLong_AsLongLong(i.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

bool truth(const py::handle& o) { return rx::as_bool(o); }

// A number as a double the way arithmetic takes one (__float__ or
// __index__): TypeError for a str, where rx::as_double's float() would
// parse it.
double real(const py::handle& o) {
    const double v = PyFloat_AsDouble(o.ptr());
    if (v == -1.0 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

// a <op> b, Python's comparison, as a bool
bool compare(const py::handle& a, const py::handle& b, int op) {
    const py::object r = steal(PyObject_RichCompare(a.ptr(), b.ptr(), op));
    return truth(r);
}

py::object opt_int_py(const std::optional<std::int64_t>& v) {
    return v ? py::object(py::int_(*v)) : py::object(py::none());
}

std::optional<std::int64_t> opt_int(const py::handle& o) {
    if (o.is_none()) return std::nullopt;
    return rx::as_int(o);
}

// erase spans as _spans returns them: (start, end) of np.float64
py::list spans_py(const lv::Erase& e) {
    const py::object f64 = rx::np_module().attr("float64");
    py::list out;
    for (const auto& [a, b] : e) out.append(py::make_tuple(f64(a), f64(b)));
    return out;
}

// the system_error utc_iso gives past gmtime's range, as the OSError
// CPython raises there (errno, strerror)
[[noreturn]] void raise_os_error(const std::system_error& e) {
    errno = e.code().value();
    PyErr_SetFromErrno(PyExc_OSError);
    throw py::error_already_set();
}

// --- arrays -------------------------------------------------------------------------------------

using CArr = py::array_t<cf, py::array::c_style | py::array::forcecast>;

// np.asarray(x, complex64), one dimension
CArr samples_of(const py::handle& x) {
    const py::object a = rx::np_module().attr("ascontiguousarray")(x, "dtype"_a = "complex64");
    if (py::array(a).ndim() != 1) throw py::value_error("expected a 1-D array of FE samples");
    return CArr(a);
}

// A 1-D numpy bool array as 0/1 bytes (nullopt: not one)
std::optional<py::array_t<std::uint8_t>> bool_bytes(const py::handle& ok) {
    if (!py::isinstance<py::array>(ok)) return std::nullopt;
    const py::array a = py::reinterpret_borrow<py::array>(ok);
    if (a.dtype().kind() != 'b' || a.ndim() != 1) return std::nullopt;
    const py::object c = rx::np_module().attr("ascontiguousarray")(a);
    return py::array_t<std::uint8_t>(c.attr("view")("uint8"));
}

// np.asarray(ok, bool) as 0/1 bytes, one dimension
py::array_t<std::uint8_t> mask_of(const py::handle& ok) {
    if (auto b = bool_bytes(ok)) return *b;
    const py::object a = rx::np_module().attr("ascontiguousarray")(ok, "dtype"_a = "bool");
    if (py::array(a).ndim() != 1) throw py::value_error("expected a 1-D mask");
    return py::array_t<std::uint8_t>(a.attr("view")("uint8"));
}

std::span<const std::uint8_t> bytes_span(const py::array_t<std::uint8_t>& a) {
    return {a.data(), static_cast<std::size_t>(a.size())};
}

std::span<const cf> cf_span(const CArr& a) { return {a.data(), static_cast<std::size_t>(a.size())}; }

py::array bool_array(std::span<const std::uint8_t> v) {
    py::array_t<bool> out(static_cast<py::ssize_t>(v.size()));
    auto* d = reinterpret_cast<std::uint8_t*>(out.mutable_data());
    std::copy(v.begin(), v.end(), d);
    return out;
}

// --- the typed way --------------------------------------------------------------------------------

// Raised by a conversion that is to fall through to the Python way.
struct Untyped {};

template <class F>
auto convert(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const py::error_already_set&) {
        throw Untyped{};
    } catch (const std::exception&) {
        throw Untyped{};
    }
}

// A native FrameSpec, or the reference's (by its fields)
std::vector<fr::FrameSpec> selection(const py::handle& spec) {
    std::vector<fr::FrameSpec> out;
    if (is_seq(spec)) {
        for (const py::handle h : py::reinterpret_borrow<py::iterable>(spec)) out.push_back(rx::spec(h));
    } else {
        out.push_back(rx::spec(spec));
    }
    return out;
}

std::int64_t typed_int(const py::handle& o) {
    if (!py::isinstance<py::int_>(o) || py::isinstance<py::bool_>(o)) throw Untyped{};
    return rx::as_int(o);
}

double typed_number(const py::handle& o) {
    if (!(py::isinstance<py::int_>(o) || py::isinstance<py::float_>(o)) || py::isinstance<py::bool_>(o))
        throw Untyped{};
    return rx::as_double(o);
}

std::vector<aq::Hint> typed_hints(const py::object& hints) {
    std::vector<aq::Hint> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(hints))) {
        aq::Hint x;
        if (is_seq(h)) {
            const std::vector<py::object> fw = objects(h);
            if (fw.size() != 2) throw Untyped{};
            x.f_hz = rx::as_double(fw[0]);
            x.waveform = rx::as_int(fw[1]);
        } else {
            x.f_hz = rx::as_double(h);
        }
        out.push_back(x);
    }
    return out;
}

// Detections, each its own object (one listed twice is the Python way's:
// the receiver tells them apart by identity)
std::vector<types::Detection> typed_dets(const py::object& dets) {
    const std::vector<py::object> objs = objects(dets);
    for (std::size_t i = 0; i < objs.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (objs[i].is(objs[j])) throw Untyped{};
    std::vector<types::Detection> out;
    for (const py::object& o : objs) out.push_back(rx::detection(o));
    return out;
}

py::list passes_py(const std::vector<types::PassResult>& v) {
    py::list out;
    for (const auto& p : v) out.append(rx::to_py(p));
    return out;
}

// The noise for `nmiss` samples, drawn now (GIL held) from the Python
// callable, as a seam that hands it over once; an empty seam for None.
lv::NoiseFn noise_seam(const py::object& noise, const py::object& seed_obj, std::size_t nmiss) {
    if (noise.is_none()) return {};
    const py::object z = noise(seed_obj, py::int_(nmiss));
    auto v = std::make_shared<std::vector<cf>>(rx::vec<cf>(z));
    if (v->size() != nmiss)
        throw py::value_error("the noise gave " + std::to_string(v->size()) + " samples for " + std::to_string(nmiss));
    return [v](std::int64_t, std::size_t n) {
        if (n != v->size()) throw std::invalid_argument("the noise was drawn for another length");
        return *v;
    };
}

std::size_t count_missing(std::span<const std::uint8_t> ok) {
    std::size_t n = 0;
    for (const std::uint8_t b : ok) n += b ? 0 : 1;
    return n;
}

// `q if seed is None else seed`, and its int64 for the C++ generator
py::object seed_or_q(const py::object& seed, const py::object& q) { return seed.is_none() ? q : seed; }

py::tuple capture_py(const lv::SlotCapture& c) {
    return py::make_tuple(rx::to_py(c.prep), spans_py(c.erase), py::float_(c.heard));
}

// The front end's stages as the reference calls them (live.frontend.blank
// / .normalise, resolved now), for the Python way.
lv::PrepareSeams python_seams() {
    const py::object frontend = live_attr("frontend");
    lv::PrepareSeams s;
    s.blank = [frontend](std::span<const cf> x) {
        const py::object r = frontend.attr("blank")(to_numpy(x));
        const std::vector<py::object> fk = unpack(r, 2);
        fe::Blanked b;
        b.fe = rx::vec<cf>(fk[0]);
        b.keep = rx::vec<float>(fk[1]);
        return b;
    };
    s.normalise = [frontend](std::span<const cf> x, std::span<const float> keep) {
        const py::object r = frontend.attr("normalise")(to_numpy(x), to_numpy(keep));
        const std::vector<py::object> fl = unpack(r, 2);
        fe::Normalised nm;
        nm.fe = rx::vec<cf>(fl[0]);
        return nm;
    };
    return s;
}

// --- capture_read ----------------------------------------------------------------------------------

// (x, ok) of FE [a, b) from a native ring, in C++ with the GIL released
py::tuple ring_read(const lv::FeRing& r, std::int64_t a, std::int64_t b) {
    if (b < a) throw py::value_error("negative dimensions are not allowed");
    const auto n = static_cast<py::ssize_t>(b - a);
    py::array_t<cf> x(n);
    py::array_t<bool> ok(n);
    std::span<cf> xs(x.mutable_data(), static_cast<std::size_t>(n));
    std::span<std::uint8_t> oks(reinterpret_cast<std::uint8_t*>(ok.mutable_data()), static_cast<std::size_t>(n));
    {
        py::gil_scoped_release nogil;
        r.read_into(a, xs, oks);
    }
    return py::make_tuple(x, ok);
}

py::tuple capture_read_py(const py::object& ring, const py::object& q, const py::object& spec,
                          const py::object& lead_s, const py::object& tail_s, bool typed) {
    if (typed) {
        try {
            const std::vector<fr::FrameSpec> sel = convert([&] { return selection(spec); });
            const std::int64_t qi = convert([&] { return typed_int(q); });
            const double lead = convert([&] { return typed_number(lead_s); });
            const double tail = convert([&] { return typed_number(tail_s); });
            const lv::CaptureSpan s = lv::capture_span(qi, lv::frame_seconds(sel), lead, tail);
            py::tuple xo;
            if (py::isinstance<lv::FeRing>(ring))
                xo = ring_read(ring.cast<const lv::FeRing&>(), s.a, s.b);
            else
                xo = py::tuple(ring.attr("read")(py::int_(s.a), py::int_(s.b)));
            const std::vector<py::object> xk = unpack(xo, 2);
            return py::make_tuple(xk[0], xk[1], py::int_(s.origin));
        } catch (const Untyped&) {
        }
    }
    // the reference's arithmetic on its own objects:
    // a = t0 * FE_FS - int(round(lead_s * FE_FS))
    // b = t0 * FE_FS + int(round((frame_seconds(spec) + tail_s) * FE_FS))
    const py::object builtins = py::module_::import("builtins");
    const py::object t0 = live_attr("slot_t0")(q);
    const py::object fs = py::int_(cfg::FE_FS);
    auto mul = [](const py::handle& x, const py::handle& y) { return steal(PyNumber_Multiply(x.ptr(), y.ptr())); };
    auto add = [](const py::handle& x, const py::handle& y) { return steal(PyNumber_Add(x.ptr(), y.ptr())); };
    auto sub = [](const py::handle& x, const py::handle& y) { return steal(PyNumber_Subtract(x.ptr(), y.ptr())); };
    auto iround = [&](const py::handle& v) { return builtins.attr("int")(builtins.attr("round")(v)); };
    const py::object base = mul(t0, fs);
    const py::object a = sub(base, iround(mul(lead_s, fs)));
    const py::object b = add(base, iround(mul(add(live_attr("frame_seconds")(spec), tail_s), fs)));
    const std::vector<py::object> xk = unpack(ring.attr("read")(a, b), 2);
    return py::make_tuple(xk[0], xk[1], sub(base, a));
}

// --- capture_prepare -------------------------------------------------------------------------------

py::object capture_prepare_py(const py::object& x, const py::object& ok, const py::object& origin,
                              const py::object& q, const py::object& spec, const py::object& seed,
                              const py::object& noise, bool typed) {
    const py::object sd = seed_or_q(seed, q);
    if (typed) {
        try {
            const std::vector<fr::FrameSpec> sel = convert([&] { return selection(spec); });
            const CArr xa = convert([&] { return samples_of(x); });
            const auto okb = bool_bytes(ok);
            if (!okb) throw Untyped{};
            const std::int64_t org = convert([&] { return typed_int(origin); });
            const std::int64_t qi = convert([&] { return typed_int(q); });
            const std::optional<std::int64_t> si = noise.is_none() ? std::optional<std::int64_t>(rx::as_int(sd))
                                                                   : std::nullopt;
            const std::span<const std::uint8_t> oks = bytes_span(*okb);
            lv::PrepareSeams seams;
            seams.noise = noise_seam(noise, sd, count_missing(oks));
            lv::SlotCapture c;
            {
                py::gil_scoped_release nogil;
                c = lv::capture_prepare(cf_span(xa), oks, org, qi, sel, si, seams);
            }
            return capture_py(c);
        } catch (const Untyped&) {
        }
    }
    // the stages through the module, the rest the C++ (GIL held)
    const double frame_s = rx::as_double(live_attr("frame_seconds")(spec));
    const CArr xa = samples_of(x);
    const py::array_t<std::uint8_t> okb = mask_of(ok);
    const std::span<const std::uint8_t> oks = bytes_span(okb);
    lv::PrepareSeams seams = python_seams();
    seams.noise = noise_seam(noise, sd, count_missing(oks));
    const std::optional<std::int64_t> si =
        noise.is_none() ? std::optional<std::int64_t>(rx::as_int(sd)) : std::nullopt;
    const lv::SlotCapture c = lv::capture_prepare(cf_span(xa), oks, index_of(origin), rx::as_int(q), frame_s, si, seams);
    return capture_py(c);
}

py::object slot_capture_py(const py::object& ring, const py::object& q, const py::object& spec,
                           const py::object& seed, const py::object& lead_s, const py::object& tail_s,
                           const py::object& noise, bool typed) {
    if (typed && py::isinstance<lv::FeRing>(ring)) {
        try {
            const std::vector<fr::FrameSpec> sel = convert([&] { return selection(spec); });
            const std::int64_t qi = convert([&] { return typed_int(q); });
            const double lead = convert([&] { return typed_number(lead_s); });
            const double tail = convert([&] { return typed_number(tail_s); });
            const py::object sd = seed_or_q(seed, q);
            const std::optional<std::int64_t> si =
                noise.is_none() ? std::optional<std::int64_t>(rx::as_int(sd)) : std::nullopt;
            const lv::FeRing& r = ring.cast<const lv::FeRing&>();
            lv::SlotRead rd;
            {
                py::gil_scoped_release nogil;
                rd = lv::capture_read(r, qi, sel, lead, tail);
            }
            lv::PrepareSeams seams;
            seams.noise = noise_seam(noise, sd, count_missing(rd.ok));
            lv::SlotCapture c;
            {
                py::gil_scoped_release nogil;
                c = lv::capture_prepare(rd.x, rd.ok, rd.origin, qi, sel, si, seams);
            }
            return capture_py(c);
        } catch (const Untyped&) {
        }
    }
    // capture_prepare(*capture_read(ring, q, spec, lead_s, tail_s), q, spec, seed)
    const std::vector<py::object> r =
        objects(live_attr("capture_read")(ring, q, spec, lead_s, tail_s));
    py::list args;
    for (const py::object& o : r) args.append(o);
    args.append(q);
    args.append(spec);
    args.append(seed);
    return live_attr("capture_prepare")(*py::tuple(args));
}

// --- finish_receive --------------------------------------------------------------------------------

struct FinishConsts {
    py::object min_heard, max_candidates, budget_s;
};

FinishConsts finish_consts() {
    return {live_attr("FINISH_SEARCH_MIN_HEARD"), live_attr("FINISH_MAX_CANDIDATES"), live_attr("FINISH_BUDGET_S")};
}

py::tuple finish_typed(const py::object& x, const py::object& ok, const py::object& origin, const py::object& q,
                       const py::object& spec, const py::object& estimator, const py::object& dets,
                       const py::object& hints, const py::object& noise, const FinishConsts& k) {
    const std::vector<fr::FrameSpec> sel = convert([&] { return selection(spec); });
    const CArr xa = convert([&] { return samples_of(x); });
    const auto okb = bool_bytes(ok);
    if (!okb) throw Untyped{};
    const std::int64_t org = convert([&] { return typed_int(origin); });
    const std::int64_t qi = convert([&] { return typed_int(q); });
    if (!py::isinstance<py::str>(estimator)) throw Untyped{};
    const std::vector<types::Detection> d = convert([&] { return typed_dets(dets); });
    const std::vector<aq::Hint> h = convert([&] { return typed_hints(hints); });
    lv::FinishOptions opt;
    opt.min_heard = convert([&] { return typed_number(k.min_heard); });
    api::ReceiveConfig config = lv::finish_config(sel, rx::as_string(estimator));
    opt.max_candidates = convert([&] { return typed_int(k.max_candidates); });
    opt.budget_s = convert([&] { return typed_number(k.budget_s); });
    const py::module_ rules = py::module_::import("sstvae.qrss.rules");
    config.old_edge_noise = truth(rules.attr("old")(rules.attr("EDGE_NOISE")));
    config.old_gating = truth(rules.attr("old")(rules.attr("GATING")));
    const std::span<const std::uint8_t> oks = bytes_span(*okb);
    // the noise only when the receive will prepare the capture, as the
    // reference makes its generator only there
    const lv::FinishGate g = lv::finish_gate(oks, org, lv::frame_seconds(sel), !d.empty(), !h.empty(), opt.min_heard);
    const py::object sd = seed_or_q(py::none(), q);
    if (g.receive) {
        opt.seams.noise = noise_seam(noise, sd, count_missing(oks));
        if (noise.is_none()) opt.seed = rx::as_int(sd);
    }
    lv::FinishResult r;
    {
        py::gil_scoped_release nogil;
        r = lv::finish_receive(cf_span(xa), oks, org, qi, config, d, h, opt);
    }
    return py::make_tuple(passes_py(r.passes), py::float_(r.heard));
}

// The reference's finish_receive over the Python objects: every stage
// through live's module attributes, the gate the C++'s.
py::tuple finish_python(const py::object& x, const py::object& ok, const py::object& origin, const py::object& q,
                        const py::object& spec, const py::object& estimator, const py::object& dets,
                        const py::object& hints, const FinishConsts& k) {
    const double frame_s = rx::as_double(live_attr("frame_seconds")(spec));
    const py::array_t<std::uint8_t> okb = mask_of(ok);
    const lv::FinishGate g = lv::finish_gate(bytes_span(okb), index_of(origin), frame_s, truth(dets), truth(hints),
                                             rx::as_double(k.min_heard));
    if (!g.receive) return py::make_tuple(py::list(), py::float_(g.heard));   // nothing to receive
    const std::vector<py::object> pe = unpack(live_attr("capture_prepare")(x, ok, origin, q, spec), 3);
    const py::object& prep = pe[0];
    const py::object& erase = pe[1];
    const py::object& heard = pe[2];
    const py::object R = live_attr("receiver");
    py::object passes = py::list();
    if (compare(heard, k.min_heard, Py_GE)) {
        passes = R.attr("receive_slot")(prep, spec, "estimator"_a = estimator, "erase_s"_a = erase,
                                        "max_candidates"_a = k.max_candidates, "budget_s"_a = k.budget_s,
                                        "hints"_a = hints);
    } else if ((truth(dets) || truth(hints)) && compare(heard, py::float_(0.0), Py_GT)) {
        py::object more = py::list();
        if (truth(hints)) {
            py::list known;
            if (truth(R.attr("is_joint")(spec))) {
                for (const py::object& d : objects(dets)) known.append(R.attr("centre_hz")(d));
            } else {
                for (const py::object& d : objects(dets)) known.append(d.attr("f_hz"));
            }
            more = R.attr("detect")(prep, spec, "live_only"_a = true, "erase_s"_a = erase, "hints"_a = hints,
                                    "skip_hz"_a = known, "max_candidates"_a = k.max_candidates,
                                    "budget_s"_a = k.budget_s);
        }
        const py::object all = steal(PyNumber_Add(py::list(dets).ptr(), more.ptr()));
        passes = R.attr("receive_slot")(prep, spec, "estimator"_a = estimator, "erase_s"_a = erase, "dets"_a = all);
    }
    return py::make_tuple(passes, heard);
}

py::tuple finish_receive_py(const py::object& x, const py::object& ok, const py::object& origin, const py::object& q,
                            const py::object& spec, const py::object& estimator, const py::object& dets,
                            const py::object& hints, const py::object& noise, bool typed) {
    const FinishConsts k = finish_consts();
    if (typed) {
        try {
            return finish_typed(x, ok, origin, q, spec, estimator, dets, hints, noise, k);
        } catch (const Untyped&) {
        }
    }
    return finish_python(x, ok, origin, q, spec, estimator, dets, hints, k);
}

// --- the dial ------------------------------------------------------------------------------------------

struct PyDialLog {
    py::object path;                       // pathlib.Path(path), as the reference keeps it
    std::shared_ptr<lv::DialLog> log;
};

// live.DIAL_GUARD_S, read at each call as the reference reads it
lv::DialGuard guard_now() {
    const py::object g = live_attr("DIAL_GUARD_S");
    return {rx::as_double(g[py::int_(0)]), rx::as_double(g[py::int_(1)])};
}

py::tuple dial_span_py(const lv::DialSpan& s) {
    return py::make_tuple(py::float_(s.lo), py::float_(s.hi), opt_int_py(s.dial));
}

// --- tiles (parity windows) -------------------------------------------------------------------------

std::string str_field(const py::handle& t, const char* name) { return rx::as_string(t.attr(name)); }

std::optional<std::string> opt_str_field(const py::handle& t, const char* name) {
    const py::object v = t.attr(name);
    if (v.is_none()) return std::nullopt;
    return rx::as_string(v);
}

std::optional<double> opt_float_field(const py::handle& t, const char* name) {
    const py::object v = t.attr(name);
    if (v.is_none()) return std::nullopt;
    return rx::as_double(v);
}

lv::Tile tile_of(const py::handle& t) {
    lv::Tile x;
    x.id = str_field(t, "id");
    x.q = rx::as_int(t.attr("q"));
    x.slot_utc = str_field(t, "slot_utc");
    x.frame = str_field(t, "frame");
    x.f_hz = rx::as_double(t.attr("f_hz"));
    x.dial_hz = opt_int(t.attr("dial_hz"));
    x.status = str_field(t, "status");
    x.snr_db = opt_float_field(t, "snr_db");
    x.z_ref = opt_float_field(t, "z_ref");
    x.callsign = opt_str_field(t, "callsign");
    x.grid = opt_str_field(t, "grid");
    x.picture_id = opt_str_field(t, "picture_id");
    x.mode = opt_str_field(t, "mode");
    x.segment = opt_int(t.attr("segment"));
    x.cw_text = opt_str_field(t, "cw_text");
    x.progress = rx::as_double(t.attr("progress"));
    x.heard = rx::as_double(t.attr("heard"));
    x.received = rx::as_double(t.attr("received"));
    x.mean_w_db = opt_float_field(t, "mean_w_db");
    for (const py::object& c : objects(t.attr("confidence")))
        x.confidence.push_back(c.is_none() ? std::nullopt : std::optional<double>(rx::as_double(c)));
    x.passes = rx::as_int(t.attr("passes"));
    x.image = opt_str_field(t, "image");
    x.image_rev = rx::as_int(t.attr("image_rev"));
    x.updated = rx::as_double(t.attr("updated"));
    x.note = str_field(t, "note");
    x.uid = opt_str_field(t, "uid");
    x.waveform = str_field(t, "waveform");
    return x;
}

std::string json_text(const py::handle& o) { return rx::as_string(o); }

// --- the bindings ------------------------------------------------------------------------------------

void bind(py::module_& qrss) {
    rx::register_translators();
    py::module_ m = qrss.def_submodule("live", "native/core/qrss/live (live.py up to LiveListener)");
    m.attr("RING_S") = lv::RING_S;
    m.attr("PREAMBLE_S") = lv::PREAMBLE_S;
    m.attr("CAPTURE_MARGIN_S") = lv::CAPTURE_MARGIN_S;
    m.attr("FINISH_SEARCH_MIN_HEARD") = lv::FINISH_SEARCH_MIN_HEARD;
    m.attr("FINISH_MAX_CANDIDATES") = lv::FINISH_MAX_CANDIDATES;
    m.attr("FINISH_BUDGET_S") = lv::FINISH_BUDGET_S;
    m.attr("DIAL_FILE") = rx::to_str(lv::DIAL_FILE);
    m.attr("DIAL_GUARD_S") = py::make_tuple(lv::DIAL_GUARD_S[0], lv::DIAL_GUARD_S[1]);
    m.attr("L_BELOW_HZ") = lv::L_BELOW_HZ;

    // --- small things
    m.def("utc_iso", [](const py::object& t) -> py::str {
        // fromtimestamp's two paths: a float rounded to the microsecond, an
        // int (anything with __index__) taken whole
        try {
            if (PyFloat_Check(t.ptr())) return rx::to_str(lv::utc_iso(PyFloat_AS_DOUBLE(t.ptr())));
            const py::object i = steal(PyNumber_Index(t.ptr()));
            int overflow = 0;
            const long long k = PyLong_AsLongLongAndOverflow(i.ptr(), &overflow);
            if (overflow) throw std::overflow_error("timestamp out of range for platform time_t");
            if (k == -1 && PyErr_Occurred()) throw py::error_already_set();
            return rx::to_str(lv::utc_iso_int(k));
        } catch (const std::system_error& e) {
            raise_os_error(e);
        }
    }, py::arg("t"));
    m.def("_utc_iso_int", [](std::int64_t t) {
        try {
            return rx::to_str(lv::utc_iso_int(t));
        } catch (const std::system_error& e) {
            raise_os_error(e);
        }
    }, py::arg("t"));

    // frame_seconds: a native spec in C++, the reference's by its own
    // keyed_end_pos * t_sym; a tuple or list the max over it (max() keeps
    // the first of equals)
    struct FrameSeconds {
        static py::object of(const py::handle& spec) {
            if (is_seq(spec)) {
                py::object best;
                for (const py::handle s : py::reinterpret_borrow<py::iterable>(spec)) {
                    py::object v = of(s);
                    if (!best || compare(v, best, Py_GT)) best = v;
                }
                if (!best) throw py::value_error("max() iterable argument is empty");
                return best;
            }
            if (py::isinstance<fr::FrameSpec>(spec)) return py::float_(lv::frame_seconds(spec.cast<fr::FrameSpec>()));
            // `spec.keyed_end_pos * spec.t_sym`, its operands in Python's order
            // (C++ leaves the order of a call's arguments open)
            const py::object k = spec.attr("keyed_end_pos");
            const py::object t = spec.attr("t_sym");
            return steal(PyNumber_Multiply(k.ptr(), t.ptr()));
        }
    };
    m.def("frame_seconds", [](const py::object& spec) { return FrameSeconds::of(spec); }, py::arg("spec"));

    m.def("in_band", [](const py::object& f_hz, const py::object& waveform) -> py::object {
        const bool l = waveform.equal(py::int_(cfg::WAVEFORM_L));
        if (PyFloat_CheckExact(f_hz.ptr()) || PyLong_CheckExact(f_hz.ptr()))
            return py::bool_(lv::in_band(rx::as_double(f_hz), l ? cfg::WAVEFORM_L : cfg::WAVEFORM_CE));
        // another number type keeps its own arithmetic and its own booleans
        // (np.float32 - 22.07 in float32, np.bool_ out), as the reference's
        // `lo = f_hz - L_BELOW_HZ ...; band[0] <= lo and f_hz <= band[1]`
        const py::object band = live_attr("CARRIER_BAND_HZ");
        const py::object lo = l ? steal(PyNumber_Subtract(f_hz.ptr(), live_attr("L_BELOW_HZ").ptr())) : f_hz;
        py::object first = steal(PyObject_RichCompare(band[py::int_(0)].ptr(), lo.ptr(), Py_LE));
        if (!truth(first)) return first;
        return steal(PyObject_RichCompare(f_hz.ptr(), band[py::int_(1)].ptr(), Py_LE));
    }, py::arg("f_hz"), py::arg("waveform") = cfg::WAVEFORM_CE);

    m.def("centre_of", [](const py::object& f_hz, const py::object& waveform) -> py::object {
        // `waveform in (WAVEFORM_L, "L")`
        const bool l = py::int_(cfg::WAVEFORM_L).equal(waveform) || py::str("L").equal(waveform);
        if (!l) return f_hz;
        if (PyFloat_CheckExact(f_hz.ptr()) || PyLong_CheckExact(f_hz.ptr()))
            return py::float_(lv::centre_of(rx::as_double(f_hz), cfg::WAVEFORM_L));
        // another number type keeps its own arithmetic (np.float32 + 9.5)
        const py::object off = py::module_::import("sstvae.qrss.receiver").attr("L_EXCLUDE_OFFSET_HZ");
        return steal(PyNumber_Add(f_hz.ptr(), off.ptr()));
    }, py::arg("f_hz"), py::arg("waveform"));

    m.def("_waveform_id", [](const py::object& w) -> py::object {
        if (py::isinstance<py::str>(w)) return py::int_(lv::waveform_id(rx::as_string(w)));
        return steal(PyNumber_Long(w.ptr()));
    }, py::arg("w"));

    m.def("_due_waveform", [](const py::object& d) -> py::object {
        // int(lframe.get_any(d["frame"]).waveform), else _waveform_id(d.get("waveform", CE))
        auto fallback = [&]() -> py::object {
            const py::object w = d.attr("get")("waveform", py::int_(cfg::WAVEFORM_CE));
            if (py::isinstance<py::str>(w)) return py::int_(lv::waveform_id(rx::as_string(w)));
            return steal(PyNumber_Long(w.ptr()));
        };
        if (py::isinstance<py::dict>(d)) {
            const py::dict dd = py::reinterpret_borrow<py::dict>(d);
            if (!dd.contains("frame")) return fallback();
            const py::object f = dd["frame"];
            if (py::isinstance<py::str>(f)) {
                if (const std::optional<int> w = lv::frame_waveform(rx::as_string(f))) return py::int_(*w);
                return fallback();
            }
        }
        try {
            const py::object f = d[py::str("frame")];
            const py::object spec = py::module_::import("sstvae.qrss.lframe").attr("get_any")(f);
            return steal(PyNumber_Long(spec.attr("waveform").ptr()));
        } catch (py::error_already_set& e) {
            if (e.matches(PyExc_KeyError) || e.matches(PyExc_ValueError) || e.matches(PyExc_TypeError) ||
                e.matches(PyExc_AttributeError))
                return fallback();
            throw;
        }
    }, py::arg("d"));

    // --- the ring
    py::class_<lv::FeRing, std::shared_ptr<lv::FeRing>>(m, "FeRing",
        "The last span_s of FE in memory, addressed by absolute FE index (t * 4000).")
        .def(py::init([](const py::object& span_s) {
            const double s = rx::as_double(span_s);
            py::gil_scoped_release nogil;
            return std::make_shared<lv::FeRing>(s);
        }), py::arg("span_s") = lv::RING_S)
        .def_property_readonly("n", &lv::FeRing::n)
        .def_property_readonly("end", [](const lv::FeRing& r) { return opt_int_py(r.end()); })
        .def("write", [](lv::FeRing& r, const py::object& n0, const py::object& x) {
            const CArr a = samples_of(x);
            const std::int64_t k = index_of(n0);
            const std::span<const cf> s = cf_span(a);
            py::gil_scoped_release nogil;
            r.write(k, s);
        }, py::arg("n0"), py::arg("x"))
        .def("_clear", [](lv::FeRing& r, const py::object& a, const py::object& b) {
            const std::int64_t ia = index_of(a), ib = index_of(b);
            py::gil_scoped_release nogil;
            r.clear(ia, ib);
        }, py::arg("a"), py::arg("b"))
        .def("read", [](const lv::FeRing& r, const py::object& a, const py::object& b) {
            return ring_read(r, index_of(a), index_of(b));
        }, py::arg("a"), py::arg("b"))
        .def_property_readonly("x", [](const lv::FeRing& r) {
            std::vector<cf> v;
            {
                py::gil_scoped_release nogil;
                v = r.samples();
            }
            return to_numpy(v);
        })
        .def_property_readonly("ok", [](const lv::FeRing& r) {
            std::vector<std::uint8_t> v;
            {
                py::gil_scoped_release nogil;
                v = r.written();
            }
            return bool_array(v);
        });

    m.def("_spans", [](const py::object& mask, const py::object& fs, const py::object& origin) {
        const py::array_t<std::uint8_t> b = mask_of(mask);
        return spans_py(lv::spans(bytes_span(b), rx::as_double(fs), rx::as_double(origin)));
    }, py::arg("mask"), py::arg("fs"), py::arg("origin"));

    // --- a slot's capture
    m.def("capture_read", &capture_read_py, py::arg("ring"), py::arg("q"), py::arg("spec"), py::arg("lead_s"),
          py::arg("tail_s"), py::arg("typed"));
    m.def("capture_prepare", &capture_prepare_py, py::arg("x"), py::arg("ok"), py::arg("origin"), py::arg("q"),
          py::arg("spec"), py::arg("seed"), py::arg("noise"), py::arg("typed"));
    m.def("slot_capture", &slot_capture_py, py::arg("ring"), py::arg("q"), py::arg("spec"), py::arg("seed"),
          py::arg("lead_s"), py::arg("tail_s"), py::arg("noise"), py::arg("typed"));
    m.def("finish_receive", &finish_receive_py, py::arg("x"), py::arg("ok"), py::arg("origin"), py::arg("q"),
          py::arg("spec"), py::arg("estimator"), py::arg("dets"), py::arg("hints"), py::arg("noise"),
          py::arg("typed"));

    // --- the dial
    py::class_<PyDialLog>(m, "DialLog", "The radio's dial frequency over time, from DIAL_FILE.")
        .def(py::init([](const py::object& path) {
            const py::object p = py::module_::import("pathlib").attr("Path")(path);
            // os.fsencode, which (unlike open) lets a NUL through: refresh
            // refuses it there, as the reference's stat does
            const py::bytes b(py::module_::import("os").attr("fsencode")(p));
            return PyDialLog{p, std::make_shared<lv::DialLog>(std::string(b))};
        }), py::arg("path"))
        .def_property_readonly("path", [](const PyDialLog& d) { return d.path; })
        .def("refresh", [](PyDialLog& d) {
            py::gil_scoped_release nogil;
            d.log->refresh();
        })
        .def_property("points", [](const PyDialLog& d) {
            py::list out;
            for (const lv::DialPoint& p : d.log->points()) out.append(py::make_tuple(py::float_(p.t), py::int_(p.dial_hz)));
            return out;
        }, [](PyDialLog& d, const py::object& v) {
            std::vector<lv::DialPoint> pts;
            for (const py::object& o : objects(v)) {
                const std::vector<py::object> th = unpack(o, 2);
                pts.push_back({real(th[0]), rx::as_int(th[1])});
            }
            d.log->set_points(std::move(pts));
        })
        .def("spans", [](const PyDialLog& d, const py::object& a, const py::object& b) {
            const lv::DialGuard g = guard_now();
            py::list out;
            for (const lv::DialSpan& s : d.log->spans(real(a), real(b), g)) out.append(dial_span_py(s));
            return out;
        }, py::arg("a"), py::arg("b"))
        .def("by_dial", [](const PyDialLog& d, const py::object& a, const py::object& b) {
            const lv::DialGuard g = guard_now();
            py::list out;
            for (const auto& [dial, spans] : d.log->by_dial(real(a), real(b), g)) {
                py::list ss;
                for (const auto& [lo, hi] : spans) ss.append(py::make_tuple(py::float_(lo), py::float_(hi)));
                out.append(py::make_tuple(opt_int_py(dial), ss));
            }
            return out;
        }, py::arg("a"), py::arg("b"))
        .def("dial_at", [](const PyDialLog& d, const py::object& t) {
            return opt_int_py(d.log->dial_at(real(t), guard_now()));
        }, py::arg("t"))
        .def("span_at", [](const PyDialLog& d, const py::object& t) {
            return dial_span_py(d.log->span_at(real(t), guard_now()));
        }, py::arg("t"));

    m.def("same_dial", [](const py::object& a, const py::object& b) -> py::object {
        if (a.is_none() || b.is_none()) return py::bool_(true);
        if (PyLong_CheckExact(a.ptr()) && PyLong_CheckExact(b.ptr())) {
            int oa = 0, ob = 0;
            const long long x = PyLong_AsLongLongAndOverflow(a.ptr(), &oa);
            const long long y = PyLong_AsLongLongAndOverflow(b.ptr(), &ob);
            if (!oa && !ob && !PyErr_Occurred()) return py::bool_(lv::same_dial(x, y));
            PyErr_Clear();
        }
        return steal(PyObject_RichCompare(a.ptr(), b.ptr(), Py_EQ));
    }, py::arg("a"), py::arg("b"));

    m.def("restrict", [](const py::object& ok, const py::object& origin, const py::object& q,
                         const py::object& spans) -> py::object {
        // `any(lo == -inf and hi == inf for lo, hi in spans)`, each span
        // unpacked in order up to the first unbounded one
        std::vector<std::pair<double, double>> sp;
        for (const py::object& o : objects(spans)) {
            const std::vector<py::object> lh = unpack(o, 2);
            const double lo = real(lh[0]), hi = real(lh[1]);
            if (lo == -std::numeric_limits<double>::infinity() && hi == std::numeric_limits<double>::infinity())
                return ok;
            sp.emplace_back(lo, hi);
        }
        const std::int64_t org = index_of(origin), qi = rx::as_int(q);
        if (const auto b = bool_bytes(ok)) {
            const std::span<const std::uint8_t> oks = bytes_span(*b);
            std::vector<std::uint8_t> out;
            {
                py::gil_scoped_release nogil;
                out = lv::restrict(oks, org, qi, sp);
            }
            return bool_array(out);
        }
        // another dtype: the ranges the C++'s, the copies numpy's
        const auto n = static_cast<std::int64_t>(py::len(ok));
        const auto ranges = lv::restrict_ranges(n, org, qi, sp);
        if (!ranges) return ok;
        py::object out = rx::np_module().attr("zeros_like")(ok);
        for (const auto& [a, b] : *ranges) {
            const py::slice s(a, b, 1);
            out[s] = ok[s];
        }
        return out;
    }, py::arg("ok"), py::arg("origin"), py::arg("q"), py::arg("spans"));

    // --- parity windows
    m.def("_tile_json", [](const py::object& t) { return rx::to_str(pyjson::dumps(tile_of(t).to_json())); },
          py::arg("tile"));
    m.def("_tile_roundtrip", [](const py::object& text) -> py::object {
        const std::optional<lv::Tile> t = lv::Tile::from_json(pyjson::loads(json_text(text)));
        if (!t) return py::none();
        return rx::to_str(pyjson::dumps(t->to_json()));
    }, py::arg("text"));
    m.def("_live_config", [](const py::object& spec, const py::object& waveforms) {
        lv::LiveConfig c;
        c.spec = rx::spec(spec);
        if (!waveforms.is_none()) {
            std::vector<std::int64_t> ws;
            for (const py::object& w : objects(waveforms)) ws.push_back(rx::as_int(w));
            c.waveforms = std::move(ws);
        }
        py::list specs, sel;
        for (const fr::FrameSpec& s : c.specs()) specs.append(rx::to_str(s.name()));
        for (const fr::FrameSpec& s : c.selection()) sel.append(rx::to_str(s.name()));
        return py::make_tuple(specs, sel, c.joint());
    }, py::arg("spec"), py::arg("waveforms"));
    m.def("_live_config_defaults", []() {
        const lv::LiveConfig c;
        py::dict d;
        d["spec"] = rx::to_str(c.spec.name());
        d["waveforms"] = py::none();
        d["refresh_s"] = c.refresh_s;
        d["first_s"] = c.first_s;
        d["detect_until_s"] = c.detect_until_s;
        d["recheck_until_s"] = c.recheck_until_s;
        d["empty_refresh_s"] = c.empty_refresh_s;
        d["keep_done_s"] = c.keep_done_s;
        d["estimator"] = rx::to_str(c.estimator);
        d["precision"] = rx::to_str(c.precision);
        d["model"] = py::none();
        d["render"] = c.render;
        d["background"] = c.background;
        d["finish_process"] = c.finish_process;
        return d;
    });
    m.def("_default_noise", [](std::int64_t seed, std::size_t n) {
        std::vector<cf> v;
        {
            py::gil_scoped_release nogil;
            v = lv::default_noise(seed, n);
        }
        return to_numpy(v);
    }, py::arg("seed"), py::arg("n"));
    m.def("_parse_dial_text", [](const py::bytes& b) {
        const std::string s = b;
        py::list out;
        for (const lv::DialPoint& p : lv::parse_dial_text(s)) out.append(py::make_tuple(py::float_(p.t), py::int_(p.dial_hz)));
        return out;
    }, py::arg("data"));
    m.def("_finish_gate", [](const py::object& ok, std::int64_t origin, double frame_s, bool have_dets,
                             bool have_hints, double min_heard) {
        const py::array_t<std::uint8_t> b = mask_of(ok);
        const lv::FinishGate g = lv::finish_gate(bytes_span(b), origin, frame_s, have_dets, have_hints, min_heard);
        return py::make_tuple(g.receive, g.k1, g.heard);
    }, py::arg("ok"), py::arg("origin"), py::arg("frame_s"), py::arg("have_dets"), py::arg("have_hints"),
       py::arg("min_heard"));
    m.def("_finish_branch", [](double heard, bool have_dets, bool have_hints, double min_heard) {
        switch (lv::finish_branch(heard, have_dets, have_hints, min_heard)) {
            case lv::FinishBranch::Search: return "search";
            case lv::FinishBranch::Known: return "known";
            case lv::FinishBranch::Nothing: break;
        }
        return "nothing";
    }, py::arg("heard"), py::arg("have_dets"), py::arg("have_hints"), py::arg("min_heard"));
    m.def("_due_waveform_json", [](const py::object& text) {
        return lv::due_waveform(pyjson::loads(json_text(text)));
    }, py::arg("text"));
    m.def("_waveform_id_json", [](const py::object& text) {
        return lv::waveform_id(pyjson::loads(json_text(text)));
    }, py::arg("text"));
}

const Registrar reg("live", 92, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
