// sstvae_native.qrss.listener: native/core/qrss/listener (live.py's
// LiveListener and run()), for tests/test_qrss_native_listener.py to run
// beside the reference. No row replaces live.LiveListener: the two are
// compared side by side (tests/qrss_native/_listener.py builds this one
// the way a test builds the reference's).
//
//   LiveListener(state_dir, cfg=None, store=None, passband=None, log=None, *,
//                clock=None, candidates=None, detect=None, receive_slot=None,
//                finish=None, noise=None, associate=None, renderer=None,
//                old_edge_noise=None, old_gating=None, low_priority=True, dial_guard=None)
//     cfg        a live.LiveConfig (the reference's dataclass); None: LiveConfig()
//     store      None, or an sstvae_native.qrss.store.Store (the C++ store)
//     passband   None, a directory, or a passband store (its .root and
//                options: the listener opens its own C++ store there)
//     log        log sink callable(line); None: stderr
//     clock      callable() -> unix seconds; None: the system clock
//     candidates, detect, receive_slot
//                the live receive, called exactly as _refresh calls
//                receiver.candidates / .detect / .receive_slot (prep is the
//                C++ capture as the reference's receiver.Prepared; spec is
//                cfg.selection);
//                None: the C++ receiver
//     finish     callable(x, ok, origin, q, spec, estimator, dets, hints) ->
//                (passes, heard): live.finish_receive's signature; None:
//                the C++ finish_receive
//     noise      callable(seed, n) -> n complex samples; None: the C++ generator
//     associate  callable(p, store) -> (key, rule); None: the C++ associate
//     renderer   None (no pictures), a native Renderer (listener_codec's
//                ListenerRenderer), or an object with load(codec_id, warn),
//                check(codec_id) and render(S, W, mode, codec_id)
//     old_edge_noise, old_gating  the receiver's rules; None: rules.old(...)
//     dial_guard (before, after) seconds taken off around a dial change;
//                None: DIAL_GUARD_S
//   .feed(x, t_end) .step(now) .write_state(listening=True) .log(msg)
//   .backfill(now) .recover(now, hours=RECOVER_HOURS) .tiles()
//   .active_slots(now) ._progress(q) ._interval(q, now) ._hints(q, dial)
//   ._match(q, f_hz, dial=None, waveform="CE") .idle() ._finish_idle()
//   .wait_idle(timeout=None) .replay_end() .close()
//   .lines .now .audio_end .busy .codec_error .source .finished .done_tiles
//   .slots (q -> live.SlotState) .recovering .dur .joint .specs .selection
//   .cfg .dir .next_index (fe.next_index) .ring (the C++ FeRing) .dial
//   run(listener, source, realtime, stop=None): live.run (source: .name and
//     .chunks() yielding (x, t_end), x None for no audio; stop: an Event)
//
// Records the seams return are read the way the reference reads them
// (attributes, with the reference's defaults where it uses getattr), and
// handed back to Python as the objects the seams returned: a detection or
// pass the C++ kept is found again by its contents, and a pass the
// listener gave a known header (_known_header) is a copy.copy of the
// original with that header, as the reference makes one. Anything not
// found is the rx_types dict.
//
// THREADS and the GIL: every method releases the GIL before the listener
// takes its lock, and every seam takes the GIL back to call Python, so a
// thread holding the listener's lock (waiting for the GIL in the clock
// seam) and a thread calling the listener (the GIL released first) cannot
// deadlock. A Python seam's exception reaches the listener as a
// live::SeamError (its type's name and str()), which the listener logs as
// the reference does. Deleting the object closes the listener with the GIL
// released (the workers may be in a seam), then frees the seams with it held.

#include "qrss/rx_types.hpp"
#include "qrss/fspath.hpp"
#include "qrss/store_types.hpp"

#include <atomic>
#include <complex>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/associate.hpp"
#include "qrss/config.hpp"
#include "qrss/listener.hpp"
#include "qrss/live.hpp"
#include "qrss/passband.hpp"
#include "qrss/receiver.hpp"
#include "qrss/receiver_slot.hpp"
#include "qrss/source.hpp"
#include "qrss/store.hpp"
#include "qrss/types.hpp"

namespace sstvae_bindings::qrss {
namespace {

namespace lv = sstvae::qrss::live;
namespace rcv = sstvae::qrss::receiver;
namespace aq = sstvae::qrss::acquire;
namespace types = sstvae::qrss::types;
namespace hdr = sstvae::qrss::header;
namespace store = sstvae::qrss::store;
namespace pb = sstvae::qrss::passband;
namespace cfg = sstvae::qrss::config;
using cf = std::complex<float>;
using namespace pybind11::literals;

py::object live_mod() { return py::module_::import("sstvae.qrss.live"); }

py::object opt_int_py(const std::optional<std::int64_t>& v) { return v ? py::object(py::int_(*v)) : py::object(py::none()); }
py::object opt_float_py(const std::optional<double>& v) { return v ? py::object(py::float_(*v)) : py::object(py::none()); }
py::object opt_str_py(const std::optional<std::string>& v) { return v ? py::object(rx::to_str(*v)) : py::object(py::none()); }

bool has(const py::handle& o, const char* name) { return rx::has_field(o, name); }

py::list int_list(const std::vector<std::int64_t>& v) {
    py::list out;
    for (const std::int64_t x : v) out.append(py::int_(x));
    return out;
}

// getattr(o, name, None) is not None
bool present(const py::handle& o, const char* name) { return has(o, name) && !rx::field(o, name).is_none(); }

// --- Python's exceptions in a seam -----------------------------------------------------------

// f() with the GIL held; a Python exception as a SeamError (its type's name, str()).
template <typename F>
auto with_gil(F&& f) -> decltype(f()) {
    py::gil_scoped_acquire gil;
    try {
        return f();
    } catch (py::error_already_set& e) {
        std::string name = "Exception";
        std::string text;
        try {
            name = py::str(e.type().attr("__name__")).cast<std::string>();
            text = py::str(e.value()).cast<std::string>();
        } catch (...) {
        }
        throw lv::SeamError(name, text);
    }
}

// --- records from Python, read as the reference reads them ------------------------------------

std::string bits(double v) {
    std::string s(sizeof v, '\0');
    std::memcpy(s.data(), &v, sizeof v);
    return s;
}

template <typename T>
std::string hash_of(const std::vector<T>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* b = reinterpret_cast<const unsigned char*>(v.data());
    for (std::size_t k = 0; k < v.size() * sizeof(T); ++k) h = (h ^ b[k]) * 1099511628211ULL;
    return std::to_string(v.size()) + ":" + std::to_string(h);
}

std::string fp(const types::Detection& d) {
    return bits(d.f_hz) + bits(d.z_ref) + bits(d.lead_in_s) + bits(d.a3_z) + std::to_string(d.waveform) + "|" + d.method;
}

std::string fp(const types::PassResult& p) {
    return p.uid + "|" + std::to_string(p.q) + "|" + p.frame + "|" + std::to_string(p.waveform) + bits(p.f_hz) +
           bits(p.report.snr2500_db) + bits(p.report.z_ref) + hash_of(p.z) + hash_of(p.w);
}

std::string fp(const hdr::HeaderFields& h) {
    return h.callsign + "|" + (h.grid ? "+" + *h.grid : std::string("-")) + "|" + std::to_string(h.picture_id) + "|" +
           std::to_string(h.mode) + "|" + std::to_string(h.segment) + "|" + std::to_string(h.codec_id) + "|" +
           std::to_string(h.version) + "|" + std::to_string(h.reserved);
}

hdr::HeaderFields header_of(const py::handle& o) {
    hdr::HeaderFields h;
    h.callsign = rx::as_string(rx::field(o, "callsign"));
    if (has(o, "grid")) h.grid = rx::opt_string(rx::field(o, "grid"));
    h.picture_id = rx::as_int(rx::field(o, "picture_id"));
    h.mode = rx::as_int(rx::field(o, "mode"));
    h.segment = rx::as_int(rx::field(o, "segment"));
    if (has(o, "codec_id")) h.codec_id = rx::as_int(rx::field(o, "codec_id"));
    if (present(o, "version")) h.version = rx::as_int(rx::field(o, "version"));
    if (present(o, "reserved")) h.reserved = rx::as_int(rx::field(o, "reserved"));
    return h;
}

types::TrackReport report_of(const py::handle& o) {
    types::TrackReport r;
    auto num = [&](const char* name, double& out) {
        if (present(o, name)) out = rx::as_double(rx::field(o, name));
    };
    num("offset_hz", r.offset_hz);
    num("drift_hz_per_min", r.drift_hz_per_min);
    num("wander_hz_rms", r.wander_hz_rms);
    num("doppler_hz", r.doppler_hz);
    num("ppm", r.ppm);
    num("snr2500_db", r.snr2500_db);
    num("z_ref", r.z_ref);
    num("kappa", r.kappa);
    if (present(o, "slip_free")) r.slip_free = rx::as_bool(rx::field(o, "slip_free"));
    if (present(o, "suspect")) r.suspect = rx::as_bool(rx::field(o, "suspect"));
    return r;
}

types::CwIdResult cw_of(const py::handle& o) {
    types::CwIdResult c;
    if (has(o, "text")) c.text = rx::opt_string(rx::field(o, "text"));
    if (present(o, "z_match")) c.z_match = rx::as_double(rx::field(o, "z_match"));
    if (present(o, "keying")) c.keying = rx::as_string(rx::field(o, "keying"));
    if (present(o, "agrees")) c.agrees = rx::as_bool(rx::field(o, "agrees"));
    return c;
}

struct Shared;

types::Detection det_of(Shared& sh, const py::handle& o);
types::PassResult pass_of(Shared& sh, const py::handle& o);

// --- what the listener shares with its seams ----------------------------------------------------

// The Python side of one listener: what the seams hand Python, and the
// objects behind the records they returned. Touched only with the GIL held.
struct Shared {
    py::object cfg;         // the reference LiveConfig
    py::object selection;   // cfg.selection: the receiver's `spec`
    py::object store;       // the store as given (associate's argument)
    bool joint = false;
    std::map<std::string, py::object> dets, passes, headers;

    template <typename T>
    void remember(std::map<std::string, py::object>& m, const T& x, const py::handle& o) {
        if (m.size() > 20000) m.clear();   // (a long run: the oldest are long gone)
        m[fp(x)] = py::reinterpret_borrow<py::object>(o);
    }

    py::object det_py(const types::Detection& d) const {
        const auto it = dets.find(fp(d));
        return it != dets.end() ? it->second : py::object(rx::to_py(d));
    }

    py::object header_py(const hdr::HeaderFields& h) const {
        const auto it = headers.find(fp(h));
        return it != headers.end() ? it->second : py::object(rx::to_py(h));
    }

    // The pass as the reference holds it: the object the seam returned, or
    // its copy with the header the listener gave it (_known_header).
    py::object pass_py(const types::PassResult& p) const {
        types::PassResult bare = p;
        bare.header.reset();
        auto it = passes.find(fp(p));
        if (it == passes.end()) it = passes.find(fp(bare));
        if (it == passes.end()) return rx::to_py(p);
        const py::object orig = it->second;
        const py::object h = py::getattr(orig, "header", py::none());
        if (p.header && h.is_none()) {
            py::object c = py::module_::import("copy").attr("copy")(orig);
            c.attr("header") = header_py(*p.header);
            return c;
        }
        return orig;
    }

    py::list dets_py(std::span<const types::Detection> ds) const {
        py::list out;
        for (const auto& d : ds) out.append(det_py(d));
        return out;
    }

    // a hint as _hints gives it: (f, waveform) on the joint path, else f
    static py::object hint_py(const aq::Hint& h) {
        if (h.waveform) return py::make_tuple(py::float_(h.f_hz), py::int_(*h.waveform));
        return py::float_(h.f_hz);
    }

    static py::list hints_py(std::span<const aq::Hint> hs) {
        py::list out;
        for (const auto& h : hs) out.append(hint_py(h));
        return out;
    }
};

types::Detection det_of(Shared& sh, const py::handle& o) {
    types::Detection d;
    d.f_hz = rx::as_double(rx::field(o, "f_hz"));
    if (present(o, "path")) d.path = rx::freq_path(rx::field(o, "path"));
    if (present(o, "timing")) d.timing = rx::timing(rx::field(o, "timing"));
    if (present(o, "z_ref")) d.z_ref = rx::as_double(rx::field(o, "z_ref"));
    if (present(o, "method")) d.method = rx::as_string(rx::field(o, "method"));   // (None: not "hint")
    if (present(o, "lead_in_s")) d.lead_in_s = rx::as_double(rx::field(o, "lead_in_s"));
    if (present(o, "waveform")) d.waveform = rx::as_int(rx::field(o, "waveform"));
    if (present(o, "a3_z")) d.a3_z = rx::as_double(rx::field(o, "a3_z"));
    sh.remember(sh.dets, d, o);
    return d;
}

types::PassResult pass_of(Shared& sh, const py::handle& o) {
    types::PassResult p;
    p.f_hz = rx::as_double(rx::field(o, "f_hz"));
    if (present(o, "uid")) p.uid = rx::as_string(rx::field(o, "uid"));
    if (present(o, "q")) p.q = rx::as_int(rx::field(o, "q"));
    if (present(o, "frame")) p.frame = rx::as_string(rx::field(o, "frame"));
    if (present(o, "waveform")) p.waveform = rx::as_int(rx::field(o, "waveform"));
    if (present(o, "timing")) p.timing = rx::timing(rx::field(o, "timing"));
    if (present(o, "report")) p.report = report_of(rx::field(o, "report"));
    if (present(o, "z")) p.z = rx::vec<float>(rx::field(o, "z"));
    if (present(o, "w")) p.w = rx::vec<float>(rx::field(o, "w"));
    if (present(o, "hdr_llr")) p.hdr_llr = rx::vec<float>(rx::field(o, "hdr_llr"));
    if (present(o, "header")) {
        const py::object h = rx::field(o, "header");
        p.header = header_of(h);
        sh.remember(sh.headers, *p.header, h);
    }
    if (present(o, "cw")) p.cw = cw_of(rx::field(o, "cw"));
    if (present(o, "ch")) p.ch = rx::vec<cf>(rx::field(o, "ch"));
    if (present(o, "ch_fs")) p.ch_fs = rx::as_int(rx::field(o, "ch_fs"));
    if (present(o, "ch_t0_index")) p.ch_t0_index = rx::as_double(rx::field(o, "ch_t0_index"));
    if (present(o, "f_mix_hz")) p.f_mix_hz = rx::rational(rx::field(o, "f_mix_hz"));
    if (present(o, "psi")) p.psi = rx::vec<float>(rx::field(o, "psi"));
    if (present(o, "estimator")) p.estimator = rx::as_string(rx::field(o, "estimator"));
    if (present(o, "em_round")) p.em_round = rx::as_int(rx::field(o, "em_round"));
    sh.remember(sh.passes, p, o);
    return p;
}

std::vector<types::Detection> dets_of(Shared& sh, const py::handle& o) {
    std::vector<types::Detection> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(o))) out.push_back(det_of(sh, h));
    return out;
}

std::vector<types::PassResult> passes_of(Shared& sh, const py::handle& o) {
    std::vector<types::PassResult> out;
    for (const py::handle h : py::reinterpret_borrow<py::iterable>(py::iter(o))) out.push_back(pass_of(sh, h));
    return out;
}

// the capture as the reference's receiver.Prepared (what _refresh hands the receiver)
py::object prep_py(const rcv::Prepared& p) {
    const py::dict d = rx::to_py(p);
    return py::module_::import("sstvae.qrss.receiver").attr("Prepared")(
        "fe"_a = d["fe"], "keep"_a = d["keep"], "q"_a = d["q"], "t0_index"_a = d["t0_index"], "fs"_a = d["fs"]);
}

// erase spans as _spans gives them: (start, end) of np.float64
py::list erase_py(const lv::Erase& e) {
    const py::object f64 = rx::np_module().attr("float64");
    py::list out;
    for (const auto& [a, b] : e) out.append(py::make_tuple(f64(a), f64(b)));
    return out;
}

// --- the seams -------------------------------------------------------------------------------------

struct PyRenderer : lv::Renderer {
    explicit PyRenderer(py::object o) : obj(std::move(o)) {}
    ~PyRenderer() override {
        py::gil_scoped_acquire gil;
        obj = py::object();
    }
    std::optional<std::string> load(std::optional<std::int64_t> codec_id, const Warn& warn) override {
        return with_gil([&]() -> std::optional<std::string> {
            const py::cpp_function w([warn](const py::object& m) {
                const std::string msg = py::str(m).cast<std::string>();
                py::gil_scoped_release nogil;   // (warn is the listener's log: its lock)
                if (warn) warn(msg);
            });
            const py::object r = obj.attr("load")(opt_int_py(codec_id), w);
            if (r.is_none()) return std::nullopt;
            return py::str(r).cast<std::string>();
        });
    }
    std::optional<std::string> check(std::optional<std::int64_t> codec_id) override {
        return with_gil([&]() -> std::optional<std::string> {
            const py::object r = obj.attr("check")(opt_int_py(codec_id));
            if (r.is_none()) return std::nullopt;
            return py::str(r).cast<std::string>();
        });
    }
    sstvae::images::Picture render(std::span<const double> S, std::span<const double> W, std::int64_t mode,
                                   std::optional<std::int64_t> codec_id) override {
        return with_gil([&]() {
            auto plane = [](std::span<const double> v) {
                py::array_t<double> a(std::vector<py::ssize_t>{3, static_cast<py::ssize_t>(v.size() / 3)});
                std::copy(v.begin(), v.end(), a.mutable_data());
                return a;
            };
            const py::object img = obj.attr("render")(plane(S), plane(W), py::int_(mode), opt_int_py(codec_id));
            const py::array a = rx::np_module().attr("ascontiguousarray")(rx::np_module().attr("asarray")(img),
                                                                          "dtype"_a = "uint8");
            if (a.ndim() != 3 || a.shape(2) != 3) throw std::invalid_argument("a picture is (height, width, 3) RGB");
            sstvae::images::Picture p(static_cast<int>(a.shape(1)), static_cast<int>(a.shape(0)));
            std::memcpy(p.rgb.data(), a.data(), p.rgb.size());
            return p;
        });
    }
    py::object obj;
};

// A Python callable, freed with the GIL held whenever the seam goes.
struct PyFn {
    explicit PyFn(py::object o) : f(new py::object(std::move(o)), [](py::object* p) {
        py::gil_scoped_acquire gil;
        delete p;
    }) {}
    std::shared_ptr<py::object> f;
    const py::object& operator*() const { return *f; }
};

lv::Seams seams_of(const std::shared_ptr<Shared>& sh, const py::object& clock, const py::object& log,
                   const py::object& candidates, const py::object& detect, const py::object& receive_slot,
                   const py::object& finish, const py::object& noise, const py::object& associate,
                   const py::object& renderer, bool old_edge_noise, bool old_gating, bool low_priority) {
    lv::Seams s;
    s.old_edge_noise = old_edge_noise;
    s.old_gating = old_gating;
    s.low_priority = low_priority;
    if (!clock.is_none()) {
        const PyFn f(clock);
        s.clock = [f]() { return with_gil([&] { return rx::as_double((*f)()); }); };
    }
    if (!log.is_none()) {
        const PyFn f(log);
        s.log = [f](const std::string& line) { with_gil([&] { (*f)(rx::to_str(line)); }); };
    }
    if (!candidates.is_none()) {
        const PyFn f(candidates);
        s.candidates = [f, sh](const rcv::Prepared& prep, lv::Selection, std::span<const aq::Hint> hints) {
            return with_gil([&] {
                const py::tuple hs(Shared::hints_py(hints));
                return dets_of(*sh, (*f)(prep_py(prep), sh->selection, "live_only"_a = true, "hints"_a = hs));
            });
        };
    }
    if (!detect.is_none()) {
        const PyFn f(detect);
        s.detect = [f, sh](const rcv::Prepared& prep, lv::Selection, const rcv::DetectOptions& o) {
            return with_gil([&] {
                py::list cands;
                if (o.cands) cands = sh->dets_py(*o.cands);
                py::object r;
                if (o.skip_hz.empty()) {
                    r = (*f)(prep_py(prep), sh->selection, "live_only"_a = o.live_only, "erase_s"_a = erase_py(o.erase_s),
                             "cands"_a = cands);
                } else {
                    py::list skip;
                    for (const double v : o.skip_hz) skip.append(py::float_(v));
                    r = (*f)(prep_py(prep), sh->selection, "live_only"_a = o.live_only, "erase_s"_a = erase_py(o.erase_s),
                             "cands"_a = cands, "skip_hz"_a = skip);
                }
                return dets_of(*sh, r);
            });
        };
    }
    if (!receive_slot.is_none()) {
        const PyFn f(receive_slot);
        s.receive_slot = [f, sh](const rcv::Prepared& prep, lv::Selection, const rcv::SlotOptions& o) {
            return with_gil([&] {
                const py::list dets = o.dets ? sh->dets_py(*o.dets) : py::list();
                return passes_of(*sh, (*f)(prep_py(prep), sh->selection, "estimator"_a = rx::to_str(o.estimator),
                                           "erase_s"_a = erase_py(o.erase_s), "round_b"_a = o.round_b,
                                           "dets"_a = dets));
            });
        };
    }
    if (!finish.is_none()) {
        const PyFn f(finish);
        s.finish = [f, sh](std::span<const cf> x, std::span<const std::uint8_t> ok, std::int64_t origin, std::int64_t q,
                           const sstvae::qrss::api::ReceiveConfig& config, std::span<const types::Detection> dets,
                           std::span<const aq::Hint> hints) {
            return with_gil([&] {
                py::array_t<cf> xa(static_cast<py::ssize_t>(x.size()));
                std::copy(x.begin(), x.end(), xa.mutable_data());
                py::array_t<bool> oka(static_cast<py::ssize_t>(ok.size()));
                auto* d = reinterpret_cast<std::uint8_t*>(oka.mutable_data());
                std::copy(ok.begin(), ok.end(), d);
                const py::object r = (*f)(xa, oka, py::int_(origin), py::int_(q), sh->selection,
                                          rx::to_str(config.estimator), sh->dets_py(dets), Shared::hints_py(hints));
                const py::tuple ph = py::tuple(r);
                if (ph.size() != 2) throw py::value_error("finish returns (passes, heard)");
                lv::FinishResult out;
                out.passes = passes_of(*sh, ph[0]);
                out.heard = rx::as_double(ph[1]);
                return out;
            });
        };
    }
    if (!noise.is_none()) {
        const PyFn f(noise);
        s.noise = [f](std::int64_t seed, std::size_t n) {
            return with_gil([&] {
                std::vector<cf> v = rx::vec<cf>((*f)(py::int_(seed), py::int_(n)));
                if (v.size() != n) throw std::invalid_argument("the noise gave " + std::to_string(v.size()) +
                                                               " samples for " + std::to_string(n));
                return v;
            });
        };
    }
    if (!associate.is_none()) {
        const PyFn f(associate);
        s.associate = [f, sh](const types::PassResult& p, store::Store&) {
            return with_gil([&] {
                const py::object r = (*f)(sh->pass_py(p), sh->store);
                const py::tuple kr = py::tuple(r);
                if (kr.size() != 2) throw py::value_error("associate returns (key, rule)");
                sstvae::qrss::associate::Result out;
                if (!kr[0].is_none()) out.key = st::key_index(kr[0]);
                if (!kr[1].is_none()) out.rule = py::str(kr[1]).cast<std::string>();
                return out;
            });
        };
    }
    if (!renderer.is_none()) {
        if (py::isinstance<lv::Renderer>(renderer))
            s.renderer = renderer.cast<std::shared_ptr<lv::Renderer>>();
        else
            s.renderer = std::make_shared<PyRenderer>(renderer);
    }
    return s;
}

// --- the configuration -------------------------------------------------------------------------------

lv::LiveConfig config_of(const py::handle& c) {
    lv::LiveConfig x;
    x.spec = rx::spec(c.attr("spec"));
    const py::object ws = c.attr("waveforms");
    if (!ws.is_none()) {
        std::vector<std::int64_t> v;
        for (const py::handle w : py::reinterpret_borrow<py::iterable>(py::iter(ws))) v.push_back(rx::as_int(w));
        x.waveforms = std::move(v);
    }
    x.refresh_s = rx::as_double(c.attr("refresh_s"));
    x.first_s = rx::as_double(c.attr("first_s"));
    x.detect_until_s = rx::as_double(c.attr("detect_until_s"));
    x.recheck_until_s = rx::as_double(c.attr("recheck_until_s"));
    x.empty_refresh_s = rx::as_double(c.attr("empty_refresh_s"));
    x.keep_done_s = rx::as_double(c.attr("keep_done_s"));
    x.estimator = rx::as_string(c.attr("estimator"));
    x.precision = rx::as_string(c.attr("precision"));
    const py::object model = c.attr("model");
    if (!model.is_none()) x.model = fspath::fs_path(model).bytes;
    x.render = rx::as_bool(c.attr("render"));
    x.background = rx::as_bool(c.attr("background"));
    x.finish_process = rx::as_bool(c.attr("finish_process"));
    return x;
}

// --- records to Python ------------------------------------------------------------------------------

py::object tile_py(const lv::Tile& t) {
    py::list conf;
    for (const auto& c : t.confidence) conf.append(opt_float_py(c));
    return live_mod().attr("Tile")(
        "id"_a = rx::to_str(t.id), "q"_a = t.q, "slot_utc"_a = rx::to_str(t.slot_utc), "frame"_a = rx::to_str(t.frame),
        "f_hz"_a = t.f_hz, "dial_hz"_a = opt_int_py(t.dial_hz), "status"_a = rx::to_str(t.status),
        "snr_db"_a = opt_float_py(t.snr_db), "z_ref"_a = opt_float_py(t.z_ref), "callsign"_a = opt_str_py(t.callsign),
        "grid"_a = opt_str_py(t.grid), "picture_id"_a = opt_str_py(t.picture_id), "mode"_a = opt_str_py(t.mode),
        "segment"_a = opt_int_py(t.segment), "cw_text"_a = opt_str_py(t.cw_text), "progress"_a = t.progress,
        "heard"_a = t.heard, "received"_a = t.received, "mean_w_db"_a = opt_float_py(t.mean_w_db), "confidence"_a = conf,
        "passes"_a = t.passes, "image"_a = opt_str_py(t.image), "image_rev"_a = t.image_rev, "updated"_a = t.updated,
        "note"_a = rx::to_str(t.note), "uid"_a = opt_str_py(t.uid), "waveform"_a = rx::to_str(t.waveform));
}

py::list tiles_py(const std::vector<lv::Tile>& ts) {
    py::list out;
    for (const lv::Tile& t : ts) out.append(tile_py(t));
    return out;
}

// --- the listener -------------------------------------------------------------------------------------

struct PyListener {
    std::shared_ptr<Shared> sh;
    py::object dir;   // pathlib.Path(state_dir)
    std::unique_ptr<lv::LiveListener> core;

    ~PyListener() {
        if (!core) return;
        {
            py::gil_scoped_release nogil;
            core->close();
        }
        core.reset();   // (the seams' Python objects go with the GIL held)
    }
};

lv::LiveListener& core_of(PyListener& L) {
    if (!L.core) throw py::value_error("the listener is gone");
    return *L.core;
}

template <typename F>
auto nogil(F&& f) -> decltype(f()) {
    py::gil_scoped_release release;
    return f();
}

std::shared_ptr<pb::PassbandStore> passband_of(const py::object& o) {
    if (o.is_none()) return nullptr;
    pb::StoreOptions opt;
    std::string root;
    if (py::hasattr(o, "root") && !py::isinstance<py::str>(o)) {
        root = fspath::fs_path(o.attr("root")).bytes;
        if (py::hasattr(o, "fs")) opt.fs = static_cast<int>(rx::as_int(o.attr("fs")));
        if (py::hasattr(o, "scale")) opt.scale = rx::as_double(o.attr("scale"));
        if (py::hasattr(o, "keep_hours")) opt.keep_hours = rx::as_double(o.attr("keep_hours"));
        if (py::hasattr(o, "pack")) opt.pack = rx::as_bool(o.attr("pack"));
    } else {
        root = fspath::fs_path(o).bytes;
    }
    return nogil([&] { return std::make_shared<pb::PassbandStore>(root, opt); });
}

std::shared_ptr<store::Store> store_of(const py::object& o) {
    if (o.is_none()) return nullptr;
    if (!py::isinstance<st::PyStore>(o))
        throw py::type_error("the C++ listener takes an sstvae_native.qrss.store.Store (the C++ store), not " +
                             st::class_name(o));
    st::PyStore& ps = o.cast<st::PyStore&>();
    if (!ps.core) throw py::value_error("the store is closed");
    // not owned: the Python object keeps the store, and this keeps the object
    auto* keep = new py::object(o);
    return std::shared_ptr<store::Store>(ps.core.get(), [keep](store::Store*) {
        py::gil_scoped_acquire gil;
        delete keep;
    });
}

bool rule_on(const py::object& v, const char* which) {
    if (!v.is_none()) return rx::as_bool(v);
    const py::module_ rules = py::module_::import("sstvae.qrss.rules");
    return rx::as_bool(rules.attr("old")(rules.attr(which)));
}

std::unique_ptr<PyListener> make_listener(const py::object& state_dir, const py::object& cfg_in, const py::object& store_in,
                                          const py::object& passband, const py::object& log, const py::object& clock,
                                          const py::object& candidates, const py::object& detect,
                                          const py::object& receive_slot, const py::object& finish,
                                          const py::object& noise, const py::object& associate,
                                          const py::object& renderer, const py::object& old_edge_noise,
                                          const py::object& old_gating, bool low_priority, const py::object& dial_guard) {
    auto L = std::make_unique<PyListener>();
    L->dir = py::module_::import("pathlib").attr("Path")(state_dir);
    const std::string dir = fspath::fs_path(L->dir).bytes;
    const py::object c = cfg_in.is_none() ? live_mod().attr("LiveConfig")() : cfg_in;
    lv::LiveConfig config = config_of(c);
    L->sh = std::make_shared<Shared>();
    L->sh->cfg = c;
    L->sh->selection = c.attr("selection");
    L->sh->store = store_in;
    std::shared_ptr<store::Store> st_core = store_of(store_in);
    std::shared_ptr<pb::PassbandStore> pbs = passband_of(passband);
    lv::Seams seams = seams_of(L->sh, clock, log, candidates, detect, receive_slot, finish, noise, associate, renderer,
                               rule_on(old_edge_noise, "EDGE_NOISE"), rule_on(old_gating, "GATING"), low_priority);
    if (!dial_guard.is_none()) {
        const py::tuple g = py::tuple(dial_guard);
        if (g.size() != 2) throw py::value_error("dial_guard is (before, after) seconds");
        seams.dial_guard = {rx::as_double(g[0]), rx::as_double(g[1])};
    }
    L->core = nogil([&] {
        return std::make_unique<lv::LiveListener>(dir, std::move(config), std::move(st_core), std::move(pbs),
                                                  std::move(seams));
    });
    L->sh->joint = L->core->joint();
    return L;
}

py::object slot_py(const Shared& sh, const lv::SlotState& s) {
    py::dict tiles;
    for (const auto& [id, t] : s.tiles) tiles[rx::to_str(id)] = tile_py(t);
    py::object cands = py::none();
    if (s.cands) cands = sh.dets_py(*s.cands);
    py::tuple hints(Shared::hints_py(s.hints));
    return live_mod().attr("SlotState")(s.q, "dets"_a = sh.dets_py(s.dets), "tiles"_a = tiles,
                                        "last_refresh"_a = opt_float_py(s.last_refresh), "done"_a = s.done,
                                        "finishing"_a = s.finishing,
                                        "reader"_a = s.reader ? py::object(py::str("passband")) : py::object(py::none()),
                                        "cands"_a = cands, "hints"_a = hints);
}

// --- run's source --------------------------------------------------------------------------------------

struct SourceError {};   // the Python source raised (kept in PySource::err)

class PySource : public lv::Source {
public:
    PySource(py::object src, py::object stop, std::atomic<bool>* flag)
        : src_(std::move(src)), stop_(std::move(stop)), flag_(flag) {}

    std::optional<lv::Chunk> next(double) override {
        py::gil_scoped_acquire gil;
        try {
            if (!it_) it_ = py::iter(src_.attr("chunks")());
            PyObject* item = PyIter_Next(it_.ptr());
            if (!item) {
                if (PyErr_Occurred()) throw py::error_already_set();
                eof_ = true;
                return std::nullopt;
            }
            const py::object chunk = py::reinterpret_steal<py::object>(item);
            const py::tuple xt = py::tuple(chunk);
            if (xt.size() != 2) throw py::value_error("a chunk is (x, t_end)");
            if (!stop_.is_none() && rx::as_bool(stop_.attr("is_set")())) flag_->store(true);
            if (xt[0].is_none()) return std::nullopt;
            lv::Chunk c;
            c.x = rx::vec<double>(xt[0]);
            c.t_end = rx::as_double(xt[1]);
            return c;
        } catch (py::error_already_set& e) {
            err = e;
            throw SourceError{};
        }
    }
    bool eof() const override { return eof_; }
    std::string name() const override {
        py::gil_scoped_acquire gil;
        return py::str(src_.attr("name")).cast<std::string>();
    }

    std::optional<py::error_already_set> err;

private:
    py::object src_, stop_, it_;
    std::atomic<bool>* flag_;
    bool eof_ = false;
};

void bind(py::module_& qrss) {
    rx::register_translators();
    py::module_ m = qrss.def_submodule("listener", "native/core/qrss/listener (live.py's LiveListener and run)");
    m.attr("CLOSE_PACKER_S") = lv::CLOSE_PACKER_S;

    py::class_<lv::Renderer, std::shared_ptr<lv::Renderer>>(m, "Renderer",
                                                            "What draws a tile's picture (native renderers only).")
        .def("load", [](lv::Renderer& r, const py::object& codec_id, const py::object& warn) -> py::object {
            const std::optional<std::int64_t> cid = codec_id.is_none() ? std::nullopt
                                                                         : std::optional<std::int64_t>(rx::as_int(codec_id));
            const PyFn w(warn);
            const lv::Renderer::Warn fn = [w](const std::string& msg) {
                with_gil([&] {
                    if (!(*w).is_none()) (*w)(rx::to_str(msg));
                });
            };
            const std::optional<std::string> e = nogil([&] { return r.load(cid, fn); });
            return opt_str_py(e);
        }, py::arg("codec_id") = py::none(), py::arg("warn") = py::none())
        .def("check", [](lv::Renderer& r, const py::object& codec_id) {
            const std::optional<std::int64_t> cid = codec_id.is_none() ? std::nullopt
                                                                         : std::optional<std::int64_t>(rx::as_int(codec_id));
            return opt_str_py(nogil([&] { return r.check(cid); }));
        }, py::arg("codec_id") = py::none())
        .def("render", [](lv::Renderer& r, const py::object& S, const py::object& W, std::int64_t mode,
                          const py::object& codec_id) {
            const std::vector<double> s = rx::vec<double>(S), w = rx::vec<double>(W);
            const std::optional<std::int64_t> cid = codec_id.is_none() ? std::nullopt
                                                                         : std::optional<std::int64_t>(rx::as_int(codec_id));
            const sstvae::images::Picture p = nogil([&] { return r.render(s, w, mode, cid); });
            py::array_t<std::uint8_t> out(std::vector<py::ssize_t>{p.height, p.width, 3});
            std::memcpy(out.mutable_data(), p.rgb.data(), p.rgb.size());
            return out;
        }, py::arg("S"), py::arg("W"), py::arg("mode"), py::arg("codec_id") = py::none());

    py::class_<lv::DialLog, std::unique_ptr<lv::DialLog, py::nodelete>>(m, "_Dial", "The listener's DialLog.")
        .def("refresh", [](lv::DialLog& d) { nogil([&] { d.refresh(); }); })
        .def("dial_at", [](const lv::DialLog& d, double t) { return opt_int_py(nogil([&] { return d.dial_at(t); })); },
             py::arg("t"))
        .def_property_readonly("path", [](const lv::DialLog& d) { return fspath::fs_text(d.path()); });

    py::class_<PyListener>(m, "LiveListener", "The live QRSSTVAE listener (live.LiveListener, in C++).")
        .def(py::init(&make_listener), py::arg("state_dir"), py::arg("cfg") = py::none(), py::arg("store") = py::none(),
             py::arg("passband") = py::none(), py::arg("log") = py::none(), py::kw_only(),
             py::arg("clock") = py::none(), py::arg("candidates") = py::none(), py::arg("detect") = py::none(),
             py::arg("receive_slot") = py::none(), py::arg("finish") = py::none(), py::arg("noise") = py::none(),
             py::arg("associate") = py::none(), py::arg("renderer") = py::none(),
             py::arg("old_edge_noise") = py::none(), py::arg("old_gating") = py::none(),
             py::arg("low_priority") = true, py::arg("dial_guard") = py::none())
        .def("feed", [](PyListener& L, const py::object& x, double t_end) {
            const std::vector<double> v = rx::vec<double>(x);
            lv::LiveListener& c = core_of(L);
            nogil([&] { c.feed(v, t_end); });
        }, py::arg("x"), py::arg("t_end"))
        .def("step", [](PyListener& L, double now) {
            lv::LiveListener& c = core_of(L);
            return nogil([&] { return c.step(now); });
        }, py::arg("now"))
        .def("write_state", [](PyListener& L, bool listening) {
            lv::LiveListener& c = core_of(L);
            nogil([&] { c.write_state(listening); });
        }, py::arg("listening") = true)
        .def("log", [](PyListener& L, const py::object& msg) {
            const std::string s = rx::as_string(py::str(msg));
            lv::LiveListener& c = core_of(L);
            nogil([&] { c.log(s); });
        }, py::arg("msg"))
        .def("backfill", [](PyListener& L, double now) {
            lv::LiveListener& c = core_of(L);
            return nogil([&] { return c.backfill(now); });
        }, py::arg("now"))
        .def("recover", [](PyListener& L, double now, double hours) {
            lv::LiveListener& c = core_of(L);
            return int_list(nogil([&] { return c.recover(now, hours); }));
        }, py::arg("now"), py::arg("hours") = lv::RECOVER_HOURS)
        .def("tiles", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return tiles_py(nogil([&] { return c.tiles(); }));
        })
        .def("active_slots", [](PyListener& L, double now) {
            lv::LiveListener& c = core_of(L);
            return int_list(nogil([&] { return c.active_slots(now); }));
        }, py::arg("now"))
        .def("_progress", [](PyListener& L, std::int64_t q) {
            lv::LiveListener& c = core_of(L);
            return nogil([&] { return c.progress(q); });
        }, py::arg("q"))
        .def("_interval", [](PyListener& L, std::int64_t q, double now) {
            lv::LiveListener& c = core_of(L);
            return nogil([&] { return c.interval(q, now); });
        }, py::arg("q"), py::arg("now"))
        .def("_hints", [](PyListener& L, std::int64_t q, const py::object& dial) {
            const std::optional<std::int64_t> d = dial.is_none() ? std::nullopt
                                                                   : std::optional<std::int64_t>(rx::as_int(dial));
            lv::LiveListener& c = core_of(L);
            return Shared::hints_py(nogil([&] { return c.hints(q, d); }));
        }, py::arg("q"), py::arg("dial") = py::none())
        .def("_match", [](PyListener& L, std::int64_t q, double f_hz, const py::object& dial,
                          const std::string& waveform) -> py::object {
            const std::optional<std::int64_t> d = dial.is_none() ? std::nullopt
                                                                   : std::optional<std::int64_t>(rx::as_int(dial));
            lv::LiveListener& c = core_of(L);
            const std::optional<lv::Tile> t = nogil([&] { return c.match(q, f_hz, d, waveform); });
            return t ? tile_py(*t) : py::object(py::none());
        }, py::arg("q"), py::arg("f_hz"), py::arg("dial") = py::none(), py::arg("waveform") = "CE")
        .def("idle", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return nogil([&] { return c.idle(); });
        })
        .def("_finish_idle", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return nogil([&] { return c.finish_idle(); });
        })
        .def("wait_idle", [](PyListener& L, const py::object& timeout) {
            const std::optional<double> t = timeout.is_none() ? std::nullopt
                                                               : std::optional<double>(rx::as_double(timeout));
            lv::LiveListener& c = core_of(L);
            return nogil([&] { return c.wait_idle(t); });
        }, py::arg("timeout") = py::none())
        .def("replay_end", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return opt_float_py(nogil([&] { return c.replay_end(); }));
        })
        .def("close", [](PyListener& L) {
            if (!L.core) return;
            lv::LiveListener& c = *L.core;
            nogil([&] { c.close(); });
        })
        .def_property_readonly("lines", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            py::list out;
            for (const std::string& s : nogil([&] { return c.lines(); })) out.append(rx::to_str(s));
            return out;
        })
        .def_property_readonly("now", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return nogil([&] { return c.now(); });
        })
        .def_property_readonly("audio_end", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return opt_float_py(nogil([&] { return c.audio_end(); }));
        })
        .def_property_readonly("busy", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return opt_str_py(nogil([&] { return c.busy(); }));
        })
        .def_property_readonly("codec_error", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return opt_str_py(nogil([&] { return c.codec_error(); }));
        })
        .def_property("source", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return rx::to_str(nogil([&] { return c.source(); }));
        }, [](PyListener& L, const py::object& v) {
            const std::string s = rx::as_string(py::str(v));
            lv::LiveListener& c = core_of(L);
            nogil([&] { c.set_source(s); });
        })
        .def_property_readonly("finished", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            py::set out;
            for (const std::int64_t q : nogil([&] { return c.finished(); })) out.add(py::int_(q));
            return out;
        })
        .def_property_readonly("done_tiles", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return tiles_py(nogil([&] { return c.done_tiles(); }));
        })
        .def_property_readonly("slots", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            const std::vector<lv::SlotState> ss = nogil([&] { return c.slots(); });
            py::dict out;
            for (const lv::SlotState& s : ss) out[py::int_(s.q)] = slot_py(*L.sh, s);
            return out;
        })
        .def_property_readonly("recovering", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return nogil([&] { return c.recovering(); });
        })
        .def_property_readonly("next_index", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return opt_int_py(nogil([&] { return c.next_index(); }));
        })
        .def_property_readonly("dur", [](PyListener& L) { return core_of(L).dur(); })
        .def_property_readonly("joint", [](PyListener& L) { return core_of(L).joint(); })
        .def_property_readonly("cfg", [](PyListener& L) { return L.sh->cfg; })
        .def_property_readonly("specs", [](PyListener& L) { return L.sh->cfg.attr("specs"); })
        .def_property_readonly("selection", [](PyListener& L) { return L.sh->selection; })
        .def_property_readonly("dir", [](PyListener& L) { return L.dir; })
        .def_property_readonly("ring", py::cpp_function([](PyListener& L) {
            return py::cast(&core_of(L).ring(), py::return_value_policy::reference);
        }, py::keep_alive<0, 1>()))
        .def_property_readonly("dial", py::cpp_function([](PyListener& L) {
            return py::cast(&core_of(L).dial(), py::return_value_policy::reference);
        }, py::keep_alive<0, 1>()))
        .def("clock", [](PyListener& L) {
            lv::LiveListener& c = core_of(L);
            return nogil([&] { return c.clock(); });
        });

    m.def("run", [](PyListener& L, const py::object& source, bool realtime, const py::object& stop) {
        lv::LiveListener& c = core_of(L);
        std::atomic<bool> flag{false};
        PySource src(source, stop, &flag);
        try {
            nogil([&] { lv::run(c, src, realtime, &flag); });
        } catch (const SourceError&) {
            if (src.err) throw *src.err;
            throw py::value_error("the source failed");
        }
    }, py::arg("listener"), py::arg("source"), py::arg("realtime"), py::arg("stop") = py::none());

    // error_name / error_text, for the tests of the log lines' texts
    m.def("_seam_error_text", [](const std::string& type_name, const std::string& text) {
        return lv::error_text(lv::SeamError(type_name, text));
    }, py::arg("type_name"), py::arg("text"));
}

const Registrar reg("listener", 97, &bind);

}  // namespace
}  // namespace sstvae_bindings::qrss
