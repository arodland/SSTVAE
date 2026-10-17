// The slot half of sstvae/qrss/receiver.py: see receiver_slot.hpp, and
// receiver_slot_flow.hpp for the orchestration this runs over the C++
// records.

#include "qrss/receiver_slot.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "math/npcomplex.hpp"
#include "qrss/ltrack_timing.hpp"
#include "qrss/npsum.hpp"
#include "qrss/rules.hpp"
#include "qrss/track.hpp"
#include "qrss/track_timing.hpp"
#include "qrss/wavread.hpp"

namespace sstvae::qrss::receiver {
namespace {

namespace np = math::np;
using cplx = std::complex<double>;
using frame::FrameSpec;
using types::Detection;
using types::PassResult;

[[noreturn]] void no_timing() {
    throw std::invalid_argument("'NoneType' object has no attribute 'tau0' (a detection without timing)");
}

double tau0_of(const Detection& d) {
    if (!d.timing) no_timing();
    return d.timing->tau0;
}

// |z|^2 as np.abs(z) ** 2
inline double abs2(cplx z) {
    const double a = np::abs(z);
    return a * a;
}

[[noreturn]] void index_error(std::int64_t i, std::size_t size) {
    throw std::out_of_range("index " + std::to_string(i) + " is out of bounds for axis 0 with size " +
                            std::to_string(size));
}

// --- policies over plain records (the rules' own entry points) ----------------------------

// Detections by index into a span.
struct DetIdx {
    using Det = std::size_t;
    std::span<const Detection> d;
    std::optional<bool> gating;
    double f_hz(Det i) const { return d[i].f_hz; }
    double centre(Det i) const { return centre_hz(d[i]); }
    bool method_is(Det i, const char* m) const { return d[i].method == m; }
    double a3_z(Det i) const { return d[i].a3_z; }
    double z_ref(Det i) const { return d[i].z_ref; }
    double tau0(Det i) const { return tau0_of(d[i]); }
    bool is_l(Det i) const { return d[i].waveform == config::WAVEFORM_L; }
    bool old_gating() const { return gating ? *gating : rules::old(rules::GATING); }
};

// Passes by index into a span (index n is `p` itself).
struct PassIdx {
    using Pass = std::size_t;
    std::span<const PassResult> out;
    const PassResult* p = nullptr;
    const PassResult& at(Pass i) const { return i == out.size() ? *p : out[i]; }
    double p_f_hz(Pass i) const { return at(i).f_hz; }
    double p_z_ref(Pass i) const { return at(i).report.z_ref; }
    bool p_suspect(Pass i) const { return at(i).report.suspect; }
    double p_centre(Pass i) const { return centre_hz(at(i)); }
};

std::vector<std::size_t> iota(std::size_t n) {
    std::vector<std::size_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = i;
    return v;
}

// --- the C++ policy ------------------------------------------------------------------------

double steady_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Typed {
    using Det = std::shared_ptr<Detection>;
    using Pass = std::shared_ptr<PassResult>;
    using Prep = std::shared_ptr<const Prepared>;
    using Cap = CaptureRef;
    using AcqCap = const Prepared*;
    using Sel = std::vector<FrameSpec>;
    using Spec = FrameSpec;
    using Chan = std::shared_ptr<const track::ChanCapture>;
    using Track = std::shared_ptr<track::TrackResult>;
    using FrameMap = acquire::FrameMap;
    using WKey = std::int64_t;
    using Others = std::vector<double>;
    using Cands = std::optional<std::vector<Det>>;
    using LiveOnly = bool;
    using Estimator = std::string;
    using Erase = receiver::Erase;
    using RoundB = bool;
    using MaxC = std::optional<std::int64_t>;
    using Budget = std::optional<double>;
    using Hints = std::vector<acquire::Hint>;
    using Skip = std::vector<double>;

    const SlotSeams& s;

    static std::vector<Det> wrap(std::vector<Detection>&& v) {
        std::vector<Det> out;
        out.reserve(v.size());
        for (Detection& d : v) out.push_back(std::make_shared<Detection>(std::move(d)));
        return out;
    }
    static std::optional<Det> wrap(std::optional<Detection>&& v) {
        if (!v) return std::nullopt;
        return std::make_shared<Detection>(std::move(*v));
    }
    track::VerifyArgs args(const Erase& e, bool fit) const {
        track::VerifyArgs a;
        a.fit = fit;
        a.erase_s = e;
        a.noise_psd = s.pass.noise_psd;
        return a;
    }

    // values
    double f_hz(const Det& d) const { return d->f_hz; }
    double centre(const Det& d) const { return centre_hz(*d); }
    bool method_is(const Det& d, const char* m) const { return d->method == m; }
    double a3_z(const Det& d) const { return d->a3_z; }
    double z_ref(const Det& d) const { return d->z_ref; }
    double tau0(const Det& d) const { return tau0_of(*d); }
    double lead_in(const Det& d) const { return d->lead_in_s; }
    void set_lead_in(Det& d, double x) const { d->lead_in_s = x; }
    bool is_l(const Det& d) const { return d->waveform == config::WAVEFORM_L; }
    double p_f_hz(const Pass& x) const { return x->f_hz; }
    double p_tau0(const Pass& x) const { return x->timing.tau0; }
    double p_z_ref(const Pass& x) const { return x->report.z_ref; }
    bool p_suspect(const Pass& x) const { return x->report.suspect; }
    double p_centre(const Pass& x) const { return centre_hz(*x); }
    bool same_waveform(const Pass& g, const Pass& x) const { return g->waveform == x->waveform; }

    // frames
    bool is_joint(const Sel& sel) const { return receiver::is_joint(std::span<const FrameSpec>(sel)); }
    Sel frames(const Sel& sel) const { return receiver::frames(std::span<const FrameSpec>(sel)); }
    Spec single(const Sel& sel) const { return receiver::single(std::span<const FrameSpec>(sel)); }
    FrameMap frame_map(const Sel& specs) const { return acquire::frame_map_of(specs); }
    std::optional<Spec> smap_get(const FrameMap& m, const Det& d) const {
        for (const auto& [w, sp] : m)
            if (w == d->waveform) return sp;
        return std::nullopt;
    }
    const FrameMap& smap_items(const FrameMap& m) const { return m; }
    bool waveform_eq(const Det& d, WKey w) const { return d->waveform == w; }
    Spec spec_for(const FrameMap& m, const Sel& specs, const Det& d) const {
        if (const auto sp = smap_get(m, d)) return *sp;
        return frame::twin(specs.at(0), static_cast<int>(d->waveform));
    }
    double t_sym(const Spec& sp) const { return sp.t_sym(); }
    bool spec_is_l(const Spec& sp) const { return sp.waveform() == frame::Waveform::L; }

    // arguments
    static bool given(const Cands& c) { return c.has_value(); }
    static const std::vector<Det>& listed(const Cands& c) { return *c; }
    Hints ce_hints(const Hints& h) const { return receiver::ce_hints(h); }
    std::vector<double> floats(const Skip& v) const { return v; }
    std::vector<double> centres(const Skip& v) const { return v; }
    double now() const { return s.now ? s.now() : steady_now(); }
    bool old_gating() const { return s.old_gating ? *s.old_gating : rules::old(rules::GATING); }
    std::optional<double> deadline(const Budget& b) const {
        if (!b) return std::nullopt;
        return now() + *b;
    }
    bool reached(const MaxC& m, std::int64_t taken) const { return m && taken >= *m; }
    Others others_f(const std::vector<Det>& dets, std::size_t i) const {
        Others o;
        for (const Det& g : dets)
            if (g != dets[i]) o.push_back(g->f_hz);
        return o;
    }
    Others others_c(const std::vector<Det>& dets, std::size_t i) const {
        Others o;
        for (const Det& g : dets)
            if (g != dets[i]) o.push_back(centre_hz(*g));
        return o;
    }

    // stages
    AcqCap capture_of(const Prep& prep) const { return prep.get(); }
    acquire::AcquireOptions acq_opt(bool live_only, const Hints& h) const {
        acquire::AcquireOptions o;
        o.live_only = live_only;
        o.preprocessed = true;
        o.hints = h;
        return o;
    }
    std::vector<Det> acquire_dets(AcqCap cap, const Sel& specs, bool live_only, const Hints& h) const {
        const acquire::Acquisition a =
            acquire::acquire(std::span<const acquire::cf>(cap->fe), cap->t0_index, static_cast<double>(cap->fs),
                             std::span<const FrameSpec>(specs), acq_opt(live_only, h), s.acquire);
        return wrap(a.detections(std::span<const FrameSpec>(specs)));
    }
    std::vector<Det> acquire_dets(AcqCap cap, const Spec& spec, bool live_only, const Hints& h) const {
        const acquire::Acquisition a =
            acquire::acquire(std::span<const acquire::cf>(cap->fe), cap->t0_index, static_cast<double>(cap->fs), spec,
                             acq_opt(live_only, h), s.acquire);
        return wrap(a.detections(spec));
    }
    std::vector<Det> candidates(const Prep& prep, const Spec& spec, bool live_only, const Hints& h) const {
        if (s.candidates) return wrap(s.candidates(*prep, std::span<const FrameSpec>(&spec, 1), false, live_only, h));
        return flow::candidates(*this, prep, Sel{spec}, live_only, h);
    }
    std::vector<Det> candidates(const Prep& prep, const Sel& specs, bool live_only, const Hints& h) const {
        if (s.candidates) return wrap(s.candidates(*prep, specs, true, live_only, h));
        return flow::candidates(*this, prep, specs, live_only, h);
    }
    Chan channel_for(const Prep& prep, const Spec& spec, const Det& d, const std::vector<double>& others) const {
        if (s.pass.channel_for) return s.pass.channel_for(*prep, spec, *d, others, std::nullopt);
        return std::make_shared<const track::ChanCapture>(receiver::channel_for(*prep, spec, *d, others));
    }
    std::optional<Det> verify(const Chan& chan, const Spec& spec, const Det& d, const Erase& e) const {
        if (s.pass.verify) return wrap(s.pass.verify(chan, spec, *d, e));
        return wrap(track::verify(chan, spec, *d, args(e, true)));
    }
    std::optional<Det> verify_l(const Chan& chan, const Spec& spec, const Det& d, const Erase& e) const {
        if (s.pass.verify_l) return wrap(s.pass.verify_l(chan, spec, *d, e));
        return wrap(ltrack::verify_l(chan, spec, *d, args(e, true)));
    }
    std::optional<Track> verify_track(const Chan& chan, const Spec& spec, const Det& v, const Erase& e) const {
        if (s.verify_track) {
            Track t = s.verify_track(chan, spec, *v, e);
            if (!t) return std::nullopt;
            return t;
        }
        return std::make_shared<track::TrackResult>(track::verify_track(chan, spec, *v, args(e, false)));
    }
    double ce_resid_ratio(const Track& tr) const {
        return s.ce_resid_ratio ? s.ce_resid_ratio(*tr) : receiver::ce_resid_ratio(*tr);
    }
    double lead_in_span(const Prep& prep, const Det& v, std::optional<double> t_sym) const {
        const double f = v->f_hz;
        const double tau0_s = tau0(v) / static_cast<double>(config::CH_FS);
        if (s.lead_in_span) return s.lead_in_span(*prep, f, tau0_s, t_sym);
        return acquire::lead_in_span(std::span<const acquire::cf>(prep->fe), prep->t0_index, f, 0.0, tau0_s,
                                     static_cast<double>(prep->fs), rxconfig::acquire::LEAD_Z,
                                     t_sym ? *t_sym : config::T_SYM, s.acquire.noise_psd);
    }
    std::optional<Det> gate(const Prep& prep, const Spec& spec, const Det& d, const std::vector<double>& others,
                            const Erase& e) const {
        if (s.gate) return wrap(s.gate(*prep, spec, *d, others, e));
        return flow::gate(*this, prep, spec, d, others, e);
    }
    flow::DetectArgs<const Typed> detect_args(bool live_only, const MaxC& m, const Budget& b, const Erase& e,
                                              const Hints& h) const {
        flow::DetectArgs<const Typed> a;
        a.live_only = live_only;
        a.max_candidates = m;
        a.budget_s = b;
        a.erase_s = e;
        a.hints = h;
        return a;
    }
    std::vector<Det> detect(const Prep& prep, const Spec& spec, bool live_only, const MaxC& m, const Budget& b,
                            const Erase& e, const Hints& h) const {
        if (s.detect) return wrap(s.detect(*prep, std::span<const FrameSpec>(&spec, 1), false, live_only, m, b, e, h));
        return flow::detect(*this, prep, Sel{spec}, detect_args(live_only, m, b, e, h));
    }
    std::vector<Det> detect(const Prep& prep, const Sel& specs, bool live_only, const MaxC& m, const Budget& b,
                            const Erase& e, const Hints& h) const {
        if (s.detect) return wrap(s.detect(*prep, specs, true, live_only, m, b, e, h));
        return flow::detect(*this, prep, specs, detect_args(live_only, m, b, e, h));
    }
    std::vector<Det> detect_joint(const Prep& prep, const Sel& specs, const flow::DetectArgs<const Typed>& a) const {
        return flow::detect_joint(*this, prep, specs, a);
    }
    std::vector<Pass> receive_slot_joint(const Prep& prep, const Sel& specs,
                                         const flow::SlotArgs<const Typed>& a) const {
        return flow::receive_slot_joint(*this, prep, specs, a);
    }
    Prep prepare(const Cap& cap) const { return s.pass.prepare ? s.pass.prepare(cap) : receiver::prepare(cap); }
    std::optional<Pass> receive_pass(const Prep& prep, const Spec& spec, const Det& d, const Estimator& est,
                                     const Others& others, bool round_b, const Erase& e) const {
        PassArgs a;
        a.estimator = est;
        a.exclude_hz = others;
        a.round_b = round_b;
        a.erase_s = e;
        std::optional<PassResult> r = s.receive_pass
                                          ? s.receive_pass(*prep, spec, *d, a)
                                          : receiver::receive_pass(CaptureRef(prep.get()), spec, *d, a, s.pass).pass;
        if (!r) return std::nullopt;
        return std::make_shared<PassResult>(std::move(*r));
    }
};

std::vector<Detection> unwrap(const std::vector<Typed::Det>& v) {
    std::vector<Detection> out;
    out.reserve(v.size());
    for (const auto& d : v) out.push_back(*d);
    return out;
}

Typed::Prep borrow(const Prepared& p) { return std::shared_ptr<const Prepared>(std::shared_ptr<const Prepared>(), &p); }

std::vector<Typed::Det> own(const std::vector<Detection>& v) {
    std::vector<Typed::Det> out;
    out.reserve(v.size());
    for (const Detection& d : v) out.push_back(std::make_shared<Detection>(d));
    return out;
}

}  // namespace

// --- the rules ----------------------------------------------------------------------------------

bool ghost_of(double f_hz, double tau0, double g_f_hz, double g_tau0, double t_sym) {
    return flow::ghost_of(f_hz, tau0, g_f_hz, g_tau0, t_sym);
}

std::vector<std::size_t> aliases_idx(std::span<const Detection> dets, double t_sym) {
    DetIdx p{dets, std::nullopt};
    return flow::aliases(p, iota(dets.size()), t_sym);
}

std::vector<Detection> aliases(std::span<const Detection> dets, double t_sym) {
    std::vector<Detection> out;
    for (const std::size_t i : aliases_idx(dets, t_sym)) out.push_back(dets[i]);
    return out;
}

std::vector<std::size_t> ghosts_idx(std::span<const Detection> dets, double t_sym) {
    DetIdx p{dets, std::nullopt};
    return flow::ghosts(p, iota(dets.size()), t_sym);
}

std::vector<Detection> ghosts(std::span<const Detection> dets, double t_sym) {
    std::vector<Detection> out;
    for (const std::size_t i : ghosts_idx(dets, t_sym)) out.push_back(dets[i]);
    return out;
}

bool suspect_neighbour(const PassResult& p, std::span<const PassResult> out) {
    PassIdx x{out, &p};
    return flow::suspect_neighbour(x, out.size(), iota(out.size()));
}

bool suspect_beside(const PassResult& p, std::span<const PassResult> out) {
    PassIdx x{out, &p};
    return flow::suspect_beside(x, out.size(), iota(out.size()));
}

std::vector<A3Line> a3_lines(std::span<const Detection> cands) {
    DetIdx p{cands, std::nullopt};
    return flow::a3_lines(p, iota(cands.size()));
}

bool owned(const Detection& d, std::span<const Detection> found, std::span<const A3Line> a3,
           std::optional<bool> old_gating) {
    // d is index n of found ++ [d]
    std::vector<Detection> all(found.begin(), found.end());
    all.push_back(d);
    DetIdx p{all, old_gating};
    return flow::owned(p, found.size(), iota(found.size()), a3);
}

double ce_resid_ratio(const track::TrackResult& tr) {
    const track::Classes* cl = tr.ce_classes();
    if (!cl) throw std::invalid_argument("_ce_resid_ratio: the track has no CE classes");
    const FrameSpec& spec = tr.spec;
    const double keyed_from = spec.keyed_start_pos() - tr.lead_in_s / config::T_SYM;
    const track::Templates t = track::templates(spec, tr.pos, cl->mu, cl->nu, nullptr, false, keyed_from);
    const auto lay = frame::layout(spec);
    if (cl->known.size() != lay->pos.size())
        throw std::out_of_range("boolean index did not match indexed array along axis 0; size of axis is " +
                                std::to_string(lay->pos.size()) + " but size of corresponding boolean axis is " +
                                std::to_string(cl->known.size()));
    const std::int64_t p_lo = tr.p_lo();
    const auto npos = static_cast<std::int64_t>(tr.pos.size());
    std::vector<std::size_t> gk;
    for (std::size_t s = 0; s < cl->known.size(); ++s) {
        if (!cl->known[s]) continue;
        const std::int64_t g = lay->pos[s] - p_lo;
        if (g < 0 || g >= npos) continue;
        if (static_cast<std::size_t>(g) >= tr.psi.size()) index_error(g, tr.psi.size());
        if (tr.psi[static_cast<std::size_t>(g)] >= rxconfig::track::PSI_MIN) gk.push_back(static_cast<std::size_t>(g));
    }
    if (gk.empty()) return 1.0;
    std::vector<double> num, den;
    for (const std::size_t g : gk) {
        const cplx m = tr.m.at(g), u = tr.u.at(g);
        const double N = tr.N.at(g), P = tr.P.at(g);
        if (g >= t.c.size()) index_error(static_cast<std::int64_t>(g), t.c.size());
        const cplx c = t.c[g];
        const double u2 = abs2(u), ci = c.imag(), c2 = abs2(c);
        const double V = ci * ci * (N * u2 + c2 * u2 * P + N * P) / 2;
        if (!(V > 0)) continue;
        const double w = ci * ci * u2 / V;
        const double om = w * w * (ci * ci) * u2;
        const cplx uc = np::mul(u, c);
        const double e2 = abs2(cplx(m.real() - uc.real(), m.imag() - uc.imag()));
        const double e0 = N + c2 * P;
        num.push_back(om * e2);
        den.push_back(om * e0);
    }
    if (num.empty()) return 1.0;
    const double d = npsum::sum(std::span<const double>(den));
    return d > 0 ? npsum::sum(std::span<const double>(num)) / d : 1.0;
}

// --- acquisition and gate V ---------------------------------------------------------------------

std::vector<Detection> candidates(const Prepared& prep, Selection sel, bool live_only,
                                  std::span<const acquire::Hint> hints, const SlotSeams& seams) {
    const Typed p{seams};
    return unwrap(flow::candidates(p, borrow(prep), Typed::Sel(sel.begin(), sel.end()), live_only,
                                   Typed::Hints(hints.begin(), hints.end())));
}

namespace {
flow::DetectArgs<const Typed> detect_args(const DetectOptions& opt) {
    flow::DetectArgs<const Typed> a;
    a.live_only = opt.live_only;
    a.max_candidates = opt.max_candidates;
    a.budget_s = opt.budget_s;
    a.erase_s = opt.erase_s;
    if (opt.cands) a.cands = own(*opt.cands);
    a.skip_hz = opt.skip_hz;
    a.hints = opt.hints;
    return a;
}
}  // namespace

std::vector<Detection> detect(const Prepared& prep, Selection sel, const DetectOptions& opt, const SlotSeams& seams) {
    const Typed p{seams};
    return unwrap(flow::detect(p, borrow(prep), Typed::Sel(sel.begin(), sel.end()), detect_args(opt)));
}

std::vector<Detection> detect_joint(const Prepared& prep, Selection specs, const DetectOptions& opt,
                                    const SlotSeams& seams) {
    const Typed p{seams};
    return unwrap(flow::detect_joint(p, borrow(prep), Typed::Sel(specs.begin(), specs.end()), detect_args(opt)));
}

std::optional<Detection> gate(const Prepared& prep, const FrameSpec& spec, const Detection& d,
                              std::span<const double> others, std::span<const std::pair<double, double>> erase_s,
                              const SlotSeams& seams) {
    const Typed p{seams};
    const std::optional<Typed::Det> v =
        flow::gate(p, borrow(prep), spec, std::make_shared<Detection>(d), std::vector<double>(others.begin(), others.end()),
                   Erase(erase_s.begin(), erase_s.end()));
    if (!v) return std::nullopt;
    return **v;
}

// --- the slot -------------------------------------------------------------------------------------

namespace {
flow::SlotArgs<const Typed> slot_args(const SlotOptions& opt) {
    flow::SlotArgs<const Typed> a;
    a.live_only = opt.live_only;
    a.estimator = opt.estimator;
    a.erase_s = opt.erase_s;
    a.round_b = opt.round_b;
    if (opt.dets) a.dets = own(*opt.dets);
    a.max_candidates = opt.max_candidates;
    a.budget_s = opt.budget_s;
    a.hints = opt.hints;
    return a;
}
std::vector<PassResult> unwrap(const std::vector<Typed::Pass>& r) {
    std::vector<PassResult> out;
    out.reserve(r.size());
    for (const auto& x : r) out.push_back(*x);
    return out;
}
}  // namespace

std::vector<PassResult> receive_slot(CaptureRef cap, Selection sel, const SlotOptions& opt, const SlotSeams& seams) {
    const Typed p{seams};
    return unwrap(flow::receive_slot(p, cap, Typed::Sel(sel.begin(), sel.end()), slot_args(opt)));
}

std::vector<PassResult> receive_slot_joint(const Prepared& prep, Selection specs, const SlotOptions& opt,
                                           const SlotSeams& seams) {
    const Typed p{seams};
    return unwrap(flow::receive_slot_joint(p, borrow(prep), Typed::Sel(specs.begin(), specs.end()), slot_args(opt)));
}

// --- a WAV file ---------------------------------------------------------------------------------------

frontend::Capture capture_from_audio(std::span<const double> x, std::int64_t q, std::optional<double> start_unix) {
    frontend::Capture c;
    c.fe = frontend::audio_to_fe(x);
    const double t0 = 900.0 * static_cast<double>(q) + 1.0;
    const double lead = start_unix ? t0 - *start_unix : 12.0;
    c.q = q;
    c.t0_index = lead * static_cast<double>(config::FE_FS);
    c.fs = config::FE_FS;
    return c;
}

frontend::Capture capture_from_wav(const std::string& path, std::int64_t q, std::optional<double> start_unix) {
    const std::vector<double> x = wav::read_wav(path);
    return capture_from_audio(x, q, start_unix);
}

std::vector<PassResult> receive_wav(const std::string& path, std::int64_t q, std::optional<double> start_unix,
                                    Selection sel, const std::string& estimator, const SlotSeams& seams) {
    const frontend::Capture cap = capture_from_wav(path, q, start_unix);
    SlotOptions opt;
    opt.estimator = estimator;
    return receive_slot(CaptureRef(&cap), sel, opt, seams);
}

}  // namespace sstvae::qrss::receiver
