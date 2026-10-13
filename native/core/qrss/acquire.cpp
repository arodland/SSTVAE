// acquire.py's arbitration and acquire(): see acquire.hpp.

#include "qrss/acquire.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "math/errors.hpp"
#include "qrss/acquire_stats.hpp"

namespace sstvae::qrss::acquire {
namespace {

namespace rxa = rxconfig::acquire;

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// The span acquire() works on, with the copy it owns when it made one.
struct Fe {
    std::vector<cf> own;
    FeView view;
};

template <typename S>
Fe prepare(std::span<const S> fe, double fs, bool preprocessed, const AcquireSeams& seams) {
    Fe out;
    if (preprocessed) {
        out.view = fe;
        return out;
    }
    if (seams.prepare) {
        out.own = seams.prepare(FeView(fe), fs);
    } else {
        // frontend.blank's np.asarray(fe, dtype=np.complex64): complex128
        // rounds each part to float32
        std::vector<cf> c64;
        std::span<const cf> x;
        if constexpr (std::is_same_v<S, cf>) {
            x = fe;
        } else {
            c64.resize(fe.size());
            for (std::size_t i = 0; i < fe.size(); ++i)
                c64[i] = cf(static_cast<float>(fe[i].real()), static_cast<float>(fe[i].imag()));
            x = c64;
        }
        frontend::Blanked b = frontend::blank(x, 5.0, 2.0, 1.0, 0.05, fs, 1.0);
        frontend::Normalised n =
            frontend::normalise(b.fe, std::span<const float>(b.keep), 100.0, fs, 10.0, 0.25);
        out.own = std::move(n.fe);
    }
    out.view = std::span<const cf>(out.own);
    return out;
}

std::vector<double> f_hz_of(std::span<const types::CarrierCandidate> c) {
    std::vector<double> out;
    out.reserve(c.size());
    for (const auto& x : c) out.push_back(x.f_hz);
    return out;
}

// The stages, through the seams when they are set.
struct Stages {
    const FeView& fe;
    double t0;
    double fs;
    const AcquireSeams& s;

    CarrierStats stats() const {
        if (s.carrier_line_stats) return s.carrier_line_stats(fe, t0, fs);
        return std::visit([&](auto x) { return carrier_line_stats(x, t0, fs); }, fe);
    }
    std::vector<types::CarrierCandidate> lines(const CarrierStats& cs) const {
        if (s.carrier_lines) return s.carrier_lines(fe, t0, fs, cs);
        return carrier_lines(cs, fs);
    }
    std::vector<types::CarrierCandidate> pilots(const CarrierStats& cs,
                                                std::span<const types::CarrierCandidate> ls) const {
        if (s.pilot_lines) return s.pilot_lines(cs, ls);
        return pilot_lines(cs, ls);
    }
    types::CarrierCandidate hint(const CarrierStats& cs, double f) const {
        if (s.hint_line) return s.hint_line(cs, f, fs);
        return hint_line(cs, f, fs);
    }
    double threshold(std::int64_t M) const {
        if (s.a1_threshold) return s.a1_threshold(M, rxa::A1_PFA, fs);
        return a1_threshold(M, rxa::A1_PFA, fs);
    }
    std::vector<PreambleHit> at_line(const types::CarrierCandidate& c,
                                     std::span<const PreambleKind* const> kinds) const {
        if (s.line_search) return s.line_search(fe, t0, c, fs, kinds);
        const A2Input inp =
            std::visit([&](auto x) { return a2_input(x, t0, c.f_hz, c.drift_hz_per_min, fs); }, fe);
        std::vector<PreambleHit> out;
        out.reserve(kinds.size());
        for (const PreambleKind* k : kinds) out.push_back(preamble_search(inp, c, rxa::A2_PFA, *k));
        return out;
    }
    PreambleHit search(const types::CarrierCandidate& c, const PreambleKind* kind) const {
        if (s.search) return s.search(fe, t0, c, fs, kind);
        const PreambleKind& k = kind ? *kind : ce_pre();
        return std::visit([&](auto x) { return preamble_search(x, t0, c, fs, rxa::A2_PFA, k); }, fe);
    }
    double lead_in(const PreambleHit& h, std::optional<double> t_sym) const {
        if (s.lead_in_span) return s.lead_in_span(fe, t0, h.f_hz, h.drift_hz_per_min, h.tau0_s, fs, t_sym);
        const double ts = t_sym ? *t_sym : config::T_SYM;
        return std::visit(
            [&](auto x) {
                return lead_in_span(x, t0, h.f_hz, h.drift_hz_per_min, h.tau0_s, fs, rxa::LEAD_Z, ts, s.noise_psd);
            },
            fe);
    }
    std::vector<TbdHit> tbd(const frame::FrameSpec& spec, std::span<const types::CarrierCandidate> ls,
                            const std::vector<frame::FrameSpec>* shape_specs) const {
        if (s.track_before_detect) return s.track_before_detect(fe, t0, spec, ls, fs, shape_specs);
        TbdOptions opt;
        if (shape_specs) {
            opt.shape_specs = *shape_specs;
            if (s.shape_frames) {
                const auto* specs = shape_specs;
                const auto& fn = s.shape_frames;
                opt.keep_fn = [specs, &fn](std::span<const double> t) { return fn(t, *specs); };
            }
        }
        const std::vector<double> cf_hz = f_hz_of(ls);
        return std::visit([&](auto x) { return track_before_detect(x, t0, spec, cf_hz, fs, opt); }, fe);
    }
};

// Python's (p_value, -lam) < (p_value', -lam'): the first unequal item decides.
bool p_then_lam_less(const PreambleHit& a, const PreambleHit& b) {
    if (!(a.p_value == b.p_value)) return a.p_value < b.p_value;
    return -a.lam < -b.lam;
}

// sorted(range(n), key=lambda i: -hits[i].lam) (stable).
std::vector<std::size_t> by_lam(std::span<const PreambleHit> hits, std::span<const std::size_t> idx) {
    std::vector<std::size_t> o(idx.begin(), idx.end());
    std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return -hits[a].lam < -hits[b].lam; });
    return o;
}

std::vector<std::size_t> iota(std::size_t n) {
    std::vector<std::size_t> v(n);
    std::iota(v.begin(), v.end(), std::size_t{0});
    return v;
}

std::vector<PreambleHit> pick(std::span<const PreambleHit> hits, std::span<const std::size_t> idx) {
    std::vector<PreambleHit> out;
    out.reserve(idx.size());
    for (const std::size_t i : idx) out.push_back(hits[i]);
    return out;
}

const frame::FrameSpec* find_spec(const FrameMap& specs, std::int64_t w) {
    for (const auto& [k, s] : specs)
        if (k == w) return &s;
    return nullptr;
}

double duration(const frame::FrameSpec& s) { return s.keyed_end_pos() * s.t_sym(); }

types::Detection hint_detection(const types::CarrierCandidate& c, types::FreqPath path, std::int64_t waveform) {
    types::Detection d;
    d.f_hz = c.f_hz;
    d.path = std::move(path);
    d.timing = std::nullopt;
    d.z_ref = NaN;
    d.method = "hint";
    d.lead_in_s = 0.0;
    d.waveform = waveform;
    return d;
}

// The reference's TypeError for a (f, waveform) hint where it wants a
// number: abs(h.f_hz - f) once there is a hit to compare it with, else
// float(f).
[[noreturn]] void tuple_hint(bool have_hits) {
    if (have_hits) throw math::TypeError("unsupported operand type(s) for -: 'float' and 'tuple'");
    throw math::TypeError("float() argument must be a string or a real number, not 'tuple'");
}

constexpr double HINT_NEAR_HZ = rxa::HINT_SPAN_HZ + 0.5;

template <typename S>
Acquisition acquire_one(std::span<const S> fe_in, double t0, double fs, const frame::FrameSpec& spec,
                        const AcquireOptions& opt, const AcquireSeams& seams) {
    const Fe fe = prepare(fe_in, fs, opt.preprocessed, seams);
    const Stages st{fe.view, t0, fs, seams};
    const CarrierStats cs = st.stats();
    Acquisition acq;
    acq.lines = st.lines(cs);
    std::vector<PreambleHit> hits;
    hits.reserve(acq.lines.size() + opt.hints.size());
    for (const auto& c : acq.lines) hits.push_back(st.search(c, nullptr));
    for (const Hint& hint : opt.hints) {
        if (hint.waveform) tuple_hint(!hits.empty());
        const double f = hint.f_hz;
        bool done = false;
        for (const auto& h : hits)
            if (h.passed && std::abs(h.f_hz - f) <= HINT_NEAR_HZ) {
                done = true;
                break;
            }
        if (done) continue;
        std::optional<std::size_t> near;
        for (std::size_t i = 0; i < hits.size(); ++i)
            if (std::abs(hits[i].f_hz - f) <= HINT_NEAR_HZ) {
                near = i;
                break;
            }
        const types::CarrierCandidate line = st.hint(cs, f);
        PreambleHit h = near ? hits[*near] : st.search(line, nullptr);
        if (h.passed)
            hits.push_back(std::move(h));
        else
            acq.hinted.push_back(line);
    }
    std::vector<PreambleHit> passed;
    for (const auto& h : hits)
        if (h.passed) passed.push_back(h);
    acq.preamble = drop_symbol_rate_aliases(passed);
    for (auto& h : acq.preamble) h.lead_in_s = st.lead_in(h, std::nullopt);
    if (!opt.live_only) acq.tbd = st.tbd(spec, acq.lines, nullptr);
    return acq;
}

template <typename S>
Acquisition acquire_joint_impl(std::span<const S> fe_in, double t0, double fs, const FrameMap& specs,
                               const AcquireOptions& opt, const AcquireSeams& seams) {
    const Fe fe = prepare(fe_in, fs, opt.preprocessed, seams);
    const Stages st{fe.view, t0, fs, seams};
    const CarrierStats cs = st.stats();
    Acquisition acq;
    acq.lines = st.lines(cs);
    std::vector<const PreambleKind*> kinds;
    for (const std::int64_t w : {std::int64_t{config::WAVEFORM_CE}, std::int64_t{config::WAVEFORM_L}})
        if (find_spec(specs, w)) kinds.push_back(&preamble_kind(w));
    std::vector<PreambleHit> hits;
    for (const auto& c : acq.lines) {
        std::vector<PreambleHit> got = st.at_line(c, kinds);
        for (auto& h : got) hits.push_back(std::move(h));
    }
    if (find_spec(specs, config::WAVEFORM_L)) acq.pilots = st.pilots(cs, acq.lines);
    for (const auto& c : acq.pilots) hits.push_back(st.search(c, &l_pre()));
    const double thr = st.threshold(cs.M);
    for (const auto& [f, ws] : hint_items(opt.hints, specs)) {
        for (const std::int64_t w : ws) {
            bool done = false;
            std::optional<std::size_t> near;
            for (std::size_t i = 0; i < hits.size(); ++i) {
                if (hits[i].waveform != w) continue;
                if (hits[i].passed && std::abs(hits[i].f_hz - f) <= HINT_NEAR_HZ) done = true;
            }
            if (done) continue;
            for (std::size_t i = 0; i < hits.size(); ++i)
                if (hits[i].waveform == w && std::abs(hits[i].f_hz - f) <= HINT_NEAR_HZ) {
                    near = i;
                    break;
                }
            const types::CarrierCandidate line = st.hint(cs, f);
            PreambleHit h;
            if (near) {
                h = hits[*near];
            } else if (w == config::WAVEFORM_L && line.metric <= thr) {
                std::optional<PreambleHit> best;
                for (const double o : rxa::L_HINT_OFFSETS_HZ) {
                    types::CarrierCandidate c;
                    c.f_hz = f + o;
                    c.drift_hz_per_min = 0.0;
                    c.metric = line.metric;
                    PreambleHit t = st.search(c, &l_pre());
                    if (!best || p_then_lam_less(t, *best)) best = std::move(t);   // min(): the first least
                }
                h = std::move(*best);
            } else {
                h = st.search(line, &preamble_kind(w));
            }
            if (h.passed) {
                hits.push_back(std::move(h));
            } else {
                types::CarrierCandidate l = line;   // dataclasses.replace(line, waveform=w)
                l.waveform = w;
                acq.hinted.push_back(std::move(l));
            }
        }
    }
    std::vector<PreambleHit> per;
    for (const auto& [w, spec] : specs) {
        std::vector<PreambleHit> own;
        for (const auto& h : hits)
            if (h.passed && h.waveform == w) own.push_back(h);
        for (auto& h : drop_symbol_rate_aliases(own, preamble_kind(w).t_sym)) per.push_back(std::move(h));
    }
    acq.preamble = merge_hits(drop_band_owned(per));
    for (auto& h : acq.preamble) {
        const frame::FrameSpec* s = find_spec(specs, h.waveform);
        if (!s) throw std::out_of_range(std::to_string(h.waveform));      // specs[h.waveform]: KeyError
        h.lead_in_s = st.lead_in(h, s->t_sym());
    }
    if (!opt.live_only) {
        const frame::FrameSpec* longest = nullptr;
        for (const auto& [w, s] : specs)
            if (!longest || duration(s) > duration(*longest)) longest = &s;     // max(): the first greatest
        std::vector<frame::FrameSpec> shape;
        for (const auto& [w, s] : specs) shape.push_back(s);
        acq.tbd = st.tbd(*longest, acq.lines, &shape);
    }
    return acq;
}

}  // namespace

// --- frames and detections -------------------------------------------------------------------

FrameMap frame_map_of(std::span<const frame::FrameSpec> specs) {
    FrameMap out;
    for (const auto& [w, i] : frame_map(specs)) out.emplace_back(w, specs[i]);
    return out;
}

std::vector<types::Detection> Acquisition::detections(const frame::FrameSpec& spec) const {
    const double dur = duration(spec);
    std::vector<types::Detection> out;
    out.reserve(preamble.size() + hinted.size() + tbd.size());
    for (const auto& h : preamble) out.push_back(h.to_detection(dur));
    for (const auto& c : hinted)
        out.push_back(hint_detection(c, path_of(c.f_hz, c.drift_hz_per_min, dur), config::WAVEFORM_CE));
    for (const auto& h : tbd) out.push_back(h.to_detection());
    return out;
}

std::vector<types::Detection> Acquisition::detections(std::span<const frame::FrameSpec> specs) const {
    return joint_detections(*this, frame_map_of(specs));
}

std::vector<types::Detection> joint_detections(const Acquisition& acq, const FrameMap& specs) {
    std::vector<types::Detection> out;
    const std::vector<std::size_t> order = by_lam(acq.preamble, iota(acq.preamble.size()));
    for (const std::size_t i : order) {
        const PreambleHit& h = acq.preamble[i];
        if (const frame::FrameSpec* s = find_spec(specs, h.waveform)) out.push_back(h.to_detection(duration(*s)));
    }
    for (const auto& c : acq.hinted) {
        std::vector<std::int64_t> ws;
        if (c.waveform)
            ws.push_back(*c.waveform);
        else
            for (const auto& [w, s] : specs) ws.push_back(w);
        for (const std::int64_t w : ws)
            if (const frame::FrameSpec* s = find_spec(specs, w))
                out.push_back(hint_detection(c, path_of(c.f_hz, c.drift_hz_per_min, duration(*s)), w));
    }
    const bool ce = find_spec(specs, config::WAVEFORM_CE) != nullptr;
    const bool l = find_spec(specs, config::WAVEFORM_L) != nullptr;
    std::vector<types::Detection> first, copies, others;
    for (const auto& h : acq.tbd) {
        if (!h.waveform) {
            if (ce) {
                first.push_back(h.to_detection(config::WAVEFORM_CE));
                if (l) copies.push_back(h.to_detection(config::WAVEFORM_L));
            } else {
                first.push_back(h.to_detection(config::WAVEFORM_L));
            }
        } else if (find_spec(specs, *h.waveform)) {
            first.push_back(h.to_detection(*h.waveform));
        } else {
            for (const auto& [w, s] : specs) others.push_back(h.to_detection(w));
        }
    }
    std::vector<types::Detection> a3;
    for (auto* group : {&first, &copies, &others})
        for (auto& d : *group) {
            bool dup = false;
            for (const auto& g : a3)
                if (g.waveform == d.waveform && std::abs(g.f_hz - d.f_hz) < rxa::TBD_DEDUP_HZ) {
                    dup = true;
                    break;
                }
            if (!dup) a3.push_back(std::move(d));
        }
    for (auto& d : a3) out.push_back(std::move(d));
    return out;
}

// --- arbitration ------------------------------------------------------------------------------

std::vector<std::size_t> drop_symbol_rate_aliases_idx(std::span<const PreambleHit> hits, double t_sym) {
    std::vector<std::size_t> out;
    for (const std::size_t i : by_lam(hits, iota(hits.size()))) {
        const PreambleHit& h = hits[i];
        bool alias = false;
        for (const std::size_t j : out) {
            const PreambleHit& g = hits[j];
            for (const int k : rxa::A2_ALIAS_K) {
                if (std::abs(std::abs(h.f_hz - g.f_hz) - static_cast<double>(k) / t_sym) < rxa::A2_ALIAS_HZ &&
                    std::abs(h.tau0_s - g.tau0_s) < t_sym / 2 && g.lam > rxa::A2_ALIAS_RATIO * h.lam) {
                    alias = true;
                    break;
                }
            }
            if (alias) break;
        }
        if (!alias) out.push_back(i);
    }
    return out;
}

std::vector<PreambleHit> drop_symbol_rate_aliases(std::span<const PreambleHit> hits, double t_sym) {
    return pick(hits, drop_symbol_rate_aliases_idx(hits, t_sym));
}

std::vector<std::size_t> drop_band_owned_idx(std::span<const PreambleHit> hits) {
    constexpr double near_t = rxa::L_OWN_T * config::L_T_SYM;
    std::vector<std::size_t> out;
    for (const std::size_t i : by_lam(hits, iota(hits.size()))) {
        const PreambleHit& h = hits[i];
        bool owned = false;
        if (h.waveform == config::WAVEFORM_L) {
            for (const std::size_t j : out) {
                const PreambleHit& g = hits[j];
                if (g.waveform == config::WAVEFORM_L && g.f_hz - rxa::L_OWN_HZ[1] <= h.f_hz &&
                    h.f_hz <= g.f_hz - rxa::L_OWN_HZ[0] && std::abs(h.tau0_s - g.tau0_s) < near_t) {
                    owned = true;
                    break;
                }
            }
        }
        if (!owned) out.push_back(i);
    }
    return out;
}

std::vector<PreambleHit> drop_band_owned(std::span<const PreambleHit> hits) {
    return pick(hits, drop_band_owned_idx(hits));
}

std::vector<std::size_t> merge_hits_idx(std::span<const PreambleHit> hits) {
    std::vector<std::size_t> order = iota(hits.size());
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return p_then_lam_less(hits[a], hits[b]); });
    std::vector<std::size_t> out;
    for (const std::size_t i : order) {
        const PreambleHit& h = hits[i];
        bool same = false;
        for (const std::size_t j : out) {
            const PreambleHit& g = hits[j];
            if (std::abs(h.f_hz - g.f_hz) < rxa::WAVEFORM_MERGE_HZ &&
                std::abs(h.tau0_s - g.tau0_s) < config::L_T_SYM / 2) {
                same = true;
                break;
            }
        }
        if (!same) out.push_back(i);
    }
    return by_lam(hits, out);
}

std::vector<PreambleHit> merge_hits(std::span<const PreambleHit> hits) { return pick(hits, merge_hits_idx(hits)); }

double live_end_s() { return rxa::A2_TAU_S + config::N_PRE * config::T_SYM + rxa::A2_MARGIN_S_; }

// --- hints ------------------------------------------------------------------------------------

std::vector<std::pair<double, std::vector<std::int64_t>>> hint_items(std::span<const Hint> hints,
                                                                     std::span<const std::int64_t> waveforms) {
    std::vector<std::pair<double, std::vector<std::int64_t>>> out;
    out.reserve(hints.size());
    for (const Hint& h : hints) {
        std::vector<std::int64_t> ws;
        if (h.waveform) {
            if (std::find(waveforms.begin(), waveforms.end(), *h.waveform) != waveforms.end())
                ws.push_back(*h.waveform);
        } else {
            ws.assign(waveforms.begin(), waveforms.end());
        }
        out.emplace_back(h.f_hz, std::move(ws));
    }
    return out;
}

std::vector<std::pair<double, std::vector<std::int64_t>>> hint_items(std::span<const Hint> hints,
                                                                     const FrameMap& specs) {
    std::vector<std::int64_t> ws;
    ws.reserve(specs.size());
    for (const auto& [w, s] : specs) ws.push_back(w);
    return hint_items(hints, std::span<const std::int64_t>(ws));
}

// --- acquire ------------------------------------------------------------------------------------

Acquisition acquire(std::span<const cf> fe, double t0_index, double fs, const frame::FrameSpec& spec,
                    const AcquireOptions& opt, const AcquireSeams& seams) {
    return acquire_one(fe, t0_index, fs, spec, opt, seams);
}
Acquisition acquire(std::span<const cd> fe, double t0_index, double fs, const frame::FrameSpec& spec,
                    const AcquireOptions& opt, const AcquireSeams& seams) {
    return acquire_one(fe, t0_index, fs, spec, opt, seams);
}
Acquisition acquire(std::span<const cf> fe, double t0_index, double fs, std::span<const frame::FrameSpec> specs,
                    const AcquireOptions& opt, const AcquireSeams& seams) {
    return acquire_joint_impl(fe, t0_index, fs, frame_map_of(specs), opt, seams);
}
Acquisition acquire(std::span<const cd> fe, double t0_index, double fs, std::span<const frame::FrameSpec> specs,
                    const AcquireOptions& opt, const AcquireSeams& seams) {
    return acquire_joint_impl(fe, t0_index, fs, frame_map_of(specs), opt, seams);
}
Acquisition acquire(const frontend::Capture& cap, const frame::FrameSpec& spec, const AcquireOptions& opt,
                    const AcquireSeams& seams) {
    return acquire_one(std::span<const cf>(cap.fe), cap.t0_index, cap.fs, spec, opt, seams);
}
Acquisition acquire(const frontend::Capture& cap, std::span<const frame::FrameSpec> specs, const AcquireOptions& opt,
                    const AcquireSeams& seams) {
    return acquire_joint_impl(std::span<const cf>(cap.fe), cap.t0_index, cap.fs, frame_map_of(specs), opt, seams);
}
Acquisition acquire_joint(std::span<const cf> fe, double t0_index, double fs, const FrameMap& specs,
                          const AcquireOptions& opt, const AcquireSeams& seams) {
    return acquire_joint_impl(fe, t0_index, fs, specs, opt, seams);
}
Acquisition acquire_joint(std::span<const cd> fe, double t0_index, double fs, const FrameMap& specs,
                          const AcquireOptions& opt, const AcquireSeams& seams) {
    return acquire_joint_impl(fe, t0_index, fs, specs, opt, seams);
}

}  // namespace sstvae::qrss::acquire
