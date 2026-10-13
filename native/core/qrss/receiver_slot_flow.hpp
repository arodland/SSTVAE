// The slot half of sstvae/qrss/receiver.py as one orchestration over a
// policy: candidates, detect / _detect_joint, _gate, receive_slot /
// _receive_slot_joint and the rules they apply (_aliases, _ghosts,
// _owned, _a3_lines, _suspect_neighbour, _suspect_beside), written once
// and run over two kinds of object.
//
//   receiver_slot.cpp's typed policy: the C++ records (types::Detection,
//     types::PassResult, track::ChanCapture, ...) and the C++ stages, each
//     replaceable by a std::function (SlotSeams). receiver_slot.hpp's
//     functions and rx_api.hpp are this.
//   the binding's Python policy (bindings/module/qrss/receiver_slot.cpp):
//     every record is the Python object a stage returned and every stage
//     is the module attribute the reference calls, resolved at each call,
//     so a test's stub objects flow through the same code untouched.
//
// So what is decided here -- which candidates are gated, in which order,
// what each is handed, what is kept -- is one code for both, the
// reference's statement for statement: the budget (one clock reading
// before the candidates, one before each), `taken += 1` before the merge
// skip, max_candidates counting what is gated or merged but not what
// `_owned` skips, the noise exclusions as tuple(sorted(set)), the merges
// strongest first (Python's stable sort on -z_ref), the per-waveform
// alias and ghost rules at each waveform's symbol, and the joint
// receiver's twin frame for a waveform not selected.
//
// A policy P provides (Python's expression in brackets):
//
//   types   Det, Pass, Prep, Cap, AcqCap, Sel (a frame or a tuple), Spec
//           (one frame), Chan, Track, FrameMap, WKey, Others, Cands (a list
//           of Det or None), and the pass-through argument types LiveOnly,
//           Estimator, Erase, RoundB, MaxC, Budget, Hints, Skip
//   values  f_hz(d) [float(d.f_hz)], centre(d) [centre_hz(d)],
//           method_is(d, m) [d.method == m], a3_z(d)
//           [float(getattr(d, "a3_z", nan))], z_ref(d), tau0(d)
//           [d.timing.tau0], lead_in(d) [d.lead_in_s], set_lead_in(d, x)
//           [d.lead_in_s = x, x as lead_in_span returned it],
//           is_l(d) [getattr(d, "waveform", CE) == L], p_f_hz, p_tau0,
//           p_z_ref [p.report.z_ref], p_suspect [bool(p.report.suspect)],
//           p_centre [centre_hz(p)], same_waveform(g, p)
//           [g.waveform == p.waveform]
//   frames  is_joint(sel), frames(sel), single(sel), frame_map(specs),
//           smap_get(smap, d) [smap.get(getattr(d, "waveform", CE))],
//           smap_items(smap), waveform_eq(d, w) [d.waveform == w],
//           spec_for(smap, specs, d) [smap[w] if w in smap else
//           lframe.twin(specs[0], w)], t_sym(spec), spec_is_l(spec)
//   args    ce_hints(hints) [_ce_hints], floats(skip) [[float(f) for f in
//           skip]], centres(skip) [_centres(skip)], deadline(budget)
//           [None or time.monotonic() + budget], reached(maxc, taken)
//           [maxc is not None and taken >= maxc], others_f(dets, i) /
//           others_c(dets, i) [tuple(g.f_hz / centre_hz(g) for g in dets
//           if g is not d)], now() [time.monotonic()], old_gating()
//           [rules.old(rules.GATING)], given(c) [c is not None], listed(c)
//           [the list]
//   stages  capture_of(prep) [frontend.Capture(prep.fe, ...)],
//           acquire_dets(cap, spec_or_specs, live_only, hints)
//           [acquire.acquire(...).detections(...)], candidates, detect,
//           channel_for, verify, verify_l, verify_track (fit=False),
//           ce_resid_ratio, lead_in_span(prep, v, t_sym or nullopt), gate,
//           prepare, receive_pass, and the two the reference reaches through
//           its module too: detect_joint(prep, specs, DetectArgs)
//           [_detect_joint], receive_slot_joint(prep, specs, SlotArgs)
//           [_receive_slot_joint]
//
// Header-only because the binding instantiates it; nothing here holds
// state of its own.

#ifndef SSTVAE_QRSS_RECEIVER_SLOT_FLOW_HPP
#define SSTVAE_QRSS_RECEIVER_SLOT_FLOW_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "qrss/config.hpp"
#include "qrss/rx_config.hpp"

namespace sstvae::qrss::receiver::flow {

namespace rr = rxconfig::receiver;

// (line, centre, Z) of a whole-slot (A3) candidate: what `_owned` weighs a
// line against.
struct A3Line {
    double line = 0.0;
    double centre = 0.0;
    double z = 0.0;
};

// --- arithmetic shared by both policies ----------------------------------------------------

// sorted(range(n), key=lambda i: -z[i]): Python's sort is stable, as
// std::stable_sort is. (A NaN key is ordered as std::stable_sort leaves it,
// not as Python's timsort does: no stage makes one.)
inline std::vector<std::size_t> by_z_desc(std::span<const double> z) {
    std::vector<std::size_t> o(z.size());
    for (std::size_t i = 0; i < o.size(); ++i) o[i] = i;
    std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return -z[a] < -z[b]; });
    return o;
}

// tuple(sorted({f for f in pool if abs(f - x0) >= MERGE_HZ})). A NaN
// fails the test, so the set holds none and the sort is total.
inline std::vector<double> others_of(std::span<const double> pool, double x0) {
    std::vector<double> out;
    for (const double f : pool)
        if (std::abs(f - x0) >= rr::MERGE_HZ) out.push_back(f);
    // stable, so of equal values (0.0 and -0.0) the first one put in stays, as in a set
    std::stable_sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end(), [](double a, double b) { return a == b; }), out.end());
    return out;
}

// _ghost_of(f_hz, tau0, g_f_hz, g_tau0, t_sym)
inline bool ghost_of(double f_hz, double tau0, double g_f_hz, double g_tau0, double t_sym = config::T_SYM) {
    return std::abs(f_hz - g_f_hz) < rr::GHOST_HZ &&
           std::abs(tau0 - g_tau0) < rr::GHOST_T * t_sym * static_cast<double>(config::CH_FS);
}

// The k = 1, 2 test of _aliases for d against a kept g, its values read as
// the reference's `and` reaches them, left operand first.
template <class FD, class FG, class TD, class TG, class ZD, class ZG>
bool alias_of(FD fd, FG fg, TD td, TG tg, ZD zd, ZG zg, double t_sym) {
    for (int k = 1; k <= 2; ++k) {
        const double f_d = fd(), f_g = fg();
        if (!(std::abs(std::abs(f_d - f_g) - static_cast<double>(k) / t_sym) < rr::ALIAS_HZ)) continue;
        const double t_d = td(), t_g = tg();
        if (!(std::abs(t_d - t_g) < static_cast<double>(config::CH_FS) * t_sym * (k + 0.5))) continue;
        const double z_g = zg(), z_d = zd();
        if (z_g > rr::ALIAS_RATIO * z_d) return true;
    }
    return false;
}

// --- the rules --------------------------------------------------------------------------------

// _aliases(dets, t_sym)
template <class P>
std::vector<typename P::Det> aliases(P& p, const std::vector<typename P::Det>& dets, double t_sym) {
    using Det = typename P::Det;
    std::vector<double> key(dets.size());
    for (std::size_t i = 0; i < dets.size(); ++i) key[i] = p.z_ref(dets[i]);
    std::vector<Det> out;
    for (const std::size_t i : by_z_desc(key)) {
        const Det& d = dets[i];
        bool drop = false;
        for (const Det& g : out) {
            if (alias_of([&] { return p.f_hz(d); }, [&] { return p.f_hz(g); }, [&] { return p.tau0(d); },
                         [&] { return p.tau0(g); }, [&] { return p.z_ref(d); }, [&] { return p.z_ref(g); }, t_sym)) {
                drop = true;
                break;
            }
        }
        if (!drop) out.push_back(d);
    }
    return out;
}

// _ghosts(dets, t_sym)
template <class P>
std::vector<typename P::Det> ghosts(P& p, const std::vector<typename P::Det>& dets, double t_sym) {
    using Det = typename P::Det;
    std::vector<double> key(dets.size());
    for (std::size_t i = 0; i < dets.size(); ++i) key[i] = p.z_ref(dets[i]);
    std::vector<Det> out;
    for (const std::size_t i : by_z_desc(key)) {
        const Det& d = dets[i];
        bool ghost = false;
        for (const Det& g : out) {
            // the call's arguments, in order
            const double f = p.f_hz(d), t = p.tau0(d), gf = p.f_hz(g), gt = p.tau0(g);
            if (ghost_of(f, t, gf, gt, t_sym)) {
                ghost = true;
                break;
            }
        }
        if (!ghost) out.push_back(d);
    }
    return out;
}

// _a3_lines(cands)
template <class P>
std::vector<A3Line> a3_lines(P& p, const std::vector<typename P::Det>& cands) {
    std::vector<A3Line> out;
    for (const auto& c : cands) {
        if (!p.method_is(c, "tbd") || !std::isfinite(p.a3_z(c))) continue;
        A3Line l;
        l.line = p.f_hz(c);
        l.centre = p.centre(c);
        l.z = p.a3_z(c);
        out.push_back(l);
    }
    return out;
}

// _owned(d, found, a3), `lines()` giving a3's lines: called once for each
// preamble-found signal d falls inside, as the reference's comprehension
// walks a3 once for each (and only for those)
template <class P, class Lines>
bool owned_by(P& p, const typename P::Det& d, const std::vector<typename P::Det>& found, Lines&& lines) {
    if (!p.method_is(d, "tbd") || p.old_gating()) return false;
    const double z = p.a3_z(d);
    if (!std::isfinite(z)) return false;
    const double f = p.f_hz(d);
    for (const auto& g : found) {
        if (!p.method_is(g, "preamble")) continue;
        bool inside;
        if (p.is_l(g))
            inside = p.f_hz(g) - rr::L_OWN_BELOW_HZ <= f && f <= p.f_hz(g) + rr::L_OWN_ABOVE_HZ;
        else
            inside = std::abs(f - p.f_hz(g)) < rr::GHOST_HZ;
        if (!inside) continue;
        const double c = p.centre(g);
        bool any = false;
        double top = 0.0;   // max(own): the first, replaced by anything larger
        const auto& a3 = lines();
        for (const A3Line& l : a3) {
            if (std::abs(l.line - p.f_hz(g)) < rr::MERGE_HZ || std::abs(l.centre - c) < rr::MERGE_HZ) {
                if (!any || l.z > top) top = l.z;
                any = true;
            }
        }
        if (any && z <= top) return true;
    }
    return false;
}

// _owned(d, found, a3)
template <class P>
bool owned(P& p, const typename P::Det& d, const std::vector<typename P::Det>& found, std::span<const A3Line> a3) {
    return owned_by(p, d, found, [a3]() { return a3; });
}

// _suspect_neighbour(p, out)
template <class P>
bool suspect_neighbour(P& p, const typename P::Pass& x, const std::vector<typename P::Pass>& out) {
    if (!p.p_suspect(x)) return false;
    for (const auto& g : out)
        if (std::abs(p.p_f_hz(x) - p.p_f_hz(g)) < rr::GHOST_HZ && !p.p_suspect(g) &&
            p.p_z_ref(x) < rr::GHOST_Z_RATIO * p.p_z_ref(g))
            return true;
    return false;
}

// _suspect_beside(p, out)
template <class P>
bool suspect_beside(P& p, const typename P::Pass& x, const std::vector<typename P::Pass>& out) {
    const double c = p.p_centre(x);
    if (!p.p_suspect(x)) return false;
    for (const auto& g : out)
        if (std::abs(c - p.p_centre(g)) < rr::GHOST_HZ && !p.p_suspect(g) &&
            p.p_z_ref(x) < rr::GHOST_Z_RATIO * p.p_z_ref(g))
            return true;
    return false;
}

// --- acquisition and gate V ---------------------------------------------------------------

// candidates(prep, spec, live_only, hints)
template <class P>
std::vector<typename P::Det> candidates(P& p, const typename P::Prep& prep, const typename P::Sel& sel,
                                        const typename P::LiveOnly& live_only, const typename P::Hints& hints) {
    const typename P::AcqCap cap = p.capture_of(prep);
    if (p.is_joint(sel)) {
        const typename P::Sel specs = p.frames(sel);
        return p.acquire_dets(cap, specs, live_only, hints);
    }
    const typename P::Spec spec = p.single(sel);
    return p.acquire_dets(cap, spec, live_only, p.ce_hints(hints));
}

// detect's arguments past (prep, spec)
template <class P>
struct DetectArgs {
    typename P::LiveOnly live_only{};
    typename P::MaxC max_candidates{};
    typename P::Budget budget_s{};
    typename P::Erase erase_s{};
    typename P::Cands cands{};   // None: candidates() are found
    typename P::Skip skip_hz{};
    typename P::Hints hints{};
};

// _gate(prep, spec, d, others, erase_s)
template <class P>
std::optional<typename P::Det> gate(P& p, const typename P::Prep& prep, const typename P::Spec& spec,
                                    const typename P::Det& d, const std::vector<double>& others,
                                    const typename P::Erase& erase_s) {
    const typename P::Chan chan = p.channel_for(prep, spec, d, others);
    std::optional<typename P::Det> v;
    if (p.spec_is_l(spec)) {
        v = p.verify_l(chan, spec, d, erase_s);
    } else {
        v = p.verify(chan, spec, d, erase_s);
        if (v) {
            const std::optional<typename P::Track> tr = p.verify_track(chan, spec, *v, erase_s);
            const double R = tr ? p.ce_resid_ratio(*tr) : 1.0;
            if (R > rr::CE_RESID_MIN && p.z_ref(*v) / std::sqrt(R) <= config::Z_ACCEPT) v.reset();
        }
    }
    if (v && !p.method_is(*v, "preamble") && p.lead_in(*v) == 0.0)
        p.set_lead_in(*v, p.lead_in_span(prep, *v, std::optional<double>(p.t_sym(spec))));
    return v;
}

// detect(prep, spec, live_only, max_candidates, budget_s, erase_s, cands, skip_hz, hints)
template <class P>
std::vector<typename P::Det> detect(P& p, const typename P::Prep& prep, const typename P::Sel& sel,
                                    DetectArgs<P> a) {
    using Det = typename P::Det;
    if (p.is_joint(sel)) return p.detect_joint(prep, p.frames(sel), a);   // (receiver._detect_joint)
    const typename P::Spec spec = p.single(sel);
    a.hints = p.ce_hints(a.hints);
    const std::optional<double> t_end = p.deadline(a.budget_s);
    const std::vector<Det> cands = p.given(a.cands) ? p.listed(a.cands) : p.candidates(prep, spec, a.live_only, a.hints);
    const std::vector<A3Line> a3 = a3_lines(p, cands);
    std::vector<Det> found;
    std::vector<double> hits;
    for (const Det& c : cands)
        if (p.method_is(c, "preamble")) hits.push_back(p.f_hz(c));
    const std::vector<double> skip = p.floats(a.skip_hz);
    hits.insert(hits.end(), skip.begin(), skip.end());
    std::int64_t taken = 0;
    for (const Det& d : cands) {
        if (t_end && p.now() > *t_end) break;
        if (owned(p, d, found, a3)) continue;   // (not gated, so not counted)
        if (p.reached(a.max_candidates, taken)) break;
        taken += 1;
        std::vector<double> known;
        for (const Det& g : found) known.push_back(p.f_hz(g));
        bool merged = false;
        for (const double f : known)
            if (std::abs(p.f_hz(d) - f) < rr::MERGE_HZ) merged = true;
        for (const double f : skip)
            if (!merged && std::abs(p.f_hz(d) - f) < rr::MERGE_HZ) merged = true;
        if (merged) continue;
        std::vector<double> pool = hits;
        pool.insert(pool.end(), known.begin(), known.end());
        const std::vector<double> others = others_of(pool, p.f_hz(d));
        const typename P::Chan chan = p.channel_for(prep, spec, d, others);
        std::optional<Det> v = p.verify(chan, spec, d, a.erase_s);
        if (!v) continue;
        if (!p.method_is(*v, "preamble") && p.lead_in(*v) == 0.0)
            p.set_lead_in(*v, p.lead_in_span(prep, *v, std::nullopt));
        found.push_back(*v);
    }
    std::vector<double> key(found.size());
    for (std::size_t i = 0; i < found.size(); ++i) key[i] = p.z_ref(found[i]);
    std::vector<Det> merged;
    for (const std::size_t i : by_z_desc(key)) {
        bool all = true;
        for (const Det& g : merged) {
            const double a = p.f_hz(found[i]), b = p.f_hz(g);
            if (!(std::abs(a - b) >= rr::MERGE_HZ)) {
                all = false;
                break;
            }
        }
        if (all) merged.push_back(found[i]);
    }
    return ghosts(p, aliases(p, merged, config::T_SYM), config::T_SYM);
}

// _detect_joint(prep, specs, live_only, max_candidates, budget_s, erase_s, cands, skip_hz, hints)
template <class P>
std::vector<typename P::Det> detect_joint(P& p, const typename P::Prep& prep, const typename P::Sel& specs,
                                          const DetectArgs<P>& a) {
    using Det = typename P::Det;
    const typename P::FrameMap smap = p.frame_map(specs);
    const std::optional<double> t_end = p.deadline(a.budget_s);
    const std::vector<Det> cands =
        p.given(a.cands) ? p.listed(a.cands) : p.candidates(prep, specs, a.live_only, a.hints);
    const std::vector<A3Line> a3 = a3_lines(p, cands);
    const std::vector<double> skip = p.centres(a.skip_hz);
    std::vector<double> hits;
    for (const Det& c : cands)
        if (p.method_is(c, "preamble")) hits.push_back(p.centre(c));
    hits.insert(hits.end(), skip.begin(), skip.end());
    std::vector<Det> found;
    std::int64_t taken = 0;
    for (const Det& d : cands) {
        if (t_end && p.now() > *t_end) break;
        if (owned(p, d, found, a3)) continue;   // (not gated, so not counted)
        if (p.reached(a.max_candidates, taken)) break;
        taken += 1;
        const std::optional<typename P::Spec> sp = p.smap_get(smap, d);
        if (!sp) continue;   // (a waveform not listened for)
        const double c0 = p.centre(d);
        // the reference's two comprehensions each call centre_hz(g)
        std::vector<double> known;
        for (const Det& g : found) known.push_back(p.centre(g));
        bool merged = false;
        for (const double f : known)
            if (std::abs(c0 - f) < rr::MERGE_HZ) merged = true;
        for (const double f : skip)
            if (std::abs(c0 - f) < rr::MERGE_HZ) merged = true;
        if (merged) continue;
        std::vector<double> pool = hits;
        for (const Det& g : found) pool.push_back(p.centre(g));
        const std::vector<double> others = others_of(pool, c0);
        std::optional<Det> v = p.gate(prep, *sp, d, others, a.erase_s);
        if (v) found.push_back(*v);
    }
    std::vector<double> key(found.size());
    for (std::size_t i = 0; i < found.size(); ++i) key[i] = p.z_ref(found[i]);
    std::vector<Det> merged;
    for (const std::size_t i : by_z_desc(key)) {
        bool all = true;
        for (const Det& g : merged) {
            const double a = p.centre(found[i]);
            const double b = p.centre(g);
            if (!(std::abs(a - b) >= rr::MERGE_HZ)) {
                all = false;
                break;
            }
        }
        if (all) merged.push_back(found[i]);
    }
    std::vector<Det> out;
    for (const auto& [w, sp] : p.smap_items(smap)) {
        std::vector<Det> mine;
        for (const Det& d : merged)
            if (p.waveform_eq(d, w)) mine.push_back(d);
        const double ts = p.t_sym(sp);
        const std::vector<Det> kept = ghosts(p, aliases(p, mine, ts), ts);
        out.insert(out.end(), kept.begin(), kept.end());
    }
    std::vector<double> zk(out.size());
    for (std::size_t i = 0; i < out.size(); ++i) zk[i] = p.z_ref(out[i]);
    std::vector<Det> sorted;
    for (const std::size_t i : by_z_desc(zk)) sorted.push_back(out[i]);
    return sorted;
}

// --- the slot ---------------------------------------------------------------------------------

// receive_slot's arguments past (cap, spec)
template <class P>
struct SlotArgs {
    typename P::LiveOnly live_only{};
    typename P::Estimator estimator{};
    typename P::Erase erase_s{};
    typename P::RoundB round_b{};
    typename P::Cands dets{};    // None: detect() finds them
    typename P::MaxC max_candidates{};
    typename P::Budget budget_s{};
    typename P::Hints hints{};
};

// _receive_slot_joint(prep, specs, live_only, estimator, *, erase_s, round_b, dets,
// max_candidates, budget_s, hints)
template <class P>
std::vector<typename P::Pass> receive_slot_joint(P& p, const typename P::Prep& prep, const typename P::Sel& specs,
                                                 const SlotArgs<P>& a) {
    using Det = typename P::Det;
    using Pass = typename P::Pass;
    const typename P::FrameMap smap = p.frame_map(specs);
    const std::vector<Det> dets = p.given(a.dets) ? p.listed(a.dets)
                                                  : p.detect(prep, specs, a.live_only, a.max_candidates,
                                                             a.budget_s, a.erase_s, a.hints);
    std::vector<Pass> out;
    for (std::size_t i = 0; i < dets.size(); ++i) {
        const Det& d = dets[i];
        const typename P::Spec sp = p.spec_for(smap, specs, d);
        const typename P::Others others = p.others_c(dets, i);
        const std::optional<Pass> x = p.receive_pass(prep, sp, d, a.estimator, others, a.round_b, a.erase_s);
        if (!x) continue;
        std::vector<Pass> same;
        for (const Pass& g : out)
            if (p.same_waveform(g, *x)) same.push_back(g);
        bool ghost = false;
        for (const Pass& g : same) {
            const double f = p.p_f_hz(*x), t = p.p_tau0(*x), gf = p.p_f_hz(g), gt = p.p_tau0(g);
            if (ghost_of(f, t, gf, gt, p.t_sym(sp))) {
                ghost = true;
                break;
            }
        }
        if (ghost) continue;
        if (suspect_beside(p, *x, out)) continue;
        const double c = p.p_centre(*x);
        std::optional<std::size_t> at;
        for (std::size_t j = 0; j < out.size(); ++j)
            if (std::abs(c - p.p_centre(out[j])) < rr::MERGE_HZ) {
                at = j;
                break;
            }
        if (at) {
            if (p.p_z_ref(*x) > p.p_z_ref(out[*at])) out[*at] = *x;
            continue;
        }
        out.push_back(*x);
    }
    return out;
}

// receive_slot(cap, spec, live_only, estimator, *, erase_s, round_b, dets, max_candidates,
// budget_s, hints)
template <class P>
std::vector<typename P::Pass> receive_slot(P& p, const typename P::Cap& cap, const typename P::Sel& sel,
                                           SlotArgs<P> a) {
    using Det = typename P::Det;
    using Pass = typename P::Pass;
    if (p.is_joint(sel)) {
        const typename P::Prep prep = p.prepare(cap);
        return p.receive_slot_joint(prep, p.frames(sel), a);   // (receiver._receive_slot_joint)
    }
    const typename P::Spec spec = p.single(sel);
    a.hints = p.ce_hints(a.hints);
    const typename P::Prep prep = p.prepare(cap);
    const std::vector<Det> dets = p.given(a.dets) ? p.listed(a.dets)
                                                  : p.detect(prep, spec, a.live_only, a.max_candidates,
                                                             a.budget_s, a.erase_s, a.hints);
    std::vector<Pass> out;
    for (std::size_t i = 0; i < dets.size(); ++i) {
        const Det& d = dets[i];
        const typename P::Others others = p.others_f(dets, i);
        const std::optional<Pass> x = p.receive_pass(prep, spec, d, a.estimator, others, a.round_b, a.erase_s);
        if (!x) continue;
        // tracking moves a pass's frequency: a ghost that got past detect()
        // can land on the stronger pass it came from
        bool ghost = false;
        for (const Pass& g : out) {
            const double f = p.p_f_hz(*x), t = p.p_tau0(*x), gf = p.p_f_hz(g), gt = p.p_tau0(g);
            if (ghost_of(f, t, gf, gt)) {
                ghost = true;
                break;
            }
        }
        if (ghost) continue;
        if (suspect_neighbour(p, *x, out)) continue;
        // two passes within MERGE_HZ are one signal: the stronger stands for it
        std::optional<std::size_t> at;
        for (std::size_t j = 0; j < out.size(); ++j)
            if (std::abs(p.p_f_hz(*x) - p.p_f_hz(out[j])) < rr::MERGE_HZ) {
                at = j;
                break;
            }
        if (at) {
            if (p.p_z_ref(*x) > p.p_z_ref(out[*at])) out[*at] = *x;
            continue;
        }
        out.push_back(*x);
    }
    return out;
}

}  // namespace sstvae::qrss::receiver::flow

#endif  // SSTVAE_QRSS_RECEIVER_SLOT_FLOW_HPP
