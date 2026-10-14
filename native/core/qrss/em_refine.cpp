// em.py's pass_path, rereceive and em_refine (see em_refine.hpp). The
// statements follow the reference in its order; comments name the Python
// they mirror.

#include "qrss/em_refine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "math/errors.hpp"
#include "qrss/ce.hpp"
#include "qrss/config.hpp"
#include "qrss/demod.hpp"
#include "qrss/frame.hpp"
#include "qrss/precoder.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/track.hpp"
#include "qrss/track_timing.hpp"

namespace sstvae::qrss::em {

namespace {

using types::Detection;
using types::EmPrior;
using types::FreqPath;
using types::PassResult;

// sequences.scrambler's refusal: q keys the scrambler as uint64_be.
std::uint64_t scrambler_q(std::int64_t q) {
    if (q < 0) throw std::invalid_argument("q must fit in uint64, not " + std::to_string(q));
    return static_cast<std::uint64_t>(q);
}

// p's own channel: receiver.channel_for(capture_from_pass(p), spec, det,
// f_mix_hz=p.f_mix_hz).
std::shared_ptr<const track::ChanCapture> own_channel(const PassResult& p, const frame::FrameSpec& spec,
                                                      const Detection& det) {
    const receiver::Prepared prep = receiver::capture_from_pass(p);
    return std::make_shared<const track::ChanCapture>(receiver::channel_for(prep, spec, det, {}, p.f_mix_hz));
}

// track.track(chan, spec, det, classes, cw_known=cw_known, timing=..., freq=..., outer=...)
// as receive_pass's round B hands cw_known on: the templates take the
// per-window keying, the result keeps the (192,) units.
track::TrackResult run_track(const std::shared_ptr<const track::ChanCapture>& chan, const frame::FrameSpec& spec,
                             const Detection& det, const track::Classes& classes,
                             const std::optional<receiver::CwKnown>& cw_known, const types::Timing& timing,
                             const std::optional<track::PathFn>& freq, int outer) {
    track::TrackArgs t;
    t.classes = classes;
    if (cw_known) {
        t.keying = ce::window_keying(std::span<const std::uint8_t>(cw_known->units), spec);
        t.ook = cw_known->ook;
    }
    t.timing = timing;
    t.freq = freq;
    t.outer = outer;
    track::TrackResult tr = track::track(chan, spec, det, t);
    if (cw_known) tr.keying = cw_known->units;
    return tr;
}

}  // namespace

// --- pass_path ------------------------------------------------------------------------------

FreqPath pass_path(const PassResult& p, std::shared_ptr<const track::ChanCapture> chan, PassPathTrace* trace) {
    const frame::FrameSpec spec = pass_spec(p);
    ce_only(spec, "em.pass_path");
    const Detection det = receiver::detection_from_pass(p);
    if (!chan) chan = own_channel(p, spec, det);
    const FreqPath line = track::line_path(*chan, spec, flat(p.f_hz)).path;
    FreqPath best = line;
    double best_z = -std::numeric_limits<double>::infinity();
    const FreqPath refined = track::refine_path(*chan, spec, line).path;
    double zs[2] = {0.0, 0.0};
    int choice = 0;
    const FreqPath* cands[2] = {&line, &refined};
    for (int k = 0; k < 2; ++k) {
        Detection d = det;   // dataclasses.replace(det, path=pth)
        d.path = *cands[k];
        track::VerifyArgs v;
        v.fit = false;
        const track::TrackResult tr = track::verify_track(chan, spec, d, v);
        zs[k] = tr.z_ref;
        if (tr.z_ref > best_z) {   // the first of equals stays, as Python's >
            best = *cands[k];
            best_z = tr.z_ref;
            choice = k;
        }
    }
    if (trace) {
        trace->line = line;
        trace->refined = refined;
        trace->z_line = zs[0];
        trace->z_refined = zs[1];
        trace->choice = choice;
    }
    return best;
}

// --- rereceive -------------------------------------------------------------------------------

Rereceived rereceive(const PassResult& p, const std::optional<EmPrior>& prior, const RereceiveArgs& args,
                     const Hooks& hooks) {
    const frame::FrameSpec spec = pass_spec(p);
    if (spec.waveform() != frame::Waveform::CE) return {};
    std::optional<header::HeaderFields> hdr = args.hdr;
    if (hdr && p.header && hdr->version != p.header->version) hdr->version = p.header->version;   // what p carried
    // est = estimator or p.estimator
    const std::string est = (args.estimator && !args.estimator->empty()) ? *args.estimator : p.estimator;
    if (p.ch.empty()) return {};
    const receiver::Prepared prep = receiver::capture_from_pass(p);
    Detection det = receiver::detection_from_pass(p);
    const std::shared_ptr<const track::ChanCapture> chan =
        std::make_shared<const track::ChanCapture>(receiver::channel_for(prep, spec, det, {}, p.f_mix_hz));
    if (!args.path) check_cancel(hooks);   // pass_path is a second or so of a FULL pass's work
    det.path = args.path ? *args.path : pass_path(p, chan);
    const std::int64_t rnd = args.em_round ? *args.em_round : p.em_round;
    check_cancel(hooks);

    if (!hdr || !spec.has_header()) {
        receiver::PassArgs a;
        a.prior = prior;
        a.estimator = est;
        a.f_mix_hz = p.f_mix_hz;
        a.return_tracks = true;
        // receive_pass's own channel_for gives this very channel (the
        // capture at the CH rate, mixed at p.f_mix_hz): hand it over, with
        // the noise table pass_path filled in
        receiver::ReceiverSeams s;
        s.channel_for = [&chan](const receiver::Prepared&, const frame::FrameSpec&, const Detection&,
                                std::span<const double>, const std::optional<Rational>&) { return chan; };
        receiver::PassOutcome o = receiver::receive_pass(&prep, spec, det, a, s);
        // new, trs = receive_pass(..., return_tracks=True): None does not unpack
        if (!o.pass) throw math::TypeError("cannot unpack non-iterable NoneType object");
        PassResult out = std::move(*o.pass);
        out.uid = p.uid;
        if (p.header) out.header = p.header;   // hdr_out = p.header if p.header is not None else new.header
        out.cw = p.cw;
        out.em_round = rnd;
        Rereceived r;
        r.path = path_of(*o.tracks.back());
        r.pass = std::move(out);
        return r;
    }

    const std::vector<std::uint8_t> bits = header::encode(*hdr);
    const frame::HdrBits hb = frame::HdrBits::of(bits);
    const track::Classes classes = track::make_classes(spec, &hb, prior ? &*prior : nullptr);
    const std::uint64_t q = scrambler_q(p.q);
    // as receive_pass's two rounds: the second starts from the first's
    // timing and re-centred frequency path
    std::optional<track::TrackResult> tr;
    tr.emplace(run_track(chan, spec, det, classes, args.cw_known, p.timing, std::nullopt, 2));
    std::vector<float> z, w, llr;
    double kappa = 1.0;
    bool suspect = false;
    if (!prior || !args.split) {
        check_cancel(hooks);
        track::TrackResult tr2 = run_track(chan, spec, det, classes, args.cw_known, tr->timing, tr->freq, 1);
        tr.emplace(std::move(tr2));
        tr->z_ref = p.report.z_ref;
        demod::Extraction ex = demod::extract(*tr, spec, q, est);
        z = std::move(ex.z);
        w = std::move(ex.w);
        llr = std::move(ex.llr);
        kappa = ex.kappa;
        suspect = ex.suspect;
    } else {
        // Block split: the symbols of a precoder block are demodulated by a
        // track whose templates hold no reference for that block (only the
        // other parity's blocks), so the reference's errors on a latent
        // cannot steer the gain and phase pass i is read with there.
        const auto n = static_cast<std::size_t>(spec.n_data());
        std::vector<std::int64_t> par = precoder::block_index(n);
        for (std::int64_t& b : par) b %= 2;   // block_index(n_data) % 2 (non-negative)
        z.assign(n, 0.0f);
        w.assign(n, 0.0f);
        const track::TrackResult first = std::move(*tr);
        tr.reset();
        for (std::int64_t k = 0; k < 2; ++k) {
            // pk = EmPrior(x_hat=np.where(own, 0.0, prior.x_hat), v=np.where(own, 1.0, prior.v))
            EmPrior pk;
            pk.x_hat.resize(prior->x_hat.size());
            pk.v.resize(prior->v.size());
            if (prior->x_hat.size() != n || prior->v.size() != n)
                throw std::invalid_argument("operands could not be broadcast together with shapes (" +
                                            std::to_string(n) + ",) () (" + std::to_string(prior->x_hat.size()) +
                                            ",) ");
            for (std::size_t i = 0; i < n; ++i) {
                const bool own = par[i] == k;
                pk.x_hat[i] = own ? 0.0 : prior->x_hat[i];
                pk.v[i] = own ? 1.0 : prior->v[i];
            }
            const track::Classes ck = track::make_classes(spec, &hb, &pk);
            check_cancel(hooks);
            track::TrackResult trk = run_track(chan, spec, det, ck, args.cw_known, first.timing, first.freq, 1);
            trk.z_ref = p.report.z_ref;
            demod::Extraction ex = demod::extract(trk, spec, q, est);
            for (std::size_t i = 0; i < n; ++i)
                if (par[i] == k) {
                    z[i] = ex.z[i];
                    w[i] = ex.w[i];
                }
            if (k == 0) {
                tr.emplace(std::move(trk));
                llr = std::move(ex.llr);
                kappa = ex.kappa;
                suspect = ex.suspect;
            }
        }
    }
    PassResult out = p;
    out.report = track::report(*tr, kappa, suspect);
    out.f_hz = tr->freq(0.0);
    out.timing = tr->timing;
    out.z = std::move(z);
    out.w = std::move(w);
    // a known spread header has been removed before extraction, so its
    // LLRs are only in the pass's first reading
    if (!(spec.hdr_rho() > 0)) out.hdr_llr = std::move(llr);
    out.psi = psi_frame(*tr);
    out.estimator = est;
    out.em_round = rnd;
    Rereceived r;
    r.path = path_of(*tr);
    r.pass = std::move(out);
    return r;
}

// --- em_refine -------------------------------------------------------------------------------

namespace {

// pass p's eligibility for a re-receive (a capture, a CE frame), for the
// progress total: a pass that cannot be read or framed counts, as the
// round will meet its error there.
bool eligible(store::Store& st, const std::string& uid) {
    try {
        const PassResult p = st.load_pass(uid);
        if (p.ch.empty()) return false;
        return pass_spec(p).waveform() == frame::Waveform::CE;
    } catch (const std::exception&) {
        return true;
    }
}

}  // namespace

std::vector<double> em_refine(store::Store& st, const store::Key& key, const EmRefineArgs& args, const Hooks& hooks,
                              const EmSeams& seams) {
    store::Accumulator acc = st.accumulator(key);
    std::vector<double> history{acc_mean_w_db(acc)};
    std::map<std::int64_t, std::optional<header::HeaderFields>> hdrs;
    std::optional<receiver::CwKnown> cw_known;
    if (args.use_header) {
        for (std::int64_t g = 0; g < acc.mode + 1; ++g) hdrs[g] = known_header(st, acc, g);
        // h0 = next((h for h in hdrs.values() if h is not None), None)
        std::optional<std::string> call;
        for (const auto& [g, h] : hdrs) {
            (void)g;
            if (h) {
                std::string c = h->callsign;
                while (!c.empty() && c.back() == ' ') c.pop_back();   // rstrip(" ")
                call = std::move(c);
                break;
            }
        }
        cw_known = known_keying(st, acc, call);
    }
    // the progress total: rounds x the members a round re-receives, counted
    // again at each round's start (from the passes done and the rounds left)
    // and never lowered, so a member a listener attaches meanwhile raises it
    // rather than running `done` past it
    std::int64_t total = 0;
    std::int64_t done = 0;
    std::map<std::string, std::optional<FreqPath>> paths;
    for (std::int64_t r = 1; r < args.rounds + 1; ++r) {
        const std::vector<std::string> uids = acc.uids();   // list(acc.uids): the round's start
        if (hooks.progress) {
            std::int64_t n = 0;
            for (const std::string& uid : uids)
                if (eligible(st, uid)) ++n;
            total = std::max(total, done + (args.rounds - r + 1) * n);
        }
        for (const std::string& uid : uids) {
            check_cancel(hooks);
            const PassResult p = st.load_pass(uid);
            if (p.ch.empty()) continue;
            const frame::FrameSpec spec = pass_spec(p);
            if (spec.waveform() != frame::Waveform::CE) {
                if (r == 1) log_line(hooks, "pass " + uid + ": waveform L, not re-received (EM is CE-only)");
                continue;
            }
            const std::optional<pyjson::Value> m = acc.member(uid);
            const EmPrior prior = em_prior(acc, uid, st, p.q, spec);
            // hdrs.get(int(m["segment"]))
            if (!m) throw math::TypeError("'NoneType' object is not subscriptable");
            const std::int64_t seg = store::py_int(m->at("segment"));
            const auto hit = hdrs.find(seg);
            const std::optional<header::HeaderFields> hdr = hit == hdrs.end() ? std::nullopt : hit->second;
            const auto pit = paths.find(uid);
            const std::optional<FreqPath> path = pit == paths.end() ? std::nullopt : pit->second;
            Rereceived got;
            if (seams.rereceive) {
                got = seams.rereceive(p, prior, hdr, cw_known, args.estimator, r, path);
            } else {
                RereceiveArgs ra;
                ra.hdr = hdr;
                ra.cw_known = cw_known;
                ra.estimator = args.estimator;
                ra.em_round = r;
                ra.path = path;
                got = rereceive(p, prior, ra, hooks);
            }
            paths[uid] = std::move(got.path);
            if (!got.pass) continue;
            st.update_pass(*got.pass);
            acc = st.accumulator(key);
            log_line(hooks, "round " + std::to_string(r) + " pass " + uid + ": mean W " +
                                frame::py_format(receiver::mean_w_db(p), "%+.2f") + " -> " +
                                frame::py_format(receiver::mean_w_db(*got.pass), "%+.2f") + " dB");
            ++done;
            report_progress(hooks, done, total, "round " + std::to_string(r) + " pass " + uid);
        }
        history.push_back(acc_mean_w_db(acc));
        if (history[history.size() - 1] - history[history.size() - 2] < args.tol_db) {
            if (r < args.rounds && total > 0) report_progress(hooks, done, done, "converged after round " + std::to_string(r));
            break;
        }
    }
    return history;
}

}  // namespace sstvae::qrss::em
