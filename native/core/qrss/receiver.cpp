// The per-pass half of sstvae/qrss/receiver.py: see receiver.hpp.

#include "qrss/receiver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "math/npstats.hpp"
#include "qrss/ce.hpp"
#include "qrss/cwid.hpp"
#include "qrss/demod.hpp"
#include "qrss/ldemod.hpp"
#include "qrss/ltrack.hpp"
#include "qrss/ltrack_timing.hpp"
#include "qrss/lwave.hpp"
#include "qrss/morse.hpp"
#include "qrss/pynum.hpp"
#include "qrss/track.hpp"
#include "qrss/track_timing.hpp"

namespace sstvae::qrss::receiver {
namespace {

namespace rxr = rxconfig::receiver;
using frame::FrameSpec;
using frame::Waveform;
using types::Detection;
using types::PassResult;

bool is_l(const FrameSpec& spec) { return spec.waveform() == Waveform::L; }

// Python's round(x, 1) of a float: the correctly rounded "%.1f" (dtoa's
// mode 3, as printf's), read back.
double round1(double x) { return *pynum::py_float(frame::py_format(x, "%.1f")); }

// cwid._combine of window_soft / window_llr's float32 (n_win, 176)
cwid::Combined combined(const cwid::Units32& u) {
    const std::vector<double> v(u.data.begin(), u.data.end());
    return cwid::combine(v, u.rows, static_cast<std::size_t>(cwid::CALL_N), false);
}

std::vector<float> to_f32(const std::vector<double>& v) {
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>(v[i]);
    return out;
}

const std::vector<double>* opt_ptr(const std::optional<std::vector<double>>& v) { return v ? &*v : nullptr; }

}  // namespace

// --- the capture --------------------------------------------------------------------------

Prepared prepare(const frontend::Capture& cap) {
    if (cap.fs == config::CH_FS) return Prepared{cap.fe, std::nullopt, cap.q, cap.t0_index, config::CH_FS};
    const double fs = static_cast<double>(cap.fs);
    // frontend.blank(cap.fe, fs=cap.fs) and frontend.normalise(fe, keep,
    // fs=cap.fs): the front end's defaults (frontend.hpp's)
    frontend::Blanked b = frontend::blank(cap.fe, 5.0, 2.0, 1.0, 0.05, fs, 1.0);
    frontend::Normalised n = frontend::normalise(b.fe, std::span<const float>(b.keep), 100.0, fs, 10.0, 0.25);
    return Prepared{std::move(n.fe), std::move(b.keep), cap.q, cap.t0_index, cap.fs};
}

std::shared_ptr<const Prepared> prepare(CaptureRef cap) {
    if (const auto* p = std::get_if<const Prepared*>(&cap)) {
        if (!*p) throw std::invalid_argument("prepare: no capture");
        return std::shared_ptr<const Prepared>(std::shared_ptr<const Prepared>(), *p);   // not owned
    }
    const frontend::Capture* c = std::get<const frontend::Capture*>(cap);
    if (!c) throw std::invalid_argument("prepare: no capture");
    return std::make_shared<const Prepared>(prepare(*c));
}

// --- which frames, and where a signal is ----------------------------------------------------

std::vector<FrameSpec> frames(std::span<const FrameSpec> specs) {
    const acquire::FrameMap m = acquire::frame_map_of(specs);
    std::vector<FrameSpec> out;
    for (const std::int64_t w : {std::int64_t(config::WAVEFORM_CE), std::int64_t(config::WAVEFORM_L)})
        for (const auto& [k, s] : m)
            if (k == w) out.push_back(s);
    return out;
}

std::vector<FrameSpec> frames(const FrameSpec& spec) { return {spec}; }

bool is_joint(std::span<const FrameSpec> specs) {
    for (const FrameSpec& s : frames(specs))
        if (is_l(s)) return true;
    return false;
}

bool is_joint(const FrameSpec& spec) { return is_l(spec); }

FrameSpec single(std::span<const FrameSpec> specs) { return frames(specs).front(); }

double centre_hz(double f_hz, std::int64_t waveform) {
    if (waveform == config::WAVEFORM_L) return f_hz + L_EXCLUDE_OFFSET_HZ;
    return f_hz;
}

double centre_hz(const Detection& d) { return centre_hz(d.f_hz, d.waveform); }
double centre_hz(const PassResult& p) { return centre_hz(p.f_hz, p.waveform); }

std::vector<acquire::Hint> ce_hints(std::span<const acquire::Hint> hints) {
    const bool any_pair = std::any_of(hints.begin(), hints.end(), [](const acquire::Hint& h) { return h.waveform.has_value(); });
    if (!any_pair) return {hints.begin(), hints.end()};
    std::vector<acquire::Hint> out;
    for (const acquire::Hint& h : hints) {
        if (!h.waveform) {
            out.push_back(h);
        } else if (*h.waveform == config::WAVEFORM_CE) {
            out.push_back(acquire::Hint{h.f_hz, std::nullopt});
        }
    }
    return out;
}

// --- one pass ---------------------------------------------------------------------------------

track::ChanCapture channel_for(const Prepared& prep, const FrameSpec& spec, const Detection& det,
                               std::span<const double> exclude_hz, const std::optional<Rational>& f_mix_hz) {
    if (prep.fs == config::CH_FS) {
        track::ChanCapture c;
        c.ch = prep.fe;
        c.fs = config::CH_FS;
        c.t0_index = prep.t0_index;
        c.f_mix = f_mix_hz ? *f_mix_hz : frontend::quantise_mix(frontend::fe_hz(det.f_hz));
        c.keep = std::nullopt;
        c.exclude_hz.assign(exclude_hz.begin(), exclude_hz.end());
        return c;
    }
    const track::PathFn fn = track::freq_path_spline(det.path);
    const double dur = spec.keyed_end_pos() * spec.t_sym();
    const double slip = 260e-6 * dur;
    double lo = -(static_cast<double>(config::LEAD_IN_MAX_S) + config::SPAN * spec.t_sym()) - MARGIN_S - slip;
    if (det.timing) lo += det.timing->tau0 / config::CH_FS;
    const double hi = dur + MARGIN_S + slip + (det.timing ? det.timing->tau0 / config::CH_FS : 2.5);
    std::optional<std::span<const float>> keep;
    if (prep.keep) keep = std::span<const float>(*prep.keep);
    return track::make_chan(prep.fe, keep, prep.t0_index, prep.fs, fn(dur / 2), lo, hi, exclude_hz);
}

std::vector<float> psi_frame(const track::TrackResult& tr) {
    const std::int64_t n = tr.spec.n_pos();
    const std::int64_t lo = tr.p_lo();
    std::vector<float> out(static_cast<std::size_t>(n));
    for (std::int64_t p = 0; p < n; ++p)
        out[static_cast<std::size_t>(p)] = static_cast<float>(tr.psi[demod::np::index(p - lo, tr.psi.size())]);
    return out;
}

CwOutcome cw_result(const track::TrackResult& tr, const FrameSpec& spec, const std::optional<header::HeaderFields>& hdr,
                    const track::NoisePsdFn& noise_psd) {
    CwOutcome out;
    if (spec.n_win() == 0) return out;
    std::optional<std::string> call;
    if (hdr) {
        std::string c = hdr->callsign;
        c.erase(c.find_last_not_of(' ') + 1);   // rstrip(" ") (npos + 1 is 0: all spaces)
        try {
            morse::check_callsign(c);
            call = std::move(c);
        } catch (const std::invalid_argument&) {
        }
    }
    const cwid::UnitSource src = cwid::unit_source(tr, noise_psd);
    std::string kt = cwid::keying_type(src, spec, call ? std::optional<std::string_view>(*call) : std::nullopt);
    const bool ook = kt == "ook";
    const cwid::Combined soft = combined(cwid::window_soft(src, spec, ook));
    const std::optional<std::string> text = cwid::read(combined(cwid::window_llr(src, spec, ook)));
    types::CwIdResult r;
    r.text = text;
    r.keying = std::move(kt);
    if (call) {
        const double z = cwid::match(soft, *call);
        // a confident read of another call disagrees, whatever the match
        r.agrees = (text && *text == *call) || (!text && z > config::Z_ACCEPT);
        if (z > config::Z_ACCEPT) out.known = CwKnown{morse::keying_units(*call), ook};
        r.z_match = z;
    } else {
        r.z_match = (text && !text->empty()) ? cwid::match(soft, *text) : 0.0;
    }
    out.cw = std::move(r);
    return out;
}

HeaderDecode decode_header(std::span<const double> llr, const std::vector<double>* llr_spread,
                           const std::vector<double>* llr_block, const FrameSpec& spec) {
    HeaderDecode out;
    if (!spec.has_header()) return out;
    if (is_l(spec)) {
        out.header = decode_header_l(llr, llr_spread, llr_block, spec);
        return out;
    }
    const std::span<const int> both(config::FORMAT_VERSIONS);
    if (spec.hdr_rho() == 0.0) {
        out.header = header::decode(llr, 8, both);
        return out;
    }
    std::optional<header::HeaderFields> hdr = header::decode(llr);
    if (hdr || spec.n_hdr() == 0) {
        out.header = std::move(hdr);
        return out;
    }
    if (llr_spread) {
        hdr = header::decode(*llr_spread);
        if (hdr) {
            out.header = std::move(hdr);
            return out;
        }
    }
    if (llr_block) {
        hdr = header::decode(*llr_block, 8, both);
        if (hdr && hdr->version == 1) {
            out.header = std::move(hdr);
            out.block = true;
            out.plain = true;
            return out;
        }
    }
    out.header = std::move(hdr);
    return out;
}

std::optional<header::HeaderFields> decode_header_l(std::span<const double> llr, const std::vector<double>* llr_spread,
                                                    const std::vector<double>* llr_block, const FrameSpec& spec) {
    const std::span<const int> v2(rxr::L_HEADER_VERSIONS);
    std::optional<header::HeaderFields> hdr = header::decode(llr, 8, v2);
    if (hdr || spec.n_hdr() == 0 || spec.hdr_rho() == 0.0) return hdr;
    for (const std::vector<double>* part : {llr_spread, llr_block}) {
        if (part) {
            hdr = header::decode(*part, 8, v2);
            if (hdr) return hdr;
        }
    }
    return std::nullopt;
}

std::string waveform_name(std::int64_t wf) {
    if (wf == config::WAVEFORM_CE) return "CE";
    if (wf == config::WAVEFORM_L) return "L";
    return std::to_string(wf);
}

std::string waveform_refusal(std::string_view det_waveform_name, const FrameSpec& spec) {
    return "a waveform " + std::string(det_waveform_name) + " detection cannot be received as frame " +
           frame::py_repr(spec.name()) + " (waveform " + waveform_name(static_cast<std::int64_t>(spec.waveform())) +
           "): pass lframe.twin(spec, det.waveform)";
}

namespace {

// Rounds A and B on a channel, CE (track, demod) or L (ltrack, ldemod).
struct Rounds {
    const ReceiverSeams& s;
    const PassArgs& a;
    const types::EmPrior* prior;

    std::shared_ptr<track::TrackResult> track_a(const std::shared_ptr<const track::ChanCapture>& chan,
                                                const FrameSpec& spec, const Detection& det) const {
        if (is_l(spec)) {
            ltrack::TrackLArgs t;
            t.classes = ltrack::make_classes_l(spec, nullptr, prior);
            t.timing = det.timing;
            t.noise_psd = s.noise_psd;
            return std::make_shared<track::LTrackResult>(ltrack::track_l(chan, spec, det, t));
        }
        track::TrackArgs t;
        t.classes = track::make_classes(spec, nullptr, prior);
        t.timing = det.timing;
        t.noise_psd = s.noise_psd;
        return std::make_shared<track::TrackResult>(track::track(chan, spec, det, t));
    }

    std::shared_ptr<track::TrackResult> track_b(const std::shared_ptr<const track::ChanCapture>& chan,
                                                const FrameSpec& spec, const Detection& det,
                                                const header::HeaderFields& hdr, const std::optional<CwKnown>& known,
                                                const track::TrackResult& tr) const {
        const std::vector<std::uint8_t> bits = header::encode(hdr);
        if (is_l(spec)) {
            ltrack::TrackLArgs t;
            t.classes = ltrack::make_classes_l(spec, &bits, prior);
            if (known) {
                t.keying = lwave::Keying::units(known->units);
                t.ook = known->ook;
            }
            t.timing = tr.timing;
            t.freq = tr.freq;
            t.outer = 1;
            t.noise_psd = s.noise_psd;
            return std::make_shared<track::LTrackResult>(ltrack::track_l(chan, spec, det, t));
        }
        const frame::HdrBits hb = frame::HdrBits::of(bits);
        track::TrackArgs t;
        t.classes = track::make_classes(spec, &hb, prior);
        if (known) {
            // the tracker's own window_keying of the (192,) mask: every window alike
            t.keying = ce::window_keying(std::span<const std::uint8_t>(known->units), spec);
            t.ook = known->ook;
        }
        t.timing = tr.timing;
        t.freq = tr.freq;
        t.outer = 1;
        t.noise_psd = s.noise_psd;
        auto out = std::make_shared<track::TrackResult>(track::track(chan, spec, det, t));
        // the result keeps cw_known's own mask, as _keying_tuple hands it on:
        // the (192,) units, not the per-window expansion the templates used
        if (known) out->keying = known->units;
        return out;
    }

    demod::Extraction extract(const track::TrackResult& tr, const FrameSpec& spec, std::int64_t q) const {
        // sequences.scrambler's refusal: q keys the scrambler as uint64_be
        if (q < 0) throw std::invalid_argument("q must fit in uint64, not " + std::to_string(q));
        const auto uq = static_cast<std::uint64_t>(q);
        if (is_l(spec)) return ldemod::extract_l(tr, spec, uq, a.estimator);
        return demod::extract(tr, spec, uq, a.estimator);
    }

    types::TrackReport report(const track::TrackResult& tr, double kappa, bool suspect) const {
        if (is_l(tr.spec)) return ltrack::report_l(dynamic_cast<const track::LTrackResult&>(tr), kappa, suspect, s.noise_psd);
        return track::report(tr, kappa, suspect, s.noise_psd);
    }
};

}  // namespace

PassOutcome receive_pass(CaptureRef cap, const FrameSpec& spec_in, const Detection& det_in, const PassArgs& a,
                         const ReceiverSeams& s) {
    if (det_in.waveform != static_cast<std::int64_t>(spec_in.waveform()))
        throw std::invalid_argument(waveform_refusal(waveform_name(det_in.waveform), spec_in));
    const bool L = is_l(spec_in);
    PassOutcome out;
    const std::shared_ptr<const Prepared> prep = s.prepare ? s.prepare(cap) : prepare(cap);
    if (!prep) throw std::invalid_argument("receive_pass: prepare gave no capture");
    std::shared_ptr<const track::ChanCapture> chan =
        s.channel_for ? s.channel_for(*prep, spec_in, det_in, a.exclude_hz, a.f_mix_hz)
                      : std::make_shared<const track::ChanCapture>(
                            channel_for(*prep, spec_in, det_in, a.exclude_hz, a.f_mix_hz));
    if (!chan) throw std::invalid_argument("receive_pass: channel_for gave no channel");
    out.chan = chan;
    std::optional<Detection> verified;
    const Detection* det = &det_in;
    if (!std::isfinite(det_in.z_ref) || !det_in.timing) {
        if (L) {
            if (s.verify_l) {
                verified = s.verify_l(chan, spec_in, det_in, a.erase_s);
            } else {
                track::VerifyArgs v;
                v.erase_s = a.erase_s;
                v.noise_psd = s.noise_psd;
                verified = ltrack::verify_l(chan, spec_in, det_in, v);
            }
        } else {
            if (s.verify) {
                verified = s.verify(chan, spec_in, det_in, a.erase_s);
            } else {
                track::VerifyArgs v;
                v.erase_s = a.erase_s;
                v.noise_psd = s.noise_psd;
                verified = track::verify(chan, spec_in, det_in, v);
            }
        }
        if (!verified) return out;
        det = &*verified;
    }
    const std::int64_t q = prep->q;
    const Rounds r{s, a, a.prior ? &*a.prior : nullptr};
    // round A
    std::shared_ptr<track::TrackResult> tr = r.track_a(chan, spec_in, *det);
    tr->z_ref = det->z_ref;
    track::erase(*tr, a.erase_s);
    demod::Extraction ex = r.extract(*tr, spec_in, q);
    const std::vector<double> llr64(ex.llr.begin(), ex.llr.end());
    const HeaderDecode hd = decode_header(llr64, opt_ptr(ex.llr_spread), opt_ptr(ex.llr_block), spec_in);
    std::vector<float> hdr_llr = hd.block ? to_f32(*ex.llr_block) : std::move(ex.llr);
    const FrameSpec spec = hd.plain ? frame::plain(spec_in) : spec_in;
    out.plain = hd.plain;
    CwOutcome cw = cw_result(*tr, spec, hd.header, s.noise_psd);
    if (a.return_tracks) out.tracks.push_back(tr);
    // round B
    if (a.round_b && hd.header) {
        std::shared_ptr<track::TrackResult> tr2 = r.track_b(chan, spec, *det, *hd.header, cw.known, *tr);
        tr2->z_ref = det->z_ref;
        track::erase(*tr2, a.erase_s);
        ex = r.extract(*tr2, spec, q);
        tr = std::move(tr2);
        if (a.return_tracks) out.tracks.push_back(tr);
    }
    PassResult p;
    p.report = r.report(*tr, ex.kappa, ex.suspect);
    p.f_hz = tr->freq(0.0);
    p.uid = types::pass_uid(q, p.f_hz, ex.z);
    p.q = q;
    p.frame = spec.name();
    p.waveform = L ? config::WAVEFORM_L : config::WAVEFORM_CE;
    p.timing = tr->timing;
    p.z = std::move(ex.z);
    p.w = std::move(ex.w);
    p.hdr_llr = std::move(hdr_llr);
    p.header = hd.header;
    p.cw = std::move(cw.cw);
    p.ch = chan->ch;
    p.ch_fs = config::CH_FS;
    p.ch_t0_index = chan->t0_index;
    p.f_mix_hz = chan->f_mix;
    p.psi = psi_frame(*tr);
    p.estimator = a.estimator;
    out.pass = std::move(p);
    return out;
}

// --- a stored pass -----------------------------------------------------------------------------

Detection detection_from_pass(const PassResult& p, double lead_in_s) {
    Detection d;
    const types::TrackReport& r = p.report;
    d.path.t_s = {-10.0, 0.0, 1800.0};
    d.path.f_hz.resize(3);
    for (std::size_t i = 0; i < 3; ++i) d.path.f_hz[i] = r.offset_hz + r.drift_hz_per_min * d.path.t_s[i] / 60.0;
    d.path.weight = {1.0, 1.0, 1.0};
    d.f_hz = p.f_hz;
    d.timing = p.timing;
    d.z_ref = r.z_ref;
    d.method = "template";
    d.lead_in_s = lead_in_s;
    d.waveform = p.waveform;
    return d;
}

Prepared capture_from_pass(const PassResult& p) {
    return Prepared{p.ch, std::nullopt, p.q, p.ch_t0_index, p.ch_fs};
}

double mean_w_db(std::span<const double> w) {
    const double m = math::mean(w);
    const double v = (1e-30 > m) ? 1e-30 : m;   // Python's max(m, 1e-30): m unless 1e-30 is greater
    return 10 * std::log10(v);
}

double mean_w_db(const PassResult& p) {
    const std::vector<double> w(p.w.begin(), p.w.end());
    return mean_w_db(w);
}

std::vector<std::optional<double>> confidence_db(const FrameSpec& spec, std::span<const double> w,
                                                 const std::vector<double>* hdr_llr, const std::vector<double>* psi,
                                                 std::optional<double> elapsed_s, std::int64_t n) {
    const auto lay = frame::layout(spec);
    std::vector<std::int64_t> lat_at, bit_at;
    if (is_l(spec)) {
        for (const std::int64_t d : lay->data) lat_at.insert(lat_at.end(), {d, d});
        for (const std::int64_t h : lay->hdr_sym) bit_at.insert(bit_at.end(), {h, h});
    } else {
        lat_at = lay->data;
        bit_at = lay->hdr_bits;
    }
    const double t_sym = spec.t_sym();
    const double end = spec.keyed_end_pos() * t_sym;
    // np.linspace(0.0, end, n + 1)
    const std::int64_t num = n + 1;
    if (num < 0) throw std::invalid_argument("Number of samples, " + std::to_string(num) + ", must be non-negative.");
    std::vector<double> edges(static_cast<std::size_t>(num));
    {
        const std::int64_t div = num - 1;
        if (div > 0) {
            const double step = end / static_cast<double>(div);
            for (std::int64_t i = 0; i < num; ++i) {
                const double y = step == 0.0 ? static_cast<double>(i) / static_cast<double>(div) * end
                                             : static_cast<double>(i) * step;
                edges[static_cast<std::size_t>(i)] = y + 0.0;
            }
            edges.back() = end;
        } else {
            for (std::int64_t i = 0; i < num; ++i) edges[static_cast<std::size_t>(i)] = static_cast<double>(i) * end + 0.0;
        }
    }
    if (n < 0) throw std::invalid_argument("negative dimensions are not allowed");   // np.zeros(n)
    const auto nb = static_cast<std::size_t>(n);
    std::vector<double> total(nb, 0.0), count(nb, 0.0);
    const bool use_psi = psi && static_cast<std::int64_t>(psi->size()) == spec.n_pos();

    // place(idx): each symbol's bin, and whether it counts
    struct Placed {
        std::vector<std::int64_t> b;
        std::vector<std::uint8_t> ok;
    };
    auto place = [&](const std::vector<std::int64_t>& idx) {
        Placed out;
        out.b.resize(idx.size());
        out.ok.resize(idx.size());
        for (std::size_t i = 0; i < idx.size(); ++i) {
            const std::int64_t pos = lay->pos[demod::np::index(idx[i], lay->pos.size())];
            const double t = static_cast<double>(pos) * t_sym;
            bool ok = use_psi ? (*psi)[demod::np::index(pos, psi->size())] > 0 : true;
            if (elapsed_s) ok = ok && t <= *elapsed_s;
            // searchsorted(edges, t, side="right") - 1, clipped to [0, n - 1]
            const auto k = static_cast<std::int64_t>(std::upper_bound(edges.begin(), edges.end(), t) - edges.begin()) - 1;
            out.b[i] = std::min(std::max(k, std::int64_t(0)), n - 1);
            out.ok[i] = ok ? 1 : 0;
        }
        return out;
    };
    auto check_bins = [&](const Placed& pl) {
        for (std::size_t i = 0; i < pl.b.size(); ++i)
            if (pl.ok[i] && pl.b[i] < 0) throw std::invalid_argument("'list' argument must have no negative elements");
    };

    if (w.size() == lat_at.size()) {
        const Placed pl = place(lat_at);
        check_bins(pl);
        std::vector<double> sum(nb, 0.0), cnt(nb, 0.0);
        for (std::size_t i = 0; i < pl.b.size(); ++i)
            if (pl.ok[i]) {
                const auto k = static_cast<std::size_t>(pl.b[i]);
                sum[k] += w[i];
                cnt[k] += 1.0;
            }
        for (std::size_t k = 0; k < nb; ++k) {
            total[k] = total[k] + sum[k];
            count[k] = count[k] + cnt[k];
        }
    }
    if (hdr_llr && spec.n_hdr() != 0 && hdr_llr->size() == bit_at.size()) {
        const Placed pl = place(bit_at);
        check_bins(pl);
        std::vector<std::int64_t> k(nb, 0);
        std::vector<double> m(nb, 0.0);
        for (std::size_t i = 0; i < pl.b.size(); ++i)
            if (pl.ok[i]) {
                const auto j = static_cast<std::size_t>(pl.b[i]);
                const double x = (*hdr_llr)[i];
                k[j] += 1;
                m[j] += x * x;
            }
        for (std::size_t j = 0; j < nb; ++j) {
            double mj = m[j] / 4.0;
            mj = k[j] > 0 ? mj / static_cast<double>(std::max<std::int64_t>(k[j], 1)) : 0.0;
            const double snr = mj > 0 ? 2.0 * mj / (1.0 + std::sqrt(1.0 + 4.0 * mj)) : 0.0;
            total[j] = total[j] + snr * static_cast<double>(k[j]);
            count[j] = count[j] + static_cast<double>(k[j]);
        }
    }
    std::vector<std::optional<double>> out;
    out.reserve(nb);
    for (std::size_t j = 0; j < nb; ++j) {
        if (count[j] == 0) {
            out.emplace_back(std::nullopt);
            continue;
        }
        const double mean = total[j] / count[j];
        const double db = mean > 0 ? 10.0 * std::log10(mean) : CONF_FLOOR_DB;
        out.emplace_back(round1(CONF_FLOOR_DB > db ? CONF_FLOOR_DB : db));
    }
    return out;
}

std::vector<std::optional<double>> confidence_db(const PassResult& p, const FrameSpec* spec,
                                                 std::optional<double> elapsed_s, std::int64_t n) {
    if (!spec || p.waveform != static_cast<std::int64_t>(spec->waveform())) spec = &frame::get_any(p.frame);
    const std::vector<double> w(p.w.begin(), p.w.end());
    const std::vector<double> llr(p.hdr_llr.begin(), p.hdr_llr.end());
    const std::vector<double> psi(p.psi.begin(), p.psi.end());
    return confidence_db(*spec, w, &llr, &psi, elapsed_s, n);
}

}  // namespace sstvae::qrss::receiver
