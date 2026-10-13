#include "qrss/tx.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/lwave.hpp"
#include "qrss/morse.hpp"
#include "qrss/precoder.hpp"

namespace sstvae::qrss::tx {

namespace {

std::string str(std::int64_t v) { return std::to_string(v); }

[[noreturn]] void refuse(const std::string& what) { throw std::invalid_argument(what); }

// Python's format(x, "g") for the band edges in the refusals ("300").
std::string fmt_g(double x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", x);
    return buf;
}

// float(L_PILOT_HZ + L_EDGE_HZ): the exact sum, divided once (as
// Fraction.__float__ divides its numerator by its denominator).
double pilot_to_data_edge_hz() {
    constexpr auto p = config::L_PILOT_HZ, e = config::L_EDGE_HZ;
    constexpr std::int64_t num = p.num * e.den + e.num * p.den, den = p.den * e.den;
    static_assert(num < (std::int64_t(1) << 53) && den < (std::int64_t(1) << 53), "exact as doubles");
    return static_cast<double>(num) / static_cast<double>(den);
}

const picture::StoredPicture* as_picture(const Source& src) { return std::get_if<picture::StoredPicture>(&src); }
const beaconfile::BeaconFile* as_beacon(const Source& src) { return std::get_if<beaconfile::BeaconFile>(&src); }

}  // namespace

SegmentAndHeader segment_and_header(const Source& src, std::optional<std::int64_t> segment,
                                    const std::optional<header::HeaderFields>& header,
                                    const std::optional<std::string>& segment_shown) {
    SegmentAndHeader out;
    if (const auto* bf = as_beacon(src)) {
        const header::HeaderFields own = bf->header_fields();
        if (header && !(*header == own))
            refuse("a beacon file carries its own header; it does not match the one given");
        if (segment && *segment != bf->segment)
            refuse("the beacon file holds segment " + str(bf->segment) + ", not " +
                   segment_shown.value_or(str(*segment)));
        out.segment = bf->segment;
        out.header = own;
        out.hdr_bits = bf->hdr_bits;
        out.keying = bf->keying;
        out.ook = bf->ook;
        return out;
    }
    const picture::StoredPicture& sp = *as_picture(src);
    const std::int64_t seg = segment.value_or(0);
    if (!(0 <= seg && seg <= sp.mode)) refuse("segment must be 0.." + str(sp.mode) + ", not " + str(seg));
    out.segment = seg;
    if (!header) return out;
    beaconfile::check_header_matches(sp, seg, *header);
    out.header = *header;
    out.hdr_bits = header::encode(*header);
    out.keying = morse::keying_units(header->callsign);
    return out;
}

std::vector<double> slot_air_latents(const Source& src, std::optional<std::int64_t> segment,
                                     std::optional<bool> use_int8) {
    if (const auto* bf = as_beacon(src)) {
        if (use_int8 && !*use_int8) refuse("a beacon file holds only the int8 latents");
        return picture::from_int8(std::span<const std::int8_t>(bf->latents_i8));
    }
    const picture::StoredPicture& sp = *as_picture(src);
    const std::int64_t seg = segment.value_or(0);
    // src.segs[seg]: a Python list index, negative from the end.
    const auto n = static_cast<std::int64_t>(sp.segs.size());
    if (seg >= n || seg < -n) throw std::out_of_range("list index out of range");
    const picture::Segment& s = sp.segs[static_cast<std::size_t>(seg < 0 ? seg + n : seg)];
    std::vector<double> air = picture::air_values(std::span<const picture::Half>(s), seg);
    if (use_int8.value_or(false)) return picture::from_int8(std::span<const std::int8_t>(picture::to_int8(air)));
    return air;
}

SlotPlan plan_symbols(const Source& src, const frame::FrameSpec& spec, std::optional<std::int64_t> segment,
                      const std::optional<header::HeaderFields>& header, std::optional<bool> use_int8,
                      const std::optional<std::string>& segment_shown) {
    SegmentAndHeader sh = segment_and_header(src, segment, header, segment_shown);
    std::vector<double> air = slot_air_latents(src, sh.segment, use_int8);
    const std::int64_t n_data = spec.n_data();
    if (static_cast<std::int64_t>(air.size()) < n_data)
        refuse("frame " + frame::py_repr(spec.name()) + " sends " + str(n_data) + " latents; the source holds " +
               std::to_string(air.size()));
    if (spec.has_header() && !sh.hdr_bits)
        refuse("frame " + frame::py_repr(spec.name()) + " has a header: pass header= for a .qrsp");
    SlotPlan plan;
    air.resize(static_cast<std::size_t>(n_data));
    plan.air = std::move(air);
    if (spec.has_header()) plan.hdr_bits = std::move(sh.hdr_bits);
    return plan;
}

Symbols assemble_symbols(const SlotPlan& plan, const frame::FrameSpec& spec, std::uint64_t q) {
    const std::vector<double> x = precoder::precode(std::span<const double>(plan.air), q);
    std::optional<frame::HdrBits> bits;
    if (plan.hdr_bits) bits = frame::HdrBits::of(*plan.hdr_bits);
    const frame::HdrBits* b = bits ? &*bits : nullptr;
    Symbols out;
    out.waveform = spec.waveform();
    if (spec.waveform() == frame::Waveform::L)
        out.l = frame::assemble_l(spec, b, std::span<const double>(x));
    else
        out.ce = frame::assemble(spec, b, std::span<const double>(x));
    return out;
}

Symbols slot_symbols(const Source& src, std::uint64_t q, const frame::FrameSpec& spec,
                     std::optional<std::int64_t> segment, const std::optional<header::HeaderFields>& header,
                     std::optional<bool> use_int8) {
    return assemble_symbols(plan_symbols(src, spec, segment, header, use_int8), spec, q);
}

Symbols slot_symbols(const Source& src, std::uint64_t q) { return slot_symbols(src, q, frame::get("full")); }

std::optional<std::vector<std::uint8_t>> slot_keying(const Source& src,
                                                     const std::optional<header::HeaderFields>& header) {
    if (const auto* bf = as_beacon(src)) return bf->keying;
    if (!header) return std::nullopt;
    return morse::keying_units(header->callsign);
}

std::int64_t slot_samples(const frame::FrameSpec& spec, double pre_s, double post_s, double fs) {
    const double end_s = spec.keyed_end_pos() * spec.t_sym() + post_s;
    const double n = std::ceil((pre_s + end_s) * fs);
    if (std::isnan(n)) refuse("cannot convert float NaN to integer");
    if (std::isinf(n)) throw std::overflow_error("cannot convert float infinity to integer");
    if (!(std::fabs(n) < 9223372036854775808.0))
        throw std::overflow_error("slot_samples: " + frame::py_repr(n) + " samples do not fit int64");
    return static_cast<std::int64_t>(n);
}

double check_pilot(double carrier_hz, const std::optional<std::string>& shown) {
    const std::string text = shown.value_or(frame::py_repr(carrier_hz));
    const double f = ce::check_carrier(carrier_hz, text);
    const double low = f - pilot_to_data_edge_hz();
    const double lo = config::CARRIER_BAND_HZ[0];
    if (low < lo) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.1f", low);
        refuse("an L pilot at " + text + " Hz puts the data down to " + buf + " Hz, below the " + fmt_g(lo) +
               " Hz receivers search");
    }
    return f;
}

// --- the pass's audio ------------------------------------------------------------------------

namespace {

using cplx = std::complex<double>;

constexpr double PI_D = std::numbers::pi;    // np.pi
constexpr double TWO_PI = 2 * PI_D;          // 2 * np.pi, exact
constexpr double FS_D = config::FS;
// ce.baseband's default chunk: the reference leaves a chunk of it wholly
// outside the keyed span (and its ramps) as +0.
constexpr std::int64_t BB_CHUNK = std::int64_t(1) << 18;
// The piece render works in, so a whole pass never needs more scratch than
// this. Every sample is computed on its own, so it changes nothing.
constexpr std::int64_t BLOCK = std::int64_t(1) << 15;
// transmit_audio's ramp allowance for CE (a literal there; ce.baseband's
// ramp_s default); L's is lwave.KEY_RAMP_S.
constexpr double CE_RAMP_S = 0.05;

const char* const L_OOK =
    "waveform L keys its callsign windows by FSK of its pilot; on-off keying is waveform CE's option (spec 2.7)";
const char* const L_FILE_OOK =
    "this beacon file keys its callsign windows on and off, which waveform L does not do: make an FSK one "
    "(spec 2.7)";

// round(x) and int(x) of a Python float refuse NaN and the infinities so.
void whole_number(double x) {
    if (std::isnan(x)) refuse("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
}

// A whole-valued double as int64, where the reference has a Python int
// that numpy refuses past int64.
std::int64_t as_int64(double r) {
    if (!(r >= -0x1p63 && r < 0x1p63)) throw std::overflow_error("Python int too large to convert to C long");
    return static_cast<std::int64_t>(r);
}

// transmit_audio / _transmit_audio_l after their carrier checks, up to
// their symbols.
PassPlan plan_common(const Source& src, const PassSpec& pass, bool l_path) {
    const frame::FrameSpec& spec = pass.spec;
    SegmentAndHeader sh = segment_and_header(src, pass.segment, pass.header, pass.shown.segment);
    if (l_path && sh.ook) refuse(L_FILE_OOK);
    if (spec.n_win() != 0 && !sh.keying)
        refuse("frame " + frame::py_repr(spec.name()) +
               " has callsign windows: a callsign is required to transmit (spec 2.7); pass header=");
    PassPlan plan;
    plan.pass = pass;
    plan.waveform_l = l_path;
    plan.ook = !l_path && pass.ook.value_or(sh.ook);   // file_ook if ook is None else bool(ook)
    // pre_s * FS != round(pre_s * FS)
    const double x = pass.pre_s * FS_D;
    whole_number(x);
    if (x != std::nearbyint(x)) refuse("pre_s must be a whole number of samples at 8 kHz");
    double lo = 0.0;
    try {
        lo = l_path ? lwave::keyed_span_pos(spec, pass.lead_in_s).first : ce::keyed_span_pos(spec, pass.lead_in_s).first;
    } catch (const ce::LeadInRefusal&) {
        if (pass.shown.lead_in_s) refuse(ce::lead_in_refusal(*pass.shown.lead_in_s));
        throw;
    } catch (const lwave::LeadInRefusal&) {
        if (pass.shown.lead_in_s) refuse(lwave::lead_in_refusal(*pass.shown.lead_in_s));
        throw;
    }
    // -lo * spec.t_sym + ramp > pre_s
    const double before = -lo * spec.t_sym();
    if (before + (l_path ? lwave::KEY_RAMP_S : CE_RAMP_S) > pass.pre_s) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.2f", before);
        refuse("pre_s = " + pass.shown.pre_s.value_or(frame::py_repr(pass.pre_s)) +
               " s does not cover the keying start (" + buf + " s before t0 plus the ramp)");
    }
    plan.keying = std::move(sh.keying);
    plan.slot = plan_symbols(src, spec, sh.segment, sh.header);
    return plan;
}

}  // namespace

Carrier::Carrier(double hz) : hz_(hz), shown_(frame::py_repr(hz)) {}

Carrier::Carrier(const Rational& hz) : hz_(hz.to_double()), exact_(hz), shown_(hz.str()) {}

Carrier::Carrier(double hz, std::optional<Rational> exact, std::string shown)
    : hz_(hz), exact_(std::move(exact)), shown_(std::move(shown)) {}

Rational Carrier::fraction() const { return exact_ ? *exact_ : ce::carrier_fraction(hz_); }

void check_pass_carrier(const frame::FrameSpec& spec, const Carrier& carrier, std::optional<bool> ook) {
    if (spec.waveform() == frame::Waveform::L) {
        if (ook.value_or(false)) refuse(L_OOK);
        check_pilot(carrier.hz(), carrier.shown());
    } else {
        ce::check_carrier(carrier.hz(), carrier.shown());
    }
}

PassPlan plan_pass(const Source& src, const PassSpec& pass) {
    check_pass_carrier(pass.spec, pass.carrier, pass.ook);
    return plan_common(src, pass, pass.spec.waveform() == frame::Waveform::L);
}

PassPlan plan_pass_l(const Source& src, const PassSpec& pass) {
    check_pilot(pass.carrier.hz(), pass.carrier.shown());
    return plan_common(src, pass, true);
}

PassAudio::PassAudio(const Source& src, const PassSpec& pass) : PassAudio(plan_pass(src, pass), pass.q) {}

PassAudio::PassAudio(const PassPlan& plan, std::uint64_t q)
    : spec_(plan.pass.spec), l_(plan.waveform_l), amplitude_(plan.pass.amplitude) {
    const PassSpec& pass = plan.pass;
    const Symbols sym = assemble_symbols(plan.slot, spec_, q);
    if (l_) {
        // lwave.frame_baseband(sym, spec, keying=keying, lead_in_s=lead_in_s),
        // which takes the symbols as complex128 (a CE frame's too, to fail
        // where the reference does, in lwave).
        std::vector<cplx> s = sym.l;
        if (sym.waveform != frame::Waveform::L) s.assign(sym.ce.begin(), sym.ce.end());
        std::optional<lwave::Keying> k;
        if (plan.keying) k = lwave::Keying::units(*plan.keying);
        lwave::Baseband bb = lwave::frame_baseband(s, spec_, k ? &*k : nullptr, pass.lead_in_s);
        bb_ = std::move(bb.z);
        bb_n0_ = bb.n0;
        bb_fs_ = bb.fs;
    }
    carrier_ = pass.carrier.fraction();
    n_ = slot_samples(spec_, pass.pre_s, pass.post_s);
    n_pre_ = as_int64(std::nearbyint(pass.pre_s * FS_D));   // whole, as plan_pass checked
    // step = max(1, int(chunk_s * FS))
    const double c = pass.chunk_s * FS_D;
    whole_number(c);
    const double t = std::trunc(c);
    step_ = t < 1.0 ? 1 : (t >= 0x1p63 ? std::numeric_limits<std::int64_t>::max() : static_cast<std::int64_t>(t));
    if (n_ < 0) refuse("negative dimensions are not allowed");   // np.empty(n)
    if (!l_) {
        vals_ = ce::at_positions(sym.ce, spec_);
        std::tie(lo_, hi_) = ce::keyed_span_pos(spec_, pass.lead_in_s);
        ramp_ = CE_RAMP_S / config::T_SYM;
        ook_ = plan.ook;
        // The reference's first chunk, when there is one, has ce.baseband
        // check the keying's shape.
        if (n_ > 0 && plan.keying) keying_ = ce::window_keying(*plan.keying, spec_);
    }
    // ...and to_audio's period, den * 8000, which numpy takes only in int64.
    if (n_ > 0 && carrier_.den() > std::numeric_limits<std::int64_t>::max() / config::FS)
        throw std::overflow_error("Python int too large to convert to C long");
    // range(0, n, step): a step past n is one chunk.
    step_ = std::max<std::int64_t>(1, std::min(step_, n_));
}

void PassAudio::ce_baseband(std::int64_t a, std::int64_t b, cplx* z) const {
    const std::int64_t L = ce::grid_step(config::FS);
    const double t_num = config::T_NUM;
    // tau of sample i: the 16 kHz grid index of t0 + (i - n_pre)/8000, in
    // int64 (numpy's, wrapping), over 485 -- ce.baseband's exact path.
    auto tau_of = [&](std::int64_t i) {
        const auto g = static_cast<std::uint64_t>(i - n_pre_) * static_cast<std::uint64_t>(L);
        return static_cast<double>(static_cast<std::int64_t>(g)) / t_num;
    };
    const ce::Keying* keying = keying_ ? &*keying_ : nullptr;
    std::vector<double> tau;
    for (std::int64_t i = a; i < b;) {
        // The reference's chunk holding sample i: transmit_audio's step,
        // then ce.baseband's 2^18 within it.
        const std::int64_t base = (i / step_) * step_;
        const std::int64_t m_k = std::min(step_, n_ - base);
        const std::int64_t j = (i - base) / BB_CHUNK;
        const std::int64_t c0 = base + j * BB_CHUNK;
        const std::int64_t c1 = base + std::min((j + 1) * BB_CHUNK, m_k);
        const std::int64_t stop = std::min(b, c1);
        cplx* out = z + (i - a);
        if (tau_of(c1 - 1) < lo_ - ramp_ || tau_of(c0) >= hi_ + ramp_) {
            std::fill(out, out + (stop - i), cplx(0.0, 0.0));   // unkeyed: stays zero
            i = stop;
            continue;
        }
        // (env * a) * _turns_to_phasor(phi / (2 pi) + theta), as ce.baseband.
        const std::int64_t len = stop - i;
        tau.resize(static_cast<std::size_t>(len));
        for (std::int64_t k = 0; k < len; ++k) tau[static_cast<std::size_t>(k)] = tau_of(i + k);
        const std::vector<double> g = ce::grid_sum(config::FS, i - n_pre_, len, vals_, ce::pulse_table());
        const ce::CwPhase cw = ce::cw_phase_turns(tau, spec_, keying, ook_);
        for (std::size_t k = 0; k < static_cast<std::size_t>(len); ++k) {
            const double amp = ce::ramp_envelope(tau[k], lo_, hi_, ramp_) * cw.a[k];
            const double phi = ce::BETA_F * g[k];
            const cplx e = ce::turns_to_phasor(phi / TWO_PI + cw.theta[k]);
            // (amp + 0j) * e, numpy's complex product
            out[k] = cplx(amp * e.real() - 0.0 * e.imag(), amp * e.imag() + 0.0 * e.real());
        }
        i = stop;
    }
}

template <typename T>
void PassAudio::render_any(std::int64_t i0, std::span<T> out) const {
    const auto m = static_cast<std::int64_t>(out.size());
    if (i0 < 0 || i0 > n_ || m > n_ - i0)
        throw std::out_of_range("PassAudio: samples " + str(i0) + ".." + str(i0) + " + " + str(m) +
                                " are outside the pass, 0.." + str(n_));
    std::vector<cplx> z;
    for (std::int64_t a = 0; a < m; a += BLOCK) {
        const std::int64_t len = std::min(BLOCK, m - a);
        const std::int64_t s0 = i0 + a;
        z.resize(static_cast<std::size_t>(len));
        if (l_)
            lwave::resample_into(bb_, bb_n0_, bb_fs_, config::FS, s0 - n_pre_, z);   // lwave.resample(bb, FS, i0 - n_pre, m)
        else
            ce_baseband(s0, s0 + len, z.data());
        const std::vector<double> audio = ce::to_audio(z, config::FS, carrier_.num(), carrier_.den(), amplitude_, s0);
        for (std::size_t k = 0; k < audio.size(); ++k) out[static_cast<std::size_t>(a) + k] = static_cast<T>(audio[k]);
    }
}

void PassAudio::render(std::int64_t i0, std::span<double> out) const { render_any(i0, out); }
void PassAudio::render(std::int64_t i0, std::span<float> out) const { render_any(i0, out); }

std::size_t PassAudio::read(std::span<double> out) {
    const auto m = std::min(static_cast<std::int64_t>(out.size()), n_ - pos_);
    render(pos_, out.first(static_cast<std::size_t>(m)));
    pos_ += m;
    return static_cast<std::size_t>(m);
}

std::size_t PassAudio::read(std::span<float> out) {
    const auto m = std::min(static_cast<std::int64_t>(out.size()), n_ - pos_);
    render(pos_, out.first(static_cast<std::size_t>(m)));
    pos_ += m;
    return static_cast<std::size_t>(m);
}

void PassAudio::seek(std::int64_t i) {
    if (i < 0 || i > n_) throw std::out_of_range("PassAudio: position " + str(i) + " is outside 0.." + str(n_));
    pos_ = i;
}

std::vector<double> transmit_audio(const Source& src, const PassSpec& pass) {
    const PassAudio audio(src, pass);
    std::vector<double> out(static_cast<std::size_t>(audio.size()));
    audio.render(0, out);
    return out;
}

}  // namespace sstvae::qrss::tx
