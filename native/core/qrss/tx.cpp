#include "qrss/tx.hpp"

#include <cmath>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

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

// ce.check_carrier: the carrier in Hz, refused outside CARRIER_BAND_HZ.
// (ce.py's own function; tx.check_pilot is its only caller in this module.
// When native/core/qrss/ce lands this should call ce::check_carrier.)
double check_carrier(double f, const std::string& shown) {
    const double lo = config::CARRIER_BAND_HZ[0], hi = config::CARRIER_BAND_HZ[1];
    if (!(std::isfinite(f) && lo <= f && f <= hi))
        refuse("carrier " + shown + " Hz is outside the " + fmt_g(lo) + "-" + fmt_g(hi) +
               " Hz audio band that receivers search");
    return f;
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
    const double f = check_carrier(carrier_hz, text);
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

}  // namespace sstvae::qrss::tx
