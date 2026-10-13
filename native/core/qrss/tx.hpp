// The transmit pipeline, the slot part: what one pass of a stored picture
// or a beacon file sends in slot q, for waveform CE and waveform L (design
// section 3). Port of sstvae/qrss/tx.py (the reference) without
// transmit_audio, which lands with the CE and L audio (ce, lwave).
//
// A format module. A pass of segment g in slot q sends
//
//     preamble | header (polar-coded fields, 1 in 16 a reference) |
//     data = precode(air latents, q), 1 in 16 a reference
//
// with the callsign windows inserted where the FrameSpec puts them.
// Everything time-like comes from the FrameSpec.
//
// A stored picture sends its fp16 values; a beacon file sends the int8
// version of the same air values (picture::to_int8), identical in every
// pass and 37 dB below the latents. use_int8 = true on a stored picture
// reproduces what a beacon would send.
//
// Every refusal is std::invalid_argument with the reference's ValueError
// text, in the reference's order (a header field or callsign refused on
// the way carries header.py's or morse.py's text).

#ifndef SSTVAE_QRSS_TX_HPP
#define SSTVAE_QRSS_TX_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "qrss/beaconfile.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/picture.hpp"

namespace sstvae::qrss::tx {

// What a pass is made from: a stored picture (.qrsp) or a beacon file (.bin).
using Source = std::variant<picture::StoredPicture, beaconfile::BeaconFile>;

// tx._segment_and_header: the segment a pass sends, its header fields (or
// none), their coded bits (or none), the callsign keying (or none) and
// whether the source keys its callsign windows on and off.
struct SegmentAndHeader {
    std::int64_t segment = 0;
    std::optional<header::HeaderFields> header;
    std::optional<std::vector<std::uint8_t>> hdr_bits;   // N_HDR_BITS
    std::optional<std::vector<std::uint8_t>> keying;     // CW_UNITS
    bool ook = false;
};

// A beacon file carries its own segment and header: a `header` given must
// equal its header_fields() and a `segment` its own ("the beacon file
// holds segment 1, not 0"; `segment_shown` quotes the argument as given).
// A stored picture takes segment 0..mode (default 0) and, when `header` is
// given, checks it describes that segment of this picture before encoding
// it and keying its callsign.
SegmentAndHeader segment_and_header(const Source& src, std::optional<std::int64_t> segment,
                                    const std::optional<header::HeaderFields>& header,
                                    const std::optional<std::string>& segment_shown = std::nullopt);

// tx.slot_air_latents: float64, the air-order latents this source sends
// (all it holds). A beacon file's are its int8 values / 20 (use_int8 =
// false is refused: it holds nothing else); a stored picture's are segment
// `segment`'s (default 0, indexed as a Python list: -1 is the last, then
// refused as a group) as stored, or through int8 with use_int8 = true.
// A segment past the list is std::out_of_range ("list index out of range").
std::vector<double> slot_air_latents(const Source& src, std::optional<std::int64_t> segment = std::nullopt,
                                     std::optional<bool> use_int8 = std::nullopt);

// The stream symbols of one pass: float64[n_sym] for CE (`ce`),
// complex128[n_sym] for an L frame (`l`).
struct Symbols {
    frame::Waveform waveform = frame::Waveform::CE;
    std::vector<double> ce;
    std::vector<std::complex<double>> l;
    std::size_t size() const { return waveform == frame::Waveform::L ? l.size() : ce.size(); }
};

// slot_symbols in two steps, every refusal that does not involve q first
// (the reference converts q only when it precodes): what the frame sends
// from this source, then the symbols of slot q.
struct SlotPlan {
    std::vector<double> air;                              // the first n_data air-order latents
    std::optional<std::vector<std::uint8_t>> hdr_bits;    // set exactly when the frame has a header
};
SlotPlan plan_symbols(const Source& src, const frame::FrameSpec& spec,
                      std::optional<std::int64_t> segment = std::nullopt,
                      const std::optional<header::HeaderFields>& header = std::nullopt,
                      std::optional<bool> use_int8 = std::nullopt,
                      const std::optional<std::string>& segment_shown = std::nullopt);
Symbols assemble_symbols(const SlotPlan& plan, const frame::FrameSpec& spec, std::uint64_t q);

// tx.slot_symbols: data = precode(the first n_data air-order latents, q);
// the header bits from header::encode(header) (a stored picture needs
// `header` whenever the frame has a header) or from the beacon file.
Symbols slot_symbols(const Source& src, std::uint64_t q, const frame::FrameSpec& spec,
                     std::optional<std::int64_t> segment = std::nullopt,
                     const std::optional<header::HeaderFields>& header = std::nullopt,
                     std::optional<bool> use_int8 = std::nullopt);
// The same on the FULL frame (the reference's default spec).
Symbols slot_symbols(const Source& src, std::uint64_t q);

// tx.slot_keying: the callsign keying for this source (CW_UNITS values), or
// none for a stored picture without a header.
std::optional<std::vector<std::uint8_t>> slot_keying(const Source& src,
                                                     const std::optional<header::HeaderFields>& header = std::nullopt);

// tx.slot_samples: samples in a slot WAV, from t0 - pre_s to the end of
// keying + post_s: ceil((pre_s + keyed_end_pos * t_sym + post_s) * fs).
// A NaN is std::invalid_argument and an infinity (or a count past int64)
// std::overflow_error, as math.ceil refuses them.
std::int64_t slot_samples(const frame::FrameSpec& spec, double pre_s = 12.0, double post_s = 3.0,
                          double fs = config::FS);

// tx.check_pilot: an L pilot in Hz, or std::invalid_argument when the
// carrier is outside CARRIER_BAND_HZ or the data (L_PILOT_HZ + L_EDGE_HZ
// below the pilot) would leave it. `shown` is the carrier as the refusal
// quotes it (default: Python's repr of the double).
double check_pilot(double carrier_hz, const std::optional<std::string>& shown = std::nullopt);

}  // namespace sstvae::qrss::tx

#endif  // SSTVAE_QRSS_TX_HPP
