// The transmit pipeline: what one pass of a stored picture or a beacon
// file sends in slot q, and its 8 kHz audio, for waveform CE and waveform
// L (design section 3). Port of sstvae/qrss/tx.py (the reference), with
// transmit_audio as a streaming PassAudio (below).
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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "qrss/beaconfile.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/picture.hpp"
#include "qrss/rational.hpp"

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
std::int64_t slot_samples(const frame::FrameSpec& spec, double pre_s = config::TX_PRE_S, double post_s = config::TX_POST_S,
                          double fs = config::FS);

// tx.check_pilot: an L pilot in Hz, or std::invalid_argument when the
// carrier is outside CARRIER_BAND_HZ or the data (L_PILOT_HZ + L_EDGE_HZ
// below the pilot) would leave it. `shown` is the carrier as the refusal
// quotes it (default: Python's repr of the double).
double check_pilot(double carrier_hz, const std::optional<std::string>& shown = std::nullopt);

// --- the pass's audio: tx.transmit_audio, streamed ----------------------------------
//
// One pass of a stored picture or a beacon file in slot q as 8 kHz audio:
// sample 0 is t0 - pre_s (the quarter hour + 1 s - pre_s) and the last is
// pre_s + the keyed frame + post_s later (slot_samples). Waveform CE is
// ce.baseband mixed to the carrier by ce.to_audio; waveform L is the whole
// frame through the clipper (lwave.frame_baseband, at 250 Hz), interpolated
// to 8 kHz (lwave.resample) and mixed to the pilot the same way.
//
// PassAudio produces any block of that audio on request, so a 30-minute
// pass is never held in memory: CE holds its symbols at their positions (a
// few hundred KB) and evaluates every sample on its own; L holds its
// clipped 250 Hz frame (7 MB for l-full), built when the PassAudio is.
// Every sample is computed on its own, so the audio does not depend on how
// it is asked for -- any block size gives the same bits -- and
// transmit_audio is one render of the whole pass. render() is const and
// touches no shared state, so blocks may be rendered from several threads.
//
// How close to the reference: each sample is the reference's own
// arithmetic (ce.baseband and ce.to_audio, lwave.resample), so CE is bit
// for bit where ce's is (numpy's float64 sin/cos are libm's and its
// complex multiply fused -- the reference platform) and within 1e-12
// elsewhere; L carries the clipper's FFT rounding, within 1e-9.
//
// The reference makes its audio in chunk_s pieces, and ce.baseband its
// baseband in 2^18-sample ones within those, and two things follow from
// where those pieces fall, both reproduced here so that no block size can
// tell: a piece wholly outside the keyed span and its ramps is left +0
// (where a computed zero can carry either sign), and ce.baseband takes its
// exact 16 kHz grid path only when the piece's start time,
// (i0 - n_pre) / 8000, times 8000 is within 1e-9 of a whole sample -- true
// of every piece of every preset's pass, and false only for a piece that
// starts more than about 2048 s from t0 (a frame longer than any preset,
// or a pre_s that long), which is evaluated in closed form (ce.phase_at)
// as the reference evaluates it. The two paths are not interchangeable:
// they disagree by up to ~1e-2 where tau is a whole symbol, because the
// pulse's cut at +-8 symbols is a step (p(8) = 0.006) that the grid takes
// at both ends and the closed form at one.

// The carrier a pass is sent on, as transmit_audio's carrier_hz takes it:
// a float (an --freq value; the audio is mixed to
// Fraction(f).limit_denominator(2^20), ce.carrier_fraction), an integer, or
// an exact Rational (a Fraction, taken as it is). The band checks test
// float(carrier_hz) (`hz`), and a refusal quotes it as Python's f-string
// would (`shown`). For an L frame it is the pilot.
class Carrier {
public:
    Carrier() : Carrier(config::TX_CARRIER_HZ) {}   // the reference's default, the int 1500
    Carrier(double hz);            // NOLINT(google-explicit-constructor): a Python float
    template <std::integral I>
        requires(!std::same_as<I, bool>)
    Carrier(I hz) : Carrier(static_cast<double>(hz), Rational(hz), std::to_string(hz)) {}  // NOLINT
    Carrier(const Rational& hz);   // NOLINT(google-explicit-constructor): a Fraction
    // Any other form (the binding's str or numpy scalar): float(carrier_hz),
    // its exact fraction (ce.carrier_fraction of the original object) and
    // its text. An unset fraction is the float's, ce.carrier_fraction(hz).
    Carrier(double hz, std::optional<Rational> exact, std::string shown);

    double hz() const { return hz_; }
    const std::string& shown() const { return shown_; }
    // ce.carrier_fraction(carrier_hz): what the audio is mixed to.
    Rational fraction() const;

private:
    double hz_ = 1500.0;
    std::optional<Rational> exact_;
    std::string shown_;
};

// One pass, as transmit_audio's arguments describe it.
struct PassSpec {
    std::uint64_t q = 0;                            // the slot (quarter-hour count): the scrambler
    Carrier carrier;                                // Hz; L's pilot (default the int 1500)
    frame::FrameSpec spec = frame::get("full");     // CE or L, which picks the waveform
    std::optional<std::int64_t> segment;            // a stored picture's (default 0); a beacon file's own
    std::optional<header::HeaderFields> header;     // a stored picture's header (its callsign keys the windows)
    double lead_in_s = 0.0;                         // plain carrier before the preamble, 0..10 s
    std::optional<bool> ook;                        // CE: on-off keyed windows (unset: the source's choice)
    double pre_s = config::TX_PRE_S;                // audio before t0 (12): whole samples, covering the keying
    double post_s = config::TX_POST_S;              // audio after the keying ends (3)
    double amplitude = config::TX_AMPLITUDE;        // peak of a unit-power baseband (0.5)
    double chunk_s = config::TX_CHUNK_S;            // the reference's piece size (120; see above)
    // How a refusal quotes an argument -- Python's f-string of what the
    // caller passed (the binding's). Unset: the value as Python's float repr.
    struct Shown {
        std::optional<std::string> segment, lead_in_s, pre_s;
    } shown;
};

// The checks transmit_audio makes before it looks at the source, in its
// order: an L frame refuses on-off keying (check_ook: "waveform L keys its
// callsign windows by FSK of its pilot; ..."), then the carrier is checked
// (check_pilot for L, ce.check_carrier for CE).
void check_ook(const frame::FrameSpec& spec, std::optional<bool> ook);
void check_pass_carrier(const frame::FrameSpec& spec, const Carrier& carrier, std::optional<bool> ook);

// Everything transmit_audio settles before it needs q (the reference
// converts q only when it precodes): the checks, then what the frame sends.
struct PassPlan {
    PassSpec pass;                                     // pass.q is not used
    bool waveform_l = false;                           // the L path (lwave), else CE (ce)
    SlotPlan slot;                                     // the first n_data air latents, the header bits
    std::optional<std::vector<std::uint8_t>> keying;   // the callsign's CW_UNITS keying, or none
    bool ook = false;                                  // CE: the windows on-off keyed
};

// transmit_audio up to its symbols, every refusal in the reference's order:
// check_pass_carrier; the source's segment and header (_segment_and_header);
// L: a beacon file that keys on and off; a frame with windows and no
// callsign; pre_s not a whole number of samples; the lead-in (0..10 s);
// pre_s not covering the keying start and its ramp; and slot_symbols'
// own (n_data, a .qrsp without a header).
PassPlan plan_pass(const Source& src, const PassSpec& pass);
// tx._transmit_audio_l's entry: the L path whatever the frame, and no ook
// check (it takes no ook). A CE frame fails where the reference's does,
// in lwave (frame::WrongWaveform).
PassPlan plan_pass_l(const Source& src, const PassSpec& pass);

class PassAudio {
public:
    // The pass of `plan` in slot q: its symbols, then (L) its clipped frame,
    // then the rest of the reference's checks -- slot_samples (NaN/inf
    // post_s), chunk_s (int(chunk_s * 8000): NaN, inf), a negative length
    // ("negative dimensions are not allowed") and, for a pass of any
    // length, the keying's shape (ce.window_keying) and a carrier
    // denominator past int64 / 8000.
    PassAudio(const PassPlan& plan, std::uint64_t q);
    // plan_pass(src, pass) in slot pass.q.
    PassAudio(const Source& src, const PassSpec& pass);

    std::int64_t size() const { return n_; }       // tx.slot_samples(spec, pre_s, post_s)
    std::int64_t n_pre() const { return n_pre_; }  // samples before t0: round(pre_s * 8000)
    std::int64_t position() const { return pos_; }
    const frame::FrameSpec& spec() const { return spec_; }
    bool waveform_l() const { return l_; }
    const Rational& carrier() const { return carrier_; }   // what the audio is mixed to
    std::int64_t chunk() const { return step_; }           // the reference's chunk, in samples (at most size(), at least 1)

    // std::out_of_range unless samples [i0, i0 + m) lie inside [0, size()):
    // render's own check, for a caller that sizes a buffer from m (the
    // binding checks before it allocates).
    void check_block(std::int64_t i0, std::int64_t m) const;
    // Samples [i0, i0 + out.size()) of the pass, float64, or rounded once to
    // float32 (numpy's astype). std::out_of_range unless inside [0, size()).
    void render(std::int64_t i0, std::span<double> out) const;
    void render(std::int64_t i0, std::span<float> out) const;
    // The next samples from position(), up to out.size() of them (fewer at
    // the end, none past it); returns how many and moves on by that much.
    std::size_t read(std::span<double> out);
    std::size_t read(std::span<float> out);
    // Move position() to i, 0 <= i <= size() (std::out_of_range otherwise).
    void seek(std::int64_t i);

private:
    template <typename T>
    void render_any(std::int64_t i0, std::span<T> out) const;
    // The CE baseband at samples [a, b), inside one of ce.baseband's pieces.
    void ce_baseband(std::int64_t a, std::int64_t b, std::complex<double>* z) const;

    frame::FrameSpec spec_;
    bool l_ = false;
    std::int64_t n_ = 0, n_pre_ = 0, step_ = 1, pos_ = 0;
    Rational carrier_;
    double amplitude_ = 0.5;
    // CE
    std::vector<double> sym_;               // the stream symbols (the closed form's)
    std::vector<double> vals_;              // ... at their positions (ce.at_positions)
    std::optional<std::vector<std::uint8_t>> keying_;   // (n_win, 192), ce.window_keying
    bool ook_ = false;
    double lo_ = 0.0, hi_ = 0.0, ramp_ = 0.0;            // keyed span and ramp, in symbols
    // L
    std::vector<std::complex<double>> bb_;  // the clipped frame at 250 Hz from bb_n0_
    std::int64_t bb_n0_ = 0, bb_fs_ = 0;
};

// tx.transmit_audio: the whole pass, float64[slot_samples] (one render).
std::vector<double> transmit_audio(const Source& src, const PassSpec& pass);

// qrss_transmit.py's WAV of a pass, streamed: peak-normalised int16
// (wavio.write_wav, which needs the peak first, so the pass is rendered
// twice) or, with float32, every sample as float32 (wavio.write_wav_float).
// Byte for byte what those write for transmit_audio's array (qrss/wav.hpp).
void write_wav(const PassAudio& audio, const std::string& path, bool float32 = false, double peak = 0.95);

}  // namespace sstvae::qrss::tx

#endif  // SSTVAE_QRSS_TX_HPP
