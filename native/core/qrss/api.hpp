// QRSSTVAE transmit, in process: what the desktop app needs to send a
// picture without running qrss_encode.py and qrss_transmit.py (plan stage 1
// item 18). Qt-free; the codec half (an encoder from a model) is
// qrss/codec/api.hpp, in sstvae_qrss_codec, so everything here builds and
// is tested without onnxruntime.
//
// The pieces, in the order a send uses them:
//
//   encode_picture(picture, mode, codec_id, encoder)  -> StoredPicture
//       (qrss_encode.py's: fit to 640x480, to [0, 1], encode, from_latents)
//   PassRequest{q, waveform, freq_hz, callsign, grid, segment, ...}
//   problem(source, request)                          -> "" or why not
//   pass_audio(source, request)                       -> float32 at 8 kHz
//       (or pass_stream for a PassAudio to render in blocks)
//
// A pass's audio is exactly what `qrss_transmit.py IN OUT --float` writes
// for the same arguments (sstvae-qrss transmit is built on these): sample
// 0 is AUDIO_LEAD_S before t0 = quarter hour + T0_OFFSET_S, the last
// AUDIO_TAIL_S after the keying ends. problem() returns the script's own
// refusal text (its SystemExit), in its order; pass_audio throws
// std::invalid_argument with the same text.
//
// Every number below comes from the generated qrss/config.hpp, i.e. from
// the reference, never from a copy typed by hand.

#ifndef SSTVAE_QRSS_API_HPP
#define SSTVAE_QRSS_API_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "images/types.hpp"
#include "qrss/codec/encode.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/picture.hpp"
#include "qrss/tx.hpp"

namespace sstvae::qrss::api {

// --- timing and band --------------------------------------------------------------------

inline constexpr int SAMPLE_RATE = config::FS;                    // 8000: the pass audio's rate
inline constexpr double SLOT_S = config::SLOT_S;                  // 900: slot q starts at SLOT_S * q (unix s)
inline constexpr double T0_OFFSET_S = config::T0_OFFSET_S;        // 1: t0 = quarter hour + 1 s
inline constexpr double AUDIO_LEAD_S = config::TX_PRE_S;          // 12: the audio starts this long before t0
inline constexpr double AUDIO_TAIL_S = config::TX_POST_S;         // 3: and ends this long after the keying
inline constexpr double PASS_S = config::FULL_DURATION_S;         // 1782.678125: a FULL frame, t0 to its end
inline constexpr double CARRIER_MIN_HZ = config::CARRIER_BAND_HZ[0];   // 300: the band receivers search
inline constexpr double CARRIER_MAX_HZ = config::CARRIER_BAND_HZ[1];   // 2700
inline constexpr double CARRIER_DEFAULT_HZ = config::TX_CARRIER_HZ;    // 1500
inline constexpr double LEAD_IN_MAX_S = config::LEAD_IN_MAX_S;    // 10: --lead-in's limit

// --- modes, waveforms and frames ----------------------------------------------------------

// "A", "B", "C": the mode index (0, 1, 2), nullopt for anything else.
std::optional<int> mode_index(std::string_view name);
// The passes (latent groups) a mode sends: 1, 2, 3; 0 for an unknown mode.
int passes_for(std::string_view mode);

enum class Waveform { CE, L };
// "CE" or "L" (qrss_transmit.py's --waveform), nullopt otherwise.
std::optional<Waveform> waveform_named(std::string_view name);
std::string_view waveform_name(Waveform w);

// qrss_transmit.py's --frame choices, sorted(frame.PRESETS).
inline constexpr std::array<std::string_view, 4> FRAMES{"full", "medium", "short", "tiny"};
// The FrameSpec of `frame` in waveform `w` (CE "short" -> frame "short",
// L "short" -> "l-short"); std::invalid_argument for a name not in FRAMES.
const frame::FrameSpec& frame_spec(Waveform w, std::string_view frame = "full");

// --- slots ---------------------------------------------------------------------------------

// q of a quarter hour given as ISO 8601 text with a zone (sequences.parse_slot,
// CPython 3.13's fromisoformat): std::invalid_argument with its texts.
std::int64_t parse_slot(std::string_view text);
// "2026-10-10T03:00Z" for slot q (std::overflow_error outside years 1..9999).
std::string slot_iso(std::int64_t q);
// q of a quarter hour in unix seconds; std::invalid_argument
// ("QRSS passes start on a quarter hour") for any other time.
std::int64_t slot_of(double quarter_hour_unix_s);
// Unix seconds of slot q's quarter hour, of its t0, and of its audio's
// first sample (t0 - AUDIO_LEAD_S).
inline double quarter_hour_s(std::int64_t q) { return SLOT_S * static_cast<double>(q); }
inline double t0_s(std::int64_t q) { return quarter_hour_s(q) + T0_OFFSET_S; }
inline double audio_start_s(std::int64_t q) { return t0_s(q) - AUDIO_LEAD_S; }
// Samples in a pass's audio (tx.slot_samples), and the unix time just
// after its last one.
std::int64_t pass_samples(Waveform w, std::string_view frame = "full");
double audio_end_s(std::int64_t q, Waveform w, std::string_view frame = "full");

// --- what a pass is made from ----------------------------------------------------------------

using Source = tx::Source;   // a StoredPicture (.qrsp) or a BeaconFile (.bin)
using Encoder = codec::Encoder;

// qrss_encode.py's encode of a picture in memory: fit to IMG_W x IMG_H
// (images::fit, cover and centre crop), to [0, 1] (images::to_array), the
// encoder, StoredPicture.from_latents. A 640x480 picture is not resized,
// so it gives what qrss_encode.py gives for the same pixels saved as PNG.
// Refusals are picture.py's (mode, codec ID) as std::invalid_argument.
picture::StoredPicture encode_picture(const images::Picture& picture, std::int64_t mode, std::int64_t codec_id,
                                      const Encoder& encode);

// qrss_transmit.py's input: a beacon file when it starts with the beacon
// magic, else a stored picture. std::invalid_argument for a file that is
// neither (beaconfile's and picture.load_qrsp's ValueError texts),
// std::runtime_error when it cannot be read.
Source read_source(const std::string& path);

// --- one pass --------------------------------------------------------------------------------

// qrss_transmit.py's arguments after --slot, with its defaults.
struct PassRequest {
    std::int64_t q = 0;                        // the slot (parse_slot / slot_of)
    Waveform waveform = Waveform::CE;
    double freq_hz = CARRIER_DEFAULT_HZ;       // CE: the carrier; L: the pilot
    std::optional<std::string> callsign;       // a stored picture needs one (as given: "K1ABC")
    std::optional<std::string> grid;           // a 4-character locator, or none
    std::int64_t segment = 0;                  // a stored picture's group, 0..mode
    std::string frame = "full";                // one of FRAMES; anything but full is a test frame
    double lead_in_s = 0.0;                    // plain carrier before the preamble, 0..LEAD_IN_MAX_S
    bool ook = false;                          // CE, stored picture: on-off keyed callsign windows
};
// A beacon file sends its own segment, header and keying: `segment` and
// `grid` are not used for one, `callsign` must be the file's when given and
// `ook` is refused for a file that keys FSK, all as the script does.

// Everything qrss_transmit.py refuses for this source and request, in its
// order, as std::invalid_argument with its SystemExit text: --ook with
// waveform L; a beacon file for another callsign or keyed FSK under ook; a
// stored picture without a callsign; then tx.transmit_audio's own (the
// carrier or pilot, the segment, the header's callsign and grid, the
// lead-in, ...); last, a slot before 1970 ("q must fit in uint64, not -1").
// A frame not in FRAMES is refused first (the script's argparse cannot get
// that far). The plan holds what the pass sends; PassAudio makes its audio.
tx::PassPlan plan_pass(const Source& src, const PassRequest& r);

// That refusal's text, or "" when the pass can be made.
std::string problem(const Source& src, const PassRequest& r);

// The pass's audio, to render in blocks (any block size gives the same
// bits; see tx::PassAudio), or whole as float32: every sample as the
// script's --float WAV holds it. std::invalid_argument as plan_pass.
tx::PassAudio pass_stream(const Source& src, const PassRequest& r);
std::vector<float> pass_audio(const Source& src, const PassRequest& r);

// qrss_transmit.py's WAV of the pass, byte for byte: float32 with
// float32 = true (--float), else peak-normalised int16. wav::WriteError
// when the file cannot be written; nothing is written on a refusal.
void write_wav(const Source& src, const PassRequest& r, const std::string& path, bool float32);

}  // namespace sstvae::qrss::api

#endif  // SSTVAE_QRSS_API_HPP
