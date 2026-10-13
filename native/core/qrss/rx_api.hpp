// QRSSTVAE receive, in process: what an application needs to receive a
// stored slot without running qrss_receive.py (the receiving half of
// api.hpp's transmit). Qt-free, and free of onnxruntime: what comes out is
// the passes (types::PassResult, the .npz record qrss_receive.py stores),
// and turning passes into a picture is the accumulator's and the codec's.
//
// The pieces, in the order a receive uses them:
//
//   capture_from_audio(audio, q, start_unix)  -> Capture
//   capture_from_wav(path, q, start_unix)     -> Capture
//       (receiver.capture_from_wav: 8 kHz audio to the front end's 4 kHz
//       complex stream, t0 = SLOT_S q + T0_OFFSET_S)
//   ReceiveConfig{frames, live_only, estimator, ...}
//   receive_slot(capture, config)             -> passes
//       (receiver.receive_slot(cap, spec, ...): A1 -> A2 (+ A3) -> gate V
//       -> receive_pass for every signal in the slot)
//
// receive_slot is the reference's, with the reference's stages under it
// (receiver_slot.hpp): for the same capture and arguments the passes are
// the ones sstvae.qrss.receiver.receive_slot returns -- the same signals
// in the same order, headers and callsign reads exact, z and w to the
// rounding of the stages' FFTs.
//
// The two rules the receiver has an earlier form of (rules.py:
// QRSS_OLD_EDGE_NOISE, QRSS_OLD_GATING) are set per call here, in
// ReceiveConfig, never read from the environment: the reference reads them
// from os.environ wherever it reaches them, which a process receiving
// several slots at once cannot vary per slot.
//
// Thread-safe: receive_slot keeps no state between calls and may run on
// several threads at once (on one Capture or each on its own: it only
// reads it); the shared state under it is two caches, each behind its own
// mutex (frame::layout's, and acquire_stats' band covariance tables), and
// constant tables built on first use (function-local statics, which C++
// initialises once, thread-safely). Nothing reads the environment. A call is
// long -- seconds for SHORT, minutes for a FULL slot with many signals --
// and not interruptible except through budget_s, which stops gating new
// candidates once spent (the passes of those already gated are still
// received).
//
// Errors: std::invalid_argument for frames receiver.frames refuses (two
// frames of one waveform, or none), and whatever the stages raise for a
// capture they cannot read (an empty one).

#ifndef SSTVAE_QRSS_RX_API_HPP
#define SSTVAE_QRSS_RX_API_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "qrss/acquire.hpp"
#include "qrss/frame.hpp"
#include "qrss/frontend.hpp"
#include "qrss/types.hpp"

namespace sstvae::qrss::api {

// A slot's capture: the front end's 4 kHz complex stream (0 Hz = 1500 Hz
// audio), the quarter hour q, and where t0 falls in it (in samples).
using Capture = frontend::Capture;

// A capture of 8 kHz real audio (float, full scale 1.0) of slot q, the
// first sample at start_unix (unix seconds; nullopt: AUDIO_LEAD_S = 12 s
// before t0, as qrss_transmit.py writes a pass's audio). float32 samples
// are widened exactly, as numpy's asarray(x, float64) widens them.
Capture capture_from_audio(std::span<const double> audio_8k, std::int64_t q,
                           std::optional<double> start_unix = std::nullopt);
Capture capture_from_audio(std::span<const float> audio_8k, std::int64_t q,
                           std::optional<double> start_unix = std::nullopt);

// The same from a WAV file, read as sstvae.wavio.read_wav reads it: PCM
// (8-32 bit) or float, any rate (resampled to 8 kHz), channels averaged.
// wav::ReadError (a std::runtime_error with errno) when the file cannot be
// read; std::invalid_argument for a WAV it does not take.
Capture capture_from_wav(const std::string& path, std::int64_t q, std::optional<double> start_unix = std::nullopt);

// receive_slot's arguments, the reference's names.
struct ReceiveConfig {
    // The frames listened for, at most one per waveform (receiver.frames):
    // one CE frame is the CE receiver; with an L frame, the joint receiver.
    // Default: CE's FULL.
    std::vector<frame::FrameSpec> frames = {frame::get_any("full")};
    // Acquisition on the preambles alone (A1 -> A2): no whole-slot search
    // (A3), which is what costs most on a FULL slot.
    bool live_only = false;
    // receive_pass's estimator: "joint" (the default) or "plain".
    std::string estimator = "joint";
    // Spans (a, b), seconds after t0, that were not heard (gate V and the
    // trackers count only the rest).
    std::vector<std::pair<double, double>> erase_s;
    // Round B (header symbols known, re-track) after a valid header.
    bool round_b = true;
    // At most this many candidates go to gate V (preamble hits first);
    // nullopt: all of them.
    std::optional<std::int64_t> max_candidates;
    // Stop gating new candidates after this many seconds (steady clock);
    // nullopt: no limit. Results then depend on the machine's speed.
    std::optional<double> budget_s;
    // Expected frequencies (Hz, audio): a bare f for every waveform, or
    // (f, waveform). acquire.acquire's hints.
    std::vector<acquire::Hint> hints;
    // rules.py's QRSS_OLD_EDGE_NOISE: the noise floor's earlier form (near
    // 300 and 2700 Hz, bins taken partly off the front end's filter skirt,
    // uncompensated).
    bool old_edge_noise = false;
    // rules.py's QRSS_OLD_GATING: gate every whole-slot candidate, as
    // before a preamble-found signal owned its band.
    bool old_gating = false;
};

// receiver.receive_slot(cap, frames, live_only, estimator, erase_s=,
// round_b=, max_candidates=, budget_s=, hints=): every signal received,
// strongest first as the receiver orders them.
std::vector<types::PassResult> receive_slot(const Capture& capture, const ReceiveConfig& config = {});

}  // namespace sstvae::qrss::api

#endif  // SSTVAE_QRSS_RX_API_HPP
