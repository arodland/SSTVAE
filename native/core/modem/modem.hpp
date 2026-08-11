// Top-level modem: latent vector <-> passband audio samples.
//
// Port of sstvae/modem/modem.py.
//
// TX layout:  silence | preamble | 2x header symbol | N frames | silence
// Frame:      1 pilot symbol + 5 data symbols (230 real latents + 5
// beacon chips on the one carrier reserved for resync/callsign).
//
// RX equalizes each data symbol against per-carrier gains interpolated
// between the surrounding frame pilots, tracks sample-clock drift from
// the pilot phase slope across carriers, and reports per-latent
// confidence weights (0 for frames that never arrived) so a decoder can
// treat missing or faded latents as erasures.

#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "beacon/beacon.hpp"
#include "config.hpp"
#include "sync/sync.hpp"

namespace sstvae::modem {

using cdouble = std::complex<double>;
using config::ModeSpec;
using sync::SyncError;

// Optional second-order loop on the pilots' *common* phase, which is
// residual carrier frequency (the phase *slope across* carriers is
// timing, which the demodulator already tracks; the two are orthogonal).
// Off by default -- see sstvae/modem/modem.py's _DriftTracker for the
// full rationale, and config::DRIFT_* for why there are two settings
// rather than one.
enum class DriftTrack { Off, Slow, Fast };

// Parse a config-file / command-line spelling. Throws
// std::invalid_argument naming what was asked for, like mode_by_name.
DriftTrack drift_track_from_name(std::string_view name);
std::string_view drift_track_name(DriftTrack track);

struct DemodResult {
    std::vector<double> latents;  // canonical order, zeros where not received
    std::vector<double> weights;  // per-latent confidence 0..1 (0 = erased)
    ModeSpec mode;
    double freq_offset;
    double sync_metric;
    int frames_received;
    std::optional<beacon::BeaconResult> beacon;
    std::string callsign;
    std::int64_t preamble_start;
    double snr_db;
};

// Result of demodulate_blind: no preamble/header was needed, so unlike
// DemodResult there is no known mode -- latents/weights are always sized
// for mode C's full canonical range (every mode is a prefix of it) and
// populated only where a demodulated frame actually landed, via the
// beacon's recovered absolute frame index.
struct BlindDemodResult {
    std::vector<double> latents;
    std::vector<double> weights;
    double freq_offset;
    std::optional<beacon::BeaconResult> beacon;
    std::string callsign;
    std::optional<int> frame_offset;  // absolute index of the first local frame
    int n_frames;
    // Sample index the transmitter's own absolute frame 0 would fall at.
    // Known only once the beacon gives frame_offset. May be negative if
    // the buffer starts mid-transmission. Note this is the start of
    // *frame 0*, one preamble+header later than a preamble-path
    // DemodResult::preamble_start for the same transmission.
    std::optional<std::int64_t> frame0_start;
    double snr_db;
};

class Modem {
   public:
    Modem();

    // Latent vector -> unit-RMS float waveform at FS.
    //
    // The on-air contract is unit-RMS latents; `normalize` enforces it.
    // `callsign` (up to 8 chars) rides the reserved beacon carrier along
    // with a resync frame counter on every frame.
    //
    // `clip_headroom_db` defaults to the configured value and exists as
    // a parameter for one reason: the reference's test suite disables
    // clipping by patching the module constant, to measure the modem's
    // own ceiling independent of how the clipper happens to be tuned.
    // A compiled-in constant is unreachable from there, so the port
    // would have silently reported the *clipped* floor for a test whose
    // whole point is to exclude it.
    std::vector<double> modulate(std::span<const double> latents,
                                 const ModeSpec& mode, bool normalize = true,
                                 const std::string& callsign = "",
                                 double clip_headroom_db =
                                     config::CLIP_HEADROOM_DB) const;

    // `search_s` restricts preamble acquisition to a time window
    // (seconds); frames are still demodulated past its end.
    DemodResult demodulate(
        std::span<const double> x,
        std::optional<std::pair<double, double>> search_s = std::nullopt,
        DriftTrack drift_track = DriftTrack::Off) const;

    // Recover frame timing purely from the pilot's own periodicity -- no
    // preamble or header needed, so this works on a recording that
    // starts mid-transmission.
    //
    // No sample-clock drift tracking (that needs a preamble phase
    // reference); fine for the bounded windows this targets.
    //
    // `drift_track` is the *carrier* drift loop, which needs no such
    // reference. It has one limit here it does not have on the preamble
    // path, and it is a cliff rather than a slope: acquire_blind
    // estimates one frequency for its whole window, so on a drifting
    // signal that describes the window's *middle* and the first frame
    // starts about half the window's total drift away. Past the
    // pilot-rate ambiguity the per-frame measurement aliases and the
    // loop locks to the wrong frequency -- measured, at 0.5 Hz/s over
    // 30 s it takes the beacon down where leaving it off decodes.
    //
    // `acquisition`, if given, skips the internal acquire_blind call and
    // demodulates at that position instead -- for a caller (rx/engine.cpp)
    // that already found it via a persistent sync::BlindAccumulator
    // rather than a fresh bounded-window search. The rest is unaffected:
    // it still demodulates every frame the whole of `x` can hold, using
    // `acquisition` only to place frame 0.
    BlindDemodResult demodulate_blind(
        std::span<const double> x,
        std::optional<std::pair<double, double>> search_s = std::nullopt,
        std::optional<sync::BlindAcquisition> acquisition = std::nullopt,
        DriftTrack drift_track = DriftTrack::Off) const;

   private:
    std::vector<cdouble> pilot_;
};

// Look a mode up by its on-air name ("A", "B", "C"). Throws
// std::out_of_range for anything else, naming what was asked for -- an
// unknown mode from a config file or a command line is a typo to report,
// never a default to silently substitute.
const ModeSpec& mode_by_name(std::string_view name);

// Pilot-based radio SNR estimate, in the same "dB SNR in a
// SNR_REF_BW_HZ noise bandwidth" convention used by the channel
// simulator -- so it is directly comparable to those numbers.
//
// `h_pilot` is (n_frames, NC) row-major. `received` may be empty to mean
// "all frames".
double estimate_snr_db(std::span<const cdouble> h_pilot, int n_frames,
                       std::span<const char> received = {});

}  // namespace sstvae::modem
