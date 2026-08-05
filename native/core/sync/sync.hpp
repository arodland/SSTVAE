// Acquisition: preamble detection, timing, and carrier frequency offset.
//
// Port of sstvae/modem/sync.py. The preamble is periodic with M
// samples, so an autocorrelation at lag M gives detection plus a
// fractional CFO estimate that is unambiguous over +/- FS/(2M) =
// +/-25 Hz. The remaining offset is a multiple of the 50 Hz carrier
// spacing, resolved by trying integer-bin candidates against the known
// preamble template. Net tolerance comfortably exceeds +/-50 Hz.

#pragma once

#include <complex>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"

namespace sstvae::sync {

using cdouble = std::complex<double>;

class SyncError : public std::runtime_error {
   public:
    explicit SyncError(const std::string& what) : std::runtime_error(what) {}
};

struct Acquisition {
    std::int64_t preamble_start;  // index of first preamble sample (CP start)
    double freq_offset;           // Hz
    double metric;                // detection confidence, ~0..1
};

// Optional [start, end) restriction on the preamble hunt. The rest of
// the signal is still used for frames.
struct SearchWindow {
    std::int64_t start;
    std::int64_t end;
};

Acquisition acquire(std::span<const cdouble> z, double threshold = config::PREAMBLE_THRESHOLD,
                    int max_bins = 2,
                    std::optional<SearchWindow> search = std::nullopt);

struct BlindAcquisition {
    std::int64_t frame_start;  // a pilot symbol's useful-window start
    double freq_offset;        // Hz
    double metric;             // prominence; not comparable to Acquisition::metric
};

// Recover frame-boundary timing and carrier frequency purely from the
// frame pilot's own periodicity (it repeats every FRAME_SAMPLES), with
// NO dependence on the transmission-start preamble -- which is sent
// once and so is useless for a recording that starts mid-transmission.
//
// For each candidate CFO bin this matched-filters the whole window
// against one bare pilot symbol and folds the energy into
// FRAME_SAMPLES-periodic phase bins, integrating across every period
// available. `metric` is the winning phase's prominence over the other
// 1151 bins (peak / median), so it is scale invariant and `threshold`
// does not need retuning per signal level.
BlindAcquisition acquire_blind(std::span<const cdouble> z,
                               double max_offset_hz = 55.0,
                               double bin_step_hz = 1.7, int min_periods = 8,
                               double threshold = 4.0,
                               std::optional<SearchWindow> search = std::nullopt);

}  // namespace sstvae::sync
