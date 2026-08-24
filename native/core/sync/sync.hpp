// Acquisition: preamble detection, timing, and carrier frequency offset.
//
// Port of sstvae/modem/sync.py. The preamble is periodic with M
// samples, so an autocorrelation at lag M gives detection plus a
// fractional CFO estimate that is unambiguous over +/- FS/(2M) =
// +/-25 Hz. The remaining offset is a multiple of the 50 Hz carrier
// spacing, resolved by trying integer-bin candidates against the known
// preamble template. Net tolerance comfortably exceeds +/-50 Hz.
//
// config::ACQUIRE_MAX_BINS is that tolerance, and is the only thing
// setting it out to about +/-700 Hz where the sync lowpass takes over.
// Detection is CFO-blind -- an offset multiplies every lag-M product by
// one constant phasor, which |.| removes -- so a wider search cannot
// move the false-alarm rate *at the true preamble's own location*, and
// measured it costs neither sensitivity nor meaningful CPU there. See
// docs/todo.md.
//
// That does not cover every *other* location a real transmission's own
// data can produce a passable lag-M metric peak at: a genuinely
// off-frequency signal has real spectral content near its true offset
// even away from the preamble, so more candidates means more chances
// one of them resonates with it instead of noise.
// config::TEMPLATE_SCORE_THRESHOLD is the second gate that catches
// that -- see its comment in config.py for the measurement.

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
                    int max_bins = config::ACQUIRE_MAX_BINS,
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
                               double max_offset_hz = config::BLIND_MAX_OFFSET_HZ,
                               double bin_step_hz = config::BLIND_BIN_STEP_HZ,
                               int min_periods = 8,
                               double threshold = config::BLIND_SCORE_THRESHOLD,
                               std::optional<SearchWindow> search = std::nullopt);

// Sub-bin CFO from the winning bin and its two neighbours. The search
// grid is deliberately far coarser than the estimate it has to produce
// (see config::BLIND_BIN_STEP_HZ), and this is what makes that safe:
// without it the demodulator gets several Hz of residual, which costs
// the picture on its own. Legitimate because the folded score is
// band-limited in CFO. Written for non-uniform abscissae because the
// grid is non-uniform -- a shift must be a whole number of block-FFT
// bins.
double refine_cfo(std::span<const double> freqs, std::span<const double> scores,
                  std::size_t i);

// Incremental counterpart to acquire_blind(). Port of
// sstvae.modem.sync.BlindAccumulator -- see that class's docstring for
// the full design rationale (block-wise overlap-save so push() costs
// O(new samples) rather than O(window length); why the block-local
// circular-shift CFO trick still folds to the correct energy; why
// window_s is exponential decay rather than exact eviction).
//
// push() requires contiguous input (no gaps, no re-sent samples) and
// throws SyncError otherwise. result() throws SyncError exactly as
// acquire_blind() does: too little data pushed yet, or no bin's peak
// clears `threshold`.
// `window_s` runs one decay timescale per entry in parallel, off the
// same (shared, expensive) per-block matched-filter result -- see the
// Python class's docstring for why a single timescale can't serve every
// mode well. result() reports whichever timescale's peak score is
// highest. A single-element vector (the default) is one timescale.
class BlindAccumulator {
   public:
    explicit BlindAccumulator(double max_offset_hz = config::BLIND_MAX_OFFSET_HZ,
                              double bin_step_hz = config::BLIND_BIN_STEP_HZ,
                              int min_periods = 8, double threshold = config::BLIND_SCORE_THRESHOLD,
                              std::optional<int> block_samples = std::nullopt,
                              std::vector<std::optional<double>> window_s = {25.0});

    void push(std::span<const cdouble> z, std::int64_t start_sample);
    BlindAcquisition result() const;
    // The current best prominence whether or not it clears the
    // threshold; 0.0 while too little has been pushed. Observability
    // for the live loop's status surfaces -- result() stays the only
    // lock gate. See the .cpp.
    double best_score() const;

   private:
    int m_;
    int min_periods_;
    double threshold_;
    int block_;
    int step_;
    std::vector<double> decay_per_block_;  // one per timescale
    std::vector<int> shift_bins_;
    std::vector<double> freqs_;
    std::vector<cdouble> kernel_f_;

    int n_bins_;
    int n_scales_;
    std::vector<double> folded_;  // n_scales_ x n_bins_ x FRAME_SAMPLES, row-major
    std::int64_t n_valid_ = 0;
    std::vector<cdouble> buf_;
    std::int64_t buf_start_ = -1;  // -1 until the first push()
};

}  // namespace sstvae::sync
