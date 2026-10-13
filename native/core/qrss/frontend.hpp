// The QRSSTVAE receiver's front end: sstvae/qrss/frontend.py without the
// passband store (stage 3), plus live.StreamFE (audio_to_fe over a stream
// of chunks).
//
// Not a format module: nothing here goes on the air, and nothing here
// draws random numbers either.
//
//   8 kHz audio --audio_to_fe--> FE 4000 Hz complex, 0 Hz = 1500 Hz audio
//   FE --blank, normalise--> FE'          (impulses out, receive AGC undone)
//   FE' --channelise(f_mix)--> CH 250 Hz complex, +-125 Hz about one signal
//
// Agreement with the Python reference (numpy 2.5, scipy 1.18):
//
// * blank, normalise, running_median, interp_to_samples, quantise_mix,
//   mix_phasor and the noise-floor masks (_fe_flat, and with them the
//   choice of noise bins) are numpy's arithmetic in numpy's order and agree
//   to the bit: float32 where numpy is float32 (|fe|^2, the blanker's
//   threshold, the 0.5 (a + b) median), numpy's fused complex multiply and
//   reciprocal-multiply complex/real division (math/npcomplex.hpp), libm
//   cos/sin, half-to-even rounding and floor division where Python has them.
// * audio_to_fe and fe_to_audio filter through scipy's oaconvolve
//   (math/convolve.hpp: same blocks, pocketfft for ducc0), so they agree to
//   ~1e-16 of the signal before the complex64 cast -- a rare 1-ulp
//   difference after it (none at all over a FULL slot's 3.68M samples,
//   measured).
// * channelise mixes with numpy's fused multiply and decimates through
//   filters::resample_poly (scipy's loop): to the bit in practice.
// * noise_psd's FFT is pocketfft where numpy's is pocketfft too: to
//   rounding (rtol ~1e-15) on the PSD, the bin choices exact.
//
// Errors are the reference's: std::invalid_argument for ValueError,
// ZeroDivision (rational.hpp) for ZeroDivisionError, std::overflow_error for
// OverflowError, std::runtime_error for scipy's RuntimeError.
//
// Threads: every function is safe to call from several threads at once.
// Python's lru_caches (_noise_response, _fold_power, _fe_power, _fe_flat,
// _fe_response) are mutex-guarded LRU caches of immutable results here.

#ifndef SSTVAE_QRSS_FRONTEND_HPP
#define SSTVAE_QRSS_FRONTEND_HPP

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "qrss/config.hpp"
#include "qrss/rational.hpp"
#include "qrss/rx_config.hpp"

namespace sstvae::qrss::frontend {

using cf = std::complex<float>;
using cd = std::complex<double>;

inline constexpr int FE_DECIM = rxconfig::frontend::FE_DECIM_;   // FS // FE_FS
inline constexpr double FE_FLAT_DB = rxconfig::frontend::FE_FLAT_DB;
inline constexpr double FE_MIN_HZ = rxconfig::frontend::FE_MIN_HZ;
inline constexpr double FE_SKIRT_DB = rxconfig::frontend::FE_SKIRT_DB;
inline constexpr int F_MIX_DEN = rxconfig::frontend::F_MIX_DEN;
static_assert(FE_DECIM == config::FS / config::FE_FS, "the front end halves the audio rate");

// live.py's StreamFE block rules: audio blocks start on multiples of ALIGN
// (to_baseband's 16-sample phasor table, FCENTER/FS = 3/16) and carry HALO
// samples of context each side (the FE FIR is 255 taps).
inline constexpr std::int64_t ALIGN = rxconfig::live::ALIGN;
inline constexpr std::int64_t HALO = rxconfig::live::HALO;

// _FE_TAPS: signal.firwin(255, 1250.0, fs=8000, window=("kaiser", 8.0)),
// through filters::firwin (bitwise scipy's; computed once).
const std::vector<double>& fe_taps();

// --- audio <-> FE -----------------------------------------------------------

// 8 kHz real audio -> complex64 FE at 4 kHz, 0 Hz = 1500 Hz audio:
// sqrt(2) to_baseband(x), oaconvolve with the FE taps ('same'), every
// second sample.
std::vector<cf> audio_to_fe(std::span<const double> x8k);

// The inverse: upsample by 2 (times 2), the FE taps, mix up to 1500 Hz,
// sqrt(2) Re.
std::vector<double> fe_to_audio(std::span<const cd> fe);

// Audio frequency -> frequency inside the FE stream (0 Hz = 1500 Hz).
inline double fe_hz(double audio_hz) { return audio_hz - config::FE_CENTER_HZ; }

// A front-end capture of one slot. t0_index: FE sample index of the
// nominal t0 (QH + 1 s) on the receiver's clock; fractional allowed.
struct Capture {
    std::vector<cf> fe;
    std::int64_t q = 0;
    double t0_index = 0.0;
    int fs = config::FE_FS;
};

// --- blanker and inverse AGC ------------------------------------------------

// (centres, medians): float32 median of p over win_s windows every hop_s,
// windows shifted inward to stay whole (or all of p when it is shorter).
struct RunningMedian {
    std::vector<std::int64_t> centres;
    std::vector<float> medians;
};
RunningMedian running_median(std::span<const float> p, double fs, double win_s = 1.0, double hop_s = 0.05);

// _interp_to_samples: np.interp(arange(n), centres, vals) as float32.
// std::invalid_argument "array of sample points is empty" for no centres
// and n > 0 (np.interp's refusal, which blank reaches for 1 <= n <= 100 at
// FE_FS -- running_median gives such a stream no centre).
std::vector<float> interp_to_samples(std::int64_t n, std::span<const double> centres, std::span<const double> vals);

// (fe * keep, keep): impulses blanked sample by sample (design 6.2).
struct Blanked {
    std::vector<cf> fe;
    std::vector<float> keep;
};
Blanked blank(std::span<const cf> fe, double k_sigma = 5.0, double guard_ms = 2.0, double win_s = 1.0,
              double hop_s = 0.05, double fs = config::FE_FS, double taper_ms = 1.0);

// (fe / sqrt(level), level): the inverse AGC (design 6.2).
struct Normalised {
    std::vector<cf> fe;
    std::vector<float> level;
};
Normalised normalise(std::span<const cf> fe, std::optional<std::span<const float>> keep = std::nullopt,
                     double win_ms = 100.0, double fs = config::FE_FS, double hop_ms = 10.0,
                     double min_kept = 0.25);

// --- channeliser ------------------------------------------------------------

// f_mix on the channeliser's 1/64 Hz grid: Fraction(round(f * 64), 64).
// std::invalid_argument "cannot convert float NaN to integer",
// std::overflow_error "cannot convert float infinity to integer".
Rational quantise_mix(double f_mix_hz);

// exp(-j 2 pi f_mix (n0 + i)/fs_in), i < n, from exact integer turns.
// std::invalid_argument "f_mix <f> is not on the 1/64 Hz grid".
std::vector<cd> mix_phasor(std::int64_t n0, std::int64_t n, const Rational& f_mix, std::int64_t fs_in);

// (ch complex64, f_mix): fe mixed down by quantise_mix(f_mix_hz) (Hz inside
// fe) and decimated to fs_out by resample_poly(., 1, fs_in/fs_out,
// window=("kaiser", 10)). Sample i of fe is stream sample n0 + i.
// std::invalid_argument "fs_out <o> must divide fs_in <i>".
struct Channelised {
    std::vector<cf> ch;
    Rational f_mix;
};
Channelised channelise(std::span<const cd> fe, std::int64_t fs_in, double f_mix_hz,
                       std::int64_t fs_out = config::CH_FS, std::int64_t n0 = 0);
Channelised channelise(std::span<const cf> fe, std::int64_t fs_in, double f_mix_hz,
                       std::int64_t fs_out = config::CH_FS, std::int64_t n0 = 0);

// float32: the blanker's keep through the channeliser's decimation (no
// mixing), aligned with channelise's output.
std::vector<float> channelise_keep(std::span<const double> keep, std::int64_t fs_in = config::FE_FS,
                                   std::int64_t fs_out = config::CH_FS);
std::vector<float> channelise_keep(std::span<const float> keep, std::int64_t fs_in = config::FE_FS,
                                   std::int64_t fs_out = config::CH_FS);

// --- the noise floor --------------------------------------------------------

// The noise band either side of the carrier, Hz (noise_psd's band_hz).
using Band = std::pair<double, double>;
inline constexpr Band NOISE_BAND{85.0, 120.0};

// _noise_response: the channeliser's aliased power response sum_k
// |H(f + k fs_out)|^2 on the fftfreq(nfft, 1/fs_out) grid (ones when
// fs_in == fs_out).
std::shared_ptr<const std::vector<double>> noise_response(std::int64_t fs_in, std::int64_t fs_out,
                                                          std::int64_t nfft);

// _fold_power: the band's bins idx, the FE frequencies fk[k][j] the k'th
// fold of bin idx[j] comes from, and the decimation filter's |H(fk)|^2
// there. fk and P are n_fold x idx.size(), row-major.
struct FoldPower {
    std::vector<std::int64_t> idx;
    std::size_t n_fold = 0;
    std::vector<double> fk;
    std::vector<double> P;
};
std::shared_ptr<const FoldPower> fold_power(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft,
                                            Band band_hz);

// _fe_power: (idx, num, den): white FE-input noise through the front end's
// low-pass and the decimation (num) and through the decimation alone (den).
struct FePower {
    std::vector<std::int64_t> idx;
    std::vector<double> num;
    std::vector<double> den;
};
std::shared_ptr<const FePower> fe_power(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, double audio_hz,
                                        Band band_hz);

// _fe_flat: bins of band_hz the front end leaves within FE_FLAT_DB of white
// (0/1 per bin of the fftfreq(nfft, 1/fs_out) grid; 0 outside the band).
std::vector<std::uint8_t> fe_flat(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, double audio_hz,
                                  Band band_hz = NOISE_BAND);

// _fe_response: num/den on band_hz's bins, 1 elsewhere.
std::vector<double> fe_response(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, double audio_hz,
                                Band band_hz = NOISE_BAND);

// noise_psd's keyword arguments. old_edge_noise nullopt reads the
// environment (rules::old(EDGE_NOISE)) as the reference does.
struct NoisePsdOptions {
    Band band_hz = NOISE_BAND;
    double guard_hz = 45.0;
    std::int64_t fs_in = config::FE_FS;
    std::optional<double> audio_hz;
    std::optional<bool> old_edge_noise;
};

// float32[n_win]: noise PSD (CH power per Hz) in win_s windows (the
// reference's docstring has the rule). std::invalid_argument
// "noise_psd: no bin in the band"; ZeroDivision "integer division or
// modulo by zero" for an empty ch.
std::vector<float> noise_psd(std::span<const cf> ch, std::int64_t fs = config::CH_FS,
                             std::span<const double> exclude_hz = {}, double win_s = 10.0,
                             const NoisePsdOptions& options = {});

// --- live.StreamFE ----------------------------------------------------------

// audio_to_fe over a stream of chunks, sample for sample: audio sample a
// (absolute, on the 8 kHz grid) becomes FE sample a / 2. A block is
// converted with HALO samples of context each side, so the output equals
// the batch conversion away from the stream's first and last HALO samples,
// and blocks start on multiples of ALIGN. push returns the FE produced as
// (first FE index, samples); a chunk that does not follow on from the last
// one restarts the stream. One stream per object; not for concurrent use.
class StreamFE {
public:
    explicit StreamFE(std::int64_t block = 8000);

    struct Chunk {
        std::int64_t index;
        std::vector<cf> fe;
    };

    std::int64_t block() const { return block_; }
    // The audio index the next chunk must start at to follow on (nullopt
    // before the first push).
    std::optional<std::int64_t> next_index() const;
    // Absolute audio index of the first pending sample, and the pending audio.
    std::optional<std::int64_t> a0() const { return a0_; }
    const std::vector<double>& pending() const { return pend_; }

    std::vector<Chunk> push(std::int64_t a, std::span<const double> x);

private:
    std::int64_t block_;
    std::optional<std::int64_t> a0_;
    std::vector<double> pend_;
};

}  // namespace sstvae::qrss::frontend

#endif  // SSTVAE_QRSS_FRONTEND_HPP
