// The narrow channel's front-end half from sstvae/qrss/track.py: make_chan
// (FE' -> one signal's 250 Hz channel) and ChanCapture's noise_psd_at and
// noise_floor, which fill the cache slot trackbase.hpp put on ChanCapture.
//
// Not a format module: nothing here goes on the air.
//
// Agreement with the Python reference: make_chan is channelise and
// channelise_keep (frontend.hpp) on the same zero-padded segment, so it
// agrees as they do (to the bit in practice). The noise table is
// noise_psd's (rtol ~1e-15, the bin choice exact) with the reference's
// window centres, padding fractions and fill-in, in its order and dtypes
// (a float32 median for the fill, float64 after); noise_psd_at is
// np.interp over it (math::interp, bit-exact).
//
// The noise PSD is a seam (NoisePsdFn): the default is frontend::noise_psd,
// and a caller that has replaced it -- the Python suite spies on
// frontend.noise_psd's arguments -- passes its own, called with exactly the
// reference's arguments (ch, fs, exclude_hz relative to the mix, win_s =
// NOISE_WIN_S, audio_hz = the mix's audio frequency).
//
// Threads: noise_psd_at and noise_floor may be called on one ChanCapture
// from several threads at once; the table is computed once (NoisePsdSlot).

#ifndef SSTVAE_QRSS_CHAN_HPP
#define SSTVAE_QRSS_CHAN_HPP

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "qrss/rx_config.hpp"
#include "qrss/trackbase.hpp"

namespace sstvae::qrss::track {

inline constexpr double NOISE_WIN_S = rxconfig::track::NOISE_WIN_S;

// Channelise FE' (and the blanker's keep) about f_audio_hz over receiver
// times [t_lo_s, t_hi_s] after nominal t0; zero-padded where the capture
// does not reach. The CH grid is FE index 16k (aligned with the stream):
// the segment starts at floor(t0_index_fe + t_lo_s fs_fe) rounded down to
// a multiple of fs_fe / CH_FS, and its length is a multiple of that too.
// Errors are the reference's: ZeroDivision "integer modulo by zero" for
// fs_fe < CH_FS, std::invalid_argument "negative dimensions are not
// allowed" for t_hi_s far enough below t_lo_s, "cannot convert float NaN to
// integer" (std::invalid_argument) and "cannot convert float infinity to
// integer" (std::overflow_error) for a non-finite time, and channelise's
// for a rate it refuses.
ChanCapture make_chan(std::span<const std::complex<float>> fe, std::optional<std::span<const float>> keep,
                      double t0_index_fe, std::int64_t fs_fe, double f_audio_hz, double t_lo_s, double t_hi_s,
                      std::span<const double> exclude_hz = {});

// frontend.noise_psd as ChanCapture.noise_psd_at calls it:
// fn(ch, fs, exclude_hz, win_s, audio_hz) -> float32 PSD per window.
using NoisePsdFn = std::function<std::vector<float>(std::span<const std::complex<float>> ch, std::int64_t fs,
                                                    std::span<const double> exclude_hz, double win_s,
                                                    std::optional<double> audio_hz)>;

// The default NoisePsdFn: frontend::noise_psd with audio_hz (and the
// environment's rules, as the reference reads them).
std::vector<float> default_noise_psd(std::span<const std::complex<float>> ch, std::int64_t fs,
                                     std::span<const double> exclude_hz, double win_s,
                                     std::optional<double> audio_hz);

// The table behind noise_psd_at: (window-centre receiver times, PSD),
// computed on first use and kept in c.psd (an empty fn is
// default_noise_psd). Windows that are mostly zero padding (<= 90% of
// samples nonzero) or read a PSD <= 0 take the median of the rest (float32,
// as numpy's); with no good window at all, every window is
// max(median, 1e-30) (NaN stays NaN).
std::shared_ptr<const NoisePsd> noise_psd_table(const ChanCapture& c, const NoisePsdFn& fn = {});

// Noise PSD (CH power per Hz) at receiver times t_s: np.interp over the
// table. std::invalid_argument "array of sample points is empty" when the
// capture gave no window (the table is cached all the same, as in Python).
double noise_psd_at(const ChanCapture& c, double t_s, const NoisePsdFn& fn = {});
std::vector<double> noise_psd_at(const ChanCapture& c, std::span<const double> t_s, const NoisePsdFn& fn = {});

// Median noise PSD over the capture (np.median of the table).
double noise_floor(const ChanCapture& c, const NoisePsdFn& fn = {});

}  // namespace sstvae::qrss::track

#endif  // SSTVAE_QRSS_CHAN_HPP
