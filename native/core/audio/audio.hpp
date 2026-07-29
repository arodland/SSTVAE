// Soundcard plumbing that is not the soundcard.
//
// Everything here is Qt-free and device-free on purpose. The audio bugs
// that have actually cost this project time were **not** in the code
// that talks to the driver -- they were in the rate conversion, the
// sample-format conversion, and the device matching around it, and all
// three were found by tests using a fake device. Keeping that logic in a
// library with no Qt in it is what keeps those tests possible; the Qt
// layer in `core/audio/qt/` is then thin enough to be mostly untestable
// without hardware, and mostly not worth testing.
//
// The three properties carried over from the reference, restated because
// they are properties of the problem rather than of any library:
//
//   * **Open at the device's own rate and resample here.** Almost no
//     capture hardware is natively 8 kHz, so asking for 8 kHz does not
//     avoid a resampler -- it delegates to whichever one the audio stack
//     has, and JACK cannot resample at all.
//   * **Capture resampling is stateful.** See `StreamResampler`.
//   * **Capture and playback need inverse ratios.** See
//     `resample_ratio`.

#ifndef SSTVAE_AUDIO_AUDIO_HPP
#define SSTVAE_AUDIO_AUDIO_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config.hpp"

namespace sstvae::audio {

// (up, down) for `dsp::resample_poly` to convert audio at `src_rate`
// into audio at `dst_rate`.
//
// Spelled out with both rates named because the two directions are
// inverses and sharing one "ratio to the device" helper between capture
// and playback silently got playback backwards: a transmission was
// decimated 48000->8000 instead of interpolated 8000->48000, so 32
// seconds of audio went out as 0.9 seconds of noise. It only shows on
// devices that *reject* 8 kHz, so testing against the default device
// proves nothing.
struct Ratio {
    int up;
    int down;
};
Ratio resample_ratio(int src_rate, int dst_rate);

// `dsp::resample_poly` for a stream that arrives in chunks.
//
// **Resampling each captured chunk independently is wrong, and wrong in
// a way that decodes rather than fails.** It is an FIR polyphase filter,
// so an isolated chunk is zero-padded at both ends and every chunk
// boundary gets a transient; at 44.1 kHz into 8 kHz the filter is 8821
// taps against ~186 output samples per chunk. Measured on a real on-air
// recording: **4.7 dB of SNR** and a badly mangled picture, while still
// syncing and reporting 440/440 frames. Per-chunk rounding is the other
// half -- each chunk's output length rounded up independently gained 684
// samples over 66 s, a 0.13% clock error the timing tracker then fights.
//
// This keeps `pad` input samples of context on *each* side of every
// block it emits and only consumes whole multiples of `down`, so the
// output is sample-for-sample what one-shot resampling of the entire
// stream would have produced. Costs `2 * pad` input samples of latency,
// ~20 ms at 44.1 kHz, which nothing here is sensitive to.
//
// Playback does not need this: the whole waveform is in hand, so it is
// resampled once.
class StreamResampler {
public:
    StreamResampler(int up, int down);

    // Feed a chunk, get however much output is ready (often none).
    std::vector<double> operator()(std::span<const double> chunk);

    int up() const { return up_; }
    int down() const { return down_; }
    std::size_t pad() const { return pad_; }

private:
    int up_;
    int down_;
    std::size_t pad_;
    std::vector<double> buf_;
};

// The device sample formats we can convert, named as QtMultimedia names
// them so the Qt layer is a lookup rather than a translation.
enum class SampleFormat { Float, Int16, Int32, UInt8 };

std::optional<SampleFormat> sample_format_from_name(std::string_view name);
const char* sample_format_name(SampleFormat f);
int bytes_per_sample(SampleFormat f);

// Raw interleaved device bytes -> mono double in [-1, 1].
//
// Channels are **mixed down** rather than taking the first, so a device
// that puts the signal only on the right channel still works. A trailing
// partial frame is dropped rather than misaligning every sample after
// it.
std::vector<double> bytes_to_mono(std::span<const std::byte> raw, SampleFormat fmt,
                                  int channels);

// The inverse, for playback: mono double -> interleaved device bytes,
// duplicated across `channels`.
//
// Integer formats scale by `scale - 1` and clip in the integer domain,
// so a sample at exactly +1.0 cannot wrap to full-scale negative. Note
// this is deliberately not the exact inverse of `bytes_to_mono`, which
// divides by `scale`; the asymmetry is the reference's and is what keeps
// both directions free of wraparound.
std::vector<std::byte> mono_to_bytes(std::span<const double> x, SampleFormat fmt,
                                     int channels);

// Index of the device to use, or nothing for "the system default".
//
// Matching is by human-readable description rather than by the backend's
// opaque device id, because the id is not stable across backends and the
// config file has to stay hand-editable. Exact match wins; otherwise a
// *unique* case-insensitive substring match, so a saved "K4 RX A" still
// finds it after the backend decorates the name. An ambiguous substring
// deliberately matches nothing -- silently capturing from the wrong
// radio is worse than falling back to the default and saying so.
std::optional<std::size_t> match_device(std::span<const std::string> descriptions,
                                        std::string_view wanted);

}  // namespace sstvae::audio

#endif
