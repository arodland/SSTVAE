// 8 kHz mono WAV read/write -- the counterpart of `sstvae/wavio.py`.
//
// Lives under core/audio/ rather than beside the modem because it is
// I/O, not DSP: the modem is handed samples and never learns where they
// came from.

#ifndef SSTVAE_AUDIO_WAVIO_HPP
#define SSTVAE_AUDIO_WAVIO_HPP

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sstvae::audio {

// Read a WAV as float64 mono at FS, resampling if needed.
//
// Two details are load-bearing and are the reference's, not choices:
//
//  - **Integer samples are scaled before the stereo mixdown.** Doing it
//    after means testing the dtype of something `mean` has already
//    turned into a float, which silently skipped normalization for
//    every stereo integer file. It decoded anyway -- the modem is
//    scale-invariant enough -- which is why the bug survived.
//  - Rate conversion is `resample_poly`, whole-signal. A file is not a
//    stream, so the stateful path does not apply here.
std::vector<double> read_wav(const std::string& path);

// Write float32 at FS with **no normalization and no quantization**.
// For diagnostics, where the question is "what exactly did we capture?"
// -- `write_wav` rescales and rounds, both of which destroy that
// evidence. `read_wav` returns these unchanged, so a dump round-trips.
void write_wav_float(const std::string& path, std::span<const double> x);

// Write int16 at FS, rescaled to `peak`.
void write_wav(const std::string& path, std::span<const double> x, double peak = 0.95);

}  // namespace sstvae::audio

#endif
