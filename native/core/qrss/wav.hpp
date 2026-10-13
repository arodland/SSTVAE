// Mono WAV files byte for byte as qrss_transmit.py writes them: through
// sstvae.wavio, which hands the samples to scipy.io.wavfile.write.
//
// Two forms, the two the CLI writes:
//
// - int16, peak-normalised (wavio.write_wav): m = max|x|, then every
//   sample np.round(x / m * peak * 32767) as numpy evaluates it --
//   ((x / m) * peak) * 32767 in float64, rounded half to even -- with no
//   scaling when m is 0. A 16-byte PCM fmt chunk.
// - float32, untouched (wavio.write_wav_float): each sample rounded once
//   to float32. An 18-byte IEEE-float fmt chunk (cbSize 0) and a fact
//   chunk holding the sample count, as scipy writes for a float dtype.
//
// This is not SSTVAE's core/audio/wavio: that one goes through dr_wav,
// whose float header has a 16-byte fmt chunk and no fact chunk, so its
// --float files are not scipy's bytes; and it takes the whole signal at
// once, where a 30-minute pass is meant to stream (tx::PassAudio). The
// int16 form is the same 44-byte header either way.
//
// The header is known before any sample is (scipy back-patches only the
// RIFF size, which a known count gives up front), so a Writer streams:
// header, then samples in any blocks, then close(). A file past 4 GiB is
// RF64 with a ds64 chunk, as scipy writes it (out of reach at 8 kHz:
// 74 hours of int16).

#ifndef SSTVAE_QRSS_WAV_HPP
#define SSTVAE_QRSS_WAV_HPP

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/config.hpp"

namespace sstvae::qrss::wav {

enum class Format { Int16, Float32 };

// A file that could not be opened or written: the path and errno, which a
// binding turns into Python's own OSError (FileNotFoundError, ...).
struct WriteError : std::runtime_error {
    WriteError(const std::string& path, int err);
    std::string path;
    int err;
};

// scipy.io.wavfile.write's bytes before the samples, for n mono samples at
// `rate`.
std::vector<std::uint8_t> header(Format format, std::int64_t n, std::int64_t rate = config::FS);

// np.max(np.abs(x)): the peak write_wav normalises by (NaN if any is NaN).
double peak_of(std::span<const double> x);
// Fold a block into a running peak (start from 0.0): the same maximum.
double peak_of(std::span<const double> x, double running);

// write_wav's sample: np.round((x / m * peak) * 32767).astype(int16), the
// division and the scaling skipped when m is not > 0. NaN is 0 (numpy's
// cast of it is unspecified).
std::int16_t to_int16(double x, double m, double peak = 0.95);

class Writer {
public:
    // Opens `path` for writing and writes the header for n samples.
    Writer(const std::string& path, Format format, std::int64_t n, std::int64_t rate = config::FS);
    ~Writer();
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    Format format() const { return format_; }
    // Samples, in order; Int16 takes int16, Float32 float32 (each its own
    // format only: std::logic_error otherwise, or past n samples).
    void write(std::span<const std::int16_t> x);
    void write(std::span<const float> x);
    // Flushes and closes; std::logic_error unless exactly n were written.
    void close();

private:
    void put(const void* p, std::size_t bytes);
    std::string path_;
    Format format_;
    std::int64_t n_ = 0, written_ = 0;
    std::FILE* f_ = nullptr;
};

// wavio.write_wav(path, x, peak) and wavio.write_wav_float(path, x).
void write_int16(const std::string& path, std::span<const double> x, double peak = 0.95);
void write_float32(const std::string& path, std::span<const double> x);

}  // namespace sstvae::qrss::wav

#endif  // SSTVAE_QRSS_WAV_HPP
