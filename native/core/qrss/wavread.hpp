// sstvae.wavio.read_wav as the QRSS receiver's capture_from_wav reads a
// WAV: scipy.io.wavfile.read's samples, integers divided by their dtype's
// maximum in float64 (np.iinfo(dtype).max: 255 for 8-bit, which scipy
// reads unsigned, 32767, 2147483647 for 16- and 32-bit and for 24-bit,
// which scipy left-justifies in int32), floats widened, channels averaged
// (numpy's mean over axis 1: each frame's pairwise sum, then / channels),
// and
// resample_poly(x, FS / g, rate / g) when the rate is not FS.
//
// This is not core/audio/wavio's read_wav: that one goes through dr_wav,
// which scales int16 by 1/32768 in float32 where the reference divides by
// 32767 in float64, so a capture read through it is not the reference's
// to the bit.
//
// Formats: RIFF little-endian, PCM 1-32 bits per sample in 1-4 byte
// containers and IEEE float 32/64, plain or WAVE_FORMAT_EXTENSIBLE; other
// chunks (fact, LIST, ...) skipped with their pad byte, as scipy skips
// them. Anything else is std::invalid_argument ("unsupported WAV: ...":
// scipy's own texts are not reproduced). A truncated data chunk gives the
// whole samples there are, as np.fromfile does.

#ifndef SSTVAE_QRSS_WAVREAD_HPP
#define SSTVAE_QRSS_WAVREAD_HPP

#include <stdexcept>
#include <string>
#include <vector>

namespace sstvae::qrss::wav {

// A file that could not be opened or read: the path and errno (a binding
// turns it into Python's own OSError).
struct ReadError : std::runtime_error {
    ReadError(const std::string& path, int err);
    std::string path;
    int err;
};

// wavio.read_wav(path): float64 mono at config::FS.
std::vector<double> read_wav(const std::string& path);

}  // namespace sstvae::qrss::wav

#endif  // SSTVAE_QRSS_WAVREAD_HPP
