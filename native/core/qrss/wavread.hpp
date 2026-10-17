// sstvae.wavio.read_wav as the QRSS receiver's capture_from_wav reads a
// WAV: scipy.io.wavfile.read's samples, integers divided by their dtype's
// maximum in float64 (np.iinfo(dtype).max: 255 for 8-bit, which scipy
// reads unsigned, 127 for the int8 scipy reads a 1-byte container of more
// than 8 bits as, 32767, 2147483647 for 16- and 32-bit and for 24-bit,
// which scipy left-justifies in int32, 2**63 - 1 for 40- to 64-bit, which
// it left-justifies in int64), floats widened, channels averaged (numpy's
// mean over axis 1: each frame's pairwise sum, then / channels), and
// resample_poly(x, FS / g, rate / g) when the rate is not FS.
//
// This is not core/audio/wavio's read_wav: that one goes through dr_wav,
// which scales int16 by 1/32768 in float32 where the reference divides by
// 32767 in float64, so a capture read through it is not the reference's
// to the bit.
//
// The file is walked the way scipy.io.wavfile.read (scipy 1.16) walks it,
// quirks included, because the reference is whatever scipy returns: RIFF,
// RIFX (big-endian) and RF64; chunks read while the position is before the
// RIFF size + 8, so a header whose size field is too small ends the walk
// early; every chunk after the data chunk read too, the last data chunk
// being the one returned, decoded with the fmt chunk current at it, and
// the rate the last fmt chunk's; the position after a data chunk where
// numpy's fromfile leaves it (only the whole elements it read, so a data
// size that is not a multiple of the container, or an 8-bit format in a
// wider container, which scipy reads as that many contiguous bytes,
// misaligns the walk exactly as it misaligns scipy's); the PCM byte-rate
// check; WAVE_FORMAT_EXTENSIBLE's subformat taken only from a GUID with
// the standard tail; the sample type from the container size
// (nBlockAlign // nChannels), so a "32-bit float" in an 8-byte container
// is float64 and one in a 2-byte container float16.
//
// A file scipy (or numpy, or read_wav's resample_poly) refuses is
// FormatError, carrying the Python exception class the reference raises
// and its text: ValueError (scipy's own texts, "Unknown wave file format:
// MULAW. Supported formats: PCM, IEEE_FLOAT", numpy's "cannot reshape
// ..." and resample_poly's "up and down must be >= 1"), struct.error
// ("unpack requires a buffer of 4 bytes"), ZeroDivisionError (no channels,
// or a block align under the channel count), TypeError (numpy's "data type
// '<i3' not understood" for a container numpy has no type for), and
// UnboundLocalError (a walk that ends with no fmt or no data chunk:
// "cannot access local variable 'fs' where it is not associated with a
// value"). The warnings the reference gives are handed to the caller in
// order, for a binding to warn with: scipy's WavFileWarnings (an unknown
// chunk, a file shorter than its header says), then numpy's
// RuntimeWarnings as numpy's floating-point flags raise them (float32
// samples holding a signalling NaN: "invalid value encountered in cast";
// the channel mean overflowing or adding infinities of both signs or a
// signalling NaN: "overflow encountered in reduce", then "invalid value
// encountered in reduce"; a 16-byte float container read as numpy's long
// double: "overflow encountered in cast" past float64's range, "invalid
// value encountered in cast" for a signalling NaN or an encoding the x87
// does not take). A 64-bit float in a 16-byte container is read as numpy
// reads it: as the platform's long double where that is 16 bytes,
// TypeError ("data type '<f16' not understood") where it is not (Windows,
// macOS arm64), as numpy refuses it there.
//
// Not as the reference: a resampling factor past what resample_poly's int
// arithmetic holds (std::bad_alloc, where scipy designs a filter of
// billions of taps and runs out of memory or time); a data size numpy
// cannot allocate up front (np.fromfile allocates the declared count
// before reading: MemoryError for an RF64 data size of a terabyte, where
// the C++ reads what is there).

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

// A warning the reference gives reading the file: scipy's WavFileWarning
// or numpy's RuntimeWarning, and its text.
struct Warning {
    enum class Kind { wav_file, runtime };
    Kind kind;
    std::string text;
    bool operator==(const Warning&) const = default;
};

// A file the reference refuses: what() is its text, `kind` the Python
// exception class it raises it as.
struct FormatError : std::invalid_argument {
    enum class Kind { value, struct_error, zero_division, type, unbound_local };
    FormatError(Kind kind, const std::string& what);
    Kind kind;
};

// wavio.read_wav(path): float64 mono at config::FS. `warnings`, when
// given, receives the reference's warnings in its order (filled before a
// FormatError is thrown, too). A path holding a NUL
// is FormatError(value, "embedded null byte"), as Python's open() refuses
// it (fopen would open the path's prefix).
std::vector<double> read_wav(const std::string& path, std::vector<Warning>* warnings = nullptr);

}  // namespace sstvae::qrss::wav

#endif  // SSTVAE_QRSS_WAVREAD_HPP
