// FLAC for the passband store (frontend.PassbandStore's packed hours):
// a small encoder of our own and a reader over the vendored dr_flac.
//
// The Python listener packs a finished hour with soundfile (libsndfile ->
// libFLAC, LPC up to order 8) and reads hours back with soundfile, seeking
// to sample ranges. The C++ must read both its own files and Python's, and
// write files soundfile reads, with no libFLAC (cpp-port-plan question 9):
//
// - The encoder writes 16-bit PCM, one or two channels, a fixed block size
//   (4096), and per subframe the smallest by exact size of CONSTANT (an
//   unwritten stretch of an hour is zeros: most of a partly heard hour),
//   VERBATIM, FIXED orders 0-4 (order chosen by the sum of |residual|, as
//   libFLAC does) and LPC up to order 12, found as libFLAC finds it --
//   tukey(0.5) and partial_tukey(2) windows, autocorrelation,
//   Levinson-Durbin, the orders ranked by the bits their error predicts,
//   coefficients quantized to 12 bits with error feedback -- with the best
//   three by that rank given residuals and the best of those by residual
//   costed exactly (flac.cpp's LPC section says what differs from
//   libFLAC). The residual is partitioned Rice with each partition's
//   parameter searched exactly over every useful value, an escape (raw
//   bits) wherever it is cheaper, partition orders 0-6 and 4- or 5-bit
//   parameters. Two channels try left/right, left/side, right/side and
//   mid/side by estimate and keep the best only when exactly smaller than
//   left/right. STREAMINFO carries the true MD5 of the samples and the
//   real minimum and maximum frame sizes; a SEEKTABLE (a point every 16
//   frames) makes partial reads fast. Frame headers carry a CRC-8, frames
//   a CRC-16 (both checked by every decoder). It is a FLAC "subset"
//   stream (block size <= 4608 and LPC order <= 12 at <= 48 kHz, sample
//   rate and size in every frame header, coefficient precision <= 15 bits,
//   partition order <= 8), and every prediction sum fits 32 bits, the
//   bound libFLAC's encoder keeps so that decoders may predict in 32.
//   Measured on eight simulated passband hours (live.StreamFE's 4 kHz I/Q
//   of channel.py transmissions over band noise at audio levels 0.01-0.3,
//   two through an SSB filter, one heard for 12% of the hour): 0.90-0.97x
//   the size of soundfile's files (libFLAC level 5: LPC to order 8,
//   tukey(0.5) alone), where the FIXED-only encoder this replaced was
//   1.22-1.55x; 0.4-1.5 s of one core per hour (RelWithDebInfo, -O2
//   without -march: the residual's sums go through int16 so that baseline
//   SSE2 vectorizes them) against soundfile's 0.4-0.8 s. Unpredictable
//   audio is the slowest: an hour of white noise (VERBATIM throughout)
//   takes about 1.9 s, Gaussian noise 1.8 s.
// - The reader is dr_flac v0.13.4 (native/third_party/dr_libs), compiled
//   with its CRC checks on, reading through rawfile (positioned reads, a
//   file opened through util/fileio's names), seeking to a sample with the
//   seek table when there is one and dr_flac's CRC-checked binary search
//   when there is not (soundfile's files have none).
//
// Nothing here is a format the project defines -- it is FLAC.
//
// Thread safety: free functions with no shared state; a Reader is one
// open file and one decoder, used by one thread at a time. Opening a
// Reader takes a process-wide mutex for dr_flac's metadata parse, which
// writes the library's CPU-feature globals (on ARM on every open), so
// two threads may open readers at once. encode_file
// blocks for the whole encode (a full 4 kHz stereo hour: up to about 1.5 s
// of one core on passband audio, 1.9 s on white noise, above) and polls
// `cancel` once per frame (a frame: at most about half a millisecond).

#ifndef SSTVAE_QRSS_FLAC_HPP
#define SSTVAE_QRSS_FLAC_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace sstvae::qrss::flac {

struct EncodeOptions {
    int blocksize = 4096;             // 16..65535; the last block may be shorter
    int max_fixed_order = 4;          // 0..4
    int max_lpc_order = 12;           // 0 (no LPC) ..32; a subset stream at <= 48 kHz allows 12
    int qlp_precision = 0;            // coefficient bits: 0 by block size, as libFLAC (12 at 4096), else 5..15
    int lpc_orders_tried = 3;         // predictors (window, order) whose residual is computed, the
                                      // best by libFLAC's estimate first (>= 1); the one whose
                                      // residual estimates smallest is costed exactly
    bool partial_tukey = true;        // libFLAC's partial_tukey(2) windows as well as tukey(0.5)
    int max_partition_order = 6;      // 0..8 (dr_flac's limit and the subset's; 8 finds 0.0002% more, slower)
    bool stereo_decorrelation = true; // try L/S, R/S, M/S as well as L/R
    int seek_every_frames = 16;       // a SEEKTABLE point every this many frames; 0 for none
};

// Pulls `n` interleaved frames (n * channels samples) starting at frame
// `first` into `out`; false to stop the encode (an error of the source's).
using Source = std::function<bool(std::uint64_t first, std::size_t n, std::int16_t* out)>;

struct EncodeStats {
    std::uint64_t bytes = 0;            // file size
    std::uint64_t frames = 0;           // FLAC frames written
    std::array<std::uint64_t, 4> stereo{};   // frames per channel assignment: L/R, L/S, R/S, M/S
    std::array<std::uint64_t, 8> subframes{}; // CONSTANT, VERBATIM, FIXED 0..4, LPC
    std::array<std::uint64_t, 33> lpc_orders{};  // LPC subframes by order (1..32)
    std::array<std::uint8_t, 16> md5{};
};

class Cancelled : public std::runtime_error {
public:
    Cancelled() : std::runtime_error("FLAC encode cancelled") {}
};

// Encodes `total_frames` frames of 16-bit PCM to `path` (created or
// truncated), synced to the device before it returns (fsync /
// FlushFileBuffers: the packer deletes the raw hour once this file is in
// place). std::system_error on an I/O error, std::invalid_argument
// for bad parameters, std::runtime_error when the source fails (or, an
// encoder bug, when a frame written is not the size it was costed at),
// Cancelled when `cancel` becomes true (the partial file is left for the
// caller).
EncodeStats encode_file(const std::string& path, int sample_rate, int channels, std::uint64_t total_frames,
                        const Source& source, const EncodeOptions& opt = {},
                        const std::atomic<bool>* cancel = nullptr);

// The same into memory, from interleaved samples (tests, small files).
std::string encode(std::span<const std::int16_t> interleaved, int channels, int sample_rate,
                   const EncodeOptions& opt = {}, EncodeStats* stats = nullptr);

// CRC-8 (poly 0x07) and CRC-16 (poly 0x8005), MSB first, initial 0: the
// frame header's and the frame's checks.
std::uint8_t crc8(const std::uint8_t* p, std::size_t n);
std::uint16_t crc16(const std::uint8_t* p, std::size_t n);

// MD5 of interleaved 16-bit samples as FLAC defines it (each sample
// little-endian, channels interleaved).
std::array<std::uint8_t, 16> pcm16_md5(std::span<const std::int16_t> interleaved);

struct Info {
    int channels = 0;
    int sample_rate = 0;
    int bits_per_sample = 0;
    std::uint64_t frames = 0;               // total PCM frames (STREAMINFO)
    std::array<std::uint8_t, 16> md5{};     // STREAMINFO's MD5 (all zero: none given)
};

// One open FLAC file, decoded with dr_flac. std::system_error when the
// file cannot be opened, std::runtime_error when it is not FLAC.
class Reader {
public:
    explicit Reader(const std::string& path);
    ~Reader();
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    const Info& info() const;
    // Interleaved int16 frames [first, first + n), clamped to the stream
    // (soundfile's start/stop reading); fewer when the stream ends first.
    // std::runtime_error when a frame fails to decode (a CRC mismatch).
    std::vector<std::int16_t> read(std::uint64_t first, std::uint64_t n);

    struct Impl;   // (flac.cpp; decode() shares it)

private:
    std::unique_ptr<Impl> impl_;
};

// The whole of an in-memory FLAC stream (tests): interleaved int16.
std::vector<std::int16_t> decode(const std::string& bytes, Info* info = nullptr);

namespace detail {

// The encoder's LPC residual kernel (exposed for its test): r[i - order] =
// x[i] - (sum over j of q[j] x[i - 1 - j]) >> shift for i in [order, n),
// and sum |r| into abs_sum. x16, when not null, holds the same samples as
// int16 (a 16-bit channel; q then fits 16 bits). Requires x's bits +
// q's bits + ilog2(order) <= 32 (every sum then fits 32 bits), 1 <= order
// <= 32, 0 <= shift <= 15. False when a residual would not fit 31 bits.
bool lpc_residual(const std::int32_t* x, const std::int16_t* x16, int n, const std::int32_t* q, int order,
                  int shift, std::vector<std::int32_t>& r, std::uint64_t& abs_sum);

}  // namespace detail

}  // namespace sstvae::qrss::flac

#endif  // SSTVAE_QRSS_FLAC_HPP
