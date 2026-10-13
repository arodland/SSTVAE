// The beacon file: one segment of a stored picture, ready for a clock-chip
// beacon (design 2.9, spec 2.5 "Storing a picture"). Port of
// sstvae/qrss/beaconfile.py (the reference).
//
// A format module: a microcontroller reads this layout directly, so it is
// fixed byte for byte. Little-endian throughout.
//
//     offset  size  field
//          0     4  magic b"QRSB"
//          4     1  file version = 1
//          5     1  waveform (0 = CE)
//          6     1  mode
//          7     1  segment
//          8     4  picture ID (u32)
//         12     2  codec ID (u16)
//         14     8  callsign, ASCII, space padded
//         22     2  grid code (u16, 32767 = none)
//         24     4  n_latents (u32), 50600
//         28     2  int8 scale (u16), 20
//         30     1  flags: bit 0 = callsign keying (0 FSK, 1 OOK); other bits 0
//         31     1  reserved = 0
//         32    24  callsign keying, 192 units, 1 bit each MSB first (1 = key-down)
//         56   310  header coded bits, 2474 bits MSB first, last 6 pad bits 0
//        366     n  int8 latents in air order, neither scrambled nor precoded
//      366+n     4  CRC-32 (zlib) of every preceding byte
//
// A full segment is 50,970 bytes. qrss_beacon_c/ (the C reference beacon)
// reads the same bytes.
//
// Every refusal is std::invalid_argument with the reference's ValueError
// text, in the reference's order; a callsign the Morse table refuses is
// morse::CallsignError (an std::invalid_argument) with morse.py's text, and
// a header field header.encode refuses carries header.py's text.

#ifndef SSTVAE_QRSS_BEACONFILE_HPP
#define SSTVAE_QRSS_BEACONFILE_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "qrss/config.hpp"
#include "qrss/header.hpp"
#include "qrss/picture.hpp"

namespace sstvae::qrss::beaconfile {

// beaconfile.MAGIC, FILE_VERSION, FLAG_OOK and the layout's sizes.
inline constexpr std::string_view MAGIC = config::BEACON_MAGIC;
inline constexpr int FILE_VERSION = config::BEACON_FILE_VERSION;
inline constexpr int FLAG_OOK = config::BEACON_FLAG_OOK;
inline constexpr std::size_t HEAD_BYTES = config::BEACON_HEAD_BYTES;  // struct "<4sBBBBIH8sHIHBB"
inline constexpr std::size_t KEYING_BYTES = config::CW_UNITS / 8;                // 24
inline constexpr std::size_t HDR_BYTES = (config::N_HDR_BITS + 7) / 8;           // 310
inline constexpr std::size_t LATENT_OFFSET = HEAD_BYTES + KEYING_BYTES + HDR_BYTES;
static_assert(LATENT_OFFSET == 366, "the latents start at byte 366");
static_assert(config::CW_UNITS % 8 == 0, "the keying mask is whole bytes");

// beaconfile.file_size: bytes in a beacon file of n latents (50,970 for a
// full segment). std::overflow_error past int64 (Python's int has no limit).
std::int64_t file_size(std::int64_t n_latents = picture::SENT);

// beaconfile.BeaconFile: one segment as a beacon sends it. The integers are
// int64 so that a wrong value can be held and refused with its own value in
// the message. `keying` and `hdr_bits` hold 0/1; any other value (a caller
// handing in something that is not a bit) never equals a bit and is refused
// by check() as a mismatch.
struct BeaconFile {
    std::int64_t waveform = config::WAVEFORM_CE;
    std::int64_t mode = 0;
    std::int64_t segment = 0;
    std::int64_t picture_id = 0;
    std::int64_t codec_id = 0;
    std::string callsign;
    std::optional<std::string> grid;
    bool ook = false;
    std::vector<std::uint8_t> keying;      // CW_UNITS
    std::vector<std::uint8_t> hdr_bits;    // N_HDR_BITS
    std::vector<std::int8_t> latents_i8;   // air order

    // The header fields this segment is sent with (version FORMAT_VERSION).
    header::HeaderFields header_fields() const;
    // BeaconFile.check: std::invalid_argument unless every field agrees with
    // every other -- waveform CE, a sendable callsign without trailing
    // spaces, the keying spelling it, the header bits encoding the fields,
    // 1..SENT latents and none of them -128.
    void check() const;
    // BeaconFile.to_bytes: the file, after check().
    std::vector<std::uint8_t> to_bytes() const;
    // BeaconFile.from_bytes: read and verify a file's bytes (magic, CRC-32,
    // version, scale, flags, size, an ASCII callsign, zero pad bits, the
    // grid code, then check()).
    static BeaconFile from_bytes(std::span<const std::uint8_t> data);
};

// The refusal of a header that does not describe this segment of this
// picture: "header (picture_id, mode, segment, codec_id) = (...) does not
// match the picture's (...)" (beaconfile.from_picture, tx._segment_and_header).
void check_header_matches(const picture::StoredPicture& sp, std::int64_t segment, const header::HeaderFields& h);

// beaconfile.from_picture: the beacon file for one segment of a stored
// picture. `h` supplies the callsign and grid; its picture ID, mode,
// segment and codec ID must match the picture's.
BeaconFile from_picture(const picture::StoredPicture& sp, std::int64_t segment, const header::HeaderFields& h,
                        bool ook = false);

// beaconfile.write / read (std::runtime_error on I/O failure). As the
// reference, write opens (and truncates) the file before it serialises,
// so a refused BeaconFile leaves an empty file behind.
void write(const std::string& path, const BeaconFile& bf);
BeaconFile read(const std::string& path);

// Python's repr() of bytes (b'QRSX', b'\x00\x01'), for the refusal texts.
std::string bytes_repr(std::span<const std::uint8_t> b);

}  // namespace sstvae::qrss::beaconfile

#endif  // SSTVAE_QRSS_BEACONFILE_HPP
