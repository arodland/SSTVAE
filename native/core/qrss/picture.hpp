// Stored pictures: segment preparation, the fp16 store, the picture ID,
// int8 beacon values, the air <-> canonical maps and `.qrsp` files (design
// 2.5, spec sections 3-4). Port of sstvae/qrss/picture.py (the reference),
// without codec_id_from_onnx (codec glue, which lives with the codec).
//
// A format module. The transmitter encodes a picture *once* and stores the
// fp16 tensor (StoredPicture, a `.qrsp` file); every pass of it sends those
// values, and the picture ID is a hash of them.
//
// Orders:
//
// - canonical: SSTVAE's flat latent order, group g at [g*52800,
//   (g+1)*52800), each group flattened (channel, h, w) C-order.
// - air: one group's 50,600 sent latents in the order of SSTVAE's frozen
//   interleaver (framing::tx_perm(g), offsets within the group). The other
//   2,200 canonical latents per group are never sent.
//
// Bit-exact against numpy, which is why three things are spelled out:
// prepare_segment's RMS is numpy's pairwise mean over the sent values *in
// air order* (npsum.hpp); the scaled values become float16 in one rounding
// straight from float64 (half.hpp; through float32 would round twice and
// change the ID); and to_int8 rounds half to even (std::nearbyint).
//
// Every refusal is std::invalid_argument carrying the reference's
// ValueError text. Where the text quotes an argument as the caller gave it
// (`{group}`, `{codec_id}`), the optional `shown` replaces the C++'s
// decimal, so the Python binding can quote Python's own str().

#ifndef SSTVAE_QRSS_PICTURE_HPP
#define SSTVAE_QRSS_PICTURE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "framing/interleaver_table.hpp"   // brings SSTVAE's config.hpp
#include "qrss/qrss_config.hpp"
#include "qrss/half.hpp"
#include "qrss/npz.hpp"

namespace sstvae::qrss::picture {

// SSTVAE's latent groups (sstvae/config.py through native/core/config.hpp).
inline constexpr int GROUP_LATENTS = ::sstvae::config::GROUP_LATENTS;            // 52,800
inline constexpr int SENT = ::sstvae::config::TRANSMIT_LATENTS_PER_GROUP;        // 50,600
inline constexpr int N_GROUPS = ::sstvae::config::LATENT_GROUPS;                 // 3
// picture.MODE_NAMES: mode m sends groups 0..m.
inline constexpr auto MODE_NAMES = ::sstvae::qrss::config::MODE_NAMES;
static_assert(MODE_NAMES.size() == static_cast<std::size_t>(N_GROUPS), "one mode per group count");
// picture.QRSP_VERSION.
inline constexpr std::int64_t QRSP_VERSION = ::sstvae::qrss::config::QRSP_VERSION;

using half::Half;
// One group's stored values: float16[GROUP_LATENTS], canonical order.
using Segment = std::vector<Half>;

// --- argument checks (picture._check_group / _check_mode) -------------------

// `group` when 0 <= group < N_GROUPS, else
// "group must be 0..2, not <shown or group>".
std::int64_t check_group(std::int64_t group, const std::optional<std::string>& shown = std::nullopt);
// `mode` when 0 <= mode < N_GROUPS, else "mode must be 0..2 (A..C), not <shown or mode>".
std::int64_t check_mode(std::int64_t mode, const std::optional<std::string>& shown = std::nullopt);

// --- the air <-> canonical maps -----------------------------------------------

// picture.air_to_canonical: the canonical offset within the group of each
// air-order latent (SENT values, the frozen interleaver table itself).
std::span<const std::uint16_t> air_to_canonical(std::int64_t group);
// picture.never_sent: 1 for the group's 2,200 canonical latents never sent,
// 0 for the rest (GROUP_LATENTS values).
std::vector<std::uint8_t> never_sent(std::int64_t group);

// picture.prepare_segment: one group's canonical latents, ready to store
// and send. `full` is the encoder's flat canonical vector (N_GROUPS *
// GROUP_LATENTS values). The never-sent are zeroed, the sent are scaled to
// RMS 1 in float64 and the result is rounded to float16. Refuses a wrong
// length and a group without energy (rms 0, inf or nan).
Segment prepare_segment(std::span<const double> full, std::int64_t group);

// picture.air_values: rows of GROUP_LATENTS canonical values (`row_len` is
// the last axis, refused unless GROUP_LATENTS) -> rows of SENT float64
// values in air order. The length is checked before the group, as in the
// reference.
std::vector<double> air_values(std::span<const Half> seg, std::int64_t group, std::size_t row_len);
std::vector<double> air_values(std::span<const float> seg, std::int64_t group, std::size_t row_len);
std::vector<double> air_values(std::span<const double> seg, std::int64_t group, std::size_t row_len);
inline std::vector<double> air_values(std::span<const Half> seg, std::int64_t group) {
    return air_values(seg, group, seg.size());
}
inline std::vector<double> air_values(std::span<const double> seg, std::int64_t group) {
    return air_values(seg, group, seg.size());
}

// picture.canonical_from_air: rows of SENT air-order values -> rows of
// GROUP_LATENTS canonical values, never-sent 0, in the input's own type
// (float16, float32 and float64 stay what they are; the reference's
// non-floating inputs become float64, which the caller converts first).
std::vector<Half> canonical_from_air(std::span<const Half> air, std::int64_t group, std::size_t row_len);
std::vector<float> canonical_from_air(std::span<const float> air, std::int64_t group, std::size_t row_len);
std::vector<double> canonical_from_air(std::span<const double> air, std::int64_t group, std::size_t row_len);
inline std::vector<double> canonical_from_air(std::span<const double> air, std::int64_t group) {
    return canonical_from_air(air, group, air.size());
}

// --- the picture ID -------------------------------------------------------------

// The refusals picture_id makes before it reads a segment, in its order.
// "codec_id must fit in 16 bits, not <shown or codec_id>"
void check_codec_id(std::int64_t codec_id, const std::optional<std::string>& shown = std::nullopt);
// "mode A stores 1 segments, got 2" (mode already checked)
void check_segment_count(std::int64_t mode, std::size_t n_segs);
// "segment shape (52801,), expected (52800,)": numpy's shape tuple.
void check_segment_shape(std::span<const std::int64_t> shape);

// picture.picture_id: the first 32 bits (big-endian) of SHA-256 over
// uint16_be(codec_id) || uint8(mode) || each of groups 0..mode as float16
// little-endian, canonical 52,800 each (decision D7).
std::uint32_t picture_id(std::int64_t codec_id, std::int64_t mode, std::span<const Segment> segs);

// f"{picture_id:08x}" (a negative value as Python formats it: "-0000005").
std::string id_hex(std::int64_t picture_id);

// --- int8 beacon values ----------------------------------------------------------

// picture.to_int8: clip(rint(20 * a), -127, 127) (round half to even). A
// NaN becomes 0, which is what numpy's cast gives on x86-64 and arm64
// (numpy's own result there is platform-defined).
std::vector<std::int8_t> to_int8(std::span<const double> air);
// picture.from_int8: q / 20 as float64 (a division, as numpy's).
std::vector<double> from_int8(std::span<const std::int8_t> q);
std::vector<double> from_int8(std::span<const double> q);

// --- the stored picture -------------------------------------------------------------

// picture.StoredPicture: a picture encoded once, what every pass of it
// sends. The integers are int64 so that a value a caller gets wrong can be
// held and refused with its own value in the message (as HeaderFields).
struct StoredPicture {
    std::int64_t mode = 0;
    std::int64_t codec_id = 0;
    std::vector<Segment> segs;   // groups 0..mode
    std::int64_t picture_id = 0;

    // StoredPicture.from_latents: the mode's segments from the encoder's
    // flat latents, and their ID.
    static StoredPicture from_latents(std::span<const double> full, std::int64_t codec_id, std::int64_t mode);
    // std::invalid_argument unless the stored ID matches the stored tensor
    // ("picture ID 00000001 does not match the stored latents (d349c671)").
    void check() const;
    std::string id_hex() const { return picture::id_hex(picture_id); }
};

// The refusal of a stored ID that is not the tensor's (StoredPicture.check).
void check_id(std::int64_t stored, std::uint32_t computed);

// --- .qrsp files ---------------------------------------------------------------------

// The .npz members save_qrsp writes, in its order: version, mode, codec_id
// (0-d int64), segs (float16 (mode+1, 52800)), picture_id (0-d int64).
// Checks the picture first.
npz::Npz qrsp_npz(const StoredPicture& sp);
// A whole .qrsp file, byte for byte what np.savez writes.
std::vector<std::uint8_t> qrsp_bytes(const StoredPicture& sp);
// load_qrsp of a file's bytes / of an archive already read; `shown` is the
// path as the refusals quote it ("<path>: .qrsp version 2, expected 1").
// Refuses what the reference refuses, with its texts and in its order: a
// missing member, a version, segs of the wrong dtype or shape, a mode
// outside 0..2, a codec ID outside 16 bits, and an ID that is not the
// tensor's.
StoredPicture parse_qrsp(std::span<const std::uint8_t> file, const std::string& shown);
StoredPicture from_npz(const npz::Npz& z, const std::string& shown);

// picture.save_qrsp / load_qrsp on files (std::runtime_error on I/O
// failure). `shown` defaults to the path.
void save_qrsp(const std::string& path, const StoredPicture& sp);
StoredPicture load_qrsp(const std::string& path, const std::optional<std::string>& shown = std::nullopt);

// numpy's name of an npz dtype ("float16", "int64", "bool"), as the
// refusals print segs.dtype.
std::string_view dtype_name(npz::DType t);

}  // namespace sstvae::qrss::picture

#endif  // SSTVAE_QRSS_PICTURE_HPP
