#include "qrss/picture.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "framing/interleaver_table.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/npsum.hpp"
#include "util/sha256.hpp"

namespace sstvae::qrss::picture {

namespace {

std::string str(std::int64_t v) { return std::to_string(v); }

[[noreturn]] void refuse(const std::string& what) { throw std::invalid_argument(what); }

constexpr auto GL = static_cast<std::size_t>(GROUP_LATENTS);
constexpr auto NS = static_cast<std::size_t>(SENT);
static_assert(framing::N_GROUPS == N_GROUPS && framing::TX_PERM_LEN == SENT, "one interleaver table per group");

double as_double(Half h) { return half::to_double(h); }
double as_double(float v) { return static_cast<double>(v); }
double as_double(double v) { return v; }

std::size_t whole_rows(std::size_t size, std::size_t row_len) {
    if (row_len == 0) return 0;
    if (size % row_len) throw std::invalid_argument("picture: the values are not whole rows of the last axis");
    return size / row_len;
}

template <typename T>
std::vector<double> air_values_of(std::span<const T> seg, std::int64_t group, std::size_t row_len) {
    if (row_len != GL)
        refuse("segment has " + std::to_string(row_len) + " latents, expected " + str(GROUP_LATENTS));
    const auto perm = air_to_canonical(group);
    const std::size_t rows = whole_rows(seg.size(), row_len);
    std::vector<double> out(rows * NS);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t k = 0; k < NS; ++k) out[r * NS + k] = as_double(seg[r * GL + perm[k]]);
    return out;
}

template <typename T>
std::vector<T> canonical_from_air_of(std::span<const T> air, std::int64_t group, std::size_t row_len) {
    if (row_len != NS) refuse("air vector has " + std::to_string(row_len) + " values, expected " + str(SENT));
    const std::size_t rows = whole_rows(air.size(), row_len);
    std::vector<T> out(rows * GL, T{});
    const auto perm = air_to_canonical(group);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t k = 0; k < NS; ++k) out[r * GL + perm[k]] = air[r * NS + k];
    return out;
}

const std::array<std::string_view, 5> QRSP_MEMBERS{"version", "mode", "codec_id", "segs", "picture_id"};

}  // namespace

// --- argument checks ----------------------------------------------------------------

std::int64_t check_group(std::int64_t group, const std::optional<std::string>& shown) {
    if (!(0 <= group && group < N_GROUPS))
        refuse("group must be 0.." + str(N_GROUPS - 1) + ", not " + shown.value_or(str(group)));
    return group;
}

std::int64_t check_mode(std::int64_t mode, const std::optional<std::string>& shown) {
    if (!(0 <= mode && mode < N_GROUPS))
        refuse("mode must be 0.." + str(N_GROUPS - 1) + " (" + std::string(MODE_NAMES.front()) + ".." +
               std::string(MODE_NAMES.back()) + "), not " + shown.value_or(str(mode)));
    return mode;
}

// --- the air <-> canonical maps -------------------------------------------------------

std::span<const std::uint16_t> air_to_canonical(std::int64_t group) {
    return framing::tx_perm(static_cast<int>(check_group(group)));
}

std::vector<std::uint8_t> never_sent(std::int64_t group) {
    std::vector<std::uint8_t> mask(GL, 1);
    for (const std::uint16_t c : air_to_canonical(group)) mask[c] = 0;
    return mask;
}

Segment prepare_segment(std::span<const double> full, std::int64_t group) {
    const auto g = static_cast<std::size_t>(check_group(group));
    if (full.size() != static_cast<std::size_t>(N_GROUPS) * GL)
        refuse("expected " + str(std::int64_t(N_GROUPS) * GROUP_LATENTS) + " latents, got " +
               std::to_string(full.size()));
    std::vector<double> seg(full.begin() + static_cast<std::ptrdiff_t>(g * GL),
                            full.begin() + static_cast<std::ptrdiff_t>((g + 1) * GL));
    const std::vector<std::uint8_t> drop = never_sent(group);
    for (std::size_t i = 0; i < GL; ++i)
        if (drop[i]) seg[i] = 0.0;
    // rms = sqrt(mean(seg[air_to_canonical(g)] ** 2)): the squares in air
    // order, numpy's pairwise mean.
    const auto perm = air_to_canonical(group);
    std::vector<double> sq(NS);
    for (std::size_t k = 0; k < NS; ++k) {
        const double v = seg[perm[k]];
        sq[k] = v * v;
    }
    const double rms = std::sqrt(npsum::mean(std::span<const double>(sq)));
    if (!std::isfinite(rms) || rms == 0.0)
        refuse("group " + std::to_string(g) + " has no energy to normalise (rms " + frame::py_repr(rms) + ")");
    Segment out(GL);
    for (std::size_t i = 0; i < GL; ++i) out[i] = half::to_half(seg[i] / rms);
    return out;
}

std::vector<double> air_values(std::span<const Half> seg, std::int64_t group, std::size_t row_len) {
    return air_values_of(seg, group, row_len);
}
std::vector<double> air_values(std::span<const float> seg, std::int64_t group, std::size_t row_len) {
    return air_values_of(seg, group, row_len);
}
std::vector<double> air_values(std::span<const double> seg, std::int64_t group, std::size_t row_len) {
    return air_values_of(seg, group, row_len);
}

std::vector<Half> canonical_from_air(std::span<const Half> air, std::int64_t group, std::size_t row_len) {
    return canonical_from_air_of(air, group, row_len);
}
std::vector<float> canonical_from_air(std::span<const float> air, std::int64_t group, std::size_t row_len) {
    return canonical_from_air_of(air, group, row_len);
}
std::vector<double> canonical_from_air(std::span<const double> air, std::int64_t group, std::size_t row_len) {
    return canonical_from_air_of(air, group, row_len);
}

// --- the picture ID -------------------------------------------------------------------

void check_codec_id(std::int64_t codec_id, const std::optional<std::string>& shown) {
    if (!(0 <= codec_id && codec_id < (std::int64_t(1) << 16)))
        refuse("codec_id must fit in 16 bits, not " + shown.value_or(str(codec_id)));
}

void check_segment_count(std::int64_t mode, std::size_t n_segs) {
    const std::int64_t m = check_mode(mode);
    if (n_segs != static_cast<std::size_t>(m + 1))
        refuse("mode " + std::string(MODE_NAMES[static_cast<std::size_t>(m)]) + " stores " + str(m + 1) +
               " segments, got " + std::to_string(n_segs));
}

void check_segment_shape(std::span<const std::int64_t> shape) {
    if (!(shape.size() == 1 && shape[0] == GROUP_LATENTS))
        refuse("segment shape " + frame::shape_repr(shape) + ", expected (" + str(GROUP_LATENTS) + ",)");
}

std::uint32_t picture_id(std::int64_t codec_id, std::int64_t mode, std::span<const Segment> segs) {
    const std::int64_t m = check_mode(mode);
    check_codec_id(codec_id);
    check_segment_count(m, segs.size());
    util::Sha256 h;
    const std::uint8_t head[3] = {static_cast<std::uint8_t>((codec_id >> 8) & 0xFF),
                                  static_cast<std::uint8_t>(codec_id & 0xFF), static_cast<std::uint8_t>(m)};
    h.update(std::span<const std::uint8_t>(head, 3));
    std::vector<std::uint8_t> le;
    for (const Segment& s : segs) {
        const std::int64_t shape[1] = {static_cast<std::int64_t>(s.size())};
        check_segment_shape(shape);
        le.resize(2 * s.size());
        for (std::size_t i = 0; i < s.size(); ++i) {   // "<f2": little-endian on every host
            le[2 * i] = static_cast<std::uint8_t>(s[i].bits & 0xFF);
            le[2 * i + 1] = static_cast<std::uint8_t>(s[i].bits >> 8);
        }
        h.update(std::span<const std::uint8_t>(le));
    }
    const util::Sha256Digest d = h.digest();
    return (std::uint32_t(d[0]) << 24) | (std::uint32_t(d[1]) << 16) | (std::uint32_t(d[2]) << 8) | std::uint32_t(d[3]);
}

std::string id_hex(std::int64_t picture_id) {
    // f"{v:08x}": lowercase hex, zero padded to width 8 after any sign.
    const bool neg = picture_id < 0;
    std::uint64_t mag = neg ? std::uint64_t(0) - static_cast<std::uint64_t>(picture_id)
                            : static_cast<std::uint64_t>(picture_id);
    std::string digits;
    do {
        digits.insert(digits.begin(), "0123456789abcdef"[mag & 0xF]);
        mag >>= 4;
    } while (mag);
    const std::size_t width = neg ? 7 : 8;
    if (digits.size() < width) digits.insert(0, width - digits.size(), '0');
    return neg ? "-" + digits : digits;
}

// --- int8 beacon values ------------------------------------------------------------------

std::vector<std::int8_t> to_int8(std::span<const double> air) {
    const double scale = static_cast<double>(config::INT8_SCALE);
    std::vector<std::int8_t> out(air.size());
    for (std::size_t i = 0; i < air.size(); ++i) {
        const double q = std::nearbyint(air[i] * scale);   // np.rint: half to even
        std::int8_t v = 0;                                  // NaN: numpy's cast gives 0 here
        if (q >= 127.0)
            v = 127;
        else if (q <= -127.0)
            v = -127;
        else if (q == q)
            v = static_cast<std::int8_t>(q);
        out[i] = v;
    }
    return out;
}

std::vector<double> from_int8(std::span<const std::int8_t> q) {
    const double scale = static_cast<double>(config::INT8_SCALE);
    std::vector<double> out(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) out[i] = static_cast<double>(q[i]) / scale;
    return out;
}

std::vector<double> from_int8(std::span<const double> q) {
    const double scale = static_cast<double>(config::INT8_SCALE);
    std::vector<double> out(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) out[i] = q[i] / scale;
    return out;
}

// --- the stored picture ---------------------------------------------------------------------

StoredPicture StoredPicture::from_latents(std::span<const double> full, std::int64_t codec_id, std::int64_t mode) {
    const std::int64_t m = check_mode(mode);
    StoredPicture sp;
    sp.mode = m;
    for (std::int64_t g = 0; g <= m; ++g) sp.segs.push_back(prepare_segment(full, g));
    sp.codec_id = codec_id;
    sp.picture_id = picture::picture_id(codec_id, m, sp.segs);
    return sp;
}

void check_id(std::int64_t stored, std::uint32_t computed) {
    if (stored != static_cast<std::int64_t>(computed))
        refuse("picture ID " + id_hex(stored) + " does not match the stored latents (" + id_hex(computed) + ")");
}

void StoredPicture::check() const { check_id(picture_id, picture::picture_id(codec_id, mode, segs)); }

// --- .qrsp files ----------------------------------------------------------------------------

npz::Npz qrsp_npz(const StoredPicture& sp) {
    sp.check();
    std::vector<Half> flat;
    flat.reserve(sp.segs.size() * GL);
    for (const Segment& s : sp.segs) flat.insert(flat.end(), s.begin(), s.end());
    npz::Npz z;
    z.add("version", npz::Array::scalar<std::int64_t>(QRSP_VERSION));
    z.add("mode", npz::Array::scalar<std::int64_t>(sp.mode));
    z.add("codec_id", npz::Array::scalar<std::int64_t>(sp.codec_id));
    z.add("segs", npz::Array::of(flat, std::vector<std::size_t>{sp.segs.size(), GL}));
    z.add("picture_id", npz::Array::scalar<std::int64_t>(sp.picture_id));
    return z;
}

std::vector<std::uint8_t> qrsp_bytes(const StoredPicture& sp) { return npz::npz_bytes(qrsp_npz(sp)); }

StoredPicture from_npz(const npz::Npz& z, const std::string& shown) {
    // The members in the reference's order of reading, each an int() but segs.
    auto member = [&](std::string_view name) -> const npz::Array& {
        if (!z.contains(name))
            refuse(shown + ": not a .qrsp file (missing '" + std::string(name) + " is not a file in the archive')");
        return z.at(name);
    };
    const std::int64_t version = member(QRSP_MEMBERS[0]).to_int64();
    const std::int64_t mode = member(QRSP_MEMBERS[1]).to_int64();
    const std::int64_t codec_id = member(QRSP_MEMBERS[2]).to_int64();
    const npz::Array& segs = member(QRSP_MEMBERS[3]);
    const std::int64_t pid = member(QRSP_MEMBERS[4]).to_int64();
    if (version != QRSP_VERSION)
        refuse(shown + ": .qrsp version " + str(version) + ", expected " + str(QRSP_VERSION));
    const bool fits = segs.dtype == npz::DType::f2 && segs.ndim() == 2 &&
                      mode < std::numeric_limits<std::int64_t>::max() && mode + 1 >= 0 &&
                      segs.shape[0] == static_cast<std::size_t>(mode + 1) && segs.shape[1] == GL;
    if (!fits) {
        std::vector<std::int64_t> shape(segs.shape.begin(), segs.shape.end());
        refuse(shown + ": segs " + std::string(dtype_name(segs.dtype)) + frame::shape_repr(shape) +
               " do not fit mode " + str(mode));
    }
    StoredPicture sp;
    sp.mode = check_mode(mode);
    sp.codec_id = codec_id;
    sp.picture_id = pid;
    const auto values = segs.values<Half>();
    for (std::size_t g = 0; g < segs.shape[0]; ++g)
        sp.segs.emplace_back(values.begin() + static_cast<std::ptrdiff_t>(g * GL),
                             values.begin() + static_cast<std::ptrdiff_t>((g + 1) * GL));
    sp.check();
    return sp;
}

StoredPicture parse_qrsp(std::span<const std::uint8_t> file, const std::string& shown) {
    return from_npz(npz::parse_npz(file), shown);
}

void save_qrsp(const std::string& path, const StoredPicture& sp) {
    const npz::Npz z = qrsp_npz(sp);   // checks the picture before the file is touched
    npz::save_npz(path, z);
}

StoredPicture load_qrsp(const std::string& path, const std::optional<std::string>& shown) {
    return from_npz(npz::load_npz(path), shown.value_or(path));
}

std::string_view dtype_name(npz::DType t) {
    switch (t) {
        case npz::DType::f2: return "float16";
        case npz::DType::f4: return "float32";
        case npz::DType::f8: return "float64";
        case npz::DType::c8: return "complex64";
        case npz::DType::c16: return "complex128";
        case npz::DType::i1: return "int8";
        case npz::DType::u1: return "uint8";
        case npz::DType::i2: return "int16";
        case npz::DType::u2: return "uint16";
        case npz::DType::i4: return "int32";
        case npz::DType::u4: return "uint32";
        case npz::DType::i8: return "int64";
        case npz::DType::u8: return "uint64";
        case npz::DType::b1: return "bool";
    }
    return "?";
}

}  // namespace sstvae::qrss::picture
