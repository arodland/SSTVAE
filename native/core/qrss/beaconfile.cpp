#include "qrss/beaconfile.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/crc32.hpp"
#include "qrss/morse.hpp"
#include "qrss/pytext.hpp"

namespace sstvae::qrss::beaconfile {

namespace {

std::string str(std::int64_t v) { return std::to_string(v); }

[[noreturn]] void refuse(const std::string& what) { throw std::invalid_argument(what); }

constexpr std::size_t N_BITS = config::N_HDR_BITS;
constexpr std::size_t N_UNITS = config::CW_UNITS;

void put_u8(std::vector<std::uint8_t>& out, std::int64_t v) { out.push_back(static_cast<std::uint8_t>(v & 0xFF)); }
void put_le(std::vector<std::uint8_t>& out, std::uint64_t v, int bytes) {
    for (int k = 0; k < bytes; ++k) out.push_back(static_cast<std::uint8_t>((v >> (8 * k)) & 0xFF));
}
std::uint64_t get_le(std::span<const std::uint8_t> d, std::size_t at, int bytes) {
    std::uint64_t v = 0;
    for (int k = bytes - 1; k >= 0; --k) v = (v << 8) | d[at + static_cast<std::size_t>(k)];
    return v;
}

// np.packbits: MSB first, the last byte padded with zero bits.
void pack_bits(std::vector<std::uint8_t>& out, std::span<const std::uint8_t> bits) {
    for (std::size_t i = 0; i < bits.size(); i += 8) {
        std::uint8_t byte = 0;
        for (std::size_t k = 0; k < 8; ++k) {
            byte = static_cast<std::uint8_t>(byte << 1);
            if (i + k < bits.size() && bits[i + k]) byte |= 1;
        }
        out.push_back(byte);
    }
}

// np.unpackbits: MSB first.
std::vector<std::uint8_t> unpack_bits(std::span<const std::uint8_t> bytes) {
    std::vector<std::uint8_t> out;
    out.reserve(8 * bytes.size());
    for (const std::uint8_t b : bytes)
        for (int k = 7; k >= 0; --k) out.push_back(static_cast<std::uint8_t>((b >> k) & 1));
    return out;
}

std::string tuple4(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d) {
    return "(" + str(a) + ", " + str(b) + ", " + str(c) + ", " + str(d) + ")";
}

}  // namespace

std::int64_t file_size(std::int64_t n_latents) { return static_cast<std::int64_t>(LATENT_OFFSET) + n_latents + 4; }

header::HeaderFields BeaconFile::header_fields() const {
    header::HeaderFields h;
    h.callsign = callsign;
    h.grid = grid;
    h.picture_id = picture_id;
    h.mode = mode;
    h.segment = segment;
    h.codec_id = codec_id;
    return h;
}

void BeaconFile::check() const {
    if (waveform != config::WAVEFORM_CE)
        refuse("waveform " + str(waveform) + " is not CE (" + str(config::WAVEFORM_CE) + ")");
    const std::string call = morse::check_callsign(callsign);
    if (call != callsign) refuse("callsign " + pytext::repr(callsign) + " has trailing spaces");
    if (keying.size() != N_UNITS || keying != morse::keying_units(call))
        refuse("keying mask does not spell " + pytext::repr(call) + " in Morse");
    // The shape first: a wrong one is refused before the fields are encoded.
    if (hdr_bits.size() != N_BITS || hdr_bits != header::encode(header_fields()))
        refuse("header coded bits do not encode the file's header fields");
    if (latents_i8.empty() || latents_i8.size() > static_cast<std::size_t>(picture::SENT))
        refuse("latents must be 1.." + str(picture::SENT) + " int8 values, not int8(" +
               std::to_string(latents_i8.size()) + ",)");
    if (std::find(latents_i8.begin(), latents_i8.end(), std::int8_t(-128)) != latents_i8.end())
        refuse("int8 latents are clipped to +-127; -128 is not a valid value");
}

std::vector<std::uint8_t> BeaconFile::to_bytes() const {
    check();
    std::vector<std::uint8_t> out;
    out.reserve(static_cast<std::size_t>(file_size(static_cast<std::int64_t>(latents_i8.size()))));
    out.insert(out.end(), MAGIC.begin(), MAGIC.end());
    put_u8(out, FILE_VERSION);
    put_u8(out, waveform);
    put_u8(out, mode);
    put_u8(out, segment);
    put_le(out, static_cast<std::uint64_t>(picture_id), 4);
    put_le(out, static_cast<std::uint64_t>(codec_id), 2);
    std::string call = callsign;   // str.ljust(8), ASCII (check() admitted only [A-Z0-9/])
    call.resize(std::max<std::size_t>(call.size(), header::CALL_CHARS), ' ');
    out.insert(out.end(), call.begin(), call.end());
    const int grid_code = header::grid_encode(grid ? std::optional<std::string_view>(*grid) : std::nullopt);
    put_le(out, static_cast<std::uint64_t>(grid_code), 2);
    put_le(out, latents_i8.size(), 4);
    put_le(out, static_cast<std::uint64_t>(config::INT8_SCALE), 2);
    put_u8(out, ook ? FLAG_OOK : 0);
    put_u8(out, 0);
    pack_bits(out, keying);
    pack_bits(out, hdr_bits);
    for (const std::int8_t v : latents_i8) out.push_back(static_cast<std::uint8_t>(v));
    put_le(out, crc32(std::span<const std::uint8_t>(out)), 4);
    return out;
}

BeaconFile BeaconFile::from_bytes(std::span<const std::uint8_t> data) {
    const std::size_t n = data.size();
    if (n < LATENT_OFFSET + 5) refuse("beacon file too short (" + std::to_string(n) + " bytes)");
    if (!std::equal(MAGIC.begin(), MAGIC.end(), data.begin(),
                    [](char a, std::uint8_t b) { return static_cast<std::uint8_t>(a) == b; }))
        refuse("not a QRSSTVAE beacon file (magic " + bytes_repr(data.first(4)) + ")");
    if (crc32(data.first(n - 4)) != static_cast<std::uint32_t>(get_le(data, n - 4, 4)))
        refuse("beacon file CRC-32 mismatch");
    const std::uint8_t version = data[4];
    const std::uint64_t scale = get_le(data, 28, 2);
    const std::uint8_t flags = data[30], reserved = data[31];
    if (version != FILE_VERSION)
        refuse("beacon file version " + str(version) + ", expected " + str(FILE_VERSION));
    if (scale != static_cast<std::uint64_t>(config::INT8_SCALE))
        refuse("int8 scale " + std::to_string(scale) + ", expected " + str(config::INT8_SCALE));
    if ((flags & ~FLAG_OOK) || reserved) {
        char hex[8];
        std::snprintf(hex, sizeof hex, "0x%02x", static_cast<unsigned>(flags));
        refuse("unknown flags " + std::string(hex) + " or reserved byte " + str(reserved));
    }
    const auto n_lat = static_cast<std::int64_t>(get_le(data, 24, 4));
    if (static_cast<std::int64_t>(n) != file_size(n_lat))
        refuse(std::to_string(n) + " bytes for " + str(n_lat) + " latents, expected " + str(file_size(n_lat)));
    const auto call = data.subspan(14, header::CALL_CHARS);
    if (std::any_of(call.begin(), call.end(), [](std::uint8_t c) { return c >= 0x80; }))
        refuse("callsign bytes " + bytes_repr(call) + " are not ASCII");
    BeaconFile bf;
    bf.callsign.assign(call.begin(), call.end());
    bf.callsign.erase(bf.callsign.find_last_not_of(' ') + 1);   // rstrip(" ") (npos + 1 == 0)
    std::size_t o = HEAD_BYTES;
    bf.keying = unpack_bits(data.subspan(o, KEYING_BYTES));
    o += KEYING_BYTES;
    std::vector<std::uint8_t> hdr_all = unpack_bits(data.subspan(o, HDR_BYTES));
    if (std::any_of(hdr_all.begin() + static_cast<std::ptrdiff_t>(N_BITS), hdr_all.end(),
                    [](std::uint8_t b) { return b != 0; }))
        refuse("header pad bits are not zero");
    hdr_all.resize(N_BITS);
    o += HDR_BYTES;
    bf.waveform = data[5];
    bf.mode = data[6];
    bf.segment = data[7];
    bf.picture_id = static_cast<std::int64_t>(get_le(data, 8, 4));
    bf.codec_id = static_cast<std::int64_t>(get_le(data, 12, 2));
    bf.grid = header::grid_decode(static_cast<std::int64_t>(get_le(data, 22, 2)));
    bf.ook = (flags & FLAG_OOK) != 0;
    bf.hdr_bits = std::move(hdr_all);
    const auto lat = data.subspan(o, static_cast<std::size_t>(n_lat));
    bf.latents_i8.resize(lat.size());
    std::transform(lat.begin(), lat.end(), bf.latents_i8.begin(),
                   [](std::uint8_t b) { return static_cast<std::int8_t>(b); });
    bf.check();
    return bf;
}

void check_header_matches(const picture::StoredPicture& sp, std::int64_t segment, const header::HeaderFields& h) {
    const bool same = h.picture_id == sp.picture_id && h.mode == sp.mode && h.segment == segment &&
                      h.codec_id == sp.codec_id;
    if (!same)
        refuse("header (picture_id, mode, segment, codec_id) = " + tuple4(h.picture_id, h.mode, h.segment, h.codec_id) +
               " does not match the picture's " + tuple4(sp.picture_id, sp.mode, segment, sp.codec_id));
}

BeaconFile from_picture(const picture::StoredPicture& sp, std::int64_t segment, const header::HeaderFields& h,
                        bool ook) {
    sp.check();
    if (!(0 <= segment && segment <= sp.mode))
        refuse("segment must be 0.." + str(sp.mode) + ", not " + str(segment));
    check_header_matches(sp, segment, h);
    const std::string call = morse::check_callsign(h.callsign);
    BeaconFile bf;
    bf.waveform = config::WAVEFORM_CE;
    bf.mode = sp.mode;
    bf.segment = segment;
    bf.picture_id = sp.picture_id;
    bf.codec_id = sp.codec_id;
    bf.callsign = call;
    bf.grid = h.grid;
    bf.ook = ook;
    bf.keying = morse::keying_units(call);
    bf.hdr_bits = header::encode(h);
    const auto& seg = sp.segs[static_cast<std::size_t>(segment)];
    bf.latents_i8 = picture::to_int8(picture::air_values(std::span<const picture::Half>(seg), segment));
    bf.check();
    return bf;
}

void write(const std::string& path, const BeaconFile& bf) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);   // opened first, as the reference
    if (!f) throw std::runtime_error("beaconfile: cannot open " + path + " for writing");
    const std::vector<std::uint8_t> data = bf.to_bytes();
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!f) throw std::runtime_error("beaconfile: cannot write " + path);
}

BeaconFile read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("beaconfile: cannot open " + path);
    std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (f.bad()) throw std::runtime_error("beaconfile: cannot read " + path);
    return BeaconFile::from_bytes(data);
}

std::string bytes_repr(std::span<const std::uint8_t> b) {
    // bytes.__repr__: single quotes unless the bytes hold ' and no ".
    const bool has_single = std::find(b.begin(), b.end(), '\'') != b.end();
    const bool has_double = std::find(b.begin(), b.end(), '"') != b.end();
    const char quote = (has_single && !has_double) ? '"' : '\'';
    std::string out = "b";
    out += quote;
    static const char* hex = "0123456789abcdef";
    for (const std::uint8_t c : b) {
        if (c == static_cast<std::uint8_t>(quote) || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c < 0x20 || c >= 0x7f) {
            out += "\\x";
            out += hex[c >> 4];
            out += hex[c & 0xF];
        } else {
            out += static_cast<char>(c);
        }
    }
    out += quote;
    return out;
}

}  // namespace sstvae::qrss::beaconfile
