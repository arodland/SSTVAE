// native/core/qrss/beaconfile against digests and texts of the Python
// reference (sstvae/qrss/beaconfile.py).
//
// The picture is test_picture.cpp's closed-form F, mode B, codec 0xD1D8
// (ID 0x73EA4EF6), and the header callsign "K1ABC/P", grid "FN42":
//   sp = StoredPicture.from_latents(F, 0xD1D8, 1)
//   h  = HeaderFields("K1ABC/P", "FN42", sp.picture_id, 1, segment, 0xD1D8)
//   hashlib.sha256(beaconfile.from_picture(sp, segment, h, ook).to_bytes())
// A beacon file is integers, bits, int8 values and a CRC-32, so its bytes
// are pinned exactly. tests/test_qrss_native_beaconfile.py compares the
// two implementations directly where Python is available, including the C
// reference beacon reading a file the C++ wrote.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/beaconfile.hpp"
#include "qrss/crc32.hpp"
#include "qrss/header.hpp"
#include "qrss/picture.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace pc = sstvae::qrss::picture;
namespace bfm = sstvae::qrss::beaconfile;
namespace hdr = sstvae::qrss::header;
using qrss_check::digest;
using qrss_check::digest_is;

namespace {

pc::StoredPicture closed_form_picture() {
    std::vector<double> v(3 * 52800);
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double g = static_cast<double>(i);
        v[i] = std::sin(std::fmod(0.001 * g * g, 6.283)) + 0.3 * std::cos(g);
    }
    return pc::StoredPicture::from_latents(v, 0xD1D8, 1);
}

hdr::HeaderFields header_for(const pc::StoredPicture& sp, std::int64_t segment,
                             std::optional<std::string> grid = std::string("FN42")) {
    hdr::HeaderFields h;
    h.callsign = "K1ABC/P";
    h.grid = std::move(grid);
    h.picture_id = sp.picture_id;
    h.mode = sp.mode;
    h.segment = segment;
    h.codec_id = sp.codec_id;
    return h;
}

template <typename F>
std::string refusal(F f) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "";
}

// A file body (without its CRC) with a valid CRC-32 appended, little-endian.
std::vector<std::uint8_t> recrc(std::vector<std::uint8_t> body) {
    const std::uint32_t c = qrss::crc32(body);
    for (int k = 0; k < 4; ++k) body.push_back(static_cast<std::uint8_t>(c >> (8 * k)));
    return body;
}

bool same(const bfm::BeaconFile& a, const bfm::BeaconFile& b) {
    return a.waveform == b.waveform && a.mode == b.mode && a.segment == b.segment && a.picture_id == b.picture_id &&
           a.codec_id == b.codec_id && a.callsign == b.callsign && a.grid == b.grid && a.ook == b.ook &&
           a.keying == b.keying && a.hdr_bits == b.hdr_bits && a.latents_i8 == b.latents_i8;
}

}  // namespace

static void test_bytes(const pc::StoredPicture& sp) {
    struct Case {
        std::int64_t segment;
        bool ook;
        bool grid;
        const char* sha;
    };
    const Case cases[] = {
        {0, false, true, "2abe695a8676e64f09fb30c5b104ec336c168faa746a26e336bbe6dc3caabc77"},
        {0, false, false, "9ccbcab0280eccc2fb2bc6b2380cd8af35eb1eb035787e1fb8e07900abfc4e42"},
        {0, true, true, "df54ee3b013133f0d6ddc3f049ff540c15ce591a17d65fc011b7e8c4d0b60336"},
        {0, true, false, "66146269c8c75c5b1195ebad8a8ef268c307fec0a5cf98f8307539af77638b84"},
        {1, false, true, "b8d3fd8666f6986392122373010c3217cf7b95f4b3f0116544dc9b6749a184f4"},
        {1, false, false, "e51930a29a2da11a3d97d5c698a52ce43abfc2b6501a7346ba19731d3d12beb8"},
        {1, true, true, "0b7d7554094bec63bcf86413a207d36d4da6c6c95ab617a5e9946dc866cf7987"},
        {1, true, false, "181758b4d59795f1d40eca7b36e0facfcf88be73fc90b7d0f49139d1098c5490"},
    };
    for (const Case& c : cases) {
        const std::string what = "segment " + std::to_string(c.segment) + (c.ook ? " OOK" : " FSK") +
                                 (c.grid ? " grid" : " no grid");
        const auto h = header_for(sp, c.segment, c.grid ? std::optional<std::string>("FN42") : std::nullopt);
        const auto bf = bfm::from_picture(sp, c.segment, h, c.ook);
        const auto bytes = bf.to_bytes();
        check::equal(bytes.size(), std::size_t(50970), what + ": size");
        check::equal(static_cast<std::int64_t>(bytes.size()), bfm::file_size(), what + ": file_size()");
        digest_is(bytes, c.sha, what + ": bytes");
        check::is_true(same(bfm::BeaconFile::from_bytes(bytes), bf), what + ": from_bytes round trip");
        check::is_true(bf.header_fields() == h, what + ": header_fields");
    }
    const auto bf = bfm::from_picture(sp, 1, header_for(sp, 1));
    digest_is(bf.keying, "46c4fb4542fce939dffea24afa991d130d650e7bfca58a7e1735023750a83089", "keying");
    digest_is(bf.hdr_bits, "278bcfcaad06f54296abb06016a3cf7504705360db598b260647b863ab0d4d0e", "hdr_bits");
    digest_is(bf.latents_i8, "325b0d7b82164d13927a286d3f35bd1d4b2499f5c1242567160c4c63addbcec0", "latents_i8");
    check::equal(bfm::file_size(1), std::int64_t(371), "file_size(1)");
    // A file of fewer latents than a segment holds is valid.
    bfm::BeaconFile part = bf;
    part.latents_i8.resize(1000);
    const auto pb = part.to_bytes();
    check::equal(pb.size(), std::size_t(366 + 1000 + 4), "a 1000-latent file");
    check::is_true(same(bfm::BeaconFile::from_bytes(pb), part), "a 1000-latent file round trip");
}

static void test_from_bytes_refusals(const pc::StoredPicture& sp) {
    const auto good = bfm::from_picture(sp, 1, header_for(sp, 1)).to_bytes();
    const std::vector<std::uint8_t> body(good.begin(), good.end() - 4);
    auto ed = [&](std::size_t off, std::uint8_t val) {
        auto b = body;
        b[off] = val;
        return recrc(b);
    };
    auto from = [](std::vector<std::uint8_t> d) { return refusal([&] { bfm::BeaconFile::from_bytes(d); }); };
    auto magic = good;
    magic[0] = 'X';
    check::equal(from(magic), std::string("not a QRSSTVAE beacon file (magic b'XRSB')"), "magic");
    check::equal(from(std::vector<std::uint8_t>(good.begin(), good.begin() + 370)),
                 std::string("beacon file too short (370 bytes)"), "too short");
    auto crc = good;
    crc.back() ^= 1;
    check::equal(from(crc), std::string("beacon file CRC-32 mismatch"), "CRC");
    check::equal(from(ed(4, 2)), std::string("beacon file version 2, expected 1"), "version");
    check::equal(from(ed(5, 1)), std::string("waveform 1 is not CE (0)"), "waveform");
    check::equal(from(ed(28, 21)), std::string("int8 scale 21, expected 20"), "scale");
    check::equal(from(ed(30, 2)), std::string("unknown flags 0x02 or reserved byte 0"), "flags");
    check::equal(from(ed(31, 7)), std::string("unknown flags 0x00 or reserved byte 7"), "reserved");
    {
        auto b = body;
        b[24] = static_cast<std::uint8_t>(50599 & 0xFF);   // n_lat 50600 -> 50599 (low byte only)
        check::equal(from(recrc(b)), std::string("50970 bytes for 50599 latents, expected 50969"), "size");
    }
    check::equal(from(ed(14, 0xC3)), std::string("callsign bytes b'\\xc31ABC/P ' are not ASCII"), "ASCII");
    check::equal(from(ed(365, static_cast<std::uint8_t>(body[365] | 1))), std::string("header pad bits are not zero"),
                 "pad bits");
    check::equal(from(ed(366 + 17, 0x80)),
                 std::string("int8 latents are clipped to +-127; -128 is not a valid value"), "-128");
    check::equal(from(ed(35, static_cast<std::uint8_t>(body[35] ^ 0x40))),
                 std::string("keying mask does not spell 'K1ABC/P' in Morse"), "keying");
    check::equal(from(ed(66, static_cast<std::uint8_t>(body[66] ^ 8))),
                 std::string("header coded bits do not encode the file's header fields"), "header bit");

    // Single-byte flips (one bit each): the head, keying and header bytes
    // all, then every 97th latent byte and the CRC.
    auto flipped = good;
    int refused = 0, tried = 0;
    for (std::size_t i = 0; i < flipped.size(); ++i) {
        if (i >= bfm::LATENT_OFFSET + 64 && i < flipped.size() - 8 && i % 97) continue;
        const auto mask = static_cast<std::uint8_t>(1u << (i % 8));
        flipped[i] ^= mask;
        const std::string why = from(flipped);
        flipped[i] ^= mask;
        ++tried;
        const bool want = i < 4 ? why.rfind("not a QRSSTVAE beacon file (magic b'", 0) == 0
                                : why == "beacon file CRC-32 mismatch";
        refused += want ? 1 : 0;
    }
    check::equal(refused, tried, "every single-byte flip refused (magic or CRC)");
}

static void test_check_refusals(const pc::StoredPicture& sp) {
    const auto bf = bfm::from_picture(sp, 1, header_for(sp, 1));
    auto b = bf;
    b.callsign = "K1ABC/P ";
    check::equal(refusal([&] { b.check(); }), std::string("callsign 'K1ABC/P ' has trailing spaces"), "trailing");
    b = bf;
    b.latents_i8.clear();
    check::equal(refusal([&] { b.check(); }), std::string("latents must be 1..50600 int8 values, not int8(0,)"),
                 "no latents");
    b = bf;
    b.latents_i8.push_back(0);
    check::equal(refusal([&] { b.to_bytes(); }), std::string("latents must be 1..50600 int8 values, not int8(50601,)"),
                 "too many latents");
    b = bf;
    b.waveform = 1;
    check::equal(refusal([&] { b.to_bytes(); }), std::string("waveform 1 is not CE (0)"), "waveform");
    b = bf;
    b.grid = "FN43";
    check::equal(refusal([&] { b.check(); }),
                 std::string("header coded bits do not encode the file's header fields"), "grid edited");
    check::equal(refusal([&] { bfm::from_picture(sp, 2, header_for(sp, 1)); }),
                 std::string("segment must be 0..1, not 2"), "from_picture segment");
    check::equal(refusal([&] { bfm::from_picture(sp, 1, header_for(sp, 0)); }),
                 std::string("header (picture_id, mode, segment, codec_id) = (1944735478, 1, 0, 53720) does not "
                             "match the picture's (1944735478, 1, 1, 53720)"),
                 "from_picture header");
    pc::StoredPicture bad = sp;
    bad.picture_id ^= 1;
    check::is_true(!refusal([&] { bfm::from_picture(bad, 1, header_for(sp, 1)); }).empty(),
                   "from_picture checks the picture");
    const std::vector<std::uint8_t> raw{'Q', 'R', 0x00, 0xff, '\'', '"', '\\', '\n', '\t', '\r', 0x7f, ' '};
    check::equal(bfm::bytes_repr(raw), std::string("b'QR\\x00\\xff\\'\"\\\\\\n\\t\\r\\x7f '"), "bytes_repr");
}

int main() {
    try {
        const auto sp = closed_form_picture();
        check::equal(sp.picture_id, std::int64_t(0x73EA4EF6), "the picture");
        test_bytes(sp);
        test_from_bytes_refusals(sp);
        test_check_refusals(sp);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_beaconfile");
}
