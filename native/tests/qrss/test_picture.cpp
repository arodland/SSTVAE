// native/core/qrss/picture against values and digests of the Python
// reference (sstvae/qrss/picture.py).
//
// Two inputs, both closed form:
//   F = np.sin(0.001 * g * g % 6.283) + 0.3 * np.cos(g),  g = np.arange(158400.)
//       (test_qrss_format's tensor of the frozen picture ID 0xD349C671)
//   E = ((i * 2654435761) % 4095 - 2047) / 1024.0,  i = np.arange(158400)  (exact)
// F goes through libm's sin and cos, which may differ from numpy's by an
// ulp on another platform; what is pinned of it is only what comes out of
// the float16 rounding (the IDs and the stored segments), which an ulp
// moves with probability ~1e-7. Digests are hashlib.sha256(a.tobytes()).
// tests/test_qrss_native_picture.py compares the two implementations
// directly where Python is available.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/npz.hpp"
#include "qrss/picture.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace pc = sstvae::qrss::picture;
using qrss_check::digest_is;

namespace {

constexpr std::size_t FULL = 3 * 52800;

std::vector<double> closed_form() {
    std::vector<double> v(FULL);
    for (std::size_t i = 0; i < FULL; ++i) {
        const double g = static_cast<double>(i);
        v[i] = std::sin(std::fmod(0.001 * g * g, 6.283)) + 0.3 * std::cos(g);
    }
    return v;
}

std::vector<double> exact() {
    std::vector<double> v(FULL);
    for (std::size_t i = 0; i < FULL; ++i)
        v[i] = static_cast<double>(static_cast<std::int64_t>((i * 2654435761ULL) % 4095) - 2047) / 1024.0;
    return v;
}

// The text of the std::invalid_argument f throws, or "" when it returns.
template <typename F>
std::string refusal(F f) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "";
}

std::vector<std::uint8_t> flat_bytes(const pc::Segment& s) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(s.data());
    return std::vector<std::uint8_t>(p, p + s.size() * sizeof(pc::Half));
}

}  // namespace

static void test_closed_form_ids() {
    const auto f = closed_form();
    check::equal(pc::StoredPicture::from_latents(f, 0xD1D8, 0).picture_id, std::int64_t(0xD349C671),
                 "the frozen picture ID (mode A, codec 0xD1D8)");
    check::equal(pc::StoredPicture::from_latents(f, 0xD1D8, 1).picture_id, std::int64_t(0x73EA4EF6), "mode B");
    const auto sp = pc::StoredPicture::from_latents(f, 0xD1D8, 2);
    check::equal(sp.picture_id, std::int64_t(0xFE3BB354), "mode C");
    check::equal(sp.id_hex(), std::string("fe3bb354"), "id_hex");
    digest_is(sp.segs[0], "106db43b28054f84717b74eaeb880989280a0a49116bada11b9f8afe9104b7a6", "F segment 0");
    digest_is(sp.segs[1], "647a2907224f1aa56979bd0b81f0910ed2a7be7068ac455cdb5135364bc8b9dd", "F segment 1");
    digest_is(sp.segs[2], "6939aa5f8893f34e5f9b6e42d78e3d71afb4dfd3c9d49a351cf3549906af46a4", "F segment 2");
    sp.check();

    // The air order and back, in each element type.
    const auto air = pc::air_values(std::span<const pc::Half>(sp.segs[1]), 1);
    digest_is(air, "9e52c90ec398708e1df9ff7f09a044c4641328390d11bf7e766c17c8b4b95e03", "air_values");
    digest_is(pc::canonical_from_air(air, 1), "651ec992ce6a388297d4c916f3c04f3c0f9da02486209b5de92ce28ea8b10220",
              "canonical_from_air float64");
    std::vector<pc::Half> air_h(air.size());
    std::vector<float> air_f(air.size());
    for (std::size_t i = 0; i < air.size(); ++i) {
        air_h[i] = qrss::half::to_half(air[i]);
        air_f[i] = static_cast<float>(air[i]);
    }
    // float16 back to canonical is the stored segment itself.
    digest_is(pc::canonical_from_air(std::span<const pc::Half>(air_h), 1, air_h.size()),
              "647a2907224f1aa56979bd0b81f0910ed2a7be7068ac455cdb5135364bc8b9dd", "canonical_from_air float16");
    digest_is(pc::canonical_from_air(std::span<const float>(air_f), 1, air_f.size()),
              "55f0a5a71b3e007d94f84ece5e8f1fe92e12129076dda63225c15b9fd31957ed", "canonical_from_air float32");
    std::vector<float> seg_f(sp.segs[1].size());
    for (std::size_t i = 0; i < seg_f.size(); ++i) seg_f[i] = static_cast<float>(qrss::half::to_double(sp.segs[1][i]));
    digest_is(pc::air_values(std::span<const float>(seg_f), 1, seg_f.size()),
              "9e52c90ec398708e1df9ff7f09a044c4641328390d11bf7e766c17c8b4b95e03", "air_values of float32");
    // Rows: two segments at once are each one's air values.
    std::vector<double> two(2 * 52800);
    for (std::size_t i = 0; i < 52800; ++i) {
        two[i] = qrss::half::to_double(sp.segs[1][i]);
        two[52800 + i] = qrss::half::to_double(sp.segs[1][i]);
    }
    const auto rows = pc::air_values(std::span<const double>(two), 1, 52800);
    check::is_true(rows.size() == 2 * 50600 && std::vector<double>(rows.begin(), rows.begin() + 50600) == air &&
                       std::vector<double>(rows.begin() + 50600, rows.end()) == air,
                   "air_values rows");
    digest_is(pc::to_int8(air), "325b0d7b82164d13927a286d3f35bd1d4b2499f5c1242567160c4c63addbcec0", "to_int8(air)");
}

static void test_exact_input() {
    const auto e = exact();
    std::vector<pc::Segment> segs;
    for (int g = 0; g < 3; ++g) segs.push_back(pc::prepare_segment(e, g));
    digest_is(segs[0], "9fe9957eac15cf0803b18403e25e7d89a8a57c04cdd1dddd9b1fbc79e48e1507", "E segment 0");
    digest_is(segs[1], "f8458b3df3d1b9282927fc28fe367adf46ed17d135ba4de3434ccdceaa06b388", "E segment 1");
    digest_is(segs[2], "8d1d66e3fac2e62640d650e5842bd21a510b9d3279041ffa71b2153024b9beb5", "E segment 2");
    check::equal(pc::picture_id(0x1234, 2, segs), std::uint32_t(0xAE38E2DE), "picture_id(0x1234, C, E)");
}

static void test_maps() {
    const char* never[] = {"bd5de7138bb283fbe4dbf98a4ba50fb751c4a557c9bfb6c9619ab8ee6169f944",
                           "baaa1abbd8a3e9321db94915b8bd23518bc599ecccdf5ff4c1aafdff758975bf",
                           "e00e615f2cf971079272096844f2ec17a7033df6b31f1dcd04c9d61fda954e7a"};
    const char* a2c[] = {"df5803c57d7d01ec8547ab388dbb8e06bc14de87d32426b72225b6764f32f108",
                         "6b32f21b7d224c96b261c67b8eb075ddf162aee239aa9b7217434428756a8627",
                         "7461954cb1f6ef5612e43771849aec788bfdd8a39df9b7607432ab264eae4fc7"};
    for (int g = 0; g < 3; ++g) {
        const auto mask = pc::never_sent(g);
        std::size_t n = 0;
        for (auto m : mask) n += m;
        check::equal(n, std::size_t(2200), "never_sent count, group " + std::to_string(g));
        digest_is(mask, never[g], "never_sent " + std::to_string(g));
        const auto perm = pc::air_to_canonical(g);
        digest_is(std::vector<std::int64_t>(perm.begin(), perm.end()), a2c[g], "air_to_canonical " + std::to_string(g));
    }
}

static void test_int8() {
    const std::vector<double> edges{0.0, -0.0, 0.025, -0.025, 0.075, -0.075, 0.125, 6.35, -6.35, 6.375, -6.375, 6.4,
                                    -6.4, 1e9, -1e9, std::numeric_limits<double>::infinity(),
                                    -std::numeric_limits<double>::infinity(), 0.0249999999, 0.0250000001, 1.0, -1.0,
                                    3.14159, 0.175, 0.225};
    const std::vector<std::int8_t> want{0,    0,   0,  0,   2, -2, 2, 127, -127, 127, -127, 127,
                                        -127, 127, -127, 127, -127, 0, 1, 20,  -20, 63,   4,    4};
    check::is_true(pc::to_int8(edges) == want, "to_int8 at the rounding and clipping edges");
    std::vector<std::int8_t> all;
    for (int v = -128; v < 128; ++v) all.push_back(static_cast<std::int8_t>(v));
    digest_is(pc::from_int8(std::span<const std::int8_t>(all)),
              "ad591f9c14fbe88db043745c43696230b3a2dcf4d73ac34ea7d8651790d1ec4f", "from_int8(-128..127)");
    check::equal(pc::to_int8(std::vector<double>{std::nan("")})[0], std::int8_t(0), "NaN -> 0");
}

static void test_refusals() {
    const auto f = closed_form();
    check::equal(refusal([&] { pc::prepare_segment(std::span<const double>(f).first(FULL - 1), 0); }),
                 std::string("expected 158400 latents, got 158399"), "prepare_segment length");
    auto z = f;
    for (std::size_t i = 52800; i < 105600; ++i) z[i] = 0.0;
    check::equal(refusal([&] { pc::prepare_segment(z, 1); }),
                 std::string("group 1 has no energy to normalise (rms 0.0)"), "prepare_segment zero energy");
    auto n = f;
    n[2 * 52800 + pc::air_to_canonical(2)[0]] = std::nan("");
    check::equal(refusal([&] { pc::prepare_segment(n, 2); }),
                 std::string("group 2 has no energy to normalise (rms nan)"), "prepare_segment NaN");
    check::equal(refusal([] { pc::check_group(3); }), std::string("group must be 0..2, not 3"), "check_group");
    check::equal(refusal([] { pc::check_mode(-1); }), std::string("mode must be 0..2 (A..C), not -1"), "check_mode");
    const auto sp = pc::StoredPicture::from_latents(f, 0xD1D8, 1);
    check::equal(refusal([&] { pc::picture_id(1 << 16, 0, std::span<const pc::Segment>(sp.segs).first(1)); }),
                 std::string("codec_id must fit in 16 bits, not 65536"), "picture_id codec");
    check::equal(refusal([&] { pc::picture_id(0xD1D8, 1, std::span<const pc::Segment>(sp.segs).first(1)); }),
                 std::string("mode B stores 2 segments, got 1"), "picture_id count");
    const std::vector<pc::Segment> short_seg{pc::Segment(5)};
    check::equal(refusal([&] { pc::picture_id(0xD1D8, 0, short_seg); }),
                 std::string("segment shape (5,), expected (52800,)"), "picture_id shape");
    pc::StoredPicture bad = sp;
    bad.picture_id = 1;
    check::equal(refusal([&] { bad.check(); }),
                 std::string("picture ID 00000001 does not match the stored latents (73ea4ef6)"), "check");
    check::equal(refusal([] { pc::air_values(std::vector<double>(5), 0); }),
                 std::string("segment has 5 latents, expected 52800"), "air_values length");
    check::equal(refusal([] { pc::canonical_from_air(std::vector<double>(5), 0); }),
                 std::string("air vector has 5 values, expected 50600"), "canonical_from_air length");
    check::equal(pc::id_hex(-5), std::string("-0000005"), "id_hex negative");
    check::equal(pc::id_hex(0xab), std::string("000000ab"), "id_hex");
}

static void test_qrsp() {
    const auto sp = pc::StoredPicture::from_latents(closed_form(), 0xD1D8, 1);
    const auto bytes = pc::qrsp_bytes(sp);
    check::equal(bytes.size(), std::size_t(212480), "qrsp size");
    digest_is(bytes, "f9b5b8fc756594a68e48dfc573c279b5d26eda89dfc8735327a70d060e70dc7b",
              "qrsp bytes (np.savez of the five members)");
    const auto back = pc::parse_qrsp(bytes, "x.qrsp");
    check::is_true(back.mode == 1 && back.codec_id == 0xD1D8 && back.picture_id == 0x73EA4EF6 &&
                       back.segs.size() == 2 && flat_bytes(back.segs[0]) == flat_bytes(sp.segs[0]) &&
                       flat_bytes(back.segs[1]) == flat_bytes(sp.segs[1]),
                   "qrsp round trip");
    // A wrong ID, a wrong version and a missing member, each refused.
    auto edit = [&](const char* member, std::int64_t value) {
        const auto z = pc::qrsp_npz(sp);
        qrss::npz::Npz out;
        for (const auto& name : z.files())
            out.add(name, name == member ? qrss::npz::Array::scalar<std::int64_t>(value) : z.at(name));
        return qrss::npz::npz_bytes(out);
    };
    check::equal(refusal([&] { pc::parse_qrsp(edit("picture_id", 0x73EA4EF7), "p.qrsp"); }),
                 std::string("picture ID 73ea4ef7 does not match the stored latents (73ea4ef6)"), "qrsp wrong ID");
    check::equal(refusal([&] { pc::parse_qrsp(edit("version", 2), "p.qrsp"); }),
                 std::string("p.qrsp: .qrsp version 2, expected 1"), "qrsp version");
    check::equal(refusal([&] { pc::parse_qrsp(edit("mode", 2), "p.qrsp"); }),
                 std::string("p.qrsp: segs float16(2, 52800) do not fit mode 2"), "qrsp mode");
    {
        const auto z = pc::qrsp_npz(sp);
        qrss::npz::Npz out;
        for (const auto& name : z.files())
            if (name != "segs") out.add(name, z.at(name));
        check::equal(refusal([&] { pc::parse_qrsp(qrss::npz::npz_bytes(out), "p.qrsp"); }),
                     std::string("p.qrsp: not a .qrsp file (missing 'segs is not a file in the archive')"),
                     "qrsp missing member");
    }
    pc::StoredPicture bad = sp;
    bad.picture_id ^= 1;
    check::is_true(!refusal([&] { pc::qrsp_bytes(bad); }).empty(), "a wrong ID is not saved");
}

int main() {
    try {
        test_closed_form_ids();
        test_exact_input();
        test_maps();
        test_int8();
        test_refusals();
        test_qrsp();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_picture");
}
