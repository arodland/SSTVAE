// native/core/qrss/tx (the slot functions) against digests, values and
// texts of the Python reference (sstvae/qrss/tx.py).
//
// The sources are test_picture.cpp's closed-form F, mode B, codec 0xD1D8,
// as a stored picture and as its segment-1 beacon file:
//   sp = StoredPicture.from_latents(F, 0xD1D8, 1)
//   h  = HeaderFields("K1ABC/P", "FN42", sp.picture_id, 1, 1, 0xD1D8)
//   bf = beaconfile.from_picture(sp, 1, h)
//   hashlib.sha256(tx.slot_symbols(src, Q, spec, ...).tobytes())
// with Q = 1954752 unless named. CE symbols are float64, L symbols
// complex128 (re, im). tests/test_qrss_native_tx.py compares the two
// implementations directly where Python is available.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/beaconfile.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/picture.hpp"
#include "qrss/tx.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace pc = sstvae::qrss::picture;
namespace bfm = sstvae::qrss::beaconfile;
namespace hdr = sstvae::qrss::header;
namespace fr = sstvae::qrss::frame;
namespace txm = sstvae::qrss::tx;
using qrss_check::digest;
using qrss_check::digest_is;

namespace {

constexpr std::uint64_t Q_TEST = 1954752;

pc::StoredPicture closed_form_picture() {
    std::vector<double> v(3 * 52800);
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double g = static_cast<double>(i);
        v[i] = std::sin(std::fmod(0.001 * g * g, 6.283)) + 0.3 * std::cos(g);
    }
    return pc::StoredPicture::from_latents(v, 0xD1D8, 1);
}

hdr::HeaderFields header_for(const pc::StoredPicture& sp, std::int64_t segment, std::string call = "K1ABC/P",
                             std::string grid = "FN42") {
    hdr::HeaderFields h;
    h.callsign = std::move(call);
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
    } catch (const std::out_of_range& e) {
        return std::string("IndexError: ") + e.what();
    } catch (const std::overflow_error& e) {
        return std::string("OverflowError: ") + e.what();
    }
    return "";
}

std::string sym_digest(const txm::Symbols& s) {
    return s.waveform == fr::Waveform::L ? digest(s.l) : digest(s.ce);
}

}  // namespace

static void test_symbols(const pc::StoredPicture& sp, const bfm::BeaconFile& bf) {
    struct Case {
        const char* spec;
        std::size_t n_sym;
        const char* fp16;   // the .qrsp as stored (use_int8 None)
        const char* int8;   // through int8: the .qrsp with use_int8, and the beacon file either way
    };
    const Case cases[] = {
        {"full", 57274, "cef09e18aad168e3784f94f9f9f411d0656806f3c0ab4fb8f368dd6ff5725d22",
         "968bf7eaebaa7592c8aecf0f06514d48a12bbe6d8243b22ae12fa8ea79ef4715"},
        {"short", 7670, "84f8709a21edbeeb22f53517a4b039437cfd57c2baf6778120abc9bde693e985",
         "2ebd3afbd01f7f4ddcb66ead8a80a8c8a68532be2e12ef1a792daf23035a5545"},
        {"tiny", 1753, "e1c3eaab8ab041a83df5e5045641ade3bbaa68673327834cdf5b82f93d839141",
         "60bc1d0fcf46fcbc57e2ad828d766940957b24b5209f755040befb8a9527207b"},
        {"l-full", 28637, "b1cb354d23f90c8fdc1ba0dd51e90c420f42efb19f8069acaf4b6b1199bedb67",
         "8d117b83e39a6f80edc70abafe07bd3b2a17b8b05868610067ea64e0f976a94b"},
        {"l-short", 3834, "826c62a58efcd3212afd3f56771774fef2b4307f8ed2a1e2de2ce32f76735597",
         "7a3fdfb9dc9ab29600cb229cee5a92ae7bd25481467995473f694482081c4143"},
        {"l-tiny", 877, "4815b75d3e30d8b2509c5e279f29bb1abde68bb18070422ea4cda2d930926bb9",
         "96867a4634ef75da20e0ba7c5256bcadfb2ccdf8590c571dd3621f73e080acfc"},
    };
    const txm::Source src_sp = sp, src_bf = bf;
    const auto h = header_for(sp, 1);
    for (const Case& c : cases) {
        const fr::FrameSpec& spec = fr::get_any(c.spec);
        const std::string n = c.spec;
        const auto a = txm::slot_symbols(src_sp, Q_TEST, spec, 1, h);
        check::equal(a.size(), c.n_sym, n + ": n_sym");
        check::equal(sym_digest(a), std::string(c.fp16), n + ": .qrsp");
        check::equal(sym_digest(txm::slot_symbols(src_sp, Q_TEST, spec, 1, h, true)), std::string(c.int8),
                     n + ": .qrsp, use_int8");
        check::equal(sym_digest(txm::slot_symbols(src_bf, Q_TEST, spec)), std::string(c.int8), n + ": beacon file");
        check::equal(sym_digest(txm::slot_symbols(src_bf, Q_TEST, spec, std::nullopt, std::nullopt, true)),
                     std::string(c.int8), n + ": beacon file, use_int8");
    }
    const fr::FrameSpec& short_spec = fr::get("short");
    check::equal(sym_digest(txm::slot_symbols(src_bf, Q_TEST + 1, short_spec)),
                 std::string("1901f416159bae3595ce269d964feed00041453e1c18496ee459c7f3ae73e4f7"), "short, Q + 1");
    check::equal(sym_digest(txm::slot_symbols(src_bf, (std::uint64_t(1) << 40) + 12345, short_spec)),
                 std::string("b14c74734eebca44c0932202aa31ff257d5ede9c9e7caf95d5dbe3ed99f2adfd"), "short, 2^40 + 12345");
    check::equal(sym_digest(txm::slot_symbols(src_sp, Q_TEST, fr::get("tiny"), 0)),
                 std::string("c31f5bffd8b15a6be6c83a99462479d42daa6329b268d89b48365b09bbfbb746"),
                 "tiny, segment 0, no header");
    check::equal(sym_digest(txm::slot_symbols(src_bf, Q_TEST)),
                 std::string("968bf7eaebaa7592c8aecf0f06514d48a12bbe6d8243b22ae12fa8ea79ef4715"), "default frame (full)");

    digest_is(txm::slot_air_latents(src_sp, 1), "9e52c90ec398708e1df9ff7f09a044c4641328390d11bf7e766c17c8b4b95e03",
              "slot_air_latents .qrsp");
    digest_is(txm::slot_air_latents(src_sp, 1, true), "9b180dbaffa884d09279a04b49ecac8066cf8fa5da4389175527bd346acdf0c9",
              "slot_air_latents .qrsp through int8");
    digest_is(txm::slot_air_latents(src_bf), "9b180dbaffa884d09279a04b49ecac8066cf8fa5da4389175527bd346acdf0c9",
              "slot_air_latents beacon file");
    const auto sh = txm::segment_and_header(src_bf, std::nullopt, std::nullopt);
    check::is_true(sh.segment == 1 && sh.header && *sh.header == bf.header_fields() && sh.hdr_bits == bf.hdr_bits &&
                       sh.keying == bf.keying && !sh.ook,
                   "segment_and_header of a beacon file");
    const auto sh2 = txm::segment_and_header(src_sp, 1, h);
    check::is_true(sh2.segment == 1 && sh2.header == h && sh2.hdr_bits == hdr::encode(h) && sh2.keying == bf.keying,
                   "segment_and_header of a .qrsp with a header");
    check::is_true(!txm::segment_and_header(src_sp, std::nullopt, std::nullopt).header, "no header, none returned");

    digest_is(*txm::slot_keying(src_bf), "46c4fb4542fce939dffea24afa991d130d650e7bfca58a7e1735023750a83089",
              "slot_keying beacon file");
    digest_is(*txm::slot_keying(src_sp, header_for(sp, 1, "W1AW")),
              "46616a4ae7c95eafbe1835b236cd9d691b8a8432c13229687d47549876a6b5f4", "slot_keying W1AW");
    check::is_true(!txm::slot_keying(src_sp), "slot_keying, no header");
}

static void test_slot_samples() {
    struct Case {
        const char* name;
        std::int64_t dflt, zero, b48k, c44k;
    };
    // slot_samples(s), (s, 0, 0), (s, 12.5, 7.25, 48000), (s, 0.1, 1e-9, 44100.0)
    const Case cases[] = {
        {"full", 14381304, 14261304, 86515823, 78619847},
        {"medium", 5344542, 5224542, 32295248, 28804694},
        {"short", 2166094, 2046094, 13224563, 11283502},
        {"tiny", 546800, 426800, 3508800, 2357146},
        {"full-v1", 14381304, 14261304, 86515823, 78619847},
        {"medium-v1", 5344542, 5224542, 32295248, 28804694},
        {"short-v1", 2166094, 2046094, 13224563, 11283502},
        {"full-spreadonly", 13741104, 13621104, 82674623, 75090745},
        {"l-full", 14381183, 14261183, 86515095, 78619179},
        {"l-medium", 5344178, 5224178, 32293065, 28802689},
        {"l-short", 2165488, 2045488, 13220926, 11280160},
        {"l-tiny", 548740, 428740, 3520440, 2367840},
    };
    std::size_t n_presets = 0;
    for (auto g : {qrss::config::PresetGroup::Presets, qrss::config::PresetGroup::Legacy,
                   qrss::config::PresetGroup::Prototypes, qrss::config::PresetGroup::LPresets})
        n_presets += fr::presets(g).size();
    check::equal(n_presets, std::size_t(12), "every preset is pinned");
    for (const Case& c : cases) {
        const fr::FrameSpec& s = fr::get_any(c.name);
        const std::string n = c.name;
        check::equal(txm::slot_samples(s), c.dflt, n + ": slot_samples");
        check::equal(txm::slot_samples(s, 0, 0), c.zero, n + ": pre 0, post 0");
        check::equal(txm::slot_samples(s, 12.5, 7.25, 48000), c.b48k, n + ": 48 kHz");
        check::equal(txm::slot_samples(s, 0.1, 1e-9, 44100.0), c.c44k, n + ": 44.1 kHz");
    }
    const fr::FrameSpec& full = fr::get("full");
    check::equal(refusal([&] { txm::slot_samples(full, std::nan(""), 3.0); }),
                 std::string("cannot convert float NaN to integer"), "slot_samples NaN");
    check::equal(refusal([&] { txm::slot_samples(full, 12.0, std::numeric_limits<double>::infinity()); }),
                 std::string("OverflowError: cannot convert float infinity to integer"), "slot_samples inf");
}

static void test_check_pilot() {
    check::equal(txm::check_pilot(1500), 1500.0, "check_pilot(1500)");
    check::equal(txm::check_pilot(2700), 2700.0, "check_pilot(2700)");
    check::equal(refusal([] { txm::check_pilot(310, "310"); }),
                 std::string("an L pilot at 310 Hz puts the data down to 287.9 Hz, below the 300 Hz receivers search"),
                 "pilot too low");
    check::equal(refusal([] { txm::check_pilot(322.0690721649484); }),
                 std::string("an L pilot at 322.0690721649484 Hz puts the data down to 300.0 Hz, below the 300 Hz "
                             "receivers search"),
                 "pilot at the edge (the sum rounds below 300)");
    check::equal(refusal([] { txm::check_pilot(2700.0001); }),
                 std::string("carrier 2700.0001 Hz is outside the 300-2700 Hz audio band that receivers search"),
                 "carrier above the band");
    check::equal(refusal([] { txm::check_pilot(299, "299"); }),
                 std::string("carrier 299 Hz is outside the 300-2700 Hz audio band that receivers search"),
                 "carrier below the band");
    check::equal(refusal([] { txm::check_pilot(std::nan("")); }),
                 std::string("carrier nan Hz is outside the 300-2700 Hz audio band that receivers search"), "NaN");
}

static void test_refusals(const pc::StoredPicture& sp, const bfm::BeaconFile& bf) {
    const txm::Source src_sp = sp, src_bf = bf;
    const fr::FrameSpec& short_spec = fr::get("short");
    const auto h = header_for(sp, 1);
    check::equal(refusal([&] { txm::slot_symbols(src_sp, Q_TEST, short_spec); }),
                 std::string("frame 'short' has a header: pass header= for a .qrsp"), "no header");
    check::equal(refusal([&] { txm::slot_symbols(src_sp, Q_TEST, short_spec, 2); }),
                 std::string("segment must be 0..1, not 2"), "segment");
    check::equal(refusal([&] { txm::slot_symbols(src_sp, Q_TEST, short_spec, 0, h); }),
                 std::string("header (picture_id, mode, segment, codec_id) = (1944735478, 1, 1, 53720) does not "
                             "match the picture's (1944735478, 1, 0, 53720)"),
                 "header for another segment");
    check::equal(refusal([&] { txm::slot_symbols(src_bf, Q_TEST, short_spec, 0); }),
                 std::string("the beacon file holds segment 1, not 0"), "beacon segment");
    check::equal(refusal([&] { txm::slot_symbols(src_bf, Q_TEST, short_spec, std::nullopt,
                                                 header_for(sp, 1, "K1ABC/P", "FN43")); }),
                 std::string("a beacon file carries its own header; it does not match the one given"),
                 "beacon header");
    check::equal(refusal([&] { txm::slot_symbols(src_bf, Q_TEST, short_spec, std::nullopt, std::nullopt, false); }),
                 std::string("a beacon file holds only the int8 latents"), "beacon use_int8 false");
    bfm::BeaconFile part = bf;
    part.latents_i8.resize(1000);
    const txm::Source src_part = part;
    check::equal(refusal([&] { txm::slot_symbols(src_part, Q_TEST, short_spec); }),
                 std::string("frame 'short' sends 4096 latents; the source holds 1000"), "too few latents");
    check::equal(refusal([&] { txm::slot_symbols(src_part, Q_TEST, fr::get_l("l-tiny")); }),
                 std::string("frame 'l-tiny' sends 1024 latents; the source holds 1000"), "too few latents, L");
    check::equal(refusal([&] { txm::slot_air_latents(src_sp, 2); }), std::string("IndexError: list index out of range"),
                 "slot_air_latents past the list");
    check::equal(refusal([&] { txm::slot_air_latents(src_sp, -1); }), std::string("group must be 0..2, not -1"),
                 "slot_air_latents -1 (the last segment, then its group)");
}

int main() {
    try {
        const auto sp = closed_form_picture();
        check::equal(sp.picture_id, std::int64_t(0x73EA4EF6), "the picture");
        const auto bf = bfm::from_picture(sp, 1, header_for(sp, 1));
        test_symbols(sp, bf);
        test_slot_samples();
        test_check_pilot();
        test_refusals(sp, bf);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_tx");
}
