// native/core/qrss/api (the in-process transmit API), codec-free: its
// constants against the reference's numbers, its slot and frame helpers,
// qrss_transmit.py's refusals in order with the reference's texts, the pass
// audio against tx::transmit_audio, and encode_picture's framing with a
// stub encoder.
//
// The source is test_tx.cpp's closed-form picture (mode B, codec 0xD1D8,
// picture ID 0x73ea4ef6). The refusal texts are the reference's, from
//     tx.transmit_audio(sp, q, carrier, spec=..., segment=..., header=...)
// tests/test_qrss_native_api.py compares the API with qrss_transmit.py
// directly where Python is available.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "check.hpp"
#include "images/images.hpp"
#include "qrss/api.hpp"
#include "qrss/beaconfile.hpp"
#include "qrss/header.hpp"
#include "qrss/picture.hpp"
#include "qrss/tx.hpp"

using namespace sstvae;
namespace api = sstvae::qrss::api;
namespace pc = sstvae::qrss::picture;
namespace bfm = sstvae::qrss::beaconfile;
namespace hdr = sstvae::qrss::header;
namespace fr = sstvae::qrss::frame;
namespace txm = sstvae::qrss::tx;

namespace {

constexpr std::int64_t Q_TEST = 1954752;   // 2025-10-01T00:00Z

// f() throws an E (and nothing else).
template <typename E, typename F>
void throws(F&& f, const std::string& what) {
    bool ok = false;
    try {
        f();
    } catch (const E&) {
        ok = true;
    } catch (...) {
    }
    check::is_true(ok, what + " throws");
}

std::vector<double> closed_form_latents() {
    std::vector<double> v(3 * 52800);
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double g = static_cast<double>(i);
        v[i] = std::sin(std::fmod(0.001 * g * g, 6.283)) + 0.3 * std::cos(g);
    }
    return v;
}

pc::StoredPicture closed_form_picture() { return pc::StoredPicture::from_latents(closed_form_latents(), 0xD1D8, 1); }

api::PassRequest tiny(std::int64_t q = Q_TEST) {
    api::PassRequest r;
    r.q = q;
    r.callsign = "K1ABC";
    r.frame = "tiny";
    return r;
}

void test_constants() {
    check::equal(api::SAMPLE_RATE, 8000, "8 kHz");
    check::equal(api::SLOT_S, 900.0, "a quarter hour");
    check::equal(api::T0_OFFSET_S, 1.0, "t0 = quarter hour + 1 s (frontend.slot_t0)");
    check::equal(api::AUDIO_LEAD_S, 12.0, "pre_s");
    check::equal(api::AUDIO_TAIL_S, 3.0, "post_s");
    check::equal(api::PASS_S, 1782.678125, "FULL frame duration");
    check::equal(api::PASS_S, fr::get("full").duration_s(), "PASS_S is FULL's");
    check::equal(api::PASS_S, fr::get_l("l-full").duration_s(), "... and l-full's");
    check::equal(api::CARRIER_MIN_HZ, 300.0, "band low");
    check::equal(api::CARRIER_MAX_HZ, 2700.0, "band high");
    check::equal(api::CARRIER_DEFAULT_HZ, 1500.0, "default carrier");
    check::equal(api::LEAD_IN_MAX_S, 10.0, "lead-in limit");
    // tx::PassSpec's own defaults are the same numbers.
    const txm::PassSpec p;
    check::equal(p.pre_s, api::AUDIO_LEAD_S, "PassSpec pre_s");
    check::equal(p.post_s, api::AUDIO_TAIL_S, "PassSpec post_s");
    check::equal(p.amplitude, qrss::config::TX_AMPLITUDE, "PassSpec amplitude");
    check::equal(p.chunk_s, qrss::config::TX_CHUNK_S, "PassSpec chunk_s");
    check::equal(p.carrier.hz(), api::CARRIER_DEFAULT_HZ, "PassSpec carrier");
}

void test_names_and_slots() {
    check::equal(api::passes_for("A"), 1, "A");
    check::equal(api::passes_for("B"), 2, "B");
    check::equal(api::passes_for("C"), 3, "C");
    check::equal(api::passes_for("D"), 0, "D");
    check::equal(api::passes_for(""), 0, "empty");
    check::is_true(api::mode_index("C") == 2 && !api::mode_index("a"), "mode_index");
    check::is_true(api::waveform_named("L") == api::Waveform::L && api::waveform_named("CE") == api::Waveform::CE &&
                       !api::waveform_named("ce"),
                   "waveform_named");
    check::equal(api::frame_spec(api::Waveform::CE, "short").name(), std::string("short"), "CE short");
    check::equal(api::frame_spec(api::Waveform::L, "short").name(), std::string("l-short"), "L short");
    check::equal(api::frame_spec(api::Waveform::L).name(), std::string("l-full"), "L default");
    throws<std::invalid_argument>([] { api::frame_spec(api::Waveform::CE, "l-short"); }, "an L name");

    check::equal(api::parse_slot("2025-10-01T00:00Z"), Q_TEST, "parse_slot");
    check::equal(api::slot_iso(Q_TEST), std::string("2025-10-01T00:00Z"), "slot_iso");
    check::equal(api::slot_of(1759276800.0), Q_TEST, "slot_of");
    check::equal(api::slot_of(-900.0), std::int64_t(-1), "slot_of before 1970");
    throws<std::invalid_argument>([] { api::slot_of(1759276801.0); }, "off the quarter hour");
    throws<std::invalid_argument>([] { api::slot_of(std::nan("")); }, "NaN");
    check::equal(api::quarter_hour_s(Q_TEST), 1759276800.0, "quarter_hour_s");
    check::equal(api::t0_s(Q_TEST), 1759276801.0, "t0_s");
    check::equal(api::audio_start_s(Q_TEST), 1759276789.0, "audio_start_s");
    // tx.slot_samples(frame.FULL) and of l-full, from the reference.
    check::equal(api::pass_samples(api::Waveform::CE), std::int64_t(14381304), "CE full samples");
    check::equal(api::pass_samples(api::Waveform::L), std::int64_t(14381183), "L full samples");
    check::equal(api::audio_end_s(Q_TEST, api::Waveform::CE), 1759276789.0 + 14381304.0 / 8000.0, "audio_end_s");
}

void test_refusals() {
    const api::Source sp = closed_form_picture();
    check::equal(api::problem(sp, tiny()), std::string(), "a pass that can go out");
    auto refused = [&](api::PassRequest r, const std::string& want, const std::string& what) {
        check::equal(api::problem(sp, r), want, what);
        throws<std::invalid_argument>([&] { api::pass_audio(sp, r); }, what + " (pass_audio)");
    };
    api::PassRequest r = tiny();
    r.callsign.reset();
    refused(r, "a .qrsp needs --callsign: every pass sends it in the header and in the callsign windows",
            "no callsign");
    r = tiny();
    r.freq_hz = 100.0;
    refused(r, "carrier 100.0 Hz is outside the 300-2700 Hz audio band that receivers search", "carrier");
    r = tiny();
    r.segment = 5;
    refused(r, "segment must be 0..1, not 5", "segment");
    r.freq_hz = 100.0;   // the carrier is checked first
    refused(r, "carrier 100.0 Hz is outside the 300-2700 Hz audio band that receivers search", "order");
    r = tiny();
    r.lead_in_s = 11.0;
    refused(r, "lead-in must be 0 to 10 s, not 11.0", "lead-in");
    r = tiny();
    r.callsign = "K1ABC!";
    refused(r, "callsign must be 1-8 characters of [A-Z0-9/], got 'K1ABC!'", "callsign");
    r = tiny();
    r.grid = "XX";
    refused(r, "grid must be a 4-character Maidenhead locator, got 'XX'", "grid");
    r = tiny();
    r.waveform = api::Waveform::L;
    r.freq_hz = 310.0;
    refused(r, "an L pilot at 310.0 Hz puts the data down to 287.9 Hz, below the 300 Hz receivers search", "pilot");
    r.ook = true;   // before anything else
    refused(r, "--ook is waveform CE's: waveform L keys its callsign windows by FSK of its pilot", "L ook");
    r = tiny(-1);
    refused(r, "q must fit in uint64, not -1", "a slot before 1970");
    r.callsign = "K1ABC!";   // ... is refused last
    refused(r, "callsign must be 1-8 characters of [A-Z0-9/], got 'K1ABC!'", "q last");
    r = tiny();
    r.frame = "huge";
    refused(r, "frame must be one of full, medium, short, tiny, not 'huge'", "frame");

    // A beacon file: its own segment, grid and keying.
    const pc::StoredPicture& pic = std::get<pc::StoredPicture>(sp);
    hdr::HeaderFields h;
    h.callsign = "K1ABC";
    h.grid = "FN42";
    h.picture_id = pic.picture_id;
    h.mode = pic.mode;
    h.segment = 1;
    h.codec_id = pic.codec_id;
    const api::Source fsk = bfm::from_picture(pic, 1, h, false);
    const api::Source ook = bfm::from_picture(pic, 1, h, true);
    r = tiny();
    r.segment = 7;
    r.grid = "XX";
    check::equal(api::problem(fsk, r), std::string(), "a beacon file ignores segment and grid");
    r.callsign.reset();
    check::equal(api::problem(fsk, r), std::string(), "... and needs no callsign");
    r.callsign = "W1AW";
    check::equal(api::problem(fsk, r), std::string("the beacon file is for K1ABC, not W1AW"), "another callsign");
    r = tiny();
    r.ook = true;
    check::equal(api::problem(fsk, r),
                 std::string("the beacon file keys its callsign windows FSK, and a beacon sends what its file says: "
                             "make the .bin with qrss_beacon.py --ook instead of passing --ook here"),
                 "--ook with an FSK file");
    check::equal(api::problem(ook, r), std::string(), "--ook with an OOK file");
    r.ook = false;
    r.waveform = api::Waveform::L;
    check::equal(api::problem(ook, r),
                 std::string("this beacon file keys its callsign windows on and off, which waveform L does not do: "
                             "make an FSK one (spec 2.7)"),
                 "an OOK file in waveform L");
}

void test_audio() {
    const pc::StoredPicture pic = closed_form_picture();
    const api::Source sp = pic;
    for (const api::Waveform w : {api::Waveform::CE, api::Waveform::L}) {
        api::PassRequest r = tiny();
        r.waveform = w;
        r.freq_hz = 1234.5;
        r.lead_in_s = 1.5;
        const std::vector<float> got = api::pass_audio(sp, r);
        // tx.transmit_audio with qrss_transmit.py's arguments.
        txm::PassSpec p;
        p.q = static_cast<std::uint64_t>(Q_TEST);
        p.carrier = txm::Carrier(1234.5);
        p.spec = api::frame_spec(w, "tiny");
        p.segment = 0;
        hdr::HeaderFields h;
        h.callsign = "K1ABC";
        h.picture_id = pic.picture_id;
        h.mode = pic.mode;
        h.codec_id = pic.codec_id;
        p.header = h;
        p.lead_in_s = 1.5;
        p.ook = false;
        const std::vector<double> want = txm::transmit_audio(sp, p);
        check::equal(static_cast<std::int64_t>(got.size()), static_cast<std::int64_t>(want.size()), "length");
        check::equal(static_cast<std::int64_t>(got.size()), api::pass_samples(w, "tiny"), "pass_samples");
        bool same = got.size() == want.size();
        for (std::size_t i = 0; same && i < got.size(); ++i) same = got[i] == static_cast<float>(want[i]);
        check::is_true(same, std::string("pass_audio is transmit_audio as float32, ") +
                                 std::string(api::waveform_name(w)));
        // The stream renders the same samples in any blocks.
        txm::PassAudio s = api::pass_stream(sp, r);
        std::vector<float> blocks(got.size());
        std::size_t at = 0;
        while (at < blocks.size()) at += s.read(std::span<float>(blocks.data() + at, std::min<std::size_t>(4099, blocks.size() - at)));
        check::is_true(blocks == got, "pass_stream in blocks");
    }

    // write_wav is the CLI's file, through the same stream.
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sstvae_qrss_test_api";
    std::filesystem::create_directories(dir);
    const api::PassRequest r = tiny();
    api::write_wav(sp, r, (dir / "a.wav").string(), true);
    txm::write_wav(api::pass_stream(sp, r), (dir / "b.wav").string(), true);
    auto bytes = [](const std::filesystem::path& p) {
        std::ifstream f(p, std::ios::binary);
        return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    };
    check::is_true(bytes(dir / "a.wav") == bytes(dir / "b.wav") && !bytes(dir / "a.wav").empty(), "write_wav");
    api::PassRequest bad = r;
    bad.segment = 9;
    throws<std::invalid_argument>([&] { api::write_wav(sp, bad, (dir / "c.wav").string(), true); },
                                         "a refused pass");
    check::is_true(!std::filesystem::exists(dir / "c.wav"), "... writes no file");
    std::filesystem::remove_all(dir);
}

void test_read_source() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sstvae_qrss_test_api_src";
    std::filesystem::create_directories(dir);
    const pc::StoredPicture pic = closed_form_picture();
    pc::save_qrsp((dir / "p.qrsp").string(), pic);
    const api::Source a = api::read_source((dir / "p.qrsp").string());
    check::is_true(std::holds_alternative<pc::StoredPicture>(a) &&
                       std::get<pc::StoredPicture>(a).picture_id == pic.picture_id,
                   "a stored picture");
    hdr::HeaderFields h;
    h.callsign = "K1ABC";
    h.picture_id = pic.picture_id;
    h.mode = pic.mode;
    h.codec_id = pic.codec_id;
    bfm::write((dir / "b.bin").string(), bfm::from_picture(pic, 0, h, true));
    const api::Source b = api::read_source((dir / "b.bin").string());
    check::is_true(std::holds_alternative<bfm::BeaconFile>(b) && std::get<bfm::BeaconFile>(b).ook, "a beacon file");
    {
        std::ofstream f(dir / "bad.bin", std::ios::binary);
        f << "QRSB....";
    }
    throws<std::invalid_argument>([&] { api::read_source((dir / "bad.bin").string()); }, "a short beacon");
    throws<std::runtime_error>([&] { api::read_source((dir / "nope").string()); }, "a missing file");
    std::filesystem::remove_all(dir);
}

void test_encode_picture() {
    // The stub sees what the encoder would: to_array(fit(picture)).
    images::Picture small(320, 240);
    for (std::size_t i = 0; i < small.rgb.size(); ++i) small.rgb[i] = static_cast<std::uint8_t>((i * 7) % 251);
    images::ImageArray seen;
    const api::Encoder stub = [&](const images::ImageArray& img) {
        seen = img;
        return closed_form_latents();
    };
    const pc::StoredPicture sp = api::encode_picture(small, 2, 0xBEEF, stub);
    const images::ImageArray want = images::to_array(images::fit(small));
    check::equal(seen.width, 640, "fitted width");
    check::equal(seen.height, 480, "fitted height");
    check::is_true(seen.chw == want.chw, "the encoder gets to_array(fit(picture))");
    const pc::StoredPicture direct = pc::StoredPicture::from_latents(closed_form_latents(), 0xBEEF, 2);
    check::equal(sp.picture_id, direct.picture_id, "then StoredPicture.from_latents");
    check::equal(sp.segs.size(), std::size_t(3), "mode C");
    // A 640x480 picture goes in untouched.
    images::Picture full(640, 480);
    for (std::size_t i = 0; i < full.rgb.size(); ++i) full.rgb[i] = static_cast<std::uint8_t>(i % 256);
    api::encode_picture(full, 0, 1, stub);
    bool same = seen.chw.size() == full.rgb.size();
    for (std::size_t c = 0; same && c < 3; ++c)
        for (std::size_t p = 0; same && p < 640 * 480; ++p)
            same = seen.chw[c * 640 * 480 + p] == static_cast<float>(full.rgb[p * 3 + c]) / 255.0f;
    check::is_true(same, "a 640x480 picture is not resized");
    throws<std::invalid_argument>([&] { api::encode_picture(full, 3, 1, stub); }, "mode 3");
    throws<std::invalid_argument>([&] { api::encode_picture(full, 0, 1 << 16, stub); }, "codec ID");
    throws<std::invalid_argument>([&] { api::encode_picture(full, 0, 1, api::Encoder()); }, "no encoder");
}

}  // namespace

int main() {
    try {
        test_constants();
        test_names_and_slots();
        test_refusals();
        test_audio();
        test_read_source();
        test_encode_picture();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_api");
}
