// native/core/qrss/tx's pass audio (PassAudio, transmit_audio, write_wav)
// and native/core/qrss/wav against values and texts of the Python
// reference (sstvae/qrss/tx.py's transmit_audio, sstvae/wavio.py through
// scipy.io.wavfile.write).
//
// The sources are test_tx.cpp's: test_picture.cpp's closed-form F, mode B,
// codec 0xD1D8, as a stored picture and as its segment-1 beacon file
// (FSK; and an OOK one):
//   g = np.arange(3 * 52800.0)
//   F = np.sin(np.fmod(0.001 * g * g, 6.283)) + 0.3 * np.cos(g)
//   sp = StoredPicture.from_latents(F, 0xD1D8, 1)
//   h  = HeaderFields("K1ABC/P", "FN42", sp.picture_id, 1, 1, 0xD1D8)
//   bf = beaconfile.from_picture(sp, 1, h)
//   a  = tx.transmit_audio(src, 1954752, carrier, spec=..., ...)
// pinned as len(a), float(np.sum(a * a)).hex() and a few float(a[i]).hex().
// The CE audio calls sin/cos, so it is pinned to 1e-12 (bit for bit on
// the reference platform, which tests/test_qrss_native_tx_audio.py checks
// against Python itself); L carries the clipper's FFT, so 1e-9.
//
// Beyond the pins, what no Python comparison can show: every block size
// gives the same bits (zeros' signs included), float32 is one rounding of
// float64, read/seek walk the pass, and the WAV files are scipy's bytes.

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/beaconfile.hpp"
#include "qrss/frame.hpp"
#include "qrss/header.hpp"
#include "qrss/npsum.hpp"
#include "qrss/picture.hpp"
#include "qrss/rational.hpp"
#include "qrss/tx.hpp"
#include "qrss/wav.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace pc = sstvae::qrss::picture;
namespace bfm = sstvae::qrss::beaconfile;
namespace hdr = sstvae::qrss::header;
namespace fr = sstvae::qrss::frame;
namespace txm = sstvae::qrss::tx;
namespace wav = sstvae::qrss::wav;

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

hdr::HeaderFields header_for(const pc::StoredPicture& sp, std::int64_t segment) {
    hdr::HeaderFields h;
    h.callsign = "K1ABC/P";
    h.grid = "FN42";
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

void near(double got, double want, double tol, const std::string& what) {
    ++check::checks;
    if (!(std::abs(got - want) <= tol)) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "got %a, want %a (|diff| %.3g > %.3g)", got, want, std::abs(got - want), tol);
        check::fail(what, buf);
    }
}

// A pass in slot Q_TEST on `spec`, every other argument the reference's default.
txm::PassSpec on(const char* spec) {
    txm::PassSpec p;
    p.q = Q_TEST;
    p.spec = fr::get_any(spec);
    return p;
}

// Bit for bit, signed zeros included.
bool same_bits(std::span<const double> a, std::span<const double> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

std::vector<std::uint8_t> file_bytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::string hex(std::span<const std::uint8_t> b) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (const std::uint8_t c : b) {
        s += d[c >> 4];
        s += d[c & 15];
    }
    return s;
}

struct Pin {
    const char* name;
    std::int64_t n;
    double energy;
    std::array<std::int64_t, 9> idx;
    std::array<double, 9> val;
    const char* wav16;   // the first 44 bytes of wavio.write_wav's file
    const char* wavf;    // the first 58 bytes of wavio.write_wav_float's
};

// A: CE tiny, the beacon file (FSK), carrier 1500 (the int)
const Pin PIN_A = {"A: CE tiny, beacon file, 1500",
                   546800,
                   0x1.a2fbf073194c0p+15,
                   {93661, 93698, 96000, 108345, 273400, 273401, 523149, 523199, 546799},
                   {-0x1.de0f76cda5bd8p-18, -0x1.0012ff851def8p-7, 0x1.45d5c093ea20ep-2, 0x1.8caa7f264c028p-3,
                    -0x1.aed023c06574ep-2, -0x1.a4a7b408d5edap-2, -0x1.2b89911a1e1a2p-6, 0x1.8c09c8009d960p-19, 0.0},
                   "5249464604b0100057415645666d74201000000001000100401f0000803e00000200100064617461e0af1000",
                   "52494646f25f210057415645666d74201200000003000100401f0000007d00000400200000006661637404000000f05708"
                   "0064617461c05f2100"};
// B: CE short, the .qrsp with its header, carrier 1500.3 (a float), lead-in 5 s, OOK
const Pin PIN_B = {"B: CE short, .qrsp, 1500.3, lead-in 5, OOK",
                   2166094,
                   0x1.e07b680e9de5ap+17,
                   {53661, 53698, 96000, 108345, 1083047, 1083048, 2142443, 2142493, 2166093},
                   {-0x1.ebe8bf797ac3bp-18, -0x1.151ae2d3fc041p-7, -0x1.f77b6b7bafb1fp-6, 0x1.6cdc62c4c5ec2p-2,
                    -0x1.789f5a964056dp-6, 0x1.d0ae829ffeb6cp-2, -0x1.088e271a3b50ap-6, 0x1.c2815fb0602b3p-21, 0.0},
                   nullptr, nullptr};
// C: CE short, the beacon file, carrier 2650, chunk_s 7.3
const Pin PIN_C = {"C: CE short, beacon file, 2650, chunk_s 7.3",
                   2166094,
                   0x1.f414e0422ba2ap+17,
                   {93661, 93698, 96000, 108345, 1083047, 1083048, 2142443, 2142493, 2166093},
                   {0x1.18e9c0ee37d84p-19, -0x1.6023378094ae9p-7, 0x1.45d5c093ea20ep-2, -0x1.ab26b1c5f33c3p-2,
                    0x1.54ec1a8751985p-2, -0x1.f46acad54abbap-2, 0x1.93633c21d82c7p-11, 0x1.92f89ba2a3de5p-20, -0.0},
                   nullptr, nullptr};
// D: l-tiny, the .qrsp with its header, pilot Fraction(30003, 20), lead-in 2 s
const Pin PIN_D = {"D: l-tiny, .qrsp, 30003/20, lead-in 2",
                   548740,
                   0x1.cc0a146da7684p+15,
                   {75504, 75541, 96000, 108345, 274370, 274371, 525310, 525360, 548739},
                   {0x1.258386b9a6cf7p-27, -0x1.148d5749c4b37p-21, 0x1.074d8d6cc0030p-1, -0x1.ecf05c24c37eap-2,
                    -0x1.c260843ddd5cfp-3, 0x1.01686e307f8aap-1, 0x1.32d8a1e276b56p-21, -0x1.52d5a28901aa5p-27, 0.0},
                   "524946462cbf100057415645666d74201000000001000100401f0000803e0000020010006461746108bf1000",
                   "52494646427e210057415645666d74201200000003000100401f0000007d00000400200000006661637404000000845f08"
                   "0064617461107e2100"};
// E: l-short, the beacon file, pilot 1200.0
const Pin PIN_E = {"E: l-short, beacon file, 1200.0",
                   2165488,
                   0x1.09238161bf7ccp+18,
                   {91504, 91541, 96000, 108345, 1082744, 1082745, 2142046, 2142096, 2165487},
                   {0x1.e7b695ecb2f5ap-28, 0x1.315ac89de1ea3p-22, 0x1.03e93fb33380ep-1, -0x1.93b1f927e89f4p-2,
                    0x1.c8899e33160f4p-9, -0x1.366353e3c3493p-6, -0x1.40dcf29356be3p-22, 0x1.f580df823c19dp-28, 0.0},
                   nullptr, nullptr};

void check_pin(const Pin& p, const std::vector<double>& a, double tol) {
    const std::string n = p.name;
    check::equal(static_cast<std::int64_t>(a.size()), p.n, n + ": length");
    if (static_cast<std::int64_t>(a.size()) != p.n) return;
    const std::vector<double> sq = [&] {
        std::vector<double> s(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) s[i] = a[i] * a[i];
        return s;
    }();
    near(qrss::npsum::sum(std::span<const double>(sq)), p.energy, tol * p.energy, n + ": energy");
    for (std::size_t j = 0; j < p.idx.size(); ++j)
        near(a[static_cast<std::size_t>(p.idx[j])], p.val[j], tol, n + ": a[" + std::to_string(p.idx[j]) + "]");
}

// Every block size gives the same bits as one render of the whole pass.
void check_blocks(txm::PassAudio& audio, const std::vector<double>& whole, const std::string& what,
                  std::initializer_list<std::int64_t> sizes) {
    for (const std::int64_t b : sizes) {
        audio.seek(0);
        std::vector<double> got;
        got.reserve(whole.size());
        std::vector<double> buf(static_cast<std::size_t>(b));
        for (;;) {
            const std::size_t k = audio.read(std::span<double>(buf));
            got.insert(got.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(k));
            if (k < buf.size()) break;
        }
        check::is_true(same_bits(got, whole), what + ": blocks of " + std::to_string(b) + " are the whole");
        check::equal(audio.position(), audio.size(), what + ": read to the end");
        check::equal(audio.read(std::span<double>(buf)), std::size_t(0), what + ": nothing past the end");
    }
}

}  // namespace

static void test_ce(const pc::StoredPicture& sp, const bfm::BeaconFile& bf) {
    check::current_step = "CE";
    const txm::Source src_sp = sp, src_bf = bf;

    txm::PassSpec a;
    a.q = Q_TEST;
    a.spec = fr::get("tiny");
    txm::PassAudio pa(src_bf, a);
    check::is_true(!pa.waveform_l(), "A is CE");
    check::equal(pa.size(), txm::slot_samples(a.spec), "A: size is slot_samples");
    check::equal(pa.n_pre(), std::int64_t(96000), "A: n_pre");
    check::is_true(pa.carrier() == qrss::Rational(1500, 1), "A: carrier 1500/1");
    check::equal(pa.chunk(), std::int64_t(546800), "A: one chunk (120 s is past the pass)");
    const std::vector<double> wa = txm::transmit_audio(src_bf, a);
    check_pin(PIN_A, wa, 1e-12);
    check_blocks(pa, wa, "A", {1, 7, 4096, 100000, 546800, 1000000});
    // float32 is one rounding of float64, from any block size
    {
        std::vector<float> f(wa.size());
        pa.render(0, std::span<float>(f));
        bool ok = true;
        for (std::size_t i = 0; i < f.size(); ++i) {
            const float w = static_cast<float>(wa[i]);
            ok = ok && std::memcmp(&f[i], &w, 4) == 0;
        }
        check::is_true(ok, "A: float32 render is float(np.float64)");
        pa.seek(0);
        std::vector<float> g;
        std::vector<float> buf(4093);
        for (std::size_t k; (k = pa.read(std::span<float>(buf))) > 0;) g.insert(g.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(k));
        check::is_true(g.size() == f.size() && std::memcmp(g.data(), f.data(), 4 * f.size()) == 0, "A: float32 blocks");
    }
    // render of a middle stretch, and out of range
    {
        std::vector<double> mid(5000);
        pa.render(270000, std::span<double>(mid));
        check::is_true(same_bits(mid, std::span<const double>(wa).subspan(270000, 5000)), "A: render(i0, m)");
        check::equal(refusal([&] { pa.render(546000, std::span<double>(mid)); }).substr(0, 23),
                     std::string("IndexError: PassAudio: "), "A: render past the end");
        check::equal(refusal([&] { pa.seek(-1); }).substr(0, 23), std::string("IndexError: PassAudio: "), "A: seek -1");
        pa.seek(pa.size());
        check::equal(pa.position(), pa.size(), "A: seek to the end");
    }

    txm::PassSpec b;
    b.q = Q_TEST;
    b.carrier = 1500.3;
    b.spec = fr::get("short");
    b.segment = 1;
    b.header = header_for(sp, 1);
    b.lead_in_s = 5.0;
    b.ook = true;
    txm::PassAudio pb(src_sp, b);
    check::is_true(pb.carrier() == qrss::Rational(15003, 10), "B: 1500.3 is 15003/10");
    const std::vector<double> wb = txm::transmit_audio(src_sp, b);
    check_pin(PIN_B, wb, 1e-12);
    check_blocks(pb, wb, "B", {7, 4096, 100000});

    txm::PassSpec c;
    c.q = Q_TEST;
    c.carrier = 2650;
    c.spec = fr::get("short");
    c.chunk_s = 7.3;
    txm::PassAudio pcc(src_bf, c);
    check::equal(pcc.chunk(), std::int64_t(58400), "C: chunk int(7.3 * 8000)");
    const std::vector<double> wc = txm::transmit_audio(src_bf, c);
    check_pin(PIN_C, wc, 1e-12);
    check::is_true(std::signbit(wc.back()), "C: the reference's -0.0 in an unkeyed chunk");
    check_blocks(pcc, wc, "C", {4096, 262144});
    // The chunking only decides the zeros' signs: the default chunking
    // gives the same values.
    c.chunk_s = 120.0;
    const std::vector<double> wc2 = txm::transmit_audio(src_bf, c);
    bool values = wc2.size() == wc.size();
    for (std::size_t i = 0; values && i < wc.size(); ++i) values = wc[i] == wc2[i];
    check::is_true(values, "C: chunk_s changes no value");
}

static void test_l(const pc::StoredPicture& sp, const bfm::BeaconFile& bf) {
    check::current_step = "L";
    const txm::Source src_sp = sp, src_bf = bf;

    txm::PassSpec d;
    d.q = Q_TEST;
    d.carrier = qrss::Rational(30003, 20);
    d.spec = fr::get_l("l-tiny");
    d.segment = 1;
    d.header = header_for(sp, 1);
    d.lead_in_s = 2.0;
    txm::PassAudio pd(src_sp, d);
    check::is_true(pd.waveform_l(), "D is L");
    check::is_true(pd.carrier() == qrss::Rational(30003, 20), "D: the fraction as given");
    const std::vector<double> wd = txm::transmit_audio(src_sp, d);
    check_pin(PIN_D, wd, 1e-9);
    check_blocks(pd, wd, "D", {1, 7, 4096, 100000, 548740});

    txm::PassSpec e;
    e.q = Q_TEST;
    e.carrier = 1200.0;
    e.spec = fr::get_l("l-short");
    txm::PassAudio pe(src_bf, e);
    const std::vector<double> we = txm::transmit_audio(src_bf, e);
    check_pin(PIN_E, we, 1e-9);
    check_blocks(pe, we, "E", {4096, 100000});
    // chunk_s is invisible on L: the interpolation is per sample.
    e.chunk_s = 0.5;
    const std::vector<double> we2 = txm::transmit_audio(src_bf, e);
    check::is_true(same_bits(we, we2), "E: chunk_s 0.5, the same bits");
}

static void test_refusals(const pc::StoredPicture& sp, const bfm::BeaconFile& bf, const bfm::BeaconFile& bf_ook) {
    check::current_step = "refusals";
    const txm::Source src_sp = sp, src_bf = bf, src_ook = bf_ook;
    const auto h = header_for(sp, 1);
    auto pass = [&](const char* spec) {
        txm::PassSpec p;
        p.q = Q_TEST;
        p.spec = fr::get_any(spec);
        p.segment = 1;
        p.header = h;
        return p;
    };
    auto text = [](const txm::Source& s, const txm::PassSpec& p) {
        return refusal([&] { txm::PassAudio a(s, p); });
    };
    {
        auto p = pass("l-short");
        p.ook = true;
        check::equal(text(src_sp, p),
                     std::string("waveform L keys its callsign windows by FSK of its pilot; on-off keying is waveform "
                                 "CE's option (spec 2.7)"),
                     "L with ook");
        p.ook = false;
        check::equal(text(src_ook, on("l-short")),
                     std::string("this beacon file keys its callsign windows on and off, which waveform L does not "
                                 "do: make an FSK one (spec 2.7)"),
                     "an OOK beacon file on L");
    }
    {
        auto p = pass("l-short");
        p.carrier = 310;
        check::equal(text(src_sp, p),
                     std::string("an L pilot at 310 Hz puts the data down to 287.9 Hz, below the 300 Hz receivers "
                                 "search"),
                     "L pilot too low");
        p.carrier = 322.0;
        check::equal(text(src_sp, p),
                     std::string("an L pilot at 322.0 Hz puts the data down to 299.9 Hz, below the 300 Hz receivers "
                                 "search"),
                     "L pilot too low, a float");
    }
    {
        auto p = pass("short");
        p.carrier = 2800;
        check::equal(text(src_sp, p),
                     std::string("carrier 2800 Hz is outside the 300-2700 Hz audio band that receivers search"),
                     "CE carrier above the band");
        p.carrier = std::nan("");
        check::equal(text(src_sp, p),
                     std::string("carrier nan Hz is outside the 300-2700 Hz audio band that receivers search"),
                     "CE carrier NaN");
    }
    for (const char* s : {"short", "l-short"}) {
        auto p = pass(s);
        p.header.reset();
        check::equal(text(src_sp, p),
                     "frame '" + std::string(s) +
                         "' has callsign windows: a callsign is required to transmit (spec 2.7); pass header=",
                     std::string(s) + ": no callsign");
    }
    {
        auto p = pass("short");
        p.pre_s = 5.0;
        p.lead_in_s = 10.0;
        check::equal(text(src_sp, p),
                     std::string("pre_s = 5.0 s does not cover the keying start (10.24 s before t0 plus the ramp)"),
                     "pre_s not covering the lead-in");
        auto l = pass("l-short");
        l.pre_s = 0.5;
        check::equal(text(src_sp, l),
                     std::string("pre_s = 0.5 s does not cover the keying start (0.48 s before t0 plus the ramp)"),
                     "pre_s not covering, L");
        p = pass("short");
        p.pre_s = 12.00001;
        check::equal(text(src_sp, p), std::string("pre_s must be a whole number of samples at 8 kHz"), "pre_s whole");
        p.pre_s = std::nan("");
        check::equal(text(src_sp, p), std::string("cannot convert float NaN to integer"), "pre_s NaN");
        p.pre_s = std::numeric_limits<double>::infinity();
        check::equal(text(src_sp, p), std::string("OverflowError: cannot convert float infinity to integer"),
                     "pre_s inf");
        p = pass("short");
        p.lead_in_s = 11.0;
        check::equal(text(src_sp, p), std::string("lead-in must be 0 to 10 s, not 11.0"), "lead-in 11");
        l = pass("l-short");
        l.lead_in_s = -1.0;
        check::equal(text(src_sp, l), std::string("lead-in must be 0 to 10 s, not -1.0"), "lead-in -1, L");
    }
    {
        txm::PassSpec t = on("tiny");
        t.post_s = -100.0;
        check::equal(text(src_bf, t), std::string("negative dimensions are not allowed"), "negative length");
        t.post_s = std::nan("");
        check::equal(text(src_bf, t), std::string("cannot convert float NaN to integer"), "post_s NaN");
        t.post_s = 3.0;
        t.chunk_s = std::nan("");
        check::equal(text(src_bf, t), std::string("cannot convert float NaN to integer"), "chunk_s NaN");
        t.chunk_s = std::numeric_limits<double>::infinity();
        check::equal(text(src_bf, t), std::string("OverflowError: cannot convert float infinity to integer"),
                     "chunk_s inf");
        t.chunk_s = 120.0;
        t.segment = 0;
        check::equal(text(src_bf, t), std::string("the beacon file holds segment 1, not 0"), "beacon segment");
        // a frame without windows or a header needs no callsign
        txm::PassSpec u = on("tiny");
        u.segment = 1;
        check::equal(txm::PassAudio(src_sp, u).size(), std::int64_t(546800), "tiny from a .qrsp, no header");
    }
}

static void test_wav(const bfm::BeaconFile& bf, const pc::StoredPicture& sp) {
    check::current_step = "wav";
    // scipy.io.wavfile.write's headers (10 samples at 8 kHz)
    check::equal(hex(wav::header(wav::Format::Int16, 10)),
                 std::string("524946463800000057415645666d74201000000001000100401f0000803e0000020010006461746114000000"),
                 "int16 header");
    check::equal(hex(wav::header(wav::Format::Float32, 10)),
                 std::string("524946465a00000057415645666d74201200000003000100401f0000007d0000040020000000666163740400"
                             "00000a0000006461746128000000"),
                 "float32 header");
    // np.round half to even after (x / m * peak) * 32767
    check::equal(wav::to_int16(0.5 / 32767.0 * 2.0, 2.0, 1.0), std::int16_t(0), "to_int16 0.5 -> 0");
    check::equal(wav::to_int16(1.5 / 32767.0, 0.0, 1.0), std::int16_t(2), "to_int16 1.5 -> 2 (m 0: no scaling)");
    check::equal(wav::to_int16(-1.0, 1.0), std::int16_t(-31129), "to_int16 -peak");

    const auto dir = std::filesystem::temp_directory_path() / ("qrss_tx_audio_" + std::to_string(Q_TEST));
    std::filesystem::create_directories(dir);
    const txm::Source src_bf = bf, src_sp = sp;
    struct Case {
        const Pin* pin;
        txm::Source src;
        txm::PassSpec pass;
    };
    txm::PassSpec a = on("tiny");
    txm::PassSpec d = on("l-tiny");
    d.carrier = qrss::Rational(30003, 20);
    d.segment = 1;
    d.header = header_for(sp, 1);
    d.lead_in_s = 2.0;
    for (const Case& c : {Case{&PIN_A, src_bf, a}, Case{&PIN_D, src_sp, d}}) {
        const txm::PassAudio audio(c.src, c.pass);
        const std::vector<double> x = txm::transmit_audio(c.src, c.pass);
        const std::string n = c.pin->name;
        const auto p16 = dir / "a16.wav", pf = dir / "af.wav", q16 = dir / "q16.wav", qf = dir / "qf.wav";
        txm::write_wav(audio, p16.string());
        txm::write_wav(audio, pf.string(), true);
        wav::write_int16(q16.string(), x);
        wav::write_float32(qf.string(), x);
        const auto b16 = file_bytes(p16), bf32 = file_bytes(pf);
        check::equal(b16.size(), std::size_t(44 + 2 * x.size()), n + ": int16 WAV size");
        check::equal(bf32.size(), std::size_t(58 + 4 * x.size()), n + ": float32 WAV size");
        check::equal(hex(std::span<const std::uint8_t>(b16).first(44)), std::string(c.pin->wav16),
                     n + ": int16 header (scipy's)");
        check::equal(hex(std::span<const std::uint8_t>(bf32).first(58)), std::string(c.pin->wavf),
                     n + ": float32 header (scipy's)");
        check::is_true(b16 == file_bytes(q16), n + ": streamed int16 WAV is the array's");
        check::is_true(bf32 == file_bytes(qf), n + ": streamed float32 WAV is the array's");
        // the samples: float32 as cast, int16 peak-normalised
        const double m = wav::peak_of(x);
        bool ok16 = true, okf = true;
        for (std::size_t i = 0; i < x.size(); ++i) {
            std::int16_t s = 0;
            std::memcpy(&s, b16.data() + 44 + 2 * i, 2);
            ok16 = ok16 && s == wav::to_int16(x[i], m);
            float f = 0;
            std::memcpy(&f, bf32.data() + 58 + 4 * i, 4);
            okf = okf && f == static_cast<float>(x[i]);
        }
        check::is_true(ok16, n + ": int16 samples");
        check::is_true(okf, n + ": float32 samples");
    }
    // An empty pass: numpy's refusal of max() for int16, a header for float32.
    check::equal(refusal([&] { wav::write_int16((dir / "e.wav").string(), std::span<const double>()); }),
                 std::string("zero-size array to reduction operation maximum which has no identity"), "empty int16");
    wav::write_float32((dir / "e.wav").string(), std::span<const double>());
    check::equal(file_bytes(dir / "e.wav").size(), std::size_t(58), "empty float32: the header");
    // A path that cannot be opened is a WriteError carrying errno.
    try {
        wav::write_float32((dir / "no" / "such" / "x.wav").string(), std::span<const double>());
        check::fail("unwritable path", "no error");
    } catch (const wav::WriteError& e) {
        check::equal(e.err, ENOENT, "unwritable path: ENOENT");
    }
    std::filesystem::remove_all(dir);
}

int main() {
    check::Watchdog dog(600, "qrss_core_tx_audio");
    try {
        const auto sp = closed_form_picture();
        check::equal(sp.picture_id, std::int64_t(0x73EA4EF6), "the picture");
        const auto h = header_for(sp, 1);
        const auto bf = bfm::from_picture(sp, 1, h);
        const auto bf_ook = bfm::from_picture(sp, 1, h, true);
        test_ce(sp, bf);
        test_l(sp, bf);
        test_refusals(sp, bf, bf_ook);
        test_wav(bf, sp);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL (%s): %s\n", check::current_step.load(), e.what());
        return 1;
    }
    return check::report("qrss_tx_audio");
}
