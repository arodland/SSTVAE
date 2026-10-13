// native/core/qrss/frame (CE and L frames) against values pinned from the
// Python reference, sstvae/qrss/frame.py and lframe.py.
//
// Digests are hashlib.sha256 of numpy's bytes: a layout's digest feeds its
// arrays in field order (pre, hdr_ref, hdr_bits, hdr_spare -- L: hdr_sym --
// data_ref, data, ref, known_idx, known_val, pos, win_pos, is_ref), and an
// assembled frame's is of the float64 (CE) or complex128 (L) symbols. The
// inputs are integer closed forms both sides evaluate exactly:
//   x_i  = ((i * 2654435761) % 1000003 - 500001) / 250000.0   (int64)
//   bit_i = ((i * 40503 + 17) >> 6) & 1                       (i < 2474)
// tests/test_qrss_native_frame.py compares the two implementations directly.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
namespace fr = sstvae::qrss::frame;
using qrss_check::digest;
using qrss_check::digest_is;

namespace {

struct PresetPin {
    const char* name;
    std::int64_t n_data_sym, n_sym, n_pos, n_ref;
    double duration_s, keyed_end_pos;
    const char* layout_digest;
};

// for s in PRESETS, LEGACY, PROTOTYPES, lframe.PRESETS (pins.py beside the
// generator's notes; values straight from the reference's properties)
const PresetPin PRESET_PINS[] = {
    {"full", 53974, 57274, 58810, 3539, 0x1.bdab666666666p+10, 0x1.cb73000000000p+15, "574dbfb0b40590b1d39162d27ffd27a3ca5c75e878ebbac6a3a9298af5451d35"},
    {"medium", 17477, 20777, 21545, 1258, 0x1.468a999999999p+9, 0x1.50a2000000000p+14, "7971c623569ba8f54b1eb56f33bedeebde0c7a9ccd337803e1142948d8421e0c"},
    {"short", 4370, 7670, 8438, 439, 0x1.ff8dc28f5c28fp+7, 0x1.07ac000000000p+13, "694a766ec106612aaf13056eb17119ff32958d6737d609a5c81dc3f3929fd9e0"},
    {"tiny", 1093, 1753, 1753, 69, 0x1.a91a3d70a3d70p+5, 0x1.b800000000000p+10, "84049e68003edd3e60c0f842ff0ef419de2cd92b955cdb0335e096da86d35ef5"},
    {"full-v1", 53974, 57274, 58810, 3539, 0x1.bdab666666666p+10, 0x1.cb73000000000p+15, "574dbfb0b40590b1d39162d27ffd27a3ca5c75e878ebbac6a3a9298af5451d35"},
    {"medium-v1", 17477, 20777, 21545, 1258, 0x1.468a999999999p+9, 0x1.50a2000000000p+14, "7971c623569ba8f54b1eb56f33bedeebde0c7a9ccd337803e1142948d8421e0c"},
    {"short-v1", 4370, 7670, 8438, 439, 0x1.ff8dc28f5c28fp+7, 0x1.07ac000000000p+13, "694a766ec106612aaf13056eb17119ff32958d6737d609a5c81dc3f3929fd9e0"},
    {"full-spreadonly", 53974, 54634, 56170, 3374, 0x1.a9a9ccccccccdp+10, 0x1.b6d3000000000p+15, "3f6e8e863aed479ac1f143f27fba9259e4486be3a05189303c6c3c2fb6545cb9"},
    {"l-full", 26987, 28637, 29405, 1770, 0x1.bdab666666666p+10, 0x1.cb72000000000p+14, "f57d18221ae5aeaa9f658c825a2db7d105194938205d4f81030c3c151bd109a0"},
    {"l-medium", 8738, 10388, 10772, 629, 0x1.4686b851eb852p+9, 0x1.509c000000000p+13, "5a524d11c7fc94140d82b3c3d969493d3bda896bb8906a78f6c250a972707cfb"},
    {"l-short", 2184, 3834, 4218, 219, 0x1.ff6eb851eb852p+7, 0x1.0798000000000p+12, "a490dc8b67e20045d90f4928dbef4bcb1c8f62a9110efcfb1e5d300c5f19700b"},
    {"l-tiny", 547, 877, 877, 35, 0x1.a95851eb851ebp+5, 0x1.ba00000000000p+9, "320c07d5475befcb0c3767aa7ab6c76b6b5956e5c7e6f5c9eed3012474231ae2"},
};

// assemble(spec, bits if spec.has_header else None, x_of(spec.n_data))
const std::pair<const char*, const char*> ASSEMBLE_PINS[] = {
    {"full", "360ddac119454a766911b5d689c10aca57d3272220ce8aa25b95c1c03ad68b8b"},
    {"medium", "321d8953b64fb4b62c69850611b677065b3ff859e869108dbcab5040e9cf3825"},
    {"short", "c8b0042c76dfbde6759124fc9ee3a9a073182edd4ced6b88d150a186585a97d5"},
    {"tiny", "55cfd9928b9b715f2e06ac91993b60de016f3964f4b54ae4e1a28b5fdde54626"},
    {"full-v1", "bc65121b00b27f632228e84b46819422935292a8e2af23c210d45b15f2f3efac"},
    {"medium-v1", "27628b4b3bdf9d321289a5d2958f9ef10cee97a6dc2ac0a05667981fea11cec6"},
    {"short-v1", "8f2085f32f93ac16e5ef56f4ebcc54af7bbc0b1277cc56825ec94b95c988e969"},
    {"full-spreadonly", "9afca6237088b1246c654ab66f0f91ef0c1c0e04b83b2d42d4db2aafd770ae84"},
    {"l-full", "4854d2118a37a2a9062c4b4bba570aafee35e8a1950a744f9acfea87a1d4712f"},
    {"l-medium", "91d2e49dea2d6182d0285f1bab204d0b144701042a8e20a5391b2e7a5a6aab12"},
    {"l-short", "c840aeed22bd0e1da439e3d715c73546d13f4a287bc1a6ea0fca97a571d11324"},
    {"l-tiny", "31f91e6d1cca2cd9e1aa3360646abb6368689731d348a32fd8a06dfdf8af8434"},
};

std::vector<double> x_of(std::size_t n) {
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = static_cast<double>((static_cast<std::int64_t>(i) * 2654435761LL) % 1000003 - 500001) / 250000.0;
    return x;
}

std::vector<std::uint8_t> bits() {
    std::vector<std::uint8_t> b(qrss::config::N_HDR_BITS);
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<std::uint8_t>(((static_cast<std::int64_t>(i) * 40503 + 17) >> 6) & 1);
    return b;
}

template <typename T>
void feed(util::Sha256& h, const std::vector<T>& v) {
    h.update(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(v.data()), v.size() * sizeof(T)));
}

std::string layout_digest(const fr::Layout& lay) {
    util::Sha256 h;
    feed(h, lay.pre);
    feed(h, lay.hdr_ref);
    if (lay.waveform == fr::Waveform::L) {
        feed(h, lay.hdr_sym);
    } else {
        feed(h, lay.hdr_bits);
        feed(h, lay.hdr_spare);
    }
    feed(h, lay.data_ref);
    feed(h, lay.data);
    feed(h, lay.ref);
    feed(h, lay.known_idx);
    if (lay.waveform == fr::Waveform::L)
        feed(h, lay.known_val_l);
    else
        feed(h, lay.known_val_ce);
    feed(h, lay.pos);
    feed(h, lay.win_pos);
    feed(h, lay.is_ref);
    return util::to_hex(h.digest());
}

template <typename F>
std::string refusal(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "(no refusal)";
}

template <typename F>
bool wrong_waveform(F&& f) {
    try {
        f();
    } catch (const fr::WrongWaveform&) {
        return true;
    }
    return false;
}

const fr::FrameSpec& any(std::string_view name) { return fr::get_any(name); }

}  // namespace

static void test_presets() {
    check::equal(fr::presets(qrss::config::PresetGroup::Presets).size(), std::size_t(4), "PRESETS");
    check::equal(fr::presets(qrss::config::PresetGroup::Legacy).size(), std::size_t(3), "LEGACY");
    check::equal(fr::presets(qrss::config::PresetGroup::Prototypes).size(), std::size_t(1), "PROTOTYPES");
    check::equal(fr::presets(qrss::config::PresetGroup::LPresets).size(), std::size_t(4), "lframe.PRESETS");
    for (const PresetPin& p : PRESET_PINS) {
        const fr::FrameSpec& s = any(p.name);
        const std::string n = p.name;
        check::equal(s.name(), n, n + " name");
        check::equal(s.n_data_sym(), p.n_data_sym, n + " n_data_sym");
        check::equal(s.n_sym(), p.n_sym, n + " n_sym");
        check::equal(s.n_pos(), p.n_pos, n + " n_pos");
        check::equal(s.n_ref(), p.n_ref, n + " n_ref");
        check::is_true(s.duration_s() == p.duration_s, n + " duration_s bitwise");
        check::is_true(s.keyed_end_pos() == p.keyed_end_pos, n + " keyed_end_pos");
        check::equal(s.keyed_start_pos(), -8.0, n + " keyed_start_pos");
        check::equal(layout_digest(*fr::layout(s)), std::string(p.layout_digest), n + " layout");
        check::equal(static_cast<std::int64_t>(fr::layout(s)->data.size()), s.n_data_payload(), n + " data slots");
    }
    // The table's own figures (design 2.2): FULL 1782.678125 s, 14,261,425 samples.
    const fr::FrameSpec& full = fr::get("full");
    check::is_true(full.duration_s() == 1782.678125, "FULL duration");
    check::is_true(full.win_start_pos() == std::vector<std::int64_t>{13888, 28734, 43580, 58426}, "FULL windows");
    check::is_true(full.last_window_ends_frame() && full.keyed_end_pos() == full.n_pos() - 0.5, "FULL ends in a window");
    const fr::FrameSpec& tiny = fr::get("tiny");
    check::is_true(!tiny.last_window_ends_frame() && tiny.keyed_end_pos() == tiny.n_sym() - 1 + 8, "TINY keyed end");
    check::is_true(!tiny.has_header() && fr::get("full-spreadonly").has_header(), "has_header");
    check::equal(fr::get_l("l-full").n_data_payload(), std::int64_t(25300), "L_FULL payload");
}

static void test_windows_and_positions() {
    const double full_spans[4][2] = {
        {0x1.a4f7000000000p+8, 0x1.b09ad70a3d70ap+8}, {0x1.b37dfae147ae1p+9, 0x1.b94fe66666666p+9},
        {0x1.4a403ae147ae1p+10, 0x1.4d2930a3d70a4p+10}, {0x1.bac17851eb852p+10, 0x1.bdaa6e147ae14p+10}};
    const double l_spans[4][2] = {
        {0x1.a4f31eb851eb8p+8, 0x1.b096f5c28f5c2p+8}, {0x1.b37c0a3d70a3dp+9, 0x1.b94df5c28f5c2p+9},
        {0x1.4a3f428f5c28fp+10, 0x1.4d283851eb852p+10}, {0x1.bac0800000000p+10, 0x1.bda975c28f5c2p+10}};
    const auto a = fr::get("full").window_spans_s(), b = fr::get_l("l-full").window_spans_s();
    for (int w = 0; w < 4; ++w) {
        check::is_true(a[w].first == full_spans[w][0] && a[w].second == full_spans[w][1], "FULL span " + std::to_string(w));
        check::is_true(b[w].first == l_spans[w][0] && b[w].second == l_spans[w][1], "L_FULL span " + std::to_string(w));
    }
    const fr::FrameSpec& full = fr::get("full");
    check::equal(full.pos_of(13887), std::int64_t(13887), "pos_of before a window");
    check::equal(full.pos_of(13888), std::int64_t(13888 + 384), "pos_of after a window");
    check::equal(full.pos_of(-5), std::int64_t(-5), "pos_of negative");
    check::equal(fr::get_l("l-full").pos_of(28636), std::int64_t(28636 + 3 * 192), "L pos_of");
    check::is_true(full.time_s(2.0) == 2.0 * qrss::config::T_SYM, "time_s");
}

static void test_lengths() {
    const std::pair<std::int64_t, std::int64_t> ds[] = {{0, 0}, {1, 2}, {2, 3}, {15, 16}, {16, 18}, {17, 19},
                                                        {100, 107}, {4096, 4370}, {50600, 53974}, {-1, -1},
                                                        {-16, -17}, {-17, -18}, {-100, -106}};
    for (const auto& [n, want] : ds) check::equal(fr::data_symbols(n), want, "_data_symbols(" + std::to_string(n) + ")");
    struct L3 {
        std::int64_t n, j0, want;
    };
    const L3 dsl[] = {{512, 0, 547}, {512, 1320, 546}, {2048, 1320, 2184}, {25300, 1320, 26987},
                      {0, 0, 0}, {0, 5, 0}, {1, 1320, 1}, {7, 3, 7}, {-5, 0, -5}, {-5, 7, -5}, {100, -40, 107}};
    for (const L3& c : dsl)
        check::equal(fr::data_symbols_l(c.n, c.j0), c.want,
                     "lframe._data_symbols(" + std::to_string(c.n) + ", " + std::to_string(c.j0) + ")");
    const std::pair<std::int64_t, std::int64_t> rb[] = {{0, 0}, {1, 1}, {15, 1}, {16, 1}, {17, 2},
                                                        {-1, 0}, {-16, -1}, {-17, -1}};
    for (const auto& [j, want] : rb) check::equal(fr::refs_below(j), want, "_refs_below(" + std::to_string(j) + ")");
    check::equal(fr::N_HDR_SLOTS_L, std::int64_t(1237), "N_HDR_SLOTS");
    // A large count is found by bisection, not by counting up (2^40 would spin).
    const std::int64_t big = std::int64_t(1) << 40;
    const std::int64_t L = fr::data_symbols(big);
    check::is_true(L - (L + 15) / 16 >= big && (L - 1) - (L + 14) / 16 < big, "_data_symbols(2^40) is the smallest");
}

static void test_assemble() {
    const auto b = bits();
    const fr::HdrBits hb = fr::HdrBits::of(b);
    for (const auto& [name, want] : ASSEMBLE_PINS) {
        const fr::FrameSpec& s = any(name);
        const auto x = x_of(static_cast<std::size_t>(s.n_data()));
        const fr::HdrBits* bp = s.has_header() ? &hb : nullptr;
        const std::string got = s.waveform() == fr::Waveform::L ? digest(fr::assemble_l(s, bp, x))
                                                                : digest(fr::assemble(s, bp, x));
        check::equal(got, std::string(want), std::string("assemble ") + name);
    }
    const fr::FrameSpec& sh = fr::get("short");
    digest_is(fr::spread_signs(sh, &hb), "76d632c0dfe83750074ccf4a5264351fb934e3e767e1b0366c3f56ff544f71cc", "spread_signs");
    digest_is(fr::spread_symbols(sh, &hb), "4f052c34e1468d8565ca0961d756db5508edc9f6575274e331384df668eee210",
              "spread_symbols");
    digest_is(fr::data_latents(fr::get_l("l-short"), x_of(4096), &hb),
              "7cc11e8a1f48b6225af4a68812a55add337a2554253799522ffb952cfc0147c3", "data_latents");
    const auto v = x_of(1000);
    const auto p = fr::pair(v);
    digest_is(p, "79d0e9f417746cb99245d06c2dfba631ba167469c852ee56d2925e4dee8eaa44", "pair");
    digest_is(fr::unpair(p), "65529e1c58b5ab05e531abb5ab6c897a55548e976aa47b21f8ed99050555ede2", "unpair(pair)");

    // disassemble inverts assemble on the data and header classes.
    const auto sym = fr::assemble(fr::get("short-v1"), &hb, x_of(4096));
    const auto [h, d] = fr::disassemble(std::span<const double>(sym), fr::get("short-v1"));
    bool ok = h.size() == b.size() && d == x_of(4096);
    for (std::size_t i = 0; ok && i < b.size(); ++i) ok = h[i] == 1.0 - 2.0 * b[i];
    check::is_true(ok, "disassemble(assemble)");
    const auto lsym = fr::assemble_l(fr::get_l("l-tiny"), nullptr, x_of(1024));
    const auto [lh, ld] = fr::disassemble_l(std::span<const std::complex<double>>(lsym), fr::get_l("l-tiny"));
    double worst = 0;
    const auto x = x_of(1024);
    for (std::size_t i = 0; i < x.size(); ++i) worst = std::max(worst, std::abs(ld[i] - x[i]));
    check::is_true(lh.empty() && ld.size() == 1024 && worst <= 4.5e-16 * 4, "L disassemble round trip");
}

static void test_names() {
    check::equal(fr::twin(fr::get("short"), 1).name(), std::string("l-short"), "twin to L");
    check::equal(fr::twin(fr::get_l("l-tiny"), 0).name(), std::string("tiny"), "twin to CE");
    check::is_true(fr::twin(fr::get("short"), 0) == fr::get("short"), "twin to itself");
    check::equal(fr::plain(fr::get("short")).name(), std::string("short-v1"), "plain is LEGACY");
    check::is_true(fr::plain(fr::get("tiny")) == fr::get("tiny"), "plain of a plain frame");
    const auto custom = fr::FrameSpec::make(fr::Waveform::CE, "custom", 660, 2640, 2000, {}, std::nullopt);
    check::is_true(custom.hdr_rho() == qrss::config::SPREAD_RHO, "default hdr_rho");
    const auto pl = fr::plain(custom);
    check::is_true(pl.name() == "custom-v1" && pl.hdr_rho() == 0.0 && pl.n_data() == 2000, "plain of a custom frame");
    check::equal(fr::get_any("short").name(), std::string("short"), "get_any CE");
    check::equal(refusal([] { fr::get("huge"); }), std::string("unknown frame 'huge'; one of full, medium, short, tiny"),
                 "get refusal");
    check::equal(refusal([] { fr::get_any("l-huge"); }),
                 std::string("unknown L frame 'l-huge'; one of l-full, l-medium, l-short, l-tiny"), "get_l refusal");
    check::equal(refusal([] { fr::twin(fr::get("full-v1"), 1); }),
                 std::string("unknown L frame 'l-full-v1'; one of l-full, l-medium, l-short, l-tiny"), "twin refusal");
    // spec.name[2:] slices code points, not UTF-8 bytes: "\u00e9-short"[2:] is
    // "short", and "\u00e9short"[2:] is "hort" (bytes would give "-short" and
    // "short", refusing the first and finding the second).
    const auto e_dash = fr::FrameSpec::make(fr::Waveform::L, "\xc3\xa9-short", 330, 1320, 4096, {2500, 3834});
    check::equal(fr::twin(e_dash, 0).name(), std::string("short"), "twin slices code points");
    const auto e_short = fr::FrameSpec::make(fr::Waveform::L, "\xc3\xa9short", 330, 1320, 4096, {2500, 3834});
    check::equal(refusal([&] { fr::twin(e_short, 0); }),
                 std::string("unknown frame 'hort'; one of full, medium, short, tiny"), "twin slices code points, refusal");
    const auto wide = fr::FrameSpec::make(fr::Waveform::L, "\xf0\x9f\x98\x80\xc3\xa9tiny", 330, 0, 1024, {});
    check::equal(fr::twin(wide, 0).name(), std::string("tiny"), "twin past a 4-byte code point");
    check::equal(fr::py_repr(std::string_view("it's")), std::string("\"it's\""), "repr with a quote");
    check::equal(fr::py_repr(std::string_view("a\\b\n")), std::string("'a\\\\b\\n'"), "repr escapes");
}

static void test_refusals() {
    using W = fr::Waveform;
    const auto make = [](W w, std::int64_t n_pre, std::int64_t n_hdr, std::int64_t n_data, std::vector<std::int64_t> cw,
                         std::optional<double> rho = std::nullopt) {
        return [=] { fr::FrameSpec::make(w, "bad", n_pre, n_hdr, n_data, cw, rho); };
    };
    check::equal(refusal(make(W::CE, 660, 2640, 4096, {5000, 4000})), std::string("bad: cw_after must be strictly increasing"),
                 "not increasing");
    check::equal(refusal(make(W::CE, 660, 2640, 4096, {5000, 5000})), std::string("bad: cw_after must be strictly increasing"),
                 "not strictly increasing");
    check::equal(refusal(make(W::CE, 660, 2640, 4096, {3300})),
                 std::string("bad: window after stream symbol 3300 is outside (3300, 7670]"), "inside the header");
    check::equal(refusal(make(W::CE, 660, 2640, 4096, {7671})),
                 std::string("bad: window after stream symbol 7671 is outside (3300, 7670]"), "past the end");
    check::equal(refusal(make(W::CE, 660, 1000, 4096, {})), std::string("n_hdr must be 0 or 2640, not 1000"), "n_hdr");
    check::equal(refusal(make(W::CE, 600, 2640, 4096, {})), std::string("n_pre must be 660, not 600"), "n_pre");
    check::equal(refusal(make(W::CE, 660, 2640, 0, {})), std::string("n_data must be positive, not 0"), "n_data");
    check::equal(refusal(make(W::L, 330, 1320, 4095, {})), std::string("n_data must be positive and even, not 4095"),
                 "L n_data odd");
    check::equal(refusal(make(W::L, 330, 1000, 4096, {})), std::string("n_hdr must be 0 or 1320, not 1000"), "L n_hdr");
    check::equal(refusal(make(W::CE, 660, 2640, 4096, {}, 1.0)), std::string("hdr_rho must be in [0, 1), not 1.0"), "rho 1");
    check::equal(refusal(make(W::CE, 660, 2640, 4096, {}, 1e-5 - 1)), std::string("hdr_rho must be in [0, 1), not -0.99999"),
                 "rho negative");
    check::equal(refusal(make(W::CE, 660, 2640, 4096, {}, std::nan(""))), std::string("hdr_rho must be in [0, 1), not nan"),
                 "rho nan");
    // Python's float repr, which the refusal texts carry.
    check::equal(fr::py_repr(1e-05), std::string("1e-05"), "repr 1e-05");
    check::equal(fr::py_repr(0.0001), std::string("0.0001"), "repr 0.0001");
    check::equal(fr::py_repr(1e16), std::string("1e+16"), "repr 1e16");
    check::equal(fr::py_repr(1e15), std::string("1000000000000000.0"), "repr 1e15");
    check::equal(fr::py_repr(-2.5e-300), std::string("-2.5e-300"), "repr -2.5e-300");
    check::equal(fr::py_repr(0.054), std::string("0.054"), "repr 0.054");
    check::equal(fr::py_repr(123.456), std::string("123.456"), "repr 123.456");

    const auto b = bits();
    const fr::HdrBits hb = fr::HdrBits::of(b);
    const fr::FrameSpec& sh = fr::get("short");
    const auto x = x_of(4096);
    check::equal(refusal([&] { fr::assemble(sh, nullptr, x); }),
                 std::string("frame 'short' has a header: hdr_bits is required"), "bits required");
    check::equal(refusal([&] { fr::assemble(sh, &hb, x_of(4097)); }), std::string("x has shape (4097,), expected (4096,)"),
                 "x shape");
    fr::HdrBits two = hb;
    two.values[7] = 2.0;
    check::equal(refusal([&] { fr::assemble(sh, &two, x); }), std::string("hdr_bits must be 2474 values of 0/1"), "bit 2");
    check::equal(refusal([&] { fr::assemble(fr::get("tiny"), &hb, x_of(1024)); }),
                 std::string("frame 'tiny' has no header but hdr_bits were given"), "bits on TINY");
    fr::HdrBits empty;
    check::equal(refusal([&] { fr::assemble(fr::get("tiny"), &empty, x_of(1024)); }), std::string("(no refusal)"),
                 "empty bits on TINY");
    check::equal(refusal([] { fr::pair(std::vector<double>(3)); }), std::string("pair takes an even number of values, not shape (3,)"),
                 "pair odd");
    check::is_true(wrong_waveform([&] { fr::assemble(fr::get_l("l-short"), &hb, x); }), "CE assemble refuses L");
    check::is_true(wrong_waveform([&] { fr::assemble_l(sh, &hb, x); }), "L assemble refuses CE");
}

// The layout cache from several threads at once, over more specs than it
// holds (16), so lookups, builds and evictions interleave: every layout handed
// out is the one a single thread builds.
static void test_layout_cache_threads() {
    std::vector<fr::FrameSpec> specs;
    for (int k = 0; k < 24; ++k) {
        const bool l = k % 2;
        specs.push_back(fr::FrameSpec::make(l ? fr::Waveform::L : fr::Waveform::CE, "t" + std::to_string(k),
                                            l ? 330 : 660, 0, 64 + 2 * k, {}));
    }
    std::vector<std::string> want;
    for (const auto& s : specs) want.push_back(layout_digest(*fr::layout(s)));
    std::vector<int> bad(8, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
        threads.emplace_back([&, t] {
            for (int i = 0; i < 300; ++i) {
                const std::size_t k = static_cast<std::size_t>((i * 7 + t * 5) % 24);
                if (layout_digest(*fr::layout(specs[k])) != want[k]) ++bad[static_cast<std::size_t>(t)];
            }
        });
    for (auto& th : threads) th.join();
    int total = 0;
    for (const int b : bad) total += b;
    check::equal(total, 0, "layouts built from 8 threads");
}

// py_format is Python's format(v, spec): a width's padding stays spaces
// (it once read as a locale's decimal point: ".1500.017"), and a NaN of
// either sign prints unsigned, as Python prints it.
static void test_py_format() {
    check::equal(fr::py_format(1500.017, "%9.3f"), std::string(" 1500.017"), "py_format width");
    check::equal(fr::py_format(-2.5, "%8.1f"), std::string("    -2.5"), "py_format negative width");
    check::equal(fr::py_format(287.9, "%.1f"), std::string("287.9"), "py_format point");
    check::equal(fr::py_format(-std::numeric_limits<double>::quiet_NaN(), "%.1f"), std::string("nan"),
                 "py_format -nan");
    check::equal(fr::py_format(-std::numeric_limits<double>::quiet_NaN(), "%6.1f"), std::string("   nan"),
                 "py_format -nan width");
    check::equal(fr::py_format(-std::numeric_limits<double>::infinity(), "%.1f"), std::string("-inf"),
                 "py_format -inf");
}

int main() {
    check::report_crashes_instead_of_prompting();
    try {
        test_presets();
        test_windows_and_positions();
        test_lengths();
        test_assemble();
        test_names();
        test_refusals();
        test_layout_cache_threads();
        test_py_format();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_frame");
}
