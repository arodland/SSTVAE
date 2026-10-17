// native/core/qrss/ldemod against sstvae/qrss/ldemod.py: every function on
// the deterministic synthetic track of demod_synth.hpp, on l-short with the
// header unknown (the spread copy as extra variance), known (subtracted)
// and with the tracker's measured-reference mask, pinned from Python by
// SHA-256 of the outputs' bytes and the kappa estimates as doubles; plus
// the refusals. tests/test_qrss_native_ldemod.py compares against Python
// on tracker runs. The pins (Q = 123456789):
//
//   tr = synth(lframe.get("l-short"), 0.054, hdr_bits, with_meas)
//   ldemod.symbol_estimates_l(tr, lframe.layout(tr.spec).data, ldemod.G_DATA_L)
//   ldemod.data_estimates_l(tr); ldemod.kappa_estimates_l(tr); ldemod.calibrate_l(tr)
//   ldemod.extract_l(None, tr, tr.spec, Q)
//   ldemod.extract_l(None, synth(lframe.get("l-tiny")), lframe.get("l-tiny"), Q, "plain")

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "demod_synth.hpp"
#include "qrss/demod.hpp"
#include "qrss/frame.hpp"
#include "qrss/ldemod.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace dm = sstvae::qrss::demod;
namespace ld = sstvae::qrss::ldemod;
namespace fr = sstvae::qrss::frame;
using qrss_check::digest_is;

constexpr std::uint64_t Q = 123456789;

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    } catch (const std::exception& e) {
        check::fail(what, std::string("threw another type: ") + e.what());
        return;
    }
    check::fail(what, "did not throw");
}

static const char* const R_DIGEST = "acf588d0fa9f1eb915e60c807013ed5c7fc923ea4721f7283f7c0a87a12fdbbd";
static const char* const S2N_DIGEST = "4415066af265d1cfdd9eb42e0d2d1c20f6411917286f87b146f5afdb37aba69e";
static const char* const S2N_X_DIGEST = "460b2ef7bb320ebd5260a0e8b81c18af559d7885d52144f189ffa038e706a10b";

static void test_unknown() {
    const auto& spec = fr::get_l("l-short");
    const auto s = demod_synth::make(spec, 0.054);
    const auto v = s.view();
    const auto se = ld::symbol_estimates_l(v, fr::layout(spec)->data, ld::G_DATA_L);
    digest_is(se.r, R_DIGEST, "symbol r");
    digest_is(se.s2n, S2N_DIGEST, "symbol s2n");
    const auto de = ld::data_estimates_l(v);
    digest_is(de.y, "0dafaf579a5b01e100b6906817e35aa5a76884ff43811ec1b26c422b78d074b8", "y_x, header unknown");
    digest_is(de.s2n, S2N_X_DIGEST, "s2n_x");
    check::equal(de.extra, 0.054, "extra = rho");
    const auto ke = ld::kappa_estimates_l(v);
    check::equal(ke.size(), std::size_t{2}, "two estimates");
    if (ke.size() == 2) {
        check::equal(ke[0].first, 2.471042771906787, "reference kappa (no meas)");
        check::equal(ke[0].second, 0.005055625642449583, "reference variance");
        check::equal(ke[1].first, -1.0970358662183957, "data kappa");
        check::equal(ke[1].second, 0.0035474182931181882, "data variance");
    }
    const auto cal = ld::calibrate_l(v);
    check::equal(cal.kappa, 0.5, "calibrate_l kappa");
    check::is_true(!cal.suspect, "calibrate_l not suspect");
    const auto ex = ld::extract_l(v, spec, Q);
    digest_is(ex.z, "3e75666765555cf7aeac61b3ef0677c7e620c291575382dd1cbcfe8638a0954b", "z");
    digest_is(ex.w, "b3ed88dae5b80328975e11ad31d56e5a614d21ddd6f87c5e1a6e630752d90c81", "w");
    digest_is(ex.llr, "457949b907a36cd667d705d6801dd9008b7b12575d4a1f7d12cbdda5780b2cd5", "llr (block + spread)");
    digest_is(ex.s2, "8816db05b0d07af496888399faef40b7ec2ee925c715ffbff3d5c737e7cf7cf3", "s2");
    check::is_true(ex.llr_block && ex.llr_spread, "both LLR parts");
}

static void test_known_and_meas() {
    const auto& spec = fr::get_l("l-short");
    const auto s = demod_synth::make(spec, 0.054, true, true);
    const auto v = s.view();
    const auto de = ld::data_estimates_l(v);
    digest_is(de.y, "09065848695acbe5ae8e8af883bda5c32e0cf6822dc7a64e57314b091258b0a7", "y_x, header subtracted");
    digest_is(de.s2n, S2N_X_DIGEST, "s2n_x, known");
    check::equal(de.extra, 0.0, "extra 0 once known");
    const auto ke = ld::kappa_estimates_l(v);
    check::equal(ke.size(), std::size_t{2}, "two estimates, known");
    if (ke.size() == 2) {
        check::equal(ke[0].first, 2.483397874688257, "reference kappa with the measured pull");
        check::equal(ke[0].second, 0.005022946112099965, "its variance");
        check::equal(ke[1].first, -0.9408465599777078, "data kappa, known");
        check::equal(ke[1].second, 0.0033352158951326146, "its variance, known");
    }
    const auto ex = ld::extract_l(v, spec, Q);
    digest_is(ex.z, "b1589b790e629ba4f3af73f18d2f6f736169d46b4e51dbdab71b5113cd8fc054", "z, known");
    digest_is(ex.w, "a448504e617de6159da272a0edffe929d2f0f3a396ed424e0bd9aed01b8a3f46", "w, known");
    digest_is(ex.llr, "5f5de95d1166c09252ea0ed18eb67c77e115cc5ea0adc328394d4aa15bb76cc4", "llr, known (block only)");
    check::is_true(ex.llr_block && !ex.llr_spread, "no spread part once known");

    const auto u = demod_synth::make(spec, 0.054, false, true);
    const auto ku = ld::kappa_estimates_l(u.view());
    if (ku.size() == 2) check::equal(ku[0].first, 2.483397874688257, "meas alone moves the reference estimate");
}

static void test_tiny_plain() {
    const auto& spec = fr::get_l("l-tiny");
    const auto s = demod_synth::make(spec);
    const auto ex = ld::extract_l(s.view(), spec, Q, "plain");
    digest_is(ex.z, "20a2741c0ab56fe6241ea0b910a10faa1950992e1c3365f5f71556b2e2046a53", "l-tiny plain z");
    digest_is(ex.w, "fe17c82372c37fb0772218900c749091220cdbb8d39f0a7105003ff86d5a3a26", "l-tiny plain w");
    check::equal(ex.llr.size(), std::size_t{0}, "no header");
    check::equal(ex.kappa, 0.5, "l-tiny kappa");
}

static void test_refusals() {
    check::equal(ld::d_pass_x(fr::get_l("l-short")), 0x1.f4cb1897a67a6p-5, "d_pass_x with the spread copy");
    check::equal(ld::d_pass_x(fr::get_l("l-tiny")), 0x1.db22d0e560419p-5, "d_pass_x without");
    check::equal(ld::d_pass_x(fr::get_l("l-short")), ld::D_PASS_LX, "D_PASS_LX");
    const auto& ce = fr::get("short");
    dm::TrackView v;
    v.spec = &ce;
    throws<fr::WrongWaveform>([&] { (void)ld::calibrate_l(v); },
                              "ldemod.calibrate_l is L's; 'short' is waveform 0: use demod's", "calibrate_l on CE");
    throws<fr::WrongWaveform>([&] { (void)ld::kappa_estimates_l(v); },
                              "ldemod.kappa_estimates_l is L's; 'short' is waveform 0: use demod's",
                              "kappa_estimates_l on CE");
    throws<fr::WrongWaveform>([&] { (void)ld::extract_l(v, fr::get("tiny"), Q); },
                              "ldemod.extract_l is L's; 'tiny' is waveform 0: use demod's", "extract_l on CE");
    auto s = demod_synth::make(fr::get_l("l-short"), 0.054, true);
    s.bits.resize(5);
    throws<std::invalid_argument>([&] { (void)ld::data_estimates_l(s.view()); },
                                  "hdr_bits must be 2474 values of 0/1", "short header bits");
}

int main() {
    try {
        test_unknown();
        test_known_and_meas();
        test_tiny_plain();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_ldemod");
}
