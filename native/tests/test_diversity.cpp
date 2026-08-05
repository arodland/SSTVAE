// sstvae::modem::diversity -- MRC combining of independently
// demodulated branches. Port of tests/test_diversity.py; see
// docs/diversity-reception.md for the derivation.
//
// Most cases here build DemodResult by hand (latents/weights/snr_db
// picked directly) rather than through a real modulate/demodulate
// round trip, since there is no C++ channel simulator (hfchannel is
// Python-only) to produce two branches with independently controlled
// SNR. One end-to-end case runs the real modem with hand-rolled AWGN to
// check the arithmetic against something that isn't also hand-rolled.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "images/types.hpp"
#include "modem/diversity.hpp"
#include "modem/modem.hpp"

using namespace sstvae;
using modem::BlindDemodResult;
using modem::DemodResult;

namespace {

const config::ModeSpec& mode_a() { return config::MODES[0]; }
const config::ModeSpec& mode_b() { return config::MODES[1]; }

DemodResult make_result(std::vector<double> latents, std::vector<double> weights,
                        double snr_db, const config::ModeSpec& mode = mode_a(),
                        int frames_received = -1) {
    DemodResult r;
    r.latents = std::move(latents);
    r.weights = std::move(weights);
    r.mode = mode;
    r.snr_db = snr_db;
    r.frames_received = frames_received >= 0 ? frames_received : mode.n_frames;
    r.freq_offset = 0.0;
    r.sync_metric = 1.0;
    r.preamble_start = 0;
    return r;
}

BlindDemodResult make_blind_result(std::vector<double> latents, std::vector<double> weights,
                                   double snr_db, int n_frames = 220) {
    BlindDemodResult r;
    r.latents = std::move(latents);
    r.weights = std::move(weights);
    r.freq_offset = 0.0;
    r.n_frames = n_frames;
    r.frame_offset = 0;
    r.frame0_start = 0;
    r.snr_db = snr_db;
    return r;
}

// Silences the lead-in/preamble/header of a real transmission, as if
// that antenna's preamble were too faded to detect at all: the header
// path fails to lock on it, but the frame pilots behind it (at exactly
// the same buffer position as an untouched branch's) are untouched, so
// blind acquisition still locks.
std::vector<double> zero_preamble(std::vector<double> x) {
    const auto end = static_cast<std::size_t>(
        config::LEADIN_SAMPLES + config::PREAMBLE_SAMPLES + config::HEADER_SAMPLES);
    std::fill(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(std::min(end, x.size())), 0.0);
    return x;
}

std::vector<double> test_latents(int n, std::uint64_t seed) {
    std::vector<double> out(static_cast<std::size_t>(n));
    std::uint64_t s = seed;
    for (double& v : out) {
        double acc = 0.0;
        for (int i = 0; i < 4; ++i) {
            s += 0x9E3779B97F4A7C15ULL;
            std::uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            acc += static_cast<double>(z >> 11) / 9007199254740992.0 - 0.5;
        }
        v = acc * 1.7;
    }
    double ms = 0.0;
    for (double v : out) ms += v * v;
    const double rms = std::sqrt(ms / static_cast<double>(out.size()));
    for (double& v : out) v /= rms;
    return out;
}

// Box-Muller, seeded independently of test_latents so two "branches" of
// the same clean transmission get uncorrelated noise.
void add_awgn(std::vector<double>& x, double sigma, std::uint64_t seed) {
    std::uint64_t s = seed;
    auto next_uniform = [&]() {
        s += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return (static_cast<double>(z >> 11) + 0.5) / 9007199254740992.0;
    };
    for (std::size_t i = 0; i + 1 < x.size(); i += 2) {
        const double u1 = next_uniform(), u2 = next_uniform();
        const double mag = sigma * std::sqrt(-2.0 * std::log(u1));
        x[i] += mag * std::cos(2.0 * std::numbers::pi * u2);
        x[i + 1] += mag * std::sin(2.0 * std::numbers::pi * u2);
    }
}

double latent_snr_db(const std::vector<double>& sent, const std::vector<double>& got,
                     const std::vector<double>& w) {
    double sig = 0.0, err = 0.0;
    int n = 0;
    for (std::size_t i = 0; i < sent.size(); ++i) {
        if (w[i] <= 0.0) continue;
        sig += sent[i] * sent[i];
        err += (sent[i] - got[i]) * (sent[i] - got[i]);
        ++n;
    }
    if (n == 0 || err <= 0.0) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(sig / err);
}

void test_single_branch_is_identity() {
    const DemodResult r = make_result({1.0, -1.0, 0.5}, {1.0, 0.8, 0.0}, 10.0);
    const DemodResult combined = modem::diversity::combine_demod_results({r});
    check::close(combined.latents, r.latents, 1e-12, "diversity/single: latents unchanged");
    check::close(combined.weights, r.weights, 1e-12, "diversity/single: weights unchanged");
}

void test_needs_at_least_one_branch() {
    bool threw = false;
    try {
        modem::diversity::combine_demod_results({});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "diversity/empty: raises on no branches");
}

void test_mode_mismatch_raises() {
    const DemodResult a = make_result({0.0}, {1.0}, 10.0, mode_a());
    const DemodResult b = make_result(std::vector<double>(mode_b().n_latents, 0.0),
                                      std::vector<double>(mode_b().n_latents, 1.0),
                                      10.0, mode_b());
    bool threw = false;
    try {
        modem::diversity::combine_demod_results({a, b});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "diversity/mismatch: raises on differing modes");
}

void test_combined_weight_never_exceeds_one() {
    const DemodResult a = make_result({1.0, 1.0, 1.0}, {1.0, 0.9, 0.3}, 15.0);
    const DemodResult b = make_result({1.0, 1.0, 1.0}, {1.0, 0.9, 0.3}, 15.0);
    const DemodResult combined = modem::diversity::combine_demod_results({a, b});
    for (double w : combined.weights)
        check::is_true(w <= 1.0 + 1e-9, "diversity/cap: combined weight <= 1");
}

void test_erasure_in_both_branches_stays_erased() {
    const DemodResult a = make_result({0.0, 2.0}, {0.0, 1.0}, 12.0);
    const DemodResult b = make_result({0.0, 2.0}, {0.0, 1.0}, 12.0);
    const DemodResult combined = modem::diversity::combine_demod_results({a, b});
    check::equal(combined.latents[0], 0.0, "diversity/erasure: latent stays 0");
    check::equal(combined.weights[0], 0.0, "diversity/erasure: weight stays 0");
    check::is_true(combined.weights[1] > 0.0, "diversity/erasure: the other latent unaffected");
}

void test_stronger_branch_dominates() {
    // Branch A carries the true value with a strong, unfaded channel;
    // branch B is at the same latent index but faded and noisy. The
    // combine should land close to A's value, not halfway between them.
    const DemodResult a = make_result({1.0}, {1.0}, 20.0);
    const DemodResult b = make_result({-5.0}, {0.05}, -5.0);
    const DemodResult combined = modem::diversity::combine_demod_results({a, b});
    check::is_true(std::abs(combined.latents[0] - 1.0) < 0.2,
                   "diversity/dominate: combine lands near the strong branch's value");
}

void test_frames_received_is_the_max() {
    const DemodResult a = make_result({0.0}, {1.0}, 10.0, mode_a(), 150);
    const DemodResult b = make_result({0.0}, {1.0}, 10.0, mode_a(), 200);
    const DemodResult combined = modem::diversity::combine_demod_results({a, b});
    check::equal(combined.frames_received, 200, "diversity/frames: reports the branch further along");
}

void test_branch_contribution_columns_sum_to_one_or_zero() {
    const DemodResult a = make_result({1.0, 0.0}, {0.9, 0.0}, 8.0);
    const DemodResult b = make_result({1.0, 0.0}, {0.4, 0.0}, 8.0);
    const auto frac = modem::diversity::branch_contribution(std::vector<DemodResult>{a, b});
    check::equal(frac.size(), std::size_t{2}, "diversity/contrib: one row per branch");
    for (std::size_t k = 0; k < a.latents.size(); ++k) {
        const double total = frac[0][k] + frac[1][k];
        const bool ok = std::abs(total - 1.0) < 1e-9 || std::abs(total) < 1e-9;
        check::is_true(ok, "diversity/contrib: column sums to 0 or 1");
    }
    check::is_true(frac[0][0] > frac[1][0],
                   "diversity/contrib: the stronger-weighted branch gets more credit");
}

void test_branch_contribution_single_branch_is_erasure_mask() {
    const DemodResult r = make_result({1.0, 0.0}, {0.6, 0.0}, 10.0);
    const auto frac = modem::diversity::branch_contribution(std::vector<DemodResult>{r});
    check::equal(frac[0][0], 1.0, "diversity/contrib1: nonzero weight -> full credit");
    check::equal(frac[0][1], 0.0, "diversity/contrib1: erased -> no credit");
}

void test_contribution_image_needs_two_branches() {
    const DemodResult r = make_result(std::vector<double>(mode_a().n_latents, 1.0),
                                      std::vector<double>(mode_a().n_latents, 1.0), 10.0);
    bool threw = false;
    try {
        modem::diversity::contribution_image(std::vector<DemodResult>{r});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "diversity/image: needs exactly two branches");
}

void test_contribution_image_shape_and_pure_branch_saturation() {
    const int n = mode_a().n_latents;
    const DemodResult good =
        make_result(std::vector<double>(n, 1.0), std::vector<double>(n, 1.0), 15.0);
    DemodResult dead =
        make_result(std::vector<double>(n, 0.0), std::vector<double>(n, 0.0), -1.0);
    dead.snr_db = -std::numeric_limits<double>::infinity();

    const images::Picture img =
        modem::diversity::contribution_image(std::vector<DemodResult>{good, dead}, 1);
    check::equal(img.width, mode_a().n_frames, "diversity/image: width is n_frames");
    check::equal(img.height, config::NC_LATENT, "diversity/image: height is NC_LATENT (carrier rows)");

    // `good` carries every latent alone: every touched cell should be
    // saturated red, with no blue at all.
    bool any_lit = false, any_blue = false;
    for (std::size_t px = 0; px + 2 < img.rgb.size(); px += 3) {
        if (img.rgb[px] > 0 || img.rgb[px + 2] > 0) {
            any_lit = true;
            if (img.rgb[px] < 250) check::fail("diversity/image", "a lit cell was not saturated red");
        }
        if (img.rgb[px + 2] > 0) any_blue = true;
    }
    check::is_true(any_lit, "diversity/image: at least one cell has data");
    check::is_true(!any_blue, "diversity/image: the dead branch contributes no blue");
}

void test_diversity_improves_snr_under_independent_awgn() {
    const modem::Modem m;
    const std::vector<double> lat = test_latents(mode_a().n_latents, 1);
    const std::vector<double> x = m.modulate(lat, mode_a());

    std::vector<double> a = x, b = x;
    // sigma picked so the clean-loopback SNR (dominated by clip/filter
    // distortion, not this noise) drops to a mid-single-digit-dB regime
    // where diversity combining has clear room to show a gain.
    add_awgn(a, 0.5, 11);
    add_awgn(b, 0.5, 22);

    const DemodResult ra = m.demodulate(a);
    const DemodResult rb = m.demodulate(b);
    const double sa = latent_snr_db(lat, ra.latents, ra.weights);
    const double sb = latent_snr_db(lat, rb.latents, rb.weights);

    const DemodResult combined = modem::diversity::combine_demod_results({ra, rb});
    const double s_combined = latent_snr_db(lat, combined.latents, combined.weights);

    check::is_true(s_combined > std::max(sa, sb) + 1.0,
                   "diversity/e2e: combining two independently-noisy branches "
                   "beats either alone by a real margin");
}

// --- blind-acquisition branches ---------------------------------------

void test_combine_blind_single_branch_is_identity() {
    const BlindDemodResult r = make_blind_result({1.0, -1.0, 0.5}, {1.0, 0.8, 0.0}, 10.0);
    const BlindDemodResult combined = modem::diversity::combine_blind_results({r});
    check::close(combined.latents, r.latents, 1e-12, "diversity/blind-single: latents unchanged");
    check::close(combined.weights, r.weights, 1e-12, "diversity/blind-single: weights unchanged");
}

void test_combine_blind_needs_at_least_one_branch() {
    bool threw = false;
    try {
        modem::diversity::combine_blind_results({});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "diversity/blind-empty: raises on no branches");
}

void test_combine_blind_weight_never_exceeds_one() {
    const BlindDemodResult a = make_blind_result({1.0, 1.0, 1.0}, {1.0, 0.9, 0.3}, 15.0);
    const BlindDemodResult b = make_blind_result({1.0, 1.0, 1.0}, {1.0, 0.9, 0.3}, 15.0);
    const BlindDemodResult combined = modem::diversity::combine_blind_results({a, b});
    for (double w : combined.weights)
        check::is_true(w <= 1.0 + 1e-9, "diversity/blind-cap: combined weight <= 1");
}

// --- combine_diversity_results: any mix of header/blind ----------------

void test_diversity_results_pure_headered_matches_combine_demod_results() {
    const DemodResult a = make_result({1.0, 1.0, 1.0}, {1.0, 0.9, 0.3}, 15.0);
    const DemodResult b = make_result({1.0, 1.0, 1.0}, {1.0, 0.9, 0.3}, 15.0);
    const modem::diversity::Branch combined =
        modem::diversity::combine_diversity_results({modem::diversity::Branch(a),
                                                      modem::diversity::Branch(b)});
    const DemodResult expected = modem::diversity::combine_demod_results({a, b});

    check::is_true(std::holds_alternative<DemodResult>(combined),
                   "diversity/mix-pure-header: returns a DemodResult");
    const DemodResult& got = std::get<DemodResult>(combined);
    check::close(got.latents, expected.latents, 1e-12,
                "diversity/mix-pure-header: latents match combine_demod_results");
    check::close(got.weights, expected.weights, 1e-12,
                "diversity/mix-pure-header: weights match combine_demod_results");
}

void test_diversity_results_pure_blind_matches_combine_blind_results() {
    const BlindDemodResult a = make_blind_result({1.0, 1.0}, {0.9, 0.5}, 10.0);
    const BlindDemodResult b = make_blind_result({1.0, 1.0}, {0.4, 0.5}, 10.0);
    const modem::diversity::Branch combined =
        modem::diversity::combine_diversity_results({modem::diversity::Branch(a),
                                                      modem::diversity::Branch(b)});
    const BlindDemodResult expected = modem::diversity::combine_blind_results({a, b});

    check::is_true(std::holds_alternative<BlindDemodResult>(combined),
                   "diversity/mix-pure-blind: returns a BlindDemodResult");
    const BlindDemodResult& got = std::get<BlindDemodResult>(combined);
    check::close(got.latents, expected.latents, 1e-12,
                "diversity/mix-pure-blind: latents match combine_blind_results");
    check::close(got.weights, expected.weights, 1e-12,
                "diversity/mix-pure-blind: weights match combine_blind_results");
}

void test_diversity_results_rejects_mismatched_header_modes() {
    const DemodResult a = make_result({0.0}, {1.0}, 10.0, mode_a());
    const DemodResult b =
        make_result(std::vector<double>(mode_b().n_latents, 0.0),
                   std::vector<double>(mode_b().n_latents, 1.0), 10.0, mode_b());
    bool threw = false;
    try {
        modem::diversity::combine_diversity_results(
            {modem::diversity::Branch(a), modem::diversity::Branch(b)});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "diversity/mix-mismatch: raises on differing header modes");
}

// One branch header-locked, the other only blind-locked (its
// preamble/header zeroed out, as if too faded to detect at all) -- real
// modem output, not hand-built, since the sizes have to agree exactly
// with what demodulate_blind actually produces (mode C's full range).
void test_diversity_results_mixed_header_and_blind_real() {
    const modem::Modem m;
    const std::vector<double> lat = test_latents(mode_a().n_latents, 40);
    const std::vector<double> x = m.modulate(lat, mode_a(), true, "MIXED");

    std::vector<double> a = x;
    add_awgn(a, 0.3, 401);
    std::vector<double> b = zero_preamble(x);
    add_awgn(b, 0.3, 402);

    const DemodResult headered = m.demodulate(a);
    const BlindDemodResult blind = m.demodulate_blind(b);
    check::is_true(blind.beacon.has_value(), "diversity/mix-real: the blind branch actually locked");

    const modem::diversity::Branch combined = modem::diversity::combine_diversity_results(
        {modem::diversity::Branch(headered), modem::diversity::Branch(blind)});
    check::is_true(std::holds_alternative<DemodResult>(combined),
                   "diversity/mix-real: a header lock present makes the result a DemodResult");
    const DemodResult& got = std::get<DemodResult>(combined);
    check::equal(got.latents.size(), static_cast<std::size_t>(mode_a().n_latents),
                "diversity/mix-real: truncated to the header-locked mode's range");
    check::is_true(got.mode.name == "A", "diversity/mix-real: mode A");
    for (double w : got.weights)
        check::is_true(w <= 1.0 + 1e-9, "diversity/mix-real: weight cap");

    const double s_combined = latent_snr_db(lat, got.latents, got.weights);
    const double s_headered = latent_snr_db(lat, headered.latents, headered.weights);
    check::is_true(s_combined > s_headered - 0.5,
                   "diversity/mix-real: combining with the blind branch doesn't hurt");
}

// --- contribution_image's Branch overload -------------------------------

void test_contribution_image_variant_overload_needs_two_branches() {
    const DemodResult r = make_result(std::vector<double>(mode_a().n_latents, 1.0),
                                      std::vector<double>(mode_a().n_latents, 1.0), 10.0);
    bool threw = false;
    try {
        modem::diversity::contribution_image(
            std::vector<modem::diversity::Branch>{modem::diversity::Branch(r)});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "diversity/image-variant: needs exactly two branches");
}

void test_contribution_image_variant_overload_matches_demod_overload() {
    const int n = mode_a().n_latents;
    const DemodResult a = make_result(std::vector<double>(n, 1.0), std::vector<double>(n, 1.0), 12.0);
    const DemodResult b = make_result(std::vector<double>(n, -1.0), std::vector<double>(n, 0.7), 12.0);

    const images::Picture direct =
        modem::diversity::contribution_image(std::vector<DemodResult>{a, b}, 1);
    const images::Picture via_variant = modem::diversity::contribution_image(
        std::vector<modem::diversity::Branch>{modem::diversity::Branch(a),
                                              modem::diversity::Branch(b)},
        1);
    check::equal(via_variant.width, direct.width, "diversity/image-variant: same width");
    check::equal(via_variant.height, direct.height, "diversity/image-variant: same height");
    check::equal(via_variant.height, config::NC_LATENT, "diversity/image-variant: carrier rows");
    check::is_true(via_variant.rgb == direct.rgb, "diversity/image-variant: pixel-identical");
}

}  // namespace

int main() {
    try {
        test_single_branch_is_identity();
        test_needs_at_least_one_branch();
        test_mode_mismatch_raises();
        test_combined_weight_never_exceeds_one();
        test_erasure_in_both_branches_stays_erased();
        test_stronger_branch_dominates();
        test_frames_received_is_the_max();
        test_branch_contribution_columns_sum_to_one_or_zero();
        test_branch_contribution_single_branch_is_erasure_mask();
        test_contribution_image_needs_two_branches();
        test_contribution_image_shape_and_pure_branch_saturation();
        test_diversity_improves_snr_under_independent_awgn();
        test_combine_blind_single_branch_is_identity();
        test_combine_blind_needs_at_least_one_branch();
        test_combine_blind_weight_never_exceeds_one();
        test_diversity_results_pure_headered_matches_combine_demod_results();
        test_diversity_results_pure_blind_matches_combine_blind_results();
        test_diversity_results_rejects_mismatched_header_modes();
        test_diversity_results_mixed_header_and_blind_real();
        test_contribution_image_variant_overload_needs_two_branches();
        test_contribution_image_variant_overload_matches_demod_overload();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("diversity");
}
