// The CW ID tone generator: timing, table correctness, and the
// click-free envelope. No receive side exists for this (a human reads
// it by ear), so this checks against the ITU/PARIS timing standard by
// total sample count -- not by scanning the waveform for on/off runs,
// because a 1000 Hz tone at an 8000 Hz sample rate is exactly zero
// every fourth sample (it sits on a carrier-spacing multiple), so a
// naive nonzero/zero scan sees dozens of spurious "gaps" inside a
// single steady tone burst.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "check.hpp"
#include "dsp/morse.hpp"

using namespace sstvae;

namespace {

constexpr int SR = 8000;
constexpr double WPM = 18.0;
constexpr double HZ = 1000.0;

std::size_t samples(double seconds) {
    return static_cast<std::size_t>(std::lround(seconds * SR));
}

double dot_s() { return 1.2 / WPM; }

void test_e_is_a_single_dot() {
    const std::vector<double> x = dsp::generate_morse("E", SR, WPM, HZ, 1.0);
    check::equal(x.size(), samples(dot_s()), "morse/e: exactly one dot");
}

void test_t_is_a_single_dash() {
    const std::vector<double> x = dsp::generate_morse("T", SR, WPM, HZ, 1.0);
    check::equal(x.size(), samples(3.0 * dot_s()), "morse/t: exactly one dash");
}

void test_letter_gap_is_three_dots() {
    // "T" (dash) + letter gap (3 dots) + "E" (dot).
    const std::vector<double> x = dsp::generate_morse("TE", SR, WPM, HZ, 1.0);
    const std::size_t want =
        samples(3.0 * dot_s()) + samples(3.0 * dot_s()) + samples(dot_s());
    check::equal(x.size(), want, "morse/gap: dash + 3-dot gap + dot");
}

void test_word_gap_is_seven_dots() {
    const std::vector<double> x = dsp::generate_morse("E E", SR, WPM, HZ, 1.0);
    const std::size_t want = samples(dot_s()) + samples(7.0 * dot_s()) + samples(dot_s());
    check::equal(x.size(), want, "morse/word: dot + 7-dot gap + dot");
}

void test_intra_character_gap_is_one_dot() {
    // "N" is dash-dot: dash + 1-dot gap + dot, no letter gap involved.
    const std::vector<double> x = dsp::generate_morse("N", SR, WPM, HZ, 1.0);
    const std::size_t want = samples(3.0 * dot_s()) + samples(dot_s()) + samples(dot_s());
    check::equal(x.size(), want, "morse/intra: dash + 1-dot gap + dot");
}

void test_case_insensitive() {
    const std::vector<double> lower = dsp::generate_morse("kc2g/p", SR, WPM, HZ, 1.0);
    const std::vector<double> upper = dsp::generate_morse("KC2G/P", SR, WPM, HZ, 1.0);
    check::is_true(lower == upper, "morse/case: lowercase matches uppercase exactly");
    check::is_true(!lower.empty(), "morse/case: a real callsign produces audio");
}

void test_unrecognized_characters_are_dropped_not_rejected() {
    // "K*C" should sound exactly like "KC" -- the '*' contributes
    // nothing, not a burst of silence or a thrown exception.
    const std::vector<double> with_junk = dsp::generate_morse("K*C", SR, WPM, HZ, 1.0);
    const std::vector<double> clean = dsp::generate_morse("KC", SR, WPM, HZ, 1.0);
    check::is_true(with_junk == clean, "morse/junk: dropped, not gapped or thrown");
}

void test_nothing_recognized_is_empty() {
    check::is_true(dsp::generate_morse("***", SR, WPM, HZ, 1.0).empty(),
                   "morse/empty: no recognizable character produces nothing");
    check::is_true(dsp::generate_morse("", SR, WPM, HZ, 1.0).empty(),
                   "morse/empty: empty text produces nothing");
}

void test_amplitude_is_respected() {
    const std::vector<double> x = dsp::generate_morse("E", SR, WPM, HZ, 0.5);
    double peak = 0.0;
    for (double v : x) peak = std::max(peak, std::abs(v));
    // The raised-cosine ramp only ever attenuates, so the peak must not
    // exceed the requested amplitude, and with a 5 ms ramp against a
    // ~67 ms dot it should still get close to it.
    check::is_true(peak <= 0.5 + 1e-9, "morse/amp: never exceeds the requested amplitude");
    check::is_true(peak > 0.4, "morse/amp: the ramp still reaches near full amplitude");
}

void test_edges_are_ramped_not_clicked() {
    // A clicked (unramped) edge would jump straight from 0 to full
    // amplitude in one sample; a raised-cosine edge starts and ends at
    // exactly 0.
    const std::vector<double> x = dsp::generate_morse("T", SR, WPM, HZ, 1.0);
    check::is_true(!x.empty(), "morse/ramp: T produces audio");
    check::close(std::vector<double>{x.front()}, std::vector<double>{0.0}, 1e-12,
                "morse/ramp: the burst starts at zero, not a click");
    check::close(std::vector<double>{x.back()}, std::vector<double>{0.0}, 1e-12,
                "morse/ramp: the burst ends at zero, not a click");
}

void test_contains_real_energy() {
    // Sanity check that the body of the burst is not just the ramped
    // edges: a dash (200 ms) has plenty of flat-envelope tone in it.
    const std::vector<double> x = dsp::generate_morse("T", SR, WPM, HZ, 1.0);
    double energy = 0.0;
    for (double v : x) energy += v * v;
    const double rms = std::sqrt(energy / static_cast<double>(x.size()));
    // A full-amplitude sine has RMS 1/sqrt(2) =~ 0.707; with the ramp
    // eating a small fraction of a 200 ms dash this should stay close.
    check::is_true(rms > 0.6, "morse/energy: the dash is mostly at full amplitude");
}

}  // namespace

int main() {
    try {
        test_e_is_a_single_dot();
        test_t_is_a_single_dash();
        test_letter_gap_is_three_dots();
        test_word_gap_is_seven_dots();
        test_intra_character_gap_is_one_dot();
        test_case_insensitive();
        test_unrecognized_characters_are_dropped_not_rejected();
        test_nothing_recognized_is_empty();
        test_amplitude_is_respected();
        test_edges_are_ramped_not_clicked();
        test_contains_real_energy();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("morse");
}
