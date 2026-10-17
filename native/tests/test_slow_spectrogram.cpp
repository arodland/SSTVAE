// The QRSSTVAE spectrogram's numbers, without the widget
// (core/dsp/slow_spectrogram.*): a tone lands at its frequency, one
// column a second stamped with the time its audio began, 15 s means for
// the history, a gap stays a gap, and the time axis's lens and history
// scales join without a jump.

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include "check.hpp"
#include "dsp/slow_spectrogram.hpp"

using namespace sstvae;
using dsp::SlowSpectrogram;

namespace {

constexpr double T0 = 1791594000.0;   // a quarter hour

// `seconds` of a tone at `hz` (amplitude `a`) plus a little noise,
// pushed in quarter-second chunks, the first sample at `start`.
void feed(SlowSpectrogram& s, double start, double seconds, double hz, double a,
          std::mt19937& rng) {
    std::normal_distribution<double> noise(0.0, 0.01);
    const int chunk = 2000;
    const int total = static_cast<int>(seconds * 8000);
    for (int i0 = 0; i0 < total; i0 += chunk) {
        std::vector<double> x(chunk);
        for (int i = 0; i < chunk; ++i) {
            const double t = (i0 + i) / 8000.0;
            x[i] = a * std::sin(2.0 * std::numbers::pi * hz * t) + noise(rng);
        }
        s.push(x, start + (i0 + chunk) / 8000.0);
    }
}

int peak_bin(const std::vector<float>& p) {
    return static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
}

void test_columns() {
    std::mt19937 rng(1);
    SlowSpectrogram s;
    feed(s, T0, 40.0, 1000.0, 0.5, rng);
    const auto& fine = s.fine();
    check::is_true(fine.size() >= 38 && fine.size() <= 40, "slow/fine: one column a second");
    check::is_true(std::abs(fine.front().t - T0) < 1e-6 && std::abs(fine[1].t - (T0 + 1)) < 1e-6,
                   "slow/fine: each stamped with the second its audio began");
    const int bin = peak_bin(fine[5].powers());
    check::is_true(std::abs(bin * SlowSpectrogram::BIN_HZ - 1000.0) < 1.0,
                   "slow/fine: a 1000 Hz tone peaks within a hertz of 1000");
    // 0.25 for amplitude 0.5, less the Hann window's scalloping off a
    // bin centre (at most 1.4 dB).
    check::is_true(fine[5].power(bin) > 0.17 && fine[5].power(bin) < 0.26,
                   "slow/fine: in amplitude squared");
    check::equal(s.coarse().size(), std::size_t{2},
                 "slow/coarse: the two whole 15 s blocks (the third is still filling)");
    check::is_true(s.coarse().front().t == T0, "slow/coarse: stamped with its block");

    double n = 0.0;
    const std::vector<float> m = s.mean_between(T0 + 10, T0 + 13, &n);
    check::is_true(std::abs(n - 3) < 1e-9, "slow/mean: the columns that overlap the range");
    double expect = 0.0;
    for (int k = 10; k <= 12; ++k) expect += fine[static_cast<std::size_t>(k)].power(bin);
    check::is_true(std::abs(m[static_cast<std::size_t>(bin)] - expect / 3) < 1e-6,
                   "slow/mean: their mean");
    check::is_true(s.mean_between(T0 - 100, T0 - 50).empty(), "slow/mean: nothing before");

    // A range that cuts columns in half takes half of each: summed in
    // power, weighted by what lies inside, divided by the total weight.
    const std::vector<float> cut = s.mean_between(T0 + 10.5, T0 + 12.5, &n);
    // A noise bin, where the columns differ enough to tell the weights.
    const int nb = bin + 40;
    const double w = 0.5 * fine[10].power(nb) + fine[11].power(nb) + 0.5 * fine[12].power(nb);
    check::is_true(std::abs(cut[static_cast<std::size_t>(nb)] - w / 2.0) < 1e-6 * w,
                   "slow/mean: a column half inside counts half");
    check::is_true(std::abs(n - 4.0 / 1.5) < 1e-9,
                   "slow/mean: and is worth its share of the noise");
    s.mean_between(T0 + 10.2, T0 + 10.7, &n);
    check::is_true(std::abs(n - 1.0) < 1e-9,
                   "slow/mean: inside one column, it is worth that whole column");
}

void test_a_gap_stays_a_gap() {
    std::mt19937 rng(2);
    SlowSpectrogram s;
    feed(s, T0, 10.0, 800.0, 0.5, rng);
    feed(s, T0 + 30.0, 10.0, 800.0, 0.5, rng);
    bool spans = false;
    for (const auto& c : s.fine()) {
        if (c.t > T0 + 9.0 && c.t < T0 + 30.0) spans = true;
    }
    check::is_true(!spans, "slow/gap: no column made of audio either side of a gap");
    check::is_true(s.mean_between(T0 + 15, T0 + 25).empty(), "slow/gap: and nothing drawn in it");
    check::is_true(std::abs(s.fine().back().t - (T0 + 38.0)) < 1e-6,
                   "slow/gap: the audio after it keeps its own times");
}

void test_history_comes_from_the_coarse_store() {
    SlowSpectrogram s;
    std::mt19937 rng(3);
    feed(s, T0, 60.0, 1200.0, 0.3, rng);
    const std::size_t fine_n = s.fine().size();
    check::is_true(s.coarse().size() == 3, "slow/coarse: three whole blocks in a minute");
    double n = 0.0;
    s.mean_between(T0 + 20, T0 + 30, &n);
    check::is_true(std::abs(n - 10) < 1e-9, "slow/mean: a short range from the fine store while it reaches back");
    const std::vector<float> wide = s.mean_between(T0, T0 + 30, &n);
    check::is_true(std::abs(n - 30) < 1e-9, "slow/mean: a block or longer from the coarse one");
    std::vector<float> by_hand(wide.size(), 0.0f);
    for (const auto& c : s.coarse()) {
        if (c.t < T0 + 30) {
            for (std::size_t k = 0; k < by_hand.size(); ++k) by_hand[k] += c.power[k] / 2;
        }
    }
    const int bin = peak_bin(wide);
    check::is_true(fine_n > 30 && std::abs(wide[static_cast<std::size_t>(bin)] -
                                           by_hand[static_cast<std::size_t>(bin)]) < 1e-6,
                   "slow/mean: the coarse blocks' own mean");
}

void test_the_fine_store_is_a_byte_a_bin() {
    for (const float p : {1e-12f, 3.7e-8f, 0.25f, 1.0f}) {
        const float back = SlowSpectrogram::dequantize(SlowSpectrogram::quantize(p));
        check::is_true(std::abs(10.0 * std::log10(back / p)) <= SlowSpectrogram::DB_STEP / 2 + 1e-6,
                       "slow/bytes: a level comes back within half a step");
    }
    check::is_true(SlowSpectrogram::quantize(0.0f) == 0, "slow/bytes: silence is the bottom level");
    check::is_true(SlowSpectrogram::FINE_KEEP_S >= 6 * 3600.0,
                   "slow/bytes: so the fine store keeps the whole six hours");
}

void test_older_than_the_fine_store() {
    std::mt19937 rng(5);
    // A short fine store, so the test need not make six hours of audio.
    constexpr double KEEP = 600.0;
    SlowSpectrogram s(KEEP);
    // Past its length, so the first minute is only coarse.
    feed(s, T0, KEEP + 120.0, 1200.0, 0.3, rng);
    check::is_true(s.fine().front().t > T0 + 60.0, "slow/old: the fine store has moved on");
    double n = 0.0;
    const std::vector<float> m = s.mean_between(T0, T0 + 5.0, &n);
    check::is_true(std::abs(n - 15) < 1e-9, "slow/old: even a short range, from the coarse block (15 s of audio)");
    const int bin = peak_bin(m);
    check::is_true(!m.empty() && std::abs(bin * SlowSpectrogram::BIN_HZ - 1200.0) < 1.0 &&
                       m[static_cast<std::size_t>(bin)] > 0.05 &&
                       m[static_cast<std::size_t>(bin)] < 0.1,
                   "slow/old: and their mean, at the tone's power");
}

void test_the_time_scale() {
    const dsp::TimeScale scale(1.0, 15.0, 360);
    check::is_true(scale.age_at(0) == 0.0 && std::abs(scale.age_at(1) - 1.0) < 0.01,
                   "scale: the newest second a pixel");
    check::is_true(scale.spp_at(0) == 1.0 && scale.spp_at(360) == 15.0 &&
                       scale.spp_at(1000) == 15.0,
                   "scale: fifteen for the history");
    bool monotonic = true;
    bool inverse = true;
    bool smooth = true;
    double prev = -1;
    double prev_step = 0;
    for (int x = 0; x <= 1200; ++x) {
        const double a = scale.age_at(x);
        if (a <= prev) monotonic = false;
        if (x > 1) {
            const double step = a - prev;
            // The scale never changes by more than a few percent from
            // one pixel to the next: no edge.
            if (step / prev_step > 1.05 || step / prev_step < 0.999) smooth = false;
            prev_step = step;
        } else if (x == 1) {
            prev_step = a - prev;
        }
        prev = a;
        if (std::abs(scale.x_of(a) - x) > 1e-6) inverse = false;
    }
    check::is_true(monotonic, "scale: older to the right, everywhere");
    check::is_true(smooth, "scale: the lens eases into the history with no edge");
    check::is_true(inverse, "scale: x_of undoes age_at");
    check::is_true(std::abs((scale.age_at(801) - scale.age_at(800)) - 15.0) < 1e-9,
                   "scale: then linear");
    dsp::TimeScale z;
    z.set_history_spp(1000);
    check::is_true(z.history_spp() == dsp::TimeScale::MAX_SPP, "scale: clamped");
}

}  // namespace

int main() {
    test_columns();
    test_a_gap_stays_a_gap();
    test_history_comes_from_the_coarse_store();
    test_older_than_the_fine_store();
    test_the_fine_store_is_a_byte_a_bin();
    test_the_time_scale();
    return check::report("slow spectrogram");
}
