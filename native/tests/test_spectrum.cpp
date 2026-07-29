// The waterfall's numbers, without the waterfall.
//
// `reduce_to_width` is the reason this file exists. Its peak-hold
// behaviour is the difference between the carriers reading as a solid
// block and reading as a ragged comb -- and a ragged comb looks like a
// *reception* problem, so getting it wrong sends the next person to
// debug the modem. It is also pure, which makes it exactly the sort of
// thing that should not need a GUI to check.

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "dsp/spectrum.hpp"

using namespace sstvae;

namespace {

void test_a_tone_lands_in_the_right_bin() {
    // A pure tone at a bin centre, so there is a single right answer.
    const int bin = 128;
    const double hz = bin * dsp::WATERFALL_BIN_HZ;
    std::vector<double> block(dsp::WATERFALL_NFFT);
    for (int i = 0; i < dsp::WATERFALL_NFFT; ++i) {
        block[i] = std::sin(2.0 * std::numbers::pi * hz * i / config::FS);
    }

    const std::vector<double> db =
        dsp::spectrum_db(block, dsp::WATERFALL_BINS);
    check::equal(static_cast<int>(db.size()), dsp::WATERFALL_BINS,
                 "spectrum: one value per displayed bin");

    const auto peak = std::max_element(db.begin(), db.end());
    check::equal(static_cast<int>(peak - db.begin()), bin,
                 "spectrum: the tone peaks in its own bin");
    // And it stands well clear of the window's skirts.
    check::is_true(*peak > db[bin + 10] + 40.0,
                   "spectrum: the peak is far above the noise around it");
}

void test_silence_is_finite() {
    // log10(0) would be -inf, which propagates into the colour index as
    // a NaN and paints garbage rather than black.
    const std::vector<double> block(dsp::WATERFALL_NFFT, 0.0);
    const std::vector<double> db = dsp::spectrum_db(block, 16);
    for (const double value : db) {
        if (!std::isfinite(value)) {
            check::fail("spectrum: silence stays finite", "got a non-finite dB");
            return;
        }
    }
    check::is_true(true, "spectrum: silence stays finite");
}

void test_a_short_block_is_refused_rather_than_read_past() {
    const std::vector<double> block(dsp::WATERFALL_NFFT - 1, 1.0);
    check::is_true(dsp::spectrum_db(block, 16).empty(),
                   "spectrum: a short block yields nothing");
}

void test_narrow_peaks_survive_being_squeezed() {
    // The property that matters. A comb of one-bin spikes six bins
    // apart -- the shape of the actual carriers -- reduced to a third of
    // its width. Point-sampling would drop two out of three; peak-hold
    // keeps them all.
    std::vector<double> values(384, -90.0);
    for (std::size_t i = 3; i < values.size(); i += 6) values[i] = -30.0;

    const std::vector<double> reduced = dsp::reduce_to_width(values, 128);
    check::equal(static_cast<int>(reduced.size()), 128,
                 "reduce: exactly the requested width");

    int loud = 0;
    for (const double value : reduced) {
        if (value > -50.0) ++loud;
    }
    // 64 spikes into 128 columns: every one should still be there.
    check::equal(loud, 64, "reduce: every carrier survives the squeeze");
}

void test_reducing_never_invents_or_loses_range() {
    std::vector<double> values(300);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(static_cast<double>(i) * 0.37) * 20.0 - 50.0;
    }
    const double hi = *std::max_element(values.begin(), values.end());
    const double lo = *std::min_element(values.begin(), values.end());

    for (const int width : {1, 17, 128, 299, 300, 301, 640}) {
        const std::vector<double> reduced = dsp::reduce_to_width(values, width);
        check::equal(static_cast<int>(reduced.size()), width,
                     "reduce: width " + std::to_string(width));
        for (const double value : reduced) {
            if (value > hi + 1e-9 || value < lo - 1e-9) {
                check::fail("reduce: stays inside the source's range",
                            "width " + std::to_string(width) + " produced " +
                                std::to_string(value));
                return;
            }
        }
    }
    check::is_true(true, "reduce: stays inside the source's range");

    // Peak-hold when shrinking means the loudest bin is never lost --
    // which is the whole point, since that bin is a carrier.
    const std::vector<double> reduced = dsp::reduce_to_width(values, 50);
    check::is_true(*std::max_element(reduced.begin(), reduced.end()) == hi,
                   "reduce: the loudest bin is always kept");
}

void test_equal_width_is_the_identity() {
    const std::vector<double> values{1.0, 2.0, 3.0, 4.0};
    check::close(dsp::reduce_to_width(values, 4), values, 0.0,
                 "reduce: same width changes nothing");
}

void test_widening_interpolates_between_the_endpoints() {
    const std::vector<double> values{0.0, 10.0};
    const std::vector<double> wide = dsp::reduce_to_width(values, 5);
    check::close(wide, {0.0, 2.5, 5.0, 7.5, 10.0}, 1e-12,
                 "reduce: widening is a straight line, endpoints included");
}

void test_degenerate_widths_do_not_crash() {
    const std::vector<double> values{1.0, 2.0, 3.0};
    check::is_true(dsp::reduce_to_width(values, 0).empty(),
                   "reduce: zero width is empty");
    check::is_true(dsp::reduce_to_width(values, -4).empty(),
                   "reduce: negative width is empty");
    check::is_true(dsp::reduce_to_width({}, 10).empty(),
                   "reduce: no input is no output");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();

    test_a_tone_lands_in_the_right_bin();
    test_silence_is_finite();
    test_a_short_block_is_refused_rather_than_read_past();
    test_narrow_peaks_survive_being_squeezed();
    test_reducing_never_invents_or_loses_range();
    test_equal_width_is_the_identity();
    test_widening_interpolates_between_the_endpoints();
    test_degenerate_widths_do_not_crash();

    return check::report("waterfall spectrum");
}
