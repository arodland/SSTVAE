// `images::fit` with a Framing, and the contract that matters most:
// adding framing must not change what the default call produces.
//
// The trap is one pixel wide. The old code cropped at
// `(scaled - target) / 2` -- integer division, i.e. truncation -- and
// the natural rewrite `lround(center * scaled - target / 2.0)` agrees
// for an even `scaled` and disagrees by one for an odd one. Every
// odd-intermediate picture would then shift a pixel, silently, for no
// reason the operator asked for.
//
// **The oracle has to be the old formula, reimplemented here.**
// Comparing `fit(img)` against `fit(img, Framing{})` proves nothing:
// the first delegates to the second, so it compares the function to
// itself and passes however wrong both are. (Verified -- that first
// draft passed with the `lround` mutation in place.) Nor can Python be
// the oracle: `tests/test_native_parity.py` deliberately does not
// compare `fit_image` on anything that needs resampling, because PIL's
// LANCZOS and stb's filter differ by design. So the reference below
// resamples through the *same* `images::resize`, which isolates the
// one thing under test -- the crop offset.

#include "images/images.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "images/types.hpp"

namespace check = sstvae::check;
using sstvae::images::Framing;
using sstvae::images::Picture;

namespace {

// A deterministic, non-uniform picture: every pixel encodes its own
// position, so a crop that lands one pixel out is visible as a
// mismatch rather than hidden by flat colour.
Picture ramp(int width, int height) {
    Picture p(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 3;
            p.rgb[i] = static_cast<std::uint8_t>(x % 256);
            p.rgb[i + 1] = static_cast<std::uint8_t>(y % 256);
            p.rgb[i + 2] = static_cast<std::uint8_t>((x + y) % 256);
        }
    }
    return p;
}

// The algorithm as it stood before framing existed, written out.
// Independent of the code under test except for `resize`, which is
// shared on purpose so the resampler cancels and only the crop offset
// is being compared.
Picture reference_fit(const Picture& img) {
    const int W = sstvae::images::IMG_W;
    const int H = sstvae::images::IMG_H;
    if (img.width == W && img.height == H) return img;
    const double scale = std::max(static_cast<double>(W) / img.width,
                                  static_cast<double>(H) / img.height);
    const int sw = std::max(W, static_cast<int>(std::lround(img.width * scale)));
    const int sh = std::max(H, static_cast<int>(std::lround(img.height * scale)));
    const Picture scaled = sstvae::images::resize(img, sw, sh);
    Picture out(W, H);
    const int left = (sw - W) / 2;  // the original integer division
    const int top = (sh - H) / 2;
    for (int y = 0; y < H; ++y) {
        const std::uint8_t* src =
            scaled.rgb.data() + (static_cast<std::size_t>(y + top) * sw + left) * 3;
        std::copy(src, src + static_cast<std::size_t>(W) * 3,
                  out.rgb.begin() + static_cast<std::size_t>(y) * W * 3);
    }
    return out;
}

void test_default_framing_matches_the_old_algorithm() {
    // Sizes chosen so both parities of the scaled intermediate appear --
    // that is where truncation and rounding disagree. 1920x1080 scales
    // to 853 wide (odd); 800x600 to 640 (even).
    const std::vector<std::pair<int, int>> sizes = {
        {640, 480},  {1920, 1080}, {800, 600},  {481, 640},
        {1001, 700}, {333, 250},   {1279, 721}, {320, 240},
    };
    int odd_seen = 0;
    for (const auto& [w, h] : sizes) {
        const Picture src = ramp(w, h);
        const Picture want = reference_fit(src);
        const std::string label = std::to_string(w) + "x" + std::to_string(h);

        // Record that the sample actually exercises the odd case; a
        // suite that only ever saw even intermediates would pass with
        // the bug present.
        const double scale =
            std::max(static_cast<double>(sstvae::images::IMG_W) / w,
                     static_cast<double>(sstvae::images::IMG_H) / h);
        const int sw = std::max(sstvae::images::IMG_W,
                                static_cast<int>(std::lround(w * scale)));
        if (sw % 2 == 1) ++odd_seen;

        for (const Picture& got :
             {sstvae::images::fit(src), sstvae::images::fit(src, Framing{})}) {
            check::equal(got.width, want.width, "framing/default: width " + label);
            check::equal(got.height, want.height, "framing/default: height " + label);
            check::is_true(got.rgb == want.rgb,
                           "framing/default: byte-identical to the old "
                           "algorithm for " +
                               label);
        }
    }
    check::is_true(odd_seen > 0,
                   "framing/default: the sample includes an odd scaled width, "
                   "which is the case that can regress");
}

// Which source pixel ended up at the output's top-left corner.
//
// This is the oracle the framing tests needed. `ramp` encodes a pixel's
// own x in red and y in green, so on a geometry where the cover scale
// is exactly 1 -- and `resize` then returns the source untouched -- the
// output's corner pixel *names the crop offset*. Inequality assertions
// cannot do this: two real mutants (a mirrored pan axis, and zoom
// applied to width only) both passed a suite built out of `a != b`.
struct Corner {
    int x;
    int y;
};
Corner corner_source_of(const Picture& out) {
    return {out.rgb[0], out.rgb[1]};
}

void test_panning_selects_the_named_columns() {
    // 1280x480: the cover scale is max(640/1280, 480/480) = 1.0
    // exactly, so no resampling happens and the crop is a pure
    // sub-rectangle whose offset the corner pixel reports.
    const Picture src = ramp(1280, 480);

    struct Case {
        double center_x;
        int want_left;
        const char* what;
    };
    // left = clamp(floor(center*1280 - 320), 0, 640)
    const Case cases[] = {
        {0.0, 0, "hard left"},
        {0.25, 0, "quarter, clamped to the left edge"},
        {0.5, 320, "centre"},
        {0.75, 640, "three quarters, clamped to the right edge"},
        {1.0, 640, "hard right"},
    };
    for (const Case& c : cases) {
        Framing f;
        f.center_x = c.center_x;
        const Picture out = sstvae::images::fit(src, f);
        const Corner got = corner_source_of(out);
        check::equal(got.x, c.want_left % 256,
                     std::string("framing/pan: ") + c.what +
                         " starts at source column " +
                         std::to_string(c.want_left));
        check::equal(got.y, 0, std::string("framing/pan: ") + c.what +
                                   " starts at source row 0");
    }

    // Directional, stated separately so a mirrored axis is unmistakable
    // rather than merely "different".
    Framing left;
    left.center_x = 0.0;
    Framing right;
    right.center_x = 1.0;
    check::is_true(corner_source_of(sstvae::images::fit(src, left)).x <
                       corner_source_of(sstvae::images::fit(src, right)).x,
                   "framing/pan: a smaller centre selects a smaller column");
}

void test_zoom_crops_both_axes_equally() {
    // 1280x960 is 4:3, so the cover scale is 0.5 and zoom 2 makes it
    // exactly 1.0 -- again no resampling. The window is then the middle
    // 640x480 of the source, and both offsets are checkable.
    const Picture src = ramp(1280, 960);
    Framing zoomed;
    zoomed.zoom = 2.0;
    const Picture out = sstvae::images::fit(src, zoomed);

    const Corner got = corner_source_of(out);
    check::equal(got.x, 320 % 256, "framing/zoom: left offset is 320");
    check::equal(got.y, 240 % 256, "framing/zoom: top offset is 240");

    // The whole sub-rectangle, not only its corner: this is what fails
    // if zoom is applied to one axis and not the other, since the
    // squashed intermediate changes every row.
    Picture want(sstvae::images::IMG_W, sstvae::images::IMG_H);
    for (int y = 0; y < sstvae::images::IMG_H; ++y) {
        const std::uint8_t* row =
            src.rgb.data() +
            ((static_cast<std::size_t>(y) + 240) * 1280 + 320) * 3;
        std::copy(row, row + static_cast<std::size_t>(sstvae::images::IMG_W) * 3,
                  want.rgb.begin() +
                      static_cast<std::size_t>(y) * sstvae::images::IMG_W * 3);
    }
    check::is_true(out.rgb == want.rgb,
                   "framing/zoom: the output is exactly the centre 640x480 "
                   "of a 2x-zoomed 1280x960 source");

    // Below 1 would expose edges with nothing behind them.
    Framing under;
    under.zoom = 0.25;
    check::is_true(
        sstvae::images::fit(src, under).rgb ==
            sstvae::images::fit(src, Framing{}).rgb,
        "framing/zoom: below 1 is clamped to cover");
}

void test_out_of_range_centres_are_clamped() {
    const Picture src = ramp(1280, 480);
    Framing far;
    far.center_x = 5.0;
    far.center_y = -3.0;
    const Picture out = sstvae::images::fit(src, far);
    check::equal(out.width, sstvae::images::IMG_W, "framing/clamp: right size");
    // Named offsets rather than a comparison against another call of
    // the same function: the rightmost legal window starts at column
    // 640, and the vertical has no room at all so it starts at row 0.
    const Corner got = corner_source_of(out);
    check::equal(got.x, 640 % 256,
                 "framing/clamp: a far-right centre lands on column 640");
    check::equal(got.y, 0, "framing/clamp: a negative centre lands on row 0");
}

void test_identity_shortcut_respects_framing() {
    // A 640x480 source with the default framing is returned untouched
    // (the parity path). With a zoom it must NOT be -- an operator who
    // zoomed an already-4:3 picture meant it.
    const Picture src = ramp(sstvae::images::IMG_W, sstvae::images::IMG_H);
    check::is_true(sstvae::images::fit(src, Framing{}).rgb == src.rgb,
                   "framing/identity: default framing returns the original");
    Framing zoomed;
    zoomed.zoom = 1.5;
    check::is_true(sstvae::images::fit(src, zoomed).rgb != src.rgb,
                   "framing/identity: a zoom is honoured on a 4:3 source");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(120.0, "framing");

    check::current_step.store("default_matches_bare");
    test_default_framing_matches_the_old_algorithm();
    check::current_step.store("pan");
    test_panning_selects_the_named_columns();
    check::current_step.store("zoom");
    test_zoom_crops_both_axes_equally();
    check::current_step.store("clamp");
    test_out_of_range_centres_are_clamped();
    check::current_step.store("identity");
    test_identity_shortcut_respects_framing();

    return check::report("framing");
}
