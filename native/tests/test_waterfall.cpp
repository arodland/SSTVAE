// The waterfall widget, in the parts that do not need eyes.
//
// Most of what makes a waterfall good is a judgement call and belongs
// in front of an operator. Three things are not: the scroll has to move
// history *down* rather than up (an in-place row copy is easy to get
// backwards, and the result still looks like a moving display), a
// resize has to keep the history rather than blank it, and a signal has
// to paint at the x its frequency says. Each of those has a right
// answer, so each is asserted here.

#include <QApplication>
#include <QImage>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "dsp/spectrum.hpp"
#include "rx/ringbuffer.hpp"
#include "waterfall.hpp"

using namespace sstvae;

namespace {

// Narrow on purpose. At this width the "SSTVAE 900-2150 Hz" caption
// does not fit and the widget drops it, which keeps the top rows clear
// for the scroll test below; the band markers (x=48, x=114), the 1 kHz
// grid ticks (bottom 8 rows) and the level meter (x>=150) are then the
// only overlays, and none of them touches the column we probe.
constexpr int W = 160;
constexpr int H = 200;
constexpr double TONE_HZ = 1500.0;

// The column a TONE_HZ tone lands in.
int tone_column() {
    return static_cast<int>(TONE_HZ / dsp::WATERFALL_DISPLAY_HZ * W);
}

// A ring holding a steady tone, so every tick paints the same row and
// the display's *motion* is the only variable.
std::shared_ptr<rx::RingBuffer> ring_with_tone(double hz) {
    // Seconds, not samples: RingBuffer sizes itself in time.
    auto ring = std::make_shared<rx::RingBuffer>(2.0);
    std::vector<double> block(dsp::WATERFALL_NFFT * 2);
    for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = 0.5 * std::sin(2.0 * std::numbers::pi * hz * static_cast<double>(i) /
                                  config::FS);
    }
    ring->write(block);
    return ring;
}

// Widgets paint on demand; render() drives it without a compositor.
QImage shot(gui::Waterfall& widget) {
    QImage image(widget.size(), QImage::Format_RGB888);
    image.fill(Qt::black);
    widget.render(&image);
    return image;
}

// One pixel, rather than a whole row. A row is never entirely black:
// the band markers are dashed vertical lines down the full height, so
// "is this row painted" has to be asked somewhere they are not.
bool is_dark(const QImage& image, int x, int y) {
    const QRgb pixel = image.pixel(x, y);
    return qRed(pixel) + qGreen(pixel) + qBlue(pixel) <= 24;
}

double brightness(const QImage& image, int x, int y) {
    const QRgb pixel = image.pixel(x, y);
    return qRed(pixel) + qGreen(pixel) + qBlue(pixel);
}

void test_it_draws_without_a_ring_buffer() {
    // Before the receive panel starts capture there is nothing to show,
    // and the widget still has to paint its grid rather than crash.
    gui::Waterfall widget;
    widget.resize(W, H);
    const QImage image = shot(widget);
    check::equal(image.width(), W, "waterfall/empty: paints at its own size");
    check::is_true(!image.isNull(), "waterfall/empty: paints something");
}

void test_a_tone_paints_where_its_frequency_says() {
    gui::Waterfall widget;
    widget.resize(W, H);
    widget.set_ring(ring_with_tone(TONE_HZ));
    // Enough ticks that the row we probe has been painted. Comparing
    // whole *columns* does not work: the band markers are dashed lines
    // down the full height, so a marker out-totals a tone that has only
    // painted one row -- which is a fact about the overlay, not about
    // where the signal went. One filled row answers the actual question.
    for (int i = 0; i < 10; ++i) {
        QMetaObject::invokeMethod(&widget, "tick", Qt::DirectConnection);
    }

    const QImage image = shot(widget);
    const int want = tone_column();
    constexpr int probe_row = 5;

    int brightest = 0;
    double best = -1.0;
    // Stop short of the level meter, a bright bar down the right edge
    // that would otherwise win every time.
    for (int x = 0; x < W - 20; ++x) {
        const double value = brightness(image, x, probe_row);
        if (value > best) {
            best = value;
            brightest = x;
        }
    }
    check::is_true(std::abs(brightest - want) <= 3,
                   "waterfall/tone: paints at the x its frequency says (got " +
                       std::to_string(brightest) + ", want " +
                       std::to_string(want) + ")");
}

void test_history_scrolls_downward() {
    gui::Waterfall widget;
    widget.resize(W, H);
    widget.set_ring(ring_with_tone(1500.0));

    // One tick: the newest row is at the top and everything below it is
    // still black.
    const int x = tone_column();
    QMetaObject::invokeMethod(&widget, "tick", Qt::DirectConnection);
    QImage image = shot(widget);
    check::is_true(!is_dark(image, x, 0),
                   "waterfall/scroll: the new row is at the top");
    check::is_true(is_dark(image, x, 5),
                   "waterfall/scroll: nothing below it yet");

    // Five more: the painted band has grown downward from the top,
    // which is what distinguishes a correct scroll from one that copies
    // rows the wrong way and leaves only the top row ever set.
    for (int i = 0; i < 5; ++i) {
        QMetaObject::invokeMethod(&widget, "tick", Qt::DirectConnection);
    }
    image = shot(widget);
    for (int y = 0; y < 6; ++y) {
        if (is_dark(image, x, y)) {
            check::fail("waterfall/scroll: history moves down the pane",
                        "row " + std::to_string(y) + " is still black");
            return;
        }
    }
    check::is_true(is_dark(image, x, 20),
                   "waterfall/scroll: and no further than it should");
}

void test_a_resize_keeps_the_history() {
    gui::Waterfall widget;
    widget.resize(W, H);
    widget.set_ring(ring_with_tone(TONE_HZ));
    for (int i = 0; i < 4; ++i) {
        QMetaObject::invokeMethod(&widget, "tick", Qt::DirectConnection);
    }

    // Wider and taller. Blanking here would be an obvious annoyance
    // every time the operator drags the splitter. The kept history is
    // point-resampled across the new width, so the tone lands in the
    // column the *new* width puts it in.
    widget.resize(W + 130, H + 60);
    const QImage image = shot(widget);
    const int moved =
        static_cast<int>(TONE_HZ / dsp::WATERFALL_DISPLAY_HZ * (W + 130));
    bool found = false;
    for (int dx = -3; dx <= 3 && !found; ++dx) {
        found = !is_dark(image, moved + dx, 0);
    }
    check::is_true(found, "waterfall/resize: the history survives a resize");
    check::equal(image.width(), W + 130,
                 "waterfall/resize: and the image follows the widget");
}

void test_clear_blanks_it() {
    gui::Waterfall widget;
    widget.resize(W, H);
    widget.set_ring(ring_with_tone(TONE_HZ));
    QMetaObject::invokeMethod(&widget, "tick", Qt::DirectConnection);
    const int x = tone_column();
    check::is_true(!is_dark(shot(widget), x, 0),
                   "waterfall/clear: something was drawn first");

    widget.clear();
    check::is_true(is_dark(shot(widget), x, 0),
                   "waterfall/clear: and clear() removes it");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QApplication app(argc, argv);

    test_it_draws_without_a_ring_buffer();
    test_a_tone_paints_where_its_frequency_says();
    test_history_scrolls_downward();
    test_a_resize_keeps_the_history();
    test_clear_blanks_it();

    return check::report("waterfall widget");
}
