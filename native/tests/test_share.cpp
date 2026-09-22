// Sharing a template: the payload, what an imported document is
// allowed to reference, and the code that carries it.
//
// **Decodability is measured, not asserted here.** Reading a QR code
// back needs a decoder, which this repository deliberately does not
// vendor (native/third_party/qrcodegen/README.md), so what these checks
// can do is prove the image is the one `qrcodegen` produced, the right
// way up and the right way round -- which is where the bugs of this
// kind live. That it *scans* was measured with OpenCV's detector on the
// largest shipped template: at the specification's 4-module quiet zone
// it would not detect at all, at 6 and 8 it read all 447 bytes first
// time, which is why `QUIET_MODULES` is 8.

#include <QApplication>
#include <QImage>

#include <optional>
#include <string>
#include <variant>

#include "check.hpp"
#include "overlay/model.hpp"
#include "overlay/share.hpp"
#include "share_dialog.hpp"

using namespace sstvae;

namespace {

overlay::Doc a_template() {
    overlay::Doc doc;
    doc.name = "Shared";
    overlay::TextItem text;
    text.text = "{theircall} de {mycall}";
    text.font = "/home/someone/fonts/private.ttf";
    text.font_family = "monospace";
    text.color = "#ff8800";
    doc.items.push_back(text);
    overlay::ImageItem inset;
    inset.source = "/home/someone/Pictures/secret.png";
    doc.items.push_back(inset);
    overlay::ImageItem rx;  // defaults to last_rx
    doc.items.push_back(rx);
    return doc;
}

// The payload is the template's own file contents -- there is no share
// format to get wrong, which is the point. Compact only because a QR
// code is paid for by the byte.
void test_the_payload_is_the_document() {
    const overlay::Doc doc = a_template();
    const std::string compact = overlay::to_json(doc, -1);
    check::is_true(compact.find('\n') == std::string::npos, "share: the payload is one line");
    check::is_true(compact.size() < overlay::to_json(doc, 2).size(),
                   "share: and smaller than the file on disk");

    const overlay::Doc back = overlay::from_json(compact);
    check::equal(back.name, doc.name, "share: the name survives");
    check::equal(back.items.size(), doc.items.size(), "share: every item survives");
    check::equal(std::get<overlay::TextItem>(back.items[0]).color, std::string("#ff8800"),
                 "share: and what they look like");
}

// A document from somewhere else must not name files here. Both
// references are cleared; nothing else is touched.
void test_an_imported_document_names_no_local_files() {
    const overlay::Doc clean = overlay::sanitize_imported(a_template());
    check::equal(clean.items.size(), std::size_t{3}, "share: import drops no items");

    const auto& text = std::get<overlay::TextItem>(clean.items[0]);
    check::is_true(text.font.empty(), "share: a font path is dropped");
    check::equal(text.font_family, std::string("monospace"),
                 "share: the family request, which travels, is kept");
    check::equal(text.text, std::string("{theircall} de {mycall}"),
                 "share: and the text itself");

    check::equal(std::get<overlay::ImageItem>(clean.items[1]).source,
                 std::string(overlay::SOURCE_LAST_RX), "share: a file inset becomes last_rx");
    check::equal(std::get<overlay::ImageItem>(clean.items[2]).source,
                 std::string(overlay::SOURCE_LAST_RX), "share: and last_rx is left alone");
}

void test_junk_is_not_a_template() {
    check::is_true(!overlay::is_share_payload("hello"), "share: prose is not a template");
    check::is_true(!overlay::is_share_payload(""), "share: nor is nothing");
    check::is_true(overlay::is_share_payload(overlay::to_json(a_template(), -1)),
                   "share: a payload is");
}

// The three finder patterns: a 7x7 dark square with a light ring and a
// dark 3x3 core, at three corners and not the fourth. Cheap, and it is
// what catches the mistakes available here -- an inverted image, a
// transposed x/y, a quiet zone that ate the code.
bool finder_at(const QImage& image, int ox, int oy) {
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 7; ++x) {
            const bool edge = x == 0 || y == 0 || x == 6 || y == 6;
            const bool core = x >= 2 && x <= 4 && y >= 2 && y <= 4;
            const bool dark = qGray(image.pixel(ox + x, oy + y)) < 128;
            if (dark != (edge || core)) return false;
        }
    }
    return true;
}

void test_the_code_is_a_code() {
    const std::string payload = overlay::to_json(a_template(), -1);
    const std::optional<QImage> image = gui::qr_image(payload);
    check::is_true(image.has_value(), "share: a template encodes");
    if (!image) return;

    const int quiet = 8;
    const int size = image->width();
    check::equal(image->height(), size, "share: the code is square");
    check::is_true(size > 2 * quiet + 21, "share: and bigger than the smallest version");

    const int last = size - quiet - 7;
    check::is_true(finder_at(*image, quiet, quiet), "share: finder, top left");
    check::is_true(finder_at(*image, last, quiet), "share: finder, top right");
    check::is_true(finder_at(*image, quiet, last), "share: finder, bottom left");
    check::is_true(!finder_at(*image, last, last), "share: and none bottom right");

    // The quiet zone is light all the way round, which is the measured
    // difference between a code a detector finds and one it does not.
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < quiet; ++j) {
            if (qGray(image->pixel(i, j)) < 128 || qGray(image->pixel(i, size - 1 - j)) < 128 ||
                qGray(image->pixel(j, i)) < 128 || qGray(image->pixel(size - 1 - j, i)) < 128) {
                check::is_true(false, "share: the quiet zone is light");
                return;
            }
        }
    }
    check::is_true(true, "share: the quiet zone is light");
}

// Not defensive: a document with a dozen decorated items outgrows the
// largest version, and the dialog says so instead of drawing something
// no camera can read.
void test_a_document_too_big_for_a_code_says_so() {
    overlay::Doc doc;
    for (int i = 0; i < 40; ++i) {
        overlay::TextItem text;
        text.text = "a line of text that is long enough to add up quickly";
        text.color = "#123456";
        doc.items.push_back(text);
    }
    const std::string payload = overlay::to_json(doc, -1);
    check::is_true(payload.size() > 2953, "share: the test document really is too big");
    check::is_true(!gui::qr_image(payload).has_value(), "share: and no code is offered");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QApplication app(argc, argv);

    test_the_payload_is_the_document();
    test_an_imported_document_names_no_local_files();
    test_junk_is_not_a_template();
    test_the_code_is_a_code();
    test_a_document_too_big_for_a_code_says_so();

    return check::report("share");
}
