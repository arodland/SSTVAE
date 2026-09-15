// The transmit pane's control strip, which must not change height.
//
// **This is the guard for a bug that broke the other pane.** The
// "Selected item" row used to hide itself when nothing was selected,
// while the comment where it is built said the opposite in bold. Two
// things followed, and the second is why this file exists:
//
//   1. The row is ~90 px, so the canvas jumped under the pointer on
//      every select and deselect -- on a composing surface, where the
//      thing being clicked is the thing that moves.
//
//   2. `PaneContainer::equalise_strips` -- the entire mechanism that
//      makes the received image and the composed image the same size --
//      runs only from `set_control_strips` and from a resize. Nothing
//      re-runs it when a strip's *content* changes height. So from the
//      first click on the canvas, both strips held minimums computed
//      for a layout that no longer existed and the two images silently
//      stopped matching, until something happened to resize the window.
//
// `test_pane_container.cpp` cannot catch this: it drives the container
// with stand-in panes whose strips are static, which is right for what
// it tests and blind to what this tests. The check has to be on the
// real panel.
//
// Deliberately a height *invariant* rather than a "the box is visible"
// assertion: what the layout cares about is the number, and hiding the
// row by some other means later would break the panes the same way.

#include <QApplication>
#include <QByteArray>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFrame>
#include <QIODevice>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPoint>
#include <QPushButton>
#include <QSize>
#include <QSlider>
#include <QTemporaryDir>
#include <QTimer>
#include <QWidget>

#include <string>
#include <variant>

#include "app_state.hpp"
#include "check.hpp"
#include "flow_layout.hpp"
#include "overlay/model.hpp"
#include "overlay_editor.hpp"
#include "tx_panel.hpp"

using namespace sstvae;
using namespace sstvae::gui;

namespace {

// The strip's height at the width it will actually have.
//
// Not `sizeHint()`: the rows inside wrap, so the height is a function
// of the width, and a hint taken at an unconstrained width reports one
// line -- the same trap `PaneContainer::strip_height_for_width` records.
int strip_height(QWidget* strip) {
    QLayout* layout = strip->layout();
    if (layout != nullptr && layout->hasHeightForWidth() && strip->width() > 0) {
        return layout->minimumHeightForWidth(strip->width());
    }
    return strip->sizeHint().height();
}

void test_the_strip_height_survives_a_selection() {
    // `AppState`'s constructor loads settings and opens the log; it
    // does not fetch a model, and `TransmitPanel` with no codec simply
    // does not arm the optimizer. So this needs no network and no
    // radio.
    AppState state;
    QWidget host;
    host.resize(900, 700);
    auto* panel = new TransmitPanel(&state, &host);
    panel->setGeometry(0, 0, 900, 700);
    host.show();
    QCoreApplication::processEvents();

    QWidget* strip = panel->control_strip();
    const int idle = strip_height(strip);
    check::is_true(idle > 0, "the strip has a height to begin with");

    // Adding a text item selects it, which is the transition that used
    // to reveal the properties row.
    auto* editor = panel->findChild<OverlayEditor*>();
    check::is_true(editor != nullptr, "the panel has an overlay editor");
    editor->add_text(std::string("KD8XYZ"));
    QCoreApplication::processEvents();
    check::equal(strip_height(strip), idle,
                 "the strip is the same height with an item selected");

    // And back to nothing selected, which is the transition that used
    // to hide the row again. Through `remove_selected`, which is a real
    // gesture (the Remove button, or Delete) and reaches the same state
    // as clicking empty canvas -- `select` itself is private, and this
    // test has no business reaching past the panel's own surface.
    editor->remove_selected();
    QCoreApplication::processEvents();
    check::equal(strip_height(strip), idle,
                 "the strip is the same height with nothing selected");
}

// The level caption, slider and readout are one item of the send bar.
//
// The send bar is a `FlowLayout`, which wraps *between* items, so three
// separate items could put the slider on one line and the number it is
// showing on the next -- or strand "Level:" above the thing it names.
// None of the three means anything alone: the slider has no scale
// printed on it, so the readout is the only way to know where it is
// set.
//
// **Measured before writing this: no reachable width actually splits
// them today.** Swept 380 to 900 px in 2 px steps against the
// un-grouped layout and the readout never left the slider's line -- the
// worst difference was 1 px of integer centring. The trio is ~285 px at
// its narrowest and the pane's floor is 380, so it fits on the first
// line every time, and the wrap lands after Send and Cancel instead.
//
// So this asserts the *structure*, not a break it cannot reproduce. A
// same-line check would pass with or without the fix, which by this
// project's standards is worse than no check at all. What is worth
// holding is that the three stay one item: the property currently
// rests on a coincidence between the mode combo's width, the slider's
// 80 px minimum and the pane floor, and a longer translation of
// "Level:" is all it would take to break it.
void test_the_level_controls_are_one_flow_item() {
    AppState state;
    QWidget host;
    host.resize(1400, 700);
    auto* panel = new TransmitPanel(&state, &host);
    host.show();

    auto* slider = panel->findChild<QSlider*>();
    check::is_true(slider != nullptr, "the panel has a level slider");

    // By its text rather than a stored pointer: this test has no
    // business reaching into the panel's members, and "the label
    // showing decibels" is what the operator is looking at.
    QLabel* readout = nullptr;
    for (QLabel* label : panel->findChildren<QLabel*>()) {
        if (label->text().endsWith(QLatin1String(" dB"))) {
            check::is_true(readout == nullptr, "exactly one dB readout");
            readout = label;
        }
    }
    check::is_true(readout != nullptr, "the panel has a dB readout");
    if (slider == nullptr || readout == nullptr) return;

    QWidget* group = slider->parentWidget();
    check::is_true(group == readout->parentWidget(),
                   "the readout shares a container with the slider");
    // And that container is not the wrapping row itself, which is what
    // it would be if the three were added to the send bar directly.
    // dynamic_cast, not qobject_cast: FlowLayout carries no Q_OBJECT
    // (it has no signals of its own), and qobject_cast static_asserts
    // on that rather than falling back.
    check::is_true(group != nullptr &&
                       dynamic_cast<FlowLayout*>(group->layout()) == nullptr,
                   "their container does not wrap between them");
}

// Editing defers the composite rebuild instead of doing it inline.
//
// `schedule_optimization` renders the whole 640x480 composite and
// converts it to float **on the GUI thread** -- measured at 6.37 ms for
// a three-item composition -- and `OverlayEditor::documentChanged` is
// emitted on every mouse move of a drag. The debounce inside
// `Speculative` does not help with that: it keeps the *worker* from
// starting a run per edit, and by the time it is consulted the
// expensive part has already happened.
//
// Asserted on the timer rather than on elapsed time: what is being
// checked is that the work was deferred and coalesced, which is a
// structural property. A stopwatch here would be a latency test
// pretending to be a correctness one.
void test_an_edit_defers_the_rebuild() {
    AppState state;
    QWidget host;
    host.resize(900, 700);
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* debounce = panel->findChild<QTimer*>(QStringLiteral("edit_debounce"));
    check::is_true(debounce != nullptr, "the panel has an edit debounce");
    if (debounce == nullptr) return;
    check::is_true(debounce->isSingleShot(),
                   "the debounce is single-shot, so a burst is one rebuild");
    check::is_true(debounce->interval() > 0,
                   "and has an interval, so it actually defers");

    auto* editor = panel->findChild<OverlayEditor*>();
    check::is_true(editor != nullptr, "the panel has an overlay editor");
    if (editor == nullptr) return;

    // A burst, as a drag produces. Every one of these restarts the same
    // timer; none of them may rebuild.
    for (int i = 0; i < 20; ++i) editor->refresh_item();
    check::is_true(debounce->isActive(),
                   "a burst of edits leaves one deferred rebuild, not none");
}

// A rebuild consumes the pending edit rather than leaving it queued.
//
// This is the primitive `send()` uses to flush, and the reason it must
// exist. The generation counter in `Speculative` is what guarantees the
// latents on the air describe the picture on the air, and it can only
// know about an edit that reached it. An edit made within the debounce
// window of a Send click has not -- so unless Send rebuilds first,
// `request_send()` waits on the *previous* generation, finds a result
// already ready for it, and transmits the current composition with
// latents refined for the one before it. That failure is invisible by
// construction: refined latents for the wrong picture still decode to a
// picture.
//
// **What this does not cover, deliberately: that `send()` calls it.**
// Driving `send()` here would open a modal message box -- no picture
// chosen, and no codec loaded -- and a modal in a headless test is a
// hang, which this suite treats as worse than a gap. The call is the
// first statement of `send()` and carries the reasoning above beside
// it. A test for it belongs wherever a panel is driven with a real
// codec, which this file is not.
void test_a_rebuild_consumes_the_pending_edit() {
    AppState state;
    QWidget host;
    host.resize(900, 700);
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* debounce = panel->findChild<QTimer*>(QStringLiteral("edit_debounce"));
    auto* editor = panel->findChild<OverlayEditor*>();
    check::is_true(debounce != nullptr && editor != nullptr,
                   "the panel has a debounce and an editor");
    if (debounce == nullptr || editor == nullptr) return;

    editor->refresh_item();
    check::is_true(debounce->isActive(), "an edit is pending");

    panel->schedule_optimization();
    check::is_true(!debounce->isActive(),
                   "a rebuild clears the pending edit rather than repeating it");
}

// The color button's size never moves.
//
// **This is the platform-independent form of a Windows-only CI
// failure.** `set_swatch` paints the current color onto the button as
// an icon, and it used to do so for the first time when a text item was
// first selected -- a QPushButton grows when it is handed an icon,
// measured 80x22 without and 80x24 with. The button therefore got 2 px
// taller at that moment and stayed there.
//
// On Linux that was invisible: a taller sibling on the same line of the
// wrapping row absorbed it, and the strip stayed 143. On Windows this
// button *is* the tallest thing on its line, so the strip went 185 to
// 189 and `test_the_strip_height_survives_a_selection` failed there and
// nowhere else. That row now lives in the floating selection palette
// rather than the strip, but the button's own size hint is unaffected
// by which container it sits in, so the same check still applies.
void test_the_color_button_does_not_change_size() {
    AppState state;
    QWidget host;
    host.resize(900, 700);
    auto* panel = new TransmitPanel(&state, &host);
    panel->setGeometry(0, 0, 900, 700);
    host.show();
    QCoreApplication::processEvents();

    // By object name, not text: three buttons on this panel now say
    // "Color..." (the text item's, and a rect's fill and stroke), so a
    // text-prefix search would silently pick whichever the traversal
    // visits last rather than the one this test means to watch.
    QPushButton* color =
        panel->findChild<QPushButton*>(QStringLiteral("text_color_button"));
    check::is_true(color != nullptr, "the panel has a color button");
    if (color == nullptr) return;
    const QSize idle = color->sizeHint();

    auto* editor = panel->findChild<OverlayEditor*>();
    check::is_true(editor != nullptr, "the panel has an overlay editor");
    // A *text* item, which is the only kind with a color and therefore
    // the only one that ever painted a swatch.
    editor->add_text(std::string("KD8XYZ"));
    QCoreApplication::processEvents();
    check::is_true(color->sizeHint() == idle,
                   "the color button is the same size with a text item selected");

    editor->remove_selected();
    QCoreApplication::processEvents();
    check::is_true(color->sizeHint() == idle,
                   "and the same size again with nothing selected");
}

// --- templates (docs/overlay-templates.md) ------------------------------

QPushButton* custom_fields_button(TransmitPanel* panel) {
    for (QPushButton* button : panel->findChildren<QPushButton*>()) {
        if (button->text().startsWith(QLatin1String("Custom fields"))) return button;
    }
    return nullptr;
}

// The built-ins ship as data beside the executable
// (`sstvae_copy_builtin_templates` in native/CMakeLists.txt, applied to
// this target too) and are listed in the fixed order the design names:
// None, then CQ, Reply, Reply with picture.
void test_the_template_combo_lists_none_then_the_builtins() {
    AppState state;
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* combo = panel->findChild<QComboBox*>(QStringLiteral("template_combo"));
    check::is_true(combo != nullptr, "the panel has a template combo");
    if (combo == nullptr) return;
    check::equal(combo->count(), 4, "None plus the three built-ins");
    if (combo->count() == 4) {
        check::equal(combo->itemText(0).toStdString(), std::string("None"), "none first");
        check::equal(combo->itemText(1).toStdString(), std::string("CQ"), "cq second");
        check::equal(combo->itemText(2).toStdString(), std::string("Reply"), "reply third");
        check::equal(combo->itemText(3).toStdString(), std::string("Reply with picture"),
                     "reply-picture fourth");
    }
}

// Selecting a template replaces the canvas and only enables the fields
// it actually uses -- "None"'s rows stay inert, "Reply"'s do not. The
// one custom field the built-ins ever declare ("Comment") fits inline
// (docs/overlay-templates.md), so the pop-up stays disabled throughout;
// its own overflow case is `test_a_template_with_more_custom_fields...`
// below, since none of the built-ins exercise it.
void test_selecting_a_template_loads_it_and_gates_its_fields() {
    AppState state;
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* combo = panel->findChild<QComboBox*>(QStringLiteral("template_combo"));
    auto* theircall = panel->findChild<QLineEdit*>(QStringLiteral("theircall_edit"));
    auto* comment = panel->findChild<QLineEdit*>(QStringLiteral("custom_field_edit_0"));
    QPushButton* custom = custom_fields_button(panel);
    auto* editor = panel->findChild<OverlayEditor*>();
    check::is_true(combo != nullptr && theircall != nullptr && comment != nullptr &&
                       custom != nullptr && editor != nullptr,
                   "the panel has a combo, a theircall field, an inline custom "
                   "field, a pop-up button and an editor");
    if (combo == nullptr || theircall == nullptr || comment == nullptr ||
        custom == nullptr || editor == nullptr) {
        return;
    }

    check::is_true(!theircall->isEnabled(), "their call starts disabled: no template");
    check::is_true(!comment->isEnabled(), "the inline Comment row starts disabled too");
    check::is_true(!custom->isEnabled(), "and the overflow pop-up has nothing to overflow");

    combo->setCurrentIndex(2);  // Reply
    QCoreApplication::processEvents();
    check::is_true(!editor->doc().items.empty(), "the template's items are on the canvas");
    check::is_true(theircall->isEnabled(), "reply uses {theircall}: enabled");
    check::is_true(comment->isEnabled(), "reply's Comment field: enabled inline");
    check::is_true(!custom->isEnabled(),
                   "still nothing for the pop-up: one field fits inline");

    combo->setCurrentIndex(0);  // None
    QCoreApplication::processEvents();
    check::is_true(editor->doc().items.empty(), "\"None\" clears the overlay");
    check::is_true(!theircall->isEnabled(), "and their call is disabled again");
    check::is_true(!comment->isEnabled(), "and so is the inline Comment row");
}

// The point of moving custom fields inline: typing updates the
// composite without a Done button, because `refresh_fields` runs on
// every keystroke via `on_custom_field_edited`.
void test_an_inline_custom_field_updates_live() {
    AppState state;
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* combo = panel->findChild<QComboBox*>(QStringLiteral("template_combo"));
    auto* comment = panel->findChild<QLineEdit*>(QStringLiteral("custom_field_edit_0"));
    auto* editor = panel->findChild<OverlayEditor*>();
    check::is_true(combo != nullptr && comment != nullptr && editor != nullptr,
                   "the panel has a combo, an inline field and an editor");
    if (combo == nullptr || comment == nullptr || editor == nullptr) return;

    combo->setCurrentIndex(1);  // CQ, which also declares a Comment field
    QCoreApplication::processEvents();
    check::is_true(comment->isEnabled(), "CQ's Comment field is enabled inline");

    comment->setText(QStringLiteral("QRZ?"));
    QCoreApplication::processEvents();
    const auto it = editor->fields().custom.find("Comment");
    check::is_true(it != editor->fields().custom.end() && it->second == "QRZ?",
                   "typing reaches the editor's fields without a Done button");

    // Switching away and back must not leave the stale value behind if
    // the label changes, and must restore it if the label is the same
    // -- CQ and Reply both declare "Comment", so the value should carry.
    combo->setCurrentIndex(2);  // Reply
    QCoreApplication::processEvents();
    check::equal(comment->text().toStdString(), std::string("QRZ?"),
                 "the same label keeps its value across a template switch");
}

// A template with more custom fields than fit inline
// (`TransmitPanel::MAX_INLINE_CUSTOM_FIELDS`) is the case the pop-up
// still exists for.
void test_a_template_with_more_custom_fields_than_fit_inline_overflows() {
    QTemporaryDir dir;
    check::is_true(dir.isValid(), "temp dir created");

    overlay::Doc doc;
    doc.name = "Many fields";
    overlay::TextItem item;
    item.text =
        "{field A}\n{field B}\n{field C}\n{field D}\n{field E}\n{field F}";
    doc.items.push_back(item);
    {
        QFile file(dir.filePath(QStringLiteral("many.json")));
        check::is_true(file.open(QIODevice::WriteOnly), "template file opened");
        file.write(QByteArray::fromStdString(overlay::to_json(doc)));
    }

    AppState state;
    state.config().folders.template_dir = dir.path().toStdString();
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* combo = panel->findChild<QComboBox*>(QStringLiteral("template_combo"));
    QPushButton* custom = custom_fields_button(panel);
    check::is_true(combo != nullptr && custom != nullptr,
                   "the panel has a combo and a pop-up button");
    if (combo == nullptr || custom == nullptr) return;

    int index = -1;
    for (int i = 0; i < combo->count(); ++i) {
        if (combo->itemText(i) == QLatin1String("Many fields")) index = i;
    }
    check::is_true(index >= 0, "the six-field template is listed");
    if (index < 0) return;

    combo->setCurrentIndex(index);
    QCoreApplication::processEvents();
    check::is_true(custom->isEnabled(), "the overflow button is enabled");
    check::is_true(custom->text().contains(QLatin1String("2")),
                   "it names the two fields (six declared, four inline)");

    // The first four (A..D) each have their own enabled inline row.
    for (int i = 0; i < 4; ++i) {
        auto* row = panel->findChild<QLineEdit*>(
            QStringLiteral("custom_field_edit_%1").arg(i));
        check::is_true(row != nullptr && row->isEnabled(),
                       "inline row is present and enabled");
    }
}

// Typing their call reaches the editor's substitution fields -- the
// wiring `refresh_fields` exists for. What substitution actually *does*
// with a filled field is `test_overlay_editor.cpp`'s to check; this is
// only "does the panel forward what the operator typed".
void test_typing_their_call_updates_the_editor_s_fields() {
    AppState state;
    state.config().callsign = "KC2G";
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* combo = panel->findChild<QComboBox*>(QStringLiteral("template_combo"));
    auto* theircall = panel->findChild<QLineEdit*>(QStringLiteral("theircall_edit"));
    auto* editor = panel->findChild<OverlayEditor*>();
    check::is_true(combo != nullptr && theircall != nullptr && editor != nullptr,
                   "the panel has a combo, a theircall field and an editor");
    if (combo == nullptr || theircall == nullptr || editor == nullptr) return;

    combo->setCurrentIndex(2);  // Reply
    QCoreApplication::processEvents();
    check::equal(editor->fields().builtin.at("mycall"), std::string("KC2G"),
                 "{mycall} came from the configured callsign, unprompted");

    theircall->setText(QStringLiteral("W1XYZ"));
    QCoreApplication::processEvents();
    const auto it = editor->fields().builtin.find("theircall");
    check::is_true(it != editor->fields().builtin.end() && it->second == "W1XYZ",
                   "their call reaches the editor's fields as typed");
}

// A template saved in the operator's configured folder is picked up the
// same way the built-ins are, at construction -- the path
// `refresh_templates` reads from `config().folders.template_dir`.
void test_a_template_from_the_configured_folder_is_listed() {
    QTemporaryDir dir;
    check::is_true(dir.isValid(), "temp dir created");

    overlay::Doc doc;
    doc.name = "Mine";
    overlay::TextItem item;
    item.text = "hi";
    doc.items.push_back(item);
    {
        QFile file(dir.filePath(QStringLiteral("mine.json")));
        check::is_true(file.open(QIODevice::WriteOnly), "template file opened for writing");
        file.write(QByteArray::fromStdString(overlay::to_json(doc)));
    }

    AppState state;
    state.config().folders.template_dir = dir.path().toStdString();
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* combo = panel->findChild<QComboBox*>(QStringLiteral("template_combo"));
    check::is_true(combo != nullptr, "the panel has a template combo");
    if (combo == nullptr) return;

    bool found = false;
    for (int i = 0; i < combo->count(); ++i) {
        if (combo->itemText(i) == QLatin1String("Mine")) found = true;
    }
    check::is_true(found, "a template from the configured folder is listed by name");
}

// --- rectangles (docs item 1) --------------------------------------------
//
// Scale and rotation moved out of this panel entirely, onto the
// editor's own drag handles and keyboard shortcuts
// (`test_overlay_editor.cpp` covers those). What is left here is what
// only lives in the floating selection palette: color/fill/stroke and
// stacking order, shown and hidden by item type rather than merely
// enabled -- the palette is not part of `control_strip()`, so it is
// exempt from that box's fixed-shape rule (see `build_selection_palette`).

// Selecting a rect shows the palette and its rect-only rows, hiding the
// text-only one; selecting text is the mirror image.
void test_selection_shows_and_hides_the_palette_s_rows_by_item_type() {
    AppState state;
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* editor = panel->findChild<OverlayEditor*>();
    auto* text_edit = panel->findChild<QPlainTextEdit*>();
    auto* fill_kind = panel->findChild<QComboBox*>(QStringLiteral("fill_kind_combo"));
    auto* text_color =
        panel->findChild<QPushButton*>(QStringLiteral("text_color_button"));
    auto* palette = panel->findChild<QFrame*>(QStringLiteral("selection_palette"));
    check::is_true(editor != nullptr && text_edit != nullptr && fill_kind != nullptr &&
                       text_color != nullptr && palette != nullptr,
                   "the panel has an editor, a text box, the fill-kind combo, "
                   "the text color button and the selection palette");
    if (editor == nullptr || text_edit == nullptr || fill_kind == nullptr ||
        text_color == nullptr || palette == nullptr) {
        return;
    }

    check::is_true(!palette->isVisible(), "the palette starts hidden: nothing selected");

    editor->add_rect();
    QCoreApplication::processEvents();
    check::is_true(palette->isVisible(), "selecting a rect shows the palette");
    check::is_true(!text_edit->isEnabled(), "the text box is disabled for a rect");
    check::is_true(fill_kind->isVisible(), "the fill row shows for a rect");
    check::is_true(!text_color->isVisible(), "and the text color row hides");

    editor->add_text("W1AW");
    QCoreApplication::processEvents();
    check::is_true(palette->isVisible(), "the palette stays visible for a text item");
    check::is_true(text_edit->isEnabled(), "the text box is enabled for text");
    check::is_true(text_color->isVisible(), "the text color row shows for text");
    check::is_true(!fill_kind->isVisible(), "and the fill row hides");

    editor->remove_selected();
    QCoreApplication::processEvents();
    check::is_true(!palette->isVisible(), "removing the selection hides the palette again");
}

// Choosing "Gradient" reveals the second color and angle; "Solid" or
// "No fill" hide them again -- `update_selection_palette` runs from the
// combo itself, not only from a fresh selection.
void test_choosing_a_gradient_fill_reveals_its_second_color_and_angle() {
    AppState state;
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* editor = panel->findChild<OverlayEditor*>();
    auto* fill_kind = panel->findChild<QComboBox*>(QStringLiteral("fill_kind_combo"));
    auto* fill_angle =
        panel->findChild<QDoubleSpinBox*>(QStringLiteral("fill_angle_spin"));
    check::is_true(editor != nullptr && fill_kind != nullptr && fill_angle != nullptr,
                   "the panel has an editor, the fill-kind combo and its angle spin");
    if (editor == nullptr || fill_kind == nullptr || fill_angle == nullptr) return;

    editor->add_rect();
    QCoreApplication::processEvents();
    check::is_true(!fill_angle->isVisible(),
                   "a freshly added rect's own fill kind is not gradient yet");

    fill_kind->setCurrentIndex(fill_kind->findData(QStringLiteral("gradient")));
    QCoreApplication::processEvents();
    check::is_true(fill_angle->isVisible(), "choosing Gradient reveals the angle row");

    fill_kind->setCurrentIndex(fill_kind->findData(QStringLiteral("solid")));
    QCoreApplication::processEvents();
    check::is_true(!fill_angle->isVisible(), "and Solid hides it again");
}

// The fill-kind combo writes straight back into the selected rect -- the
// same wiring `on_selection` reads it from.
void test_rect_fill_kind_combo_writes_back_to_the_item() {
    AppState state;
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* editor = panel->findChild<OverlayEditor*>();
    auto* fill_kind = panel->findChild<QComboBox*>(QStringLiteral("fill_kind_combo"));
    check::is_true(editor != nullptr && fill_kind != nullptr,
                   "the panel has an editor and the fill-kind combo");
    if (editor == nullptr || fill_kind == nullptr) return;

    editor->add_rect();
    QCoreApplication::processEvents();

    fill_kind->setCurrentIndex(fill_kind->findData(QStringLiteral("gradient")));
    QCoreApplication::processEvents();
    check::equal(std::get<overlay::RectItem>(*editor->selected_item()).fill_kind,
                 std::string("gradient"), "the fill-kind combo writes fill_kind");
}

// Remove lives in the palette now, beside color and stacking order
// rather than in the tool row -- see `build_selection_palette`'s
// comment on why it belongs with the selection's own controls.
void test_the_palette_s_remove_button_removes_the_selection() {
    AppState state;
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* editor = panel->findChild<OverlayEditor*>();
    auto* remove = panel->findChild<QPushButton*>(QStringLiteral("remove_button"));
    check::is_true(editor != nullptr && remove != nullptr,
                   "the panel has an editor and a remove button");
    if (editor == nullptr || remove == nullptr) return;

    editor->add_text("N0CALL");
    QCoreApplication::processEvents();
    check::equal(static_cast<int>(editor->doc().items.size()), 1, "one item to begin with");

    remove->click();
    QCoreApplication::processEvents();
    check::equal(static_cast<int>(editor->doc().items.size()), 0,
                 "clicking Remove takes it out of the document");
}

// --- stacking order (docs item 2) -----------------------------------------

void test_the_z_order_buttons_reorder_the_selected_item() {
    AppState state;
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* editor = panel->findChild<OverlayEditor*>();
    auto* raise = panel->findChild<QPushButton*>(QStringLiteral("raise_button"));
    auto* lower = panel->findChild<QPushButton*>(QStringLiteral("lower_button"));
    check::is_true(editor != nullptr && raise != nullptr && lower != nullptr,
                   "the panel has an editor and raise/lower buttons");
    if (editor == nullptr || raise == nullptr || lower == nullptr) return;

    check::is_true(!raise->isEnabled(), "raise starts disabled: nothing selected");

    editor->add_text("A");
    editor->add_text("B");  // selected, already on top
    QCoreApplication::processEvents();
    check::is_true(!raise->isEnabled(), "the top item cannot be raised further");
    check::is_true(lower->isEnabled(), "but it can be lowered");

    lower->click();
    QCoreApplication::processEvents();
    check::equal(std::get<overlay::TextItem>(editor->doc().items[0]).text,
                 std::string("B"), "clicking Lower moves the selection down");
    check::is_true(raise->isEnabled(), "now it can be raised back");
}

// --- "Save as template..." defaults to the loaded name --------------------

// `save_as_template` opens a modal `QInputDialog`; the test answers it
// from a zero-delay timer, which is the standard way to drive a Qt
// modal without blocking the test itself.
void test_saving_a_loaded_template_defaults_to_its_own_name() {
    AppState state;
    QWidget host;
    auto* panel = new TransmitPanel(&state, &host);
    host.show();
    QCoreApplication::processEvents();

    auto* combo = panel->findChild<QComboBox*>(QStringLiteral("template_combo"));
    QPushButton* save = nullptr;
    for (QPushButton* button : panel->findChildren<QPushButton*>()) {
        if (button->text().contains(QLatin1String("Save as template"))) save = button;
    }
    check::is_true(combo != nullptr && save != nullptr,
                   "the panel has a template combo and a save button");
    if (combo == nullptr || save == nullptr) return;

    combo->setCurrentIndex(2);  // "Reply"
    QCoreApplication::processEvents();

    QString seen_default;
    QTimer::singleShot(0, save, [&seen_default] {
        QWidget* modal = QApplication::activeModalWidget();
        if (auto* dialog = qobject_cast<QDialog*>(modal)) {
            if (auto* edit = dialog->findChild<QLineEdit*>()) {
                seen_default = edit->text();
            }
            dialog->reject();  // decline the save; only the default is under test
        }
    });
    save->click();
    QCoreApplication::processEvents();

    check::equal(seen_default.toStdString(), std::string("Reply"),
                 "the dialog opened pre-filled with the loaded template's name");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);

    test_the_strip_height_survives_a_selection();
    test_the_level_controls_are_one_flow_item();
    test_the_color_button_does_not_change_size();
    test_an_edit_defers_the_rebuild();
    test_a_rebuild_consumes_the_pending_edit();
    test_the_template_combo_lists_none_then_the_builtins();
    test_selecting_a_template_loads_it_and_gates_its_fields();
    test_an_inline_custom_field_updates_live();
    test_a_template_with_more_custom_fields_than_fit_inline_overflows();
    test_typing_their_call_updates_the_editor_s_fields();
    test_a_template_from_the_configured_folder_is_listed();
    test_selection_shows_and_hides_the_palette_s_rows_by_item_type();
    test_choosing_a_gradient_fill_reveals_its_second_color_and_angle();
    test_rect_fill_kind_combo_writes_back_to_the_item();
    test_the_palette_s_remove_button_removes_the_selection();
    test_the_z_order_buttons_reorder_the_selected_item();
    test_saving_a_loaded_template_defaults_to_its_own_name();
    return check::report("transmit panel");
}
