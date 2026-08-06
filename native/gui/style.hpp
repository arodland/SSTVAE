// Shared appearance: colours, secondary text, and the two widgets that
// existed in three copies each.
//
// This file is where things go that more than one panel has to agree
// about. Everything in it was previously either duplicated verbatim
// (`place_banner`, `to_pixmap`), kept in step by hand (the empty-canvas
// colours), or done three different ways in three files (dimmed text:
// a palette override in the settings dialog, `setEnabled(false)` on the
// receive card, bold in the log pane).
//
// **No stylesheets, here or anywhere.** Setting one on any widget makes
// Qt wrap the application style in `QStyleSheetStyle`, whose defaults
// are not the platform's -- most visibly, padding drops to zero, so
// every combo, spin box and line edit in the app gets its text jammed
// against the left border. One `color:` rule on one label did that to a
// whole window once. Everything here goes through `QPalette` or a
// painter.

#ifndef SSTVAE_GUI_STYLE_HPP
#define SSTVAE_GUI_STYLE_HPP

#include <QColor>
#include <QLabel>
#include <QString>

#include <initializer_list>

#include "images/types.hpp"

class QImage;
class QPalette;
class QPixmap;
class QWidget;

namespace sstvae::gui {

class ErrorBanner;

namespace style {

// --- colour ------------------------------------------------------------------

// WCAG 2.1 relative luminance and contrast ratio. Exposed because the
// point of `secondary_text` is a number, and a number nobody can check
// is a preference wearing a lab coat.
double relative_luminance(const QColor& color);
double contrast_ratio(const QColor& a, const QColor& b);

// The minimum this project holds itself to: WCAG AA for body text.
inline constexpr double MIN_CONTRAST = 4.5;

// A dimmed foreground that is still legible, for explanatory text.
//
// **Not `QPalette::Disabled, QPalette::WindowText`,** which is what
// this replaced and which measured **1.62:1** against the window on the
// default light theme -- against a 4.5:1 requirement, with normal
// labels in the same dialog at 18.3:1. That colour is chosen by the
// style to say "you cannot use this control", and help text is not a
// disabled control; borrowing it made the app's entire documentation
// surface the least readable thing in the window.
//
// Derived rather than hardcoded so it follows the theme: start at
// `WindowText`, walk toward `Window`, and stop at the last step that
// still clears `MIN_CONTRAST`. So it is as quiet as it can be while
// remaining readable, in a light theme and a dark one alike.
QColor secondary_text(const QPalette& palette);

// Semantic colours, from one base each, so "the red" cannot drift.
//
// It had: `#b3261e` for the PTT lamp, `#7a1f1a` for the error banner
// and `rgb(255,60,60)` for the waterfall's CLIP marker were three
// literals in three files for one idea.
namespace color {

QColor danger();          // the base: "this is wrong / the radio is keyed"
QColor danger_surface();  // a fill to put light text on
QColor on_danger();       // that light text
QColor danger_bright();   // over the waterfall's own dark image
QColor caution();         // approaching clipping
QColor ok();              // a healthy level
QColor on_ok();           // dark text that reads on an ok() fill

// The picture viewport, deliberately theme-independent: a photograph is
// judged against a neutral dark ground whatever the desktop is doing,
// which is what every image viewer does. Shared so the receive box and
// the composer cannot drift apart -- they are a pair, and the operator
// sees them side by side.
QColor viewport();        // around the picture
QColor viewport_frame();  // the 4:3 area before a picture arrives
QColor viewport_edge();   // its hairline
QColor viewport_text();   // the empty-state caption

}  // namespace color

// --- text --------------------------------------------------------------------

// The width a note reserves at its left for a disclosure triangle.
//
// Reserved whether or not the note has one, so plain help and a
// disclosure's summary start in the same column and the triangle hangs
// in the margin beside the text rather than pushing it right. Measured
// from `reference`'s style, so it matches the platform.
int note_gutter(const QWidget* reference);

// Dimmed explanatory text. Wraps, and reserves `note_gutter` at the left.
QLabel* note(const QString& text, QWidget* parent);

// The same treatment applied to a widget that already exists.
//
// The alternative in use was `setEnabled(false)`, which is an
// appearance change expressed as a state change: it also takes the
// widget out of the accessibility tree as interactive-but-unavailable,
// and it says "you cannot use this" about a label nobody was going to
// click.
void dim(QWidget* widget);

// Undo `dim`. Clears the widget's palette override entirely, so it goes
// back to inheriting whatever its parent and the theme say -- which is
// the only way back that stays correct across a theme change.
void undim(QWidget* widget);

// A long note as one line, with the rest behind a disclosure.
//
// The settings dialog had grown to where the Transmit tab was roughly
// 60% grey prose by area -- nine lines for one checkbox -- which is the
// point at which nobody reads any of it. `summary` is always visible;
// `detail` appears when the operator asks. Default collapsed, and the
// state is deliberately not persisted: it is a reading aid, not a
// setting.
//
// A factory rather than a class: the toggle is a lambda bound to the
// returned widget, so there is nothing for moc to do.
QWidget* note_with_detail(const QString& summary, const QString& detail,
                          QWidget* parent);

// A row of widgets, left-aligned, with the slack pushed to the right.
// `stretch_last` names the index that should absorb the slack instead.
QWidget* row(QWidget* parent, std::initializer_list<QWidget*> widgets,
             int stretch_last = -1);

// A QLabel that shortens with an ellipsis instead of being cut off.
//
// Several labels carry `QSizePolicy::Ignored` horizontally so they
// cannot pin the window's minimum width -- which is right, and which
// means they are routinely handed less width than their text needs. A
// plain QLabel then *clips*, and the settings dialog's own scroll-area
// comment says why that is the wrong answer: "the operator cannot tell
// whether the text is cut off or simply ends". A callsign missing from
// the end of a status line reads as a callsign that was not decoded.
//
// **Elides in `paintEvent`, and does not override `setText`.**
// `QLabel::setText` is not virtual, so a subclass that shadows it is
// silently bypassed through a `QLabel*` -- and half of Qt holds labels
// that way. Eliding at paint time leaves `text()` returning the truth,
// needs no new API at the call sites, and cannot be routed around.
//
// No Q_OBJECT: a plain virtual override, nothing for moc to do.
class ElidingLabel : public QLabel {
public:
    using QLabel::QLabel;

protected:
    void paintEvent(QPaintEvent* event) override;
    // The full text on hover, so nothing is unrecoverable. `event()` is
    // virtual where `setToolTip` is not, which is why the tooltip is
    // answered here rather than written from `paintEvent` -- setting a
    // property during a paint is how a repaint loop starts.
    bool event(QEvent* event) override;

private:
    bool elided_ = false;
};

// --- pictures ----------------------------------------------------------------

QImage to_qimage(const images::Picture& picture);
QPixmap to_pixmap(const images::Picture& picture);

// SNR for display: "SNR 8.3 dB", or a placeholder when there is none.
//
// **Not `rx::fmt_snr`,** which the panels used to borrow. That one is
// the *CLI's* format and mirrors `sstvae/rx/engine.py` exactly, so it
// is not ours to change -- and it produces "8.3dB" with no space, has
// the English word "SNR" baked into a layer below the GUI where no
// translator can reach it, and returns an empty string for NaN. On
// screen an absent field is indistinguishable from a field that was
// never going to be there.
QString fmt_snr_db(double snr_db);

// --- layout ------------------------------------------------------------------

// Keep a floating error banner across the top of `target`.
//
// Floating rather than laid out: in the layout the banner displaced
// everything below it, so an error in one pane pushed that pane's
// picture down and the two stopped lining up -- precisely what the
// equal-panes work exists to prevent. This costs no layout at all.
//
// `heightForWidth`, not `sizeHint`: the message wraps, and a hint taken
// at unconstrained width is one line, which clipped a long device name
// to a sliver of its own text.
void place_over(ErrorBanner* banner, const QWidget* target);

}  // namespace style
}  // namespace sstvae::gui

#endif
