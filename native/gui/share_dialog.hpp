// Handing a template to another station, without a cable or a file
// manager.
//
// A template is a few hundred bytes of JSON -- small enough to fit in
// one QR code that a phone reads off the screen, which is the whole
// reason this is a dialog and not an export file. The same payload sits
// in a text box underneath it, because the code is useless on a device
// with no camera and because a template pasted into a chat window or an
// email is the same thing sent a slower way.
//
// There is deliberately no share *format*: the payload is what
// `overlay::to_json` writes, so the text in this box is the contents of
// a `.json` in the template folder. Anything that already understands a
// template understands this, and the operator can read it.
//
// Import is `overlay::sanitize_imported` plus a save into the template
// folder -- see that function for why a document from elsewhere is not
// handed the local filesystem.

#ifndef SSTVAE_GUI_SHARE_DIALOG_HPP
#define SSTVAE_GUI_SHARE_DIALOG_HPP

#include <QDialog>
#include <QImage>
#include <QString>
#include <QWidget>

#include <optional>
#include <string>

#include "overlay/model.hpp"

class QLabel;
class QPlainTextEdit;

namespace sstvae::gui {

// The code for `payload`, as a 1-bit-per-module image at one pixel per
// module -- scaled up by whoever draws it, with `Qt::FastTransformation`
// so the modules stay square.
//
// `std::nullopt` when the payload does not fit in any QR version. That
// is a real case (a document with a dozen items), not a defensive one,
// and the dialog says so rather than showing a code nothing can read.
std::optional<QImage> qr_image(const std::string& payload);

class ShareDialog : public QDialog {
    Q_OBJECT

public:
    // `doc` is shown as a code and as text. The dialog is read-only: it
    // does not write to the template folder, since what it is sharing
    // is already there (or on the canvas, which is the operator's).
    ShareDialog(const overlay::Doc& doc, QWidget* parent = nullptr);

private:
    void copy_payload();

    std::string payload_;
    QLabel* code_ = nullptr;
    QPlainTextEdit* text_ = nullptr;
};

}  // namespace sstvae::gui

#endif
