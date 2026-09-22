#include "share_dialog.hpp"

#include <QApplication>
#include <QClipboard>
#include <QDialogButtonBox>
#include <QFontDatabase>
#include <QGuiApplication>
#include <QLabel>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScreen>
#include <QVBoxLayout>

#include <algorithm>
#include <exception>

#include "qrcodegen.hpp"
#include "style.hpp"

namespace sstvae::gui {

namespace {

// The white margin, in modules, painted *into* the image rather than
// left to the dialog behind it. Two reasons, and the first was measured:
// at the specification's minimum of 4, OpenCV's detector could not find
// this code at all until the image was given a wider border, while 6
// and 8 read first time. And the app's palette is dark, so a code that
// relied on its background for quiet zone would have a dark one.
constexpr int QUIET_MODULES = 8;

}  // namespace

std::optional<QImage> qr_image(const std::string& payload) {
    qrcodegen::QrCode code = qrcodegen::QrCode::encodeText("", qrcodegen::QrCode::Ecc::LOW);
    try {
        // LOW, deliberately: this is read off a lit screen from a few
        // inches away, where the error budget buys nothing and the
        // version it costs makes the modules smaller -- which is the
        // thing that actually decides whether a phone can read it.
        code = qrcodegen::QrCode::encodeText(payload.c_str(), qrcodegen::QrCode::Ecc::LOW);
    } catch (const std::exception&) {
        return std::nullopt;  // longer than any version holds
    }

    const int size = code.getSize() + 2 * QUIET_MODULES;
    QImage image(size, size, QImage::Format_RGB32);
    image.fill(Qt::white);
    for (int y = 0; y < code.getSize(); ++y) {
        for (int x = 0; x < code.getSize(); ++x) {
            if (code.getModule(x, y)) {
                image.setPixel(x + QUIET_MODULES, y + QUIET_MODULES, qRgb(0, 0, 0));
            }
        }
    }
    return image;
}

ShareDialog::ShareDialog(const overlay::Doc& doc, QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("Share template"));
    // -1: the compact form. The payload is scanned and pasted, never
    // read as a file, and the indentation is a third of the bytes --
    // which on a QR code is versions, which is module size.
    payload_ = overlay::to_json(doc, -1);

    auto* column = new QVBoxLayout(this);

    code_ = new QLabel(this);
    code_->setObjectName(QStringLiteral("share_qr"));
    code_->setAlignment(Qt::AlignCenter);
    if (const std::optional<QImage> image = qr_image(payload_)) {
        // **`FastTransformation`, and an integer multiple.** A smoothed
        // or fractionally-scaled QR code is blurred module edges, which
        // is exactly what a decoder is trying to measure.
        const int modules = image->width();
        const int target = 440;
        const int scale = std::max(1, target / modules);
        code_->setPixmap(QPixmap::fromImage(
            image->scaled(modules * scale, modules * scale, Qt::KeepAspectRatio,
                          Qt::FastTransformation)));
    } else {
        // A real case, not a defensive one: a document with a dozen
        // items outgrows the largest version. The text below still
        // works, which is what the message points at.
        code_->setText(tr("This template is too large for a QR code.\n"
                          "Copy the text below instead."));
        code_->setWordWrap(true);
    }
    column->addWidget(code_);

    auto* hint = new QLabel(tr("Scan the code, or copy the text below."), this);
    hint->setWordWrap(true);
    column->addWidget(hint);

    text_ = new QPlainTextEdit(QString::fromStdString(payload_), this);
    text_->setObjectName(QStringLiteral("share_text"));
    text_->setReadOnly(true);
    text_->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    text_->setMaximumHeight(120);
    column->addWidget(text_);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    QPushButton* copy = buttons->addButton(tr("&Copy"), QDialogButtonBox::ActionRole);
    copy->setObjectName(QStringLiteral("share_copy"));
    connect(copy, &QPushButton::clicked, this, &ShareDialog::copy_payload);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    column->addWidget(buttons);
}

void ShareDialog::copy_payload() {
    QGuiApplication::clipboard()->setText(QString::fromStdString(payload_));
}

}  // namespace sstvae::gui
