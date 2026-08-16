#include "images/qt/qtimages.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <QBuffer>
#include <QByteArray>
#include <QFileInfo>
#include <QImage>
#include <QImageIOHandler>
#include <QImageReader>
#include <QIODevice>
#include <QList>
#include <QString>
#include <QTransform>

#include "images/images.hpp"

namespace sstvae::images::qt {
namespace {

// The formats stb reads and Qt does not, or does not reliably: Qt has no
// PSD, HDR or PIC handler at all, and TGA and the PNM family arrive in
// the `qtimageformats` module, which a distribution may not have
// installed. They are in the dialog's filter because the fallback in
// `decode` really does open them -- offering a format the loader cannot
// read is what the constant this replaced did.
constexpr const char* STB_EXTENSIONS[] = {
    "bmp", "gif", "hdr", "jpeg", "jpg", "pgm", "pic",
    "png", "pnm", "ppm", "psd", "tga",
};

// Clockwise, which is what a positive angle means in Qt's y-down
// coordinate system and what the EXIF table's rotations are stated as.
QImage rotate_cw(const QImage& img, int degrees) {
    return img.transformed(QTransform().rotate(degrees));
}

// `images::apply_orientation` in the QImage domain, so an inset keeps its
// alpha channel -- going through `Picture` to reuse that function would
// flatten transparency, and a transparent PNG inset is an ordinary thing
// to compose with. Held equal to that function on every one of the eight
// values by `test_images_qt.cpp`, which is the only reason writing it a
// second time is acceptable.
//
// Two `transformed` calls rather than one composed QTransform: the
// composition order is precisely what is easy to get backwards, and
// applying the rotation and then the mirror is unambiguous on the page.
// `QTransform().scale(-1, 1)` rather than `QImage::mirrored`, which Qt
// 6.9 deprecates in favour of `flipped` -- one spelling that works
// across the whole supported range.
QImage oriented(const QImage& img, int orientation) {
    // The overwhelmingly common case, and free: no transform is built.
    if (orientation <= 1 || orientation > 8) return img;
    const QTransform mirror_h = QTransform().scale(-1, 1);
    const QTransform mirror_v = QTransform().scale(1, -1);
    switch (orientation) {
        case 2: return img.transformed(mirror_h);                       // mirror
        case 3: return rotate_cw(img, 180);                             // 180
        case 4: return img.transformed(mirror_v);                       // flip
        case 5: return rotate_cw(img, 90).transformed(mirror_h);        // transpose
        case 6: return rotate_cw(img, 90);                              // 90 CW
        case 7: return rotate_cw(img, 90).transformed(mirror_v);        // transverse
        case 8: return rotate_cw(img, 270);                             // 90 CCW
        default: return img;
    }
}

QImage to_qimage(const Picture& p) {
    if (p.empty()) return QImage();
    // The Picture's rows are tightly packed; QImage would otherwise
    // assume a 4-byte-aligned stride and shear the picture. copy()
    // because the wrapper does not own the bytes.
    const QImage view(p.rgb.data(), p.width, p.height, p.width * 3,
                      QImage::Format_RGB888);
    return view.copy();
}

Picture from_qimage(const QImage& image) {
    const QImage rgb = image.convertToFormat(QImage::Format_RGB888);
    Picture out(rgb.width(), rgb.height());
    for (int y = 0; y < rgb.height(); ++y) {
        std::copy_n(rgb.constScanLine(y), static_cast<std::size_t>(rgb.width()) * 3,
                    out.rgb.data() + static_cast<std::size_t>(y) * rgb.width() * 3);
    }
    return out;
}

// One decode attempt. `format` empty means "work it out from the bytes".
QImage qt_read(const std::vector<std::uint8_t>& bytes, const QByteArray& format,
               QImageIOHandler::Transformations* transform, QString* error) {
    // fromRawData: no copy, and `bytes` outlives the reader. The buffer
    // is opened read-only, so nothing here can write through it.
    QByteArray raw = QByteArray::fromRawData(reinterpret_cast<const char*>(bytes.data()),
                                             static_cast<qsizetype>(bytes.size()));
    QBuffer buffer(&raw);
    if (!buffer.open(QIODevice::ReadOnly)) {
        *error = QStringLiteral("could not open the file's bytes for reading");
        return QImage();
    }
    QImageReader reader(&buffer, format);
    // Off, deliberately: the orientation tag is applied once, by
    // `oriented`, from whichever reader understood the file. See the
    // header.
    reader.setAutoTransform(false);
    const QImage image = reader.read();
    if (image.isNull()) {
        *error = reader.errorString();
        return QImage();
    }
    *transform = reader.transformation();
    return image;
}

// Qt first, stb second, and a null return with `error` set to a complete
// message when neither could read it -- complete because the file-read
// failures already name the file and a caller prepending the path to
// those would say it twice.
QImage decode(const std::string& path, std::string* error) {
    std::vector<std::uint8_t> bytes;
    try {
        bytes = images::read_picture_bytes(path);
    } catch (const std::exception& e) {
        *error = e.what();
        return QImage();
    }

    QImageIOHandler::Transformations transform = QImageIOHandler::TransformationNone;
    QString qt_error;
    // Detected from the content first, and only then from the extension.
    // The reader is given the bytes rather than the path, which is what
    // keeps the file read shared with the EXIF parser -- but it also
    // means Qt has no suffix to try, so the suffix attempt is added back
    // here. This order rather than Qt's own (suffix, then content): a
    // misnamed file is the ordinary case and content detection cannot be
    // fooled by one, while the suffix rescues only the formats whose
    // headers are too weak to detect at all -- an old TGA with no footer
    // above all.
    QImage image = qt_read(bytes, QByteArray(), &transform, &qt_error);
    if (image.isNull()) {
        const QByteArray suffix =
            QFileInfo(QString::fromStdString(path)).suffix().toLower().toLatin1();
        if (!suffix.isEmpty()) {
            QString ignored;
            image = qt_read(bytes, suffix, &transform, &ignored);
        }
    }

    if (!image.isNull()) {
        // Our tag reader knows JPEG and PNG and is the one held to
        // Pillow's answers; Qt's covers the formats it does not.
        int orientation = images::exif_orientation(bytes.data(), bytes.size());
        if (orientation == 1) orientation = exif_orientation_of(transform);
        return oriented(image, orientation);
    }

    // Whatever Qt has no handler for. `images::load` applies the
    // orientation itself, so nothing more is done to what it returns.
    try {
        return to_qimage(images::load(path));
    } catch (const std::exception& e) {
        // Both reasons: which one is informative depends on which
        // decoder recognised the format, and the caller cannot know.
        // stb's already names the file, so it carries the message and
        // Qt's is appended to it.
        *error = e.what();
        const std::string qt = qt_error.toStdString();
        if (!qt.empty() && qt != *error) *error += " (Qt: " + qt + ")";
        return QImage();
    }
}

}  // namespace

// Written out rather than composed, because 5 and 7 are the pair that is
// easy to swap -- both are a rotation plus a mirror -- and a silent
// disagreement with Qt here is a sideways transmission.
int exif_orientation_of(QImageIOHandler::Transformations transformation) {
    switch (static_cast<int>(transformation)) {
        case QImageIOHandler::TransformationNone:               return 1;
        case QImageIOHandler::TransformationMirror:             return 2;
        case QImageIOHandler::TransformationRotate180:          return 3;
        case QImageIOHandler::TransformationFlip:               return 4;
        case QImageIOHandler::TransformationFlipAndRotate90:    return 5;
        case QImageIOHandler::TransformationRotate90:           return 6;
        case QImageIOHandler::TransformationMirrorAndRotate90:  return 7;
        case QImageIOHandler::TransformationRotate270:          return 8;
        default:                                                return 1;
    }
}

std::vector<std::string> readable_extensions() {
    std::vector<std::string> out;
    const QList<QByteArray> qt_formats = QImageReader::supportedImageFormats();
    out.reserve(static_cast<std::size_t>(qt_formats.size()) + std::size(STB_EXTENSIONS));
    for (const QByteArray& format : qt_formats) {
        out.push_back(QString::fromLatin1(format).toLower().toStdString());
    }
    for (const char* extension : STB_EXTENSIONS) out.emplace_back(extension);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

Picture load(const std::string& path) {
    std::string error;
    const QImage image = decode(path, &error);
    if (image.isNull()) {
        throw std::runtime_error(error.empty() ? "cannot read " + path : error);
    }
    return from_qimage(image);
}

QImage load_qimage(const std::string& path) {
    std::string ignored;
    return decode(path, &ignored);
}

}  // namespace sstvae::images::qt
