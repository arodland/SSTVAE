// Opening a picture with Qt's decoders, for the two apps that have Qt.
//
// `core/images/images.hpp` is the Qt-free loader: stb, a fixed list of
// formats (PNG, JPEG, BMP, GIF, TGA, PSD, HDR, PIC, PNM), and the
// implementation the golden vectors and `pytest --native` are written
// against. It stays exactly that. This is the layer the *application*
// loads through, and the only thing it adds is Qt's format list --
// which is the platform's, not ours: TIFF, WEBP, ICO, ICNS, and
// whatever else the imageformats plugins present on the machine can
// read.
//
// **Why not simply make `images::load` do this.** The headless CLI, the
// golden-vector tests and the pybind11 parity module must build with no
// Qt at all, and the stb loader is the one Python's `images.py` is
// compared against. So this is an added layer rather than a
// replacement, and it is a separate library for the same reason
// `core/audio/qt/` is.
//
// **Nothing that used to open stops opening.** Qt has no PSD, HDR or
// PIC handler and its TGA handler is a plugin that may be absent, so a
// file Qt declines is handed to `images::load` before it is reported as
// unreadable. The consequence for `readable_extensions` below is that
// the list is Qt's *plus* stb's, and both dialogs in the app can offer
// the same one.
//
// **EXIF orientation is applied exactly once, and by preference from
// our own tag reader.** `images::exif_orientation` reads the JPEG APP1
// segment and the PNG `eXIf` chunk and is held to Pillow's answers by
// `tests/test_native_parity.py -k exif`; Qt's `setAutoTransform` is
// therefore off, and Qt's own reading of the tag
// (`QImageReader::transformation`) is consulted only for a file our
// reader has nothing to say about -- a TIFF or a WEBP. Getting this
// wrong in the obvious direction (leave autoTransform on *and* apply
// our tag) rotates every photograph from a phone twice.

#ifndef SSTVAE_IMAGES_QT_QTIMAGES_HPP
#define SSTVAE_IMAGES_QT_QTIMAGES_HPP

#include <string>
#include <vector>

#include <QImage>
#include <QImageIOHandler>

#include "images/types.hpp"

namespace sstvae::images::qt {

// The EXIF orientation (1..8) a Qt transformation corresponds to -- the
// inverse of what Qt's own handlers apply on the way in, and the one
// piece of arithmetic in this layer with no oracle in `sstvae_core`.
//
// Public so `test_images_qt.cpp` can pin it against the *installed* Qt
// rather than against a copy of the table, the same reason
// `tx::PttWatchdog` is in a header. Reached only for a file our own tag
// reader cannot parse; anything unrecognised is 1, which is upright.
int exif_orientation_of(QImageIOHandler::Transformations transformation);

// Lowercase extensions, no dots, sorted and deduplicated: everything
// `QImageReader` reports on this machine plus the stb-only formats the
// fallback still opens. Meant for a file dialog's filter, so that what
// the dialog offers and what `load` accepts are one list rather than
// two -- the hardcoded filter this replaced advertised `*.webp`, which
// the stb loader behind it could not read at all.
std::vector<std::string> readable_extensions();

// Open any of those, upright. Throws with the file name and the
// decoder's reason on failure -- both decoders' reasons, when Qt and
// stb each declined, since the useful one is whichever recognised the
// format.
Picture load(const std::string& path);

// The same decode, stopping at the QImage -- so the overlay renderer,
// which paints inset images and wants their alpha channel, gets the
// same format list and the same orientation handling as the picture
// being composed. Returns a null QImage on failure, which is that
// caller's existing contract for an unreadable inset.
QImage load_qimage(const std::string& path);

}  // namespace sstvae::images::qt

#endif
