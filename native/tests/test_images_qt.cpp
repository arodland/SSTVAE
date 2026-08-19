// Loading a picture through Qt's decoders: the format list, the fallback
// to stb, and the one thing that can be silently wrong -- orientation.
//
// The formats themselves need no test of ours; they are Qt's plugins and
// Qt tests them. What is checked here is the three joins this layer
// makes, each of which fails quietly:
//
//   * A file Qt cannot read must still open, because stb reads formats Qt
//     has no handler for (PSD, HDR, PIC) and the app must not lose a
//     format by gaining a loader.
//   * A file Qt *can* read must produce the same picture the stb path
//     would, wherever both can read it -- byte for byte on a lossless
//     format. That is what makes this an added layer rather than a
//     second implementation.
//   * The EXIF orientation tag must be applied exactly once. Ours is
//     applied for JPEG and PNG (the reader held to Pillow's answers) and
//     Qt's reported transformation for anything else, and the ways of
//     getting that wrong -- twice, not at all, or with 5 and 7 swapped --
//     all leave a picture that loads and looks like a decoder bug.
//
// The oracle throughout is `images::load` and `images::apply_orientation`,
// which have their own tests and are what the Python parity suite
// compares against.

#include "images/qt/qtimages.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

#include <QBuffer>
#include <QByteArray>
#include <QGuiApplication>
#include <QIODevice>
#include <QImage>
#include <QImageReader>
#include <QImageWriter>
#include <QString>

#include "check.hpp"
#include "images/images.hpp"
#include "images/types.hpp"
#include "stb_image_write.h"

namespace check = sstvae::check;
namespace images = sstvae::images;
using sstvae::images::Picture;

namespace {

// Every pixel encodes its own position, so no two of the eight EXIF
// transforms of it are equal and a wrong one cannot hide behind a
// symmetry. Non-square, because 5..8 exchange the axes.
Picture ramp(int width, int height) {
    Picture p(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 3;
            p.rgb[i] = static_cast<std::uint8_t>(x * 3 % 251);
            p.rgb[i + 1] = static_cast<std::uint8_t>(y * 7 % 241);
            p.rgb[i + 2] = static_cast<std::uint8_t>((x * 5 + y * 11) % 239);
        }
    }
    return p;
}

void write_bytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
}

double mean_abs_diff(const Picture& a, const Picture& b) {
    if (a.width != b.width || a.height != b.height || a.rgb.size() != b.rgb.size()) {
        return 1e9;
    }
    double err = 0.0;
    for (std::size_t i = 0; i < a.rgb.size(); ++i) {
        err += std::abs(static_cast<double>(a.rgb[i]) - b.rgb[i]);
    }
    return a.rgb.empty() ? 0.0 : err / a.rgb.size();
}

// --- files with an orientation tag ------------------------------------
//
// The same construction `test_images.cpp` uses: a minimal but real APP1
// segment, big-endian, spliced in after SOI where a camera puts it.

std::vector<std::uint8_t> app1_segment(int orientation) {
    std::vector<std::uint8_t> tiff = {
        'M',  'M',  0x00, 0x2a,
        0x00, 0x00, 0x00, 0x08,
        0x00, 0x01,
        0x01, 0x12,              // tag: Orientation
        0x00, 0x03,              // type: SHORT
        0x00, 0x00, 0x00, 0x01,  // count
        static_cast<std::uint8_t>(orientation >> 8),
        static_cast<std::uint8_t>(orientation & 0xff),
        0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,  // no next IFD
    };
    std::vector<std::uint8_t> body = {'E', 'x', 'i', 'f', 0x00, 0x00};
    body.insert(body.end(), tiff.begin(), tiff.end());
    const std::size_t length = body.size() + 2;
    std::vector<std::uint8_t> seg = {0xFF, 0xE1,
                                     static_cast<std::uint8_t>(length >> 8),
                                     static_cast<std::uint8_t>(length & 0xff)};
    seg.insert(seg.end(), body.begin(), body.end());
    return seg;
}

std::vector<std::uint8_t> jpeg_with_orientation(const Picture& p, int orientation) {
    std::vector<std::uint8_t> jpeg;
    stbi_write_jpg_to_func(
        [](void* ctx, void* data, int size) {
            auto* v = static_cast<std::vector<std::uint8_t>*>(ctx);
            const auto* b = static_cast<const std::uint8_t*>(data);
            v->insert(v->end(), b, b + size);
        },
        &jpeg, p.width, p.height, 3, p.rgb.data(), 95);
    const std::vector<std::uint8_t> app1 = app1_segment(orientation);
    jpeg.insert(jpeg.begin() + 2, app1.begin(), app1.end());
    return jpeg;
}

// PNG's CRC-32, so the eXIf chunk below is *valid* rather than merely
// present. `test_images.cpp` can leave the CRC zero because it only
// drives our own chunk walk, which does not check it; here libpng reads
// the same file, and a chunk it rejects would make this test pass for
// the wrong reason -- with the tag simply unread.
std::uint32_t crc32_of(const std::vector<std::uint8_t>& bytes) {
    static std::uint32_t table[256];
    static bool built = false;
    if (!built) {
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[n] = c;
        }
        built = true;
    }
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::uint8_t b : bytes) c = table[(c ^ b) & 0xff] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// A real PNG -- written by stb, so the pixels are exact -- with an eXIf
// chunk spliced in before IEND. Lossless, which is what lets the two
// loaders be compared byte for byte.
std::vector<std::uint8_t> png_with_orientation(const Picture& p, int orientation,
                                               const std::string& scratch) {
    const std::string tmp = scratch + "/oriented_source.png";
    images::save_png(p, tmp);
    std::ifstream in(tmp, std::ios::binary);
    std::vector<std::uint8_t> png((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
    std::error_code ec;
    std::filesystem::remove(tmp, ec);

    // The APP1 body minus its marker, length and `Exif\0\0` signature:
    // a PNG stores the bare TIFF stream.
    const std::vector<std::uint8_t> app1 = app1_segment(orientation);
    const std::vector<std::uint8_t> tiff(app1.begin() + 4 + 6, app1.end());

    std::vector<std::uint8_t> typed = {'e', 'X', 'I', 'f'};
    typed.insert(typed.end(), tiff.begin(), tiff.end());
    const std::uint32_t crc = crc32_of(typed);
    const std::size_t n = tiff.size();
    std::vector<std::uint8_t> chunk = {
        static_cast<std::uint8_t>((n >> 24) & 0xff),
        static_cast<std::uint8_t>((n >> 16) & 0xff),
        static_cast<std::uint8_t>((n >> 8) & 0xff),
        static_cast<std::uint8_t>(n & 0xff),
    };
    chunk.insert(chunk.end(), typed.begin(), typed.end());
    for (int shift : {24, 16, 8, 0}) {
        chunk.push_back(static_cast<std::uint8_t>((crc >> shift) & 0xff));
    }

    // Before IEND, which is the final 12 bytes of any PNG.
    //
    // Returned early rather than merely recorded: `check::is_true` notes a
    // failure and carries on, and `png.end() - 12` on a short vector is
    // undefined behaviour -- so a missing scratch file would have become a
    // segfault, which says less in a CI log than a named failure does.
    check::is_true(png.size() > 12, "images/qt: the source PNG was written");
    if (png.size() <= 12) return {};
    png.insert(png.end() - 12, chunk.begin(), chunk.end());
    return png;
}

// --- a format stb reads and Qt does not -------------------------------
//
// Radiance HDR, 4x3, uncompressed -- stb takes the flat scanline path
// below a width of 8, so this needs no RLE encoder. Qt has no HDR
// handler at any version, which is what makes it the right probe for the
// fallback: if this opens, the fallback ran.
std::vector<std::uint8_t> radiance_hdr(int width, int height) {
    const std::string header = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y " +
                               std::to_string(height) + " +X " +
                               std::to_string(width) + "\n";
    std::vector<std::uint8_t> out(header.begin(), header.end());
    for (int i = 0; i < width * height; ++i) {
        // RGBE with an exponent of 128 is a value in [0,1); the exact
        // subpixels are not the claim here, only that it decoded.
        out.push_back(static_cast<std::uint8_t>(200));
        out.push_back(static_cast<std::uint8_t>(150));
        out.push_back(static_cast<std::uint8_t>(100));
        out.push_back(128);
    }
    return out;
}

// --- the tests --------------------------------------------------------

void test_the_qt_transformation_table_matches_qt() {
    check::step("qt transformation table");
    // The one piece of arithmetic here with no oracle in `sstvae_core`:
    // the map from Qt's transformation flags back to an EXIF value, used
    // for the formats our own tag reader does not parse. Pinned against
    // the *installed* Qt by tagging a JPEG -- which both readers
    // understand -- with each of the eight values and requiring the
    // round trip. A future Qt that renumbered the flags would fail here
    // rather than in a sideways TIFF nobody photographs.
    const Picture src = ramp(24, 16);
    for (int o = 1; o <= 8; ++o) {
        std::vector<std::uint8_t> jpeg = jpeg_with_orientation(src, o);
        QByteArray raw(reinterpret_cast<const char*>(jpeg.data()),
                       static_cast<qsizetype>(jpeg.size()));
        QBuffer buffer(&raw);
        buffer.open(QIODevice::ReadOnly);
        QImageReader reader(&buffer);
        reader.setAutoTransform(false);
        const QImage image = reader.read();
        check::is_true(!image.isNull(),
                       "images/qt: Qt reads the tagged JPEG, orientation " +
                           std::to_string(o));
        check::equal(images::qt::exif_orientation_of(reader.transformation()), o,
                     "images/qt: Qt's transformation maps back to orientation " +
                         std::to_string(o));
    }
}

void test_a_tagged_png_matches_the_stb_loader_exactly(const std::string& tmpdir) {
    check::step("tagged png vs stb");
    // PNG is lossless and both loaders see the same eXIf chunk, so this
    // is an equality rather than a tolerance -- and it is the test that
    // catches the orientation being applied *twice*, which no
    // single-loader test can: a double rotation is a perfectly good
    // picture.
    const Picture src = ramp(37, 22);
    for (int o = 1; o <= 8; ++o) {
        const std::string path = tmpdir + "/tagged_" + std::to_string(o) + ".png";
        const std::vector<std::uint8_t> tagged = png_with_orientation(src, o, tmpdir);
        if (tagged.empty()) return;  // already reported above
        write_bytes(path, tagged);
        const std::string label = "orientation " + std::to_string(o);

        const Picture want = images::load(path);
        const Picture got = images::qt::load(path);
        check::equal(got.width, want.width, "images/qt: PNG width, " + label);
        check::equal(got.height, want.height, "images/qt: PNG height, " + label);
        check::is_true(got.rgb == want.rgb,
                       "images/qt: PNG pixels identical to the stb loader, " + label);

        // And that the transform actually happened, rather than both
        // loaders agreeing on doing nothing.
        const Picture expected = images::apply_orientation(src, o);
        check::is_true(got.rgb == expected.rgb,
                       "images/qt: PNG is upright per the tag, " + label);
    }
}

void test_a_tagged_jpeg_is_oriented(const std::string& tmpdir) {
    check::step("tagged jpeg");
    // JPEG is lossy, so this is geometry plus a margin: the intended
    // transform must beat every other one, which is what makes it a test
    // of the transform rather than of blurriness.
    const Picture src = ramp(64, 32);
    for (int o = 1; o <= 8; ++o) {
        const std::string path = tmpdir + "/tagged_" + std::to_string(o) + ".jpg";
        write_bytes(path, jpeg_with_orientation(src, o));
        const Picture got = images::qt::load(path);
        const Picture want = images::apply_orientation(src, o);
        const std::string label = "orientation " + std::to_string(o);
        check::equal(got.width, want.width, "images/qt: JPEG width, " + label);
        check::equal(got.height, want.height, "images/qt: JPEG height, " + label);

        const double err = mean_abs_diff(got, want);
        double best_wrong = 1e9;
        for (int other = 1; other <= 8; ++other) {
            if (other == o) continue;
            const Picture cand = images::apply_orientation(src, other);
            if (cand.width != got.width || cand.height != got.height) continue;
            best_wrong = std::min(best_wrong, mean_abs_diff(got, cand));
        }
        check::is_true(err < 8.0, "images/qt: JPEG matches the transform, " + label +
                                     " (mae " + std::to_string(err) + ")");
        check::is_true(err * 3.0 < best_wrong,
                       "images/qt: the intended transform beats every other, " + label);
    }
}

void test_a_format_qt_cannot_read_still_opens(const std::string& tmpdir) {
    check::step("stb fallback (hdr)");
    // The fallback. Radiance HDR: stb reads it, no Qt build has a
    // handler for it, and an app that gained Qt's formats must not have
    // lost stb's.
    const std::string path = tmpdir + "/fallback.hdr";
    write_bytes(path, radiance_hdr(4, 3));

    // The premise, checked rather than assumed: if some future Qt gains
    // an HDR handler this test stops proving anything, and it should say
    // so instead of passing.
    const bool qt_has_hdr =
        QImageReader::supportedImageFormats().contains(QByteArray("hdr"));
    check::is_true(!qt_has_hdr,
                   "images/qt: Qt still has no HDR handler, so this probes the "
                   "fallback");

    // Caught rather than allowed to escape: a missing fallback throws,
    // and an uncaught exception here would end the process with no
    // failure named -- which reads as a crash in the loader rather than
    // as the one thing this function is about.
    Picture got;
    std::string error;
    try {
        got = images::qt::load(path);
    } catch (const std::exception& e) {
        error = e.what();
    }
    check::is_true(error.empty(),
                   "images/qt: a format only stb reads still opens (error: " + error +
                       ")");
    if (!error.empty()) return;
    check::equal(got.width, 4, "images/qt: the fallback loader's width");
    check::equal(got.height, 3, "images/qt: the fallback loader's height");
    check::is_true(got.rgb == images::load(path).rgb,
                   "images/qt: the fallback is the stb loader, unchanged");
}

void test_formats_qt_adds_are_readable(const std::string& tmpdir) {
    check::step("formats qt adds");
    // The point of the whole layer: a format stb has no handler for
    // opens because Qt does. Which ones exist depends on the machine --
    // TIFF and WEBP arrive in the `qtimageformats` module -- so every
    // candidate Qt can both write and read is exercised and the count is
    // asserted, because a loop over an empty list is a test that passes
    // having checked nothing.
    struct Candidate {
        const char* format;
        bool lossless;
    };
    // XPM and ICO are in qtbase itself, so at least one of these is
    // always available and the assertion below is reachable everywhere.
    const Candidate candidates[] = {
        {"tiff", true}, {"webp", false}, {"xpm", true}, {"ico", true},
    };

    const Picture src = ramp(48, 36);
    const std::vector<std::string> readable = images::qt::readable_extensions();
    int exercised = 0;
    for (const Candidate& candidate : candidates) {
        const QByteArray format(candidate.format);
        if (!QImageWriter::supportedImageFormats().contains(format)) continue;
        if (!QImageReader::supportedImageFormats().contains(format)) continue;
        const std::string extension = candidate.format;
        // Per candidate, because "formats qt adds" covers four of them
        // and a crash inside one is otherwise indistinguishable.
        const std::string phase = "formats qt adds: " + extension;
        check::step(phase.c_str());
        check::is_true(std::find(readable.begin(), readable.end(), extension) !=
                           readable.end(),
                       "images/qt: readable_extensions offers " + extension);

        const std::string path = tmpdir + "/added." + extension;
        QImage out(src.width, src.height, QImage::Format_RGB888);
        for (int y = 0; y < src.height; ++y) {
            std::copy_n(src.rgb.data() + static_cast<std::size_t>(y) * src.width * 3,
                        static_cast<std::size_t>(src.width) * 3, out.scanLine(y));
        }
        if (!out.save(QString::fromStdString(path), candidate.format)) continue;

        // stb must be the one that cannot: if it can, this candidate is
        // not evidence about Qt adding anything.
        bool stb_read = true;
        try {
            images::load(path);
        } catch (const std::exception&) {
            stb_read = false;
        }
        if (stb_read) continue;

        const Picture got = images::qt::load(path);
        check::equal(got.width, src.width, "images/qt: " + extension + " width");
        check::equal(got.height, src.height, "images/qt: " + extension + " height");
        if (candidate.lossless) {
            check::is_true(got.rgb == src.rgb,
                           "images/qt: " + extension + " pixels are exact");
        } else {
            // No tolerance for a lossy format, because there is no
            // honest one: this ramp is nearly noise, which is the worst
            // case for WEBP at its default quality, and a threshold
            // loose enough to pass would also pass for a grey
            // rectangle. So the claim is comparative, as it is for the
            // JPEG orientations above -- the decoded picture must look
            // more like the source than like the source upside down.
            const double err = mean_abs_diff(got, src);
            const double wrong = mean_abs_diff(got, images::apply_orientation(src, 3));
            check::is_true(err < wrong * 0.5,
                           "images/qt: " + extension + " decoded the right picture "
                           "(mae " + std::to_string(err) + " against " +
                               std::to_string(wrong) + " for it inverted)");
        }
        ++exercised;
    }
    check::is_true(exercised > 0,
                   "images/qt: at least one Qt-only format was actually tested");
}

void test_the_extension_list_is_usable_as_a_filter() {
    check::step("extension list");
    const std::vector<std::string> extensions = images::qt::readable_extensions();
    check::is_true(!extensions.empty(), "images/qt: the extension list is not empty");
    check::is_true(std::is_sorted(extensions.begin(), extensions.end()),
                   "images/qt: the extension list is sorted");
    check::is_true(std::adjacent_find(extensions.begin(), extensions.end()) ==
                       extensions.end(),
                   "images/qt: the extension list has no duplicates");
    for (const std::string& extension : extensions) {
        check::is_true(!extension.empty(),
                       "images/qt: no empty extension in the list");
        check::is_true(extension.find('.') == std::string::npos,
                       "images/qt: extensions carry no dot: " + extension);
        check::is_true(std::none_of(extension.begin(), extension.end(),
                                    [](unsigned char c) { return std::isupper(c); }),
                       "images/qt: extensions are lowercase: " + extension);
    }
    // The formats behind the fallback, which no QImageReader would
    // report and which the dialog must still offer.
    for (const char* stb_only : {"psd", "hdr", "pic"}) {
        check::is_true(std::find(extensions.begin(), extensions.end(),
                                 std::string(stb_only)) != extensions.end(),
                       "images/qt: the list includes the stb-only " +
                           std::string(stb_only));
    }
}

void test_a_bundled_plugin_adds_its_format(const std::string& fixtures) {
    check::step("bundled heif plugin");
    // The bundled HEIF plugin, which is the point of building one at all:
    // a format neither Qt nor stb carries, reached through the same
    // `images::qt::load` with no code in this project referring to the
    // plugin.
    //
    // Not part of the "formats Qt adds" loop above, because that loop
    // needs a *writer* to make its probe file and this plugin registers
    // none -- writing HEIC needs a GPL encoder we deliberately do not
    // ship. Hence a committed fixture; see fixtures/README.md.
    const std::string path = fixtures + "/heif_ramp.heic";
    const bool heic_offered =
        QImageReader::supportedImageFormats().contains(QByteArray("heic"));
    if (!heic_offered) {
        // **The build system says whether this may skip.** A plugin that
        // failed to land on the search path looks exactly like
        // `-DSSTVAE_BUILD_HEIF=OFF` from in here, and the difference is
        // the whole question -- so CMake, which knows which it built,
        // sets SSTVAE_REQUIRE_HEIF and turns the skip into a failure.
        // Same hazard and same answer as SSTVAE_REQUIRE_CODEC: these are
        // the checks with a build-time prerequisite, which is exactly the
        // combination that rots into testing nothing.
        const char* required = std::getenv("SSTVAE_REQUIRE_HEIF");
        check::is_true(required == nullptr || std::string(required) != "1",
                       "images/qt: the HEIF plugin was built, so Qt must offer "
                       "heic -- it does not, so the plugin did not load (check "
                       "QT_PLUGIN_PATH and the plugin's own dependencies)");
        return;
    }

    const std::vector<std::string> readable = images::qt::readable_extensions();
    for (const char* extension : {"heic", "heif"}) {
        check::is_true(std::find(readable.begin(), readable.end(),
                                 std::string(extension)) != readable.end(),
                       "images/qt: readable_extensions offers " +
                           std::string(extension));
    }

    Picture got;
    std::string error;
    check::step("bundled heif plugin: decode");
    try {
        got = images::qt::load(path);
    } catch (const std::exception& e) {
        error = e.what();
    }
    check::step("bundled heif plugin: compare");
    check::is_true(error.empty(), "images/qt: the HEIC fixture opens (error: " +
                                      error + ")");
    if (!error.empty()) return;
    check::equal(got.width, 48, "images/qt: HEIC width");
    check::equal(got.height, 36, "images/qt: HEIC height");

    // Lossy, and through subsampled chroma, so comparative rather than
    // toleranced -- the same reasoning as the WEBP case above.
    const Picture src = ramp(48, 36);
    const double err = mean_abs_diff(got, src);
    const double wrong = mean_abs_diff(got, images::apply_orientation(src, 3));
    check::is_true(err < wrong * 0.5,
                   "images/qt: the HEIC fixture decoded to the right picture "
                   "(mae " + std::to_string(err) + " against " +
                       std::to_string(wrong) + " for it inverted)");

    // And stb must still be the one that cannot, or this proves nothing
    // about the plugin.
    bool stb_read = true;
    try {
        images::load(path);
    } catch (const std::exception&) {
        stb_read = false;
    }
    check::is_true(!stb_read, "images/qt: stb still cannot read HEIC");
}

void test_an_inset_keeps_its_alpha(const std::string& tmpdir) {
    check::step("inset alpha");
    // `load_qimage` exists so the overlay renderer gets a QImage rather
    // than a `Picture`, and the reason is in a comment there: going
    // through `Picture` to reuse `images::apply_orientation` would
    // flatten transparency, and a transparent PNG inset is an ordinary
    // thing to compose with. That is a claim about behaviour, so it is
    // checked here -- and checked on an *oriented* file, since the
    // orientation path is the one that would have done the flattening.
    const int w = 12, h = 8;
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(w) * h * 4);
    for (int i = 0; i < w * h; ++i) {
        rgba[i * 4 + 0] = static_cast<std::uint8_t>(i * 3 % 256);
        rgba[i * 4 + 1] = 40;
        rgba[i * 4 + 2] = 200;
        // Half transparent, half opaque, so a dropped channel is not
        // hidden by every pixel happening to be one or the other.
        rgba[i * 4 + 3] = static_cast<std::uint8_t>(i < w * h / 2 ? 0 : 255);
    }
    const std::string path = tmpdir + "/transparent.png";
    if (stbi_write_png(path.c_str(), w, h, 4, rgba.data(), w * 4) == 0) {
        check::is_true(false, "images/qt: the transparent PNG was written");
        return;
    }

    const QImage got = images::qt::load_qimage(path);
    check::is_true(!got.isNull(), "images/qt: load_qimage reads a transparent PNG");
    check::is_true(got.hasAlphaChannel(),
                   "images/qt: load_qimage keeps the alpha channel");
    check::equal(qAlpha(got.pixel(0, 0)), 0,
                 "images/qt: the transparent half is still transparent");
    check::equal(qAlpha(got.pixel(w - 1, h - 1)), 255,
                 "images/qt: the opaque half is still opaque");
}

void test_an_unreadable_file_reports_the_file(const std::string& tmpdir) {
    check::step("unreadable file");
    const std::string missing = tmpdir + "/does_not_exist.png";
    std::string message;
    try {
        images::qt::load(missing);
    } catch (const std::exception& e) {
        message = e.what();
    }
    check::is_true(message.find("does_not_exist.png") != std::string::npos,
                   "images/qt: a missing file's error names it (got: " + message + ")");
    check::is_true(images::qt::load_qimage(missing).isNull(),
                   "images/qt: load_qimage returns null rather than throwing");

    // Junk that is not a picture in any format: both decoders decline,
    // and the message must still name the file rather than being one
    // decoder's bare complaint.
    const std::string junk = tmpdir + "/junk.png";
    write_bytes(junk, std::vector<std::uint8_t>(4096, 0xAB));
    message.clear();
    try {
        images::qt::load(junk);
    } catch (const std::exception& e) {
        message = e.what();
    }
    check::is_true(message.find("junk.png") != std::string::npos,
                   "images/qt: junk reports the file name (got: " + message + ")");
}

void test_the_file_size_limit_still_applies(const std::string& tmpdir) {
    check::step("file size limit");
    // The guard lives in `read_picture_bytes`, which this layer shares
    // with the stb loader -- so it applies here too, and a second copy
    // of the limit was the thing worth avoiding. A *sparse* file, so it
    // costs no disk; if the platform declines, the check is skipped
    // rather than failed.
    const std::string path = tmpdir + "/enormous.png";
    {
        std::ofstream create(path, std::ios::binary);
        if (!create) return;
    }
    std::error_code ec;
    std::filesystem::resize_file(path, images::MAX_FILE_BYTES + 1, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        return;
    }
    std::string message;
    try {
        images::qt::load(path);
    } catch (const std::exception& e) {
        message = e.what();
    }
    std::filesystem::remove(path, ec);
    check::is_true(message.find("limit") != std::string::npos,
                   "images/qt: a file above MAX_FILE_BYTES is refused for its size "
                   "(got: " +
                       message + ")");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    // Qt resolves its image format plugins through the application's
    // library paths, so this needs an application object; offscreen so
    // it runs on a CI box with no display, and set before constructing
    // rather than in the environment of whatever launched us.
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QGuiApplication app(argc, argv);

    const std::string tmpdir = argc > 1 ? argv[1] : ".";
    const std::string fixtures = argc > 2 ? argv[2] : "fixtures";

    test_the_qt_transformation_table_matches_qt();
    test_a_tagged_png_matches_the_stb_loader_exactly(tmpdir);
    test_a_tagged_jpeg_is_oriented(tmpdir);
    test_a_format_qt_cannot_read_still_opens(tmpdir);
    test_formats_qt_adds_are_readable(tmpdir);
    test_the_extension_list_is_usable_as_a_filter();
    test_a_bundled_plugin_adds_its_format(fixtures);
    test_an_inset_keeps_its_alpha(tmpdir);
    test_an_unreadable_file_reports_the_file(tmpdir);
    test_the_file_size_limit_still_applies(tmpdir);

    return check::report("Qt image loading");
}
