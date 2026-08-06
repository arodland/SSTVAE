// EXIF orientation: the transform, the tag reader, and the load path
// that joins them.
//
// The transform is eight cases of index arithmetic, and the way it goes
// wrong is not subtle in its effect (a sideways picture) but is very
// easy to get backwards in the source: `dst(x,y) = src(y,x)` and its
// seven relatives all look equally plausible on the page. So the oracle
// here is a *second derivation* -- each orientation expressed as a
// composition of three primitives (transpose, mirror-x, mirror-y) taken
// straight from the EXIF table -- rather than a restatement of the same
// inverse maps, which would only prove the file was copied correctly.
//
// The tag reader is vendored (third_party/easyexif), so what is checked
// here is not its parsing but *our* handling of what it returns: the
// defaults on a file it rejects, and the range clamp. Those are the
// lines that decide whether a corrupt photo loads sideways or not at
// all.

#include "images/images.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "check.hpp"
#include "images/types.hpp"
#include "stb_image_write.h"

namespace check = sstvae::check;
using sstvae::images::Picture;

namespace {

// Every pixel encodes its own position, and no two of the eight
// transforms of it are equal -- which is what makes a wrong case
// visible rather than absorbed by symmetry. Deliberately non-square for
// the same reason: 5..8 must exchange the axes.
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

// --- the independent derivation ---------------------------------------
//
// The EXIF table says which corner of the *stored* picture holds the
// visual top-left, and each row of it is naturally read as a sequence of
// these three primitives. Composing them is a different mental operation
// from writing an inverse map, which is the point.

Picture transpose(const Picture& p) {
    Picture out(p.height, p.width);
    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < out.width; ++x) {
            const std::size_t d = (static_cast<std::size_t>(y) * out.width + x) * 3;
            const std::size_t s = (static_cast<std::size_t>(x) * p.width + y) * 3;
            for (int c = 0; c < 3; ++c) out.rgb[d + c] = p.rgb[s + c];
        }
    }
    return out;
}

Picture mirror_x(const Picture& p) {
    Picture out(p.width, p.height);
    for (int y = 0; y < p.height; ++y) {
        for (int x = 0; x < p.width; ++x) {
            const std::size_t d = (static_cast<std::size_t>(y) * p.width + x) * 3;
            const std::size_t s =
                (static_cast<std::size_t>(y) * p.width + (p.width - 1 - x)) * 3;
            for (int c = 0; c < 3; ++c) out.rgb[d + c] = p.rgb[s + c];
        }
    }
    return out;
}

Picture mirror_y(const Picture& p) {
    Picture out(p.width, p.height);
    for (int y = 0; y < p.height; ++y) {
        const std::size_t d = static_cast<std::size_t>(y) * p.width * 3;
        const std::size_t s = static_cast<std::size_t>(p.height - 1 - y) * p.width * 3;
        std::copy(p.rgb.begin() + s, p.rgb.begin() + s + p.width * 3, out.rgb.begin() + d);
    }
    return out;
}

// Transpose first where it appears; the two mirrors commute, so their
// relative order is not a choice this has to make.
Picture reference_orientation(const Picture& p, int orientation) {
    switch (orientation) {
        case 1: return p;                                   // top-left
        case 2: return mirror_x(p);                         // top-right
        case 3: return mirror_y(mirror_x(p));               // bottom-right
        case 4: return mirror_y(p);                         // bottom-left
        case 5: return transpose(p);                        // left-top
        case 6: return mirror_x(transpose(p));              // right-top
        case 7: return mirror_y(mirror_x(transpose(p)));    // right-bottom
        case 8: return mirror_y(transpose(p));              // left-bottom
        default: return p;
    }
}

void test_orientation_matches_the_exif_table() {
    const Picture src = ramp(37, 23);  // odd both ways: no accidental symmetry
    for (int o = 1; o <= 8; ++o) {
        const Picture got = sstvae::images::apply_orientation(src, o);
        const Picture want = reference_orientation(src, o);
        const std::string label = "orientation " + std::to_string(o);
        check::equal(got.width, want.width, "images/exif: width, " + label);
        check::equal(got.height, want.height, "images/exif: height, " + label);
        check::is_true(got.rgb == want.rgb, "images/exif: pixels, " + label);
    }
}

void test_the_eight_transforms_are_distinguishable() {
    // Without this the test above could pass on a picture symmetric
    // enough that several cases coincide -- and then a swapped pair of
    // cases would be invisible. Asserting the sample's fitness for
    // purpose, not the code.
    const Picture src = ramp(37, 23);
    for (int a = 1; a <= 8; ++a) {
        for (int b = a + 1; b <= 8; ++b) {
            const Picture pa = sstvae::images::apply_orientation(src, a);
            const Picture pb = sstvae::images::apply_orientation(src, b);
            check::is_true(pa.width != pb.width || pa.height != pb.height || pa.rgb != pb.rgb,
                           "images/exif: transforms " + std::to_string(a) + " and " +
                               std::to_string(b) + " differ on the test picture");
        }
    }
}

void test_axes_swap_only_above_four() {
    const Picture src = ramp(37, 23);
    for (int o = 1; o <= 8; ++o) {
        const Picture got = sstvae::images::apply_orientation(src, o);
        const bool swapped = got.width == src.height && got.height == src.width;
        check::equal(swapped, o >= 5,
                     "images/exif: axes exchanged iff orientation >= 5, at " +
                         std::to_string(o));
    }
}

void test_out_of_range_is_the_identity() {
    const Picture src = ramp(9, 5);
    // 0 is easyexif's "absent"; 9 and above are undefined in the spec.
    // Both must leave the picture alone rather than fall through to some
    // case's arithmetic.
    for (int o : {-3, 0, 1, 9, 65535}) {
        const Picture got = sstvae::images::apply_orientation(src, o);
        check::is_true(got.width == src.width && got.height == src.height &&
                           got.rgb == src.rgb,
                       "images/exif: orientation " + std::to_string(o) + " is the identity");
    }
}

// --- the tag reader ----------------------------------------------------

// A minimal but real APP1 segment: TIFF header, one IFD entry holding
// tag 0x0112, no next IFD. Big-endian ("MM"), so the byte order path is
// the non-native one on every machine this is built for.
std::vector<std::uint8_t> app1_segment(int orientation) {
    std::vector<std::uint8_t> tiff = {
        'M',  'M',  0x00, 0x2a,              // byte order, magic
        0x00, 0x00, 0x00, 0x08,              // offset of IFD0
        0x00, 0x01,                          // one entry
        0x01, 0x12,                          // tag: Orientation
        0x00, 0x03,                          // type: SHORT
        0x00, 0x00, 0x00, 0x01,              // count
        static_cast<std::uint8_t>(orientation >> 8),
        static_cast<std::uint8_t>(orientation & 0xff),
        0x00, 0x00,                          // value, left-aligned in 4 bytes
        0x00, 0x00, 0x00, 0x00,              // no next IFD
    };
    std::vector<std::uint8_t> body = {'E', 'x', 'i', 'f', 0x00, 0x00};
    body.insert(body.end(), tiff.begin(), tiff.end());

    const std::size_t length = body.size() + 2;  // the length field counts itself
    std::vector<std::uint8_t> seg = {0xFF, 0xE1,
                                     static_cast<std::uint8_t>(length >> 8),
                                     static_cast<std::uint8_t>(length & 0xff)};
    seg.insert(seg.end(), body.begin(), body.end());
    return seg;
}

std::vector<std::uint8_t> jpeg_bytes(const Picture& p) {
    std::vector<std::uint8_t> out;
    stbi_write_jpg_to_func(
        [](void* ctx, void* data, int size) {
            auto* v = static_cast<std::vector<std::uint8_t>*>(ctx);
            const auto* b = static_cast<const std::uint8_t*>(data);
            v->insert(v->end(), b, b + size);
        },
        &out, p.width, p.height, 3, p.rgb.data(), 95);
    return out;
}

// A JPEG with the segment spliced in immediately after SOI, which is
// where a camera puts it.
std::vector<std::uint8_t> jpeg_with_orientation(const Picture& p, int orientation) {
    std::vector<std::uint8_t> jpeg = jpeg_bytes(p);
    const std::vector<std::uint8_t> app1 = app1_segment(orientation);
    jpeg.insert(jpeg.begin() + 2, app1.begin(), app1.end());
    return jpeg;
}

void write_bytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
}

void test_tag_is_read_from_a_jpeg() {
    const Picture src = ramp(32, 16);
    for (int o = 1; o <= 8; ++o) {
        const std::vector<std::uint8_t> jpeg = jpeg_with_orientation(src, o);
        check::equal(sstvae::images::exif_orientation(jpeg.data(), jpeg.size()), o,
                     "images/exif: tag read back, orientation " + std::to_string(o));
    }
}

void test_unreadable_metadata_defaults_to_upright() {
    const Picture src = ramp(32, 16);
    const std::vector<std::uint8_t> plain = jpeg_bytes(src);
    const std::vector<std::uint8_t> tagged = jpeg_with_orientation(src, 6);

    // No EXIF at all.
    check::equal(sstvae::images::exif_orientation(plain.data(), plain.size()), 1,
                 "images/exif: a JPEG with no APP1 reads as upright");
    // Nothing at all.
    check::equal(sstvae::images::exif_orientation(nullptr, 0), 1,
                 "images/exif: an empty buffer reads as upright");
    // Not a JPEG.
    const std::vector<std::uint8_t> junk(64, 0xAB);
    check::equal(sstvae::images::exif_orientation(junk.data(), junk.size()), 1,
                 "images/exif: junk reads as upright");
    // An APP1 that claims more bytes than are present -- the shape a
    // truncated download has, and the one that must not read past the
    // end. Cut inside the segment, after the length field.
    for (std::size_t cut : {tagged.size() - 4, tagged.size() / 2, std::size_t{8}}) {
        const std::vector<std::uint8_t> truncated(tagged.begin(), tagged.begin() + cut);
        const int got = sstvae::images::exif_orientation(truncated.data(), truncated.size());
        check::is_true(got >= 1 && got <= 8,
                       "images/exif: a truncated file yields a legal orientation");
    }
    // A value outside the spec's range, which buggy writers do emit.
    for (int bad : {0, 9, 240}) {
        const std::vector<std::uint8_t> odd = jpeg_with_orientation(src, bad);
        check::equal(sstvae::images::exif_orientation(odd.data(), odd.size()), 1,
                     "images/exif: out-of-range tag " + std::to_string(bad) +
                         " reads as upright");
    }
}

// A PNG carries EXIF in an `eXIf` chunk holding a bare TIFF stream --
// no `Exif\0\0` signature. Nothing here validates CRCs, so a synthetic
// buffer exercises the chunk walk without needing a PNG encoder or a
// CRC32: the walk is what is under test, and it is reached the same way
// by a real file.
std::vector<std::uint8_t> png_chunk(const std::string& type,
                                    const std::vector<std::uint8_t>& body) {
    const std::size_t n = body.size();
    std::vector<std::uint8_t> out = {
        static_cast<std::uint8_t>((n >> 24) & 0xff), static_cast<std::uint8_t>((n >> 16) & 0xff),
        static_cast<std::uint8_t>((n >> 8) & 0xff),  static_cast<std::uint8_t>(n & 0xff),
    };
    out.insert(out.end(), type.begin(), type.end());
    out.insert(out.end(), body.begin(), body.end());
    out.insert(out.end(), 4, 0x00);  // CRC, unchecked
    return out;
}

std::vector<std::uint8_t> png_with_orientation(int orientation) {
    std::vector<std::uint8_t> png = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    // A chunk before it, so a walk that only ever looks at the first one
    // fails here rather than passing by luck.
    const std::vector<std::uint8_t> ihdr(13, 0x01);
    const std::vector<std::uint8_t> header = png_chunk("IHDR", ihdr);
    png.insert(png.end(), header.begin(), header.end());

    // The APP1 body without its `Exif\0\0` prefix -- which is exactly
    // what a PNG stores and what the code under test has to re-add.
    const std::vector<std::uint8_t> app1 = app1_segment(orientation);
    const std::vector<std::uint8_t> tiff(app1.begin() + 4 + 6, app1.end());
    const std::vector<std::uint8_t> exif = png_chunk("eXIf", tiff);
    png.insert(png.end(), exif.begin(), exif.end());
    return png;
}

void test_png_exif_chunk_is_read() {
    // Pillow reads this chunk, so not reading it would mean a tagged PNG
    // rotating on one side of the port and not the other.
    for (int o = 1; o <= 8; ++o) {
        const std::vector<std::uint8_t> png = png_with_orientation(o);
        check::equal(sstvae::images::exif_orientation(png.data(), png.size()), o,
                     "images/exif: PNG eXIf chunk, orientation " + std::to_string(o));
    }

    // A chunk whose declared length runs past the end -- the walk must
    // stop rather than read past it.
    std::vector<std::uint8_t> truncated = png_with_orientation(6);
    truncated.resize(truncated.size() - 6);
    const int got = sstvae::images::exif_orientation(truncated.data(), truncated.size());
    check::equal(got, 1, "images/exif: a truncated eXIf chunk reads as upright");

    // A PNG with no eXIf chunk at all.
    std::vector<std::uint8_t> plain = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    const std::vector<std::uint8_t> ihdr = png_chunk("IHDR", std::vector<std::uint8_t>(13, 0x01));
    plain.insert(plain.end(), ihdr.begin(), ihdr.end());
    check::equal(sstvae::images::exif_orientation(plain.data(), plain.size()), 1,
                 "images/exif: a PNG with no eXIf chunk reads as upright");
}

// --- the two joined ----------------------------------------------------

void test_load_applies_the_tag(const std::string& tmpdir) {
    // JPEG is lossy, so this checks geometry and gross structure rather
    // than pixels -- the exact-transform claim is the tests above. What
    // it does prove is that `load` reads the tag and acts on it at all,
    // which no unit test of either half can.
    const Picture src = ramp(64, 32);
    for (int o = 1; o <= 8; ++o) {
        const std::string path = tmpdir + "/exif_" + std::to_string(o) + ".jpg";
        write_bytes(path, jpeg_with_orientation(src, o));
        const Picture got = sstvae::images::load(path);
        const Picture want = reference_orientation(src, o);
        const std::string label = "orientation " + std::to_string(o);
        check::equal(got.width, want.width, "images/exif: loaded width, " + label);
        check::equal(got.height, want.height, "images/exif: loaded height, " + label);

        // Mean absolute error against the intended transform, versus the
        // best of the other seven. A tolerance alone could be met by a
        // picture that is merely blurry; requiring the right transform to
        // win by a margin is what makes this a test of the transform.
        double best_wrong = 1e9;
        for (int other = 1; other <= 8; ++other) {
            if (other == o) continue;
            const Picture cand = reference_orientation(src, other);
            if (cand.width != got.width || cand.height != got.height) continue;
            double err = 0.0;
            for (std::size_t i = 0; i < got.rgb.size(); ++i) {
                err += std::abs(static_cast<double>(got.rgb[i]) - cand.rgb[i]);
            }
            best_wrong = std::min(best_wrong, err / got.rgb.size());
        }
        double err = 0.0;
        for (std::size_t i = 0; i < got.rgb.size(); ++i) {
            err += std::abs(static_cast<double>(got.rgb[i]) - want.rgb[i]);
        }
        err /= got.rgb.size();
        check::is_true(err < 8.0, "images/exif: loaded picture matches the transform, " +
                                      label + " (mae " + std::to_string(err) + ")");
        check::is_true(err * 3.0 < best_wrong,
                       "images/exif: the intended transform beats every other, " + label +
                           " (mae " + std::to_string(err) + " vs " +
                           std::to_string(best_wrong) + ")");
    }
}

void test_an_enormous_file_is_refused(const std::string& tmpdir) {
    // A *sparse* file: `resize_file` sets the length without writing the
    // bytes on every filesystem this is built for (ext4, APFS, NTFS), so
    // this costs no disk and no time. If the platform declines, the
    // check is skipped rather than failed -- an unwritable temp
    // directory is not evidence about the loader.
    const std::string path = tmpdir + "/enormous.jpg";
    {
        std::ofstream create(path, std::ios::binary);
        if (!create) return;
    }
    std::error_code ec;
    std::filesystem::resize_file(path, sstvae::images::MAX_FILE_BYTES + 1, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        return;
    }

    const auto load_error = [&](const std::string& p) {
        try {
            sstvae::images::load(p);
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string();
    };

    // Refused *for its size*, which is the claim. A file this large is
    // not a valid JPEG either, so asserting only that something was
    // thrown would pass with the limit deleted -- the decoder would
    // reject it a moment later and a gigabyte further on.
    const std::string over = load_error(path);
    check::is_true(over.find("limit") != std::string::npos,
                   "images/limit: a file above MAX_FILE_BYTES is refused for its size "
                   "(got: " +
                       over + ")");

    // The boundary is inclusive: at exactly the limit the file reaches
    // the decoder and fails there instead, with stb's message. Resized
    // before the removal below -- deleting it first is how the first
    // draft of this silently skipped everything after this point.
    std::filesystem::resize_file(path, sstvae::images::MAX_FILE_BYTES, ec);
    const std::string message = ec ? std::string() : load_error(path);
    std::filesystem::remove(path, ec);
    if (message.empty()) return;
    check::is_true(message.find("limit") == std::string::npos,
                   "images/limit: a file at exactly the limit is not refused for size "
                   "(got: " +
                       message + ")");
}

void test_a_png_is_untouched(const std::string& tmpdir) {
    // No EXIF path exists for PNG here, and the orientation reader must
    // not corrupt the load of one. Lossless, so this is exact.
    const Picture src = ramp(19, 41);
    const std::string path = tmpdir + "/plain.png";
    sstvae::images::save_png(src, path);
    const Picture got = sstvae::images::load(path);
    check::equal(got.width, src.width, "images/exif: PNG width unchanged");
    check::equal(got.height, src.height, "images/exif: PNG height unchanged");
    check::is_true(got.rgb == src.rgb, "images/exif: PNG pixels unchanged");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    const std::string tmpdir = argc > 1 ? argv[1] : ".";

    test_orientation_matches_the_exif_table();
    test_the_eight_transforms_are_distinguishable();
    test_axes_swap_only_above_four();
    test_out_of_range_is_the_identity();
    test_tag_is_read_from_a_jpeg();
    test_unreadable_metadata_defaults_to_upright();
    test_png_exif_chunk_is_read();
    test_load_applies_the_tag(tmpdir);
    test_an_enormous_file_is_refused(tmpdir);
    test_a_png_is_untouched(tmpdir);

    return check::report("images");
}
