#include "images/images.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef SSTVAE_HAVE_LIBHEIF
#include <libheif/heif.h>
#endif
#ifdef SSTVAE_HAVE_LIBWEBP
#include <webp/decode.h>
#endif
#ifdef SSTVAE_HAVE_LIBOPENJP2
#include <openjpeg.h>
#endif

// Vendored; read for the orientation tag alone. See
// native/third_party/easyexif/README.md.
#include "easyexif/exif.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

namespace sstvae::images {
namespace {

// Whole file into memory. Both consumers -- the stb decoder and the
// EXIF parser -- want the bytes rather than a path, and a picture is
// already being held decoded at several times this size.
std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<std::uint8_t> bytes;
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) throw std::runtime_error("cannot read " + path);
    // Refused before the allocation, not after -- the point is to not
    // reserve a gigabyte on the strength of a file name. The message
    // names the size, because the usual cause is a file that is not a
    // picture at all and the operator will recognise it by its length.
    if (static_cast<std::uintmax_t>(size) > MAX_FILE_BYTES) {
        throw std::runtime_error(path + " is " + std::to_string(size) +
                                 " bytes; the limit for a picture file is " +
                                 std::to_string(MAX_FILE_BYTES));
    }
    in.seekg(0, std::ios::beg);
    bytes.resize(static_cast<std::size_t>(size));
    if (size > 0) {
        in.read(reinterpret_cast<char*>(bytes.data()), size);
        if (!in) throw std::runtime_error("cannot read " + path);
    }
    return bytes;
}

}  // namespace

Picture resize(const Picture& img, int width, int height) {
    if (img.width == width && img.height == height) return img;
    if (img.width <= 0 || img.height <= 0) {
        throw std::runtime_error("cannot resize an empty picture");
    }
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("resize target must be positive");
    }
    Picture out(width, height);
    if (stbir_resize_uint8_srgb(img.rgb.data(), img.width, img.height, 0, out.rgb.data(),
                                width, height, 0, STBIR_RGB) == nullptr) {
        throw std::runtime_error("image resize failed");
    }
    return out;
}

Picture fit(const Picture& img) { return fit(img, Framing{}); }

double min_zoom(int width, int height) {
    if (width <= 0 || height <= 0) return 1.0;
    const double src = static_cast<double>(width) / height;
    const double target = static_cast<double>(IMG_W) / IMG_H;
    // contain / cover, which is the ratio of the two aspects whichever
    // way round they are.
    return std::min(src, target) / std::max(src, target);
}

Picture fit(const Picture& img, const Framing& framing) {
    // The identity short-circuit only applies to the default framing:
    // an operator who has zoomed or panned an already-4:3 picture means
    // it, and returning the original would silently ignore them.
    const bool defaulted = framing.zoom == 1.0 && framing.center_x == 0.5 &&
                           framing.center_y == 0.5;
    if (defaulted && img.width == IMG_W && img.height == IMG_H) return img;
    if (img.width <= 0 || img.height <= 0) {
        throw std::runtime_error("cannot fit an empty picture");
    }

    // Scale relative to *cover* the target, then crop -- the same shape
    // of operation as images.py, though not the same filter. Below
    // `min_zoom` there is nothing left to reveal, only more black.
    const double zoom =
        std::max(min_zoom(img.width, img.height), framing.zoom);
    const double scale = std::max(static_cast<double>(IMG_W) / img.width,
                                  static_cast<double>(IMG_H) / img.height) *
                         zoom;
    // At zoom 1 and above the scaled picture covers the target, and the
    // floor of IMG_W/IMG_H is the old guard against rounding a pixel
    // short of it. Below 1 an axis is *meant* to come up short, so the
    // guard would be exactly the bug -- it would silently cancel the
    // padding the operator asked for.
    const int floor_w = zoom >= 1.0 ? IMG_W : 1;
    const int floor_h = zoom >= 1.0 ? IMG_H : 1;
    const int sw = std::max(floor_w, static_cast<int>(std::lround(img.width * scale)));
    const int sh = std::max(floor_h, static_cast<int>(std::lround(img.height * scale)));

    const Picture scaled = resize(img, sw, sh);

    // `floor`, not `lround`: for the default centre this has to reduce
    // to the old `(sw - IMG_W) / 2` integer division exactly, and those
    // agree only under truncation of a non-negative value. With an odd
    // `sw` they differ by one pixel, so adding framing would have
    // silently shifted every odd-intermediate picture -- 1920x1080
    // scales to 853 wide, so that is most photographs. `test_framing`
    // pins this against the old formula written out; the Python parity
    // suite cannot, since it deliberately skips `fit_image` wherever
    // resampling is involved (PIL LANCZOS vs stb).
    //
    // The two clamp bounds are `0` and `scaled - target` in whichever
    // order they come: when the scaled picture is the larger the window
    // must stay inside it (the original rule), and when it is the
    // smaller the same expression keeps the *picture* inside the
    // window, so the padding never lands all on one side with the
    // photograph half off the canvas.
    const auto offset = [](double center, int scaled_size, int target) {
        const double raw = center * scaled_size - target / 2.0;
        const int slack = scaled_size - target;
        return std::clamp(static_cast<int>(std::floor(raw)), std::min(0, slack),
                          std::max(0, slack));
    };
    const int left = offset(framing.center_x, sw, IMG_W);
    const int top = offset(framing.center_y, sh, IMG_H);

    // Zero-initialized, which is the black the uncovered part is padded
    // with -- so the copy below only has to skip what it cannot fill.
    Picture out(IMG_W, IMG_H);
    const int x0 = std::max(0, -left);            // first covered column
    const int x1 = std::min(IMG_W, sw - left);    // one past the last
    for (int y = 0; y < IMG_H; ++y) {
        const int sy = y + top;
        if (sy < 0 || sy >= sh || x1 <= x0) continue;
        const std::uint8_t* src =
            scaled.rgb.data() + (static_cast<std::size_t>(sy) * sw + (x0 + left)) * 3;
        std::copy(src, src + static_cast<std::size_t>(x1 - x0) * 3,
                  out.rgb.begin() + (static_cast<std::size_t>(y) * IMG_W + x0) * 3);
    }
    return out;
}

ImageArray to_array(const Picture& img) {
    if (img.width != IMG_W || img.height != IMG_H) {
        throw std::runtime_error("to_array wants a fitted " + std::to_string(IMG_W) +
                                 "x" + std::to_string(IMG_H) + " picture");
    }
    ImageArray out;
    out.width = img.width;
    out.height = img.height;
    out.chw.resize(static_cast<std::size_t>(3) * IMG_H * IMG_W);

    // (H, W, 3) interleaved -> (3, H, W) planar, /255.
    //
    // The divide is exact in the sense that matters: 255 is a power-of-
    // two-free integer, so v/255.0f is correctly rounded from an exact
    // integer numerator and matches numpy's float32 divide bit for bit.
    const std::size_t plane = static_cast<std::size_t>(IMG_H) * IMG_W;
    for (std::size_t i = 0; i < plane; ++i) {
        for (int c = 0; c < 3; ++c) {
            out.chw[c * plane + i] = static_cast<float>(img.rgb[i * 3 + c]) / 255.0f;
        }
    }
    return out;
}

namespace {

// PNG stores EXIF in an `eXIf` chunk holding a bare TIFF stream -- no
// `Exif\0\0` signature, which is what a JPEG APP1 segment carries and
// what easyexif's segment entry point expects. So the chunk is located
// here and the signature prepended, leaving the TIFF parsing (byte
// order, IFD offsets, bounds) where it belongs: in the vendored code.
//
// Walking PNG chunks is not the same kind of undertaking as walking a
// TIFF: every chunk is a 4-byte big-endian length followed by a 4-byte
// type, so the whole walk is bounds-checkable by inspection and there
// are no internal offsets that can point anywhere.
//
// Pillow honours this chunk, so not reading it would mean a tagged PNG
// rotating in `sstvae/images.py` and not here -- exactly the silent,
// consequential disagreement the parity test exists to prevent.
std::vector<std::uint8_t> png_exif_segment(const std::uint8_t* data, std::size_t len) {
    static constexpr std::uint8_t SIGNATURE[8] = {0x89, 0x50, 0x4E, 0x47,
                                                  0x0D, 0x0A, 0x1A, 0x0A};
    if (len < sizeof(SIGNATURE) || !std::equal(SIGNATURE, SIGNATURE + sizeof(SIGNATURE), data)) {
        return {};
    }
    std::size_t pos = sizeof(SIGNATURE);
    while (pos + 8 <= len) {
        const std::size_t size = (static_cast<std::size_t>(data[pos]) << 24) |
                                 (static_cast<std::size_t>(data[pos + 1]) << 16) |
                                 (static_cast<std::size_t>(data[pos + 2]) << 8) |
                                 static_cast<std::size_t>(data[pos + 3]);
        const std::uint8_t* type = data + pos + 4;
        const std::size_t body = pos + 8;
        // The 4-byte CRC follows the body. Overflow is not reachable --
        // `size` is at most 2^32-1 and `body` at most `len` -- but the
        // comparison is written so it would not matter if it were.
        if (size > len || body > len - size || body + size + 4 > len) break;
        if (std::equal(type, type + 4, "eXIf")) {
            std::vector<std::uint8_t> out = {'E', 'x', 'i', 'f', 0x00, 0x00};
            out.insert(out.end(), data + body, data + body + size);
            return out;
        }
        pos = body + size + 4;
    }
    return {};
}

}  // namespace

int exif_orientation(const std::uint8_t* data, std::size_t len) {
    if (data == nullptr || len == 0) return 1;
    easyexif::EXIFInfo info;
    // Anything either parser objects to -- not a JPEG or PNG, no EXIF, a
    // truncated or corrupt segment -- is an unrotated picture, not a
    // failure. The return code is deliberately not distinguished: there
    // is nothing a caller could do differently with "no EXIF" than with
    // "corrupt".
    const std::vector<std::uint8_t> png = png_exif_segment(data, len);
    const int status =
        png.empty()
            ? info.parseFrom(data, static_cast<unsigned>(len))
            : info.parseFromEXIFSegment(png.data(), static_cast<unsigned>(png.size()));
    if (status != PARSE_EXIF_SUCCESS) return 1;
    const int value = static_cast<int>(info.Orientation);
    // 0 is easyexif's "tag absent"; anything above 8 is undefined in the
    // spec and has been seen in the wild from buggy writers.
    return (value >= 1 && value <= 8) ? value : 1;
}

Picture apply_orientation(const Picture& img, int orientation) {
    if (orientation <= 1 || orientation > 8 || img.empty()) return img;

    // Each case is the *inverse* map: where in the source does the
    // destination pixel come from. Written out rather than composed from
    // flip/transpose flags because the composition order is exactly the
    // thing that is easy to get backwards, and eight explicit lines can
    // be read against the spec's table one at a time.
    //
    // 5..8 exchange the axes, so the output's geometry changes -- which
    // is the whole point, and why this must run before `fit` decides
    // what to crop.
    const int sw = img.width, sh = img.height;
    const bool swap_axes = orientation >= 5;
    Picture out(swap_axes ? sh : sw, swap_axes ? sw : sh);

    for (int dy = 0; dy < out.height; ++dy) {
        for (int dx = 0; dx < out.width; ++dx) {
            int sx = 0, sy = 0;
            switch (orientation) {
                case 2: sx = sw - 1 - dx; sy = dy;             break;  // mirror
                case 3: sx = sw - 1 - dx; sy = sh - 1 - dy;    break;  // 180
                case 4: sx = dx;          sy = sh - 1 - dy;    break;  // flip
                case 5: sx = dy;          sy = dx;             break;  // transpose
                case 6: sx = dy;          sy = sh - 1 - dx;    break;  // 90 CW
                case 7: sx = sw - 1 - dy; sy = sh - 1 - dx;    break;  // transverse
                case 8: sx = sw - 1 - dy; sy = dx;             break;  // 90 CCW
                default: sx = dx;         sy = dy;             break;
            }
            const std::size_t src = (static_cast<std::size_t>(sy) * sw + sx) * 3;
            const std::size_t dst = (static_cast<std::size_t>(dy) * out.width + dx) * 3;
            out.rgb[dst] = img.rgb[src];
            out.rgb[dst + 1] = img.rgb[src + 1];
            out.rgb[dst + 2] = img.rgb[src + 2];
        }
    }
    return out;
}

bool heif_supported() {
#ifdef SSTVAE_HAVE_LIBHEIF
    return true;
#else
    return false;
#endif
}

bool looks_like_heif(const std::uint8_t* data, std::size_t len) {
    if (len < 12 || std::memcmp(data + 4, "ftyp", 4) != 0) return false;
    static const char* const brands[] = {"heic", "heix", "hevc", "hevx", "heim", "heis",
                                         "mif1", "msf1", "avif", "avis", "mif2"};
    for (const char* b : brands) {
        if (std::memcmp(data + 8, b, 4) == 0) return true;
    }
    return false;
}

bool webp_supported() {
#ifdef SSTVAE_HAVE_LIBWEBP
    return true;
#else
    return false;
#endif
}

bool jp2_supported() {
#ifdef SSTVAE_HAVE_LIBOPENJP2
    return true;
#else
    return false;
#endif
}

bool looks_like_webp(const std::uint8_t* data, std::size_t len) {
    return len >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WEBP", 4) == 0;
}

bool looks_like_jp2(const std::uint8_t* data, std::size_t len) {
    static const std::uint8_t jp2[] = {0, 0, 0, 0x0C, 'j', 'P', ' ', ' ', 0x0D, 0x0A, 0x87, 0x0A};
    static const std::uint8_t j2k[] = {0xFF, 0x4F, 0xFF, 0x51};
    return (len >= sizeof jp2 && std::memcmp(data, jp2, sizeof jp2) == 0) ||
           (len >= sizeof j2k && std::memcmp(data, j2k, sizeof j2k) == 0);
}

namespace {

[[noreturn]] void missing_library(const std::string& path, const std::string& what,
                                  const std::string& packages) {
    throw std::runtime_error("cannot read " + path + ": " + what + " pictures need " +
                             packages + ". Install it, then rebuild the app.");
}

#ifdef SSTVAE_HAVE_LIBWEBP
Picture load_webp(const std::vector<std::uint8_t>& bytes, const std::string& path) {
    int w = 0, h = 0;
    std::uint8_t* data = WebPDecodeRGB(bytes.data(), bytes.size(), &w, &h);
    if (data == nullptr || w <= 0 || h <= 0) {
        if (data) WebPFree(data);
        throw std::runtime_error("cannot read " + path + ": not a readable WebP picture");
    }
    Picture out(w, h);
    std::copy(data, data + static_cast<std::size_t>(w) * h * 3, out.rgb.begin());
    WebPFree(data);
    return out;
}
#endif

#ifdef SSTVAE_HAVE_LIBOPENJP2
struct MemoryStream {
    const std::vector<std::uint8_t>* bytes;
    std::size_t pos = 0;
};

OPJ_SIZE_T mem_read(void* buf, OPJ_SIZE_T n, void* user) {
    auto* m = static_cast<MemoryStream*>(user);
    const std::size_t left = m->bytes->size() - m->pos;
    if (left == 0) return static_cast<OPJ_SIZE_T>(-1);
    const std::size_t k = std::min<std::size_t>(n, left);
    std::memcpy(buf, m->bytes->data() + m->pos, k);
    m->pos += k;
    return k;
}

OPJ_OFF_T mem_skip(OPJ_OFF_T n, void* user) {
    auto* m = static_cast<MemoryStream*>(user);
    const auto size = static_cast<OPJ_OFF_T>(m->bytes->size());
    OPJ_OFF_T to = static_cast<OPJ_OFF_T>(m->pos) + n;
    to = std::clamp<OPJ_OFF_T>(to, 0, size);
    const OPJ_OFF_T moved = to - static_cast<OPJ_OFF_T>(m->pos);
    m->pos = static_cast<std::size_t>(to);
    return moved;
}

OPJ_BOOL mem_seek(OPJ_OFF_T to, void* user) {
    auto* m = static_cast<MemoryStream*>(user);
    if (to < 0 || static_cast<std::size_t>(to) > m->bytes->size()) return OPJ_FALSE;
    m->pos = static_cast<std::size_t>(to);
    return OPJ_TRUE;
}

// One component sample at picture position (x, y), scaled to 8 bits.
// Components may be subsampled (dx, dy), as chroma often is.
int jp2_sample(const opj_image_comp_t& c, int x, int y) {
    const int cx = std::min<int>(x / static_cast<int>(std::max(1u, c.dx)), static_cast<int>(c.w) - 1);
    const int cy = std::min<int>(y / static_cast<int>(std::max(1u, c.dy)), static_cast<int>(c.h) - 1);
    long v = c.data[static_cast<std::size_t>(cy) * c.w + cx];
    if (c.sgnd) v += 1L << (c.prec - 1);
    if (c.prec > 8) v >>= (c.prec - 8);
    else if (c.prec < 8) v = v * 255 / ((1L << c.prec) - 1);
    return static_cast<int>(std::clamp<long>(v, 0, 255));
}

std::uint8_t clamp8(double v) {
    return static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L));
}

Picture load_jp2(const std::vector<std::uint8_t>& bytes, const std::string& path) {
    const bool boxed = bytes.size() >= 4 && bytes[0] == 0 && bytes[3] == 0x0C;
    struct Codec {
        opj_codec_t* p;
        ~Codec() { if (p) opj_destroy_codec(p); }
    } codec{opj_create_decompress(boxed ? OPJ_CODEC_JP2 : OPJ_CODEC_J2K)};
    struct Stream {
        opj_stream_t* p = opj_stream_create(OPJ_J2K_STREAM_CHUNK_SIZE, OPJ_TRUE);
        ~Stream() { if (p) opj_stream_destroy(p); }
    } stream;
    struct Image {
        opj_image_t* p = nullptr;
        ~Image() { if (p) opj_image_destroy(p); }
    } image;
    auto fail = [&] {
        throw std::runtime_error("cannot read " + path + ": not a readable JPEG 2000 picture");
    };
    if (codec.p == nullptr || stream.p == nullptr) fail();

    MemoryStream mem{&bytes};
    opj_stream_set_user_data(stream.p, &mem, nullptr);
    opj_stream_set_user_data_length(stream.p, bytes.size());
    opj_stream_set_read_function(stream.p, mem_read);
    opj_stream_set_skip_function(stream.p, mem_skip);
    opj_stream_set_seek_function(stream.p, mem_seek);

    opj_dparameters_t params;
    opj_set_default_decoder_parameters(&params);
    if (!opj_setup_decoder(codec.p, &params) ||
        !opj_read_header(stream.p, codec.p, &image.p) ||
        !opj_decode(codec.p, stream.p, image.p) || !opj_end_decompress(codec.p, stream.p)) {
        fail();
    }
    const opj_image_t& img = *image.p;
    if (img.numcomps < 1 || img.comps[0].data == nullptr) fail();
    for (OPJ_UINT32 i = 0; i < img.numcomps && i < 3; ++i) {
        if (img.comps[i].data == nullptr || img.comps[i].w == 0 || img.comps[i].h == 0) fail();
    }

    // The picture's size is the first component's at full resolution.
    const int w = static_cast<int>(img.comps[0].w * std::max(1u, img.comps[0].dx));
    const int h = static_cast<int>(img.comps[0].h * std::max(1u, img.comps[0].dy));
    if (w <= 0 || h <= 0) fail();
    Picture out(w, h);
    const bool colour = img.numcomps >= 3;
    const bool ycc = colour && (img.color_space == OPJ_CLRSPC_SYCC ||
                                img.color_space == OPJ_CLRSPC_EYCC);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::uint8_t* px = &out.rgb[(static_cast<std::size_t>(y) * w + x) * 3];
            const int a = jp2_sample(img.comps[0], x, y);
            if (!colour) {
                px[0] = px[1] = px[2] = static_cast<std::uint8_t>(a);
                continue;
            }
            const int b = jp2_sample(img.comps[1], x, y);
            const int c = jp2_sample(img.comps[2], x, y);
            if (ycc) {  // BT.601 full range, as opj_decompress converts it
                px[0] = clamp8(a + 1.402 * (c - 128));
                px[1] = clamp8(a - 0.344136 * (b - 128) - 0.714136 * (c - 128));
                px[2] = clamp8(a + 1.772 * (b - 128));
            } else {
                px[0] = static_cast<std::uint8_t>(a);
                px[1] = static_cast<std::uint8_t>(b);
                px[2] = static_cast<std::uint8_t>(c);
            }
        }
    }
    return out;
}
#endif

#ifdef SSTVAE_HAVE_LIBHEIF
// The primary image, decoded to interleaved 8-bit RGB. libheif applies
// the container's rotation and mirroring (irot/imir) while decoding, so
// the result needs no EXIF orientation step.
Picture load_heif(const std::vector<std::uint8_t>& bytes, const std::string& path) {
    static std::once_flag init;
    std::call_once(init, [] { heif_init(nullptr); });  // loads decoder plugins

    auto fail = [&](const heif_error& err) {
        throw std::runtime_error("cannot read " + path + ": " +
                                 (err.message ? err.message : "libheif error"));
    };
    struct Ctx {
        heif_context* p = heif_context_alloc();
        ~Ctx() { heif_context_free(p); }
    } ctx;
    heif_error err = heif_context_read_from_memory_without_copy(ctx.p, bytes.data(),
                                                                bytes.size(), nullptr);
    if (err.code != heif_error_Ok) fail(err);

    struct Handle {
        heif_image_handle* p = nullptr;
        ~Handle() { if (p) heif_image_handle_release(p); }
    } handle;
    err = heif_context_get_primary_image_handle(ctx.p, &handle.p);
    if (err.code != heif_error_Ok) fail(err);

    struct Image {
        heif_image* p = nullptr;
        ~Image() { if (p) heif_image_release(p); }
    } img;
    err = heif_decode_image(handle.p, &img.p, heif_colorspace_RGB, heif_chroma_interleaved_RGB,
                            nullptr);
    if (err.code != heif_error_Ok) fail(err);

    const int w = heif_image_get_width(img.p, heif_channel_interleaved);
    const int h = heif_image_get_height(img.p, heif_channel_interleaved);
    int stride = 0;
    const std::uint8_t* plane = heif_image_get_plane_readonly(img.p, heif_channel_interleaved,
                                                              &stride);
    if (plane == nullptr || w <= 0 || h <= 0) {
        throw std::runtime_error("cannot read " + path + ": no picture in the HEIF file");
    }
    Picture out(w, h);
    const std::size_t row = static_cast<std::size_t>(w) * 3;
    for (int y = 0; y < h; ++y) {
        std::copy(plane + static_cast<std::size_t>(y) * stride,
                  plane + static_cast<std::size_t>(y) * stride + row,
                  out.rgb.begin() + static_cast<std::ptrdiff_t>(y * row));
    }
    return out;
}
#endif

}  // namespace

Picture load(const std::string& path) {
    // Read the file once and decode from memory, because the orientation
    // tag and the pixels come from the same bytes: stb ignores EXIF, and
    // easyexif wants the whole file. `stbi_load(path, ...)` would mean
    // opening it twice.
    std::vector<std::uint8_t> bytes = read_file(path);

    if (looks_like_heif(bytes.data(), bytes.size())) {
#ifdef SSTVAE_HAVE_LIBHEIF
        return load_heif(bytes, path);
#else
        missing_library(path, "HEIC/HEIF", "libheif (libheif-dev and libheif-plugin-libde265)");
#endif
    }
    if (looks_like_webp(bytes.data(), bytes.size())) {
#ifdef SSTVAE_HAVE_LIBWEBP
        return apply_orientation(load_webp(bytes, path), exif_orientation(bytes.data(), bytes.size()));
#else
        missing_library(path, "WebP", "libwebp (libwebp-dev)");
#endif
    }
    if (looks_like_jp2(bytes.data(), bytes.size())) {
#ifdef SSTVAE_HAVE_LIBOPENJP2
        return load_jp2(bytes, path);
#else
        missing_library(path, "JPEG 2000", "openjpeg (libopenjp2-7-dev)");
#endif
    }

    int w = 0, h = 0, channels = 0;
    // Forced to 3 channels: an RGBA or greyscale source is converted the
    // way `Image.convert("RGB")` would.
    std::uint8_t* data = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                                               &w, &h, &channels, 3);
    if (data == nullptr) {
        throw std::runtime_error("cannot read " + path + ": " + stbi_failure_reason());
    }
    Picture out(w, h);
    std::copy(data, data + static_cast<std::size_t>(w) * h * 3, out.rgb.begin());
    stbi_image_free(data);
    return apply_orientation(out, exif_orientation(bytes.data(), bytes.size()));
}

ImageArray load_array(const std::string& path) { return to_array(fit(load(path))); }

void save_png(const Picture& img, const std::string& path) {
    if (stbi_write_png(path.c_str(), img.width, img.height, 3, img.rgb.data(),
                       img.width * 3) == 0) {
        throw std::runtime_error("cannot write " + path);
    }
}

}  // namespace sstvae::images
