// sstvae.wavio.read_wav over scipy.io.wavfile.read (wavread.hpp).

#include "qrss/wavread.hpp"

#include <algorithm>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dsp/dsp.hpp"
#include "qrss/config.hpp"
#include "qrss/npsum.hpp"

namespace sstvae::qrss::wav {

namespace {

using Kind = FormatError::Kind;
using Bytes = std::span<const std::uint8_t>;

constexpr std::uint64_t FORMAT_PCM = 0x0001;
constexpr std::uint64_t FORMAT_IEEE_FLOAT = 0x0003;
constexpr std::uint64_t FORMAT_EXTENSIBLE = 0xFFFE;

std::string errno_text(int err) {
    return std::string("[Errno ") + std::to_string(err) + "] " + std::strerror(err);
}

[[noreturn]] void fail(Kind k, const std::string& what) { throw FormatError(k, what); }

bool is(Bytes b, std::string_view id) {
    return b.size() == id.size() && std::memcmp(b.data(), id.data(), id.size()) == 0;
}

// repr(b) for a bytes object
std::string bytes_repr(Bytes b) {
    const bool sq = std::find(b.begin(), b.end(), '\'') != b.end();
    const bool dq = std::find(b.begin(), b.end(), '"') != b.end();
    const char q = (sq && !dq) ? '"' : '\'';
    std::string out = "b";
    out += q;
    static const char* hex = "0123456789abcdef";
    for (const std::uint8_t c : b) {
        if (c == static_cast<std::uint8_t>(q) || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c < 0x20 || c >= 0x7f) {
            out += "\\x";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else {
            out += static_cast<char>(c);
        }
    }
    out += q;
    return out;
}

// scipy.io.wavfile.WAVE_FORMAT: each value's name as WAVE_FORMAT(value).name
// gives it (scipy 1.16; tests/test_qrss_native_receiver_slot.py checks every
// one against scipy's enum)
struct FormatName {
    std::uint32_t tag;
    const char* name;
};
constexpr FormatName WAVE_FORMAT_NAMES[] = {
    {0x0000, "UNKNOWN"}, {0x0001, "PCM"}, {0x0002, "ADPCM"}, {0x0003, "IEEE_FLOAT"}, {0x0004, "VSELP"},
    {0x0005, "IBM_CVSD"}, {0x0006, "ALAW"}, {0x0007, "MULAW"}, {0x0008, "DTS"}, {0x0009, "DRM"}, {0x000A, "WMAVOICE9"},
    {0x000B, "WMAVOICE10"}, {0x0010, "OKI_ADPCM"}, {0x0011, "DVI_ADPCM"}, {0x0012, "MEDIASPACE_ADPCM"},
    {0x0013, "SIERRA_ADPCM"}, {0x0014, "G723_ADPCM"}, {0x0015, "DIGISTD"}, {0x0016, "DIGIFIX"},
    {0x0017, "DIALOGIC_OKI_ADPCM"}, {0x0018, "MEDIAVISION_ADPCM"}, {0x0019, "CU_CODEC"}, {0x001A, "HP_DYN_VOICE"},
    {0x0020, "YAMAHA_ADPCM"}, {0x0021, "SONARC"}, {0x0022, "DSPGROUP_TRUESPEECH"}, {0x0023, "ECHOSC1"},
    {0x0024, "AUDIOFILE_AF36"}, {0x0025, "APTX"}, {0x0026, "AUDIOFILE_AF10"}, {0x0027, "PROSODY_1612"},
    {0x0028, "LRC"}, {0x0030, "DOLBY_AC2"}, {0x0031, "GSM610"}, {0x0032, "MSNAUDIO"}, {0x0033, "ANTEX_ADPCME"},
    {0x0034, "CONTROL_RES_VQLPC"}, {0x0035, "DIGIREAL"}, {0x0036, "DIGIADPCM"}, {0x0037, "CONTROL_RES_CR10"},
    {0x0038, "NMS_VBXADPCM"}, {0x0039, "CS_IMAADPCM"}, {0x003A, "ECHOSC3"}, {0x003B, "ROCKWELL_ADPCM"},
    {0x003C, "ROCKWELL_DIGITALK"}, {0x003D, "XEBEC"}, {0x0040, "G721_ADPCM"}, {0x0041, "G728_CELP"},
    {0x0042, "MSG723"}, {0x0043, "INTEL_G723_1"}, {0x0044, "INTEL_G729"}, {0x0045, "SHARP_G726"}, {0x0050, "MPEG"},
    {0x0052, "RT24"}, {0x0053, "PAC"}, {0x0055, "MPEGLAYER3"}, {0x0059, "LUCENT_G723"}, {0x0060, "CIRRUS"},
    {0x0061, "ESPCM"}, {0x0062, "VOXWARE"}, {0x0063, "CANOPUS_ATRAC"}, {0x0064, "G726_ADPCM"}, {0x0065, "G722_ADPCM"},
    {0x0066, "DSAT"}, {0x0067, "DSAT_DISPLAY"}, {0x0069, "VOXWARE_BYTE_ALIGNED"}, {0x0070, "VOXWARE_AC8"},
    {0x0071, "VOXWARE_AC10"}, {0x0072, "VOXWARE_AC16"}, {0x0073, "VOXWARE_AC20"}, {0x0074, "VOXWARE_RT24"},
    {0x0075, "VOXWARE_RT29"}, {0x0076, "VOXWARE_RT29HW"}, {0x0077, "VOXWARE_VR12"}, {0x0078, "VOXWARE_VR18"},
    {0x0079, "VOXWARE_TQ40"}, {0x007A, "VOXWARE_SC3"}, {0x007B, "VOXWARE_SC3_1"}, {0x0080, "SOFTSOUND"},
    {0x0081, "VOXWARE_TQ60"}, {0x0082, "MSRT24"}, {0x0083, "G729A"}, {0x0084, "MVI_MVI2"}, {0x0085, "DF_G726"},
    {0x0086, "DF_GSM610"}, {0x0088, "ISIAUDIO"}, {0x0089, "ONLIVE"}, {0x008A, "MULTITUDE_FT_SX20"},
    {0x008B, "INFOCOM_ITS_G721_ADPCM"}, {0x008C, "CONVEDIA_G729"}, {0x008D, "CONGRUENCY"}, {0x0091, "SBC24"},
    {0x0092, "DOLBY_AC3_SPDIF"}, {0x0093, "MEDIASONIC_G723"}, {0x0094, "PROSODY_8KBPS"}, {0x0097, "ZYXEL_ADPCM"},
    {0x0098, "PHILIPS_LPCBB"}, {0x0099, "PACKED"}, {0x00A0, "MALDEN_PHONYTALK"}, {0x00A1, "RACAL_RECORDER_GSM"},
    {0x00A2, "RACAL_RECORDER_G720_A"}, {0x00A3, "RACAL_RECORDER_G723_1"}, {0x00A4, "RACAL_RECORDER_TETRA_ACELP"},
    {0x00B0, "NEC_AAC"}, {0x00FF, "RAW_AAC1"}, {0x0100, "RHETOREX_ADPCM"}, {0x0101, "IRAT"}, {0x0111, "VIVO_G723"},
    {0x0112, "VIVO_SIREN"}, {0x0120, "PHILIPS_CELP"}, {0x0121, "PHILIPS_GRUNDIG"}, {0x0123, "DIGITAL_G723"},
    {0x0125, "SANYO_LD_ADPCM"}, {0x0130, "SIPROLAB_ACEPLNET"}, {0x0131, "SIPROLAB_ACELP4800"},
    {0x0132, "SIPROLAB_ACELP8V3"}, {0x0133, "SIPROLAB_G729"}, {0x0134, "SIPROLAB_G729A"}, {0x0135, "SIPROLAB_KELVIN"},
    {0x0136, "VOICEAGE_AMR"}, {0x0140, "G726ADPCM"}, {0x0141, "DICTAPHONE_CELP68"}, {0x0142, "DICTAPHONE_CELP54"},
    {0x0150, "QUALCOMM_PUREVOICE"}, {0x0151, "QUALCOMM_HALFRATE"}, {0x0155, "TUBGSM"}, {0x0160, "MSAUDIO1"},
    {0x0161, "WMAUDIO2"}, {0x0162, "WMAUDIO3"}, {0x0163, "WMAUDIO_LOSSLESS"}, {0x0164, "WMASPDIF"},
    {0x0170, "UNISYS_NAP_ADPCM"}, {0x0171, "UNISYS_NAP_ULAW"}, {0x0172, "UNISYS_NAP_ALAW"}, {0x0173, "UNISYS_NAP_16K"},
    {0x0174, "SYCOM_ACM_SYC008"}, {0x0175, "SYCOM_ACM_SYC701_G726L"}, {0x0176, "SYCOM_ACM_SYC701_CELP54"},
    {0x0177, "SYCOM_ACM_SYC701_CELP68"}, {0x0178, "KNOWLEDGE_ADVENTURE_ADPCM"}, {0x0180, "FRAUNHOFER_IIS_MPEG2_AAC"},
    {0x0190, "DTS_DS"}, {0x0200, "CREATIVE_ADPCM"}, {0x0202, "CREATIVE_FASTSPEECH8"},
    {0x0203, "CREATIVE_FASTSPEECH10"}, {0x0210, "UHER_ADPCM"}, {0x0215, "ULEAD_DV_AUDIO"},
    {0x0216, "ULEAD_DV_AUDIO_1"}, {0x0220, "QUARTERDECK"}, {0x0230, "ILINK_VC"}, {0x0240, "RAW_SPORT"},
    {0x0241, "ESST_AC3"}, {0x0249, "GENERIC_PASSTHRU"}, {0x0250, "IPI_HSX"}, {0x0251, "IPI_RPELP"}, {0x0260, "CS2"},
    {0x0270, "SONY_SCX"}, {0x0271, "SONY_SCY"}, {0x0272, "SONY_ATRAC3"}, {0x0273, "SONY_SPC"}, {0x0280, "TELUM_AUDIO"},
    {0x0281, "TELUM_IA_AUDIO"}, {0x0285, "NORCOM_VOICE_SYSTEMS_ADPCM"}, {0x0300, "FM_TOWNS_SND"}, {0x0350, "MICRONAS"},
    {0x0351, "MICRONAS_CELP833"}, {0x0400, "BTV_DIGITAL"}, {0x0401, "INTEL_MUSIC_CODER"}, {0x0402, "INDEO_AUDIO"},
    {0x0450, "QDESIGN_MUSIC"}, {0x0500, "ON2_VP7_AUDIO"}, {0x0501, "ON2_VP6_AUDIO"}, {0x0680, "VME_VMPCM"},
    {0x0681, "TPC"}, {0x08AE, "LIGHTWAVE_LOSSLESS"}, {0x1000, "OLIGSM"}, {0x1001, "OLIADPCM"}, {0x1002, "OLICELP"},
    {0x1003, "OLISBC"}, {0x1004, "OLIOPR"}, {0x1100, "LH_CODEC"}, {0x1101, "LH_CODEC_CELP"}, {0x1102, "LH_CODEC_SBC8"},
    {0x1103, "LH_CODEC_SBC12"}, {0x1104, "LH_CODEC_SBC16"}, {0x1400, "NORRIS"}, {0x1401, "ISIAUDIO_2"},
    {0x1500, "SOUNDSPACE_MUSICOMPRESS"}, {0x1600, "MPEG_ADTS_AAC"}, {0x1601, "MPEG_RAW_AAC"}, {0x1602, "MPEG_LOAS"},
    {0x1608, "NOKIA_MPEG_ADTS_AAC"}, {0x1609, "NOKIA_MPEG_RAW_AAC"}, {0x160A, "VODAFONE_MPEG_ADTS_AAC"},
    {0x160B, "VODAFONE_MPEG_RAW_AAC"}, {0x1610, "MPEG_HEAAC"}, {0x181C, "VOXWARE_RT24_SPEECH"},
    {0x1971, "SONICFOUNDRY_LOSSLESS"}, {0x1979, "INNINGS_TELECOM_ADPCM"}, {0x1C07, "LUCENT_SX8300P"},
    {0x1C0C, "LUCENT_SX5363S"}, {0x1F03, "CUSEEME"}, {0x1FC4, "NTCSOFT_ALF2CM_ACM"}, {0x2000, "DVM"}, {0x2001, "DTS2"},
    {0x3313, "MAKEAVIS"}, {0x4143, "DIVIO_MPEG4_AAC"}, {0x4201, "NOKIA_ADAPTIVE_MULTIRATE"}, {0x4243, "DIVIO_G726"},
    {0x434C, "LEAD_SPEECH"}, {0x564C, "LEAD_VORBIS"}, {0x5756, "WAVPACK_AUDIO"}, {0x674F, "OGG_VORBIS_MODE_1"},
    {0x6750, "OGG_VORBIS_MODE_2"}, {0x6751, "OGG_VORBIS_MODE_3"}, {0x676F, "OGG_VORBIS_MODE_1_PLUS"},
    {0x6770, "OGG_VORBIS_MODE_2_PLUS"}, {0x6771, "OGG_VORBIS_MODE_3_PLUS"}, {0x6C61, "ALAC"}, {0x7000, "_3COM_NBX"},
    {0x704F, "OPUS"}, {0x706D, "FAAD_AAC"}, {0x7361, "AMR_NB"}, {0x7362, "AMR_WB"}, {0x7363, "AMR_WP"},
    {0x7A21, "GSM_AMR_CBR"}, {0x7A22, "GSM_AMR_VBR_SID"}, {0xA100, "COMVERSE_INFOSYS_G723_1"},
    {0xA101, "COMVERSE_INFOSYS_AVQSBC"}, {0xA102, "COMVERSE_INFOSYS_SBC"}, {0xA103, "SYMBOL_G729_A"},
    {0xA104, "VOICEAGE_AMR_WB"}, {0xA105, "INGENIENT_G726"}, {0xA106, "MPEG4_AAC"}, {0xA107, "ENCORE_G726"},
    {0xA108, "ZOLL_ASAO"}, {0xA109, "SPEEX_VOICE"}, {0xA10A, "VIANIX_MASC"}, {0xA10B, "WM9_SPECTRUM_ANALYZER"},
    {0xA10C, "WMF_SPECTRUM_ANAYZER"}, {0xA10D, "GSM_610"}, {0xA10E, "GSM_620"}, {0xA10F, "GSM_660"},
    {0xA110, "GSM_690"}, {0xA111, "GSM_ADAPTIVE_MULTIRATE_WB"}, {0xA112, "POLYCOM_G722"}, {0xA113, "POLYCOM_G728"},
    {0xA114, "POLYCOM_G729_A"}, {0xA115, "POLYCOM_SIREN"}, {0xA116, "GLOBAL_IP_ILBC"},
    {0xA117, "RADIOTIME_TIME_SHIFT_RADIO"}, {0xA118, "NICE_ACA"}, {0xA119, "NICE_ADPCM"}, {0xA11A, "VOCORD_G721"},
    {0xA11B, "VOCORD_G726"}, {0xA11C, "VOCORD_G722_1"}, {0xA11D, "VOCORD_G728"}, {0xA11E, "VOCORD_G729"},
    {0xA11F, "VOCORD_G729_A"}, {0xA120, "VOCORD_G723_1"}, {0xA121, "VOCORD_LBC"}, {0xA122, "NICE_G728"},
    {0xA123, "FRACE_TELECOM_G729"}, {0xA124, "CODIAN"}, {0xF1AC, "FLAC"}, {0xFFFE, "EXTENSIBLE"},
    {0xFFFF, "DEVELOPMENT"},
};

// _raise_bad_format
[[noreturn]] void raise_bad_format(std::uint64_t tag) {
    std::string name;
    const auto* it = std::lower_bound(std::begin(WAVE_FORMAT_NAMES), std::end(WAVE_FORMAT_NAMES), tag,
                                      [](const FormatName& f, std::uint64_t t) { return f.tag < t; });
    if (it != std::end(WAVE_FORMAT_NAMES) && it->tag == tag) {
        name = it->name;
    } else {
        // f'{format_tag:#06x}'
        static const char* hex = "0123456789abcdef";
        std::string digits;
        for (std::uint64_t v = tag; v; v >>= 4) digits.insert(digits.begin(), hex[v & 15]);
        while (digits.size() < 4) digits.insert(digits.begin(), '0');
        name = "0x" + digits;
    }
    fail(Kind::value, "Unknown wave file format: " + name + ". Supported formats: PCM, IEEE_FLOAT");
}

// The open file as scipy's `fid` behaves: read(n) stops at the end of the
// file, seek past it is allowed, tell() is wherever that left it.
struct Fid {
    const std::vector<std::uint8_t>& b;
    std::uint64_t pos = 0;

    Bytes read(std::uint64_t n) {
        if (pos >= b.size()) return {};
        const std::uint64_t k = std::min<std::uint64_t>(n, b.size() - pos);
        const Bytes s(b.data() + pos, static_cast<std::size_t>(k));
        pos += k;
        return s;
    }
    std::uint64_t avail() const { return pos < b.size() ? b.size() - pos : 0; }
    // _handle_pad_byte
    void pad(std::uint64_t size) {
        if (size % 2) pos += 1;
    }
};

// an unsigned integer of `s`, struct.unpack's: the buffer must be exactly
// `width` bytes
std::uint64_t unpack(Bytes s, std::size_t width, bool be) {
    if (s.size() != width) fail(Kind::struct_error, "unpack requires a buffer of " + std::to_string(width) + " bytes");
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < width; ++i) v = v << 8 | s[be ? i : width - 1 - i];
    return v;
}

std::uint64_t field(Bytes s, std::size_t at, std::size_t width, bool be) {
    return unpack(s.subspan(at, width), width, be);
}

// _skip_unknown_chunk
void skip_unknown(Fid& fid, bool be) {
    const Bytes d = fid.read(4);
    if (!d.empty()) {
        const std::uint64_t size = unpack(d, 4, be);
        fid.pos += size;
        fid.pad(size);
    }
}

// numpy's float16 -> float64 (npy_halfbits_to_doublebits)
double half_to_double(std::uint16_t h) {
    std::uint16_t h_exp = h & 0x7c00u;
    const std::uint64_t d_sgn = static_cast<std::uint64_t>(h & 0x8000u) << 48;
    std::uint64_t bits;
    if (h_exp == 0) {
        std::uint16_t h_sig = h & 0x03ffu;
        if (h_sig == 0) {
            bits = d_sgn;
        } else {
            h_sig = static_cast<std::uint16_t>(h_sig << 1);
            while ((h_sig & 0x0400u) == 0) {
                h_sig = static_cast<std::uint16_t>(h_sig << 1);
                ++h_exp;
            }
            const std::uint64_t d_exp = static_cast<std::uint64_t>(1023 - 15 - h_exp) << 52;
            bits = d_sgn + d_exp + (static_cast<std::uint64_t>(h_sig & 0x03ffu) << 42);
        }
    } else if (h_exp == 0x7c00u) {
        bits = d_sgn + 0x7ff0000000000000ull + (static_cast<std::uint64_t>(h & 0x03ffu) << 42);
    } else {
        bits = d_sgn + ((static_cast<std::uint64_t>(h & 0x7fffu) + 0xfc000u) << 42);
    }
    double d;
    std::memcpy(&d, &bits, 8);
    return d;
}

// numpy's sample types scipy picks, and what read_wav divides each by
enum class Dtype { u1, i1, i2, i4, i8, v3, v567, f2, f4, f8, f16 };

// numpy's '<f16' is the platform's long double when that is 16 bytes (x87
// extended on x86-64, IEEE quad on aarch64 Linux); elsewhere there is no
// such type and numpy refuses the name -- on Windows too, where numpy's
// long double is the compiler's 8-byte one even if ours (MinGW) is not
#if defined(_WIN32)
constexpr bool LONG_DOUBLE_16 = false;
#else
constexpr bool LONG_DOUBLE_16 = sizeof(long double) == 16 && std::numeric_limits<long double>::digits >= 64;
#endif

struct Format {
    std::uint64_t tag = 0;
    std::uint64_t channels = 0;
    std::uint64_t fs = 0;
    std::uint64_t block_align = 0;
    std::uint64_t bit_depth = 0;
};

// _read_fmt_chunk, the file pointer after the 'fmt ' id
Format read_fmt(Fid& fid, bool be) {
    const std::uint64_t size = unpack(fid.read(4), 4, be);
    if (size < 16) fail(Kind::value, "Binary structure of wave file is not compliant");
    const Bytes res = fid.read(16);
    if (res.size() != 16) fail(Kind::struct_error, "unpack requires a buffer of 16 bytes");
    Format f;
    f.tag = field(res, 0, 2, be);
    f.channels = field(res, 2, 2, be);
    f.fs = field(res, 4, 4, be);
    const std::uint64_t bytes_per_second = field(res, 8, 4, be);
    f.block_align = field(res, 12, 2, be);
    f.bit_depth = field(res, 14, 2, be);
    std::uint64_t bytes_read = 16;
    if (f.tag == FORMAT_EXTENSIBLE && size >= 16 + 2) {
        const std::uint64_t ext_chunk_size = unpack(fid.read(2), 2, be);
        bytes_read += 2;
        if (ext_chunk_size >= 22) {
            const Bytes ext = fid.read(22);
            bytes_read += 22;
            // extensible_chunk_data[2+4:2+4+16], clamped as a slice clamps
            const Bytes raw_guid = ext.size() > 6 ? ext.subspan(6) : Bytes{};
            static const std::uint8_t tail_le[12] = {0x00, 0x00, 0x10, 0x00, 0x80, 0x00,
                                                     0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
            static const std::uint8_t tail_be[12] = {0x00, 0x00, 0x00, 0x10, 0x80, 0x00,
                                                     0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
            const std::uint8_t* tail = be ? tail_be : tail_le;
            if (raw_guid.size() >= 12 && std::memcmp(raw_guid.data() + raw_guid.size() - 12, tail, 12) == 0)
                f.tag = unpack(raw_guid.first(4), 4, be);
        } else {
            fail(Kind::value, "Binary structure of wave file is not compliant");
        }
    }
    if (f.tag != FORMAT_PCM && f.tag != FORMAT_IEEE_FLOAT) raise_bad_format(f.tag);
    if (size > bytes_read) fid.read(size - bytes_read);
    fid.pad(size);
    if (f.tag == FORMAT_PCM && bytes_per_second != f.fs * f.block_align)
        fail(Kind::value, "WAV header is invalid: nAvgBytesPerSec must equal product of nSamplesPerSec and "
                          "nBlockAlign, but file has nSamplesPerSec = " +
                              std::to_string(f.fs) + ", nBlockAlign = " + std::to_string(f.block_align) +
                              ", and nAvgBytesPerSec = " + std::to_string(bytes_per_second));
    return f;
}

// A data chunk as read_wav holds it: the samples already divided by their
// dtype's maximum (or widened), and the channel count they were read with.
struct Data {
    std::vector<double> x;
    std::uint64_t channels = 1;
    // read_wav's cast to float64 raising FE_OVERFLOW / FE_INVALID (a
    // float32 signalling NaN; a long double past float64's range, or a
    // signalling NaN, or an encoding the x87 does not take)
    bool cast_overflow = false;
    bool cast_invalid = false;
};

// (double)v as numpy's cast does it, the flags it raises noted
double long_double_to_double(const std::uint8_t* p, bool be, Data& d) {
    std::uint8_t b[16];
    for (std::size_t k = 0; k < 16; ++k) b[k] = p[be ? 15 - k : k];
    long double v;
    std::memcpy(&v, b, sizeof v);
    std::feclearexcept(FE_OVERFLOW | FE_INVALID);
    volatile long double in = v;
    volatile double out = static_cast<double>(in);
    const int raised = std::fetestexcept(FE_OVERFLOW | FE_INVALID);
    if (raised & FE_OVERFLOW) d.cast_overflow = true;
    if (raised & FE_INVALID) d.cast_invalid = true;
    return out;
}

bool signalling(double v) {
    std::uint64_t b;
    std::memcpy(&b, &v, 8);
    return (b & 0x7ff0000000000000ull) == 0x7ff0000000000000ull && (b & 0x000fffffffffffffull) != 0 &&
           (b & 0x0008000000000000ull) == 0;
}

// numpy's floating-point flags over an add reduction (FE_OVERFLOW,
// FE_INVALID), as the hardware raises them
struct FpFlags {
    bool overflow = false;
    bool invalid = false;
};

double add(double a, double b, FpFlags& fp) {
    if (signalling(a) || signalling(b) || (std::isinf(a) && std::isinf(b) && std::signbit(a) != std::signbit(b)))
        fp.invalid = true;
    const double r = a + b;
    if (std::isfinite(a) && std::isfinite(b) && std::isinf(r)) fp.overflow = true;
    return r;
}

// npsum::pairwise_sum (numpy's DOUBLE_pairwise_sum, the same association),
// its flags kept
double pairwise_sum(const double* a, std::size_t n, FpFlags& fp) {
    if (n < 8) {
        double res = -0.0;
        for (std::size_t i = 0; i < n; ++i) res = add(res, a[i], fp);
        return res;
    }
    if (n <= npsum::PW_BLOCKSIZE) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (std::size_t k = 0; k < 8; ++k) r[k] = add(r[k], a[i + k], fp);
        double res = add(add(add(r[0], r[1], fp), add(r[2], r[3], fp), fp),
                         add(add(r[4], r[5], fp), add(r[6], r[7], fp), fp), fp);
        for (; i < n; ++i) res = add(res, a[i], fp);
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return add(pairwise_sum(a, n2, fp), pairwise_sum(a + n2, n - n2, fp), fp);
}

std::string dtype_text(bool be, char kind, std::uint64_t bytes) {
    return std::string("data type '") + (be ? '>' : '<') + kind + std::to_string(bytes) + "' not understood";
}

[[noreturn]] void cannot_reshape(std::uint64_t size, std::uint64_t cols) {
    fail(Kind::value,
         "cannot reshape array of size " + std::to_string(size) + " into shape (" + std::to_string(cols) + ")");
}

// _read_data_chunk (mmap=False), the file pointer after the 'data' id
Data read_data(Fid& fid, const Format& f, bool be, bool rf64, std::uint64_t rf64_chunk_size) {
    std::uint64_t size;
    if (!rf64) {
        size = unpack(fid.read(4), 4, be);
    } else {
        size = rf64_chunk_size;
        fid.read(4);
    }
    if (f.channels == 0) fail(Kind::zero_division, "integer division or modulo by zero");
    const std::uint64_t bytes_per_sample = f.block_align / f.channels;
    if (bytes_per_sample == 0) fail(Kind::zero_division, "integer division or modulo by zero");
    const std::uint64_t n_samples = size / bytes_per_sample;
    Dtype dt;
    std::uint64_t itemsize;
    if (f.tag == FORMAT_PCM) {
        if (1 <= f.bit_depth && f.bit_depth <= 8) {
            dt = Dtype::u1;   // read as n_samples single bytes whatever the container
            itemsize = 1;
        } else if (bytes_per_sample == 3) {
            dt = Dtype::v3;
            itemsize = 1;
        } else if (bytes_per_sample >= 5 && bytes_per_sample <= 7) {
            dt = Dtype::v567;
            itemsize = 1;
        } else if (f.bit_depth <= 64) {
            switch (bytes_per_sample) {
                case 1: dt = Dtype::i1; break;
                case 2: dt = Dtype::i2; break;
                case 4: dt = Dtype::i4; break;
                case 8: dt = Dtype::i8; break;
                default: fail(Kind::type, dtype_text(be, 'i', bytes_per_sample));
            }
            itemsize = bytes_per_sample;
        } else {
            fail(Kind::value, "Unsupported bit depth: the WAV file has " + std::to_string(f.bit_depth) +
                                  "-bit integer data.");
        }
    } else {   // IEEE_FLOAT: read_fmt refused every other tag
        if (f.bit_depth == 32 || f.bit_depth == 64) {
            switch (bytes_per_sample) {
                case 2: dt = Dtype::f2; break;
                case 4: dt = Dtype::f4; break;
                case 8: dt = Dtype::f8; break;
                case 16:
                    if (!LONG_DOUBLE_16) fail(Kind::type, dtype_text(be, 'f', bytes_per_sample));
                    dt = Dtype::f16;
                    break;
                default: fail(Kind::type, dtype_text(be, 'f', bytes_per_sample));
            }
            itemsize = bytes_per_sample;
        } else {
            fail(Kind::value, "Unsupported bit depth: the WAV file has " + std::to_string(f.bit_depth) +
                                  "-bit floating-point data.");
        }
    }
    // np.fromfile(fid, dtype, count): the whole elements there are, the
    // file left after every byte fread consumed (a partial element at the
    // end of the file included)
    const bool raw = dt == Dtype::v3 || dt == Dtype::v567;
    const std::uint64_t count = raw ? size : n_samples;
    const std::uint64_t start = fid.pos;
    const std::uint64_t avail = fid.avail();
    std::uint64_t elements, consumed;
    if (count <= avail / itemsize) {
        elements = count;
        consumed = count * itemsize;
    } else {
        elements = avail / itemsize;
        consumed = avail;
    }
    const std::uint8_t* p = fid.b.data() + std::min<std::uint64_t>(start, fid.b.size());
    fid.pos = start + consumed;
    Data d;
    // element i as an unsigned integer of `width` bytes in the file's order
    auto word = [&](std::uint64_t i, std::size_t width) {
        const std::uint8_t* q = p + i * width;
        std::uint64_t v = 0;
        for (std::size_t k = 0; k < width; ++k) v = v << 8 | q[be ? k : width - 1 - k];
        return v;
    };
    if (raw) {
        // bytes into the high end of an int32 / int64, as scipy places them
        if (elements % bytes_per_sample != 0) cannot_reshape(elements, bytes_per_sample);
        const std::uint64_t n = elements / bytes_per_sample;
        const unsigned shift = static_cast<unsigned>(8 * ((dt == Dtype::v3 ? 4 : 8) - bytes_per_sample));
        d.x.resize(static_cast<std::size_t>(n));
        for (std::uint64_t i = 0; i < n; ++i) {
            const std::uint64_t v = word(i, static_cast<std::size_t>(bytes_per_sample)) << shift;
            d.x[i] = dt == Dtype::v3 ? static_cast<double>(static_cast<std::int32_t>(static_cast<std::uint32_t>(v))) /
                                           2147483647.0
                                     : static_cast<double>(static_cast<std::int64_t>(v)) / 9223372036854775807.0;
        }
    } else {
        d.x.resize(static_cast<std::size_t>(elements));
        for (std::uint64_t i = 0; i < elements; ++i) {
            double v;
            switch (dt) {
                case Dtype::u1: v = static_cast<double>(p[i]) / 255.0; break;
                case Dtype::i1: v = static_cast<double>(static_cast<std::int8_t>(p[i])) / 127.0; break;
                case Dtype::i2:
                    v = static_cast<double>(static_cast<std::int16_t>(static_cast<std::uint16_t>(word(i, 2)))) /
                        32767.0;
                    break;
                case Dtype::i4:
                    v = static_cast<double>(static_cast<std::int32_t>(static_cast<std::uint32_t>(word(i, 4)))) /
                        2147483647.0;
                    break;
                case Dtype::i8:
                    v = static_cast<double>(static_cast<std::int64_t>(word(i, 8))) / 9223372036854775807.0;
                    break;
                case Dtype::f2: v = half_to_double(static_cast<std::uint16_t>(word(i, 2))); break;
                case Dtype::f4: {
                    const auto u = static_cast<std::uint32_t>(word(i, 4));
                    float s;
                    std::memcpy(&s, &u, 4);
                    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0 && (u & 0x00400000u) == 0)
                        d.cast_invalid = true;
                    v = static_cast<double>(s);
                    break;
                }
                case Dtype::f16: v = long_double_to_double(p + 16 * i, be, d); break;
                default: {   // f8
                    const std::uint64_t u = word(i, 8);
                    std::memcpy(&v, &u, 8);
                    break;
                }
            }
            d.x[i] = v;
        }
    }
    fid.pad(size);
    const std::uint64_t got = static_cast<std::uint64_t>(d.x.size());
    if (f.channels > 1 && got % f.channels != 0) cannot_reshape(got, f.channels);
    d.channels = f.channels;
    return d;
}

// "{file_size:d}" for file_size = size_field + 8, which RF64's 64-bit size
// field can carry past 2**64
std::string plus8(std::uint64_t size_field) {
    if (size_field <= UINT64_MAX - 8) return std::to_string(size_field + 8);
    return "184467440737095516" + std::to_string(16 + (size_field + 8));   // 2**64 + 0..7
}

}  // namespace

ReadError::ReadError(const std::string& p, int e)
    : std::runtime_error(errno_text(e) + ": '" + p + "'"), path(p), err(e) {}

FormatError::FormatError(Kind k, const std::string& what) : std::invalid_argument(what), kind(k) {}

std::vector<double> read_wav(const std::string& path, std::vector<Warning>* warnings) {
    if (path.find('\0') != std::string::npos) fail(Kind::value, "embedded null byte");
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> file(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (!file) throw ReadError(path, errno);
    std::vector<std::uint8_t> bytes;
    {
        std::vector<std::uint8_t> buf(std::size_t{1} << 20);
        std::size_t n;
        while ((n = std::fread(buf.data(), 1, buf.size(), file.get())) > 0)
            bytes.insert(bytes.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(n));
        if (std::ferror(file.get())) throw ReadError(path, errno ? errno : EIO);
    }
    auto warn = [&](std::string s, Warning::Kind k = Warning::Kind::wav_file) {
        if (warnings) warnings->push_back({k, std::move(s)});
    };

    // _read_riff_chunk
    Fid fid{bytes};
    const Bytes str1 = fid.read(4);
    bool be = false, rf64 = false;
    if (is(str1, "RIFF")) {
    } else if (is(str1, "RIFX")) {
        be = true;
    } else if (is(str1, "RF64")) {
        rf64 = true;
    } else {
        fail(Kind::value,
             "File format " + bytes_repr(str1) + " not understood. Only 'RIFF', 'RIFX', and 'RF64' supported.");
    }
    std::uint64_t size_field;   // file_size - 8
    std::uint64_t rf64_chunk_size = 0;
    Bytes str2;
    if (!rf64) {
        size_field = unpack(fid.read(4), 4, be);
        str2 = fid.read(4);
    } else {
        fid.read(4);
        str2 = fid.read(4);
        if (!is(fid.read(4), "ds64")) fail(Kind::value, "Invalid RF64 file: ds64 chunk not found.");
        const std::uint64_t ds64_size = unpack(fid.read(4), 4, false);
        size_field = unpack(fid.read(8), 8, false);
        rf64_chunk_size = unpack(fid.read(8), 8, false);
        fid.pos = fid.pos + ds64_size - 16;   // seek(ds64_size - 16, 1), from 36: never before the start
    }
    if (!is(str2, "WAVE")) fail(Kind::value, "Not a WAV file. RIFF form type is " + bytes_repr(str2) + ".");

    // read's chunk loop: while fid.tell() < file_size
    bool fmt_received = false, data_received = false;
    Format f;
    std::uint64_t fs = 0;
    Data data;
    while (fid.pos < 8 || fid.pos - 8 < size_field) {
        const Bytes id = fid.read(4);
        if (id.empty()) {
            if (data_received) {
                warn("Reached EOF prematurely; finished at " + std::to_string(fid.pos) + " bytes, expected " +
                     plus8(size_field) + " bytes from header.");
                break;
            }
            fail(Kind::value, "Unexpected end of file.");
        } else if (id.size() < 4) {
            const std::string msg = "Incomplete chunk ID: " + bytes_repr(id);
            if (fmt_received && data_received) warn(msg + ", ignoring it.");
            else fail(Kind::value, msg);
        }
        if (is(id, "fmt ")) {
            fmt_received = true;
            f = read_fmt(fid, be);
            fs = f.fs;
        } else if (is(id, "fact") || is(id, "LIST") || is(id, "JUNK") || is(id, "Fake")) {
            skip_unknown(fid, be);
        } else if (is(id, "data")) {
            data_received = true;
            if (!fmt_received) fail(Kind::value, "No fmt chunk before data");
            data = read_data(fid, f, be, rf64, rf64_chunk_size);
        } else {
            warn("Chunk (non-data) not understood, skipping it.");
            skip_unknown(fid, be);
        }
    }
    // return fs, data
    if (!fmt_received)
        fail(Kind::unbound_local, "cannot access local variable 'fs' where it is not associated with a value");
    if (!data_received)
        fail(Kind::unbound_local, "cannot access local variable 'data' where it is not associated with a value");

    // read_wav: (already scaled) cast to float64, channels averaged, then
    // the rate
    if (data.cast_overflow) warn("overflow encountered in cast", Warning::Kind::runtime);
    if (data.cast_invalid) warn("invalid value encountered in cast", Warning::Kind::runtime);
    std::vector<double> x = std::move(data.x);
    if (data.channels > 1) {
        const auto ch = static_cast<std::size_t>(data.channels);
        const std::size_t frames = x.size() / ch;
        std::vector<double> mono(frames);
        // data.mean(axis=1) on the C-contiguous (frames, channels) array:
        // each row's pairwise sum onto 0, then / channels (a division
        // raises no flag numpy warns of: no operand is a signalling NaN)
        FpFlags fp;
        for (std::size_t i = 0; i < frames; ++i)
            mono[i] = add(0.0, pairwise_sum(x.data() + i * ch, ch, fp), fp) / static_cast<double>(ch);
        if (fp.overflow) warn("overflow encountered in reduce", Warning::Kind::runtime);
        if (fp.invalid) warn("invalid value encountered in reduce", Warning::Kind::runtime);
        x = std::move(mono);
    }
    if (fs != static_cast<std::uint64_t>(config::FS)) {
        const auto rate = static_cast<std::int64_t>(fs);
        const std::int64_t g = std::gcd(static_cast<std::int64_t>(config::FS), rate);
        const std::int64_t up = config::FS / g, down = rate / g;
        if (up < 1 || down < 1) fail(Kind::value, "up and down must be >= 1");
        // resample_poly's 2 * 10 * max(up, down) + 1 taps in an int
        if (std::max(up, down) > (INT_MAX - 1) / 20) throw std::bad_alloc();
        x = dsp::resample_poly(std::span<const double>(x), static_cast<int>(up), static_cast<int>(down));
    }
    return x;
}

}  // namespace sstvae::qrss::wav
