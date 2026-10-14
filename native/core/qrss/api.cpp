// QRSSTVAE transmit in process (api.hpp): qrss_transmit.py's flow after
// its arguments are read, over tx::plan_pass and tx::PassAudio.

#include "qrss/api.hpp"

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "images/images.hpp"
#include "qrss/beaconfile.hpp"
#include "qrss/header.hpp"
#include "qrss/pytext.hpp"
#include "qrss/slot.hpp"
#include "util/fileio.hpp"

namespace sstvae::qrss::api {

std::optional<int> mode_index(std::string_view name) {
    for (std::size_t i = 0; i < config::MODE_NAMES.size(); ++i)
        if (config::MODE_NAMES[i] == name) return static_cast<int>(i);
    return std::nullopt;
}

int passes_for(std::string_view mode) {
    const std::optional<int> m = mode_index(mode);
    return m ? *m + 1 : 0;
}

std::optional<Waveform> waveform_named(std::string_view name) {
    if (name == "CE") return Waveform::CE;
    if (name == "L") return Waveform::L;
    return std::nullopt;
}

std::string_view waveform_name(Waveform w) { return w == Waveform::L ? "L" : "CE"; }

const frame::FrameSpec& frame_spec(Waveform w, std::string_view name) {
    bool known = false;
    for (std::string_view f : FRAMES) known = known || f == name;
    if (!known)
        throw std::invalid_argument("frame must be one of full, medium, short, tiny, not '" + std::string(name) +
                                    "'");
    return w == Waveform::L ? frame::get_l("l-" + std::string(name)) : frame::get(name);
}

std::int64_t parse_slot(std::string_view text) { return slot::parse_slot(text); }

std::string slot_iso(std::int64_t q) { return slot::slot_iso(q); }

std::int64_t slot_of(double quarter_hour_unix_s) {
    const double t = quarter_hour_unix_s;
    // Exactly a multiple of 900 s, and a q that int64 holds.
    if (!std::isfinite(t) || std::fmod(t, SLOT_S) != 0.0 || std::fabs(t / SLOT_S) > 9.0e15)
        throw std::invalid_argument("QRSS passes start on a quarter hour");
    return static_cast<std::int64_t>(t / SLOT_S);
}

std::int64_t pass_samples(Waveform w, std::string_view name) {
    return tx::slot_samples(frame_spec(w, name), AUDIO_LEAD_S, AUDIO_TAIL_S);
}

double audio_end_s(std::int64_t q, Waveform w, std::string_view name) {
    return audio_start_s(q) + static_cast<double>(pass_samples(w, name)) / SAMPLE_RATE;
}

picture::StoredPicture encode_picture(const images::Picture& pic, std::int64_t mode, std::int64_t codec_id,
                                      const Encoder& encode) {
    // The refusals of the arguments come before the (slow) encode.
    picture::check_mode(mode);
    picture::check_codec_id(codec_id);
    if (!encode) throw std::invalid_argument("encode_picture: no encoder");
    // A picture whose pixels are not width x height x 3 would be read out
    // of bounds by the resize.
    if (pic.width <= 0 || pic.height <= 0)
        throw std::invalid_argument("encode_picture: an empty picture (" + std::to_string(pic.width) + "x" +
                                    std::to_string(pic.height) + ")");
    const std::uint64_t bytes = static_cast<std::uint64_t>(pic.width) * static_cast<std::uint64_t>(pic.height) * 3;
    if (pic.rgb.size() != bytes)
        throw std::invalid_argument("encode_picture: a " + std::to_string(pic.width) + "x" +
                                    std::to_string(pic.height) + " picture needs " + std::to_string(bytes) +
                                    " bytes of RGB, not " + std::to_string(pic.rgb.size()));
    return codec::encode_picture(images::to_array(images::fit(pic)), mode, codec_id, encode);
}

ReadError::ReadError(const std::string& p, int e)
    : std::runtime_error("[Errno " + std::to_string(e) + "] " + std::strerror(e) + ": " + pytext::repr(p)),
      path(p),
      err(e) {}

Source read_source(const std::string& path) {
    // open(path, "rb").read(), with its OSError's errno: a stream would say
    // nothing useful (libstdc++ throws "basic_filebuf::underflow error
    // reading the file" for a directory).
    errno = 0;
    const util::File f = util::open_file(path, "rb");
    if (!f) throw ReadError(path, errno ? errno : EIO);
    std::vector<std::uint8_t> data;
    std::vector<std::uint8_t> buf(std::size_t(1) << 16);
    for (;;) {
        errno = 0;
        const std::size_t n = std::fread(buf.data(), 1, buf.size(), f.get());
        data.insert(data.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(n));
        if (n < buf.size()) {
            if (std::ferror(f.get())) throw ReadError(path, errno ? errno : EIO);
            break;
        }
    }
    const std::string_view magic = config::BEACON_MAGIC;
    if (data.size() >= magic.size() && std::string_view(reinterpret_cast<const char*>(data.data()), magic.size()) == magic)
        return beaconfile::BeaconFile::from_bytes(data);
    return picture::parse_qrsp(data, path);
}

tx::PassPlan plan_pass(const Source& src, const PassRequest& r) {
    const frame::FrameSpec& spec = frame_spec(r.waveform, r.frame);
    if (r.waveform == Waveform::L && r.ook)
        throw std::invalid_argument("--ook is waveform CE's: waveform L keys its callsign windows by FSK of its pilot");
    tx::PassSpec pass;
    pass.carrier = tx::Carrier(r.freq_hz);
    pass.spec = spec;
    pass.lead_in_s = r.lead_in_s;
    if (const auto* bf = std::get_if<beaconfile::BeaconFile>(&src)) {
        // segment, header and ook None: the file's own.
        if (r.callsign && *r.callsign != bf->callsign)
            throw std::invalid_argument("the beacon file is for " + bf->callsign + ", not " + *r.callsign);
        if (r.ook && !bf->ook)
            throw std::invalid_argument(
                "the beacon file keys its callsign windows FSK, and a beacon sends what its file says: make the "
                ".bin with qrss_beacon.py --ook instead of passing --ook here");
    } else {
        const picture::StoredPicture& sp = std::get<picture::StoredPicture>(src);
        if (!r.callsign)
            throw std::invalid_argument(
                "a .qrsp needs --callsign: every pass sends it in the header and in the callsign windows");
        header::HeaderFields h;
        h.callsign = *r.callsign;
        h.grid = r.grid;
        h.picture_id = sp.picture_id;
        h.mode = sp.mode;
        h.segment = r.segment;
        h.codec_id = sp.codec_id;
        pass.segment = r.segment;
        pass.header = std::move(h);
        pass.ook = r.ook;
    }
    tx::PassPlan plan = tx::plan_pass(src, pass);
    // precode's q check (sequences.scrambler) comes after every other.
    if (r.q < 0) throw std::invalid_argument("q must fit in uint64, not " + std::to_string(r.q));
    return plan;
}

std::string problem(const Source& src, const PassRequest& r) {
    try {
        plan_pass(src, r);
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return {};
}

tx::PassAudio pass_stream(const Source& src, const PassRequest& r) {
    return tx::PassAudio(plan_pass(src, r), static_cast<std::uint64_t>(r.q));
}

std::vector<float> pass_audio(const Source& src, const PassRequest& r) {
    const tx::PassAudio audio = pass_stream(src, r);
    std::vector<float> out(static_cast<std::size_t>(audio.size()));
    audio.render(0, std::span<float>(out));
    return out;
}

void write_wav(const Source& src, const PassRequest& r, const std::string& path, bool float32) {
    tx::write_wav(pass_stream(src, r), path, float32);
}

}  // namespace sstvae::qrss::api
