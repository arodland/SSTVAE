#include "settings/settings.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace sstvae::settings {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

fs::path home() {
#ifdef _WIN32
    if (const char* p = std::getenv("USERPROFILE")) return fs::path(p);
    const char* drive = std::getenv("HOMEDRIVE");
    const char* rest = std::getenv("HOMEPATH");
    if (drive && rest) return fs::path(std::string(drive) + rest);
#else
    if (const char* p = std::getenv("HOME")) return fs::path(p);
#endif
    return fs::current_path();
}

// Collects notes without every reader having to check for null.
class Notes {
public:
    explicit Notes(std::vector<Note>* sink) : sink_(sink) {}
    void add(const std::string& key, const std::string& problem) const {
        if (sink_) sink_->push_back({key, problem});
    }

private:
    std::vector<Note>* sink_;
};

std::string type_name(const json& v) {
    if (v.is_string()) return "a string";
    if (v.is_boolean()) return "a boolean";
    if (v.is_number_integer()) return "an integer";
    if (v.is_number()) return "a number";
    if (v.is_array()) return "an array";
    if (v.is_object()) return "an object";
    if (v.is_null()) return "null";
    return "an unknown type";
}

// Each reader keeps the existing value and notes the problem rather
// than throwing. `null` is silently treated as absent, because that is
// how the Python side spells "unset" for its optional fields.
struct Reader {
    const json& obj;
    std::string prefix;
    const Notes& notes;

    std::string path(const std::string& key) const { return prefix + key; }

    bool present(const std::string& key, const json** out) const {
        const auto it = obj.find(key);
        if (it == obj.end() || it->is_null()) return false;
        *out = &*it;
        return true;
    }

    void get(const std::string& key, std::string& dst) const {
        const json* v;
        if (!present(key, &v)) return;
        if (!v->is_string()) {
            notes.add(path(key), "expected a string, found " + type_name(*v));
            return;
        }
        dst = v->get<std::string>();
    }

    void get(const std::string& key, bool& dst) const {
        const json* v;
        if (!present(key, &v)) return;
        if (!v->is_boolean()) {
            notes.add(path(key), "expected a boolean, found " + type_name(*v));
            return;
        }
        dst = v->get<bool>();
    }

    void get(const std::string& key, int& dst) const {
        const json* v;
        if (!present(key, &v)) return;
        if (!v->is_number_integer()) {
            notes.add(path(key), "expected an integer, found " + type_name(*v));
            return;
        }
        dst = v->get<int>();
    }

    void get(const std::string& key, double& dst) const {
        const json* v;
        if (!present(key, &v)) return;
        if (!v->is_number()) {
            notes.add(path(key), "expected a number, found " + type_name(*v));
            return;
        }
        dst = v->get<double>();
    }

    // Nested section. A section of the wrong type is noted and skipped,
    // leaving the whole section at its defaults.
    std::optional<Reader> section(const std::string& key) const {
        const json* v;
        if (!present(key, &v)) return std::nullopt;
        if (!v->is_object()) {
            notes.add(path(key), "expected an object, found " + type_name(*v));
            return std::nullopt;
        }
        return Reader{*v, path(key) + ".", notes};
    }

    // Anything in the file we never asked about. Not an error -- an
    // older build reading a newer config hits this, and must not
    // rewrite the file in a way that discards the operator's settings
    // for options it does not know about.
    //
    // `string_view` rather than `std::string`: the callers pass literal
    // lists, and taking them as strings allocated a throwaway vector per
    // section. (It also tripped a -Wfree-nonheap-object false positive
    // in GCC 16 on the inlined temporary, which is its own reason not to
    // do it -- a spurious warning in a -Wall -Wextra build is noise that
    // hides real ones.)
    void report_unknown(std::initializer_list<std::string_view> known) const {
        for (const auto& item : obj.items()) {
            const std::string_view key = item.key();
            if (std::find(known.begin(), known.end(), key) == known.end()) {
                notes.add(path(std::string(key)),
                          "not a setting this build knows about (ignored)");
            }
        }
    }
};

void read_audio(const Reader& r, AudioConfig& c) {
    r.get("backend", c.backend);
    r.get("input_device", c.input_device);
    r.get("output_device", c.output_device);
    r.get("samplerate", c.samplerate);
    if (c.backend != "qt" && c.backend != "portaudio") {
        r.notes.add(r.path("backend"),
                    "unknown backend '" + c.backend + "'; expected qt or portaudio");
        c.backend = "qt";
    }
    // The one value here that is not a preference. A ring buffer filled
    // at the wrong rate decodes to nothing, and the failure looks like a
    // bad radio rather than a bad config, so it is worth refusing.
    if (c.samplerate != config::FS) {
        r.notes.add(r.path("samplerate"),
                    "is fixed by the modem at " + std::to_string(config::FS) +
                        " Hz and is not a device setting; ignoring " +
                        std::to_string(c.samplerate));
        c.samplerate = config::FS;
    }
    r.report_unknown({"backend", "input_device", "output_device", "samplerate"});
}

void read_rig(const Reader& r, RigConfig& c) {
    r.get("enabled", c.enabled);
    // v1 wrote the model as a string ("1"), because it was passed
    // straight to a rigctld command line; it is a number and is stored
    // as one now. Accept both, so migrating does not report a type
    // error against a key the operator never edited -- the same
    // courtesy the dead v1 keys get below.
    const json* model_value;
    if (r.present("model", &model_value)) {
        if (model_value->is_number_integer()) {
            c.model = model_value->get<int>();
        } else if (model_value->is_string()) {
            try {
                c.model = std::stoi(model_value->get<std::string>());
            } catch (const std::exception&) {
                r.notes.add(r.path("model"),
                            "expected a Hamlib model number, found \"" +
                                model_value->get<std::string>() + "\"");
            }
        } else {
            r.notes.add(r.path("model"),
                        "expected an integer, found " + type_name(*model_value));
        }
    }
    r.get("poll_interval_s", c.poll_interval_s);
    r.get("ptt_lead_s", c.ptt_lead_s);
    r.get("ptt_tail_s", c.ptt_tail_s);
    r.get("device", c.device);
    r.get("baud", c.baud);
    r.get("data_bits", c.data_bits);
    r.get("stop_bits", c.stop_bits);
    r.get("parity", c.parity);
    r.get("handshake", c.handshake);
    r.get("dtr", c.dtr);
    r.get("rts", c.rts);
    r.get("ptt_method", c.ptt_method);
    r.get("ptt_device", c.ptt_device);
    r.get("mode", c.mode);
    // The v1 keys are listed as known so that a config written by the
    // Python app is *quietly* migrated rather than reported as four
    // typos. They carry no information the new schema can use -- a
    // rigctld host and port describe a daemon this app does not have --
    // so the rig simply comes up on its defaults and the operator sets
    // it once. Naming them here is the difference between "your config
    // changed shape" and a wall of complaints about a file they did not
    // write.
    r.report_unknown({"enabled", "model", "poll_interval_s", "ptt_lead_s",
                      "ptt_tail_s", "device", "baud", "data_bits", "stop_bits",
                      "parity", "handshake", "dtr", "rts", "ptt_method",
                      "ptt_device", "mode",
                      // v1, ignored:
                      "host", "port", "spawn_local"});
}

void read_folders(const Reader& r, FolderConfig& c) {
    r.get("receive_dir", c.receive_dir);
    r.get("transmit_dir", c.transmit_dir);
    r.get("template_dir", c.template_dir);
    r.report_unknown({"receive_dir", "transmit_dir", "template_dir"});
}

void read_receive(const Reader& r, ReceiveConfig& c) {
    r.get("autosave", c.autosave);
    r.get("low_cpu", c.low_cpu);
    r.get("buffer_seconds", c.buffer_seconds);
    r.get("poll_interval", c.poll_interval);
    r.get("blind_search_seconds", c.blind_search_seconds);
    r.get("end_grace", c.end_grace);
    r.get("save_size", c.save_size);
    r.get("save_audio", c.save_audio);
    r.get("filename_template", c.filename_template);
    r.report_unknown({"autosave", "low_cpu", "buffer_seconds", "poll_interval",
                      "blind_search_seconds", "end_grace", "save_size", "save_audio",
                      "filename_template"});
}

void read_transmit(const Reader& r, TransmitConfig& c) {
    r.get("mode", c.mode);
    r.get("level", c.level);
    r.get("optimize", c.optimize);
    r.report_unknown({"mode", "level", "optimize"});
}

// Empty string <-> JSON null, for the fields Python declares optional.
json or_null(const std::string& s) {
    if (s.empty()) return nullptr;
    return s;
}

}  // namespace

FolderConfig::FolderConfig()
    : receive_dir((home() / "SSTVAE" / "received").string()),
      transmit_dir((home() / "Pictures").string()),
      template_dir((home() / "SSTVAE" / "templates").string()) {}

// This mirrors `platformdirs.user_config_dir("sstvae", appauthor=False)`,
// which is what the GUI app actually uses -- *not* the ImportError
// fallback in `sstvae/gui/settings.py`. The two differ, and the
// difference is easy to test against by accident: platformdirs lives in
// the `gui` extra, so an environment installing only `[cli]` takes the
// fallback and reports `~/.config` on every platform. Matching that
// would put the native app's config somewhere the GUI never looks.
//
// XDG_CONFIG_HOME is honoured on macOS as well as Linux, because
// platformdirs honours it on both. An all-whitespace value counts as
// unset, also matching (it does `.strip()` before testing).
namespace {

std::optional<fs::path> xdg_config_home() {
    const char* raw = std::getenv("XDG_CONFIG_HOME");
    if (!raw) return std::nullopt;
    std::string_view v(raw);
    const auto first = v.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return std::nullopt;
    const auto last = v.find_last_not_of(" \t\r\n");
    return fs::path(std::string(v.substr(first, last - first + 1)));
}

}  // namespace

fs::path config_dir() {
#if defined(_WIN32)
    // platformdirs maps user_config_dir to user_data_dir on Windows, and
    // defaults to roaming=False -- so this is CSIDL_LOCAL_APPDATA
    // (AppData\Local), not APPDATA (AppData\Roaming). Getting this
    // backwards is invisible on Linux and macOS and puts the config in
    // the wrong place on the one platform where it matters.
    if (const char* local = std::getenv("LOCALAPPDATA")) {
        if (*local) return fs::path(local) / "sstvae";
    }
    return home() / "AppData" / "Local" / "sstvae";
#elif defined(__APPLE__)
    if (const auto xdg = xdg_config_home()) return *xdg / "sstvae";
    return home() / "Library" / "Application Support" / "sstvae";
#else
    if (const auto xdg = xdg_config_home()) return *xdg / "sstvae";
    return home() / ".config" / "sstvae";
#endif
}

fs::path config_path() { return config_dir() / "config.json"; }

std::vector<std::string> LoadResult::messages() const {
    std::vector<std::string> out;
    out.reserve(notes.size());
    for (const Note& n : notes) out.push_back(n.key + ": " + n.problem);
    return out;
}

Config from_json(const std::string& text, std::vector<Note>* sink) {
    const Notes notes(sink);
    Config c;

    json root = json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (root.is_discarded()) {
        notes.add("<file>", "is not valid JSON; using defaults throughout");
        return c;
    }
    if (!root.is_object()) {
        notes.add("<file>", "is not a JSON object; using defaults throughout");
        return c;
    }

    const Reader r{root, "", notes};
    r.get("callsign", c.callsign);
    r.get("model_path", c.model_path);
    r.get("precision", c.precision);
    r.get("version", c.version);
    if (auto s = r.section("audio")) read_audio(*s, c.audio);
    if (auto s = r.section("rig")) read_rig(*s, c.rig);
    if (auto s = r.section("folders")) read_folders(*s, c.folders);
    if (auto s = r.section("receive")) read_receive(*s, c.receive);
    if (auto s = r.section("transmit")) read_transmit(*s, c.transmit);
    r.report_unknown({"callsign", "model_path", "precision", "version", "audio", "rig",
                      "folders", "receive", "transmit"});

    if (c.version > CONFIG_VERSION) {
        notes.add("version",
                  "was written by a newer build (" + std::to_string(c.version) +
                      " > " + std::to_string(CONFIG_VERSION) +
                      "); unknown settings are kept in the file but not applied");
    }
    return c;
}

std::string to_json(const Config& c) {
    // Field order matches the Python dataclasses so a diff between a
    // config written by either app stays readable.
    const json root = {
        {"callsign", c.callsign},
        {"model_path", or_null(c.model_path)},
        {"precision", c.precision},
        {"audio",
         {{"backend", c.audio.backend},
          {"input_device", or_null(c.audio.input_device)},
          {"output_device", or_null(c.audio.output_device)},
          {"samplerate", c.audio.samplerate}}},
        {"rig",
         {{"enabled", c.rig.enabled},
          {"model", c.rig.model},
          {"poll_interval_s", c.rig.poll_interval_s},
          {"ptt_lead_s", c.rig.ptt_lead_s},
          {"ptt_tail_s", c.rig.ptt_tail_s},
          {"device", c.rig.device},
          {"baud", c.rig.baud},
          {"data_bits", c.rig.data_bits},
          {"stop_bits", c.rig.stop_bits},
          {"parity", c.rig.parity},
          {"handshake", c.rig.handshake},
          {"dtr", c.rig.dtr},
          {"rts", c.rig.rts},
          {"ptt_method", c.rig.ptt_method},
          {"ptt_device", c.rig.ptt_device},
          {"mode", c.rig.mode}}},
        {"folders",
         {{"receive_dir", c.folders.receive_dir},
          {"transmit_dir", c.folders.transmit_dir},
          {"template_dir", c.folders.template_dir}}},
        {"receive",
         {{"autosave", c.receive.autosave},
          {"low_cpu", c.receive.low_cpu},
          {"buffer_seconds", c.receive.buffer_seconds},
          {"poll_interval", c.receive.poll_interval},
          {"blind_search_seconds", c.receive.blind_search_seconds},
          {"end_grace", c.receive.end_grace},
          {"save_size", or_null(c.receive.save_size)},
          {"save_audio", c.receive.save_audio},
          {"filename_template", c.receive.filename_template}}},
        {"transmit",
         {{"mode", c.transmit.mode},
          {"level", c.transmit.level},
          {"optimize", c.transmit.optimize}}},
        {"version", c.version},
    };
    return root.dump(2) + "\n";
}

LoadResult load(const fs::path& where) {
    LoadResult result;
    const fs::path path = where.empty() ? config_path() : where;

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        // Absent is normal -- a first run. Not worth a note.
        return result;
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    if (in.bad()) {
        result.notes.push_back({"<file>", "could not be read; using defaults"});
        return result;
    }
    result.config = from_json(buf.str(), &result.notes);
    return result;
}

fs::path save(const Config& c, const fs::path& where) {
    const fs::path path = where.empty() ? config_path() : where;
    if (path.has_parent_path()) fs::create_directories(path.parent_path());

    fs::path tmp = path;
    tmp += ".tmp";
    const std::string text = to_json(c);
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write " + tmp.string());
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.flush();
        if (!out) throw std::runtime_error("failed writing " + tmp.string());
    }

    // fsync the file before the rename. Without it, a crash can leave
    // the rename durable while the contents are not -- which is exactly
    // the truncated-config failure the atomic write exists to prevent.
#ifndef _WIN32
    if (FILE* f = std::fopen(tmp.string().c_str(), "rb")) {
        ::fsync(::fileno(f));
        std::fclose(f);
    }
#else
    {
        const HANDLE h = CreateFileA(tmp.string().c_str(), GENERIC_READ | GENERIC_WRITE,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            FlushFileBuffers(h);
            CloseHandle(h);
        }
    }
#endif

    // Atomic on POSIX and on Windows (rename over an existing file).
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        throw std::runtime_error("cannot replace " + path.string());
    }
    return path;
}

std::optional<std::string> codec_precision(const Config& c) {
    // A .pt selects the torch backend, which has no precision and
    // rejects the argument rather than ignoring it. The native app
    // cannot load one at all, but the config may still name one.
    if (!c.model_path.empty() && fs::path(c.model_path).extension() == ".pt") {
        return std::nullopt;
    }
    return c.precision;
}

std::string format_filename(const std::string& tmpl, const FilenameFields& fields) {
    const std::int64_t epoch =
        fields.when.value_or(static_cast<std::int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count()));
    const std::time_t t = static_cast<std::time_t>(epoch);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &t);
#else
    gmtime_r(&t, &utc);
#endif

    char date[16], clock[16];
    std::strftime(date, sizeof date, "%Y-%m-%d", &utc);
    std::strftime(clock, sizeof clock, "%H%M%S", &utc);

    std::string freq;
    if (fields.freq_hz && *fields.freq_hz != 0.0) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.3fMHz", *fields.freq_hz / 1e6);
        freq = buf;
    }
    std::string callsign = fields.callsign;
    // Trim, then make it safe for a filename.
    const auto first = callsign.find_first_not_of(" \t\r\n");
    const auto last = callsign.find_last_not_of(" \t\r\n");
    callsign = (first == std::string::npos) ? "" : callsign.substr(first, last - first + 1);
    for (char& ch : callsign) {
        if (ch == '/') ch = '-';
    }

    const std::vector<std::pair<std::string, std::string>> values{
        {"{date}", date}, {"{time}", clock},   {"{freq}", freq},
        {"{callsign}", callsign}, {"{mode}", fields.mode},
    };

    // Split on '_', substitute, and drop chunks that came out empty --
    // that is what removes the stranded separators.
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= tmpl.size()) {
        const std::size_t at = tmpl.find('_', start);
        std::string chunk = tmpl.substr(start, at == std::string::npos
                                                   ? std::string::npos
                                                   : at - start);
        for (const auto& [token, value] : values) {
            std::size_t pos;
            while ((pos = chunk.find(token)) != std::string::npos) {
                chunk.replace(pos, token.size(), value);
            }
        }
        if (!chunk.empty()) parts.push_back(chunk);
        if (at == std::string::npos) break;
        start = at + 1;
    }

    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i) out += '_';
        out += parts[i];
    }
    if (out.empty()) out = std::string(date) + "_" + clock + "Z";
    return out;
}

}  // namespace sstvae::settings
