// Persistent application configuration.
//
// Plain structs over JSON, matching `sstvae/gui/settings.py`'s schema so
// an existing config.json loads unchanged. Two robustness rules carry
// over from there, both learned from config files that ate themselves:
//
//  * **Writes are atomic** (temp file + rename), so losing power or
//    filling the disk mid-save leaves the previous config intact rather
//    than a truncated one.
//  * **Loading never fails.** A corrupt config must not stop the
//    application starting, because the settings dialog is how the
//    operator would fix it.
//
// One deliberate improvement on the reference. Python silently drops
// keys it does not recognise or cannot use; that is the right *effect*
// but it makes a typo in a hand-edited config invisible -- the operator
// changes a value, nothing happens, and nothing says why. `load()`
// returns the same never-failing Config plus a list of notes saying
// exactly what was ignored and what was used instead. Still never
// fatal; just no longer silent.

#ifndef SSTVAE_SETTINGS_SETTINGS_HPP
#define SSTVAE_SETTINGS_SETTINGS_HPP

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "config.hpp"

namespace sstvae::settings {

inline constexpr int CONFIG_VERSION = 1;

// Where the config lives, per platform. Matches what platformdirs'
// user_config_dir("sstvae", appauthor=False) picks, so the C++ and
// Python apps read the same file on the same machine.
std::filesystem::path config_dir();
std::filesystem::path config_path();

struct AudioConfig {
    // Which library talks to the soundcard. "qt" (QtMultimedia) is the
    // default because its realtime side is C++ inside Qt -- a PortAudio
    // callback written in Python sits on the audio thread and loses the
    // GIL race, which cost 5 dB and a mangled picture on JACK. The
    // native app does not have that problem, but the two backends also
    // *enumerate different devices* (Qt does not list PulseAudio or
    // PipeWire monitor sources), so the choice stays meaningful.
    std::string backend = "qt";
    // Device description under the active backend; empty = system
    // default. A name rather than an opaque id, so the config stays
    // hand-editable and a name that no longer resolves is kept and
    // flagged rather than silently reset.
    std::string input_device;
    std::string output_device;
    // What lands in the ring buffer. Fixed by the modem -- not a device
    // setting, and changing it produces silent garbage.
    int samplerate = config::FS;
};

struct RigConfig {
    bool enabled = false;
    std::string host = "127.0.0.1";
    int port = 4532;
    bool spawn_local = false;  // start our own rigctld rather than reuse one
    std::string model = "1";   // Hamlib rig model number; 1 is the dummy rig
    std::string device = "/dev/ttyUSB0";
    int baud = 19200;
    double ptt_lead_s = 0.3;
    double ptt_tail_s = 0.3;
    double poll_interval_s = 5.0;
};

struct FolderConfig {
    std::string receive_dir;
    std::string transmit_dir;
    std::string template_dir;

    FolderConfig();  // defaults are home-relative, so not constant
};

struct ReceiveConfig {
    bool autosave = true;
    bool low_cpu = false;
    double buffer_seconds = 130.0;
    double poll_interval = 5.0;
    double blind_search_seconds = 25.0;
    double end_grace = 8.0;
    std::string save_size;  // e.g. "320x240"; empty keeps 640x480
    // Diagnostic: also write the captured audio beside each received
    // picture, float32 at FS with nothing rescaled. Answers "was the
    // picture bad because the audio was bad?" without needing the
    // hardware again -- the dump decodes offline with sstvae-decode.
    bool save_audio = false;
    // Fields: date, time, freq, callsign, mode. Missing ones drop out
    // of the name rather than leaving an empty gap.
    std::string filename_template = "{date}_{time}Z_{freq}_{callsign}";
};

struct TransmitConfig {
    std::string mode = "B";
    double level = 0.9;
};

struct Config {
    std::string callsign;
    // Empty = the published ONNX artifacts, fetched and cached on first
    // use. May also be a .onnx artifact or a directory of them.
    std::string model_path;
    // Purely local: every precision decodes every other precision's
    // transmissions, so this never has to match the far end.
    std::string precision = "fp16";
    AudioConfig audio;
    RigConfig rig;
    FolderConfig folders;
    ReceiveConfig receive;
    TransmitConfig transmit;
    int version = CONFIG_VERSION;
};

// What `load` found but could not use. Never an error -- the
// corresponding default or previous value is in the Config.
struct Note {
    std::string key;      // dotted path, e.g. "audio.samplerate"
    std::string problem;  // what was wrong
};

struct LoadResult {
    Config config;
    std::vector<Note> notes;

    bool clean() const { return notes.empty(); }
    // One line per note, for a log or the settings dialog.
    std::vector<std::string> messages() const;
};

// Read the config. Never throws: anything missing, unparseable or of
// the wrong type falls back to its default and is reported in `notes`.
LoadResult load(const std::filesystem::path& path = {});

// Serialize to JSON text (2-space indent, trailing newline), which is
// what `save` writes. Exposed so a test can round-trip without a file.
std::string to_json(const Config& config);
Config from_json(const std::string& text, std::vector<Note>* notes = nullptr);

// Atomic write: temp file, flush, fsync, rename. Returns the path
// written. Throws only if the file genuinely cannot be written --
// unlike loading, a failed save must not be silent.
std::filesystem::path save(const Config& config,
                           const std::filesystem::path& path = {});

// The `precision` to hand the codec for this config, or nullopt when
// the model path selects a backend that has no precision.
std::optional<std::string> codec_precision(const Config& config);

// Render a received-picture filename from the configured template.
//
// Fields the reception did not supply drop out along with any separator
// that would be left stranded, so a decode with no callsign and no rig
// yields `2026-07-28_011542Z` rather than `2026-07-28_011542Z__`.
struct FilenameFields {
    std::string callsign;
    std::optional<double> freq_hz;
    std::string mode;
    // Seconds since the epoch, UTC. Defaults to now.
    std::optional<std::int64_t> when;
};
std::string format_filename(const std::string& tmpl, const FilenameFields& fields = {});

}  // namespace sstvae::settings

#endif
