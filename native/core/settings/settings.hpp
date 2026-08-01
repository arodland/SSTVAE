// Persistent application configuration.
//
// Plain structs over JSON. The schema comes from the Python GUI's
// `settings.py` (deleted 2026-08-01), so a config.json that app wrote
// still loads -- see CONFIG_VERSION for the one section that changed
// shape.
// Two robustness rules carry over from there, both learned from config
// files that ate themselves:
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

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "config.hpp"

namespace sstvae::settings {

// 2 since 2026-07-29: the rig section was reshaped around libhamlib
// rather than a rigctld socket (see RigConfig). Everything else in the
// file is unchanged, and unknown keys are still ignored rather than
// fatal, so the bump is a record of the change rather than a gate.
inline constexpr int CONFIG_VERSION = 2;

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

// What Hamlib needs to talk to a radio.
//
// This section deliberately **breaks compatibility** with the config
// the Python app writes (decided 2026-07-29), and the version bump is
// how a reader finds out. The old shape -- host, port, spawn_local --
// described a *rigctld socket*, which is the one part of rig control
// the native app does not have: it links libhamlib in-process, so there
// is no daemon to address or to start. Hamlib model 2 ("NET rigctl") is
// the rigctld client, so talking to a remote daemon is now a model
// number in the same picker rather than a parallel set of fields.
//
// The rest is modelled on WSJT-X's Radio tab, because that is the set a
// real radio actually needs and the one operators already know. Every
// field maps to a `rig_set_conf` token (see `rig/hamlib.cpp`); the
// enumerated ones use our own lowercase spellings rather than Hamlib's,
// so the config file does not become a place where Hamlib's combo-box
// strings leak into a user's editor.
//
// "default" everywhere means *do not set the token at all* and let the
// backend's own default stand -- which for most rigs is the right
// answer and is why WSJT-X offers it as a distinct choice rather than
// making the user guess 8-N-1.
struct RigConfig {
    bool enabled = false;
    int model = 1;  // Hamlib model number; 1 is the dummy rig, 2 is NET rigctl
    double poll_interval_s = 5.0;
    double ptt_lead_s = 0.3;
    double ptt_tail_s = 0.3;

    // --- CAT ---------------------------------------------------------
    // Serial device, or "host:port" when the model is NET rigctl.
    //
    // Keeps v1's key name, and not by accident: v1 also had a `port`,
    // an integer holding a rigctld TCP port. Reusing *that* name for a
    // device string would make every migrated config report a type
    // error on a key the operator never touched. `device` meant this
    // already.
    std::string device;
    int baud = 0;                       // 0 = the backend's default
    std::string data_bits = "default";  // default | seven | eight
    std::string stop_bits = "default";  // default | one | two
    std::string parity = "default";     // default | none | odd | even
    std::string handshake = "default";  // default | none | xonxoff | hardware
    // Held high or low for the life of the session, which is how an
    // interface that steals its power from the control lines stays fed.
    std::string dtr = "default";  // default | high | low
    std::string rts = "default";  // default | high | low

    // --- PTT ---------------------------------------------------------
    // "vox" means do not key at all: the operator's radio is keyed by
    // the audio itself, so the transmit engine must be given nothing to
    // key rather than something that fails.
    std::string ptt_method = "cat";  // vox | cat | dtr | rts
    // Often a *different* port from the CAT one -- a serial adapter
    // whose control lines key the rig while CAT runs elsewhere, or a
    // rig with no CAT keying at all. Empty means "the CAT device".
    std::string ptt_device;

    // Set on the rig when the session opens; "none" leaves whatever the
    // operator has dialled in alone.
    std::string mode = "none";  // none | usb | pkt_usb
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

    // Spend the composing time improving the latents for this
    // particular picture (docs/latent-optimization.md). Off by default:
    // it fetches an extra 18 MB artifact on first use, and a station
    // that does not want it should never pay for it.
    //
    // Deliberately one switch and no dials. The objective SNR is a
    // measured constant rather than a setting, and the budgets only
    // matter through a plateau test that usually ends the run first.
    bool optimize = false;
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
