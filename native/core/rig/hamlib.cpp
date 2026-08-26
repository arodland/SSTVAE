#include "rig/hamlib.hpp"


#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hamlib/rig.h>
#include <hamlib/riglist.h>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace sstvae::rig {

namespace {

// The level `SSTVAE_HAMLIB_DEBUG` asks for, which is also the level to
// go back to when a debug sink is removed. Read every time rather than
// cached: it is two getenv calls in a lifetime, and a cached copy is one
// more thing that can be stale.
enum rig_debug_level_e env_debug_level() {
    const char* debug = std::getenv("SSTVAE_HAMLIB_DEBUG");
    const bool verbose =
        debug != nullptr && *debug != '\0' && std::strcmp(debug, "0") != 0;
    return verbose ? RIG_DEBUG_TRACE : RIG_DEBUG_NONE;
}

// Hamlib's backend registry is global and loading it is not reentrant,
// so it is done once. `list_models` and `rig_init` both need it -- a
// `rig_init` on a model whose backend has not been loaded simply fails.
void load_backends_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        // Quiet by default: Hamlib logs to stderr, and a CAT timeout is
        // reported through our own status text rather than as a wall of
        // library chatter behind whatever the operator is reading.
        //
        // SSTVAE_HAMLIB_DEBUG turns that back on, because "quiet" and
        // "unfalsifiable" are close together when the library owns the
        // serial port: the questions an operator's bug report actually
        // raises -- which CAT command did the rig refuse, how far did
        // open get -- are answerable only from Hamlib's own trace. The
        // rig tests set it so a failure arrives with that trace already
        // attached rather than needing the run repeated.
        rig_set_debug(env_debug_level());
        rig_load_all_backends();
    });
}

std::string hamlib_error(const char* what, int code) {
    const char* text = rigerror(code);
    return std::string(what) + ": " + (text != nullptr ? text : "unknown error") +
           " (" + std::to_string(code) + ")";
}

// --- Hamlib's own trace, routed somewhere an app can show it ---------
//
// Hamlib formats trace lines with `rig_debug(level, fmt, ...)` and hands
// a registered callback the *unformatted* fmt plus a `va_list`, so the
// formatting is ours to do. Three details are load-bearing:
//
//   * **A line is not a call.** `dump_hex` and the frame tracers emit a
//     line in several calls and sometimes several lines in one, so the
//     text is accumulated and split on newlines. A sink handed raw calls
//     would show a CI-V frame smeared across a dozen entries.
//   * **The `va_list` is the caller's.** `vsnprintf` consumes it, so
//     both the sizing pass and the real one take a `va_copy`.
//   * **Nothing may escape.** This is called from C, through a function
//     pointer, while Hamlib holds its own debug mutex -- an exception
//     crossing that boundary is a terminate, and a call back into
//     `rig_debug` from the sink is a deadlock. Hence the try/catch and
//     the contract in the header.
std::mutex& debug_mu() {
    static std::mutex m;
    return m;
}

DebugSink& debug_sink() {
    static DebugSink sink;
    return sink;
}

std::string& debug_partial() {
    static std::string partial;
    return partial;
}

// A frame trace at 115200 is verbose but bounded; this is only here so a
// sink that has stopped consuming, or a format with no newline in it at
// all, cannot grow the buffer without limit.
constexpr std::size_t kMaxPartialBytes = 8192;

std::string format_debug(const char* fmt, va_list ap) {
    va_list sizing;
    va_copy(sizing, ap);
    char stack[512];
    const int n = std::vsnprintf(stack, sizeof(stack), fmt, sizing);
    va_end(sizing);
    if (n < 0) return {};
    if (static_cast<std::size_t>(n) < sizeof(stack)) {
        return std::string(stack, static_cast<std::size_t>(n));
    }
    std::string big(static_cast<std::size_t>(n), '\0');
    va_list again;
    va_copy(again, ap);
    // `&big[0]` has n+1 writable chars, the last of which holds the
    // terminator vsnprintf is going to write there anyway.
    std::vsnprintf(&big[0], static_cast<std::size_t>(n) + 1, fmt, again);
    va_end(again);
    return big;
}

int debug_trampoline(enum rig_debug_level_e, rig_ptr_t, const char* fmt, va_list ap) {
    try {
        if (fmt == nullptr) return 0;
        const std::string text = format_debug(fmt, ap);
        if (text.empty()) return 0;

        std::vector<std::string> lines;
        DebugSink sink;
        {
            std::lock_guard<std::mutex> lock(debug_mu());
            sink = debug_sink();
            if (!sink) return 0;
            std::string& partial = debug_partial();
            partial += text;
            std::size_t start = 0;
            for (;;) {
                const std::size_t nl = partial.find('\n', start);
                if (nl == std::string::npos) break;
                lines.emplace_back(partial, start, nl - start);
                start = nl + 1;
            }
            partial.erase(0, start);
            if (partial.size() > kMaxPartialBytes) {
                lines.push_back(partial);
                partial.clear();
            }
        }

        // Outside the lock: the sink is the app's, may take a lock of
        // its own, and holding ours across it would make the ordering
        // between the two a property of whoever wrote the sink.
        for (const std::string& line : lines) {
            if (!line.empty() && line.back() == '\r') {
                sink(line.substr(0, line.size() - 1));
            } else {
                sink(line);
            }
        }
    } catch (...) {
        // Nowhere to report to, and this is a C boundary.
    }
    return 0;
}

class HamlibBackend final : public RigBackend {
public:
    explicit HamlibBackend(HamlibConfig config) : config_(std::move(config)) {}

    ~HamlibBackend() override { close(); }

    void open() override {
        load_backends_once();
        if (rig_ != nullptr) return;

        rig_ = rig_init(static_cast<rig_model_t>(config_.model));
        if (rig_ == nullptr) {
            throw RigError("Hamlib does not know rig model " +
                           std::to_string(config_.model));
        }

        // Everything is configured through rig_set_conf rather than by
        // writing to the port struct. The RIGPORT/PTTPORT macros are
        // guarded by IN_HAMLIB -- they are Hamlib's internals, and the
        // header says so and that they are subject to change. The
        // token route is the supported one and survives the struct
        // being rearranged, which upstream is explicitly planning.
        //
        // For MODEL_NET_RIGCTL the pathname is "host:port"; netrigctl
        // reads the same setting, so there is no branch here.
        if (!config_.device.empty()) set_conf("rig_pathname", config_.device);
        if (config_.baud > 0) set_conf("serial_speed", std::to_string(config_.baud));
        apply_serial_settings();
        apply_ptt_settings();

        // One timeout and one retry, both ours. See HamlibConfig: the
        // reference has two layers of each and controls only the outer,
        // and `RigController::stop()` leans on these being short.
        set_conf("timeout", std::to_string(config_.timeout_ms));
        set_conf("retry", std::to_string(config_.retries));

        // Hamlib's own polling thread, off.
        //
        // `rig_open` otherwise starts one (poll_interval defaults to
        // 1000 ms) that issues its own CAT commands for transceive
        // emulation. That is a direct conflict with what RigController
        // is for: it exists to keep exactly one command in flight and
        // to guarantee keying never waits behind a status read, and it
        // cannot do either if the library is talking to the same serial
        // port behind its back. We poll at our own interval, on our own
        // thread, and want the port quiet in between -- which also means
        // one fewer thread to shut down when a wedged rig is abandoned.
        set_conf("poll_interval", "0");

        const int rc = rig_open(rig_);
        if (rc != RIG_OK) {
            const std::string msg = hamlib_error("could not open the rig", rc);
            rig_cleanup(rig_);
            rig_ = nullptr;
            throw RigError(msg);
        }
        open_ = true;
        apply_mode();
    }

    void close() noexcept override {
        if (rig_ == nullptr) return;
        if (open_) {
            rig_close(rig_);
            open_ = false;
        }
        rig_cleanup(rig_);
        rig_ = nullptr;
    }

    void set_ptt(bool on) override {
        require_open();
        // RIG_PTT_ON for the mic, not RIG_PTT_ON_MIC: they differ on a
        // MICDATA rig (`TX` against `TX0`) and only the first is what
        // this app has always sent, so the default keys the radio the
        // way it did before this setting existed. Hamlib downgrades
        // ON_DATA to ON on a rig that is not MICDATA, so the branch is
        // safe even if a config claims otherwise.
        const ptt_t key = config_.ptt_audio == PttAudio::Data && key_by_cat()
                              ? RIG_PTT_ON_DATA
                              : RIG_PTT_ON;
        const int rc = rig_set_ptt(rig_, RIG_VFO_CURR, on ? key : RIG_PTT_OFF);
        if (rc != RIG_OK) {
            throw RigError(hamlib_error(on ? "PTT on failed" : "PTT off failed", rc));
        }
    }

    double frequency_hz() override {
        require_open();
        freq_t freq = 0;
        const int rc = rig_get_freq(rig_, RIG_VFO_CURR, &freq);
        if (rc != RIG_OK) throw RigError(hamlib_error("could not read frequency", rc));
        return static_cast<double>(freq);
    }

    std::string description() const override {
        // Via rig_get_caps_cptr rather than rig_->caps->..., and that is
        // deliberate: **nothing here may dereference a RIG\***.
        //
        // `struct rig_state`, which `struct RIG` embeds, contains
        // pthread_mutex_t members. MSVC has no pthread.h, so Windows
        // builds compile against a shim
        // (native/third_party/msvc-pthread/) whose type sizes cannot be
        // guaranteed to match the winpthreads the bundled MinGW-built
        // DLL was compiled with. Reading through a RIG* would then find
        // every field after the first mutex at the wrong offset --
        // silently. Taking the model number instead means no struct
        // layout is relied on at all. `struct rig_caps`, the one thing
        // list_models() does read through, has no pthread members.
        const char* mfg =
            rig_get_caps_cptr(static_cast<rig_model_t>(config_.model),
                              RIG_CAPS_MFG_NAME_CPTR);
        const char* model =
            rig_get_caps_cptr(static_cast<rig_model_t>(config_.model),
                              RIG_CAPS_MODEL_NAME_CPTR);
        if (mfg != nullptr && model != nullptr) {
            return std::string(mfg) + " " + model;
        }
        return "model " + std::to_string(config_.model);
    }

private:
    void require_open() const {
        if (rig_ == nullptr || !open_) throw RigError("the rig is not open");
    }

    // The token names and their allowed values are Hamlib's, taken from
    // `src/serial_cfg_params.h` and `src/conf.c` in the pinned release
    // rather than from memory -- a misspelled token is not an error,
    // `rig_token_lookup` simply returns RIG_CONF_END and the setting is
    // silently ignored, which is the failure mode this whole file is
    // written to avoid.
    void apply_serial_settings() {
        switch (config_.data_bits) {
            case DataBits::Seven: set_conf("data_bits", "7"); break;
            case DataBits::Eight: set_conf("data_bits", "8"); break;
            case DataBits::Default: break;
        }
        switch (config_.stop_bits) {
            case StopBits::One: set_conf("stop_bits", "1"); break;
            case StopBits::Two: set_conf("stop_bits", "2"); break;
            case StopBits::Default: break;
        }
        switch (config_.parity) {
            case Parity::None: set_conf("serial_parity", "None"); break;
            case Parity::Odd: set_conf("serial_parity", "Odd"); break;
            case Parity::Even: set_conf("serial_parity", "Even"); break;
            case Parity::Default: break;
        }
        switch (config_.handshake) {
            case Handshake::None: set_conf("serial_handshake", "None"); break;
            case Handshake::XonXoff:
                set_conf("serial_handshake", "XONXOFF");
                break;
            case Handshake::Hardware:
                set_conf("serial_handshake", "Hardware");
                break;
            case Handshake::Default: break;
        }
        // "Unset" is Hamlib's own name for "leave it alone", so
        // Default could equally be spelled by setting it; not setting
        // the token at all is the same thing and one fewer assumption.
        switch (config_.dtr) {
            case LineState::High: set_conf("dtr_state", "ON"); break;
            case LineState::Low: set_conf("dtr_state", "OFF"); break;
            case LineState::Default: break;
        }
        switch (config_.rts) {
            case LineState::High: set_conf("rts_state", "ON"); break;
            case LineState::Low: set_conf("rts_state", "OFF"); break;
            case LineState::Default: break;
        }
    }

    bool key_by_cat() const { return config_.ptt_method == PttMethod::Cat; }

    void apply_ptt_settings() {
        switch (config_.ptt_method) {
            // Vox means the rig is keyed by its own audio detector, so
            // Hamlib must not key it: "None" is the token value that
            // says exactly that. Callers are additionally expected not
            // to hand this backend to the transmit engine as a PTT.
            case PttMethod::Vox: set_conf("ptt_type", "None"); break;
            // "RIG" would *downgrade* a MICDATA rig: the token
            // overwrites caps->ptt_type, and RIG_PTT_RIG makes Hamlib
            // fold ON_DATA back into a plain mic key.
            case PttMethod::Cat:
                set_conf("ptt_type",
                         config_.ptt_audio == PttAudio::Data ? "RIGMICDATA" : "RIG");
                break;
            case PttMethod::Dtr: set_conf("ptt_type", "DTR"); break;
            case PttMethod::Rts: set_conf("ptt_type", "RTS"); break;
        }
        // Only meaningful for the line-keyed methods, and only worth
        // setting when it differs -- an empty value means "the CAT
        // device", which Hamlib already assumes.
        if (!config_.ptt_device.empty() &&
            (config_.ptt_method == PttMethod::Dtr ||
             config_.ptt_method == PttMethod::Rts)) {
            set_conf("ptt_pathname", config_.ptt_device);
        }
    }

    // After open, not before: it is a rig command rather than a port
    // setting. A failure is reported and not fatal, because a rig that
    // will not change mode is still a rig we can key and read.
    void apply_mode() {
        rmode_t mode = RIG_MODE_NONE;
        switch (config_.mode) {
            case RigMode::Usb: mode = RIG_MODE_USB; break;
            case RigMode::PktUsb: mode = RIG_MODE_PKTUSB; break;
            case RigMode::None: return;
        }
        rig_set_mode(rig_, RIG_VFO_CURR, mode, RIG_PASSBAND_NOCHANGE);
    }

    void set_conf(const char* name, const std::string& value) {
        const hamlib_token_t token = rig_token_lookup(rig_, name);
        // Not every backend has every setting -- netrigctl has no serial
        // retry, for one -- and a missing knob is not a failure worth
        // refusing to open the radio over.
        if (token == RIG_CONF_END) return;
        rig_set_conf(rig_, token, value.c_str());
    }

    HamlibConfig config_;
    RIG* rig_ = nullptr;
    bool open_ = false;
};

}  // namespace

std::string RigModel::label() const {
    std::string out = manufacturer;
    if (!out.empty() && !name.empty()) out += " ";
    out += name;
    if (!status.empty()) out += " (" + status + ")";
    return out;
}

std::vector<RigModel> list_models() {
    load_backends_once();
    std::vector<RigModel> models;
    rig_list_foreach(
        [](const struct rig_caps* caps, rig_ptr_t data) -> int {
            auto* out = static_cast<std::vector<RigModel>*>(data);
            RigModel m;
            m.model = caps->rig_model;
            m.manufacturer = caps->mfg_name != nullptr ? caps->mfg_name : "";
            m.name = caps->model_name != nullptr ? caps->model_name : "";
            m.version = caps->version != nullptr ? caps->version : "";
            const char* status = rig_strstatus(caps->status);
            m.status = status != nullptr ? status : "";
            out->push_back(std::move(m));
            return 1;  // keep going
        },
        &models);

    std::sort(models.begin(), models.end(), [](const RigModel& a, const RigModel& b) {
        if (a.manufacturer != b.manufacturer) return a.manufacturer < b.manufacturer;
        if (a.name != b.name) return a.name < b.name;
        return a.model < b.model;
    });
    return models;
}

SerialDefaults serial_defaults(int model) {
    load_backends_once();

    struct Search {
        int model;
        SerialDefaults out;
        bool found = false;
    } search{model, SerialDefaults{}, false};

    // `rig_list_foreach` rather than a `RIG*`: there is no
    // `rig_get_caps_int` token for the serial fields, and reading them
    // out of an initialised RIG would mean dereferencing a pointer whose
    // struct carries pthread members -- the thing CLAUDE.md forbids on
    // Windows. `struct rig_caps` has none, which is why it is the one
    // Hamlib struct this file reads through.
    rig_list_foreach(
        [](const struct rig_caps* caps, rig_ptr_t data) -> int {
            auto* s = static_cast<Search*>(data);
            if (static_cast<int>(caps->rig_model) != s->model) return 1;
            s->found = true;
            // `serial_rate_max`, not a mean or a minimum: `rig_init`
            // takes the default from that field and its own comment
            // says "fastest !". Matching it is the whole point -- a
            // different choice here would make a bridged rig run at a
            // different speed than the same rig on a desktop.
            if (caps->serial_rate_max > 0) s->out.baud = caps->serial_rate_max;
            if (caps->serial_data_bits > 0) s->out.data_bits = caps->serial_data_bits;
            if (caps->serial_stop_bits > 0) s->out.stop_bits = caps->serial_stop_bits;
            switch (caps->serial_parity) {
                case RIG_PARITY_ODD: s->out.parity = SerialParams::kOdd; break;
                case RIG_PARITY_EVEN: s->out.parity = SerialParams::kEven; break;
                default: s->out.parity = SerialParams::kNoParity; break;
            }
            switch (caps->serial_handshake) {
                case RIG_HANDSHAKE_HARDWARE: s->out.flow = SerialParams::kRtsCts; break;
                case RIG_HANDSHAKE_XONXOFF: s->out.flow = SerialParams::kXonXoff; break;
                default: s->out.flow = SerialParams::kNoFlow; break;
            }
            return 0;  // stop
        },
        &search);

    // A model Hamlib does not know is a configuration the operator has
    // to fix anyway, and `rig_init` will refuse it a moment later with a
    // better message than this function could give. 9600 8-N-1 rather
    // than throwing, so a settings screen can still render.
    return search.out;
}

bool supports_ptt_audio_source(int model) {
    load_backends_once();
    return rig_get_caps_int(static_cast<rig_model_t>(model), RIG_CAPS_PTT_TYPE) ==
           RIG_PTT_RIG_MICDATA;
}

std::unique_ptr<RigBackend> make_hamlib_backend(const HamlibConfig& config) {
    return std::make_unique<HamlibBackend>(config);
}

std::string hamlib_version() {
    // rig_version(), not the `hamlib_version2` variable, and that is a
    // Windows requirement rather than taste. Hamlib exports the version
    // string as *data*, and MSVC cannot import a data symbol from a DLL
    // without __declspec(dllimport) -- which Hamlib's headers only emit
    // when the consumer defines `DLL_EXPORT`, a name far too generic to
    // want in a translation unit. Function symbols have no such problem:
    // the import library thunks them. This linked on Linux and macOS and
    // failed only on Windows, with exactly one unresolved symbol.
    const char* v = rig_version();
    return v != nullptr ? v : "";
}

void set_debug_sink(DebugSink sink) {
    // Not for the registry, which this does not need, but because it is
    // where the env-driven level is established -- installing a sink
    // before it would have that call overwrite the level a moment later.
    load_backends_once();

    const bool active = static_cast<bool>(sink);
    {
        std::lock_guard<std::mutex> lock(debug_mu());
        debug_sink() = std::move(sink);
        debug_partial().clear();
    }

    // **Registered only while a sink exists.** A callback that is
    // installed permanently and returns early swallows the trace rather
    // than passing it on: `rig_debug` writes to stderr *or* to the
    // callback, never both, so leaving ours in place would silently
    // break `SSTVAE_HAMLIB_DEBUG` for every desktop and test run.
    rig_set_debug_callback(active ? &debug_trampoline : nullptr, nullptr);
    rig_set_debug(active ? RIG_DEBUG_TRACE : env_debug_level());
}

}  // namespace sstvae::rig
