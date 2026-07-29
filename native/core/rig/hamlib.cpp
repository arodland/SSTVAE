#include "rig/hamlib.hpp"


#include <algorithm>
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
        const char* debug = std::getenv("SSTVAE_HAMLIB_DEBUG");
        const bool verbose = debug != nullptr && *debug != '\0' &&
                             std::strcmp(debug, "0") != 0;
        rig_set_debug(verbose ? RIG_DEBUG_TRACE : RIG_DEBUG_NONE);
        rig_load_all_backends();
    });
}

std::string hamlib_error(const char* what, int code) {
    const char* text = rigerror(code);
    return std::string(what) + ": " + (text != nullptr ? text : "unknown error") +
           " (" + std::to_string(code) + ")";
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
        const int rc =
            rig_set_ptt(rig_, RIG_VFO_CURR, on ? RIG_PTT_ON : RIG_PTT_OFF);
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

}  // namespace sstvae::rig
