// The listening session, and the only thing that owns one.
//
// **This inverts the desktop's `AppState`** (docs/android.md). There,
// the window owns the engine and the engine dies with it, which is
// right for a machine whose app is either on screen or not running. A
// phone's listening session has to survive the screen going off, the
// activity being destroyed on rotation, and the app being swiped away,
// so ownership cannot sit anywhere the UI can take with it when it
// goes.
//
// So: the session is process-wide, `ListenerService` guarantees the
// process stays alive around it, and every UI is a *view* that attaches
// and detaches. Nothing outside here holds a ring buffer, a stream or
// an engine thread, and nothing outside here caches what they say --
// `progress()` is read on demand, so a view that was not attached when
// something happened is in exactly the same position as one that was.
//
// The corollary is the battery answer: with no view attached nothing
// renders at all, because there is nothing accumulating for a view to
// render later.

#ifndef SSTVAE_ANDROID_SESSION_HPP
#define SSTVAE_ANDROID_SESSION_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "audio/android/androidaudio.hpp"
#include "codec/codec.hpp"
#include "rx/engine.hpp"
#include "rx/ringbuffer.hpp"

namespace sstvae::androidapp {

// What the model is doing, for a UI that must say so without guessing.
// `Downloading` is a distinct state rather than a flavour of `Loading`
// because it is the one that can take minutes on a phone's connection,
// and the one where "nothing is happening" is the wrong conclusion.
enum class ModelState { Absent, Downloading, Loading, Ready, Failed };

class Session {
public:
    // The one instance. Deliberately not injectable: two sessions would
    // mean two claims on the microphone and two foreground-service
    // lifetimes, and there is no arrangement of this app in which that
    // is what anyone wants.
    static Session& instance();

    // `device_name` empty means the platform default. Returns false and
    // sets `last_error()` if capture could not be started; already
    // running is a no-op returning true.
    bool start(const std::string& device_name);
    void stop();

    bool running() const;

    // A snapshot, not a reference: the engine thread writes this
    // continuously, so handing out anything else would be handing out a
    // race.
    rx::Progress progress() const;

    // Empty unless capture went wrong. Sticky until the next `start`,
    // because the desktop proved that an error written where the next
    // message overwrites it is an error nobody reads.
    std::string last_error() const;

    // Fetch and load the decoder, off the calling thread. Safe to call
    // before, during or after a session: **listening never waits on the
    // model.** The engine reports mode, frames, callsign and SNR from
    // the moment capture starts, and pictures begin appearing whenever
    // the decoder arrives -- which on a first run means a download, and
    // a receiver that refused to listen until it finished would be
    // useless exactly when someone is trying it out.
    void load_model_async();

    // Where finished receptions are written, as a PNG plus a JSON
    // sidecar. Empty (the default) means nothing is saved.
    void set_picture_dir(std::string dir);

    // The path of a reception saved since this was last called, once.
    // **Consuming**, because the caller is the notification poller and
    // a non-consuming read would repost the same picture every tick.
    std::optional<std::string> take_saved_picture();

    // The metadata line for that reception, for the notification's
    // text. Read alongside the path and not from live state, which the
    // engine wipes two seconds later -- the same reason the sidecar
    // exists.
    std::string last_saved_summary() const;

    ModelState model_state() const;
    std::string model_error() const;
    // Bytes received / total for the current download, both 0 when not
    // downloading. `total` is -1 when the server does not say.
    std::pair<std::int64_t, std::int64_t> model_progress() const;

    // Device-level facts, empty when not running. Read from the stream
    // rather than remembered, for the same reason as `progress()`.
    int device_rate() const;
    std::string routed_device() const;
    std::string routing_warning() const;
    double peak_level() const;
    double near_zero_fraction() const;
    double capture_drift_ppm() const;

    // Whether the notification line carries diagnostics. Atomic rather
    // than under `mu_`: it is set from the GUI thread and read by the
    // service's handler, and it must not be able to wait behind a
    // session that is starting or stopping.
    void set_show_technical(bool on) { show_technical_.store(on); }
    bool show_technical() const { return show_technical_.load(); }

    // The most recent `n` samples, for the waterfall. `tail` rather
    // than `snapshot` deliberately: snapshot copies the whole 130 s
    // buffer, and doing that at display rates is how the desktop tore
    // holes in its own audio.
    std::vector<double> audio_tail(std::size_t n) const;

private:
    Session() = default;
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    std::optional<std::string> save_reception(const rx::Reception& r);

    std::atomic<bool> show_technical_{false};

    // Guards the members below against a view thread reading while the
    // UI thread starts or stops. It is never held across a decode: the
    // engine thread touches `ring_` and `state_`, which have their own
    // synchronisation, and not these pointers.
    mutable std::mutex mu_;
    std::unique_ptr<rx::RingBuffer> ring_;
    std::unique_ptr<audio::android::InputStream> stream_;
    std::unique_ptr<rx::SharedState> state_;
    std::unique_ptr<rx::StopFlag> stop_;
    std::thread thread_;
    std::string error_;
    std::string picture_dir_;
    std::optional<std::string> saved_picture_;
    std::string saved_summary_;

    // Separate lock from `mu_`: the engine thread reads `codec_` on
    // every decode and the model thread writes it once, and neither has
    // any business waiting on a start/stop.
    mutable std::mutex model_mu_;
    // `shared_ptr` so the decode lambda can take a copy and use it
    // outside the lock. Nothing replaces a loaded codec today, but a
    // reload would otherwise be free to destroy one mid-inference.
    std::shared_ptr<codec::OnnxCodec> codec_;
    ModelState model_state_ = ModelState::Absent;
    std::string model_error_;
    std::int64_t model_received_ = 0;
    std::int64_t model_total_ = 0;
    std::thread model_thread_;
};

}  // namespace sstvae::androidapp

#endif
