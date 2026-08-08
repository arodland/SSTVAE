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

#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "audio/android/androidaudio.hpp"
#include "rx/engine.hpp"
#include "rx/ringbuffer.hpp"

namespace sstvae::androidapp {

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

    // Device-level facts, empty when not running. Read from the stream
    // rather than remembered, for the same reason as `progress()`.
    int device_rate() const;
    std::string routed_device() const;
    std::string routing_warning() const;
    double peak_level() const;
    double near_zero_fraction() const;

private:
    Session() = default;
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

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
};

}  // namespace sstvae::androidapp

#endif
