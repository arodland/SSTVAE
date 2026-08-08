// Soundcard capture and playback on Android.
//
// **Entry point for entry point the same surface as `core/audio/qt/`**,
// so `InputStream` and `play()` drop into the engines' existing seams
// with nothing above this line changed. Seven functions is the whole
// layer; everything with logic in it -- `bytes_to_mono`, the sample
// format conversions, `resample_ratio`, `StreamResampler`,
// `CapturePipeline`, `match_device` -- stays in `core/audio/audio.hpp`,
// Qt-free, device-free and tested against bytes rather than hardware.
// That split is the design, not an accident of porting: every audio bug
// this project has had lived in the conversions and not in the code
// talking to the driver.
//
// **Why Java `AudioRecord` and not AAudio/Oboe** (docs/android.md):
// there is no latency requirement here at all -- 2 s of buffer, 5 s
// decode polls -- so AAudio's one real benefit is worth nothing, while
// its `setDeviceId` is silently ignored on the OpenSL ES fallback and
// USB capture through it has open glitch reports. Enumeration needs
// Java either way. And a blocking read on a plain thread is the
// architecture the desktop wanted and could not have, PortAudio's
// blocking API having corrupted the heap on JACK.
//
// **The JNI direction rule.** Java calls into C++ on the data path and
// never the reverse: the reader thread is already attached, so pushing a
// chunk is a plain call. C++ calls *into* Java only for control --
// enumerate, open, close -- which is rare, off the audio path, and
// attaches explicitly. Getting this backwards would put an
// AttachCurrentThread on every buffer.

#ifndef SSTVAE_AUDIO_ANDROID_ANDROIDAUDIO_HPP
#define SSTVAE_AUDIO_ANDROID_ANDROIDAUDIO_HPP

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "audio/audio.hpp"
#include "config.hpp"
#include "rx/ringbuffer.hpp"

namespace sstvae::audio::android {

// How much audio the Java side buffers ahead of us, matching the Qt
// layer's 2 s. Slack rather than latency: nothing here is
// latency-sensitive, and this is what makes a late drain harmless.
inline constexpr double BUFFER_SECONDS = 2.0;

using Report = std::function<void(const std::string&)>;

// Must be called once before anything else, from `JNI_OnLoad` or an
// equivalent. Without it every call here throws: there is no ambient
// way to find the VM from a thread we did not create.
struct JavaVM_;
void set_java_vm(JavaVM_* vm);
bool ready();

// Human-readable device descriptions, in the platform's order. These are
// what `audio::match_device` matches against and what the config file
// stores -- the same reasoning as the desktop's, and the same accepted
// ambiguity when two identical interfaces are attached.
std::vector<std::string> input_device_names();
std::vector<std::string> output_device_names();

std::string default_input_name();
std::string default_output_name();

// Capture into a ring buffer.
//
// Opens at the *device's* rate and resamples through
// `audio::CapturePipeline`; `samplerate` is what lands in the ring and
// the modem fixes it at FS. It is not a device setting -- passing
// anything else fills the ring with wrong-rate audio that decodes to
// nothing.
//
// Two callbacks, because there are two kinds of message, and collapsing
// them cost the desktop twice. `on_opened` reports what was actually
// obtained -- the device, its rate, whether our resampler is in the path
// -- which is information, since almost nothing is natively 8 kHz and
// reporting that as a failure told operators their working setup was
// broken. `on_error` is for capture going wrong once running, which
// deserves a sticky banner.
class InputStream {
public:
    // `device_name` empty means the platform default. Throws if capture
    // cannot be started.
    InputStream(const std::string& device_name, rx::RingBuffer& ring,
                int samplerate = config::FS, Report on_opened = {},
                Report on_error = {});
    ~InputStream();

    InputStream(const InputStream&) = delete;
    InputStream& operator=(const InputStream&) = delete;

    void stop();

    // The rate the device is actually running at, usually not
    // `samplerate`.
    int device_rate() const;

    // Where capture actually landed, which is **not** necessarily what
    // was asked for. Read after the stream is running: before that there
    // is no routing to report, and `setPreferredDevice`'s return value
    // is not an answer -- measured against a TH-D75 over USB it returned
    // false while the routing had plainly taken effect.
    std::string routed_device() const;

    // Empty unless capture landed somewhere other than the requested
    // device. Not "did the call succeed" but "is the audio coming from
    // where the operator thinks": capturing from the built-in mic while
    // the UI claims USB is the worst outcome available here.
    std::string routing_warning() const;

    std::uint64_t samples_captured() const;

    // Peak sample magnitude and the share of near-silent samples over
    // the last reporting window. **Both**, because they answer different
    // questions: a path that is merely quiet and a path delivering
    // silence have the same mean level and are not the same failure --
    // which is how an emulator's zeroed audio got diagnosed twice as
    // something else.
    double peak_level() const;
    double near_zero_fraction() const;

    // Public because the JNI callbacks at the bottom of
    // androidaudio.cpp have to reach it, and they are free functions
    // with C linkage -- there is no `this` for them to be a friend of.
    // Opaque either way: the definition is in the .cpp.
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// Play a waveform, blocking until it has finished or been stopped.
// Matches `tx::Player`, so it drops into `TxEngine`'s seam unchanged and
// the PTT guarantee is unaffected by which player is in use.
//
// Resamples up front rather than per chunk: unlike capture the whole
// waveform is in hand, so one clean conversion avoids polyphase edge
// effects entirely.
bool play(const std::string& device_name, std::span<const double> samples,
          int samplerate = config::FS,
          const std::function<void(double)>& on_progress = {},
          const std::function<bool()>& should_stop = {},
          const Report& on_error = {});

}  // namespace sstvae::audio::android

#endif
