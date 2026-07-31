// Soundcard capture and playback through QtMultimedia.
//
// The thin part. Everything with logic in it -- rate conversion, sample
// format conversion, device matching -- is in `core/audio/audio.hpp`,
// Qt-free and tested against a fake device, because that is where every
// audio bug this project has had actually lived. What is left here is
// device enumeration and moving bytes, which needs hardware to test and
// so is kept as small as possible.
//
// **Why QtMultimedia and not PortAudio** (docs/native-app.md): the
// Python app's worst bug was that its PortAudio callback is a *Python*
// function on the host's realtime thread, so it needs the GIL, and a Qt
// thread painting a 640x480 preview after every decode poll stopped it
// running. On JACK -- a couple of milliseconds of period with nothing
// queued -- the audio was silently skipped: 200-350 samples per poll,
// 5 dB of SNR, a mangled picture, while sync succeeded and every frame
// was reported. PortAudio's blocking API would have fixed it, but
// `stream.read()` corrupts the heap on its JACK backend.
//
// C++ has no GIL, so that specific bug cannot recur here and the choice
// is on merit: `QAudioSource` is pull-based over a buffered QIODevice,
// one fewer native dependency, and `QMediaDevices` gives hot-plug and
// default-change notifications PortAudio does not.
//
// This is an optional library (`SSTVAE_BUILD_QTAUDIO`). The modem, the
// codec and both engines build and test with no Qt present at all --
// the engines take their player as a seam precisely so that stays true.

#ifndef SSTVAE_AUDIO_QT_QTAUDIO_HPP
#define SSTVAE_AUDIO_QT_QTAUDIO_HPP

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "audio/audio.hpp"
#include "config.hpp"
#include "rx/ringbuffer.hpp"

namespace sstvae::audio::qt {

// How much audio Qt buffers for us. This slack is what makes the design
// work: we can be this far late in draining without losing a sample.
//
// Measured in the Python app on a K4 RX A with a thread deliberately
// blocking: clean through 400 ms at 1.0 s of buffer, losing samples at
// 800 ms. 2 s is generous rather than tight and costs 384 KB. Nothing
// here is latency-sensitive -- the decode loop polls every 5 seconds.
inline constexpr double BUFFER_SECONDS = 2.0;

using Report = std::function<void(const std::string&)>;

// Human-readable device descriptions, in the backend's order. These are
// what `audio::match_device` matches against and what the config file
// stores -- see its comment on why not the opaque device id.
std::vector<std::string> input_device_names();
std::vector<std::string> output_device_names();

std::string default_input_name();
std::string default_output_name();

// Capture into a ring buffer.
//
// **Runs its own thread with its own event loop**, and the QAudioSource
// lives on it. The reference lets the source live on the GUI thread and
// drains it from there, which is safe in Python only because Qt's own
// backend does the realtime work. Giving it a dedicated thread here
// costs a few lines and means a busy GUI thread cannot delay the drain
// at all -- the same hazard that cost 5 dB in the PortAudio version,
// removed structurally rather than absorbed by the buffer.
//
// Opens at the *device's* rate and resamples through
// `audio::StreamResampler`; `samplerate` is what lands in the ring, and
// the modem fixes it at FS. It is not a device setting: passing anything
// else fills the ring with wrong-rate audio that decodes to nothing.
class InputStream {
public:
    // `device_name` empty means the system default. Throws if capture
    // cannot be started.
    InputStream(const std::string& device_name, rx::RingBuffer& ring,
                int samplerate = config::FS, Report on_error = {});
    ~InputStream();

    InputStream(const InputStream&) = delete;
    InputStream& operator=(const InputStream&) = delete;

    void stop();

    // The rate the device is actually running at, which is usually not
    // `samplerate`.
    int device_rate() const;
    std::string device_name() const;
    std::uint64_t samples_captured() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Play a waveform, blocking until it has finished or been stopped.
// Matches `tx::Player`, so it drops into `TxEngine`'s seam unchanged --
// which means the PTT guarantee is unaffected by which player is in use.
//
// **Runs its own QThread with its own event loop**, like InputStream.
// Push mode with explicit writes paced on `bytesFree`, same as before,
// but on Windows the WASAPI backend still needs an event loop pumped on
// the owning thread to actually move written bytes to the device --
// without one, write() queues into the sink's buffer and nothing drains,
// which reads as total silence until the stream is torn down and the
// buffered tail escapes in one burst. TxEngine calls this from a plain
// std::thread with no event loop, which is exactly the thread that hit
// this; PulseAudio/PipeWire/CoreAudio service the stream from their own
// native thread and never showed it.
//
// Resamples up front rather than per chunk. Unlike capture the whole
// waveform is in hand, so one clean conversion avoids polyphase edge
// effects entirely -- and this is not hypothetical: a K4's transmit
// device advertises 12 kHz.
bool play(const std::string& device_name, std::span<const double> samples,
          int samplerate = config::FS,
          const std::function<void(double)>& on_progress = {},
          const std::function<bool()>& should_stop = {},
          const Report& on_error = {});

}  // namespace sstvae::audio::qt

#endif
