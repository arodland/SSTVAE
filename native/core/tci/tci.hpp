// TCI (ExpertSDR3's Transceiver Control Interface, protocol 2.1): the
// radio's audio, PTT and dial over one WebSocket, in place of a
// soundcard and Hamlib. Ported from Data2G's `app/tci.*`.
//
// Its own QThread and event loop, like the soundcard streams
// (audio/qt/qtaudio.hpp): the server's TX_CHRONO frames ask for transmit
// audio and are answered at once, whatever the GUI thread is doing.
//
// - receive: after READY, mono float32 at RATE; RX audio frames go
//   through an `audio::CapturePipeline` into whichever ring buffer is
//   attached (`set_capture`), resampled to FS.
// - transmit: `play` matches `tx::Player`. The waveform is resampled to
//   RATE up front and each TX_CHRONO is answered with as many samples. A
//   server that sends no TX_CHRONO (AetherSDR, per Decodium 1.0.542) is
//   fed on our clock instead, from half a second into the over (at once
//   if none was ever seen).
// - PTT: `trx:0,true,tci;` / `trx:0,false;`.
// - dial: the server's `vfo:0,0,Hz` notifications, unasked.
//
// Transceiver 0, channel 0 only. A lost connection is retried every 2 s.
//
// **Created once and never destroyed while the app runs**, like
// `RigController`: the transmit engine holds `set_ptt` and `play` for a
// whole over, so a settings change reconnects in place (`start`) rather
// than replacing the object under it.

#ifndef SSTVAE_TCI_TCI_HPP
#define SSTVAE_TCI_TCI_HPP

#include <QThread>

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace sstvae::rx {
class RingBuffer;
}

namespace sstvae::tci {

// The rate asked of the server: every TCI server has it, and it is a
// multiple of FS.
inline constexpr int RATE = 48000;
inline constexpr int DEFAULT_PORT = 50001;  // ExpertSDR3's

// "HOST" or "HOST:PORT" (DEFAULT_PORT when absent); nothing if not one.
std::optional<std::pair<std::string, int>> parse_address(const std::string& s);

class Tci {
public:
    // `on_status(text, error)`: connected, lost, refused -- from the
    // worker thread.
    using Status = std::function<void(const std::string&, bool error)>;

    explicit Tci(Status on_status = {});
    ~Tci();  // disconnects, the thread joined
    Tci(const Tci&) = delete;
    Tci& operator=(const Tci&) = delete;

    // (Re)connect to `host:port`, in the background. `modulation` is set
    // after READY ("usb", "digu"); empty leaves the operator's.
    void start(const std::string& host, int port, const std::string& modulation);
    void stop();

    // Where RX audio goes; nullptr detaches. Kept across reconnects.
    void set_capture(std::shared_ptr<rx::RingBuffer> ring);

    // A tx::Ptt. Throws std::runtime_error when not connected.
    void set_ptt(bool on);

    // A tx::Player (no device: the server is the device). Blocks until
    // the waveform has been handed to the server, `should_stop` says so,
    // or the connection is lost.
    bool play(std::span<const double> samples, int samplerate,
              const std::function<void(double)>& on_progress,
              const std::function<bool()>& should_stop,
              const std::function<void(const std::string&)>& on_error);

    // The dial the server last reported, or nothing.
    std::optional<double> frequency_hz() const;

    struct Worker;

private:
    QThread thread_;
    Worker* w_;  // lives on thread_
};

}  // namespace sstvae::tci

#endif
