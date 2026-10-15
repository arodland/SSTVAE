// Sending a picture as QRSSTVAE from the desktop app.
//
// The transmitter runs in process: the C++ port of qrss_encode.py and
// qrss_transmit.py (qrss/api.hpp, qrss/codec/api.hpp), whose audio is
// bit-exact against the scripts. SSTVAE_QRSS_BACKEND=python runs the two
// scripts instead, as the app did before the port, for comparing the two;
// the audio either way is keyed through the app's own transmitter
// (`TxEngine`), which keeps its PTT guarantees: a scope guard and an
// independent watchdog per pass.
//
// A picture is encoded once into a stored picture. Mode A sends one
// 50,600-latent group in one pass, B two and C three, one pass per half
// hour: a pass lasts 1782.7 s from t0 = quarter hour + 1 s, and its audio
// starts 12 s before t0, so one pass's audio ends about 2 s before the
// next one's starts. So the first pass's audio is made before it is
// keyed, and each later pass's is made while the one before it plays. A
// scheduled send (gui/qrss_schedule.hpp) hands over its own list of
// passes, which can repeat the picture back to back for hours; making
// each pass during the one before keeps that to two passes' audio in
// memory at a time.
// Each pass carries the callsign in its own Morse ID windows (spec
// 2.7), so the SSTVAE CW ID is not added.
//
// A different picture can follow straight on (`Chain`): it joins the
// run on the air, is encoded while the pass before it plays, and its
// first pass keys about 2 s after the last one ends, exactly as a
// run's own passes do. There is no time to start a fresh run there.

#ifndef SSTVAE_GUI_QRSS_TX_HPP
#define SSTVAE_GUI_QRSS_TX_HPP

#include <QString>
#include <QStringList>

#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "images/types.hpp"
#include "qrss/api.hpp"
#include "tx/engine.hpp"

namespace sstvae::gui::qrss_tx {

// From the transmitter's own constants (qrss/config.hpp, generated from
// the Python reference), never typed by hand.
inline constexpr double PASS_S = qrss::api::PASS_S;              // FULL frame, t0 to the end
inline constexpr double AUDIO_LEAD_S = qrss::api::AUDIO_LEAD_S;  // the audio starts this long before t0
inline constexpr double AUDIO_TAIL_S = qrss::api::AUDIO_TAIL_S;  // and ends this long after the frame
inline constexpr double T0_OFFSET_S = qrss::api::T0_OFFSET_S;    // t0 = quarter hour + 1 s
inline constexpr double PASS_SPACING_S = 2.0 * qrss::api::SLOT_S;
inline constexpr double PREP_S = 60.0;          // make the first pass's audio this long ahead
inline constexpr double FREQ_MIN_HZ = qrss::api::CARRIER_MIN_HZ;
inline constexpr double FREQ_MAX_HZ = qrss::api::CARRIER_MAX_HZ;
inline constexpr double FREQ_DEFAULT_HZ = qrss::api::CARRIER_DEFAULT_HZ;

// The two QRSSTVAE waveforms (docs/qrss/waveform-l.md). "CE": constant
// envelope, one latent per symbol, about 50 Hz wide. "L": linear, two
// latents per symbol plus a pilot, about 22 Hz wide; needs a linear
// transmit chain with the ALC out of it, and receivers that hear L.
inline constexpr const char* WAVEFORM_CE = "CE";
inline constexpr const char* WAVEFORM_L = "L";
bool known_waveform(const std::string& waveform);

// Passes a QRSS mode sends ("A" 1, "B" 2, "C" 3); 0 for anything else.
int passes_for(const std::string& mode);

// Total airtime label minutes for a mode: 30, 60, 90.
int minutes_for(const std::string& mode);

// The quarter hours (unix seconds) of a send's passes, started at `now`:
// the first whose audio starts at least `prep_s` from now, then one
// every half hour.
std::vector<double> plan(double now, int passes, double prep_s = PREP_S);

// When a pass's audio starts, and when it ends, for its quarter hour.
double audio_start(double slot);
double audio_end(double slot);

// One pass on the air: its quarter hour and which of the picture's
// latent groups it carries (0 .. passes_for(mode) - 1).
struct Pass {
    double slot = 0.0;
    int segment = 0;
};

// "2026-10-10T03:00Z" for a quarter hour.
QString slot_iso(double slot);

// Which transmitter makes the audio: the C++ one, unless
// SSTVAE_QRSS_BACKEND is "python".
enum class Backend { Native, Python };
Backend backend();
const char* backend_name(Backend b);

// Where the QRSS scripts are (the python backend only): $SSTVAE_QRSS_DIR, else the directory
// holding qrss_transmit.py, looked for beside the executable and up to
// four directories above it. Python is $SSTVAE_QRSS_PYTHON, else that
// directory's .venv, else python3.
struct Tools {
    QString python;
    QString dir;
};
std::optional<Tools> find_tools();

// A picture that follows the run's passes straight on, in its own mode
// and on its own carrier.
struct Part {
    images::Picture picture;
    std::string mode = "A";
    std::string waveform = WAVEFORM_CE;
    double freq_hz = FREQ_DEFAULT_HZ;
    std::vector<Pass> passes;   // the first a half hour (or more) after the run's last
};

// Pictures to add to a run while it is on the air. `append` is called
// from any thread; the run takes what is waiting when it keys its last
// pass, and from then on (or once the run has ended) refuses more, so a
// part is either sent or refused, never silently dropped.
class Chain {
public:
    // False when the run has already keyed its last pass or ended.
    bool append(Part part);
    // For `run`: the next part waiting, else nothing, and the chain closes.
    std::optional<Part> take_or_close();
    void close();

private:
    std::mutex mutex_;
    std::deque<Part> parts_;
    bool closed_ = false;
};

struct Request {
    images::Picture picture;
    std::string mode = "A";            // A, B or C
    std::string waveform = WAVEFORM_CE;  // CE or L
    double freq_hz = FREQ_DEFAULT_HZ;
    std::string callsign;
    std::string grid;                  // used only when it is a 4-character locator
    tx::TxConfig tx;                   // device, level, PTT timing
    double prep_s = PREP_S;            // start making the first pass's audio this long ahead
    // The passes to send, in order. Empty: the mode's passes, from the
    // first quarter hour `plan` finds once the picture is encoded.
    std::vector<Pass> passes;
    // Pictures that follow straight on; optional.
    std::shared_ptr<Chain> chain;
};

// The command lines `run` executes, for tests and the log.
QStringList encode_args(const Tools& tools, const QString& png, const QString& qrsp,
                        const std::string& mode);
QStringList transmit_args(const Tools& tools, const QString& qrsp, const QString& wav,
                          double slot, int segment, const Request& request);

// Empty when the request can go out; otherwise why not, for the operator.
std::string problem(const Request& request);
// The same for a part following a run whose last pass is at `after`.
std::string problem(const Part& part, double after);

// qrss_transmit.py's arguments for one pass of a request, for the C++
// transmitter: the slot, waveform, carrier (to 0.1 Hz, as the script's
// --freq is given), callsign in capitals, grid when it is a locator, and
// the pass's group.
qrss::api::PassRequest pass_request(const Request& request, const Pass& pass);

using Clock = std::function<double()>;               // unix seconds
using Runner = std::function<bool(const QStringList& argv, QString* output)>;

// What makes a run's audio. `run` calls these from its worker thread and
// from one helper thread, never both at once. Picture `index` is the
// run's index-th picture (0 the request's, then the chain's).
class Maker {
public:
    virtual ~Maker() = default;
    // Encode the picture once; empty on success, else why not.
    virtual std::string encode(std::size_t index, const Request& picture) = 0;
    // One pass of an encoded picture: float samples at 8 kHz, as the
    // transmitter writes them (before the output level); empty with
    // `*error` set when it cannot be made.
    virtual std::vector<double> make(std::size_t index, const Request& picture, const Pass& pass,
                                     std::string* error) = 0;
};

// Test / Tune: a steady `hz` sine for `seconds` at 8 kHz, peak 1, with
// 10 ms raised-cosine ends so keying it on and off does not click.
// Whole Hz only: the phase is reduced exactly, (n * hz) mod 8000.
std::vector<double> test_tone(double seconds, int hz = 1500);

// The C++ transmitter. The encoder (the codec's, from the model cache)
// is opened on the first encode and kept for the life of the app.
std::unique_ptr<Maker> native_maker();
// The scripts, through `runner` (a QProcess that cancelling the engine
// kills, when empty).
std::unique_ptr<Maker> script_maker(tx::TxEngine& engine, const Tools& tools, Runner runner = {});

// The whole sequence, on the calling (worker) thread: encode, make the
// first pass's audio, then key each pass on its start while the next
// one's audio is made. True when every pass was sent. Errors go to
// `on_error`; cancelling the engine stops it at the next wait or
// mid-pass.
bool run(tx::TxEngine& engine, const Request& request, Maker& maker,
         const std::function<void(const std::string&)>& on_error, Clock clock = {});
// The same with the scripts.
bool run(tx::TxEngine& engine, const Request& request, const Tools& tools,
         const std::function<void(const std::string&)>& on_error,
         Clock clock = {}, Runner runner = {});

}  // namespace sstvae::gui::qrss_tx

#endif
