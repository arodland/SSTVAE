// Sending a picture as QRSSTVAE CE from the desktop app.
//
// The QRSSTVAE transmitter is Python (`qrss_encode.py`,
// `qrss_transmit.py`) and is not ported, so this runs the two scripts
// and keys their audio through the app's own transmitter (`TxEngine`),
// which keeps its PTT guarantees: a scope guard and an independent
// watchdog per pass.
//
// A picture is encoded once into a stored picture (.qrsp). Mode A sends
// one 50,600-latent group in one pass, B two and C three, one pass per
// half hour: a pass lasts 1782.7 s from t0 = quarter hour + 1 s, and its
// audio (qrss_transmit.py's WAV) starts 12 s before t0. Each pass is
// made a minute before its quarter hour, and carries the callsign in
// its own Morse ID windows (spec 2.7), so the SSTVAE CW ID is not added.

#ifndef SSTVAE_GUI_QRSS_TX_HPP
#define SSTVAE_GUI_QRSS_TX_HPP

#include <QString>
#include <QStringList>

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "images/types.hpp"
#include "tx/engine.hpp"

namespace sstvae::gui::qrss_tx {

inline constexpr double PASS_S = 1782.7;        // FULL frame, t0 to the end
inline constexpr double AUDIO_LEAD_S = 12.0;    // the WAV starts this long before t0
inline constexpr double T0_OFFSET_S = 1.0;      // t0 = quarter hour + 1 s
inline constexpr double PASS_SPACING_S = 1800.0;
inline constexpr double PREP_S = 60.0;          // make a pass's audio this long ahead
inline constexpr double FREQ_MIN_HZ = 300.0;    // CARRIER_BAND_HZ
inline constexpr double FREQ_MAX_HZ = 2700.0;
inline constexpr double FREQ_DEFAULT_HZ = 1500.0;

// Passes a QRSS mode sends ("A" 1, "B" 2, "C" 3); 0 for anything else.
int passes_for(const std::string& mode);

// Total airtime label minutes for a mode: 30, 60, 90.
int minutes_for(const std::string& mode);

// The quarter hours (unix seconds) of a send's passes, started at `now`:
// the first whose audio starts at least `prep_s` from now, then one
// every half hour.
std::vector<double> plan(double now, int passes, double prep_s = PREP_S);

// "2026-10-10T03:00Z" for a quarter hour.
QString slot_iso(double slot);

// Where the QRSS scripts are: $SSTVAE_QRSS_DIR, else the directory
// holding qrss_transmit.py, looked for beside the executable and up to
// four directories above it. Python is $SSTVAE_QRSS_PYTHON, else that
// directory's .venv, else python3.
struct Tools {
    QString python;
    QString dir;
};
std::optional<Tools> find_tools();

struct Request {
    images::Picture picture;
    std::string mode = "A";            // A, B or C
    double freq_hz = FREQ_DEFAULT_HZ;
    std::string callsign;
    std::string grid;                  // used only when it is a 4-character locator
    tx::TxConfig tx;                   // device, level, PTT timing
    double prep_s = PREP_S;            // make each pass's audio this long ahead
};

// The command lines `run` executes, for tests and the log.
QStringList encode_args(const Tools& tools, const QString& png, const QString& qrsp,
                        const std::string& mode);
QStringList transmit_args(const Tools& tools, const QString& qrsp, const QString& wav,
                          double slot, int segment, const Request& request);

// Empty when the request can go out; otherwise why not, for the operator.
std::string problem(const Request& request);

using Clock = std::function<double()>;               // unix seconds
using Runner = std::function<bool(const QStringList& argv, QString* output)>;

// The whole sequence, on the calling (worker) thread: encode, then for
// each pass wait, make its audio, wait for its start and key it. True
// when every pass was sent. Errors go to `on_error`; cancelling the
// engine stops it at the next wait or mid-pass.
bool run(tx::TxEngine& engine, const Request& request, const Tools& tools,
         const std::function<void(const std::string&)>& on_error,
         Clock clock = {}, Runner runner = {});

}  // namespace sstvae::gui::qrss_tx

#endif
