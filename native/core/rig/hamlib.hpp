// The real radio, through libhamlib linked in-process.
//
// `docs/native-app.md` ("Bundling Hamlib") is the decision record. In
// short: no child `rigctld`, no IPC, no nested binary to sign, no
// per-platform rigctld build in CI. The reference's socket architecture
// existed because the SWIG Hamlib bindings are in the system
// site-packages where a virtualenv cannot reach them -- a Python
// packaging constraint with no C++ equivalent.
//
// Two consequences worth stating plainly:
//
// **Sharing a radio still works.** Hamlib model 2, "Hamlib NET rigctl",
// is a backend that speaks the rigctld protocol as a *client*, so
// pointing at a `rigctld` the operator already runs for WSJT-X or
// fldigi is just another model in the same picker with `host:port` in
// place of a serial device. No port conflict, because we are a client;
// no second transport in the codebase.
//
// **A backend segfault now takes the app with it.** Accepted in the
// design doc: these backends are widely exercised across the ham
// software ecosystem, and a silently hung child process is arguably
// worse than a visible crash. Model 2 is the way back to isolation for
// anyone who wants it.
//
// Optional (`SSTVAE_BUILD_RIG`), like the codec and the Qt audio layer,
// so the modem and both engines still build with no libhamlib present.

#ifndef SSTVAE_RIG_HAMLIB_HPP
#define SSTVAE_RIG_HAMLIB_HPP

#include <memory>
#include <string>
#include <vector>

#include "rig/backend.hpp"
#include "rig/transport.hpp"

namespace sstvae::rig {

// The rigctld protocol as a client -- the "share the radio with other
// software" option. Named because it is referred to by number in the
// settings and in the docs, and a bare 2 in the code would be a puzzle.
inline constexpr int MODEL_NET_RIGCTL = 2;

// Hamlib's own dummy, which is what the reference's test recipe uses
// (`rigctld -m 1`) and what makes PTT and frequency exercisable with no
// radio attached.
inline constexpr int MODEL_DUMMY = 1;

struct RigModel {
    int model;
    std::string manufacturer;
    std::string name;
    std::string version;
    std::string status;  // Hamlib's backend status: Alpha, Beta, Stable...

    // "Elecraft K4 (Stable)" -- what a picker shows.
    std::string label() const;
};

// Every model this Hamlib knows, sorted by manufacturer then name.
//
// `rig_list_foreach` replaces parsing `rigctld -l`, which deletes the
// fixed-width column slicing CLAUDE.md flags as a trap: splitting on
// whitespace runs looks fine and silently drops rows, because fields
// contain single spaces ("N2ADR James Ahlstrom") and at least one Model
// fills its column exactly. Reading a struct cannot have that bug, and
// model 2 appears in the list for free.
std::vector<RigModel> list_models();

// How PTT is asserted. `Vox` is not "key by VOX" -- it is *do not key
// at all*, because the operator's audio is doing it; the caller must
// then hand the transmit engine nothing to key rather than something
// that fails.
enum class PttMethod { Vox, Cat, Dtr, Rts };

// Serial line settings. `Default` means do not set the token, leaving
// the backend's own value -- which is right for almost every rig, and
// is why it is a distinct choice rather than a guess at 8-N-1.
enum class DataBits { Default, Seven, Eight };
enum class StopBits { Default, One, Two };
enum class Parity { Default, None, Odd, Even };
enum class Handshake { Default, None, XonXoff, Hardware };
// Held for the life of the session: an interface that steals its power
// from the control lines needs them parked, not toggled.
enum class LineState { Default, High, Low };

// What to put the rig in when the session opens. `None` leaves whatever
// the operator has dialled in alone -- the safe default, since changing
// a stranger's rig mode on connect is a surprise.
enum class RigMode { None, Usb, PktUsb };

struct HamlibConfig {
    int model = MODEL_DUMMY;
    // Serial device, or "host:port" for MODEL_NET_RIGCTL.
    std::string device;
    int baud = 0;  // 0 = the backend's default

    DataBits data_bits = DataBits::Default;
    StopBits stop_bits = StopBits::Default;
    Parity parity = Parity::Default;
    Handshake handshake = Handshake::Default;
    LineState dtr = LineState::Default;
    LineState rts = LineState::Default;

    PttMethod ptt_method = PttMethod::Cat;
    // Empty means the CAT device. Often different: a serial adapter
    // whose control lines key the rig while CAT runs elsewhere.
    std::string ptt_device;

    RigMode mode = RigMode::None;

    // Set on the rig via `rig_set_conf` before opening.
    //
    // An improvement on the reference rather than a translation of it:
    // there, an app-side socket timeout and retry sit on top of
    // rigctld's own rig timeout and retry, and only the outer pair is
    // under our control. Here there is one of each and we set it. This
    // is load-bearing for `RigController::stop()`, whose abandoned
    // worker only exits once the in-flight call gives up.
    int timeout_ms = 1000;
    int retries = 1;
};

std::unique_ptr<RigBackend> make_hamlib_backend(const HamlibConfig& config);

// What this backend would have set a serial port to, had it been given
// one.
//
// Read straight out of `struct rig_caps`, which is where `rig_init`
// takes them from -- including the deliberate choice of
// `serial_rate_max` ("fastest!") as the default rate. It exists for the
// bridged path, where Hamlib never touches the hardware and so cannot
// apply them itself; see `SerialParams` in `rig/transport.hpp`.
//
// `rig_caps` is the one Hamlib struct this project reads through, for
// the reason `description()` records: it has no pthread members, so the
// Windows shim's type sizes cannot silently misplace its fields.
// Falls back to 9600 8-N-1 for a model Hamlib does not know.
SerialDefaults serial_defaults(int model);

// Hamlib's version string, for an about box or a bug report.
std::string hamlib_version();

}  // namespace sstvae::rig

#endif
