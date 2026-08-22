// A rig reached through a `SerialTransport` instead of a serial port.
//
// This is the piece that makes `docs/android.md`'s "no CAT on Android"
// wrong. It composes three things that each already work:
//
//   * a `SerialTransport` -- USB or Bluetooth, from Java (transport.hpp)
//   * a `LoopbackBridge` -- that transport as a socket (bridge.hpp)
//   * whatever `RigBackend` the caller's factory builds, pointed at
//     that socket's address rather than at a device path
//
// The result is an ordinary `RigBackend`, so `RigController` -- one
// worker, PTT as priority work, `stop()` that detaches -- is used
// unchanged, and so is every radio Hamlib knows.
//
// **Two jobs, and only two.** Hand the inner backend a loopback
// endpoint in place of a device, and key DTR/RTS on the transport
// rather than through Hamlib. Everything else about a rig -- the model,
// the timeouts, reading frequency, CAT PTT, putting the radio into
// PKT-USB on connect -- goes straight through, because it is all just
// bytes on the link and the link is the only thing that changed.
//
// **The factory is a seam for the same reason `rx::Decoder` is.** It
// keeps this file in `sstvae_core`, so the composition, the PTT
// routing and the no-bridge shortcut are all tested in a build with no
// libhamlib at all (`native/tests/test_rig_bridged.cpp`, against a stub
// inner backend). `native/tests/test_rig_hamlib.cpp` is the complement
// that holds the real thing.

#ifndef SSTVAE_RIG_BRIDGED_HPP
#define SSTVAE_RIG_BRIDGED_HPP

#include <functional>
#include <memory>
#include <string>

#include "rig/backend.hpp"
#include "rig/bridge.hpp"
#include "rig/hamlib.hpp"
#include "rig/transport.hpp"

namespace sstvae::rig {

// Builds the backend that actually talks to Hamlib, given a config
// whose `device` has been rewritten to point wherever it should.
//
// `rig/hamlib.hpp` is included above for `HamlibConfig` alone -- that
// header declares plain structs and three free functions and pulls in
// nothing from libhamlib, which is what lets `sstvae_core` name the
// configuration type without linking the implementation of it.
using BackendFactory =
    std::function<std::unique_ptr<RigBackend>(const HamlibConfig&)>;

// Wrap `transport` and hand the result to `make_inner`.
//
// **The presence of a transport is the decision, not the shape of the
// device string.** The first draft sniffed `config.device` with
// `is_network_device` and branched on that, which is wrong in a way
// worth recording: an app-level device identifier like
// `usb:1a86:7523` contains no slash and does not start with `com`, so
// Hamlib's own rules read it as a *hostname* -- and the bridge was
// skipped for exactly the devices it exists to serve. The caller
// already knows which the operator picked, because a picker offered
// them the choice; a heuristic can only guess, and has to guess on a
// string that was never a device path to begin with.
//
// So: no transport means the operator chose a network rig, and Hamlib
// dials `config.device` itself -- one code path covering both model 2
// (NET rigctl, sharing a radio with a station PC) and a native backend
// pointed at a ser2net-style server, because in Hamlib they *are* one
// code path. A transport means the operator chose a USB or Bluetooth
// device, and `config.device` is replaced by the bridge's loopback
// address.
//
// `is_network_device` survives as the *validation* of that choice: with
// no transport, a device Hamlib will not read as a host is refused here
// rather than becoming a failed connection later.
//
// Throws RigError for a combination that cannot work, at build time
// rather than at `open()`, because it is a programming error rather
// than an operating one.
std::unique_ptr<RigBackend> make_bridged_backend(
    HamlibConfig config, std::shared_ptr<SerialTransport> transport,
    BackendFactory make_inner);

// Whether keying this configuration drives a control line rather than
// sending a CAT command. Exposed because the app has to know: it is the
// difference between a PTT that works over Bluetooth and one that
// cannot.
inline bool ptt_uses_control_line(PttMethod method) {
    return method == PttMethod::Dtr || method == PttMethod::Rts;
}

}  // namespace sstvae::rig

#endif
