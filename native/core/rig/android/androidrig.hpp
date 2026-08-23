// USB and Bluetooth serial on Android, as a `rig::SerialTransport`.
//
// The Java half of this is `java/org/cleverdomain/sstvae/SerialBridge.java`;
// this is the JNI shim in front of it. It is a **separate library**
// (`SSTVAE_BUILD_ANDROIDRIG`), for the same reason `core/audio/qt/` and
// `core/audio/android/` are: everything with logic in it -- the bridge,
// the composition, the PTT routing, the line-setting resolution --
// lives in `sstvae_core` and is tested against fakes on a desktop, and
// what is left here is only the code that talks to the platform.
//
// **Why Java at all.** Android hands an unprivileged app no
// `/dev/ttyUSB0`; USB serial goes through the USB host API, and the
// standard library for the chip-level protocols is
// `usb-serial-for-android` (MIT, FTDI/CP210x/CH34x/PL2303/CDC-ACM).
// Bluetooth is an RFCOMM socket on the Serial Port Profile UUID, which
// is also Java-only. There is no C API under either of them to reach
// for instead.
//
// **Bonded Bluetooth devices only, deliberately.** Listing paired
// devices needs `BLUETOOTH_CONNECT` and nothing else; *discovering*
// them needs `BLUETOOTH_SCAN` and, before API 31, location permission,
// which is a large ask for a screen whose whole job is to show the
// radio the operator already paired in Settings. Pairing is the
// system's job and it does it better.
//
// **The direction rule is the opposite of the audio layer's**, and that
// is not an inconsistency. Audio is a stream with a thread pushing it,
// so Java calls into C++ and never waits. A rig link is request and
// response driven by `LoopbackBridge`'s pump threads, so C++ calls into
// Java -- which means those threads must be attached, and attaching and
// detaching per call would be two JNI transitions per idle wakeup.
// `androidrig.cpp` attaches each calling thread once and detaches when
// it exits.

#ifndef SSTVAE_RIG_ANDROID_ANDROIDRIG_HPP
#define SSTVAE_RIG_ANDROID_ANDROIDRIG_HPP

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rig/transport.hpp"

namespace sstvae::rig::android {

// Must be called once from the UI thread before anything else here,
// exactly like `audio::android::set_java_vm` and for the same reason:
// `FindClass` from a thread we created cannot see an application class,
// so the class reference has to be taken on a thread that has the app's
// loader on its stack.
struct JavaVM_;
void set_java_vm(JavaVM_* vm);
bool ready();

// One row of a device picker.
struct SerialDevice {
    // What goes in the config file. `usb:10c4:ea60` (vendor and product,
    // lower-case hex, with `#N` appended for the second and later ports
    // of a multi-port adapter) or `bt:00:11:22:33:44:55`.
    //
    // **Deliberately not the USB device node or the enumeration
    // index**, either of which changes when the cable is replugged --
    // and a rig setting that stops working after unplugging the radio is
    // not a setting. The accepted cost is the same one the audio layer
    // takes on device names: two identical adapters are indistinguishable
    // and the first one wins.
    std::string id;
    std::string label;

    // USB: whether the app already holds permission for it. Bluetooth:
    // always true, since only bonded devices are listed. A device that
    // is present but not permitted is worth showing -- it is the one the
    // operator is about to grant.
    bool permitted = false;
};

std::vector<SerialDevice> usb_devices();
std::vector<SerialDevice> bluetooth_devices();

// Whether the app may open `id` right now. Cheap, and safe to poll.
bool has_permission(const std::string& id);

// Ask the user. **Returns immediately**: this raises a system dialog,
// so it must be called from the UI thread and the answer arrives later
// through the callback below. Nothing here blocks on a human.
void request_permission(const std::string& id);

// Called from whichever thread Android delivers the broadcast on --
// which is the main thread in practice, but do not rely on it.
using PermissionResult = std::function<void(const std::string& id, bool granted)>;
void set_permission_callback(PermissionResult callback);

// The transport itself. Construction is cheap and does not touch the
// device; `open()` does, and throws `RigError` if it cannot -- which is
// what puts every blocking operation on the rig worker thread, where
// `RigController` requires it.
std::shared_ptr<SerialTransport> make_transport(const SerialParams& params);

}  // namespace sstvae::rig::android

#endif
