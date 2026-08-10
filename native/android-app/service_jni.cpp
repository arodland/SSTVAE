// What `ListenerService` calls to drive the session.
//
// **The service starts and stops the session; the UI never does.** A
// view asks the service (via `startForegroundService`) and the service
// asks here, so there is exactly one path into `Session::start` and it
// is the one whose lifetime Android guarantees. The alternative --
// starting the session from the activity and posting a notification
// alongside it -- reads simpler and is the bug: the session would then
// belong to something the system may destroy while a reception is in
// progress.

#include <jni.h>

#include <cstdio>
#include <string>

#include "rx/engine.hpp"
#include "session.hpp"
#include "tx/engine.hpp"

namespace {

using sstvae::androidapp::Session;

std::string to_utf8(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

}  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL Java_org_cleverdomain_sstvae_ListenerService_nativeStart(
    JNIEnv* env, jclass, jstring device) {
    return Session::instance().start(to_utf8(env, device)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_org_cleverdomain_sstvae_ListenerService_nativeStop(JNIEnv*, jclass) {
    Session::instance().stop();
}

// One line for the ongoing notification.
//
// Deliberately *not* the UI's status string. The notification is read
// at a glance from a lock screen and has room for one clause, so it
// answers the only question worth answering there -- is this still
// working, and is something arriving -- while the pane gives the
// diagnostics. Built here rather than in Java so that the field
// meanings live next to the engine that publishes them.
JNIEXPORT jstring JNICALL
Java_org_cleverdomain_sstvae_ListenerService_nativeStatusLine(JNIEnv* env, jclass) {
    // Transmitting first, and unconditionally: it is the state where the
    // radio is on the air, it is short, and half duplex means the
    // receive line has nothing to say meanwhile anyway. An operator
    // glancing at the shade during an over needs to see that the station
    // is transmitting before anything else.
    if (Session::instance().transmitting()) {
        const sstvae::tx::TxState t = Session::instance().tx_state();
        std::string s = "Transmitting";
        if (t.phase == sstvae::tx::TxPhase::Sending) {
            char pct[16];
            std::snprintf(pct, sizeof pct, "  %.0f%%", 100.0 * t.progress);
            s += pct;
        } else if (!t.message.empty()) {
            s = "Transmitting - " + t.message;
        }
        return env->NewStringUTF(s.c_str());
    }

    const sstvae::rx::Progress p = Session::instance().progress();
    std::string s;
    switch (p.status) {
        case sstvae::rx::Status::Receiving: {
            s = "Receiving";
            if (p.mode_name) s += " mode " + *p.mode_name;
            if (p.frames_received) {
                s += "  " + std::to_string(*p.frames_received);
                if (p.n_frames_expected) s += "/" + std::to_string(*p.n_frames_expected);
                s += " frames";
            }
            if (!p.callsign.empty()) s += "  " + p.callsign;
            break;
        }
        case sstvae::rx::Status::Done:
            s = "Reception complete";
            break;
        case sstvae::rx::Status::Listening:
        default:
            // The poll count is the difference between "alive and
            // hearing nothing" and "wedged", which is otherwise
            // invisible from outside and is the question a listener
            // left running for an hour actually has -- but it is also
            // the single most technical thing on an otherwise plain
            // notification, so it follows the same switch as the pane.
            s = "Listening";
            if (Session::instance().show_technical()) {
                s += "  " + std::to_string(p.polls) + " polls";
            }
            break;
    }
    return env->NewStringUTF(s.c_str());
}

// The path of a reception saved since the last call, or null.
// Consuming, so the service posts each picture exactly once.
JNIEXPORT jstring JNICALL
Java_org_cleverdomain_sstvae_ListenerService_nativeTakeSavedPicture(JNIEnv* env, jclass) {
    const auto path = Session::instance().take_saved_picture();
    if (!path) return nullptr;
    return env->NewStringUTF(path->c_str());
}

JNIEXPORT jstring JNICALL
Java_org_cleverdomain_sstvae_ListenerService_nativeLastSavedSummary(JNIEnv* env, jclass) {
    return env->NewStringUTF(Session::instance().last_saved_summary().c_str());
}

// The gallery-export switch, read by the service rather than pushed to
// it: the export happens on the Java side (that is where the Context
// is), so the setting has to be readable from there, and the session is
// already the one place both halves of the app agree about.
JNIEXPORT jboolean JNICALL
Java_org_cleverdomain_sstvae_ListenerService_nativeSaveToGallery(JNIEnv*, jclass) {
    return Session::instance().save_to_gallery() ? JNI_TRUE : JNI_FALSE;
}

// The other direction: an export failure has to reach a screen, and the
// service has no UI of its own. Empty clears it.
JNIEXPORT void JNICALL
Java_org_cleverdomain_sstvae_ListenerService_nativeReportGalleryError(JNIEnv* env, jclass,
                                                                     jstring message) {
    Session::instance().set_gallery_error(to_utf8(env, message));
}

// The over the UI staged. The picture never crosses this boundary -- see
// Session::stage_transmit for why it is two calls rather than one.
JNIEXPORT jboolean JNICALL
Java_org_cleverdomain_sstvae_ListenerService_nativeStartTransmit(JNIEnv*, jclass) {
    return Session::instance().start_staged_transmit() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_org_cleverdomain_sstvae_ListenerService_nativeCancelTransmit(JNIEnv*, jclass) {
    Session::instance().cancel_transmit();
}

JNIEXPORT jboolean JNICALL
Java_org_cleverdomain_sstvae_ListenerService_nativeTransmitting(JNIEnv*, jclass) {
    return Session::instance().transmitting() ? JNI_TRUE : JNI_FALSE;
}

}  // extern "C"
