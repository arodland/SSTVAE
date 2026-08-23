package org.cleverdomain.sstvae;

import android.app.Activity;
import android.os.Bundle;

/**
 * Receives {@code USB_DEVICE_ATTACHED} and does nothing with it.
 *
 * <p>The point is the side effect. Android drops a USB permission when the
 * device is detached, so without an intent filter the operator re-grants it
 * every time they plug the radio in — and a radio gets plugged in at the start
 * of every session. An app that <i>can handle</i> the attach intent is granted
 * permission automatically when the user answers "always open with this app",
 * and that grant survives replugging.
 *
 * <p><b>Its own activity, not the main one, and that is the whole design.</b>
 * The filter fires for anything in {@code res/xml/device_filter.xml} — which
 * includes every CDC-ACM device and four whole vendor ranges, so an Arduino or
 * a USB modem matches it too. Putting the filter on {@code QtActivity} would
 * mean plugging any of those in yanks the operator into a radio app they were
 * not using. This one starts, finishes, and is excluded from Recents, so the
 * grant is collected and nothing is shown.
 */
public final class UsbAttach extends Activity {
    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        finish();
    }
}
