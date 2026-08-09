package org.cleverdomain.sstvae;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;

/**
 * The one permission Qt has no type for.
 *
 * <p>The microphone goes through {@code QMicrophonePermission}, which
 * owns the asynchronous result plumbing; {@code POST_NOTIFICATIONS}
 * (API 33+) has no Qt equivalent, so it is requested directly.
 *
 * <p><b>The result is deliberately not awaited.</b> Denial costs the
 * ongoing notification and nothing else — the session still runs, and
 * `ListenerService` is written to treat a missing notification as
 * non-fatal — so there is no decision waiting on the answer and no
 * reason to carry the callback through {@code
 * onRequestPermissionsResult}. The microphone is the opposite case, and
 * that one is awaited.
 */
public final class Permissions {
    private static final int REQUEST_NOTIFICATIONS = 1001;

    private Permissions() {}

    public static void requestNotifications(Context context) {
        if (Build.VERSION.SDK_INT < 33) return;
        if (!(context instanceof Activity)) return;
        final Activity activity = (Activity) context;
        if (activity.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                == PackageManager.PERMISSION_GRANTED) {
            return;
        }
        activity.requestPermissions(
                new String[] {Manifest.permission.POST_NOTIFICATIONS},
                REQUEST_NOTIFICATIONS);
    }
}
