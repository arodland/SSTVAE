package org.cleverdomain.sstvae;

import android.app.Activity;
import android.content.Context;

/**
 * Sends the task to the background, the way Back does on a home screen.
 *
 * <p>This is what Back means while a session is running. The app has an
 * ongoing foreground service and a notification promising that it is
 * listening; ending the activity would kill the process that owns the
 * engine, so the promise would stop being true at the moment the user
 * did the single most ordinary thing on a phone. Backgrounding keeps the
 * session, the notification and the poller that keeps the notification
 * honest, and it is what recorders, navigation and media apps all do.
 *
 * <p>{@code moveTaskToBack} rather than {@code finish}: the task stays in
 * Recents and the activity is preserved, so returning by the launcher or
 * by tapping the notification comes back to the screen that was left
 * rather than to a cold start. The {@code true} argument means "even if
 * this is not the root activity", which it always is here.
 */
public final class Background {
    private Background() {}

    public static void moveToBack(Context context) {
        if (!(context instanceof Activity)) return;
        ((Activity) context).moveTaskToBack(true);
    }
}
