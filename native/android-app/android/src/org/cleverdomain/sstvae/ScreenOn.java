package org.cleverdomain.sstvae;

import android.app.Activity;
import android.content.Context;
import android.view.WindowManager;

/**
 * Holds the screen awake while a reception is in progress and the app
 * is on screen.
 *
 * <p><b>A window flag, deliberately not a wake lock.</b>
 * {@code FLAG_KEEP_SCREEN_ON} is a property of the window, so the
 * platform honours it only while that window is actually showing and
 * drops it the moment the app is backgrounded or the activity is
 * destroyed. That is exactly the requested behaviour — "the GUI is open
 * and receiving is active" — with no lifecycle tracking on our side and
 * nothing to leak.
 *
 * <p>A {@code PowerManager.WakeLock} would look like the more powerful
 * answer and would be a bug: it survives backgrounding, so a listening
 * session left running in a pocket would hold the screen on until the
 * battery went flat. The whole point of the foreground service is that
 * receiving continues <i>without</i> the display, which is most of the
 * battery answer in docs/android.md.
 */
public final class ScreenOn {
    private ScreenOn() {}

    /**
     * @param context the app's context; ignored unless it is an
     *     Activity, since there is no window to flag otherwise. That is
     *     the right no-op rather than an error: the same call is
     *     harmless from a service context.
     */
    public static void set(Context context, final boolean on) {
        if (!(context instanceof Activity)) return;
        final Activity activity = (Activity) context;
        // Window flags are main-thread only, and this is called from
        // whichever thread noticed the session change.
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (on) {
                    activity.getWindow().addFlags(
                            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                } else {
                    activity.getWindow().clearFlags(
                            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                }
            }
        });
    }
}
