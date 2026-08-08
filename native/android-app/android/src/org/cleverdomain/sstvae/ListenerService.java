package org.cleverdomain.sstvae;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;

/**
 * Keeps a listening session alive independently of any UI.
 *
 * <p>This is the Android half of the ownership inversion in
 * docs/android.md: the session belongs to the service, and the activity
 * is a view that attaches to it. A reception therefore survives the
 * screen going off, a rotation, and the app being swiped away —
 * {@code stopWithTask="false"} in the manifest is what buys the last of
 * those, and it is the case that matters most, because on a phone
 * putting the app away is the normal thing to do while waiting for a
 * transmission.
 *
 * <p><b>The service holds no state of its own.</b> Everything on the
 * notification is read out of the native session at the moment it is
 * drawn, so there is nothing here that can disagree with what the UI
 * shows, and nothing accumulating while nobody is looking.
 */
public class ListenerService extends Service {
    private static final String TAG = "SSTVAE";
    private static final String CHANNEL_ID = "sstvae.listening";
    private static final int NOTIFICATION_ID = 1;
    private static final String PICTURE_CHANNEL_ID = "sstvae.received";
    // Each reception gets its own id, so a second picture does not
    // replace the first in the shade.
    private int nextPictureId = 100;

    public static final String ACTION_START = "org.cleverdomain.sstvae.START";
    public static final String ACTION_STOP = "org.cleverdomain.sstvae.STOP";
    public static final String EXTRA_DEVICE = "device";

    /** How often the notification is refreshed. Not a decode cadence —
     *  the engine polls on its own 5 s schedule and this only reads what
     *  it has already published. */
    private static final long REFRESH_MS = 2000;

    private static native boolean nativeStart(String device);
    private static native void nativeStop();
    private static native String nativeStatusLine();
    private static native String nativeTakeSavedPicture();
    private static native String nativeLastSavedSummary();

    private final Handler handler = new Handler(Looper.getMainLooper());
    private boolean running = false;

    private final Runnable refresh = new Runnable() {
        @Override
        public void run() {
            if (!running) return;
            NotificationManager nm = getSystemService(NotificationManager.class);
            if (nm != null) {
                nm.notify(NOTIFICATION_ID, buildNotification(nativeStatusLine()));
                // Polled here rather than pushed from C++: the engine
                // finishes a reception on its own thread, and this is
                // already the place with a Looper and a
                // NotificationManager.
                final String saved = nativeTakeSavedPicture();
                if (saved != null) postPicture(nm, saved, nativeLastSavedSummary());
            }
            handler.postDelayed(this, REFRESH_MS);
        }
    };

    /**
     * Ask the service to start listening. Must be called while the app
     * is in the foreground: from API 34 a microphone-typed foreground
     * service may not be started from the background, which is why
     * nothing but a button press reaches this.
     */
    public static void startListening(Context context, String device) {
        Intent i = new Intent(context, ListenerService.class);
        i.setAction(ACTION_START);
        i.putExtra(EXTRA_DEVICE, device);
        context.startForegroundService(i);
    }

    public static void stopListening(Context context) {
        Intent i = new Intent(context, ListenerService.class);
        i.setAction(ACTION_STOP);
        context.startService(i);
    }

    @Override
    public IBinder onBind(Intent intent) {
        // Nothing binds: the UI reaches the session directly through
        // its own JNI, so a binder would only be a second, slower path
        // to the same object — and a second path is how the two get to
        // disagree.
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        final String action = intent != null ? intent.getAction() : null;
        if (ACTION_STOP.equals(action)) {
            stopListening();
            return START_NOT_STICKY;
        }

        final String device = intent != null ? intent.getStringExtra(EXTRA_DEVICE) : "";

        // Go foreground *before* opening the microphone. The other
        // order is the one that gets the process killed mid-open on a
        // busy device, and it is also what the microphone service type
        // is asserting to the system.
        createChannel();
        startAsForeground();

        if (!nativeStart(device == null ? "" : device)) {
            Log.e(TAG, "capture did not start; stopping the service");
            stopListening();
            return START_NOT_STICKY;
        }

        running = true;
        handler.postDelayed(refresh, REFRESH_MS);

        // Not START_STICKY: a restarted service would arrive with a
        // null intent and no device, and silently opening the *wrong*
        // microphone is worse than not restarting. The operator can
        // see the session ended and start it again.
        return START_NOT_STICKY;
    }

    @Override
    public void onDestroy() {
        stopListening();
        super.onDestroy();
    }

    private void stopListening() {
        running = false;
        handler.removeCallbacks(refresh);
        nativeStop();
        stopForeground(STOP_FOREGROUND_REMOVE);
        stopSelf();
    }

    private void startAsForeground() {
        Notification n = buildNotification("Starting…");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE);
        } else {
            startForeground(NOTIFICATION_ID, n);
        }
    }

    /**
     * A finished reception, with the picture itself on the lock screen.
     *
     * <p>This is the reason to want the app on a phone rather than a
     * laptop: the operator is not watching, and a decoded picture is
     * worth interrupting for in a way that "listening, 43 polls" is
     * not. Hence its own channel at DEFAULT importance while the
     * ongoing one stays at LOW — they are different kinds of event and
     * sharing a channel would force one setting on both.
     */
    private void postPicture(NotificationManager nm, String path, String summary) {
        final Bitmap bmp = BitmapFactory.decodeFile(path);
        if (bmp == null) return;

        Intent open = new Intent(this, org.qtproject.qt.android.bindings.QtActivity.class);
        open.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent pi = PendingIntent.getActivity(
                this, nextPictureId, open, PendingIntent.FLAG_IMMUTABLE);

        Notification n = new Notification.Builder(this, PICTURE_CHANNEL_ID)
                .setContentTitle("Picture received")
                .setContentText(summary)
                .setSmallIcon(android.R.drawable.ic_menu_gallery)
                .setLargeIcon(bmp)
                .setStyle(new Notification.BigPictureStyle()
                        .bigPicture(bmp)
                        // Drop the thumbnail once expanded, so the
                        // picture is not shown twice in the same card.
                        .bigLargeIcon((Bitmap) null))
                .setContentIntent(pi)
                .setAutoCancel(true)
                .build();
        nm.notify(nextPictureId++, n);
    }

    private void createChannel() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm == null) return;
        // LOW: this updates every couple of seconds for as long as the
        // station is listening, so anything that makes a sound or peeks
        // would be unusable within a minute.
        NotificationChannel ch = new NotificationChannel(
                CHANNEL_ID, "Listening", NotificationManager.IMPORTANCE_LOW);
        ch.setDescription("Shown while SSTVAE is receiving");
        ch.setShowBadge(false);
        nm.createNotificationChannel(ch);

        NotificationChannel pics = new NotificationChannel(
                PICTURE_CHANNEL_ID, "Received pictures",
                NotificationManager.IMPORTANCE_DEFAULT);
        pics.setDescription("One notification per decoded picture");
        nm.createNotificationChannel(pics);
    }

    private Notification buildNotification(String text) {
        Intent open = new Intent(this, org.qtproject.qt.android.bindings.QtActivity.class);
        open.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent pi = PendingIntent.getActivity(
                this, 0, open, PendingIntent.FLAG_IMMUTABLE);

        return new Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("SSTVAE")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setContentIntent(pi)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .build();
    }
}
