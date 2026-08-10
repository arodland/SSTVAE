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
import android.graphics.drawable.Icon;
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
    public static final String ACTION_TRANSMIT = "org.cleverdomain.sstvae.TRANSMIT";
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
    /** Whether the operator asked for receptions in the shared gallery. */
    private static native boolean nativeSaveToGallery();
    /** Empty clears it; anything else is shown on the Settings screen. */
    private static native void nativeReportGalleryError(String message);
    /** Begins the over the UI already staged on the native session. */
    private static native boolean nativeStartTransmit();
    private static native void nativeCancelTransmit();
    private static native boolean nativeTransmitting();

    private final Handler handler = new Handler(Looper.getMainLooper());
    private boolean running = false;
    /** True between a transmit request and the engine going idle again. */
    private boolean transmitting = false;

    private final Runnable refresh = new Runnable() {
        @Override
        public void run() {
            if (!running && !transmitting) return;
            // An over ends on its own, on a worker thread, and this is
            // the only place watching. Noticing here is what lets a
            // transmit-only service stop itself afterwards rather than
            // sitting foreground forever.
            if (transmitting && !nativeTransmitting()) {
                transmitting = false;
                if (!running) {
                    stopEverything();
                    return;
                }
            }
            NotificationManager nm = getSystemService(NotificationManager.class);
            if (nm != null) {
                nm.notify(NOTIFICATION_ID, buildNotification(nativeStatusLine()));
                // Polled here rather than pushed from C++: the engine
                // finishes a reception on its own thread, and this is
                // already the place with a Looper and a
                // NotificationManager.
                final String saved = nativeTakeSavedPicture();
                if (saved != null) {
                    postPicture(nm, saved, nativeLastSavedSummary());
                    exportToGallery(saved);
                }
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

    /**
     * Send the over the UI has staged on the native session.
     *
     * <p>Routed through the service for the same reason listening is: an
     * over is 32–95 s of committed airtime, and a transmission that stops
     * halfway because the activity was destroyed puts a truncated picture
     * on the band. The activity is in the foreground when Send is pressed
     * — that is what makes starting the service legal — but it need not
     * still be there when the audio ends.
     */
    public static void transmit(Context context) {
        Intent i = new Intent(context, ListenerService.class);
        i.setAction(ACTION_TRANSMIT);
        context.startForegroundService(i);
    }

    public static void cancelTransmit(Context context) {
        nativeCancelTransmit();
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
            stopEverything();
            return START_NOT_STICKY;
        }

        if (ACTION_TRANSMIT.equals(action)) {
            createChannel();
            // Re-asserted every time, because the type has to cover what
            // the service is about to do: a station that is not listening
            // has no microphone claim to transmit under.
            startAsForeground();
            if (!nativeStartTransmit()) {
                Log.e(TAG, "the session refused the transmit request");
                if (!running) stopEverything();
                return START_NOT_STICKY;
            }
            transmitting = true;
            handler.removeCallbacks(refresh);
            handler.postDelayed(refresh, REFRESH_MS);
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
            stopEverything();
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
        stopEverything();
        super.onDestroy();
    }

    /**
     * End the session, in both directions.
     *
     * <p>Cancelling the transmission first is deliberate and is the one
     * ordering that matters here: {@code nativeStop()} would otherwise
     * return while an over was still playing, and the service would go
     * away underneath a transmission it is supposed to be protecting.
     */
    private void stopEverything() {
        running = false;
        transmitting = false;
        handler.removeCallbacks(refresh);
        nativeCancelTransmit();
        nativeStop();
        stopForeground(STOP_FOREGROUND_REMOVE);
        stopSelf();
    }

    /**
     * Both types, whenever both are available.
     *
     * <p>The service does two things Android wants declared, and which of
     * them applies changes during a session: listening is {@code
     * microphone}, an over is {@code mediaPlayback}, and half duplex
     * means the first stops for the duration of the second. Re-declaring
     * on each transition is a thing to get wrong once, so both are
     * declared for the whole lifetime, as in the manifest.
     *
     * <p><b>Except that {@code microphone} may not be claimed without
     * RECORD_AUDIO</b>, and from API 34 asking anyway throws. A station
     * that denied the microphone and only wants to transmit is a real
     * configuration — one this app should support rather than crash on —
     * so the type is dropped when the permission is not held.
     */
    private void startAsForeground() {
        Notification n = buildNotification("Starting…");
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, n);
            return;
        }
        int types = ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK;
        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO)
                == android.content.pm.PackageManager.PERMISSION_GRANTED) {
            types |= ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE;
        }
        startForeground(NOTIFICATION_ID, n, types);
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
                .setSmallIcon(smallIcon())
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

    /**
     * Mirror a reception into the shared gallery, if the operator asked
     * for that.
     *
     * <p>Here rather than beside the file write in {@code
     * Session::save_reception}, and that placement is the point: this
     * poller is already the one thing holding a {@link Context}, and a
     * {@code Context} is what a MediaStore insert needs. Doing it from
     * C++ would mean calling into an application class from a thread the
     * engine created, which is the {@code FindClass} hazard
     * docs/android.md records — real, and entirely avoidable by exporting
     * from the side that was in Java to begin with.
     *
     * <p>A thread per reception rather than an executor: these arrive
     * 32–95 s apart at the very best, so there is nothing to pool, and a
     * plain thread has no lifecycle to get wrong when the service stops.
     * It must not run inline — this runnable is on the main Looper and
     * the copy is about a megabyte of I/O.
     */
    private void exportToGallery(final String path) {
        if (!nativeSaveToGallery()) return;
        final Context context = getApplicationContext();
        new Thread(new Runnable() {
            @Override
            public void run() {
                final String error = Gallery.save(context, path);
                if (error != null) Log.w(TAG, "gallery export failed: " + error);
                // Reported either way, so a failure that has since been
                // fixed (storage was full, and is not any more) stops
                // being shown. A sticky error nobody can clear is the
                // one the operator learns to ignore.
                nativeReportGalleryError(error == null ? "" : error);
            }
        }, "sstvae-gallery").start();
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

    /** The bird silhouette, by name so no generated R class is needed. */
    private int smallIcon() {
        int id = getResources().getIdentifier("ic_stat_sstvae", "drawable", getPackageName());
        return id != 0 ? id : android.R.drawable.ic_btn_speak_now;
    }

    private Notification buildNotification(String text) {
        Intent open = new Intent(this, org.qtproject.qt.android.bindings.QtActivity.class);
        open.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent pi = PendingIntent.getActivity(
                this, 0, open, PendingIntent.FLAG_IMMUTABLE);

        // Stop from the shade, without going through the UI.
        //
        // This is not a shortcut for the button on the Listen screen —
        // it is the only control that exists in the state the service
        // is built for. An ongoing notification is what the operator
        // sees after swiping the app away, and {@code
        // stopWithTask="false"} means that session is still holding the
        // microphone and the wake path with no activity to return to.
        // Without this, ending it means relaunching the app or digging
        // through the system settings, and a foreground service the
        // user cannot stop from its own notification is the kind that
        // gets uninstalled.
        //
        // Request code 1, not 0: a PendingIntent is identified by
        // (context, requestCode, intent-modulo-extras), so reusing 0
        // would collide with the content intent above and one of the
        // two would silently become the other.
        Intent stop = new Intent(this, ListenerService.class);
        stop.setAction(ACTION_STOP);
        PendingIntent stopPi = PendingIntent.getService(
                this, 1, stop, PendingIntent.FLAG_IMMUTABLE);

        return new Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("SSTVAE")
                .setContentText(text)
                .setSmallIcon(smallIcon())
                .setContentIntent(pi)
                .addAction(new Notification.Action.Builder(
                                   Icon.createWithResource(
                                           this, android.R.drawable.ic_menu_close_clear_cancel),
                                   "Stop", stopPi)
                                   .build())
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .build();
    }
}
