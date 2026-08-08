package org.cleverdomain.sstvae.smoke;

import android.annotation.SuppressLint;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * A blocking {@link AudioRecord} reader on its own thread.
 *
 * <p>Note what this architecture <em>is</em>: the blocking-read design the
 * desktop app wanted and could not have. Reading in a loop was the right answer
 * to the PortAudio GIL bug and had to be abandoned because {@code
 * stream.read()} corrupts the heap on PortAudio's JACK backend. Here there is
 * no interpreter on any thread and no realtime callback at all — just a thread
 * that blocks until bytes are ready — so the hazard class that cost 5 dB is
 * absent by construction rather than tuned around.
 *
 * <p>There is no latency requirement whatsoever: the ring buffer holds 130
 * seconds and the decode loop polls every five. That is the whole reason this
 * is Java rather than AAudio, whose only real benefit is latency.
 *
 * <p>The device is opened at <em>its own</em> rate and resampled in our code
 * ({@code audio::CapturePipeline} on the native side), never by asking the
 * device for 8 kHz. Almost nothing is natively 8 kHz, so requesting it does not
 * avoid a resampler — it delegates to whichever one the platform has.
 */
final class CaptureThread extends Thread {

    interface Listener {
        void onOpened(int sampleRate, int channels, String format);

        void onError(String message);
    }

    private static final String TAG = "sstvae";

    /** Candidate rates, most-likely-native first. */
    private static final int[] RATES = {48000, 44100, 32000, 16000, 8000};

    private String sourceName = "?";

    // Polled by the UI every 500 ms. Volatile rather than locked: these are
    // a display, and a torn read of a level meter is not worth a mutex on
    // the capture thread.
    volatile String openedAs = "";
    volatile String routedTo = "";
    volatile String routingWarning = "";
    volatile int levelPeak = 0;
    volatile double levelNearZeroPct = 100.0;

    private final AudioManager audioManager;
    private final int preferredDeviceId;
    private final Listener listener;
    private volatile boolean running = true;
    private AudioRecord record;

    CaptureThread(AudioManager audioManager, int preferredDeviceId, Listener listener) {
        super("sstvae-capture");
        this.audioManager = audioManager;
        this.preferredDeviceId = preferredDeviceId;
        this.listener = listener;
    }

    void shutdown() {
        running = false;
        AudioRecord r = record;
        if (r != null) {
            try {
                r.stop();
            } catch (IllegalStateException ignored) {
                // Already stopped; nothing to do.
            }
        }
        interrupt();
    }

    @SuppressLint("MissingPermission")  // Checked by the caller before starting.
    @Override
    public void run() {
        int rate = 0;
        int minBytes = 0;
        AudioRecord r = null;

        for (int candidate : RATES) {
            int min = AudioRecord.getMinBufferSize(candidate,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
            if (min <= 0) continue;
            // Two seconds of slack, matching the desktop's Qt buffer. This is
            // what makes a late drain harmless; measured there, capture stayed
            // clean through 800 ms of deliberate blocking at 1 s of buffer.
            int bytes = Math.max(min * 4, candidate * 2 * 2);
            try {
                r = new AudioRecord(MediaRecorder.AudioSource.UNPROCESSED, candidate,
                        AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                        bytes);
                sourceName = "UNPROCESSED";
            } catch (IllegalArgumentException e) {
                r = null;
            }
            if (r != null && r.getState() == AudioRecord.STATE_INITIALIZED) {
                rate = candidate;
                minBytes = bytes;
                break;
            }
            if (r != null) {
                r.release();
                r = null;
            }
        }

        if (r == null) {
            // UNPROCESSED is not guaranteed. Falling back to VOICE_RECOGNITION
            // rather than MIC because both skip most of the processing chain,
            // and AGC or noise suppression on an SSTV signal is destructive in
            // a way that looks like a bad radio rather than a bad setting.
            for (int candidate : RATES) {
                int min = AudioRecord.getMinBufferSize(candidate,
                        AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                if (min <= 0) continue;
                int bytes = Math.max(min * 4, candidate * 2 * 2);
                try {
                    r = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,
                            candidate, AudioFormat.CHANNEL_IN_MONO,
                            AudioFormat.ENCODING_PCM_16BIT, bytes);
                    sourceName = "VOICE_RECOGNITION";
                } catch (IllegalArgumentException e) {
                    r = null;
                }
                if (r != null && r.getState() == AudioRecord.STATE_INITIALIZED) {
                    rate = candidate;
                    minBytes = bytes;
                    break;
                }
                if (r != null) {
                    r.release();
                    r = null;
                }
            }
        }

        if (r == null) {
            listener.onError("could not open any capture rate");
            return;
        }
        record = r;

        if (preferredDeviceId != 0) {
            AudioDeviceInfo target = null;
            for (AudioDeviceInfo d :
                    audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)) {
                if (d.getId() == preferredDeviceId) {
                    target = d;
                    break;
                }
            }
            // The return value is checked because this is exactly the call
            // that silently does nothing on the paths we are avoiding. If the
            // platform declines the routing, the operator has to be told —
            // capturing from the built-in mic while the UI claims USB is the
            // worst available outcome.
            boolean ok = target != null && r.setPreferredDevice(target);
            if (!ok) {
                // Sticky, not a Toast. "Capturing from the built-in mic while
                // the UI claims USB" is the worst available outcome here, and
                // a message that fades after two seconds is how it happens --
                // the same reasoning as the desktop's error tiers.
                routingWarning = "the system refused to route capture to the "
                        + "selected device; using its own choice instead";
                listener.onError(routingWarning);
            }
        }

        String error = Native.start("Int16", 1, rate, App.modelDir(), App.outDir());
        if (!error.isEmpty()) {
            listener.onError("native session: " + error);
            r.release();
            record = null;
            return;
        }

        AudioDeviceInfo routed = r.getRoutedDevice();
        // What it *actually* opened and where it *actually* went. Displayed,
        // not just logged: on a USB test the whole question is whether the
        // interface got used, and answering it should not need adb.
        openedAs = sourceName + " " + r.getSampleRate() + " Hz";
        routedTo = routed == null ? "(unknown)"
                : AudioDevices.describeType(routed.getType()) + " \""
                        + routed.getProductName() + "\"";
        Log.i(TAG, "AudioRecord: source=" + sourceName
                + " requested=" + rate + " actual=" + r.getSampleRate()
                + " channels=" + r.getChannelCount()
                + " encoding=" + r.getAudioFormat()
                + " routedTo=" + (routed == null ? "(unknown)"
                        : routed.getProductName() + "/type" + routed.getType()));

        listener.onOpened(rate, 1, "Int16");

        // Direct, so the bytes are visible to C++ without a copy across the
        // boundary. Sized well under the device buffer so a read returns
        // promptly rather than waiting to fill something large.
        final int chunk = Math.max(4096, minBytes / 8);
        ByteBuffer buf = ByteBuffer.allocateDirect(chunk).order(ByteOrder.nativeOrder());

        try {
            r.startRecording();
        } catch (IllegalStateException e) {
            listener.onError("startRecording: " + e.getMessage());
            r.release();
            record = null;
            return;
        }

        long reportedAt = System.nanoTime();
        long bytesSinceReport = 0;
        int peak = 0;
        long nearZero = 0;
        long counted = 0;

        while (running) {
            buf.clear();
            int n = r.read(buf, chunk);
            if (n > 0) {
                for (int i = 0; i + 1 < n; i += 2) {
                    int v = Math.abs(buf.getShort(i));
                    if (v > peak) peak = v;
                    if (v < 16) nearZero++;
                    counted++;
                }
                bytesSinceReport += n;
                long now = System.nanoTime();
                if (now - reportedAt > 5_000_000_000L) {
                    double secs = (now - reportedAt) / 1e9;
                    Log.i(TAG, String.format(
                            "input: %.0f bytes/s (expect %d)  peak %d  %.1f%% below 16 LSB",
                            bytesSinceReport / secs, r.getSampleRate() * 2, peak,
                            100.0 * nearZero / Math.max(1, counted)));
                    levelPeak = peak;
                    levelNearZeroPct = 100.0 * nearZero / Math.max(1, counted);
                    reportedAt = now;
                    bytesSinceReport = 0;
                    peak = 0;
                    nearZero = 0;
                    counted = 0;
                }
                Native.push(buf, n);
            } else if (n < 0) {
                // ERROR_INVALID_OPERATION and friends. A negative read while
                // still running means the stream is gone, e.g. the USB device
                // was unplugged; stopping is better than spinning on it.
                listener.onError("capture read failed (" + n + ")");
                break;
            }
        }

        try {
            r.stop();
        } catch (IllegalStateException ignored) {
            // Already stopped by shutdown().
        }
        r.release();
        record = null;
        Log.i(TAG, "capture thread finished");
    }
}
