package org.cleverdomain.sstvae;

import android.annotation.SuppressLint;
import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The Java half of {@code core/audio/android/}.
 *
 * <p>Part of the audio layer rather than of any app: an app supplying its own
 * would be free to get the blocking-read loop subtly wrong, and that loop is
 * the thing this layer exists to own.
 *
 * <p><b>Direction rule.</b> This class calls into C++ on the data path
 * ({@code nativePush}) from a thread it owns and which is therefore already
 * attached. C++ calls in here only for control — enumerate, open, close.
 * Reversing that would put an {@code AttachCurrentThread} on every buffer.
 *
 * <p>Call {@link #init} once with an application context before anything else.
 */
public final class AudioBridge {

    private static final String TAG = "sstvae";

    /** Candidate capture rates, most-likely-native first. */
    private static final int[] RATES = {48000, 44100, 32000, 16000, 8000};

    private static Context context;
    private static final AtomicInteger nextToken = new AtomicInteger(1);
    private static final List<Reader> readers = new ArrayList<>();
    private static AudioTrack track;

    private AudioBridge() {}

    public static void init(Context appContext) {
        context = appContext.getApplicationContext();
    }

    private static AudioManager manager() {
        return (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
    }

    // --- enumeration ----------------------------------------------------
    //
    // Names, not ids: `audio::match_device` matches descriptions, and the
    // config file has to stay hand-editable. The type is part of the name
    // because several phones report every input with the same product name,
    // and "USB Device" versus "Builtin Mic" is exactly the distinction the
    // operator is making.

    static String[] inputDeviceNames() {
        return names(AudioManager.GET_DEVICES_INPUTS);
    }

    static String[] outputDeviceNames() {
        return names(AudioManager.GET_DEVICES_OUTPUTS);
    }

    private static String[] names(int flags) {
        AudioManager am = manager();
        if (am == null) return new String[0];
        AudioDeviceInfo[] devices = am.getDevices(flags);
        String[] out = new String[devices.length];
        for (int i = 0; i < devices.length; i++) out[i] = describe(devices[i]);
        return out;
    }

    static String defaultInputName() {
        return "";  // Empty means "whatever the platform routes to".
    }

    static String defaultOutputName() {
        return "";
    }

    static String describe(AudioDeviceInfo d) {
        String product = String.valueOf(d.getProductName()).trim();
        String type = typeName(d.getType());
        return product.isEmpty() ? type : product + " (" + type + ")";
    }

    static String typeName(int type) {
        switch (type) {
            case AudioDeviceInfo.TYPE_BUILTIN_MIC: return "Builtin Mic";
            case AudioDeviceInfo.TYPE_BUILTIN_SPEAKER: return "Speaker";
            case AudioDeviceInfo.TYPE_USB_DEVICE: return "USB Device";
            case AudioDeviceInfo.TYPE_USB_HEADSET: return "USB Headset";
            case AudioDeviceInfo.TYPE_USB_ACCESSORY: return "USB Accessory";
            case AudioDeviceInfo.TYPE_WIRED_HEADSET: return "Wired Headset";
            case AudioDeviceInfo.TYPE_WIRED_HEADPHONES: return "Wired Headphones";
            case AudioDeviceInfo.TYPE_BLUETOOTH_SCO: return "Bluetooth SCO";
            case AudioDeviceInfo.TYPE_BLUETOOTH_A2DP: return "Bluetooth A2DP";
            case AudioDeviceInfo.TYPE_TELEPHONY: return "Telephony";
            default: return "Type " + type;
        }
    }

    private static AudioDeviceInfo findDevice(int flags, String name) {
        if (name == null || name.isEmpty()) return null;
        AudioManager am = manager();
        if (am == null) return null;
        AudioDeviceInfo exact = null;
        AudioDeviceInfo partial = null;
        int partialCount = 0;
        for (AudioDeviceInfo d : am.getDevices(flags)) {
            String n = describe(d);
            if (n.equals(name)) {
                exact = d;
            } else if (n.toLowerCase().contains(name.toLowerCase())) {
                partial = d;
                partialCount++;
            }
        }
        if (exact != null) return exact;
        // An ambiguous substring deliberately matches nothing: silently
        // capturing from the wrong radio is worse than falling back and
        // saying so. Same rule as `audio::match_device`.
        return partialCount == 1 ? partial : null;
    }

    // --- capture --------------------------------------------------------

    /** Returns a token, or -1. The C++ side keys its stream on it. */
    @SuppressLint("MissingPermission")  // RECORD_AUDIO is the caller's job.
    static int openInput(String deviceName, int ringRate) {
        int token = nextToken.getAndIncrement();
        Reader reader = new Reader(token, deviceName, ringRate);
        if (!reader.open()) return -1;
        synchronized (readers) {
            readers.add(reader);
        }
        reader.start();
        return token;
    }

    static void closeInput(int token) {
        Reader found = null;
        synchronized (readers) {
            for (Reader r : readers) {
                if (r.token == token) {
                    found = r;
                    break;
                }
            }
            if (found != null) readers.remove(found);
        }
        if (found != null) found.shutdown();
    }

    private static final class Reader extends Thread {
        final int token;
        private final String deviceName;
        private final int ringRate;
        private AudioRecord record;
        private int rate;
        private int chunkBytes;
        private String sourceName = "?";
        private volatile boolean running = true;

        Reader(int token, String deviceName, int ringRate) {
            super("sstvae-capture-" + token);
            this.token = token;
            this.deviceName = deviceName;
            this.ringRate = ringRate;
        }

        @SuppressLint("MissingPermission")
        boolean open() {
            // UNPROCESSED first: AGC or noise suppression on an OFDM signal
            // degrades it in a way that reads like a bad radio rather than a
            // bad setting. VOICE_RECOGNITION is the fallback because it also
            // skips most of the chain; plain MIC does not.
            int[] sources = {MediaRecorder.AudioSource.UNPROCESSED,
                             MediaRecorder.AudioSource.VOICE_RECOGNITION};
            String[] sourceNames = {"UNPROCESSED", "VOICE_RECOGNITION"};

            for (int s = 0; s < sources.length && record == null; s++) {
                for (int candidate : RATES) {
                    int min = AudioRecord.getMinBufferSize(candidate,
                            AudioFormat.CHANNEL_IN_MONO,
                            AudioFormat.ENCODING_PCM_16BIT);
                    if (min <= 0) continue;
                    // Two seconds of slack, matching the desktop's Qt buffer.
                    int bytes = Math.max(min * 4,
                            (int) (candidate * 2 * BUFFER_SECONDS));
                    AudioRecord r;
                    try {
                        r = new AudioRecord(sources[s], candidate,
                                AudioFormat.CHANNEL_IN_MONO,
                                AudioFormat.ENCODING_PCM_16BIT, bytes);
                    } catch (IllegalArgumentException e) {
                        continue;
                    }
                    if (r.getState() == AudioRecord.STATE_INITIALIZED) {
                        record = r;
                        rate = candidate;
                        chunkBytes = Math.max(4096, bytes / 8);
                        sourceName = sourceNames[s];
                        break;
                    }
                    r.release();
                }
            }
            if (record == null) return false;

            AudioDeviceInfo target =
                    findDevice(AudioManager.GET_DEVICES_INPUTS, deviceName);
            if (target != null) record.setPreferredDevice(target);
            // Deliberately not checking that return value. Measured against a
            // TH-D75 over USB it returned false while the routing had plainly
            // taken effect; the verdict is deferred to getRoutedDevice() once
            // the stream is live, which is the outcome rather than the API's
            // opinion of the request.
            requested = target;
            return true;
        }

        private AudioDeviceInfo requested;

        private static final double BUFFER_SECONDS = 2.0;

        void shutdown() {
            running = false;
            AudioRecord r = record;
            if (r != null) {
                try {
                    r.stop();
                } catch (IllegalStateException ignored) {
                    // Already stopped.
                }
            }
            interrupt();
        }

        @Override
        public void run() {
            AudioRecord r = record;
            try {
                r.startRecording();
            } catch (IllegalStateException e) {
                nativeError(token, "startRecording: " + e.getMessage());
                r.release();
                return;
            }

            // Only meaningful once the stream is live: before this there is
            // no routing to report, which is what made an earlier version
            // warn about a device that was working perfectly.
            String route = routeName();
            String warning = routeWarning();
            Log.i(TAG, "AudioRecord: source=" + sourceName + " rate=" + rate
                    + " -> " + route + (warning.isEmpty() ? "" : "  ! " + warning));
            nativeOpened(token, rate, 1, route, warning);

            ByteBuffer buf = ByteBuffer.allocateDirect(chunkBytes)
                    .order(ByteOrder.nativeOrder());
            long reportedAt = System.nanoTime();
            int peak = 0;
            long nearZero = 0;
            long counted = 0;
            int lastPeak = 0;
            double lastNearZero = 1.0;

            while (running) {
                buf.clear();
                int n = r.read(buf, chunkBytes);
                if (n > 0) {
                    for (int i = 0; i + 1 < n; i += 2) {
                        int v = Math.abs(buf.getShort(i));
                        if (v > peak) peak = v;
                        if (v < 16) nearZero++;
                        counted++;
                    }
                    long now = System.nanoTime();
                    if (now - reportedAt > 1_000_000_000L) {
                        lastPeak = peak;
                        lastNearZero = (double) nearZero / Math.max(1, counted);
                        reportedAt = now;
                        peak = 0;
                        nearZero = 0;
                        counted = 0;
                    }
                    nativePush(token, buf, n, lastPeak, lastNearZero);
                } else if (n < 0) {
                    // A negative read while still running means the stream is
                    // gone -- the USB device unplugged, typically. Stopping
                    // beats spinning on it.
                    nativeError(token, "capture read failed (" + n + ")");
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
        }

        private String routeName() {
            AudioDeviceInfo routed = record.getRoutedDevice();
            return routed == null ? "(not reported)" : describe(routed);
        }

        private String routeWarning() {
            if (requested == null) return "";
            AudioDeviceInfo routed = record.getRoutedDevice();
            if (routed == null || routed.getId() == requested.getId()) return "";
            return "routed to " + describe(routed) + ", not the selected "
                    + describe(requested);
        }
    }

    // --- playback -------------------------------------------------------

    /** Returns the rate actually opened, or -1. */
    static int openOutput(String deviceName, int wantRate) {
        int rate = wantRate;
        int min = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT);
        if (min <= 0) {
            rate = 48000;
            min = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT);
        }
        if (min <= 0) return -1;
        try {
            track = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                            .build())
                    .setAudioFormat(new AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(rate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build())
                    .setBufferSizeInBytes(Math.max(min * 4, rate * 2))
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build();
        } catch (Exception e) {
            Log.e(TAG, "openOutput", e);
            return -1;
        }
        AudioDeviceInfo target =
                findDevice(AudioManager.GET_DEVICES_OUTPUTS, deviceName);
        if (target != null) track.setPreferredDevice(target);
        track.play();
        return rate;
    }

    /** Blocking write. Returns bytes written, or -1. */
    static int writeOutput(byte[] data, int offset, int length) {
        AudioTrack t = track;
        if (t == null) return -1;
        int n = t.write(data, offset, length, AudioTrack.WRITE_BLOCKING);
        return n < 0 ? -1 : n;
    }

    static void closeOutput() {
        AudioTrack t = track;
        track = null;
        if (t == null) return;
        try {
            // Blocks until queued audio has actually played. Without it the
            // tail is discarded on stop() -- and on a transmit path the tail
            // is the end of the picture.
            t.stop();
        } catch (IllegalStateException ignored) {
            // Never started.
        }
        t.release();
    }

    private static native void nativeOpened(int token, int deviceRate, int channels,
                                            String routedTo, String warning);

    private static native void nativePush(int token, ByteBuffer buf, int length,
                                          int peak, double nearZero);

    private static native void nativeError(int token, String message);
}
