package org.cleverdomain.sstvae.smoke;

import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Feeds a 16-bit mono WAV through the capture path, bypassing only the
 * microphone driver.
 *
 * <p>This exists because the emulator's virtual microphone is not a usable
 * audio source — measured here at 1/60th of the reference amplitude with two
 * thirds of its power outside the 900–2150 Hz band, i.e. not our signal at
 * all. That is an emulator limitation and says nothing about a real device, but
 * it does block the one thing the smoke test is for.
 *
 * <p>So this splits the question in two. Everything downstream of the driver —
 * {@code CapturePipeline}, the ring buffer, sync, framing, the beacon, and
 * onnxruntime decoding an actual picture — is exercised on real Android with
 * real arm64/x86_64 code, at the same 48 kHz and in the same ragged chunks
 * {@code AudioRecord} delivers. What remains untested is the driver itself,
 * which was always going to need hardware (docs/android.md).
 *
 * <p>Deliberately reuses {@link Native#push} rather than adding a native entry
 * point: a test path that does not go through the code under test is not a
 * test of it.
 */
final class WavFeeder extends Thread {

    interface Listener {
        void onDone(String message);
    }

    private final File file;
    private final Listener listener;
    private volatile boolean running = true;

    WavFeeder(File file, Listener listener) {
        super("sstvae-wav-feeder");
        this.file = file;
        this.listener = listener;
    }

    void shutdown() {
        running = false;
    }

    @Override
    public void run() {
        try (FileInputStream in = new FileInputStream(file)) {
            // Minimal WAV parse: skip to `data`. Enough for a file we
            // generated; anything more belongs in the real app, which uses
            // the C++ reader.
            byte[] header = new byte[12];
            if (in.read(header) != 12) throw new Exception("short file");
            int rate = 0;
            int channels = 1;
            long dataLen = 0;
            while (true) {
                byte[] ch = new byte[8];
                if (in.read(ch) != 8) throw new Exception("no data chunk");
                String id = new String(ch, 0, 4, "US-ASCII");
                int size = ByteBuffer.wrap(ch, 4, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
                if (id.equals("fmt ")) {
                    byte[] fmt = new byte[size];
                    if (in.read(fmt) != size) throw new Exception("short fmt");
                    ByteBuffer b = ByteBuffer.wrap(fmt).order(ByteOrder.LITTLE_ENDIAN);
                    b.getShort();               // audio format
                    channels = b.getShort();
                    rate = b.getInt();
                } else if (id.equals("data")) {
                    dataLen = size & 0xffffffffL;
                    break;
                } else {
                    if (in.skip(size) != size) throw new Exception("short skip");
                }
            }
            if (rate == 0) throw new Exception("no fmt chunk");

            String error = Native.start("Int16", channels, rate, App.modelDir(),
                    App.outDir());
            if (!error.isEmpty()) throw new Exception("native session: " + error);

            // Ragged on purpose, and never aligned to anything: chunk-boundary
            // handling is exactly what the per-chunk-resampling bug got wrong,
            // and a feeder that used one tidy size would not exercise it.
            final int[] sizes = {4096, 1024, 7680, 512, 16384};
            ByteBuffer buf = ByteBuffer.allocateDirect(16384).order(ByteOrder.nativeOrder());
            byte[] scratch = new byte[16384];
            long fed = 0;
            int k = 0;
            while (running && fed < dataLen) {
                int want = (int) Math.min(sizes[k++ % sizes.length], dataLen - fed);
                want -= want % (2 * channels);       // whole frames only
                if (want <= 0) break;
                int got = in.read(scratch, 0, want);
                if (got <= 0) break;
                buf.clear();
                buf.put(scratch, 0, got);
                Native.push(buf, got);
                fed += got;
                // Real time, so the decode loop's 5-second polls see the
                // transmission arrive the way they would on the air rather
                // than all at once.
                long ms = (long) (1000.0 * (got / (2.0 * channels)) / rate);
                if (ms > 0) Thread.sleep(ms);
            }
            Log.i("sstvae", "fed " + fed + " bytes at " + rate + " Hz");
            listener.onDone("fed " + fed + " bytes @" + rate + " Hz");
        } catch (Exception e) {
            Log.e("sstvae", "feeder failed", e);
            listener.onDone("feeder failed: " + e);
        }
    }
}
