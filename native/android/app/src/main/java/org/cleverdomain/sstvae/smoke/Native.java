package org.cleverdomain.sstvae.smoke;

import java.nio.ByteBuffer;

/**
 * The JNI surface, which is deliberately tiny.
 *
 * <p>Java calls into C++ on the data path and never the reverse: the capture
 * thread is already attached so {@link #push} is a plain call, while a C++
 * thread calling back would need {@code AttachCurrentThread} and buy nothing.
 * The decode loop therefore publishes into {@code rx::SharedState} and the UI
 * polls {@link #status()}.
 */
final class Native {

    static {
        System.loadLibrary("sstvae_smoke");
    }

    private Native() {}

    /** Empty on success, otherwise the reason. */
    static native String start(String format, int channels, int deviceRate,
                               String modelDir, String outDir);

    /** {@code buf} must be a direct ByteBuffer. */
    static native void push(ByteBuffer buf, int length);

    /** A small JSON object; see the C++ side for the fields. */
    static native String status();

    static native void stop();

    /** Write the ring buffer to a WAV. Empty on success, else the reason. */
    static native String dumpAudio(String path);

    /** False when built with {@code -DSSTVAE_BUILD_CODEC=OFF}. */
    static native boolean hasCodec();
}
