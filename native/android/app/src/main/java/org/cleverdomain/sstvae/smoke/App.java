package org.cleverdomain.sstvae.smoke;

import android.app.Application;

import java.io.File;

/**
 * Holds the app-private paths the native side needs.
 *
 * <p>The environment variables are the interesting part. {@code
 * core/settings/settings.cpp} and {@code core/checkpoint/checkpoint.cpp} both
 * already honour {@code XDG_CONFIG_HOME} and {@code XDG_CACHE_HOME} on every
 * platform, so setting them here before anything reads them makes the entire
 * path layer work on Android with <em>no source change</em>. That is worth more
 * than it looks: an {@code #ifdef __ANDROID__} in those files would be a fourth
 * platform's worth of untested code in the one place where getting it wrong
 * means silently writing a config nobody reads.
 *
 * <p>Note this is set from Java rather than in the native library's
 * constructor, because it has to happen before any native static
 * initialisation that might read it.
 */
public final class App extends Application {

    private static App instance;
    private static File filesDirectory;
    private static File cacheDirectory;

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;
        filesDirectory = getFilesDir();
        cacheDirectory = getCacheDir();
        new File(filesDirectory, "models").mkdirs();
        File external = getExternalFilesDir("models");
        if (external != null) external.mkdirs();
        new File(filesDirectory, "received").mkdirs();
    }

    /** The decoder the smoke test needs; there is no Hub fetcher here. */
    static final String DECODER = "v3-decoder-fp16.onnx";

    /**
     * Where to look for the model, in order.
     *
     * <p>The internal directory is first because it is what the app owns, but
     * it is also the one a person cannot write to without {@code run-as} — so
     * the external ones are what make this usable on a real phone: {@code adb
     * push} reaches {@code getExternalFilesDir} with no root and no {@code
     * run-as}, and it is visible over MTP, so the file can simply be dragged
     * across.
     */
    static File[] modelDirs() {
        File external = instance == null ? null : instance.getExternalFilesDir("models");
        return new File[] {
            new File(filesDirectory, "models"),
            external,
            new File("/sdcard/Download"),
        };
    }

    /** The directory holding the decoder, or null if it is nowhere. */
    static File foundModelDir() {
        for (File d : modelDirs()) {
            if (d != null && new File(d, DECODER).isFile()) return d;
        }
        return null;
    }

    /** Where the operator should be told to put it. */
    static String preferredModelDir() {
        File external = instance == null ? null : instance.getExternalFilesDir("models");
        return (external != null ? external : new File(filesDirectory, "models"))
                .getAbsolutePath();
    }

    static String modelDir() {
        File found = foundModelDir();
        return (found != null ? found : new File(filesDirectory, "models"))
                .getAbsolutePath();
    }

    static String outDir() {
        return new File(filesDirectory, "received").getAbsolutePath();
    }

    static File filesDirectory() {
        return filesDirectory;
    }

    static File cacheDirectory() {
        return cacheDirectory;
    }
}
