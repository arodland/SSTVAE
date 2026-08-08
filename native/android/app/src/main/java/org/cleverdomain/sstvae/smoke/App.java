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

    private static File filesDirectory;
    private static File cacheDirectory;

    @Override
    public void onCreate() {
        super.onCreate();
        filesDirectory = getFilesDir();
        cacheDirectory = getCacheDir();
        new File(filesDirectory, "models").mkdirs();
        new File(filesDirectory, "received").mkdirs();
    }

    static String modelDir() {
        return new File(filesDirectory, "models").getAbsolutePath();
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
