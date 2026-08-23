package com.hoho.android.usbserial;

/**
 * The one class the vendored library needs and Gradle does not generate for us.
 *
 * <p><b>Ours, not upstream's</b> — which is why it is in {@code shim/} and not
 * in {@code java/}. That directory is a byte-for-byte drop of the release and
 * has to stay that way, or updating it means diffing our edits back out.
 *
 * <p>Gradle synthesises a {@code BuildConfig} per Android <i>module</i>, in that
 * module's own package. Consuming the library as an {@code .aar} you get
 * {@code com.hoho.android.usbserial.BuildConfig} for free; compiling its sources
 * into this app, Gradle generates one for {@code org.cleverdomain.sstvae} and
 * nothing at all for the library's package, so two of its drivers fail to
 * compile on an import.
 *
 * <p><b>{@code DEBUG} is false, and that is the correct value rather than a
 * convenient one.</b> The single place upstream reads it
 * ({@code Ch34xSerialDriver:206}) is a test-only escape hatch that strips a bit
 * from the requested baud rate — its own comment says "for testing purpose
 * bypass dedicated baud rate handling". False is what a release build of the
 * library compiles to, and it is what a radio should be talking to.
 * {@code ProlificSerialDriver} imports the class and never reads it.
 *
 * <p>Deliberately not wired to this app's own {@code BuildConfig}: AGP 8 does
 * not generate one unless {@code android.buildFeatures.buildConfig} is turned
 * on, so that would trade a compile error we understand for one we would have
 * to rediscover.
 */
public final class BuildConfig {
    public static final boolean DEBUG = false;

    private BuildConfig() {}
}
