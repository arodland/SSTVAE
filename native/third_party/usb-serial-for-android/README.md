# usb-serial-for-android (vendored)

The Java sources of https://github.com/mik3y/usb-serial-for-android,
tag `3.11.0` (commit `16d84116a03880a7842a9439b83f5f62ac892df2`),
fetched 2026-08-22. MIT -- see `LICENSE`, which is upstream's own file.
19 files, ~4200 lines, under `java/com/hoho/android/usbserial/`.

## What it is for

Android hands an unprivileged app no `/dev/ttyUSB0`. USB serial goes
through the USB host API, which gives you bulk endpoints and nothing
else -- so *every* chip-level protocol has to be implemented on top:
FTDI's modem-status prefix on every packet, CP210x's vendor control
requests, CH34x's undocumented register writes, PL2303's several
incompatible generations, and CDC-ACM's line-coding descriptors. This
library is that work, and it is the standard answer for it in the ham
software that already runs on Android.

`core/rig/android/SerialBridge.java` uses it: `UsbSerialProber` to find
and probe devices, and `UsbSerialPort` for `read`/`write`/
`setParameters`/`setDTR`/`setRTS`/`setFlowControl`. `setDTR` and
`setRTS` are load-bearing rather than incidental -- they are how PTT is
asserted on the bridged path, where Hamlib is holding a socket and its
own `ser_set_dtr` (a `TIOCMSET` ioctl) cannot work.

## Why vendored rather than a Gradle dependency

Qt's generated `build.gradle` already carries
`implementation fileTree(dir: 'libs', include: ['*.jar', '*.aar'])`,
so dropping an `.aar` in would have worked with no template surgery.
It was not done, for the reason `tools/make_installer.sh` declines
`choco install nsis`: the artifact is published on JitPack, which
**builds on demand from a git tag** rather than serving a fixed file.
That makes a build's success a property of a third-party build
service's current state, which is the thing pinning exists to prevent
-- and there is no `repo1.maven.org` copy to pin instead. Vendoring
4200 lines of Java is the smaller cost, and it is the same trade
`../easyexif` and `../stb` already record.

The consequence to know about: **updating means replacing `java/` and
this file**, not bumping a version string. Do it deliberately, in step
with checking that `SerialBridge.java` still compiles -- upstream has
changed the `read` and flow-control signatures within the 3.x series.

## `PATCHES.md` — deviations from the release

`java/` was a byte-for-byte drop of the release until 2026-08-23 and is
no longer. **One line is patched**, and `PATCHES.md` is the exhaustive
list: `Cp21xxSerialDriver.openInt()` no longer writes the CP210x
`SET_FLOW` structure when no flow control is wanted, because sending
that chip sixteen zero bytes stopped an IC-9700 answering CI-V at all.
Read that file before re-vendoring -- every deviation is also marked in
the source with `// SSTVAE PATCH`, so a replacement drop cannot lose one
without the marker going with it.

The rule the `shim/` directory exists for still stands and is why there
is exactly one patch: **anything reachable from outside the library
belongs in `core/rig/android/java/`**, which is ours. This one is not --
`openInt` runs inside `port.open()`, before any of our code is asked
anything.

## `shim/` — one class Gradle would have generated

Gradle synthesises a `BuildConfig` per Android *module*, in that module's
own package. Consumed as an `.aar` you get
`com.hoho.android.usbserial.BuildConfig` for free; compiling the sources
into an app, Gradle generates one for `org.cleverdomain.sstvae` and none
at all for the library's package — so `Ch34xSerialDriver` and
`ProlificSerialDriver` fail to compile on an import, and only during a
full Gradle run.

`shim/com/hoho/android/usbserial/BuildConfig.java` supplies it. It is in
`shim/` rather than in `java/` to keep edits out of the upstream drop:
anything in there has to be diffed back out at every update, which is
the sort of thing that gets forgotten once and then silently reverted.
`PATCHES.md` above exists because one such edit turned out to be
unavoidable; this one was not.

`DEBUG` is `false`, and that is the correct value rather than a
convenient one. The single place upstream reads it
(`Ch34xSerialDriver:206`) is a test-only escape hatch that strips a bit
from the requested baud rate — upstream's own comment says "for testing
purpose bypass dedicated baud rate handling". False is what a release
build of the library compiles to. `ProlificSerialDriver` imports the
class and never reads it.

It is deliberately *not* wired to the app's own `BuildConfig`: AGP 8
does not generate one unless `android.buildFeatures.buildConfig` is
enabled, so that would trade a compile error we understand for one that
would have to be rediscovered.

Nothing else in `java/` depends on generated code — no `R.` references,
and the only import outside `java.*`, `android.*` and `com.hoho.*` is
`androidx.annotation.IntDef`.

## Build integration

`native/android-app/CMakeLists.txt` stages `java/` and `shim/` into the assembled
`QT_ANDROID_PACKAGE_SOURCE_DIR`, alongside the app's own Java and the
audio and rig layers'. It is the only Android-only entry in
`third_party/`, so nothing else in the tree references it and the
desktop build never sees it.

Its one external dependency is `androidx.annotation.IntDef`, which
arrives transitively through the `androidx.core` dependency Qt's own
template already declares. Nothing else outside `java.*` and
`android.*` is imported.
