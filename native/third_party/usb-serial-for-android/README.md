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

## Build integration

`native/android-app/CMakeLists.txt` stages `java/` into the assembled
`QT_ANDROID_PACKAGE_SOURCE_DIR`, alongside the app's own Java and the
audio and rig layers'. It is the only Android-only entry in
`third_party/`, so nothing else in the tree references it and the
desktop build never sees it.

Its one external dependency is `androidx.annotation.IntDef`, which
arrives transitively through the `androidx.core` dependency Qt's own
template already declares. Nothing else outside `java.*` and
`android.*` is imported.
