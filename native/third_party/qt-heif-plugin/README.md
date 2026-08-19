# qt-heif-plugin (vendored)

A Qt image format plugin for HEIF/HEIC, so the applications can open the
pictures a modern phone takes. `heif.cpp`, `heif_p.h`, `util_p.h` and
`heif.json` from
<https://github.com/novomesk/qt-heic-image-plugin>, commit
`65c5c488e96f4fe1132f4c72e113f7f1aa50699d`, fetched 2026-08-16.

Built by `native/CMakeLists.txt` into `plugins/imageformats/`, against
the libheif pinned by `native/cmake/libheif.cmake`. Nothing in this
project references it in code: it is found at run time by Qt's plugin
loader, which is what makes `images::qt::load` and
`images::qt::readable_extensions` pick the format up with no change to
either.

## Licence

`LGPL-2.0-or-later`, per the SPDX header in every file. `LICENSE` is
upstream's copy of the LGPL-2.1 text, which is the "or later" version
upstream chose to ship.

The plugin is a separate shared object that Qt `dlopen`s, and libheif is
linked into it dynamically, so the LGPL obligations land where they are
easiest to meet: the operator can replace the plugin or the codec
library outright without relinking anything of ours. Recorded in the
root `NOTICE`.

## Why this and not KDE's kimageformats

Upstream is the same author as kimageformats' `heif.cpp` and the code is
substantially the same. kimageformats would have been the tidier
dependency -- it is one pinned source for JXL, HEIF, AVIF and more --
but its master requires **ECM 6.29 and Qt 6.9**, which would make "which
image formats the app supports" a function of the builder's Qt version.
That is exactly the property `hamlib.cmake` exists to prevent, and it
would have broken the build on Ubuntu 24.04's Qt 6.4.2. Three
self-contained files with no framework behind them do not have that
problem.

Note for the JPEG XL counterpart, when it lands: the *standalone*
`novomesk/qt-jpegxl-image-plugin` is **GPL-3.0** and cannot be bundled
here -- the icon in `NOTICE` is licensed artwork that cannot be
relicensed under GPLv3, and it is compiled into the executable. KDE's
`jxl.cpp` is BSD-2-Clause and is the copy to take.

## Modifications

One, and it must be stated because the LGPL requires modifications to be
marked:

- **`heif.json` had `hej2` and `avci` removed from its Keys and
  MimeTypes.** Those are JPEG 2000 and H.264 payloads inside a HEIF
  container, decoded by OpenJPEG and openh264 respectively, and
  `libheif.cmake` builds with both off. Left in, the plugin would
  *advertise* two formats it cannot decode -- so they would appear in
  `readable_extensions()` and therefore in the file dialog's filter,
  and refuse to open. That is the precise bug this whole change set
  exists to retire; it would have been silly to import a fresh copy of
  it.

The `.cpp` and `.h` files are unmodified. Keep it that way if you can:
an unmodified vendored file is one an operator can diff against
upstream, and the version above is what says which upstream.
