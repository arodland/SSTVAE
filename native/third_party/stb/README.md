# stb (vendored)

`stb_image.h` (v2.30), `stb_image_write.h`, `stb_image_resize2.h` from
https://github.com/nothings/stb, branch `master`, fetched 2026-07-28.
Public domain / MIT dual licence — see `LICENSE`, which is the licence
block from the headers themselves.

Vendored rather than fetched at configure time, for the same reason as
`../pocketfft`: single headers with no build system, and a
`FetchContent` would make every CI job depend on GitHub being reachable
during `cmake`.

## Why these, rather than QtGui

Loading, saving and scaling pictures is the sort of thing `QImage` does
well, and Qt arrives in Phase 3 anyway. Two reasons not to reach for it
here:

- **`core/` stays Qt-free through Phase 2.** The layering rule allows
  QtGui only under `core/overlay/`; using it for image I/O would mean
  widening that rule, and the point of the rule is that it is narrow.
- **The headless CLI stays genuinely headless.** Phase 2's exit
  criterion is a command-line tool that turns a WAV into a picture. It
  should not link a GUI toolkit, and CI should not install one on three
  platforms to test it — the workflow says Qt gets added with the GUI,
  and this keeps that true.

### …and where the applications do reach for it anyway

Both reasons are about **this** loader, and both still hold for it:
`images::load` is still stb, still Qt-free, and still what the golden
vectors, `pytest --native` and `sstvae-decode` use.

What they do not justify is the *app* being limited to stb's format
list, which is fixed at compile time and has no TIFF, WEBP or ICO in it.
So since 2026-08-16 the desktop and Android apps open pictures through
`core/images/qt/`, a separate library over `QImageReader` — Qt's list is
the platform's, and grows with an installed plugin. It is an added
layer, not a replacement: a file Qt declines is handed to `images::load`
before it is reported unreadable, because Qt has no handler for PSD, HDR
or PIC and an app that gained Qt's formats must not lose stb's.

The layering rule was widened by exactly one directory to allow it, and
`tools/check_layering.py` gained teeth in the same change — its QtGui
rule had been written against the `<QtGui/QImage>` spelling that nothing
in this tree uses.

## Scaling is deliberately not bit-identical to Pillow

`sstvae/images.py` frames pictures with `Image.LANCZOS`, and
`stb_image_resize2` will not reproduce it exactly.

This was checked rather than assumed: Pillow's resampler *is* exactly
reproducible — a reimplementation of `precompute_coeffs` plus the two
fixed-point passes from `Resample.c` matched it on every subpixel across
four different source geometries. So an exact port was available and
was declined, not unavailable.

**Declined because framing is transmit-side and cosmetic.** It decides
which pixels of an oversized source get sent, not what the waveform
means; a receiver never runs it, and two stations framing a photo one
pixel differently is not an interop question. That is the opposite of
the situation in `core/codec/`, where exactness *was* worth paying for
because it lands in the decoded picture.

The consequence for tests: parity tests feed images that are already
640x480, so `fit_image` is a mode conversion and nothing else, and the
resampler is not on the compared path.
