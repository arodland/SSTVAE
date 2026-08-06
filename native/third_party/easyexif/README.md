# easyexif (vendored)

`exif.h` and `exif.cpp` from https://github.com/mayanklahiri/easyexif,
branch `master`, commit `cd994a3b6009bc3c1f84062e96bd7f5ad16e85f6`
(2020-05-22), fetched 2026-08-06. BSD-2-Clause -- see `LICENSE`, which
is upstream's own file.

**Vendored rather than fetched at configure time**, for the same reason
as `../pocketfft` and `../stb`: two files with no build system of their
own, and a `FetchContent` would make every CI job on three platforms
depend on GitHub being reachable during `cmake`. Updating means
replacing two files.

Unlike the other four vendored libraries this one is not header-only,
so `exif.cpp` is listed in `sstvae_core`'s sources. It compiles clean
under `-Wall -Wextra`, so it is *not* given the `-w` treatment stb gets
-- if a future update introduces warnings, suppress them then rather
than pre-emptively.

## What it is used for

Exactly one field: `EXIFInfo::Orientation`, read by
`core/images/images.cpp` so a phone photograph is framed the way it was
taken. Nothing else in the struct is looked at. The library is doing
the part that is genuinely fiddly and easy to get subtly wrong --
walking the JPEG marker segments to find APP1, the `II`/`MM` byte
order, the TIFF IFD offsets, and bounds-checking all of it against a
possibly-hostile file. Applying the resulting transform to the pixels
is ours (`images::apply_orientation`), because that part is eight cases
of index arithmetic with no format subtleties in it.

One other thing stayed ours, and it is worth knowing why. easyexif
parses **JPEG only**, but Pillow also honours orientation in a PNG
`eXIf` chunk -- so leaving PNG alone would have meant a tagged file
rotating in `sstvae/images.py` and not here, silently. `images.cpp`
therefore locates that chunk itself and prepends the `Exif\0\0`
signature a PNG omits, handing the TIFF stream to
`parseFromEXIFSegment`. That keeps the split intact: walking PNG chunks
is a length-prefixed scan that is bounds-checkable by inspection, with
no internal offsets that can point anywhere, which is not the kind of
thing TIFF is.

## Why this one

The alternative considered was
[TinyEXIF](https://github.com/cdcseacave/TinyEXIF) (MIT, also two
files). Two things decided it:

- easyexif's documented entry point is a **memory buffer**
  (`parseFrom(data, len)`), which is the shape `images::load` needs
  since it hands the same bytes to `stbi_load_from_memory`. TinyEXIF's
  documented usage is stream-based.
- TinyEXIF parses XMP as well, which needs tinyxml2. It is optional,
  but it makes "which dependencies does this pull in" a build
  configuration question, for a feature this project has no use for.

The tradeoff accepted in exchange: easyexif is **dormant** -- 41
commits, nothing since May 2020. That matters much less than usual
here. EXIF orientation is a frozen format, we read one tag of it, and
vendoring makes upstream's activity irrelevant to us either way. If it
ever does matter, TinyEXIF is the maintained one.

## It is a parser, and the input is a file the operator picked

A malformed or malicious APP1 segment must fail to a sane default, not
propagate. `images::load` therefore treats **any** non-zero return from
`parseFrom` -- and an `Orientation` outside 1..8 -- as orientation 1,
the identity, so an unreadable tag costs the operator a rotation rather
than a load failure. `test_images.cpp` plants a truncated APP1 to hold
that.
