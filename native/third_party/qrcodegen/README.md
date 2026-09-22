# qrcodegen (vendored)

`qrcodegen.cpp` / `qrcodegen.hpp` from
https://github.com/nayuki/QR-Code-generator, branch `master`, fetched
2026-09-21. MIT — see `LICENSE`, which is the licence block from the
sources themselves. Unmodified.

Vendored rather than fetched at configure time, for the same reason as
`../stb` and `../pocketfft`: two files with no build system of their
own, and a `FetchContent` would make every CI job depend on GitHub
being reachable during `cmake`.

## Why an encoder and no decoder

The desktop app *shows* a template as a QR code
(`docs/overlay-templates.md`); nothing in this repository reads one.
Encoding is a few hundred lines of Reed-Solomon and mask selection with
an exact answer; decoding is image processing — perspective, lighting,
blur — and belongs to whatever the phone's camera stack offers rather
than to a vendored library that would have to be as good as it.
