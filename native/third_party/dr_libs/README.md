# dr_libs (vendored)

`dr_wav.h` from https://github.com/mackron/dr_libs, branch `master`,
fetched 2026-07-28. Public domain / MIT-0 dual licence — see `LICENSE`,
which is the licence block from the header itself.

Vendored rather than fetched at configure time, for the same reason as
`../pocketfft` and `../stb`: a single header with no build system, and
a `FetchContent` would make every CI job depend on GitHub being
reachable during `cmake`.

## Why not a hand-rolled RIFF parser

WAV looks like a weekend's work and is not. `sstvae/wavio.py` gets to be
47 lines because `scipy.io.wavfile` is doing the work underneath, and
the files this has to read are not ones we wrote: they are recordings
made by whatever the operator had — SDR software, a phone, Audacity,
`arecord` — which in practice means 8/16/24/32-bit integer, IEEE float,
`WAVE_FORMAT_EXTENSIBLE`, and chunk layouts with metadata before `data`.

Getting one of those wrong does not produce an error. It produces
samples that are quiet, or offset, or half-rate, and the modem is
scale-invariant enough to *nearly* decode them — which is exactly the
class of bug that has cost this project the most time (see the
`read_wav` stereo-scaling note in CLAUDE.md, which decoded anyway and
so went unnoticed).

## dr_flac.h

`dr_flac.h` v0.13.4 from the same repository,
https://raw.githubusercontent.com/mackron/dr_libs/master/dr_flac.h,
fetched 2026-10-14, sha256
`111144e778f55738db6851cb226015c419e00d04b916a09506d4856d9cff945c`.
Same dual licence as `dr_wav.h` (public domain / MIT-0; `LICENSE` above
is the identical block from this header too). Unmodified.

It reads the QRSSTVAE passband store's packed hours
(`native/core/qrss/flac.cpp`, the only translation unit that defines
`DR_FLAC_IMPLEMENTATION`): both the files the C++ packer writes and the
ones Python's `soundfile` (libsndfile -> libFLAC, LPC) wrote, seeking to
a sample range. dr_flac decodes only; the encoder is ours
(`native/core/qrss/flac.cpp`), small enough to check, because the store
needs FLAC written with no libFLAC. Built with its CRC checks on: they
are what makes its seek without a SEEKTABLE (a binary search over frame
headers) safe on soundfile's files, which carry none.
