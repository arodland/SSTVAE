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
