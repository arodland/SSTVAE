"""QRSSTVAE: slow, narrow, multi-pass picture beacons built on SSTVAE.

One group of 50,600 SSTVAE latents goes out as a single constant-envelope
carrier (waveform CE) about 50 Hz wide over a ~30-minute slot, and a
receiver averages every pass of the same picture into one accumulator.
See `docs/qrss/design.md` for the build design and the spec it follows.

Modules marked *format* in the design (`constants`, `sequences`,
`precoder`, `frame`, `morse`, `picture`, `header`, `polar`, `ce`, `tx`,
`beaconfile`, `si5351`) define what goes on the air: closed-form
sequences only, no seeded random generators. This package imports
nothing heavy at import time; import the submodule you need.
"""

__all__ = [
    "constants", "sequences", "precoder", "frame", "morse", "picture",
    "types", "polar", "header", "ce", "tx", "beaconfile", "si5351",
    "channel", "frontend", "acquire", "track", "demod", "cwid",
    "receiver", "store", "associate", "render", "em",
]
