"""The CE transmit pipeline: slot symbols from a stored picture or a
beacon file, and the slot's transmit audio (design section 3).

A **format module**. A pass of segment g in slot q sends

    preamble | header (polar-coded fields, 1 in 16 a reference) |
    data = precode(air latents, q), 1 in 16 a reference

with the callsign windows inserted at the positions `frame.FrameSpec`
gives. Everything time-like comes from the FrameSpec, so the slot timing
stays one edit in `frame.py`.

A `.qrsp` source sends its fp16 values; a beacon file sends the int8
version of the same air values (`picture.to_int8`), which is identical in
every pass and 37 dB below the latents. `use_int8=True` on a `.qrsp`
reproduces what a beacon would send.

Every transmit path keys the callsign windows (spec 2.7, a legal
requirement): the keying comes from the header's callsign, or from the
beacon file's stored mask, and a frame with windows refuses to transmit
without one.
"""

import math

import numpy as np

from . import ce
from .beaconfile import BeaconFile
from .constants import FS, T_SYM
from .frame import FULL, FrameSpec, assemble
from .header import HeaderFields, encode
from .morse import keying_units
from .picture import StoredPicture, air_values, from_int8, to_int8
from .precoder import precode


def _segment_and_header(src, segment, header):
    """(segment, header fields or None, hdr bits or None, keying or None, ook flag)."""
    if isinstance(src, BeaconFile):
        if header is not None and header != src.header_fields():
            raise ValueError("a beacon file carries its own header; it does not match "
                             "the one given")
        if segment not in (None, src.segment):
            raise ValueError(f"the beacon file holds segment {src.segment}, not {segment}")
        return src.segment, src.header_fields(), np.asarray(src.hdr_bits), \
            np.asarray(src.keying), src.ook
    if not isinstance(src, StoredPicture):
        raise TypeError(f"source must be a StoredPicture or a BeaconFile, not {type(src)}")
    segment = 0 if segment is None else int(segment)
    if not 0 <= segment <= src.mode:
        raise ValueError(f"segment must be 0..{src.mode}, not {segment}")
    if header is None:
        return segment, None, None, None, False
    want = (src.picture_id, src.mode, segment, src.codec_id)
    got = (header.picture_id, header.mode, header.segment, header.codec_id)
    if got != want:
        raise ValueError(f"header (picture_id, mode, segment, codec_id) = {got} does "
                         f"not match the picture's {want}")
    return segment, header, encode(header), keying_units(header.callsign), False


def slot_air_latents(src, segment: int | None = None, use_int8: bool | None = None) -> np.ndarray:
    """float64: the air-order latents this source sends (all it holds)."""
    if isinstance(src, BeaconFile):
        if use_int8 is False:
            raise ValueError("a beacon file holds only the int8 latents")
        return from_int8(src.latents_i8)
    seg = 0 if segment is None else int(segment)
    air = air_values(src.segs[seg], seg)
    return from_int8(to_int8(air)) if use_int8 else air


def slot_symbols(src: StoredPicture | BeaconFile, q: int, spec: FrameSpec = FULL, *,
                 segment: int | None = None, header: HeaderFields | None = None,
                 use_int8: bool | None = None) -> np.ndarray:
    """float64[spec.n_sym]: the stream symbols of one pass in slot q.

    Data = precode(the first n_data air-order latents, q). The header bits
    come from `header.encode(header)` (a `.qrsp` needs `header` whenever
    the frame has a header) or from the beacon file.
    """
    seg, h, bits, _, _ = _segment_and_header(src, segment, header)
    air = slot_air_latents(src, seg, use_int8)
    if len(air) < spec.n_data:
        raise ValueError(f"frame {spec.name!r} sends {spec.n_data} latents; the source "
                         f"holds {len(air)}")
    if spec.has_header and bits is None:
        raise ValueError(f"frame {spec.name!r} has a header: pass header= for a .qrsp")
    x = precode(air[:spec.n_data], q)
    return assemble(spec, bits if spec.has_header else None, x)


def slot_keying(src, header: HeaderFields | None = None) -> np.ndarray | None:
    """uint8[192]: the callsign keying for this source, or None if it has none."""
    if isinstance(src, BeaconFile):
        return np.asarray(src.keying)
    return None if header is None else keying_units(header.callsign)


def slot_samples(spec: FrameSpec, pre_s: float = 12.0, post_s: float = 3.0,
                 fs: int = FS) -> int:
    """Samples in a slot WAV: from t0 - pre_s to the end of keying + post_s."""
    end_s = spec.keyed_end_pos * T_SYM + post_s
    return int(math.ceil((pre_s + end_s) * fs))


def transmit_audio(src, q: int, carrier_hz=1500, *, spec: FrameSpec = FULL,
                   segment: int | None = None, header: HeaderFields | None = None,
                   lead_in_s: float = 0.0, ook: bool | None = None, pre_s: float = 12.0,
                   post_s: float = 3.0, amplitude: float = 0.5,
                   chunk_s: float = 120.0) -> np.ndarray:
    """float64 8 kHz audio of one pass; sample 0 is t0 - pre_s (QH + 1 - pre_s).

    `ook` None uses the source's keying (the beacon file's flag; FSK for a
    `.qrsp`). The callsign windows are always keyed: a frame with windows
    and no callsign is refused. `carrier_hz` may be an int, a float or a
    Fraction; the mixing is exact in integer turns. Generated in chunks
    so a full slot never holds more than `chunk_s` of complex baseband.
    """
    ce.check_carrier(carrier_hz)
    seg, h, _, keying, file_ook = _segment_and_header(src, segment, header)
    if spec.n_win and keying is None:
        raise ValueError(f"frame {spec.name!r} has callsign windows: a callsign is "
                         "required to transmit (spec 2.7); pass header=")
    use_ook = file_ook if ook is None else bool(ook)
    if pre_s * FS != round(pre_s * FS):
        raise ValueError("pre_s must be a whole number of samples at 8 kHz")
    lo, _ = ce.keyed_span_pos(spec, lead_in_s)
    if -lo * T_SYM + 0.05 > pre_s:
        raise ValueError(f"pre_s = {pre_s} s does not cover the keying start "
                         f"({-lo * T_SYM:.2f} s before t0 plus the ramp)")
    sym = slot_symbols(src, q, spec, segment=seg, header=h)
    num, den = ce.carrier_fraction(carrier_hz)
    n = slot_samples(spec, pre_s, post_s)
    n_pre = round(pre_s * FS)
    step = max(1, int(chunk_s * FS))
    out = np.empty(n, dtype=np.float64)
    for i0 in range(0, n, step):
        m = min(step, n - i0)
        z = ce.baseband(sym, spec, FS, (i0 - n_pre) / FS, m, keying=keying, ook=use_ook,
                        lead_in_s=lead_in_s)
        out[i0:i0 + m] = ce.to_audio(z, FS, num, den, amplitude, n0=i0)
    return out
