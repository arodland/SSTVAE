"""The beacon file: one segment of a stored picture, ready for a
clock-chip beacon (design 2.9, spec 2.5 "Storing a picture").

A **format module**: a microcontroller reads this layout directly, so it
is fixed byte for byte. Little-endian throughout.

    offset  size  field
         0     4  magic b"QRSB"
         4     1  file version = 1
         5     1  waveform (0 = CE)
         6     1  mode
         7     1  segment
         8     4  picture ID (u32)
        12     2  codec ID (u16)
        14     8  callsign, ASCII, space padded
        22     2  grid code (u16, 32767 = none)
        24     4  n_latents (u32), 50600
        28     2  int8 scale (u16), 20
        30     1  flags: bit 0 = callsign keying (0 FSK, 1 OOK); other bits 0
        31     1  reserved = 0
        32    24  callsign keying, 192 units, 1 bit each MSB first (1 = key-down)
        56   310  header coded bits, 2474 bits MSB first, last 6 pad bits 0
       366     n  int8 latents in air order, neither scrambled nor precoded
     366+n     4  CRC-32 (zlib) of every preceding byte

A full segment is 50,970 bytes, which fits a 24LC512. The keying mask is
stored so a microcontroller needs no Morse table; the beacon itself
computes the preamble, references, scrambler, precoder, phase and window
phase every pass.
"""

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

from .constants import CW_UNITS, INT8_SCALE, N_HDR_BITS, WAVEFORM_CE
from .header import HeaderFields, encode, grid_decode, grid_encode
from .morse import check_callsign, keying_units
from .picture import SENT, StoredPicture, air_values, to_int8

MAGIC = b"QRSB"
FILE_VERSION = 1
FLAG_OOK = 0x01
_HEAD = struct.Struct("<4sBBBBIH8sHIHBB")        # offsets 0..31
HEAD_BYTES = _HEAD.size                           # 32
KEYING_BYTES = CW_UNITS // 8                      # 24
HDR_BYTES = -(-N_HDR_BITS // 8)                   # 310
LATENT_OFFSET = HEAD_BYTES + KEYING_BYTES + HDR_BYTES   # 366
assert LATENT_OFFSET == 366


def file_size(n_latents: int = SENT) -> int:
    """Bytes in a beacon file of n latents (50,970 for a full segment)."""
    return LATENT_OFFSET + int(n_latents) + 4


@dataclass
class BeaconFile:
    """One segment as a beacon sends it."""
    waveform: int
    mode: int
    segment: int
    picture_id: int
    codec_id: int
    callsign: str
    grid: str | None
    ook: bool
    keying: np.ndarray = field(repr=False)        # uint8[192]
    hdr_bits: np.ndarray = field(repr=False)      # uint8[2474]
    latents_i8: np.ndarray = field(repr=False)    # int8[n], air order

    def header_fields(self) -> HeaderFields:
        return HeaderFields(callsign=self.callsign, grid=self.grid,
                            picture_id=self.picture_id, mode=self.mode,
                            segment=self.segment, codec_id=self.codec_id)

    def check(self) -> None:
        """ValueError unless every field agrees with every other."""
        if self.waveform != WAVEFORM_CE:
            raise ValueError(f"waveform {self.waveform} is not CE ({WAVEFORM_CE})")
        call = check_callsign(self.callsign)
        if call != self.callsign:
            raise ValueError(f"callsign {self.callsign!r} has trailing spaces")
        k = np.asarray(self.keying)
        if k.shape != (CW_UNITS,) or not np.array_equal(k, keying_units(call)):
            raise ValueError(f"keying mask does not spell {call!r} in Morse")
        b = np.asarray(self.hdr_bits)
        if b.shape != (N_HDR_BITS,) or not np.array_equal(b, encode(self.header_fields())):
            raise ValueError("header coded bits do not encode the file's header fields")
        q = np.asarray(self.latents_i8)
        if q.dtype != np.int8 or q.ndim != 1 or not 0 < len(q) <= SENT:
            raise ValueError(f"latents must be 1..{SENT} int8 values, not {q.dtype}{q.shape}")
        if np.any(q == -128):
            raise ValueError("int8 latents are clipped to +-127; -128 is not a valid value")

    def to_bytes(self) -> bytes:
        self.check()
        call = self.callsign.ljust(8).encode("ascii")
        head = _HEAD.pack(MAGIC, FILE_VERSION, self.waveform, self.mode, self.segment,
                          self.picture_id, self.codec_id, call, grid_encode(self.grid),
                          len(self.latents_i8), INT8_SCALE,
                          FLAG_OOK if self.ook else 0, 0)
        keying = np.packbits(np.asarray(self.keying, np.uint8)).tobytes()
        hdr = np.packbits(np.asarray(self.hdr_bits, np.uint8)).tobytes()   # pads with 0
        body = head + keying + hdr + np.asarray(self.latents_i8, np.int8).tobytes()
        return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BeaconFile":
        data = bytes(data)
        if len(data) < LATENT_OFFSET + 5:
            raise ValueError(f"beacon file too short ({len(data)} bytes)")
        if data[:4] != MAGIC:
            raise ValueError(f"not a QRSSTVAE beacon file (magic {data[:4]!r})")
        (crc,) = struct.unpack("<I", data[-4:])
        if zlib.crc32(data[:-4]) & 0xFFFFFFFF != crc:
            raise ValueError("beacon file CRC-32 mismatch")
        (_, version, waveform, mode, segment, pid, codec_id, call, grid, n_lat, scale,
         flags, reserved) = _HEAD.unpack(data[:HEAD_BYTES])
        if version != FILE_VERSION:
            raise ValueError(f"beacon file version {version}, expected {FILE_VERSION}")
        if scale != INT8_SCALE:
            raise ValueError(f"int8 scale {scale}, expected {INT8_SCALE}")
        if flags & ~FLAG_OOK or reserved:
            raise ValueError(f"unknown flags {flags:#04x} or reserved byte {reserved}")
        if len(data) != file_size(n_lat):
            raise ValueError(f"{len(data)} bytes for {n_lat} latents, "
                             f"expected {file_size(n_lat)}")
        try:
            callsign = call.decode("ascii").rstrip(" ")
        except UnicodeDecodeError:
            raise ValueError(f"callsign bytes {call!r} are not ASCII") from None
        o = HEAD_BYTES
        keying = np.unpackbits(np.frombuffer(data[o:o + KEYING_BYTES], np.uint8))
        o += KEYING_BYTES
        hdr_all = np.unpackbits(np.frombuffer(data[o:o + HDR_BYTES], np.uint8))
        if np.any(hdr_all[N_HDR_BITS:]):
            raise ValueError("header pad bits are not zero")
        o += HDR_BYTES
        bf = cls(waveform=waveform, mode=mode, segment=segment, picture_id=pid,
                 codec_id=codec_id, callsign=callsign, grid=grid_decode(grid),
                 ook=bool(flags & FLAG_OOK), keying=keying.astype(np.uint8),
                 hdr_bits=hdr_all[:N_HDR_BITS].astype(np.uint8),
                 latents_i8=np.frombuffer(data[o:o + n_lat], np.int8).copy())
        bf.check()
        return bf


def write(path, bf: BeaconFile) -> None:
    with open(path, "wb") as f:
        f.write(bf.to_bytes())


def read(path) -> BeaconFile:
    """Read and verify a beacon file: ValueError on CRC, magic or keying mismatch."""
    with open(path, "rb") as f:
        return BeaconFile.from_bytes(f.read())


def from_picture(sp: StoredPicture, segment: int, header: HeaderFields,
                 ook: bool = False) -> BeaconFile:
    """The beacon file for one segment of a stored picture.

    `header` supplies the callsign and grid; its picture ID, mode, segment
    and codec ID must match the picture's.
    """
    sp.check()
    segment = int(segment)
    if not 0 <= segment <= sp.mode:
        raise ValueError(f"segment must be 0..{sp.mode}, not {segment}")
    want = (sp.picture_id, sp.mode, segment, sp.codec_id)
    got = (header.picture_id, header.mode, header.segment, header.codec_id)
    if got != want:
        raise ValueError(f"header (picture_id, mode, segment, codec_id) = {got} "
                         f"does not match the picture's {want}")
    call = check_callsign(header.callsign)
    bf = BeaconFile(waveform=WAVEFORM_CE, mode=sp.mode, segment=segment,
                    picture_id=sp.picture_id, codec_id=sp.codec_id, callsign=call,
                    grid=header.grid, ook=bool(ook), keying=keying_units(call),
                    hdr_bits=encode(header),
                    latents_i8=to_int8(air_values(sp.segs[segment], segment)))
    bf.check()
    return bf
