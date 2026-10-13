#!/usr/bin/env python3
"""The pins of native/tests/qrss/test_simfile.cpp, from the Python reference.

Writes native/tests/qrss/simfile_pins.hpp, which is committed so the ctest
needs no Python. Re-run it from the repository root whenever
sstvae.qrss.channel.save_npz/load_npz change on purpose:

    python native/tests/qrss/gen_simfile_pins.py

Each case is a small file in qrss_simulate.py's .npz form (np.savez with
save_npz's keywords, in its order) or a broken one, and what
channel.load_npz makes of it: the receiver's part of the Sim (fs,
t0_index, q, the frame's name, fe) or the exception's class and str().
The ctest reads every file through simfile::parse. Two cases are files the
reference reads and the port refuses on purpose (complex128 fe, a
compressed archive); the pins record the reference's success and the test
its own refusal.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("simfile_pins.hpp")


def savez(**members) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **members)
    return buf.getvalue()


def sim_members(frame="l-short", fe=None, fs=4000, t0_index=48000.25, q=1954752, optional=True, drop=(),
                **override):
    """save_npz's call: fe, fs, t0_index, q, frame, the truth scalars, the truth arrays."""
    if fe is None:
        fe = np.array([1 + 2j, -0.5j, 3.25, 1e-30 - 7j, -0.0 + 0.0j], np.complex64)
    arrays = {"truth_u": np.array([0.5, -1.0]), "truth_f_hz": np.array([1500.0, 1500.25]),
              "truth_t_pos_s": np.array([0.0, 0.125]), "truth_impulse_mask": np.array([False, True])}
    if optional:
        arrays["truth_agc_gain"] = np.array([1.0, 0.75], np.float32)
        arrays["truth_u_data"] = np.array([0.25, -0.25])
    m = dict(fe=fe, fs=fs, t0_index=t0_index, q=q, frame=frame, truth_noise_var_fe=0.125,
             truth_signal_power=2.0, truth_time_scale=1.0 + 2 ** -40, **arrays)
    m.update(override)
    for k in drop:
        del m[k]
    return m


def cases() -> list[tuple[str, bytes]]:
    out = [
        ("sim_l", savez(**sim_members())),
        ("sim_ce", savez(**sim_members(frame="short", optional=False, fs=np.int32(4000), q=np.uint32(1954753)))),
        ("fs_float", savez(**sim_members(fs=4000.75, t0_index=np.float32(0.1)))),
        ("frame_unicode", savez(**sim_members(frame="l-sh\u00f6rt\U0001f4e1"))),
        ("fe_empty", savez(**sim_members(fe=np.zeros(0, np.complex64)))),
        ("no_truth_u", savez(**sim_members(drop=("truth_u",)))),
        ("no_signal_power", savez(**sim_members(drop=("truth_signal_power",)))),
        ("no_fe", savez(**sim_members(drop=("fe",)))),
        ("no_frame", savez(**sim_members(drop=("frame",)))),
        ("t0_complex", savez(**sim_members(t0_index=np.complex128(1 + 1j)))),
        ("fs_1d", savez(**sim_members(fs=np.array([4000, 4000])))),
        ("noise_var_1d", savez(**sim_members(truth_noise_var_fe=np.array([0.5, 0.5])))),
        ("q_complex", savez(**sim_members(q=np.complex64(3)))),
        ("q_nan", savez(**sim_members(q=np.float64("nan")))),
        ("q_inf", savez(**sim_members(q=np.float64("inf")))),
        ("empty", b""),
        ("pickle", b"hello, this is not an archive\n"),
        ("npy", _npy()),
        ("not_zip", b"PK\x03\x04" + bytes(range(64))),
        ("bad_crc", _flipped()),
        # The reference reads these; the port refuses them (simfile.hpp).
        ("fe_c16", savez(**sim_members(fe=np.array([1 + 2j, 3j], np.complex128)))),
        ("compressed", _compressed()),
    ]
    return out


def _npy() -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.arange(4))
    return buf.getvalue()


def _flipped() -> bytes:
    """sim_l with one bit of fe's data flipped: zipfile's CRC check refuses it."""
    data = bytearray(savez(**sim_members()))
    at = bytes(data).index(np.array([1 + 2j], np.complex64).tobytes())
    data[at] ^= 0x01
    return bytes(data)


def _compressed() -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **sim_members(frame="short"))
    return buf.getvalue()


def outcome(data: bytes):
    from sstvae.qrss import channel

    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as d:
            # load_npz without lframe.get_any (the frame's name is the
            # port's output; the receive subcommand resolves it).
            for k in ("u", "f_hz", "t_pos_s", "impulse_mask"):
                d[f"truth_{k}"]
            float(d["truth_noise_var_fe"])
            if "truth_agc_gain" in d:
                d["truth_agc_gain"]
            float(d["truth_signal_power"])
            float(d["truth_time_scale"])
            if "truth_u_data" in d:
                d["truth_u_data"]
            sim = (d["fe"], int(d["fs"]), float(d["t0_index"]), int(d["q"]), str(d["frame"]))
    except Exception as e:  # noqa: BLE001 -- the class is the pin
        cls = type(e).__name__
        if type(e).__module__ == "zipfile":
            cls = "zipfile." + cls
        return ("error", cls, str(e))
    assert channel.load_npz.__doc__   # the reference this mirrors
    return ("ok",) + sim


def c_str(s: str) -> str:
    out = []
    for b in s.encode("utf-8"):
        out.append(chr(b) if 0x20 <= b < 0x7f and chr(b) not in '"\\?' else f"\\x{b:02x}\"\"")
    return '"' + "".join(out) + '"'


def main() -> None:
    lines = [
        "// GENERATED by native/tests/qrss/gen_simfile_pins.py from the Python reference",
        f"// (numpy {np.__version__}, CPython {sys.version.split()[0]}). Do not edit.",
        "",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace simfile_pins {",
        "",
        "struct Pin {",
        "    const char* name;",
        "    const char* const* hex;",
        "    std::size_t n_hex;",
        "    bool ok;              // channel.load_npz reads it",
        "    const char* cls;      // else its exception's class",
        "    const char* msg;      // and str()",
        "    std::int64_t fs;",
        "    double t0_index;",
        "    std::int64_t q;",
        "    const char* frame;    // UTF-8",
        "    const float* fe;      // re, im, re, im ...",
        "    std::size_t n_fe;     // complex values",
        "};",
        "",
    ]
    table = []
    for name, data in cases():
        hx = data.hex()
        chunks = [hx[i:i + 96] for i in range(0, len(hx), 96)] or [""]
        lines.append(f"inline const char* const {name.upper()}_HEX[] = {{")
        lines += [f'    "{c}",' for c in chunks]
        lines.append("};")
        res = outcome(data)
        if res[0] == "ok":
            fe, fs, t0, q, frame = res[1:]
            fe = np.asarray(fe).astype(np.complex64).ravel()
            vals = [x for z in fe for x in (z.real, z.imag)]
            if vals:
                lines.append(f"inline const float {name.upper()}_FE[] = {{" +
                             ", ".join(float(np.float32(v)).hex() + "f" for v in vals) + "};")
                fe_ref = f"{name.upper()}_FE"
            else:
                fe_ref = "nullptr"
            table.append(f'    {{"{name}", {name.upper()}_HEX, std::size({name.upper()}_HEX), true, nullptr, '
                         f'nullptr, {fs}, {float(t0).hex()}, {q}, {c_str(frame)}, {fe_ref}, {fe.size}}},')
        else:
            _, cls, msg = res
            table.append(f'    {{"{name}", {name.upper()}_HEX, std::size({name.upper()}_HEX), false, '
                         f'{c_str(cls)}, {c_str(msg)}, 0, 0.0, 0, nullptr, nullptr, 0}},')
        lines.append("")
    lines.append("inline const Pin PINS[] = {")
    lines += table
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace simfile_pins")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
