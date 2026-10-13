"""sstvae.qrss.tx (the slot functions) -> sstvae_native.qrss.tx
(native/core/qrss/tx). transmit_audio and _transmit_audio_l stay Python
until the CE and L audio are ported; they reach these through the
module's globals, so they run on the native slots.

Sources and headers are the reference's dataclasses, passed through. The
one HeaderFields that comes back (a beacon file's own, from
_segment_and_header) arrives as a tuple and is wrapped here.
"""

STAGE = "tx"


def rows(q):
    from sstvae.qrss.header import HeaderFields

    n = q.tx
    t = "sstvae.qrss.tx"

    def _segment_and_header(src, segment, header):
        seg, h, bits, keying, ook = n._segment_and_header(src, segment, header)
        if isinstance(h, tuple):
            h = HeaderFields(*h)
        return seg, h, bits, keying, ook

    return {
        (t, "_segment_and_header"): _segment_and_header,
        (t, "slot_air_latents"): n.slot_air_latents,
        (t, "slot_symbols"): n.slot_symbols,
        (t, "slot_keying"): n.slot_keying,
        (t, "slot_samples"): n.slot_samples,
        (t, "check_pilot"): n.check_pilot,
    }
