"""sstvae.qrss.tx -> sstvae_native.qrss.tx (native/core/qrss/tx): the slot
functions and the pass's audio, transmit_audio and _transmit_audio_l
(natively a PassAudio rendered whole). With these in place qrss_transmit.py
and every test that transmits run on the C++ transmitter, so
test_qrss_smoke.py is the C++-transmit -> Python-receive chain.

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
        (t, "transmit_audio"): n.transmit_audio,
        (t, "_transmit_audio_l"): n._transmit_audio_l,
    }
