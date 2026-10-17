"""sstvae.qrss.cwid -> sstvae_native.qrss.cwid (native/core/qrss/cwid).

Every function of the module, the reader's dynamic programme and trace
included. The C++ reads a track by attribute as the reference does and
calls back `chan.t_s` / `chan.noise_psd_at` in Python, so the track's
ChanCapture (its native rows or the reference) answers for the noise.

The module's tables `_CHARS` / `_PATS` are left as they are: the C++
carries its own copy of the same alphabet (`_patterns()`), which the
parity test holds equal to them.
"""

STAGE = "rx"


def rows(q):
    c = q.cwid
    m = "sstvae.qrss.cwid"
    return {
        (m, "_unit_sums"): c._unit_sums,
        (m, "window_soft"): c.window_soft,
        (m, "_fsk_only"): c._fsk_only,
        (m, "_ln_i0"): c._ln_i0,
        (m, "window_llr"): c.window_llr,
        (m, "_combine"): c._combine,
        (m, "decode_units"): c.decode_units,
        (m, "_patterns"): c._patterns,
        (m, "_dp"): c._dp,
        (m, "_trace"): c._trace,
        (m, "read_ml"): c.read_ml,
        (m, "read"): c.read,
        (m, "match"): c.match,
        (m, "keying_type"): c.keying_type,
    }
