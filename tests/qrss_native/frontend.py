"""The receiver's front end, sstvae.qrss.frontend -> sstvae_native.qrss.frontend
(native/core/qrss/frontend).

Everything but the passband store (stage 3) and the `Capture` record:
the audio <-> FE conversions, the blanker and inverse AGC, the
channeliser and the noise floor with the lru_cached tables behind it.
The cached helpers are replaced too, so the noise floor's masks are the
C++'s whichever way they are reached (`noise_psd`, or a test calling
`_fe_flat` directly). `_FE_TAPS` is filters_rx.py's row. `live.StreamFE`
has a C++ twin (frontend.StreamFE) for the parity test only: the live
listener is stage 3's.
"""

STAGE = "rx"


def rows(q):
    f = q.frontend
    t = "sstvae.qrss.frontend"
    names = ("audio_to_fe", "fe_to_audio", "fe_hz", "running_median", "_interp_to_samples", "blank",
             "normalise", "quantise_mix", "mix_phasor", "channelise", "channelise_keep",
             "_noise_response", "_fold_power", "_fe_power", "_fe_flat", "_fe_response", "noise_psd")
    return {(t, name): getattr(f, name, None) for name in names}
