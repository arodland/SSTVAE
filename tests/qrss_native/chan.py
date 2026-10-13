"""The narrow channel's front-end half of sstvae.qrss.track ->
sstvae_native.qrss.chan (native/core/qrss/chan).

`make_chan` returns the reference's ChanCapture (`_rx.to_ref`), and
`ChanCapture.noise_psd_at` / `noise_floor` become the C++'s, keeping their
cache in the object's own `_psd` as the reference does. The noise PSD they
compute it with is whatever `frontend.noise_psd` is when they are called:
the C++ (directly, no round trip through Python) while it is the native
function, and any replacement -- the rules suite spies on its arguments
with a monkeypatch -- called with the reference's arguments otherwise.
"""

STAGE = "rx"


def rows(q):
    from . import _rx

    c = q.chan
    native_noise_psd = q.frontend.noise_psd

    def _noise_psd():
        from sstvae.qrss import frontend

        fn = frontend.noise_psd
        return None if fn is native_noise_psd else fn

    def make_chan(fe, keep, t0_index_fe, fs_fe, f_audio_hz, t_lo_s, t_hi_s, exclude_hz=()):
        return _rx.to_ref(c.make_chan(fe, keep, t0_index_fe, fs_fe, f_audio_hz, t_lo_s, t_hi_s, exclude_hz))

    def noise_psd_at(self, t_s):
        return c.noise_psd_at(self, t_s, _noise_psd())

    def noise_floor(self):
        return c.noise_floor(self, _noise_psd())

    t = "sstvae.qrss.track"
    return {
        (t, "make_chan"): make_chan,
        (t + ".ChanCapture", "noise_psd_at"): noise_psd_at,
        (t + ".ChanCapture", "noise_floor"): noise_floor,
    }
