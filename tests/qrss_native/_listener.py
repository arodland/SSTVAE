"""Builds the C++ live listener (sstvae_native.qrss.listener, native/core/
qrss/listener) the way a test builds the reference's, for
tests/test_qrss_native_listener.py to run the two side by side.

Not a row module (the leading underscore keeps `apply` from loading it):
no row replaces live.LiveListener. The reference's tests patch the module
functions its listener calls (receiver.candidates / .detect /
.receive_slot, associate.associate) and set module constants
(DIAL_GUARD_S); the C++ listener cannot see a Python module, so it is
handed seams that look each function up at every call:

- `receivers()`: the live receive, receiver.<name>(...) as _refresh calls it;
- `finish(q)`: the end-of-slot receive, the C++ finish_receive's decisions
  (live.cpp's binding, over the Python objects) calling the module's
  receivers, with numpy's fill (`_live.noise()`), so a capture is the
  reference's bit for bit;
- `associate()`: associate.associate(p, store), for a test that replaces it;
- `CodecRenderer`: a Python codec (render.load's, or a test's fake) behind
  the Renderer seam, called as live._render calls render.load,
  check_codec_id and render;
- `frozen_time(monkeypatch, t)`: the reference's time.time() fixed, and
  the same clock for the C++ (state.json's "updated", the log's time
  before the first step, the "(N s)" of a receive).
"""

from __future__ import annotations

import time as _time
from types import SimpleNamespace

import numpy as np

from . import _live


def receivers():
    """The live receive seams: sstvae.qrss.receiver's functions looked up at each call."""
    from sstvae.qrss import receiver as RX

    return dict(candidates=lambda *a, **k: RX.candidates(*a, **k),
                detect=lambda *a, **k: RX.detect(*a, **k),
                receive_slot=lambda *a, **k: RX.receive_slot(*a, **k))


def finish(q):
    """The end-of-slot seam: the native finish_receive (its decisions in
    C++) over the module's current receivers, with numpy's noise."""
    n = q.live

    def finish_receive(x, ok, origin, slot, spec, estimator, dets, hints=()):
        return n.finish_receive(x, ok, origin, slot, spec, estimator, dets, hints, _live.noise(), False)

    return finish_receive


def associate():
    """associate.associate(p, store), looked up at each call."""
    import sstvae.qrss.associate as assoc

    return lambda p, store: assoc.associate(p, store)


class CodecRenderer:
    """A Python codec behind the C++ listener's Renderer seam: load is
    live._codec's render.load (once; its failure's text, never retried by
    the listener), check is render.check_codec_id's refusal, render is
    render.render's picture as (height, width, 3) uint8."""

    def __init__(self, codec=None, precision="fp32", model=None):
        self.codec, self.precision, self.model = codec, precision, model

    def load(self, codec_id, warn):
        if self.codec is None:
            from sstvae.qrss import render

            try:
                self.codec = render.load(self.precision, self.model, codec_id=codec_id, warn=warn)
            except BaseException as e:      # CodecMismatch is a SystemExit
                return str(e) or type(e).__name__
        return None

    def check(self, codec_id):
        from sstvae.qrss import render

        try:
            render.check_codec_id(self.codec, codec_id)
        except SystemExit as e:
            return str(e)
        return None

    def render(self, S, W, mode, codec_id):
        from sstvae.qrss import render

        return np.asarray(render.render(self.codec, S, W, mode).convert("RGB"))


def frozen_time(monkeypatch, t):
    """The reference's time.time() fixed at t (sleep and monotonic the real
    ones); returns the same clock for the C++ listener."""
    from sstvae.qrss import live

    monkeypatch.setattr(live, "time", SimpleNamespace(time=lambda: t, sleep=_time.sleep,
                                                      monotonic=_time.monotonic))
    return lambda: t


def listener(native, state_dir, cfg=None, store=None, passband=None, log=None, *, stubbed=True,
             clock=None, renderer=None, stub_associate=False, **kw):
    """The C++ listener as `live.LiveListener(state_dir, cfg, store, passband, log)`:
    with `stubbed`, every receive goes through the module's (patched)
    receivers; else the C++ receiver. DIAL_GUARD_S is the module's now."""
    from sstvae.qrss import live

    seams = {}
    if stubbed:
        seams.update(receivers())
        seams["finish"] = finish(native.qrss)
    if stub_associate:
        seams["associate"] = associate()
    return native.qrss.listener.LiveListener(
        state_dir, cfg, store, passband, log, clock=clock, noise=_live.noise(), renderer=renderer,
        dial_guard=tuple(live.DIAL_GUARD_S), **seams, **kw)
