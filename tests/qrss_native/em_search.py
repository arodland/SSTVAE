"""The template search and retroactive detection of sstvae.qrss.em ->
sstvae_native.qrss.em_search (native/core/qrss/em_search): `_candidates`,
`_chunk_template`, `_search_chan`, `_search_input`, `_from_passband`,
`template_search`, `receive_template` and `retro_detect`. (The EM loop --
`pass_path`, `rereceive`, `em_refine` -- is tests/qrss_native/em_refine.py's;
the shared pieces tests/qrss_native/em.py's.)

The C++ search, receive and retro loop call their own C++ for everything
beneath them (prepare, channel_for, make_chan, track, demod, the header,
association): no suite spies on those through these entry points. What the
suites do replace is one level up, and the rows hand it to the C++:

- **retro_detect** takes `_seams`: `template_search`, `receive_template`
  and `associate` resolved at each call from em's and associate's module
  attributes. A name still holding the native row (marked
  `__qrss_native__` here, or associate's two `associate_rows` closures of
  tests/qrss_native/_store.py) is the C++; anything else -- a test's
  monkeypatch (test_qrss_lreceiver.py stubs both search and receive), or
  the reference when this file's rows are not applied -- is called in
  Python with the reference's arguments, and what it returns is what the
  loop goes on with (and what retro_detect returns: the seam's own pass,
  key and rule objects).
- **A store that is not the native Store** (test_qrss_native_passband.py
  hands retro_detect a SimpleNamespace with a stubbed template_search)
  goes to the reference's own function, captured when the rows were
  built, which then calls these rows by module name as it always did.
  `_candidates` falls back the same way.
- `pb` may be any object with slots(dur) / capture(q, dur) (a
  SimpleNamespace in test_qrss_lreceiver.py): the C++ reads a
  PassbandStore's directory itself and calls anything else in Python.

Records come back as the reference's (`_rx.to_ref`): Detection, Prepared,
PassResult; `_from_passband` the reference's frontend.Capture;
`_search_input` hands a Prepared argument back as itself, as
receiver.prepare does.
"""

STAGE = "rx"


def _is_assoc_row(fn, name):
    return (getattr(fn, "__module__", None) == "qrss_native._store"
            and getattr(fn, "__qualname__", None) == f"associate_rows.<locals>.{name}")


def rows(q):
    import sstvae.qrss.em as ref

    from . import _rx

    n = q.em_search
    to_ref = _rx.to_ref
    native_store = q.store.Store
    # the reference's own functions, for a store the C++ cannot read
    r_candidates = ref._candidates
    r_template_search = ref.template_search
    r_receive_template = ref.receive_template
    r_retro_detect = ref.retro_detect

    def _native(fn):
        return getattr(fn, "__qrss_native__", False)

    def seams():
        """What retro_detect is to call in Python: the names whose attribute
        is not the native row now."""
        from sstvae.qrss import associate as asc
        from sstvae.qrss import em

        s = {}
        if not _native(em.template_search):
            s["template_search"] = em.template_search
        if not _native(em.receive_template):
            fn = em.receive_template
            s["receive_template"] = lambda store, key, cap, spec, det=None, estimator="joint", segment=None: fn(
                store, key, cap, spec, det=to_ref(det), estimator=estimator, segment=segment)
        if not (_is_assoc_row(asc.associate, "associate") and _is_assoc_row(asc._soft_header, "_soft_header")):
            fa = asc.associate
            s["associate"] = lambda p, store: fa(to_ref(p), store)
        return s or None

    def _candidates(store, acc, extra_hz=()):
        if not isinstance(store, native_store):
            return r_candidates(store, acc, extra_hz)
        return n._candidates(store, acc, extra_hz)

    def _search_input(cap, f_mix_hz):
        return to_ref(n._search_input(cap, f_mix_hz))

    def _from_passband(cap, q, spec):
        return n._from_passband(cap, q, spec)

    def template_search(store, key, cap, spec, *, freqs=(), segments=None, tau_s=ref.TS_TAU_S,
                        df_hz=ref.TS_DF_HZ, ppm_steps=ref.TS_PPM_STEPS, z_accept=ref.Z_ACCEPT,
                        f_mix_hz=None, q=None, return_stats=False):
        kw = dict(freqs=freqs, segments=segments, tau_s=tau_s, df_hz=df_hz, ppm_steps=ppm_steps,
                  z_accept=z_accept, f_mix_hz=f_mix_hz, q=q, return_stats=return_stats)
        if not isinstance(store, native_store):
            return r_template_search(store, key, cap, spec, **kw)
        return to_ref(n.template_search(store, key, cap, spec, **kw))

    def receive_template(store, key, cap, spec, det=None, estimator="joint", segment=None, **search):
        if not isinstance(store, native_store):
            return r_receive_template(store, key, cap, spec, det, estimator, segment, **search)
        return to_ref(n.receive_template(store, key, cap, spec, det, estimator, segment, **search))

    def retro_detect(store, key, pb, spec, *, slots=None, estimator="joint", log=None, **search):
        if not isinstance(store, native_store):
            return r_retro_detect(store, key, pb, spec, slots=slots, estimator=estimator, log=log, **search)
        out = n.retro_detect(store, key, pb, spec, slots=slots, estimator=estimator, log=log,
                             _seams=seams(), **search)
        return [(to_ref(p), k, rule) for p, k, rule in out]

    for fn in (template_search, receive_template, retro_detect):
        fn.__qrss_native__ = True

    t = "sstvae.qrss.em"
    return {
        (t, "_candidates"): _candidates,
        (t, "_chunk_template"): n._chunk_template,
        (t, "_search_chan"): n._search_chan,
        (t, "_search_input"): _search_input,
        (t, "_from_passband"): _from_passband,
        (t, "template_search"): template_search,
        (t, "receive_template"): receive_template,
        (t, "retro_detect"): retro_detect,
    }
