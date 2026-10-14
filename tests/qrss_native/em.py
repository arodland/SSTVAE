"""Leave-one-out EM and the template search, the shared half:
sstvae.qrss.em -> sstvae_native.qrss.em (native/core/qrss/em).

Every function this lane ports is substituted: the references
(`lmmse_latents`, `prior_from`, `em_prior`, `acc_mean_w_db`), what every
member is known to have sent (`known_header`, `known_keying`), a stored
pass's frame (`pass_spec`, `is_l`, `_ce_only`), the paths (`_psi_frame`,
`_path_of`, `_flat`) and the tail statistics (`_gauss_z`, `template_z`).
None of the suites spies on them: test_qrss_em.py's P4 smoke spies on
`rereceive` (still the reference's, a later lane's) and calls `em_prior`
beside it, which is this row -- so the spy and the loop it watches read
the same native prior. The loop and the search themselves (`pass_path`,
`rereceive`, `em_refine`, `template_search`, `receive_template`,
`retro_detect`, ...) stay Python here and call these rows by their module
names.

The records come back as the reference's (`_rx.to_ref`): EmPrior,
HeaderFields and FreqPath dataclasses, and `pass_spec` the module's own
preset FrameSpec (test_qrss_spread.py asserts `is frame.FULL_V1`). The
store and accumulator are the native ones (tests/qrss_native/store.py),
which the C++ reads directly.
"""

STAGE = "rx"


def rows(q):
    from . import _rx

    n = q.em
    to_ref = _rx.to_ref

    def prior_from(S, W, q):
        return to_ref(n.prior_from(S, W, q))

    def em_prior(acc, uid, store, q, spec):
        return to_ref(n.em_prior(acc, uid, store, q, spec))

    def known_header(store, acc, segment):
        return to_ref(n.known_header(store, acc, segment))

    def pass_spec(p):
        return to_ref(n.pass_spec(p))

    def _path_of(tr):
        return to_ref(n._path_of(tr))

    def _flat(f_hz):
        return to_ref(n._flat(f_hz))

    t = "sstvae.qrss.em"
    return {
        (t, "lmmse_latents"): n.lmmse_latents,
        (t, "prior_from"): prior_from,
        (t, "em_prior"): em_prior,
        (t, "acc_mean_w_db"): n.acc_mean_w_db,
        (t, "known_header"): known_header,
        (t, "known_keying"): n.known_keying,
        (t, "_psi_frame"): n._psi_frame,
        (t, "pass_spec"): pass_spec,
        (t, "is_l"): n.is_l,
        (t, "_ce_only"): n._ce_only,
        (t, "_path_of"): _path_of,
        (t, "_flat"): _flat,
        (t, "_gauss_z"): n._gauss_z,
        (t, "template_z"): n.template_z,
    }
