"""The receiver rules, sstvae.qrss.rules -> sstvae_native.qrss.rules
(native/core/qrss/rules).

`old` reads the C environment and `set_old` writes through os.environ (so
os.environ, monkeypatch and the C++'s getenv agree); `apply` is the
native one over the same switches. `add_arguments` is argparse and stays.
"""

STAGE = "rx"


def rows(q):
    r = q.rules
    t = "sstvae.qrss.rules"
    return {
        (t, "old"): r.old,
        (t, "set_old"): r.set_old,
        (t, "apply"): r.apply,
    }
