"""sstvae.qrss.picture -> sstvae_native.qrss.picture (native/core/qrss/picture).

StoredPicture stays the reference's dataclass: the native functions take
any object with its attributes and return its fields as a dict, and the
adapters here make the dataclass. Its methods are replaced by plain
functions (a native function assigned to a class does not bind as a
method). `codec_id_from_onnx` is codec glue and is not ported.
"""

STAGE = "tx"


def rows(q):
    from sstvae.qrss import picture as P

    n = q.picture
    t, c = "sstvae.qrss.picture", "sstvae.qrss.picture.StoredPicture"

    def from_latents(cls, full, codec_id, mode):
        return cls(**n.from_latents(full, codec_id, mode))

    def check(self):
        n.check(self)

    def load_qrsp(path):
        return P.StoredPicture(**n.load_qrsp(path))

    return {
        (t, "_check_group"): n._check_group,
        (t, "_check_mode"): n._check_mode,
        (t, "air_to_canonical"): n.air_to_canonical,
        (t, "never_sent"): n.never_sent,
        (t, "prepare_segment"): n.prepare_segment,
        (t, "air_values"): n.air_values,
        (t, "canonical_from_air"): n.canonical_from_air,
        (t, "picture_id"): n.picture_id,
        (t, "to_int8"): n.to_int8,
        (t, "from_int8"): n.from_int8,
        (t, "save_qrsp"): n.save_qrsp,
        (t, "load_qrsp"): load_qrsp,
        (c, "from_latents"): classmethod(from_latents),
        (c, "check"): check,
        (c, "id_hex"): property(lambda self: n.id_hex(self.picture_id)),
    }
