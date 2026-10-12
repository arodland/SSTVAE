"""sstvae.qrss.frame and .lframe -> sstvae_native.qrss.frame / .lframe
(native/core/qrss/frame: CE and L as one FrameSpec type with a waveform).

The reference's frames stay the reference's dataclasses -- `get`, `twin`
and `plain` hand back `FrameSpec`/`LFrameSpec` objects (the module's own
preset objects where the C++ named one, so `frame.get("short") is
frame.SHORT` still holds), and `layout` a `Layout` -- but every decision
behind them is the C++'s: the validation in `__post_init__`, every
derived length and position, the name lookups, the layouts and the
assembled symbols. The native functions take the reference's spec
objects directly (by their fields), so most rows need no adapter.
"""

import functools

STAGE = "tx"


def rows(q):
    from sstvae.qrss import frame as F
    from sstvae.qrss import lframe as LF

    f, lf = q.frame, q.lframe
    CE_SPEC, L_SPEC = F.FrameSpec, LF.LFrameSpec
    PRESET_OBJECTS = {s: s for s in (*F.PRESETS.values(), *F.LEGACY.values(),
                                     *F.PROTOTYPES.values(), *LF.PRESETS.values())}

    @functools.lru_cache(maxsize=256)
    def nspec(spec):
        """The native FrameSpec of a reference one (both are immutable)."""
        return f.of(spec)

    def pyspec(n, keep=None):
        """The reference dataclass for a native spec: `keep` or the module's
        own preset object when equal to one, else a new one."""
        cls = L_SPEC if n.waveform == LF.WAVEFORM_L else CE_SPEC
        s = cls(n.name, n.n_pre, n.n_hdr, n.n_data, n.cw_after, n.hdr_rho)
        if keep is not None and s == keep:
            return keep
        return PRESET_OBJECTS.get(s, s)

    def post_init(self):
        n = f.make(self.waveform, self.name, self.n_pre, self.n_hdr, self.n_data,
                   self.cw_after, self.hdr_rho)
        object.__setattr__(self, "cw_after", n.cw_after)
        object.__setattr__(self, "hdr_rho", n.hdr_rho)

    def prop(name):
        return property(lambda self: getattr(nspec(self), name))

    def get(name_or_spec):
        if isinstance(name_or_spec, CE_SPEC):
            return name_or_spec
        return pyspec(f.get(name_or_spec))

    def get_l(name_or_spec):
        if isinstance(name_or_spec, L_SPEC):
            return name_or_spec
        return pyspec(lf.get(name_or_spec))

    def get_any(name_or_spec):
        if isinstance(name_or_spec, (L_SPEC, CE_SPEC)):
            return name_or_spec
        return pyspec(lf.get_any(name_or_spec))

    def twin(spec, waveform):
        return pyspec(lf.twin(spec, waveform), keep=spec)

    def plain(spec):
        return pyspec(f.plain(spec), keep=spec)

    @functools.lru_cache(maxsize=16)
    def layout(spec):
        return F.Layout(**f.layout(spec))

    @functools.lru_cache(maxsize=16)
    def layout_l(spec):
        return LF.Layout(**lf.layout(spec))

    t, tl = "sstvae.qrss.frame", "sstvae.qrss.lframe"
    out = {
        (t, "_data_symbols"): f._data_symbols,
        (t, "get"): get,
        (t, "plain"): plain,
        (t, "layout"): layout,
        (t, "spread_signs"): f.spread_signs,
        (t, "spread_symbols"): f.spread_symbols,
        (t, "assemble"): f.assemble,
        (t, "disassemble"): f.disassemble,
        (tl, "N_HDR_SLOTS"): lf.N_HDR_SLOTS,
        (tl, "_refs_below"): lf._refs_below,
        (tl, "_data_symbols"): lf._data_symbols,
        (tl, "get"): get_l,
        (tl, "get_any"): get_any,
        (tl, "twin"): twin,
        (tl, "layout"): layout_l,
        (tl, "pair"): lf.pair,
        (tl, "unpair"): lf.unpair,
        (tl, "data_latents"): lf.data_latents,
        (tl, "assemble"): lf.assemble,
        (tl, "disassemble"): lf.disassemble,
    }
    # The dataclasses themselves: validation and every derived property.
    derived = ("has_header", "n_data_sym", "n_sym", "n_win", "n_pos", "n_ref", "duration_s",
               "win_start_pos", "keyed_end_pos", "keyed_start_pos")
    for cls, target, extra in ((CE_SPEC, f"{t}.FrameSpec", ()),
                               (L_SPEC, f"{tl}.LFrameSpec", ("n_data_payload",))):
        out[(target, "__post_init__")] = post_init
        for name in derived + extra:
            out[(target, name)] = prop(name)
        out[(target, "pos_of")] = lambda self, s: nspec(self).pos_of(s)
        out[(target, "time_s")] = lambda self, pos: nspec(self).time_s(pos)
        out[(target, "window_spans_s")] = lambda self: nspec(self).window_spans_s()
    return out
