"""Symbol extraction, weights and header LLRs for waveform L (the L
counterpart of `demod`; docs/qrss/waveform-l.md).

Not a format module.

BEING BUILT (2026-10-12): only the pinned constants below exist so far.
`extract_l` and `calibrate_l` follow `demod.extract`'s contract.
"""

from __future__ import annotations

# Class gains of the L clipper relative to the reference symbols' gain (the
# tracker's u is the reference gain: a reference a arrives as u a). Provisional,
# from noiseless loopbacks (`lwave.loopback_stats`) of L_SHORT and L_MEDIUM with
# Gaussian latents, 6 frames: data/ref 0.878-0.890, header/ref 0.986-1.000.
# G_DATA_L is spec 2.2's figure for real v5 latents.
G_REF_L = 1.0
G_HDR_L = 0.995
G_DATA_L = 0.881

# The pilot's gain through the clipper, against the unclipped pilot
# (`lwave.pilot(tau) / lwave.A_ID`), per section of the frame. Where the pilot
# runs at average power (lead-in, window units 8..183: W = 1) nothing clips it;
# elsewhere it shares the clipper with the symbols. Provisional, same 6 frames:
# the preamble and header are the same symbols in every frame; the data figure
# moves with the latents (0.703-0.710 Gaussian, 0.672-0.681 tanh).
PILOT_GAIN_PRE = 0.656
PILOT_GAIN_HDR = 0.666
PILOT_GAIN_DATA = 0.703
PILOT_GAIN_ID = 1.0
# The references' own gain on the unit-average-power baseband (1.000-1.015).
G_REF_ABS = 1.005

# Clipper distortion at the matched filter, per latent (l units), same frames:
# data SINR 12.05-12.36 dB (Gaussian), references 16.1-16.8 dB.
D_PASS_L = 0.060
D_REF_L = 0.022
D_HDR_L = 0.0205
