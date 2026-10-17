"""Every on-air constant of QRSSTVAE waveforms CE and L (design section 2.1).

This is a **format module**: what it says is what goes on the air, so
nothing here may be computed from a seeded random generator, and nothing
here may be redefined anywhere else. Frame *lengths* are deliberately not
here: they are properties of `frame.FrameSpec`, so that the slot timing
(windows, data length) can be changed in one place.

SSTVAE's own layout numbers -- `config.TRANSMIT_LATENTS_PER_GROUP`
(50,600), `config.GROUP_LATENTS` (52,800) and `config.SNR_REF_BW_HZ`
(2500) -- are imported from `sstvae.config` where needed and never
restated.
"""

import math
from fractions import Fraction

from sstvae import config as _c

FS = _c.FS                                   # 8000, audio sample rate

# --- CE symbol clock and pulse ---------------------------------------------
# T = 485/16000 s exactly, so a symbol is an integer number of samples on
# the 16 kHz grid and every rate dividing 16000 sees an exact pulse table.
T_NUM, T_DEN = 485, 16000
T_SYM = T_NUM / T_DEN                        # 0.0303125 s
ALPHA = Fraction(3, 20)                      # RRC roll-off
SPAN = 8                                     # pulse cut at |tau| <= 8 symbols

# --- phase modulation ------------------------------------------------------
BETA = Fraction(4, 5)                        # rad rms
A0 = math.exp(-0.32)                         # e^(-beta^2/2) = 0.7261490370736908
K_LIN = 0.8 * A0                             # 0.5809192296589527, linear-part gain
CARRIER_FRAC = math.exp(-0.64)               # 0.5272924240430485, carrier power share
USEFUL_FRAC = 0.64 * CARRIER_FRAC            # 0.3374671513875510, linear-part share

# --- frame building blocks ---------------------------------------------------
N_PRE, N_HDR, REF_PERIOD = 660, 2640, 16     # preamble, header symbols; 1 in 16 a reference
N_HDR_BITS, N_INFO_BITS, POLAR_N = 2474, 142, 2048
# 2 (2026-10-10, spec 5.1): every CE frame with a header block also carries
# the spread copy of the header on its data symbols. 1 was sent only in
# tests; its block still decodes, so receivers fall back to it.
FORMAT_VERSION = 2
FORMAT_VERSIONS = (1, 2)                     # what a header may say
SPREAD_RHO = 0.054                           # spread-header power, latent units (spec 5.1)
WAVEFORM_CE, WAVEFORM_L = 0, 1               # CE: frame.py, ce.py; L: lframe.py, lwave.py
WAVEFORM_NAMES = {WAVEFORM_CE: "CE", WAVEFORM_L: "L"}
LEAD_IN_MAX_S = 10

# --- callsign windows (spec 2.7) ----------------------------------------------
CW_WINDOW = 384                              # CE symbols per callsign window (11.64 s)
CW_UNIT = 2                                  # CE symbols per Morse unit (60.625 ms)
CW_UNITS = CW_WINDOW // CW_UNIT              # 192 units per window
CW_LEAD_UNITS, CW_ROOM_UNITS = 8, 176        # plain units before the call; room for the call
CW_AFTER_FULL = (13888, 28350, 42812, 57274)  # windows follow these stream symbols (2x L's)

# --- beacon storage ------------------------------------------------------------
INT8_SCALE = 20                              # beacon latent = q/20, q in [-127, 127]

# --- receiver rates and gates ---------------------------------------------------
FE_FS, FE_CENTER_HZ = 4000, 1500             # receiver front end: complex, 300-2700 Hz
CARRIER_BAND_HZ = (300.0, 2700.0)            # audio carriers a receiver searches (and a sender may use)
CH_FS, CH_DECIM = 250, 16                    # per-signal channel: complex, +-125 Hz
Z_ACCEPT = 6.0                               # single detection/association gate

Q_EPOCH_NOTE = "q = floor(unix_utc(QH)/900); unix time ignores leap seconds"

# --- waveform L (spec 2, 2.2, 2.7, 2.8; docs/qrss/waveform-l.md) ------------------
# One linear carrier plus a pilot tone, for SSB rigs. Two latents per complex
# symbol at half CE's rate, so a pair of latents takes the same 60.625 ms in
# both waveforms and every frame time (preamble, header, windows, end) agrees.
L_T_NUM = 2 * T_NUM                          # 970 grid steps: T_L = 0.060625 s
L_T_SYM = L_T_NUM / T_DEN                    # 16.495 Bd
L_N_PRE, L_N_HDR = 330, 1320                 # QPSK preamble, header symbols (83 references)
L_CW_WINDOW = CW_WINDOW // 2                 # 192 L symbols; one Morse unit = one L symbol
L_CW_AFTER_FULL = tuple(c // 2 for c in CW_AFTER_FULL)   # (6944, 14175, 21406, 28637)
# The pilot sits 3.1 Hz above the data band's upper edge (1 + alpha) Rs / 2,
# 12207/970 Hz above the data centre, exactly: a rational keeps the mixing in
# integer turns. The carrier a sender names is the pilot's frequency.
L_EDGE_HZ = (1 + ALPHA) * Fraction(T_DEN, L_T_NUM) / 2      # 920/97 = 9.4845 Hz
L_PILOT_HZ = L_EDGE_HZ + Fraction(31, 10)                   # 12207/970 = 12.5845 Hz
L_PILOT_FRAC = Fraction(3, 20)               # pilot share of the power before clipping
# Clip and filter (spec 2.2): SSTVAE's three passes with overshoot, the
# threshold 0 dB above the nominal RMS envelope, each pass's correction
# filtered to the occupied band plus a guard.
L_CLIP_HEADROOM_DB = 0.0
L_CLIP_OVERSHOOT = (1.0, 1.5, 2.0)
L_CLIP_GUARD_HZ = 1.0

assert T_DEN % FS == 0 and FE_FS == CH_FS * CH_DECIM  # rates nest
assert CW_LEAD_UNITS + CW_ROOM_UNITS + CW_LEAD_UNITS == CW_UNITS
assert L_PILOT_HZ == Fraction(12207, 970) and 2 * L_CW_WINDOW == CW_WINDOW
