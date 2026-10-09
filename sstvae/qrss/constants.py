"""Every on-air constant of QRSSTVAE waveform CE (design section 2.1).

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
FORMAT_VERSION = 1
WAVEFORM_CE, WAVEFORM_L = 0, 1               # L reserved
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

assert T_DEN % FS == 0 and FE_FS == CH_FS * CH_DECIM  # rates nest
assert CW_LEAD_UNITS + CW_ROOM_UNITS + CW_LEAD_UNITS == CW_UNITS
