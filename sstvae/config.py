"""Central waveform and latent-layout constants.

Every number that must agree between the modem, the channel simulator,
and (later) the neural network training loop lives here.
"""

from dataclasses import dataclass, field

FS = 8000  # audio sample rate, Hz

# --- OFDM waveform ---------------------------------------------------------
RS = 50  # carrier spacing == symbol rate of one carrier, Hz
NC = 24  # data carriers
M = FS // RS  # useful symbol length, samples (160)
NCP = 32  # cyclic prefix, samples (4 ms)
NSYM = M + NCP  # full symbol, samples (192)

# Carriers sit on integer multiples of RS so the cyclic prefix is truly
# cyclic: passband carrier k is at CARRIER0 + k*RS.
CARRIER0 = 950  # Hz; occupied band ~925..2125 Hz, centered ~1525 Hz
FCENTER = 1500  # Hz; baseband conversion frequency (carriers land on
#                 integer multiples of RS: bin k-11 for carrier k)

# --- framing ---------------------------------------------------------------
# Short frames keep pilots 144 ms apart so interpolated equalization can
# follow ~1 Hz Doppler fading (coherence time ~400 ms). Costs 16.7%
# pilot overhead vs 8.3% for 12-symbol frames.
SYMS_PER_FRAME = 6  # 1 pilot + 5 data
DATA_SYMS_PER_FRAME = SYMS_PER_FRAME - 1
FRAME_SAMPLES = SYMS_PER_FRAME * NSYM  # 1152 samples = 144 ms
LATENTS_PER_FRAME = NC * DATA_SYMS_PER_FRAME * 2  # 240 real values

# Preamble: one OFDM symbol repeated twice with a double-length cyclic
# prefix, so the waveform is periodic with M over the whole block.
PREAMBLE_CP = 2 * NCP
PREAMBLE_SAMPLES = PREAMBLE_CP + 2 * M  # 384
HEADER_SYMS = 2  # identical Golay-coded BPSK symbols
HEADER_SAMPLES = HEADER_SYMS * NSYM  # 384

LEADIN_SAMPLES = 800  # 100 ms of silence before the preamble
LEADOUT_SAMPLES = 800

# --- latent layout ---------------------------------------------------------
LATENT_H = 30  # spatial grid for a 240-high image, x8 downsample
LATENT_W = 40
LATENT_GROUPS = 3
CHANNELS_PER_GROUP = 44
LATENT_CHANNELS = LATENT_GROUPS * CHANNELS_PER_GROUP  # 132
GROUP_LATENTS = CHANNELS_PER_GROUP * LATENT_H * LATENT_W  # 52800
FRAMES_PER_GROUP = GROUP_LATENTS // LATENTS_PER_FRAME  # exactly 220
assert FRAMES_PER_GROUP * LATENTS_PER_FRAME == GROUP_LATENTS

# --- TX conditioning -------------------------------------------------------
CLIP_HEADROOM_DB = 5.0  # envelope clip threshold above mean envelope power;
#                         gives ~6.7 dB envelope PAPR at ~20 dB latent-SNR floor
TX_BANDPASS = (850.0, 2200.0)  # Hz, post-clip filter
DEMOD_BACKOFF = 6  # samples: demod window starts this early inside the CP

INTERLEAVER_SEED = 1000  # + group index
PILOT_SEED = 42
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class ModeSpec:
    name: str
    index: int
    groups: int  # latent channel groups transmitted

    @property
    def n_frames(self) -> int:
        return self.groups * FRAMES_PER_GROUP

    @property
    def n_latents(self) -> int:
        return self.n_frames * LATENTS_PER_FRAME

    @property
    def duration_s(self) -> float:
        n = (
            LEADIN_SAMPLES
            + PREAMBLE_SAMPLES
            + HEADER_SAMPLES
            + self.n_frames * FRAME_SAMPLES
            + LEADOUT_SAMPLES
        )
        return n / FS


MODES = {
    "A": ModeSpec("A", 0, 1),  # ~32 s
    "B": ModeSpec("B", 1, 2),  # ~64 s
    "C": ModeSpec("C", 2, 3),  # ~95 s
}
MODES_BY_INDEX = {m.index: m for m in MODES.values()}
