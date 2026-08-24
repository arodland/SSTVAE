// GENERATED FILE -- DO NOT EDIT.
//
// Written by tools/gen_config_header.py from sstvae/config.py, which
// is the single source of truth for every waveform and latent number
// in this project. Editing this file by hand will be reverted by the
// next generator run, and CI fails if the two disagree.
//
// To change a constant: edit sstvae/config.py, re-run the generator,
// and commit both.

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <string_view>

namespace sstvae::config {

// --- scalars ---------------------------------------------------------
inline constexpr int FS = 8000;
inline constexpr int RS = 50;
inline constexpr int NC = 24;
inline constexpr int M = 160;
inline constexpr int NCP = 32;
inline constexpr int NSYM = 192;
inline constexpr int CARRIER0 = 950;
inline constexpr int FCENTER = 1500;
inline constexpr int SYMS_PER_FRAME = 6;
inline constexpr int DATA_SYMS_PER_FRAME = 5;
inline constexpr int BEACON_CARRIER = 23;
inline constexpr int NC_LATENT = 23;
inline constexpr int CHIPS_PER_FRAME = 5;
inline constexpr int FRAME_SAMPLES = 1152;
inline constexpr int LATENTS_PER_FRAME = 230;
inline constexpr int BEACON_COUNTER_BITS = 10;
inline constexpr int BEACON_CALLSIGN_CHARS = 8;
inline constexpr int BEACON_CALLSIGN_CHAR_BITS = 6;
inline constexpr int BEACON_CALLSIGN_BITS = 48;
inline constexpr int BEACON_MODE_BITS = 2;
inline constexpr int BEACON_RESERVED_BITS = 8;
inline constexpr int BEACON_RESERVED_VALUE = 170;
inline constexpr int BEACON_CRC_BITS = 16;
inline constexpr int PREAMBLE_REPEATS = 4;
inline constexpr int PREAMBLE_CP = 64;
inline constexpr int PREAMBLE_SAMPLES = 704;
inline constexpr int PREAMBLE_CORR_WINDOW = 480;
inline constexpr int HEADER_SYMS = 2;
inline constexpr int HEADER_SAMPLES = 384;
inline constexpr int LEADIN_SAMPLES = 800;
inline constexpr int LEADOUT_SAMPLES = 800;
inline constexpr int LATENT_H = 30;
inline constexpr int LATENT_W = 40;
inline constexpr int LATENT_GROUPS = 3;
inline constexpr int CHANNELS_PER_GROUP = 44;
inline constexpr int LATENT_CHANNELS = 132;
inline constexpr int GROUP_LATENTS = 52800;
inline constexpr int FRAMES_PER_GROUP = 220;
inline constexpr int TRANSMIT_LATENTS_PER_GROUP = 50600;
inline constexpr int DROPPED_LATENTS_PER_GROUP = 2200;
inline constexpr int DEMOD_BACKOFF = 6;
inline constexpr int INTERLEAVER_SEED = 1000;
inline constexpr int PROTOCOL_VERSION = 4;
inline constexpr int ACQUIRE_MAX_BINS = 12;
inline constexpr int FIRST_PATH_SEARCH = 32;

inline constexpr double PREAMBLE_THRESHOLD = 0x1.ae147ae147ae1p-2;  // 0.42
inline constexpr double CLIP_HEADROOM_DB = 0x0.0p+0;  // 0.0
inline constexpr double SNR_REF_BW_HZ = 0x1.3880000000000p+11;  // 2500.0
inline constexpr double BLIND_BIN_STEP_HZ = 0x1.9000000000000p+3;  // 12.5
inline constexpr double BLIND_BLOCK_RES_HZ = 0x1.b333333333333p+0;  // 1.7
inline constexpr double BLIND_MAX_OFFSET_HZ = 0x1.b800000000000p+5;  // 55.0
inline constexpr double BLIND_WIDE_MAX_OFFSET_HZ = 0x1.3880000000000p+9;  // 625.0
inline constexpr double DRIFT_SLOW_ALPHA = 0x1.999999999999ap-4;  // 0.1
inline constexpr double DRIFT_SLOW_BETA = 0x1.47ae147ae147bp-7;  // 0.01
inline constexpr double DRIFT_FAST_ALPHA = 0x1.3333333333333p-2;  // 0.3
inline constexpr double DRIFT_FAST_BETA = 0x1.999999999999ap-5;  // 0.05
inline constexpr double TEMPLATE_SCORE_THRESHOLD = 0x1.999999999999ap-2;  // 0.4
inline constexpr double BLIND_SCORE_THRESHOLD = 0x1.2000000000000p+3;  // 9.0
inline constexpr double FIRST_PATH_FRAC = 0x1.0000000000000p-1;  // 0.5

// Post-clip transmit filter, Hz.
inline constexpr double TX_BANDPASS_LO = 0x1.a900000000000p+9;  // 850.0
inline constexpr double TX_BANDPASS_HI = 0x1.1300000000000p+11;  // 2200.0

// --- beacon sync word ------------------------------------------------
// Barker-13: a clean, unambiguous chip-level autocorrelation peak, so
// superframe phase is recoverable from any contiguous run of frames.
inline constexpr int BEACON_SYNC_LEN = 13;
inline constexpr std::array<int, BEACON_SYNC_LEN> BEACON_SYNC = {1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1};

// --- pilot sequence --------------------------------------------------
// Copied from config.PILOT_PHASE_NUM -- not re-derived here.
//
// A minimized crest-factor phase set, carried as an exact rational
// turn: phi_k = 2*pi * NUM[k] / DEN. Integers rather than radians
// because sin/cos disagree between libms and between x86-64 and Apple
// silicon, which would make the pilot a property of the machine rather
// than of the format -- and this file exists so C++ evaluates the
// *same expression* on the *same values* Python does.
//
// Not Zadoff-Chu, deliberately: see the note in config.py. Its
// delay-Doppler equivalence makes CFO and timing confusable, and this
// sequence is also the acquisition template.
inline constexpr int PILOT_PHASE_DEN = 1024;
inline constexpr std::array<int, NC> PILOT_PHASE_NUM = {
    725, 497, 359, 322, 193, 849, 710, 345, 960, 628, 347, 570,
    551, 678, 448, 713, 839, 90, 236, 545, 1020, 403, 985, 304,
};

// --- modes -----------------------------------------------------------
struct ModeSpec {
    std::string_view name;
    int index;
    int groups;          // latent channel groups transmitted
    int n_frames;
    int n_latents;       // model-facing contract (groups * GROUP_LATENTS)
    int n_tx_latents;    // actually carried on air (23-carrier capacity)
    double duration_s;
};

inline constexpr int N_MODES = 3;
inline constexpr std::array<ModeSpec, N_MODES> MODES = {{
    {"A", 0, 1, 220, 52800, 50600, 0x1.0020c49ba5e35p+5},  // ~32 s
    {"B", 1, 2, 440, 105600, 101200, 0x1.fd916872b020cp+5},  // ~64 s
    {"C", 2, 3, 660, 158400, 151800, 0x1.7d810624dd2f2p+6},  // ~95 s
}};

// Modes are indexed by their on-air index, which is also their position
// in the table; the static_assert keeps that true if a mode is ever
// added out of order.
static_assert(MODES[0].index == 0 && MODES[N_MODES - 1].index == N_MODES - 1,
              "MODES must be stored in on-air index order");

// --- invariants ------------------------------------------------------
// Restated from config.py's own asserts, checked at compile time here.
static_assert(M == FS / RS);
static_assert(NSYM == M + NCP);
static_assert(FRAME_SAMPLES == SYMS_PER_FRAME * NSYM);
static_assert(LATENTS_PER_FRAME == NC_LATENT * DATA_SYMS_PER_FRAME * 2);
static_assert(LATENT_CHANNELS == LATENT_GROUPS * CHANNELS_PER_GROUP);
static_assert(GROUP_LATENTS == CHANNELS_PER_GROUP * LATENT_H * LATENT_W);
static_assert(FRAMES_PER_GROUP * NC * DATA_SYMS_PER_FRAME * 2 == GROUP_LATENTS,
              "FRAMES_PER_GROUP is pinned to the pre-beacon capacity");
static_assert(TRANSMIT_LATENTS_PER_GROUP == FRAMES_PER_GROUP * LATENTS_PER_FRAME);
static_assert(DROPPED_LATENTS_PER_GROUP ==
                  GROUP_LATENTS - TRANSMIT_LATENTS_PER_GROUP &&
              DROPPED_LATENTS_PER_GROUP >= 0 &&
              DROPPED_LATENTS_PER_GROUP < GROUP_LATENTS,
              "the per-group remainder is a permanent erasure");
static_assert(BEACON_CARRIER == NC - 1 && NC_LATENT == NC - 1,
              "one carrier is reserved for the beacon side-channel");

}  // namespace sstvae::config
