/* qrss_ce.h -- QRSSTVAE waveform CE: the reference phase generator of a
 * clock-chip (Si5351-class) beacon, from a beacon file (design section 9).
 *
 * C99, no allocation, integer arithmetic only. A beacon holds one beacon
 * file (design 2.9: header bits, callsign keying and int8 latents, 50,970
 * bytes for a full segment) and computes everything else itself, every
 * pass: the preamble, references and scrambler by SHA-256, the precoder
 * (sign flips, then a 64-point Walsh-Hadamard transform per block), the
 * spread copy of the header on the data symbols (spec 5.1, format
 * version 2: sqrt(rho) h_i on data symbol i, from the file's header bits
 * and a fourth SHA-256 sequence), the phase pulse sum and the
 * callsign-window phase. This file is the parity
 * oracle for any port (tests/test_qrss_c_ref.py, C1-C3).
 *
 * Units and time.
 *  - Phase is in 2^-32 turn, as a uint32_t: it wraps exactly like the
 *    phase accumulator of a synthesizer. The callsign-window phase of FSK
 *    keying ends every window on a whole number of turns (design 2.7), so
 *    the wrapped phase is continuous across it.
 *  - Time is counted in updates u from the start of keying, t0 - 8T,
 *    where t0 = QH + 1 s is the centre of the first preamble symbol and
 *    T = 485/16000 s: update u is at t0 - 8T + u * fu_den / fu_num s.
 *    Internally time is tau = (t - t0)/T symbols in Q22.
 *  - The frame (header length, data latents, callsign windows, the
 *    spread copy) is a qrss_frame_t, so the slot timing stays a table edit: QRSS_PRESETS in
 *    qrss_tables.h is generated from frame.PRESETS.
 *
 * The callsign windows are always sent (spec 2.7, a legal requirement):
 * qrss_ce_init refuses a file whose keying mask is empty when the frame
 * has windows. With FSK keying (flags bit 0 clear) they are part of the
 * phase; with on-off keying (bit 0 set) the phase runs on and
 * qrss_ce_keyed() says when the output is on. An optional lead-in carrier
 * (spec 2.8, up to 10 s before t0 - 8T) is the carrier at phase 0 with no
 * symbols, so it needs nothing from this module: key the output early.
 */
#ifndef QRSS_CE_H
#define QRSS_CE_H

#include <stdint.h>

#include "qrss_tables.h"

/* qrss_ce_init error codes. */
#define QRSS_OK            0
#define QRSS_E_SHORT      -1   /* file too short or length mismatch */
#define QRSS_E_MAGIC      -2   /* not "QRSB" */
#define QRSS_E_CRC        -3   /* CRC-32 mismatch */
#define QRSS_E_FORMAT     -4   /* version, waveform, scale, flags or reserved */
#define QRSS_E_FRAME      -5   /* frame does not fit the file or is malformed */
#define QRSS_E_CALLSIGN   -6   /* frame has windows but the file keys no callsign */

typedef struct {
    uint32_t blk;              /* hash block number; UINT32_MAX when empty */
    uint8_t d[32];
} qrss_hash_slot_t;

typedef struct {
    int32_t start;             /* first data index of the block; -1 when empty */
    uint8_t m;                 /* block size */
    int16_t v[64];             /* unnormalised WHT of c_q * int8 latents */
} qrss_blk_t;

typedef struct {
    /* the beacon file */
    const uint8_t *bf;
    uint32_t n_lat;
    uint64_t q;
    uint8_t flags;             /* bit 0: on-off keyed callsign windows */
    uint8_t keying[24];        /* 192 units, MSB first, 1 = key-down */
    const uint8_t *hdr;        /* 2474 coded header bits, MSB first */
    const int8_t *lat;         /* int8 latents, air order */
    /* the frame */
    uint32_t n_hdr, n_data, n_data_sym, n_sym, n_pos, n_win;
    uint32_t spread;           /* 1: data symbols carry the spread header */
    uint32_t cw_after[QRSS_MAX_WINDOWS];
    uint32_t win_pos[QRSS_MAX_WINDOWS];   /* P_w, first position of window w */
    int64_t keyed_end_q22;     /* end of keying, tau in Q22 */
    /* caches: two hash blocks per sequence and two precoder blocks, so
     * the 17-symbol pulse span never thrashes across a block boundary */
    qrss_hash_slot_t pre[2], ref[2], scr[2], spr[2];
    uint8_t pre_next, ref_next, scr_next, spr_next;
    qrss_blk_t blk[2];
    uint8_t blk_next;
} qrss_ce_t;

/* Bind a beacon file (it must stay in memory) to slot q, FULL frame with
 * n_data = the file's latent count. 0 ok; <0 one of QRSS_E_*. */
int qrss_ce_init(qrss_ce_t *s, const uint8_t *file, uint32_t len, uint64_t q);

/* The same with an explicit frame; frame->n_data may be less than the
 * file's latent count (a test frame sends a prefix of air order). */
int qrss_ce_init_frame(qrss_ce_t *s, const uint8_t *file, uint32_t len, uint64_t q,
                       const qrss_frame_t *frame);

/* A preset by name ("full", "medium", "short", "tiny"), or NULL. */
const qrss_frame_t *qrss_frame_preset(const char *name);

/* Stream symbol k (0 <= k < n_sym). Preamble, references, the header
 * spare and header bits are +-1 (class 0); a data symbol of a size-M
 * precoder block is in units of 1/(20 sqrt(M)) (class 1 + log2 M), i.e.
 * the unnormalised WHT of the sign-flipped int8 latents. */
int32_t  qrss_ce_symbol(qrss_ce_t *s, uint32_t k);
int      qrss_ce_symbol_class(qrss_ce_t *s, uint32_t k);

/* The spread header's part of stream symbol k, in units of sqrt(rho)
 * (QRSS_SPREAD_Q32 turns): h_i = w_i (1 - 2 c[i mod 2474]) = +-1 on data
 * symbol i of a spread frame, else 0. Symbol k is sent as
 * qrss_ce_symbol (scaled by its class) plus sqrt(rho) times this. */
int32_t  qrss_ce_spread(qrss_ce_t *s, uint32_t k);

/* Total phase phi + 2 pi theta_cw in 2^-32 turn at t = t0 - 8T + u*fu_den/fu_num.
 * fu_num <= 2^21. */
uint32_t qrss_ce_phase(qrss_ce_t *s, uint32_t u, uint32_t fu_num, uint32_t fu_den);

/* The same at tau = (t - t0)/T symbols, in Q22 (any sign). */
uint32_t qrss_ce_phase_q22(qrss_ce_t *s, int64_t tau_q22);

/* 1 while the output is keyed (from t0 - 8T to the end of the frame,
 * following the Morse in an on-off-keyed window), else 0. */
int      qrss_ce_keyed(qrss_ce_t *s, uint32_t u, uint32_t fu_num, uint32_t fu_den);

/* Si5351 error-feedback steps (design 2.10, si5351.si5351_steps).
 * `reached` is the phase the synthesizer has reached, in units of
 * 2^-32 turn / (1000 fu_num), kept modulo 2^32 * 1000 * fu_num; start it
 * with qrss_si5351_start. qrss_si5351_next returns the frequency of
 * interval u (from update u to u + 1) in steps of step_mhz milli-Hz,
 * aiming at the target phase at update u + 1 and rounding half away from
 * zero, and advances `reached` by exactly that many steps. Exact (no
 * overflow) for fu_num <= 2^21 and step_mhz * fu_den < 2^31. */
int64_t  qrss_si5351_start(qrss_ce_t *s, uint32_t fu_num, uint32_t fu_den);
int32_t  qrss_si5351_next(qrss_ce_t *s, uint32_t u, uint32_t fu_num, uint32_t fu_den,
                          uint32_t step_mhz, int64_t *reached);

/* Bits [start, start + n) of a sequence, one 0/1 per byte: which = 0
 * preamble, 1 references, 2 scrambler of slot q, 3 the spread header's
 * whitener (not keyed by q). For tests and ports. */
void     qrss_seq_bits(int which, uint64_t q, uint32_t start, uint32_t n, uint8_t *out);

/* zlib's CRC-32. */
uint32_t qrss_crc32(const uint8_t *p, uint32_t n);

#endif /* QRSS_CE_H */
