/* qrss_ce.c -- QRSSTVAE CE reference phase generator. See qrss_ce.h.
 *
 * The phase at tau symbols after t0 is
 *     phi = beta * sum_s x_s p(tau - pos(s))       (17 positions at most)
 * plus, inside an FSK callsign window, 2 pi theta_cw. Each x_s is an
 * integer times a per-class scale (QRSS_SCALE_Q32, turns per unit), p is
 * the Q14 half-table at T/64 interpolated linearly to Q22, and the sum is
 * an int64 in 2^-54 turn. All of it mirrors sstvae/qrss/ce.py; where they
 * differ, Python is the definition.
 */
#include "qrss_ce.h"

#include <string.h>

#include "sha256.h"

#define Q22 ((int64_t)1 << 22)
#define HDR_OFFSET 56u
#define LAT_OFFSET 366u

/* --- small integer helpers ------------------------------------------------ */

/* floor(x / 2^n) for any sign (C99 leaves >> of a negative value to the
 * implementation). */
static int64_t sar(int64_t x, unsigned n)
{
    return x >= 0 ? x >> n : -(((-x) - 1) >> n) - 1;
}

static int64_t floordiv(int64_t a, int64_t b)        /* b > 0 */
{
    int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

static uint32_t rd_u32le(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint32_t rd_u16le(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8;
}

uint32_t qrss_crc32(const uint8_t *p, uint32_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    uint32_t i;
    int k;
    for (i = 0; i < n; i++) {
        c ^= p[i];
        for (k = 0; k < 8; k++)
            c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return c ^ 0xFFFFFFFFu;
}

/* --- SHA-256 sequences (design 2.3) ---------------------------------------
 * Block b of a stream is SHA256(domain || prefix || uint32_be(b)); bit i
 * is bit 7 - i%8 of byte (i%256)/8 of block i/256; bit 0 sends +1. */

static void hash_block(int which, uint64_t q, uint32_t b, uint8_t out[32])
{
    static const char *const dom[3] = {
        QRSS_PREAMBLE_DOMAIN, QRSS_REFERENCE_DOMAIN, QRSS_SCRAMBLE_DOMAIN,
    };
    uint8_t tail[12];
    uint32_t n = 0;
    int i;
    sha256_ctx c;
    sha256_init(&c);
    sha256_update(&c, dom[which], strlen(dom[which]));
    if (which == 2)
        for (i = 0; i < 8; i++)
            tail[n++] = (uint8_t)(q >> (56 - 8 * i));     /* uint64_be(q) */
    for (i = 0; i < 4; i++)
        tail[n++] = (uint8_t)(b >> (24 - 8 * i));          /* uint32_be(b) */
    sha256_update(&c, tail, n);
    sha256_final(&c, out);
}

static int digest_bit(const uint8_t d[32], uint32_t i)
{
    return (d[(i & 255u) >> 3] >> (7u - (i & 7u))) & 1;
}

void qrss_seq_bits(int which, uint64_t q, uint32_t start, uint32_t n, uint8_t *out)
{
    uint8_t d[32];
    uint32_t blk = 0xFFFFFFFFu, i;
    for (i = 0; i < n; i++) {
        uint32_t k = start + i;
        if (k >> 8 != blk) {
            blk = k >> 8;
            hash_block(which, q, blk, d);
        }
        out[i] = (uint8_t)digest_bit(d, k);
    }
}

static int cached_bit(qrss_ce_t *s, int which, uint32_t i)
{
    qrss_hash_slot_t *slot = which == 0 ? s->pre : which == 1 ? s->ref : s->scr;
    uint8_t *next = which == 0 ? &s->pre_next : which == 1 ? &s->ref_next : &s->scr_next;
    uint32_t b = i >> 8;
    int k;
    for (k = 0; k < 2; k++)
        if (slot[k].blk == b)
            return digest_bit(slot[k].d, i);
    k = *next;
    *next ^= 1;
    slot[k].blk = b;
    hash_block(which, s->q, b, slot[k].d);
    return digest_bit(slot[k].d, i);
}

/* --- frame ----------------------------------------------------------------- */

const qrss_frame_t *qrss_frame_preset(const char *name)
{
    int i;
    for (i = 0; i < QRSS_N_PRESETS; i++)
        if (strcmp(QRSS_PRESETS[i].name, name) == 0)
            return &QRSS_PRESETS[i];
    return NULL;
}

/* Smallest L with L - ceil(L/16) == n_data (frame._data_symbols). */
static uint32_t data_symbols(uint32_t n_data)
{
    uint32_t L = n_data + n_data / QRSS_REF_PERIOD;
    while (L - (L + QRSS_REF_PERIOD - 1) / QRSS_REF_PERIOD < n_data)
        L++;
    return L;
}

/* Stream index at position p, or -1 inside a callsign window (or outside the frame). */
static int64_t stream_at(const qrss_ce_t *s, int64_t p)
{
    uint32_t w;
    int64_t before = 0;
    if (p < 0 || p >= (int64_t)s->n_pos)
        return -1;
    for (w = 0; w < s->n_win; w++) {
        if (p < (int64_t)s->win_pos[w])
            break;
        if (p < (int64_t)s->win_pos[w] + QRSS_CW_WINDOW)
            return -1;
        before++;
    }
    return p - (int64_t)QRSS_CW_WINDOW * before;
}

static int key_down(const qrss_ce_t *s, uint32_t unit)
{
    return (s->keying[unit >> 3] >> (7u - (unit & 7u))) & 1;
}

int qrss_ce_init_frame(qrss_ce_t *s, const uint8_t *file, uint32_t len, uint64_t q,
                       const qrss_frame_t *frame)
{
    uint32_t n_lat, w, u, key_count = 0;
    memset(s, 0, sizeof *s);
    if (len < LAT_OFFSET + 5)
        return QRSS_E_SHORT;
    if (memcmp(file, "QRSB", 4) != 0)
        return QRSS_E_MAGIC;
    n_lat = rd_u32le(file + 24);
    if (n_lat == 0 || n_lat > QRSS_MAX_LATENTS || len != LAT_OFFSET + n_lat + 4)
        return QRSS_E_SHORT;
    if (qrss_crc32(file, len - 4) != rd_u32le(file + len - 4))
        return QRSS_E_CRC;
    if (file[4] != 1 || file[5] != QRSS_WAVEFORM_CE || rd_u16le(file + 28) != QRSS_INT8_SCALE
        || (file[30] & ~1u) != 0 || file[31] != 0)
        return QRSS_E_FORMAT;

    s->bf = file;
    s->n_lat = n_lat;
    s->q = q;
    s->flags = file[30];
    memcpy(s->keying, file + 32, sizeof s->keying);
    s->hdr = file + HDR_OFFSET;
    s->lat = (const int8_t *)(file + LAT_OFFSET);

    if (frame->n_data == 0 || frame->n_data > n_lat
        || (frame->n_hdr != 0 && frame->n_hdr != QRSS_N_HDR) || frame->n_win > QRSS_MAX_WINDOWS)
        return QRSS_E_FRAME;
    s->n_hdr = frame->n_hdr;
    s->n_data = frame->n_data;
    s->n_data_sym = data_symbols(frame->n_data);
    s->n_sym = QRSS_N_PRE + s->n_hdr + s->n_data_sym;
    s->n_win = frame->n_win;
    for (w = 0; w < s->n_win; w++) {
        uint32_t c = frame->cw_after[w];
        if (c <= QRSS_N_PRE + s->n_hdr || c > s->n_sym || (w && c <= frame->cw_after[w - 1]))
            return QRSS_E_FRAME;
        s->cw_after[w] = c;
        s->win_pos[w] = c + QRSS_CW_WINDOW * w;
    }
    s->n_pos = s->n_sym + QRSS_CW_WINDOW * s->n_win;
    if (s->n_win && s->cw_after[s->n_win - 1] == s->n_sym)
        s->keyed_end_q22 = (int64_t)s->n_pos * Q22 - Q22 / 2;        /* the last window ends it */
    else
        s->keyed_end_q22 = ((int64_t)s->n_pos - 1 + QRSS_SPAN) * Q22; /* pos(n_sym-1) + 8 */

    /* Every transmit path sends the callsign (spec 2.7): with windows in
     * the frame, the mask must key something, and only in the call's room. */
    for (u = 0; u < QRSS_CW_UNITS; u++) {
        int k = key_down(s, u);
        if (k && (u < QRSS_CW_LEAD_UNITS || u >= QRSS_CW_LEAD_UNITS + QRSS_CW_ROOM_UNITS))
            return QRSS_E_CALLSIGN;
        key_count += (uint32_t)k;
    }
    if (s->n_win && key_count == 0)
        return QRSS_E_CALLSIGN;

    for (w = 0; w < 2; w++) {
        s->pre[w].blk = s->ref[w].blk = s->scr[w].blk = 0xFFFFFFFFu;
        s->blk[w].start = -1;
    }
    return QRSS_OK;
}

int qrss_ce_init(qrss_ce_t *s, const uint8_t *file, uint32_t len, uint64_t q)
{
    qrss_frame_t f = *qrss_frame_preset("full");
    if (len >= 28)
        f.n_data = rd_u32le(file + 24);
    return qrss_ce_init_frame(s, file, len, q, &f);
}

/* --- symbols (design 2.2, 2.5) -------------------------------------------- */

static int log2u(uint32_t m)
{
    int b = 0;
    while (m > 1u) {
        m >>= 1;
        b++;
    }
    return b;
}

/* The precoder block holding data index i: its start and size (precoder.block_sizes:
 * 64s, then the binary decomposition of n_data % 64, descending). */
static void find_block(const qrss_ce_t *s, uint32_t i, uint32_t *start, uint32_t *m)
{
    uint32_t full = s->n_data & ~63u, rem = s->n_data & 63u, off = full, sz;
    if (i < full) {
        *start = i & ~63u;
        *m = 64;
        return;
    }
    for (sz = 32; sz; sz >>= 1) {
        if (!(rem & sz))
            continue;
        if (i < off + sz)
            break;
        off += sz;
    }
    *start = off;
    *m = sz;
}

/* Unnormalised Sylvester WHT of c_q * int8 latents over one block. */
static const qrss_blk_t *get_block(qrss_ce_t *s, uint32_t start, uint32_t m)
{
    qrss_blk_t *b;
    uint32_t k, h;
    int j;
    for (j = 0; j < 2; j++)
        if (s->blk[j].start == (int32_t)start)
            return &s->blk[j];
    b = &s->blk[s->blk_next];
    s->blk_next ^= 1;
    b->start = (int32_t)start;
    b->m = (uint8_t)m;
    for (k = 0; k < m; k++) {
        int16_t a = s->lat[start + k];
        b->v[k] = cached_bit(s, 2, start + k) ? (int16_t)-a : a;
    }
    for (h = 1; h < m; h <<= 1)
        for (k = 0; k < m; k += 2 * h) {
            uint32_t i;
            for (i = k; i < k + h; i++) {
                int16_t x = b->v[i], y = b->v[i + h];
                b->v[i] = (int16_t)(x + y);
                b->v[i + h] = (int16_t)(x - y);
            }
        }
    return b;
}

/* Value and scale class of stream symbol k (see qrss_ce_symbol). */
static int32_t symbol(qrss_ce_t *s, uint32_t k, int *cls)
{
    uint32_t j, i;
    *cls = 0;
    if (k < QRSS_N_PRE)
        return cached_bit(s, 0, k) ? -1 : 1;
    j = k - QRSS_N_PRE;
    if (j % QRSS_REF_PERIOD == 0)                       /* reference m = j/16 */
        return cached_bit(s, 1, j / QRSS_REF_PERIOD) ? -1 : 1;
    if (j < s->n_hdr) {
        i = j - j / QRSS_REF_PERIOD - 1;                /* header non-reference index */
        if (i >= QRSS_N_HDR_BITS)
            return 1;                                   /* the spare */
        return ((s->hdr[i >> 3] >> (7u - (i & 7u))) & 1u) ? -1 : 1;
    }
    j -= s->n_hdr;                                      /* n_hdr is a multiple of 16 */
    i = j - j / QRSS_REF_PERIOD - 1;                    /* data index */
    {
        uint32_t start, m;
        const qrss_blk_t *b;
        find_block(s, i, &start, &m);
        b = get_block(s, start, m);
        *cls = 1 + log2u(m);
        return b->v[i - start];
    }
}

int32_t qrss_ce_symbol(qrss_ce_t *s, uint32_t k)
{
    int cls;
    return symbol(s, k, &cls);
}

int qrss_ce_symbol_class(qrss_ce_t *s, uint32_t k)
{
    int cls;
    symbol(s, k, &cls);
    return cls;
}

/* --- phase (design 2.4, 2.7) ---------------------------------------------- */

/* p(|d|) in Q22 for |d| <= 8 symbols, d in Q22. */
static int64_t pulse_q22(int64_t d)
{
    int64_t a = d < 0 ? -d : d;
    int64_t idx = a >> 16, fr = a & 0xFFFF;
    int64_t lo, hi;
    if (idx >= QRSS_SPAN * QRSS_PULSE_STEPS)
        return idx == QRSS_SPAN * QRSS_PULSE_STEPS && fr == 0
            ? (int64_t)QRSS_PULSE_Q14[idx] * 256 : 0;
    lo = QRSS_PULSE_Q14[idx];
    hi = QRSS_PULSE_Q14[idx + 1];
    return lo * 256 + sar((hi - lo) * fr + 128, 8);    /* lo may be < 0: no << */
}

/* R(y) (the integral of the Hann CDF) in Q30, y in Q24. */
static int64_t hann_r_q30(int64_t y)
{
    const int64_t half = (int64_t)1 << 23;
    int64_t z, idx, fr, lo, hi;
    if (y < -half)
        return 0;
    if (y > half)
        return y << 6;
    z = y + half;                                       /* 0 .. 2^24 */
    idx = z >> 18;
    fr = z & (((int64_t)1 << 18) - 1);                 /* int may be 16 bits */
    if (idx >= QRSS_HANN_STEPS)
        return QRSS_HANN_R_Q30[QRSS_HANN_STEPS];
    lo = QRSS_HANN_R_Q30[idx];
    hi = QRSS_HANN_R_Q30[idx + 1];
    return lo + sar((hi - lo) * fr + ((int64_t)1 << 17), 18);
}

/* theta_cw modulo one turn, Q30 turns, at tau (Q22) inside window w.
 * theta = -sum_u k[u] (R(x - u) - R(x - u - 1)), x in Morse units from the
 * window's start; units before f - 1 are complete (whole turns, dropped). */
static int64_t cw_theta_q30(const qrss_ce_t *s, uint32_t w, int64_t tau_q22)
{
    int64_t x = (tau_q22 - (int64_t)s->win_pos[w] * Q22 + Q22 / 2) * 2;   /* Q24 units */
    int64_t f = sar(x, 24), th = 0;
    int j;
    for (j = -1; j <= 1; j++) {
        int64_t u = f + j, y;
        if (u < 0 || u >= QRSS_CW_UNITS || !key_down(s, (uint32_t)u))
            continue;
        y = x - u * ((int64_t)1 << 24);
        th -= hann_r_q30(y) - hann_r_q30(y - ((int64_t)1 << 24));
    }
    return th;
}

/* The window holding tau (Q22), or -1. Window w spans [P_w - 1/2, P_w + 383.5). */
static int window_at(const qrss_ce_t *s, int64_t tau_q22)
{
    uint32_t w;
    for (w = 0; w < s->n_win; w++) {
        int64_t a = (int64_t)s->win_pos[w] * Q22 - Q22 / 2;
        if (tau_q22 >= a && tau_q22 < a + (int64_t)QRSS_CW_WINDOW * Q22)
            return (int)w;
    }
    return -1;
}

uint32_t qrss_ce_phase_q22(qrss_ce_t *s, int64_t tau_q22)
{
    int64_t p_lo = floordiv(tau_q22 - QRSS_SPAN * Q22 + Q22 - 1, Q22);    /* ceil(tau - 8) */
    int64_t p_hi = floordiv(tau_q22 + QRSS_SPAN * Q22, Q22);              /* floor(tau + 8) */
    int64_t p, acc = 0, ph;
    int w;
    for (p = p_lo; p <= p_hi; p++) {
        int64_t k = stream_at(s, p);
        int cls;
        int32_t v;
        if (k < 0)
            continue;
        v = symbol(s, (uint32_t)k, &cls);
        acc += (int64_t)v * QRSS_SCALE_Q32[cls] * pulse_q22(tau_q22 - p * Q22);
    }
    ph = sar(acc + ((int64_t)1 << 21), 22);             /* 2^-32 turn */
    w = window_at(s, tau_q22);
    if (w >= 0 && !(s->flags & 1u))
        ph += cw_theta_q30(s, (uint32_t)w, tau_q22) * 4;
    return (uint32_t)((uint64_t)ph & 0xFFFFFFFFu);
}

/* tau in Q22 at update u: -8 + u * fu_den * 16000 / (fu_num * 485), rounded. */
static int64_t tau_of_update(uint32_t u, uint32_t fu_num, uint32_t fu_den)
{
    uint64_t num = (uint64_t)u * fu_den * QRSS_T_DEN;
    uint64_t den = (uint64_t)fu_num * QRSS_T_NUM;
    uint64_t whole = num / den, rem = num % den;
    uint64_t frac = ((rem << 22) + den / 2) / den;      /* rem < den <= 2^21 * 485 */
    return ((int64_t)whole - QRSS_SPAN) * Q22 + (int64_t)frac;
}

uint32_t qrss_ce_phase(qrss_ce_t *s, uint32_t u, uint32_t fu_num, uint32_t fu_den)
{
    return qrss_ce_phase_q22(s, tau_of_update(u, fu_num, fu_den));
}

int qrss_ce_keyed(qrss_ce_t *s, uint32_t u, uint32_t fu_num, uint32_t fu_den)
{
    int64_t tau = tau_of_update(u, fu_num, fu_den);
    int w;
    if (tau < -(int64_t)QRSS_SPAN * Q22 || tau >= s->keyed_end_q22)
        return 0;
    w = window_at(s, tau);
    if (w >= 0 && (s->flags & 1u)) {
        int64_t x = (tau - (int64_t)s->win_pos[w] * Q22 + Q22 / 2) * 2;
        uint32_t unit = (uint32_t)(x >> 24);
        if (unit >= QRSS_OOK_FIRST_UNIT && unit <= QRSS_OOK_LAST_UNIT)
            return key_down(s, unit);
    }
    return 1;
}

/* --- Si5351 steps (design 2.10) ------------------------------------------- */

int64_t qrss_si5351_start(qrss_ce_t *s, uint32_t fu_num, uint32_t fu_den)
{
    return (int64_t)qrss_ce_phase(s, 0, fu_num, fu_den) * (1000 * (int64_t)fu_num);
}

int32_t qrss_si5351_next(qrss_ce_t *s, uint32_t u, uint32_t fu_num, uint32_t fu_den,
                         uint32_t step_mhz, int64_t *reached)
{
    /* Unsigned throughout: mod <= 1000 * 2^21 * 2^32 < 2^63, so a sum of two
     * values below mod never wraps, and nothing here overflows for any
     * fu_num <= 2^21. */
    const uint64_t scale = 1000 * (uint64_t)fu_num;
    const uint64_t mod = scale << 32;                   /* one turn */
    const uint64_t k = ((uint64_t)step_mhz * fu_den) << 32;  /* phase of one step over Tu */
    uint64_t tgt = (uint64_t)qrss_ce_phase(s, u + 1, fu_num, fu_den) * scale;
    uint64_t r = (uint64_t)*reached, diff, a, f, fk;
    int neg;
    diff = tgt >= r ? tgt - r : tgt + (mod - r);        /* [0, mod) */
    neg = diff >= mod / 2;                              /* diff - mod in [-1/2, 0) turn */
    a = neg ? mod - diff : diff;
    f = a / k;
    if (a % k >= k - a % k)
        f++;                                            /* round half away from zero */
    fk = (f * k) % mod;
    if (neg)
        r = r >= fk ? r - fk : r + (mod - fk);
    else {
        r += fk;
        if (r >= mod)
            r -= mod;
    }
    *reached = (int64_t)r;
    return neg ? -(int32_t)f : (int32_t)f;
}
