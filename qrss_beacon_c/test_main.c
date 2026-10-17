/* test_main.c -- command-line driver for the QRSSTVAE CE reference.
 *
 * Reads a beacon file and prints phases and Si5351 steps; also exposes the
 * pieces tests/test_qrss_c_ref.py compares against Python (C1-C3).
 *
 *   qrss_ce_test selftest                      SHA-256 against FIPS 180-4
 *   qrss_ce_test sha256 HEX                    digest of the bytes HEX
 *   qrss_ce_test seq WHICH Q START N           sequence bits as '0'/'1'
 *                                              (WHICH: pre | ref | scr | spr)
 *   qrss_ce_test show FILE Q [FRAME]           a readable summary: frame,
 *                                              first phases and steps at 990 Hz
 *   qrss_ce_test sym FILE Q FRAME              int32 (value, class, spread) per
 *                                              stream symbol
 *   qrss_ce_test phase FILE Q FRAME FNUM FDEN U0 N STEP
 *                                              uint32 phase at u = U0 + i*STEP
 *   qrss_ce_test keyed FILE Q FRAME FNUM FDEN U0 N STEP
 *                                              one byte 0/1 per update
 *   qrss_ce_test si5351 FILE Q FRAME FNUM FDEN STEP_MHZ N
 *                                              int32 steps of intervals 0..N-1
 *
 * Binary outputs are little-endian-native arrays on stdout. FRAME is a
 * preset name (full, medium, short, tiny) or an explicit
 * N_HDR,N_DATA[,CW_AFTER...] list, which carries the spread copy when it
 * has a header block (as frame.FrameSpec does by default).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qrss_ce.h"
#include "sha256.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

static uint8_t file_buf[70000];
static qrss_ce_t ce;

static int hexval(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void print_hex(const uint8_t d[32])
{
    int i;
    for (i = 0; i < 32; i++)
        printf("%02x", d[i]);
    printf("\n");
}

static int selftest(void)
{
    static const struct { const char *msg; const char *hex; } v[] = {
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
         "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
         "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"},
    };
    uint8_t d[32], a[1000];
    char hex[65];
    int i, j, bad = 0;
    sha256_ctx c;
    for (i = 0; i < (int)(sizeof v / sizeof v[0]); i++) {
        sha256(v[i].msg, strlen(v[i].msg), d);
        for (j = 0; j < 32; j++)
            sprintf(hex + 2 * j, "%02x", d[j]);
        if (strcmp(hex, v[i].hex) != 0) {
            printf("FAIL vector %d: %s\n", i, hex);
            bad++;
        }
    }
    /* one million 'a', fed in uneven pieces */
    memset(a, 'a', sizeof a);
    sha256_init(&c);
    for (i = 0; i < 1000000;) {
        int n = 1 + (i * 7) % 997;
        if (n > 1000000 - i)
            n = 1000000 - i;
        sha256_update(&c, a, (size_t)n);
        i += n;
    }
    sha256_final(&c, d);
    for (j = 0; j < 32; j++)
        sprintf(hex + 2 * j, "%02x", d[j]);
    if (strcmp(hex, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0") != 0) {
        printf("FAIL million a: %s\n", hex);
        bad++;
    }
    printf(bad ? "sha256 selftest FAILED\n" : "sha256 selftest ok\n");
    return bad ? 1 : 0;
}

/* "N_HDR,N_DATA[,CW_AFTER...]" -> frame; 0 ok. */
static int parse_frame(const char *spec, qrss_frame_t *fr)
{
    char *end;
    uint32_t vals[2 + QRSS_MAX_WINDOWS];
    uint32_t n = 0;
    while (*spec && n < 2 + QRSS_MAX_WINDOWS) {
        vals[n++] = (uint32_t)strtoul(spec, &end, 10);
        if (end == spec)
            return -1;
        spec = *end == ',' ? end + 1 : end;
    }
    if (n < 2 || *spec)
        return -1;
    memset(fr, 0, sizeof *fr);
    fr->name = "custom";
    fr->n_hdr = vals[0];
    fr->n_data = vals[1];
    fr->n_win = n - 2;
    fr->spread = fr->n_hdr != 0;
    memcpy(fr->cw_after, vals + 2, (n - 2) * sizeof vals[0]);
    return 0;
}

static int load(const char *path, const char *q_str, const char *frame_name)
{
    FILE *f = fopen(path, "rb");
    size_t n;
    const qrss_frame_t *fr = NULL;
    qrss_frame_t custom;
    int rc;
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return -100;
    }
    n = fread(file_buf, 1, sizeof file_buf, f);
    fclose(f);
    if (frame_name) {
        fr = qrss_frame_preset(frame_name);
        if (!fr && parse_frame(frame_name, &custom) == 0)
            fr = &custom;
        if (!fr) {
            fprintf(stderr, "unknown frame %s\n", frame_name);
            return -101;
        }
        rc = qrss_ce_init_frame(&ce, file_buf, (uint32_t)n, strtoull(q_str, NULL, 10), fr);
    } else {
        rc = qrss_ce_init(&ce, file_buf, (uint32_t)n, strtoull(q_str, NULL, 10));
    }
    if (rc)
        fprintf(stderr, "qrss_ce_init failed: %d\n", rc);
    return rc;
}

static uint32_t arg_u32(const char *s)
{
    return (uint32_t)strtoul(s, NULL, 10);
}

int main(int argc, char **argv)
{
    const char *cmd = argc > 1 ? argv[1] : "";
    uint32_t i;
#ifdef _WIN32
    /* Binary arrays go to stdout; text mode would turn each 0x0A into 0D 0A. */
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (!strcmp(cmd, "selftest"))
        return selftest();
    if (!strcmp(cmd, "sha256") && argc == 3) {
        static uint8_t msg[4096];
        size_t n = strlen(argv[2]) / 2, k;
        uint8_t d[32];
        if (n > sizeof msg)
            return 2;
        for (k = 0; k < n; k++)
            msg[k] = (uint8_t)(hexval(argv[2][2 * k]) << 4 | hexval(argv[2][2 * k + 1]));
        sha256(msg, n, d);
        print_hex(d);
        return 0;
    }
    if (!strcmp(cmd, "seq") && argc == 6) {
        int which = !strcmp(argv[2], "pre") ? 0 : !strcmp(argv[2], "ref") ? 1
                  : !strcmp(argv[2], "scr") ? 2 : !strcmp(argv[2], "spr") ? 3 : -1;
        uint32_t start = arg_u32(argv[4]), n = arg_u32(argv[5]);
        uint8_t b[256];
        if (which < 0)
            return 2;
        for (i = 0; i < n; i += 256) {
            uint32_t m = n - i < 256 ? n - i : 256, j;
            qrss_seq_bits(which, strtoull(argv[3], NULL, 10), start + i, m, b);
            for (j = 0; j < m; j++)
                putchar('0' + b[j]);
        }
        putchar('\n');
        return 0;
    }
    if (!strcmp(cmd, "show") && (argc == 4 || argc == 5)) {
        int64_t reached;
        if (load(argv[2], argv[3], argc == 5 ? argv[4] : NULL))
            return 1;
        printf("latents %u, frame: n_hdr %u n_data %u n_sym %u n_pos %u windows %u (%s)"
               " spread %u\n",
               ce.n_lat, ce.n_hdr, ce.n_data, ce.n_sym, ce.n_pos, ce.n_win,
               (ce.flags & 1u) ? "on-off keyed" : "FSK", ce.spread);
        printf("duration %.9f s from t0 - T/2\n", ce.n_pos * (double)QRSS_T_NUM / QRSS_T_DEN);
        reached = qrss_si5351_start(&ce, 990, 1);
        printf("   u   phase(deg)   step(0.4 Hz)\n");
        for (i = 0; i < 20; i++) {
            uint32_t ph = qrss_ce_phase(&ce, i, 990, 1);
            int32_t st = qrss_si5351_next(&ce, i, 990, 1, 400, &reached);
            printf("%4u %11.4f %8d\n", i, ph * (360.0 / 4294967296.0), st);
        }
        return 0;
    }
    if (!strcmp(cmd, "sym") && argc == 5) {
        if (load(argv[2], argv[3], argv[4]))
            return 1;
        for (i = 0; i < ce.n_sym; i++) {
            int32_t vc[3];
            vc[0] = qrss_ce_symbol(&ce, i);
            vc[1] = qrss_ce_symbol_class(&ce, i);
            vc[2] = qrss_ce_spread(&ce, i);
            fwrite(vc, sizeof vc, 1, stdout);
        }
        return 0;
    }
    if ((!strcmp(cmd, "phase") || !strcmp(cmd, "keyed")) && argc == 10) {
        uint32_t fn = arg_u32(argv[5]), fd = arg_u32(argv[6]);
        uint32_t u0 = arg_u32(argv[7]), n = arg_u32(argv[8]), step = arg_u32(argv[9]);
        int keyed = !strcmp(cmd, "keyed");
        if (load(argv[2], argv[3], argv[4]))
            return 1;
        for (i = 0; i < n; i++) {
            uint32_t u = u0 + i * step;
            if (keyed) {
                uint8_t k = (uint8_t)qrss_ce_keyed(&ce, u, fn, fd);
                fwrite(&k, 1, 1, stdout);
            } else {
                uint32_t ph = qrss_ce_phase(&ce, u, fn, fd);
                fwrite(&ph, sizeof ph, 1, stdout);
            }
        }
        return 0;
    }
    if (!strcmp(cmd, "si5351") && argc == 9) {
        uint32_t fn = arg_u32(argv[5]), fd = arg_u32(argv[6]), step = arg_u32(argv[7]);
        uint32_t n = arg_u32(argv[8]);
        int64_t reached;
        if (load(argv[2], argv[3], argv[4]))
            return 1;
        reached = qrss_si5351_start(&ce, fn, fd);
        for (i = 0; i < n; i++) {
            int32_t st = qrss_si5351_next(&ce, i, fn, fd, step, &reached);
            fwrite(&st, sizeof st, 1, stdout);
        }
        return 0;
    }
    fprintf(stderr, "usage: see the comment at the top of test_main.c\n");
    return 2;
}
