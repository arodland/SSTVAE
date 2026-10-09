/* sha256.h -- a minimal SHA-256 (FIPS 180-4) for the QRSSTVAE reference beacon.
 *
 * C99, no allocation, no tables beyond the 64 round constants. Small
 * enough for an 8-bit microcontroller: the beacon hashes its preamble,
 * reference and scrambler sequences on the device (design section 9)
 * rather than storing them, and every one of those messages is a single
 * 64-byte block.
 */
#ifndef QRSS_SHA256_H
#define QRSS_SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t h[8];
    uint64_t n_bytes;          /* message length so far */
    uint8_t buf[64];
    uint32_t n_buf;            /* bytes waiting in buf */
} sha256_ctx;

void sha256_init(sha256_ctx *c);
void sha256_update(sha256_ctx *c, const void *data, size_t len);
void sha256_final(sha256_ctx *c, uint8_t out[32]);

/* One-shot convenience. */
void sha256(const void *data, size_t len, uint8_t out[32]);

#endif /* QRSS_SHA256_H */
