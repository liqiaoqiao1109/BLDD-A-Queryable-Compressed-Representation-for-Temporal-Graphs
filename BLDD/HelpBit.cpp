#include <iostream>
#include"Help.h"
#include"Graph.h"
#include"Bit.h"
#include<vector>
#include <intrin.h>


void bw_init(BitWriter* bw, uint64_t init_bits)
{
    bw->capacity_words = (init_bits >> 5) + 2;
    if (bw->capacity_words > (uint64_t)(SIZE_MAX / sizeof(unsigned int))) {
        fprintf(stderr, "BitWriter capacity exceeds addressable memory\n");
        exit(EXIT_FAILURE);
    }
    bw->data = (unsigned int*)calloc((size_t)bw->capacity_words, sizeof(unsigned int));
    bw->bitpos = 0;

    if (!bw->data) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
}


void bw_reserve(BitWriter* bw, uint64_t more_bits)
{
    if (more_bits > UINT64_MAX - bw->bitpos) {
        fprintf(stderr, "BitWriter bit position overflow\n");
        exit(EXIT_FAILURE);
    }
    uint64_t need = bw->bitpos + more_bits;
    uint64_t need_words = (need >> 5) + 2;

    if (need_words <= bw->capacity_words)
        return;

    uint64_t new_cap = bw->capacity_words > UINT64_MAX / 2
        ? need_words : bw->capacity_words * 2;
    if (new_cap < need_words)
        new_cap = need_words;

    if (new_cap > (uint64_t)(SIZE_MAX / sizeof(unsigned int))) {
        fprintf(stderr, "BitWriter capacity exceeds addressable memory\n");
        exit(EXIT_FAILURE);
    }
    void* resized = realloc(bw->data, (size_t)new_cap * sizeof(unsigned int));
    if (!resized) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    bw->data = (unsigned int*)resized;

    memset(bw->data + bw->capacity_words, 0,
        (size_t)(new_cap - bw->capacity_words) * sizeof(unsigned int));

    bw->capacity_words = new_cap;
}


void bw_write_bit(BitWriter* bw, unsigned int b)
{
    bw_reserve(bw, 1);

    uint64_t word = bw->bitpos >> 5;
    unsigned int off = bw->bitpos & 31;

    bw->data[word] |= (b << off);

    bw->bitpos++;
}


void bw_write_bits(BitWriter* bw, unsigned int v, unsigned int bits) {
    bw_reserve(bw, bits);

    uint64_t word = bw->bitpos >> 5;
    unsigned int off = bw->bitpos & 31;

    bw->data[word] |= v << off;

    if (off + bits > 32)
        bw->data[word + 1] |= v >> (32 - off);

    bw->bitpos += bits;
}

void bw_write_bits64(BitWriter* bw, uint64_t value, unsigned int bits)
{
    if (bits > 64u) {
        fprintf(stderr, "BitWriter field exceeds 64 bits\n");
        exit(EXIT_FAILURE);
    }
    bw_reserve(bw, bits);
    for (unsigned int bit = 0; bit < bits; ++bit) {
        if (((value >> bit) & 1u) != 0u) {
            const uint64_t position = bw->bitpos + bit;
            bw->data[position >> 5] |= 1u << (position & 31u);
        }
    }
    bw->bitpos += bits;
}


void br_init(BitReader* br,
    const unsigned int* data,
    uint64_t start)
{
    br->data = data;
    br->bitpos = start;
}


static inline unsigned int reverse_k_bits(unsigned int x, unsigned int k) {
    unsigned int r = 0;
    for (unsigned int i = 0; i < k; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}


static inline unsigned int br_read_bits(BitReader* br, unsigned int k)
{
    uint64_t pos = br->bitpos;
    br->bitpos += k;

    uint64_t word = pos >> 5;
    unsigned int offset = pos & 31;

    unsigned long long v = br->data[word];

    if (offset + k > 32)
        v |= ((unsigned long long)br->data[word + 1] << 32);

    v >>= offset;

    unsigned int low = (unsigned int)(v & ((1ULL << k) - 1));
    return reverse_k_bits(low, k);
}


static inline unsigned int my_ctz(unsigned int x)
{
#if defined(_MSC_VER)

    unsigned long idx;
    _BitScanForward(&idx, x);
    return (unsigned int)idx;

#else

    return __builtin_ctz(x);

#endif
}


unsigned int br_read_gamma(BitReader* br)
{
    unsigned int zeros = 0;

    while (1)
    {
        uint64_t word = br->bitpos >> 5;
        unsigned int offset = br->bitpos & 31;

        unsigned int w = br->data[word] >> offset;

        if (w == 0)
        {
            zeros += 32 - offset;
            br->bitpos += 32 - offset;
            continue;
        }

        unsigned int tz = my_ctz(w);
        zeros += tz;
        br->bitpos += tz;
        break;
    }


    br->bitpos++;

    unsigned int rest = 0;
    if (zeros > 0)
        rest = br_read_bits(br, zeros);

    return (1u << zeros) | rest;
}
