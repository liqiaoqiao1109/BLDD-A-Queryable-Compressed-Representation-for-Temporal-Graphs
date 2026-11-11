#include "Graph.h"
#include "Bit.h"
#include "Help.h"
#include <intrin.h>


FixBitArray* CompressFixBits(const U64Array* src, unsigned int bits, unsigned int level)
{
    // Skip values use a fixed-width bit stream with one end offset per level.
    if (!src || bits == 0 || bits > 64)
        exit(EXIT_FAILURE);

    uint64_t total_bits = (uint64_t)src->Count * bits;
    uint64_t word_count = (total_bits + 31) >> 5;

    FixBitArray* dst = InitFixBitArray(word_count + 1, level, bits);
    if (!dst)
        exit(EXIT_FAILURE);

    for (unsigned int i = 0; i < level; ++i) dst->Index[i] = src->Index[i];

    BitWriter writer;
    bw_init(&writer, total_bits);
    for (uint64_t i = 0; i < src->Count; ++i)
        bw_write_bits64(&writer, src->List[i], bits);
    free(dst->List);
    dst->List = writer.data;

    dst->Count = total_bits;

    return dst;
}


static inline void bw_write_gamma(BitWriter* bw, uint64_t x)
{
    if (x == 0u) exit(EXIT_FAILURE);
#if defined(__GNUC__) || defined(__clang__)
    unsigned int len = 63u - (unsigned int)__builtin_clzll(x);
#else
    unsigned int len = 0;
    uint64_t t = x;
    while (t >>= 1) len++;
#endif

    bw_reserve(bw, (uint64_t)len * 2u + 1u);
    bw->bitpos += len;
    for (int bit = (int)len; bit >= 0; --bit)
        bw_write_bit(bw, (unsigned int)((x >> bit) & 1u));
}


static int constant_layer(const unsigned int* bits, uint64_t start, uint64_t end)
{
    unsigned int first = BITSET_GET(bits, start);

    for (uint64_t i = start + 1; i < end; i++)
    {
        if (BITSET_GET(bits, i) != first)
            return -1;
    }

    return first;
}


RunBitArray* Compress01Gamma(const UIntArray* src, unsigned int level)
{
    // Constant levels use no payload; mixed levels encode alternating run lengths.
    uint64_t wordCount = ((uint64_t)src->Count + 31) >> 5;

    RunBitArray* dst = InitRunBitArray(wordCount * 2 + 64, level, level);
    if (!dst) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    BitWriter bw;
    bw_init(&bw, wordCount * 2 + 64);

    uint64_t start = 0;

    for (unsigned int lvl = 0; lvl < level - 1; lvl++) {

        uint64_t end = src->Index[lvl];

        if (end < start || end > src->Count) {
            fprintf(stderr, "Invalid index\n");
            exit(EXIT_FAILURE);
        }

        uint64_t len = end - start;

        if (len == 0) {
            dst->Flag[lvl] = 1;
            dst->First[lvl] = 0;
            dst->Index[lvl] = bw.bitpos;
            start = end;
            continue;
        }

        if (len == 1) {
            dst->Flag[lvl] = 1;

            unsigned int bit =
                (src->List[start >> 5] >> (start & 31)) & 1;

            dst->First[lvl] = bit;

            dst->Index[lvl] = bw.bitpos;

            start = end;
            continue;
        }

        int c = constant_layer(src->List, start, end);

        if (c != -1) {

            dst->Flag[lvl] = 1;
            dst->First[lvl] = c;

            dst->Index[lvl] = bw.bitpos;
        }
        else {

            dst->Flag[lvl] = 0;

            unsigned int cur = BITSET_GET(src->List, start);
            dst->First[lvl] = cur;

            uint64_t run = 1;

            for (uint64_t i = start + 1; i < end; i++) {
                unsigned int b = BITSET_GET(src->List, i);

                if (b == cur) {
                    run++;
                }
                else {
                    bw_write_gamma(&bw, run);
                    cur = b;
                    run = 1;
                }
            }

            bw_write_gamma(&bw, run);
            dst->Index[lvl] = bw.bitpos;
        }

        start = end;
    }

    if (level != 0u) {
        dst->Flag[level - 1u] = 1u;
        dst->First[level - 1u] = 0u;
        dst->Index[level - 1u] = bw.bitpos;
    }

    dst->List = bw.data;
    dst->Count = bw.bitpos;

    return dst;
}

static inline void bw_write_unary(BitWriter* bw, uint64_t q)
{
    if (q == UINT64_MAX) {
        fprintf(stderr, "Rice unary length overflow\n");
        exit(EXIT_FAILURE);
    }
    bw_reserve(bw, q + 1u);
    bw->bitpos += q;
    bw_write_bit(bw, 1u);
}


RiceArray* CompressRice(const U64Array* src, unsigned int level)
{
    if (src == NULL) {
        fprintf(stderr, "Invalid input\n");
        exit(EXIT_FAILURE);
    }

    // Select the Rice parameter independently for every decision-diagram level.
    RiceArray* dst = InitRiceArray(src->Count, src->Count, level, level);

    BitWriter bw_q;
    BitWriter bw_r;

    bw_init(&bw_q, src->Count * 16);
    bw_init(&bw_r, src->Count * 16);

    uint64_t start = 0;

    for (unsigned int lvl = 0; lvl < level; lvl++)
    {
        uint64_t end = src->Index[lvl];

        if (end < start || end > src->Count) {
            fprintf(stderr, "Invalid index\n");
            exit(EXIT_FAILURE);
        }

        uint64_t len = end - start;


        if (len == 0) {
            dst->quoIndex[lvl] = bw_q.bitpos;
            dst->remIndex[lvl] = bw_r.bitpos;
            dst->RiceParameter[lvl] = 0;
            start = end;
            continue;
        }



        unsigned int best_k = 0;
        uint64_t best_cost = UINT64_MAX;

        for (unsigned int k = 0; k <= 63; k++)
        {
            uint64_t cost = 0;

            for (uint64_t i = start; i < end; i++)
            {
                const uint64_t q = src->List[i] >> k;
                const uint64_t item = q + 1u + k;
                if (cost > UINT64_MAX - item) { cost = UINT64_MAX; break; }
                cost += item;
            }

            if (cost < best_cost)
            {
                best_cost = cost;
                best_k = k;
            }
        }


        for (uint64_t i = start; i < end; i++)
        {
            const uint64_t value = src->List[i];
            const uint64_t q = value >> best_k;
            const uint64_t mask = best_k == 0u ? 0u :
                ((UINT64_C(1) << best_k) - 1u);
            const uint64_t r = value & mask;


            bw_write_unary(&bw_q, q);


            if (best_k > 0)
                bw_write_bits64(&bw_r, r, best_k);
        }


        dst->RiceParameter[lvl] = best_k;

        dst->quoIndex[lvl] = bw_q.bitpos;
        dst->remIndex[lvl] = bw_r.bitpos;

        start = end;
    }


    free(dst->quotient);
    free(dst->remainder);
    dst->quotient = bw_q.data;
    dst->remainder = bw_r.data;

    dst->quoCount = bw_q.bitpos;
    dst->remCount = bw_r.bitpos;

    return dst;
}


BitArray* CompressSameBit(const UIntArray* src, unsigned int level)
{
    // Constant levels store only a two-bit tag; other levels retain their bits.
    BitArray* dst = (BitArray*)malloc(sizeof(BitArray));

    dst->Index = (uint64_t*)malloc(sizeof(uint64_t) * level);
    dst->Flag = (unsigned int*)malloc(sizeof(unsigned int) * level);


    uint64_t wordCount = ((uint64_t)src->Count + 31) >> 5;
    dst->List = (unsigned int*)calloc(wordCount ? wordCount : 1u, sizeof(unsigned int));

    uint64_t bitpos = 0;

    for (unsigned int lvl = 0; lvl < level - 1; lvl++)
    {
        uint64_t start = (lvl == 0) ? 0 : src->Index[lvl - 1];
        uint64_t end = src->Index[lvl];

        if (start == end) {
            dst->Flag[lvl] = 1;
            dst->Index[lvl] = bitpos;
            continue;
        }

        if (start + 1 == end) {
            unsigned int bit =
                (src->List[start >> 5] >> (start & 31)) & 1;
            dst->Flag[lvl] = bit;

            dst->Index[lvl] = bitpos;
            continue;
        }

        int c = constant_layer(src->List, start, end);

        if (c != -1)
        {
            dst->Flag[lvl] = c;
            dst->Index[lvl] = bitpos;
        }
        else
        {
            dst->Flag[lvl] = 2;

            for (uint64_t i = start; i < end; i++)
            {
                unsigned int bit =
                    (src->List[i >> 5] >> (i & 31)) & 1;

                if (bit)
                    dst->List[bitpos >> 5] |=
                    (1u << (bitpos & 31));

                bitpos++;
            }

            dst->Index[lvl] = bitpos;
        }
    }

    if (level != 0u) {
        dst->Flag[level - 1u] = 0u;
        dst->Index[level - 1u] = bitpos;
    }

    dst->Count = bitpos;
    return dst;
}

Graph_Compress* CreateBitGraph(Graph_Array* GA, Graph_Compress* CG) {
    CG->nips_t = GA->nips_t;
    CG->nips_v = GA->nips_v;
    unsigned int totallevel = CG->nips_v * 2 + CG->nips_t + 1;
    CG->VariatyNumList = (uint64_t*)malloc(sizeof(uint64_t) * totallevel);
    memcpy(CG->VariatyNumList, GA->VariatyNumList, sizeof(uint64_t) * totallevel);

    CG->NormalFlagList = CompressSameBit(GA->NormalFlagList, totallevel);
    CG->ShareFlagList = CompressSameBit(GA->ShareFlagList, totallevel);


    const uint64_t skip_limit = GA->MaxSkip == UINT64_MAX
        ? UINT64_MAX : GA->MaxSkip + 1u;
    unsigned int skip_bits = ComputeRequiredBits64(skip_limit);
    if (skip_bits == 0u) skip_bits = 1u;
    CG->SkipValue = CompressFixBits(
        GA->SkipValue, skip_bits, totallevel);


    CG->SkipFlagList = Compress01Gamma(GA->SkipFlagList, totallevel);
    CG->ShareNodeList = Compress01Gamma(GA->ShareNodeList, totallevel);


    CG->ShareValue = CompressRice(GA->ShareValue, totallevel);

    return CG;
}
