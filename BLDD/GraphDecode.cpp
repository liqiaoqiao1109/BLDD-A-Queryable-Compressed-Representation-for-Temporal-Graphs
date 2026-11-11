#include"Graph.h"
#include"Help.h"
#include"Bit.h"
#include"QueryAccelerate.h"




static inline unsigned int GetSameBitArrayBit(
    const BitArray* dst,
    unsigned int lvl,
    unsigned int pos)
{
    unsigned int flag = dst->Flag[lvl];
    

    if (flag == 0)
        return 0;
    if (flag == 1)
        return 1;


    uint64_t start =
        (lvl == 0) ? 0 : dst->Index[lvl - 1];

	uint64_t bitpos = start + pos;

	uint64_t word = bitpos >> 5;
    unsigned int off = bitpos & 31;


    return (dst->List[word] >> off) & 1u;
}



static inline uint32_t BitPopCount(const unsigned int* arr, uint64_t a, uint64_t b) {
    if (arr == NULL) return 0;
    if (b < a) return 0;

    uint32_t count = 0;


    for (uint64_t i = a; ; ++i) {
        count += (arr[i >> 5] >> (i & 31)) & 1u;

        if (i == b) break;
    }

    return count;
}


unsigned int SameBitPopCount(BitArray* src, uint32_t a, uint32_t b, unsigned int level) {
    if (src == NULL) return 0;
    if (b < a) return 0;


    if (src->Flag[level] == 1) {
        return b - a + 1;
    }


    if (src->Flag[level] == 0) {
        return 0;
    }


    uint64_t layer_start = (level == 0) ? 0u : src->Index[level - 1];

    uint64_t abs_a = layer_start + a;
    uint64_t abs_b = layer_start + b;

    return BitPopCount(src->List, abs_a, abs_b);
}



static inline unsigned int read_bits(
    const unsigned int* data,
    uint64_t pos,
    unsigned int k)
{
    uint64_t word = pos >> 5;
    unsigned int offset = pos & 31;

    unsigned long long v = data[word];

    if (offset + k > 32)
        v |= ((unsigned long long)data[word + 1] << 32);

    v >>= offset;

    return (unsigned int)(v & ((1ULL << k) - 1));
}


unsigned int RiceAccess(const RiceArray* ra, unsigned int level, unsigned int i) {
    if (ra == NULL) return 0;

    unsigned int k = ra->RiceParameter[level];

    uint64_t quo_start = (level == 0) ? 0 : ra->quoIndex[level - 1];
    uint64_t rem_start = (level == 0) ? 0 : ra->remIndex[level - 1];

    uint64_t pos = quo_start;


    for (unsigned int j = 0; j < i; j++) {
        while (BITSET_GET(ra->quotient, pos) == 0) {
            pos++;
        }
        pos++;
    }


    unsigned int q = 0;
    while (BITSET_GET(ra->quotient, pos) == 0) {
        q++;
        pos++;
    }


    unsigned int r = 0;
    if (k > 0) {
        uint64_t rpos = rem_start + (uint64_t)i * k;
        r = read_bits(ra->remainder, rpos, k);
    }

    return (q << k) | r;
}



unsigned int Select1_RunGamma(const RunBitArray* ra, unsigned int lvl, unsigned int i) {
    if (ra == NULL) return (unsigned int)-1;


    if (ra->Flag[lvl]) {
        if (ra->First[lvl] == 0) {
            return (unsigned int)-1;
        }


        return i;
    }

    uint64_t start = (lvl == 0) ? 0 : ra->Index[lvl - 1];
    uint64_t end = ra->Index[lvl];

    BitReader br;
    br_init(&br, ra->List, start);

    unsigned int pos = 0;
    unsigned int rank = 0;
    unsigned int cur_bit = ra->First[lvl];

    while (br.bitpos < end) {
        unsigned int run = br_read_gamma(&br);

        if (cur_bit == 1) {
            if (rank + run > i) {
                return pos + (i - rank);
            }

            rank += run;
        }

        pos += run;
        cur_bit ^= 1;
    }

    return (unsigned int)-1;
}


static inline unsigned int GetRunGammaBit(const RunBitArray* ra, unsigned int lvl, unsigned int pos)
{
    if (ra == NULL) return 0;


    if (ra->Flag[lvl]) {
        return ra->First[lvl];
    }

    uint64_t start = (lvl == 0) ? 0 : ra->Index[lvl - 1];
    uint64_t end = ra->Index[lvl];

    BitReader br;
    br_init(&br, ra->List, start);

    unsigned int elem_sum = 0;
    unsigned int cur_bit = ra->First[lvl];

    while (br.bitpos < end) {
        unsigned int run = br_read_gamma(&br);
        unsigned int new_sum = elem_sum + run;


        if (pos < new_sum) {
            return cur_bit;
        }

        elem_sum = new_sum;
        cur_bit ^= 1;
    }

    fprintf(stderr, "GetRunGammaBit: pos out of range\n");
    return 0;
}


unsigned int PopCount_RunGamma(const RunBitArray* ra, unsigned int lvl, unsigned int a, unsigned int b) {
    if (ra == NULL) return 0;
    if (b < a) return 0;


    if (ra->Flag[lvl]) {
        if (ra->First[lvl] == 0) {
            return 0;
        }

        return b - a + 1;
    }

    uint64_t start = (lvl == 0) ? 0 : ra->Index[lvl - 1];
    uint64_t end = ra->Index[lvl];

    BitReader br;
    br_init(&br, ra->List, start);

    unsigned int pos = 0;
    unsigned int cur_bit = ra->First[lvl];
    unsigned int count = 0;

    while (br.bitpos < end) {
        unsigned int run = br_read_gamma(&br);

        unsigned int run_start = pos;
        unsigned int run_end = pos + run - 1;


        if (run_start > b) {
            break;
        }


        if (cur_bit == 1 && run_end >= a) {
            unsigned int L = (run_start > a) ? run_start : a;
            unsigned int R = (run_end < b) ? run_end : b;

            count += R - L + 1;
        }

        pos += run;
        cur_bit ^= 1;
    }

    return count;
}



static inline unsigned int GetFixBitsLayerValue(const FixBitArray* fa, unsigned int lvl, unsigned int i) {
    if (fa == NULL) return 0;

    uint64_t start = (lvl == 0) ? 0 : fa->Index[lvl - 1];
    uint64_t idx = start + i;
    unsigned int bits = fa->BitsPerElement;

    if (bits == 0) return 0;

    unsigned long long bitpos = (unsigned long long)idx * bits;

    unsigned int value = 0;

    for (unsigned int j = 0; j < bits; j++) {
        unsigned long long cur = bitpos + j;

        uint64_t word = cur >> 5;
        unsigned int offset = (unsigned int)(cur & 31);

        unsigned int bit = (fa->List[word] >> offset) & 1u;

        value |= bit << j;
    }

    return value;
}



int* GetChildIndex(MinHeap* visitlist) {
    int* childindex = (int*)malloc(sizeof(int) * visitlist->size * 2);
    unsigned int parentid = 0;

    for (int i = 0; i < visitlist->size; i++) {
        parentid = visitlist->data[i].parent.id;
        childindex[2 * i] = 2 * parentid;
        childindex[2 * i + 1] = 2 * parentid + 1;
    }

    return childindex;
}




static inline int* GetSameBitArrayBit_Array(
    const BitArray* dst,
    unsigned int lvl,
    const int* pos_arr,
    unsigned int n)
{
    int* out_bits = (int*)malloc(sizeof(int) * n);
    if (!out_bits) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    unsigned int flag = dst->Flag[lvl];


    for (unsigned int i = 0; i < n; i++) {
        if (pos_arr[i] == -1)
            out_bits[i] = -1;
    }

    if (flag == 0) {
        for (unsigned int i = 0; i < n; i++)
            if (pos_arr[i] != -1) out_bits[i] = 0;
        return out_bits;
    }

    if (flag == 1) {
        for (unsigned int i = 0; i < n; i++)
            if (pos_arr[i] != -1) out_bits[i] = 1;
        return out_bits;
    }

    uint64_t start = (lvl == 0) ? 0 : dst->Index[lvl - 1];

    unsigned int i = 0;

    while (i < n) {


        if (pos_arr[i] == -1) {
            i++;
            continue;
        }

        uint64_t bitpos = start + (uint64_t)(unsigned int)pos_arr[i];
        uint64_t word = bitpos >> 5;


        unsigned int word_val = dst->List[word];


        while (i < n && pos_arr[i] != -1) {

            uint64_t cur_bitpos = start + (uint64_t)(unsigned int)pos_arr[i];


            if ((cur_bitpos >> 5) != word)
                break;

            unsigned int off = cur_bitpos & 31;
            out_bits[i] = (word_val >> off) & 1u;

            i++;
        }
    }

    return out_bits;
}



int* SameBitRankArray(
    BitArray* src,
    const int32_t* index_arr,
    uint32_t n,
    unsigned int level)
{
#if defined(QUERYBDD_REFERENCE_BUILD)
    int* out = ReferenceSameBitRankArray(src, index_arr, n, level);
#else
    int* out = AcceleratedSameBitRankArray(src, index_arr, n, level);
#endif
    if (out == NULL) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }
    return out;
}



static inline int* RiceAccess_Array(
    const RiceArray* ra,
    unsigned int level,
    const int* idx_arr,
    unsigned int n)
{
#if !defined(QUERYBDD_REFERENCE_BUILD)
    int* accelerated = AcceleratedRiceAccessArray(ra, level, idx_arr, n);
    if (accelerated == NULL) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }
    return accelerated;
#else
    int* out = (int*)malloc(sizeof(int) * n);
    if (!out) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    unsigned int k = ra->RiceParameter[level];
    uint64_t quo_start = (level == 0) ? 0 : ra->quoIndex[level - 1];
    uint64_t rem_start = (level == 0) ? 0 : ra->remIndex[level - 1];


    for (unsigned int i = 0; i < n; i++) {
        if (idx_arr[i] == -1)
            out[i] = -1;
    }

    unsigned int cur_idx = 0;
    uint64_t pos = quo_start;

    for (unsigned int i = 0; i < n; i++) {

        int idx = idx_arr[i] - 1;
        if (idx + 1 == -1) continue;


        while (cur_idx < (unsigned int)idx) {


            while (BITSET_GET(ra->quotient, pos) == 0) {
                pos++;
            }


            pos++;

            cur_idx++;
        }


        unsigned int q = 0;

        while (BITSET_GET(ra->quotient, pos) == 0) {
            q++;
            pos++;
        }


        pos++;

        unsigned int r = 0;
        if (k > 0) {
            uint64_t rpos = rem_start + (uint64_t)idx * k;
            r = read_bits(ra->remainder, rpos, k);
        }

        out[i] = (q << k) | r;

        cur_idx++;
    }

    return out;
#endif
}


static void radix_sort_queries(RankQuery* Q, RankQuery* tmp, uint32_t n)
{
    const uint32_t B = 1 << 16;
    uint32_t* cnt = (uint32_t*)calloc(B, sizeof(uint32_t));


    for (uint32_t i = 0; i < n; i++)
        cnt[Q[i].idx & 0xFFFF]++;
    for (uint32_t i = 1; i < B; i++)
        cnt[i] += cnt[i - 1];
    for (int i = n - 1; i >= 0; i--)
        tmp[--cnt[Q[i].idx & 0xFFFF]] = Q[i];

    memset(cnt, 0, B * sizeof(uint32_t));


    for (uint32_t i = 0; i < n; i++)
        cnt[(tmp[i].idx >> 16) & 0xFFFF]++;
    for (uint32_t i = 1; i < B; i++)
        cnt[i] += cnt[i - 1];
    for (int i = n - 1; i >= 0; i--)
        Q[--cnt[(tmp[i].idx >> 16) & 0xFFFF]] = tmp[i];

    free(cnt);
}

static inline uint32_t lower_bound_query(
    RankQuery* Q, uint32_t l, uint32_t r, uint32_t target)
{
    while (l < r) {
        uint32_t mid = (l + r) >> 1;
        if (Q[mid].idx < target)
            l = mid + 1;
        else
            r = mid;
    }
    return l;
}


int* Select1_RunGamma_batch(const RunBitArray* ra,
    unsigned int lvl,
    const int* index_arr,
    unsigned int n)
{
    int* result = (int*)malloc(sizeof(int) * n);
    if (!result) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }


    uint32_t valid_cnt = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (index_arr[i] != -1) {
            valid_cnt++;
        }
        else {
            result[i] = -1;
        }
    }
    if (valid_cnt == 0) return result;


    RankQuery* Q = (RankQuery*)malloc(sizeof(RankQuery) * valid_cnt);
    RankQuery* tmp = (RankQuery*)malloc(sizeof(RankQuery) * valid_cnt);
    if (!Q || !tmp) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    uint32_t k = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (index_arr[i] != -1) {
            Q[k].idx = (uint32_t)index_arr[i];
            Q[k].pos = i;
            k++;
        }
    }


    radix_sort_queries(Q, tmp, valid_cnt);


    if (ra->Flag[lvl]) {
        if (ra->First[lvl] == 0) {
            for (uint32_t i = 0; i < valid_cnt; i++)
                result[Q[i].pos] = -1;
        }
        else {
            for (uint32_t i = 0; i < valid_cnt; i++)
                result[Q[i].pos] = Q[i].idx;
        }
        free(Q);
        free(tmp);
        return result;
    }


    uint64_t start = (lvl == 0) ? 0 : ra->Index[lvl - 1];
    uint64_t end = ra->Index[lvl];

    BitReader br;
    br_init(&br, ra->List, start);

    unsigned int cur = ra->First[lvl];
    unsigned int pos = 0;
    unsigned int count = 0;
    uint32_t qptr = 0;

    uint32_t max_query = Q[valid_cnt - 1].idx;

    while (br.bitpos < end && qptr < valid_cnt) {

        unsigned int run = br_read_gamma(&br);

        if (cur) {
            unsigned int end_count = count + run;


            if (count > max_query) break;


            uint32_t next_qptr = lower_bound_query(
                Q, qptr, valid_cnt, end_count);

            for (uint32_t i = qptr; i < next_qptr; i++) {
                result[Q[i].pos] = pos + (Q[i].idx - count);
            }

            qptr = next_qptr;
            count = end_count;
        }

        pos += run;
        cur ^= 1;
    }


    while (qptr < valid_cnt) {
        result[Q[qptr].pos] = -1;
        qptr++;
    }

    free(Q);
    free(tmp);
    return result;
}

static inline int* GetRunGammaBit_Array(
    const RunBitArray* ra,
    unsigned int lvl,
    const int* pos_arr,
    unsigned int n)
{
    int* out = (int*)malloc(sizeof(int) * n);
    if (!out) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }


    if (ra->Flag[lvl]) {
        int val = ra->First[lvl];
        for (uint32_t i = 0; i < n; i++) {
            out[i] = (pos_arr[i] == -1) ? -1 : val;
        }
        return out;
    }


    uint64_t start = (lvl == 0) ? 0 : ra->Index[lvl - 1];
    uint64_t end = ra->Index[lvl];

    BitReader br;
    br_init(&br, ra->List, start);

    unsigned int cur_bit = ra->First[lvl];
    unsigned int sum = 0;
    uint32_t qi = 0;


    while (qi < n && pos_arr[qi] == -1) {
        out[qi] = -1;
        qi++;
    }


    while (br.bitpos < end && qi < n) {

        unsigned int run = br_read_gamma(&br);
        unsigned int next_sum = sum + run;


        while (qi < n && pos_arr[qi] != -1 && pos_arr[qi] < next_sum) {
            out[qi] = cur_bit;
            qi++;
        }

        sum = next_sum;
        cur_bit ^= 1;


        while (qi < n && pos_arr[qi] == -1) {
            out[qi] = -1;
            qi++;
        }
    }


    while (qi < n) {
        if (pos_arr[qi] == -1) {
            out[qi] = -1;
        }
        else {

            out[qi] = 0;
        }
        qi++;
    }

    return out;
}

static inline int* PopCount_RunGamma_RankArray(
    const RunBitArray* ra,
    unsigned int lvl,
    const int* index_arr,
    unsigned int n)
{
    int* out = (int*)malloc(sizeof(int) * n);
    if (!out) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }


    if (ra->Flag[lvl]) {
        if (ra->First[lvl] == 0) {
            for (uint32_t i = 0; i < n; i++) {
                out[i] = (index_arr[i] == -1) ? -1 : 0;
            }
        }
        else {
            for (uint32_t i = 0; i < n; i++) {
                out[i] = (index_arr[i] == -1) ? -1 : (index_arr[i] + 1);
            }
        }
        return out;
    }


    uint64_t start = (lvl == 0) ? 0 : ra->Index[lvl - 1];
    uint64_t end = ra->Index[lvl];

    BitReader br;
    br_init(&br, ra->List, start);

    unsigned int cur = ra->First[lvl];
    unsigned int sum = 0;
    unsigned int rank = 0;
    uint32_t qi = 0;


    while (qi < n && index_arr[qi] == -1) {
        out[qi] = -1;
        qi++;
    }


    while (br.bitpos < end && qi < n) {

        unsigned int run = br_read_gamma(&br);
        unsigned int next_sum = sum + run;


        while (qi < n && index_arr[qi] != -1 && index_arr[qi] < next_sum) {

            if (cur) {

                unsigned int offset = index_arr[qi] - sum;
                out[qi] = rank + offset + 1;
            }
            else {

                out[qi] = rank;
            }

            qi++;
        }


        if (cur) {
            rank += run;
        }

        sum = next_sum;
        cur ^= 1;


        while (qi < n && index_arr[qi] == -1) {
            out[qi] = -1;
            qi++;
        }
    }


    while (qi < n) {
        if (index_arr[qi] == -1) {
            out[qi] = -1;
        }
        else {
            out[qi] = rank;
        }
        qi++;
    }

    return out;
}


static inline int* GetFixBitsLayerValue_Array(
    const FixBitArray* fa,
    unsigned int lvl,
    const int* idx_arr,
    unsigned int n)
{
#if !defined(QUERYBDD_REFERENCE_BUILD)
    int* accelerated = AcceleratedFixBitsAccessArray(fa, lvl, idx_arr, n);
    if (!accelerated) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }
    return accelerated;
#else
    int* out = (int*)malloc(sizeof(int) * n);
    if (!out) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    uint64_t start = (lvl == 0) ? 0 : fa->Index[lvl - 1];
    unsigned int bits = fa->BitsPerElement;
    unsigned int mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1);

    uint64_t prev_bitpos = 0;
    int first = 1;

    for (unsigned int i = 0; i < n; i++) {

        if (idx_arr[i] == -1) {
            out[i] = -1;
            continue;
        }

        uint64_t bitpos;

        if (first) {

            uint64_t idx = start + (unsigned int)(idx_arr[i] - 1);
            bitpos = (uint64_t)idx * bits;
            first = 0;
        }
        else {

            int delta = idx_arr[i] - idx_arr[i - 1];
            bitpos = prev_bitpos + (uint64_t)delta * bits;
        }

        prev_bitpos = bitpos;

        uint64_t word = bitpos >> 5;
        unsigned int offset = bitpos & 31;

        unsigned int v = fa->List[word] >> offset;


        if (offset + bits <= 32) {
            out[i] = v & mask;
        }
        else {

            unsigned int remain = 32 - offset;
            unsigned int next = fa->List[word + 1] << remain;
            out[i] = (v | next) & mask;
        }
    }

    return out;
#endif
}



unsigned int UnpackOne(const Graph_Compress& CG, unsigned int nodeindex, unsigned int* level) {
#if defined(QUERYBDD_REFERENCE_BUILD)
    return ReferenceUnpackOne(CG, nodeindex, level);
#else
    return AcceleratedUnpackOne(CG, nodeindex, level);
#endif
}



MinHeap* UnpackAll(const Graph_Compress& CG, MinHeap* visitlist) {
#if !defined(QUERYBDD_REFERENCE_BUILD)
    if (QueryAccelerationBytes(CG) != 0u) {
        return AcceleratedUnpackAll(CG, visitlist);
    }
#endif
    unsigned int curlevel = visitlist->data->parent.level;
    int* shareindex = NULL, * ChildIndex = NULL, * normalcount = NULL, * normalflag = NULL, * shareflag = NULL, * sharecount = NULL,
        * skipindex = NULL, * skipflag = NULL, * skipcount = NULL, * skipvalue = NULL, * sharevalue = NULL;
    unsigned int totallevel = CG.nips_v * 2 + CG.nips_t;
    unsigned int i = 0, count = 0;

    ChildIndex = GetChildIndex(visitlist);
    shareindex = (int*)malloc(sizeof(int) * visitlist->size * 2);


    if (curlevel == (totallevel - 1)) {
        for (i = 0; i < visitlist->size; i++) {
            visitlist->data[i].left.id = GetSameBitArrayBit(CG.NormalFlagList, curlevel, 2 * visitlist->data[i].parent.id);
            visitlist->data[i].right.id = GetSameBitArrayBit(CG.NormalFlagList, curlevel, 2 * visitlist->data[i].parent.id + 1);
            visitlist->data[i].left.level = curlevel + 1;
            visitlist->data[i].right.level = curlevel + 1;
        }
        free(shareindex);
        free(ChildIndex);
        return visitlist;
    }

    normalflag = GetSameBitArrayBit_Array(CG.NormalFlagList, curlevel, ChildIndex, visitlist->size * 2);
    normalcount = SameBitRankArray(CG.NormalFlagList, ChildIndex, visitlist->size * 2, curlevel);
    for (i = 0; i < visitlist->size * 2; i++) {
        if (normalflag[i]) {
            shareindex[i] = -1;
            if (i % 2 == 0) {
                visitlist->data[i / 2].left.id = normalcount[i] - 1;
                visitlist->data[i / 2].left.level = curlevel + 1;
                count++;
            }
            else {
                visitlist->data[i / 2].right.id = normalcount[i] - 1;
                visitlist->data[i / 2].right.level = curlevel + 1;
                count++;
            }
        }
        else {
            shareindex[i] = ChildIndex[i] - normalcount[i];
        }
        if (count == (visitlist->size * 2)) {
            free(shareindex);
            free(ChildIndex);
            free(normalcount);
            free(normalflag);
            return visitlist;
        }
    }

    shareflag = GetSameBitArrayBit_Array(CG.ShareFlagList, curlevel, shareindex, visitlist->size * 2);
    sharecount = SameBitRankArray(CG.ShareFlagList, shareindex, visitlist->size * 2, curlevel);
    skipindex = (int*)malloc(sizeof(int) * visitlist->size * 2);
    for (i = 0; i < visitlist->size * 2; i++) {
		if (shareflag[i] == 1) {
            skipindex[i] = -1;
        }
		else if (shareflag[i] == 0) {
            skipindex[i] = shareindex[i] - sharecount[i];
            sharecount[i] = -1;
        }
        else {
            skipindex[i] = -1;
        }
    }
    sharevalue = RiceAccess_Array(CG.ShareValue, curlevel, sharecount, visitlist->size * 2);
    {
        int* selected = Select1_RunGamma_batch(CG.ShareNodeList, curlevel + 1, sharevalue, visitlist->size * 2);
        free(sharevalue);
        sharevalue = selected;
    }
    for (i = 0; i < visitlist->size * 2; i++) {
        if (sharecount[i] != -1) {
            count++;
            if (i % 2 == 0) {
                visitlist->data[i / 2].left.id = sharevalue[i];
                visitlist->data[i / 2].left.level = curlevel + 1;
            }
            else {
                visitlist->data[i / 2].right.id = sharevalue[i];
                visitlist->data[i / 2].right.level = curlevel + 1;
            }
        }
        if (count == (visitlist->size * 2)) {
            free(shareindex);
            free(ChildIndex);
            free(normalcount);
            free(normalflag);
            free(shareflag);
            free(sharecount);
            free(skipindex);
            free(sharevalue);
            return visitlist;
        }
    }

#if defined(QUERYBDD_REFERENCE_BUILD)
    skipflag = GetRunGammaBit_Array(CG.SkipFlagList, curlevel, skipindex, visitlist->size * 2);
#else
    if (!AcceleratedRunGammaBitRankArray(
        CG.SkipFlagList,
        curlevel,
        skipindex,
        visitlist->size * 2,
        &skipflag,
        &skipcount)) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }
#endif
    for (i = 0; i < visitlist->size * 2; i++) {
        if (skipflag[i] == 0) {
            if (i % 2 == 0) {
                visitlist->data[i / 2].left.id = 0;
                visitlist->data[i / 2].left.level = totallevel;
            }
            else {
                visitlist->data[i / 2].right.id = 0;
                visitlist->data[i / 2].right.level = totallevel;
            }
            skipindex[i] = -1;
#if !defined(QUERYBDD_REFERENCE_BUILD)
            skipcount[i] = -1;
#endif
        }
    }
#if defined(QUERYBDD_REFERENCE_BUILD)
    skipcount = PopCount_RunGamma_RankArray(CG.SkipFlagList, curlevel, skipindex, visitlist->size * 2);
#endif
    skipvalue = GetFixBitsLayerValue_Array(CG.SkipValue, curlevel, skipcount, visitlist->size * 2);
    for (i = 0; i < visitlist->size * 2; i++) {
        if (skipindex[i] != -1) {
            count++;
            unsigned int truelevel = curlevel + 2;
            
            while (truelevel < totallevel &&
                skipvalue[i] >= CG.VariatyNumList[truelevel] - CG.VariatyNumList[truelevel - 1]) {
                skipvalue[i] -= (CG.VariatyNumList[truelevel] - CG.VariatyNumList[truelevel - 1]);
                truelevel++;
            }
            if (i % 2 == 0) {
                visitlist->data[i / 2].left.id = skipvalue[i];
                visitlist->data[i / 2].left.level = truelevel;
            }
            else {
                visitlist->data[i / 2].right.id = skipvalue[i];
                visitlist->data[i / 2].right.level = truelevel;
            }
        }
        if (count == (visitlist->size * 2)) {
            free(shareindex);
            free(ChildIndex);
            free(normalcount);
            free(normalflag);
            free(shareflag);
            free(sharecount);
            free(skipindex);
            free(skipflag);
            free(skipcount);
            free(skipvalue);
            free(sharevalue);
            return visitlist;
        }
    }

    free(shareindex);
    free(ChildIndex);
    free(normalcount);
    free(normalflag);
    free(shareflag);
    free(sharecount);
    free(skipindex);
    free(skipflag);
    free(skipcount);
    free(skipvalue);
    free(sharevalue);
    return visitlist;
}
