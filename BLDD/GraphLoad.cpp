#define _CRT_SECURE_NO_WARNINGS
#include "Graph.h"
#include "Bit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <errno.h>

#define OUT_DIR "output"

static FILE* open_in_dir(const char* dir, const char* name, const char* mode)
{
    if (!dir || dir[0] == '\0') {
        dir = OUT_DIR;
    }

    char path[512];
    int written = snprintf(path, sizeof(path), "%s/%s", dir, name);
    if (written < 0 || written >= (int)sizeof(path)) {
        fprintf(stderr, "open_in_dir: path too long (%s/%s)\n", dir, name);
        return NULL;
    }

    FILE* fp = fopen(path, mode);
    if (!fp) {
        fprintf(stderr, "open_in_dir: cannot open %s (mode=%s)\n", path, mode);
    }
    return fp;
}


static void bytes_to_words(const unsigned char* in, uint64_t bit_count, unsigned int* words)
{
    uint64_t word_count = (bit_count + 31) >> 5;
    memset(words, 0, sizeof(unsigned int) * (size_t)word_count);
    uint64_t byte_count = (bit_count + 7) >> 3;
    for (uint64_t b = 0; b < byte_count; b++) {
        words[b >> 2] |= ((unsigned int)in[b]) << ((b & 3) * 8);
    }
}

static void unpack_small_ints(const unsigned char* in, unsigned int n, unsigned int bits_per_elem,
    unsigned int* dst)
{
    unsigned int bitpos = 0;
    for (unsigned int i = 0; i < n; i++) {
        unsigned int v = 0;
        for (unsigned int k = 0; k < bits_per_elem; k++) {
            unsigned int bit = (in[bitpos >> 3] >> (bitpos & 7)) & 1u;
            v |= (bit << k);
            bitpos++;
        }
        dst[i] = v;
    }
}



// File size distinguishes legacy 32-bit indexes from the current 64-bit format.
typedef enum ArrayFileFormat {
    ARRAY_FORMAT_LEGACY32 = 0,
    ARRAY_FORMAT_BBDD64 = 1
} ArrayFileFormat;

static int seek_file(FILE* fp, long long offset, int origin)
{
#if defined(_WIN32)
    return _fseeki64(fp, (__int64)offset, origin);
#else
    return fseeko(fp, (off_t)offset, origin);
#endif
}

static int tell_file(FILE* fp, long long* position)
{
#if defined(_WIN32)
    __int64 value = _ftelli64(fp);
    if (value < 0) return -1;
    *position = (long long)value;
#else
    off_t value = ftello(fp);
    if (value < 0) return -1;
    *position = (long long)value;
#endif
    return 0;
}

static int get_file_size(FILE* fp, uint64_t* size)
{
    long long saved = 0;
    if (tell_file(fp, &saved) != 0 || seek_file(fp, 0, SEEK_END) != 0) return -1;
    long long end = 0;
    if (tell_file(fp, &end) != 0 || end < 0 || seek_file(fp, saved, SEEK_SET) != 0) return -1;
    *size = (uint64_t)end;
    return 0;
}

static uint64_t bits_to_bytes(uint64_t bits)
{
    if (bits > UINT64_MAX - 7u) return UINT64_MAX;
    return (bits + 7u) >> 3;
}

static int read_bit_stream(FILE* fp, uint64_t bit_count, unsigned int* words)
{
    uint64_t byte_count = bits_to_bytes(bit_count);
    if (byte_count > SIZE_MAX) return -1;
    return byte_count == 0 ||
        fread((unsigned char*)words, 1, (size_t)byte_count, fp) == (size_t)byte_count
        ? 0 : -1;
}

static int select_array_format(FILE* fp, uint64_t legacy_size, uint64_t bbdd_size,
    ArrayFileFormat* format, const char* name)
{
    uint64_t actual = 0;
    if (get_file_size(fp, &actual) != 0) return -1;

    if (actual == bbdd_size) {
        *format = ARRAY_FORMAT_BBDD64;
        return 0;
    }
    if (actual == legacy_size) {
        *format = ARRAY_FORMAT_LEGACY32;
        return 0;
    }

    fprintf(stderr,
        "load %s: unsupported file size=%llu (legacy=%llu, BBDD64=%llu)\n",
        name, (unsigned long long)actual,
        (unsigned long long)legacy_size, (unsigned long long)bbdd_size);
    return -1;
}

static int read_index_array(FILE* fp, ArrayFileFormat format, uint64_t* dst,
    unsigned int count, uint64_t limit, const char* what)
{
    for (unsigned int i = 0; i < count; ++i) {
        uint64_t value = 0;
        if (format == ARRAY_FORMAT_BBDD64) {
            if (fread(&value, sizeof(uint64_t), 1, fp) != 1) return -1;
        }
        else {
            unsigned int legacy = 0;
            if (fread(&legacy, sizeof(unsigned int), 1, fp) != 1) return -1;
            value = legacy;
        }
        if (i + 1 == count) {


            dst[i] = i == 0 ? 0 : dst[i - 1];
            continue;
        }
        if (value > limit) {
            fprintf(stderr, "load %s: index[%u]=%llu exceeds count=%llu\n",
                what, i, (unsigned long long)value, (unsigned long long)limit);
            return -1;
        }
        dst[i] = value;
    }
    return 0;
}

static int peek_count32(FILE* fp, unsigned int* count)
{
    if (seek_file(fp, 0, SEEK_SET) != 0 ||
        fread(count, sizeof(unsigned int), 1, fp) != 1 ||
        seek_file(fp, 0, SEEK_SET) != 0) return -1;
    return 0;
}

static int peek_count64(FILE* fp, uint64_t* count)
{
    if (seek_file(fp, 0, SEEK_SET) != 0 ||
        fread(count, sizeof(uint64_t), 1, fp) != 1 ||
        seek_file(fp, 0, SEEK_SET) != 0) return -1;
    return 0;
}

static int load_meta(Graph_Compress* CG, const char* input_dir, unsigned int totallevel,
    unsigned int expect_v, unsigned int expect_t)
{
    FILE* fp = open_in_dir(input_dir, "meta.txt", "rb");
    if (!fp) return -1;
    unsigned char prefix[8] = {};
    if (fread(prefix, 1, sizeof(prefix), fp) != sizeof(prefix) ||
        seek_file(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }
    static const unsigned char magic[8] = {
        'B', 'B', 'D', 'D', 'M', '6', '4', 2
    };
    unsigned int v = 0, t = 0;
    const int is_wide = memcmp(prefix, magic, sizeof(magic)) == 0;
    if (is_wide) {
        uint32_t version = 0;
        if (fread(prefix, 1, sizeof(prefix), fp) != sizeof(prefix) ||
            fread(&version, sizeof(version), 1, fp) != 1 || version != 2u ||
            fread(&v, sizeof(v), 1, fp) != 1 ||
            fread(&t, sizeof(t), 1, fp) != 1) {
            fclose(fp);
            return -1;
        }
    }
    else if (fread(&v, sizeof(unsigned int), 1, fp) != 1 ||
        fread(&t, sizeof(unsigned int), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    if (v != expect_v || t != expect_t) {
        fprintf(stderr, "load_meta: nips mismatch (file=%u/%u, expect=%u/%u)\n",
            v, t, expect_v, expect_t);
        fclose(fp);
        return -1;
    }
    CG->nips_v = v;
    CG->nips_t = t;
    CG->VariatyNumList = (uint64_t*)calloc(totallevel, sizeof(uint64_t));
    if (!CG->VariatyNumList) { fclose(fp); return -1; }
    if (is_wide) {
        if (fread(CG->VariatyNumList, sizeof(uint64_t), totallevel, fp) != totallevel) {
            fclose(fp); return -1;
        }
    }
    else {
        for (unsigned int i = 0; i < totallevel; ++i) {
            uint32_t value = 0;
            if (fread(&value, sizeof(value), 1, fp) != 1) {
                fclose(fp);
                return -1;
            }
            CG->VariatyNumList[i] = value;
        }
    }
    if (totallevel > 1 && CG->VariatyNumList[totallevel - 1] <
        CG->VariatyNumList[totallevel - 2]) {
        CG->VariatyNumList[totallevel - 1] = CG->VariatyNumList[totallevel - 2];
    }
    fclose(fp);
    return 0;
}

static int load_order(Graph_Compress* CG, const char* input_dir)
{
    if (!input_dir || !*input_dir) return 0;
    char path[512];
    const int written = snprintf(path, sizeof(path), "%s/order.txt", input_dir);
    if (written < 0 || written >= (int)sizeof(path)) return -1;
    FILE* fp = fopen(path, "rb");
    if (!fp) return errno == ENOENT ? 0 : -1;
    char order[4] = {};
    const bool read_ok = fread(order, 1, 3, fp) == 3;
    const bool close_ok = fclose(fp) == 0;
    const bool ok = read_ok && close_ok &&
        (memcmp(order, "UVT", 3) == 0 || memcmp(order, "UTV", 3) == 0);
    if (!ok) {
        fprintf(stderr, "load_order: invalid order.txt\n");
        return -1;
    }
    memcpy(CG->order, order, sizeof(order));
    return 0;
}

static BitArray* load_bit_array(const char* input_dir, const char* name, unsigned int totallevel)
{
    FILE* fp = open_in_dir(input_dir, name, "rb");
    if (!fp) return NULL;

    unsigned int legacy_count = 0;
    uint64_t bbdd_count = 0;
    if (peek_count32(fp, &legacy_count) != 0 || peek_count64(fp, &bbdd_count) != 0) {
        fclose(fp); return NULL;
    }

    uint64_t flag_bytes = ((uint64_t)2 * totallevel + 7u) >> 3;
    uint64_t legacy_size = 4u + (uint64_t)4 * totallevel + flag_bytes + bits_to_bytes(legacy_count);
    uint64_t bbdd_size = 8u + (uint64_t)8 * totallevel + flag_bytes + bits_to_bytes(bbdd_count);
    ArrayFileFormat format;
    if (select_array_format(fp, legacy_size, bbdd_size, &format, name) != 0) {
        fclose(fp); return NULL;
    }

    uint64_t count64 = 0;
    if (format == ARRAY_FORMAT_BBDD64) {
        if (fread(&count64, sizeof(uint64_t), 1, fp) != 1) { fclose(fp); return NULL; }
    }
    else {
        count64 = legacy_count;
        if (fread(&legacy_count, sizeof(unsigned int), 1, fp) != 1) { fclose(fp); return NULL; }
    }
    uint64_t count = count64;
    uint64_t word_count = (count + 31u) >> 5;
    unsigned int flag_bytes32 = (unsigned int)flag_bytes;

    BitArray* a = InitBitArray(word_count ? word_count : 1, totallevel, totallevel);
    if (!a) { fclose(fp); return NULL; }
    a->Count = count;

    if (read_index_array(fp, format, a->Index, totallevel, count64, name) != 0) goto err;

    if (flag_bytes32) {
        unsigned char* flag_buf = (unsigned char*)malloc(flag_bytes32);
        if (!flag_buf) goto err;
        if (fread(flag_buf, 1, flag_bytes32, fp) != flag_bytes32) { free(flag_buf); goto err; }
        unpack_small_ints(flag_buf, totallevel, 2, a->Flag);
        free(flag_buf);
    }

    if (read_bit_stream(fp, count, a->List) != 0) goto err;

    fclose(fp);
    return a;
err:
    free(a->List); free(a->Index); free(a->Flag); free(a);
    fclose(fp);
    return NULL;
}

static RunBitArray* load_run_bit_array(const char* input_dir, const char* name, unsigned int totallevel)
{
    FILE* fp = open_in_dir(input_dir, name, "rb");
    if (!fp) return NULL;

    unsigned int legacy_count = 0;
    uint64_t bbdd_count = 0;
    if (peek_count32(fp, &legacy_count) != 0 || peek_count64(fp, &bbdd_count) != 0) {
        fclose(fp); return NULL;
    }

    uint64_t small_bytes64 = ((uint64_t)totallevel + 7u) >> 3;
    uint64_t legacy_size = 4u + (uint64_t)4 * totallevel + 2u * small_bytes64 + bits_to_bytes(legacy_count);
    uint64_t bbdd_size = 8u + (uint64_t)8 * totallevel + 2u * small_bytes64 + bits_to_bytes(bbdd_count);
    ArrayFileFormat format;
    if (select_array_format(fp, legacy_size, bbdd_size, &format, name) != 0) {
        fclose(fp); return NULL;
    }

    uint64_t count64 = 0;
    if (format == ARRAY_FORMAT_BBDD64) {
        if (fread(&count64, sizeof(uint64_t), 1, fp) != 1) { fclose(fp); return NULL; }
    }
    else {
        count64 = legacy_count;
        if (fread(&legacy_count, sizeof(unsigned int), 1, fp) != 1) { fclose(fp); return NULL; }
    }
    uint64_t count = count64;
    uint64_t word_count = (count + 31u) >> 5;
    unsigned int small_bytes = (unsigned int)small_bytes64;

    RunBitArray* a = InitRunBitArray(word_count ? word_count : 1, totallevel, totallevel);
    if (!a) { fclose(fp); return NULL; }
    a->Count = count;

    if (read_index_array(fp, format, a->Index, totallevel, count64, name) != 0) goto err;

    if (small_bytes) {
        unsigned char* buf = (unsigned char*)malloc(small_bytes);
        if (!buf) goto err;
        if (fread(buf, 1, small_bytes, fp) != small_bytes) { free(buf); goto err; }
        unpack_small_ints(buf, totallevel, 1, a->Flag);
        if (fread(buf, 1, small_bytes, fp) != small_bytes) { free(buf); goto err; }
        unpack_small_ints(buf, totallevel, 1, a->First);
        free(buf);
    }

    if (read_bit_stream(fp, count, a->List) != 0) goto err;

    fclose(fp);
    return a;
err:
    free(a->List); free(a->Index); free(a->Flag); free(a->First); free(a);
    fclose(fp);
    return NULL;
}

static FixBitArray* load_fix_bit_array(const char* input_dir, const char* name, unsigned int totallevel)
{
    FILE* fp = open_in_dir(input_dir, name, "rb");
    if (!fp) return NULL;
    uint64_t* raw_index = NULL;
    uint64_t element_count = 0;
    int indexes_are_bits = 0;

    unsigned int legacy_count = 0;
    uint64_t bbdd_count = 0;
    if (peek_count32(fp, &legacy_count) != 0 || peek_count64(fp, &bbdd_count) != 0) {
        fclose(fp); return NULL;
    }

    uint64_t legacy_size = 4u + 4u + (uint64_t)4 * totallevel + bits_to_bytes(legacy_count);
    uint64_t bbdd_size = 8u + 4u + (uint64_t)8 * totallevel + bits_to_bytes(bbdd_count);
    ArrayFileFormat format;
    if (select_array_format(fp, legacy_size, bbdd_size, &format, name) != 0) {
        fclose(fp); return NULL;
    }

    uint64_t count64 = 0;
    unsigned int bpe = 0;
    if (format == ARRAY_FORMAT_BBDD64) {
        if (fread(&count64, sizeof(uint64_t), 1, fp) != 1 ||
            fread(&bpe, sizeof(unsigned int), 1, fp) != 1) { fclose(fp); return NULL; }
    }
    else {
        if (fread(&legacy_count, sizeof(unsigned int), 1, fp) != 1 ||
            fread(&bpe, sizeof(unsigned int), 1, fp) != 1) { fclose(fp); return NULL; }
        count64 = legacy_count;
    }
    uint64_t count = count64;
    // The current BDD decoder uses 32-bit internal node indexes and skip
    // offsets. External UTV node and time IDs are decoded from BDD bits and
    // remain 64-bit; this field stores internal offsets, not those IDs.
    if (bpe == 0 || bpe > 32) {
        fprintf(stderr, "load %s: invalid BitsPerElement=%u\n", name, bpe);
        fclose(fp); return NULL;
    }

    uint64_t word_count = (count + 31u) >> 5;

    FixBitArray* a = InitFixBitArray(word_count ? word_count : 1, totallevel, bpe);
    if (!a) { fclose(fp); return NULL; }
    a->Count = count;

    raw_index = (uint64_t*)calloc(totallevel, sizeof(uint64_t));
    if (!raw_index) goto err;
    for (unsigned int i = 0; i < totallevel; ++i) {
        if (format == ARRAY_FORMAT_BBDD64) {
            if (fread(&raw_index[i], sizeof(uint64_t), 1, fp) != 1) {
                free(raw_index); goto err;
            }
        }
        else {
            unsigned int legacy = 0;
            if (fread(&legacy, sizeof(unsigned int), 1, fp) != 1) {
                free(raw_index); goto err;
            }
            raw_index[i] = legacy;
        }
    }
    if (totallevel > 1) raw_index[totallevel - 1] = raw_index[totallevel - 2];



    element_count = (count64 % bpe == 0) ? count64 / bpe : count64;
    if (format == ARRAY_FORMAT_BBDD64 && bpe > 1) {
        for (unsigned int i = 0; i < totallevel; ++i) {
            if (raw_index[i] > element_count) {
                indexes_are_bits = 1;
                break;
            }
        }
    }
    for (unsigned int i = 0; i < totallevel; ++i) {
        if (indexes_are_bits) {
            if (raw_index[i] % bpe != 0) {
                fprintf(stderr, "load %s: bit index[%u]=%llu is not divisible by bpe=%u\n",
                    name, i, (unsigned long long)raw_index[i], bpe);
                free(raw_index); goto err;
            }
            raw_index[i] /= bpe;
        }
        if (raw_index[i] > element_count) {
            free(raw_index); goto err;
        }
        a->Index[i] = raw_index[i];
    }
    free(raw_index);

    if (read_bit_stream(fp, count, a->List) != 0) goto err;

    fclose(fp);
    return a;
err:
    free(a->List); free(a->Index); free(a);
    fclose(fp);
    return NULL;
}

static RiceArray* load_rice_array(const char* input_dir, const char* name, unsigned int totallevel)
{
    FILE* fp = open_in_dir(input_dir, name, "rb");
    if (!fp) return NULL;

    unsigned int legacy_qcnt = 0, legacy_rcnt = 0;
    uint64_t bbdd_qcnt = 0, bbdd_rcnt = 0;
    if (seek_file(fp, 0, SEEK_SET) != 0 ||
        fread(&legacy_qcnt, sizeof(unsigned int), 1, fp) != 1 ||
        fread(&legacy_rcnt, sizeof(unsigned int), 1, fp) != 1 ||
        seek_file(fp, 0, SEEK_SET) != 0 ||
        fread(&bbdd_qcnt, sizeof(uint64_t), 1, fp) != 1 ||
        fread(&bbdd_rcnt, sizeof(uint64_t), 1, fp) != 1 ||
        seek_file(fp, 0, SEEK_SET) != 0) {
        fclose(fp); return NULL;
    }

    uint64_t legacy_size = 8u + (uint64_t)12 * totallevel +
        bits_to_bytes(legacy_qcnt) + bits_to_bytes(legacy_rcnt);
    uint64_t bbdd_size = 16u + (uint64_t)20 * totallevel +
        bits_to_bytes(bbdd_qcnt) + bits_to_bytes(bbdd_rcnt);
    ArrayFileFormat format;
    if (select_array_format(fp, legacy_size, bbdd_size, &format, name) != 0) {
        fclose(fp); return NULL;
    }

    uint64_t qcnt64 = 0, rcnt64 = 0;
    if (format == ARRAY_FORMAT_BBDD64) {
        if (fread(&qcnt64, sizeof(uint64_t), 1, fp) != 1 ||
            fread(&rcnt64, sizeof(uint64_t), 1, fp) != 1) { fclose(fp); return NULL; }
    }
    else {
        if (fread(&legacy_qcnt, sizeof(unsigned int), 1, fp) != 1 ||
            fread(&legacy_rcnt, sizeof(unsigned int), 1, fp) != 1) { fclose(fp); return NULL; }
        qcnt64 = legacy_qcnt;
        rcnt64 = legacy_rcnt;
    }
    uint64_t qcnt = qcnt64;
    uint64_t rcnt = rcnt64;
    uint64_t qw = (qcnt + 31u) >> 5;
    uint64_t rw = (rcnt + 31u) >> 5;

    RiceArray* a = InitRiceArray(qw ? qw : 1, rw ? rw : 1, totallevel, totallevel);
    if (!a) { fclose(fp); return NULL; }
    a->quoCount = qcnt;
    a->remCount = rcnt;

    if (read_index_array(fp, format, a->quoIndex, totallevel, qcnt64, "ShareValue.quoIndex") != 0) goto err;
    if (read_index_array(fp, format, a->remIndex, totallevel, rcnt64, "ShareValue.remIndex") != 0) goto err;
    if (fread(a->RiceParameter, sizeof(unsigned int), totallevel, fp) != totallevel) goto err;

    if (read_bit_stream(fp, qcnt, a->quotient) != 0) goto err;
    if (read_bit_stream(fp, rcnt, a->remainder) != 0) goto err;

    fclose(fp);
    return a;
err:
    free(a->quotient); free(a->remainder);
    free(a->quoIndex); free(a->remIndex); free(a->RiceParameter);
    free(a);
    fclose(fp);
    return NULL;
}

Graph_Compress* LoadBitGraphFromTxt(Graph_Compress* CG, unsigned int nips_v, unsigned int nips_t, const char* input_dir)
{
    if (!CG) {
        fprintf(stderr, "LoadBitGraphFromTxt: NULL Graph_Compress\n");
        return NULL;
    }
    unsigned int totallevel = nips_v * 2 + nips_t + 1;

    if (load_meta(CG, input_dir, totallevel, nips_v, nips_t) != 0) {
        fprintf(stderr, "LoadBitGraphFromTxt: meta.txt failed\n");
        goto fail;
    }
    if (load_order(CG, input_dir) != 0) {
        fprintf(stderr, "LoadBitGraphFromTxt: order.txt failed\n");
        goto fail;
    }

    CG->NormalFlagList = load_bit_array(input_dir, "NormalFlagList.txt", totallevel);
    if (!CG->NormalFlagList) {
        fprintf(stderr, "LoadBitGraphFromTxt: NormalFlagList.txt failed\n");
        goto fail;
    }
    CG->ShareFlagList = load_bit_array(input_dir, "ShareFlagList.txt", totallevel);
    if (!CG->ShareFlagList) {
        fprintf(stderr, "LoadBitGraphFromTxt: ShareFlagList.txt failed\n");
        goto fail;
    }
    CG->SkipFlagList = load_run_bit_array(input_dir, "SkipFlagList.txt", totallevel);
    if (!CG->SkipFlagList) {
        fprintf(stderr, "LoadBitGraphFromTxt: SkipFlagList.txt failed\n");
        goto fail;
    }
    CG->ShareNodeList = load_run_bit_array(input_dir, "ShareNodeList.txt", totallevel);
    if (!CG->ShareNodeList) {
        fprintf(stderr, "LoadBitGraphFromTxt: ShareNodeList.txt failed\n");
        goto fail;
    }
    CG->SkipValue = load_fix_bit_array(input_dir, "SkipValue.txt", totallevel);
    if (!CG->SkipValue) {
        fprintf(stderr, "LoadBitGraphFromTxt: SkipValue.txt failed\n");
        goto fail;
    }
    CG->ShareValue = load_rice_array(input_dir, "ShareValue.txt", totallevel);
    if (!CG->ShareValue) {
        fprintf(stderr, "LoadBitGraphFromTxt: ShareValue.txt failed\n");
        goto fail;
    }

    return CG;
fail:
    fprintf(stderr, "LoadBitGraphFromTxt: failed (dir=%s)\n", input_dir ? input_dir : OUT_DIR);
    FreeGraphCompress(CG);
    return NULL;
}
