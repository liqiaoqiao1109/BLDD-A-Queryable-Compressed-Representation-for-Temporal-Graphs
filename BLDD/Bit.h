#define BIT_WORD 32

#define BITSET_WORD(i)   ((uint64_t)(i) >> 5)
#define BITSET_OFFSET(i) ((unsigned int)((i) & 31))

#define BITSET_SET(arr, i) \
    ((arr)[BITSET_WORD(i)] |= (1u << BITSET_OFFSET(i)))

#define BITSET_CLR(arr, i) \
    ((arr)[BITSET_WORD(i)] &= ~(1u << BITSET_OFFSET(i)))

#define BITSET_GET(arr, i) \
    (((arr)[BITSET_WORD(i)] >> BITSET_OFFSET(i)) & 1u)


typedef struct BitWriter {
    unsigned int* data;
    uint64_t capacity_words;
    uint64_t bitpos;
} BitWriter;

typedef struct BitReader {
    const unsigned int* data;
    uint64_t bitpos;
} BitReader;


extern void bw_init(BitWriter* bw, uint64_t init_bits);
extern void br_init(BitReader* br, const unsigned int* data, uint64_t start);
extern void bw_reserve(BitWriter* bw, uint64_t more_bits);
extern void bw_write_bit(BitWriter* bw, unsigned int bit);
extern void bw_write_bits(BitWriter* bw, unsigned int v, unsigned int bits);
extern void bw_write_bits64(BitWriter* bw, uint64_t v, unsigned int bits);
extern unsigned int br_read_gamma(BitReader* br);
