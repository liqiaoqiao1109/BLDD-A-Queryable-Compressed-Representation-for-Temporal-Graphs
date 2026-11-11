#include "QueryAccelerate.h"
#include "Bit.h"
#include "Help.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <intrin.h>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace {

// The query path always builds this prune-oriented cache; no runtime mode exists.
constexpr std::uint64_t kCacheBudgetBytes = 256ull * 1024ull * 1024ull;
constexpr std::uint64_t kResidentCacheBudgetBytes = 240ull * 1024ull * 1024ull;
constexpr std::uint64_t kRankBudgetBytes = 70ull * 1024ull * 1024ull;
constexpr std::uint64_t kRiceBudgetBytes = 30ull * 1024ull * 1024ull;
constexpr std::uint64_t kGammaBudgetBytes = 10ull * 1024ull * 1024ull;
constexpr std::size_t kNeighborMemoBudgetBytes = 4ull * 1024ull * 1024ull;
constexpr std::size_t kBranchChildBudgetBytes = 4ull * 1024ull * 1024ull;
constexpr std::size_t kNeighborStackBudgetBytes = 1ull * 1024ull * 1024ull;
constexpr std::size_t kParityCacheBudgetBytes = 1ull * 1024ull * 1024ull;
constexpr std::size_t kBatchWorkspaceBudgetBytes = 4ull * 1024ull * 1024ull;
constexpr std::size_t kTemporalRootCacheCapacity = 1ull << 16;
constexpr unsigned int kNormalRankBlockShift = 6u;
constexpr unsigned int kShareRankBlockShift = 7u;
constexpr unsigned int kDefaultRiceStride = 16u;
constexpr unsigned int kDefaultGammaStride = 32u;
constexpr unsigned int kBuildCheckMask = (1u << 16) - 1u;

using BuildClock = std::chrono::steady_clock;
using BuildDeadline = BuildClock::time_point;

struct BitRankDir {
    unsigned int blockShift = kShareRankBlockShift;
    unsigned int levelCount = 0;
    std::uint64_t entryCount = 0;
    bool uses64 = false;
    std::unique_ptr<std::uint32_t[]> prefix32;
    std::unique_ptr<std::uint64_t[]> prefix64;
    std::unique_ptr<std::uint64_t[]> levelBaseRanks;

    std::size_t bytes() const {
        return static_cast<std::size_t>(entryCount) *
            (uses64 ? sizeof(std::uint64_t) : sizeof(std::uint32_t)) +
            static_cast<std::size_t>(levelCount) * sizeof(std::uint64_t);
    }
};

struct RiceSelectDir {
    unsigned int stride = 0;
    unsigned int levelCount = 0;
    std::uint64_t oneCount = 0;
    std::uint64_t sampleCount = 0;
    bool uses64 = false;
    std::unique_ptr<std::uint32_t[]> samples32;
    std::unique_ptr<std::uint64_t[]> samples64;
    std::unique_ptr<std::uint64_t[]> levelBaseOnes;

    std::size_t bytes() const {
        return static_cast<std::size_t>(sampleCount) *
            (uses64 ? sizeof(std::uint64_t) : sizeof(std::uint32_t)) +
            static_cast<std::size_t>(levelCount) * sizeof(std::uint64_t);
    }
};

struct GammaCheckpoint {
    std::uint32_t bitPosition;
    std::uint32_t position;
    std::uint32_t rank;
    std::uint8_t currentBit;
    std::uint8_t padding[3];
};
static_assert(sizeof(GammaCheckpoint) == 16, "Gamma checkpoints must stay compact");

struct RunGammaCkpt {
    unsigned int stride = 0;
    unsigned int levelCount = 0;
    std::uint64_t checkpointCount = 0;
    std::unique_ptr<std::uint64_t[]> levelOffsets;
    std::unique_ptr<std::uint32_t[]> levelPositions;
    std::unique_ptr<std::uint32_t[]> levelOneCounts;
    std::unique_ptr<GammaCheckpoint[]> checkpoints;

    std::size_t bytes() const {
        return static_cast<std::size_t>(checkpointCount) * sizeof(GammaCheckpoint) +
            static_cast<std::size_t>(levelCount + 1u) * sizeof(std::uint64_t) +
            static_cast<std::size_t>(levelCount) * sizeof(std::uint32_t) * 2u;
    }
};

struct NeighborMemoEntry {
    std::uint32_t nodeId;
    std::uint16_t generation;
    std::uint8_t nodeLevel;
    std::uint8_t state;
};
static_assert(sizeof(NeighborMemoEntry) == 8,
    "Neighbor memo entries must stay compact");

struct BranchChildEntry {
    std::uint32_t nodeId;
    std::uint32_t childId;
    std::uint8_t nodeLevel;
    std::uint8_t childLevel;
    std::uint8_t branch;
    std::uint8_t valid;
    std::uint32_t padding;
};
static_assert(sizeof(BranchChildEntry) == 16,
    "Branch child cache entries must stay compact");
constexpr std::size_t kBranchChildCapacity =
    kBranchChildBudgetBytes / sizeof(BranchChildEntry);
static_assert((kBranchChildCapacity & (kBranchChildCapacity - 1u)) == 0u,
    "Branch child capacity must be a power of two");

struct BranchChildCache {
    std::unique_ptr<BranchChildEntry[]> entries;

    std::size_t bytes() const {
        return entries == nullptr ? 0u : kBranchChildBudgetBytes;
    }
};

struct NeighborDfsState {
    std::uint32_t nodeId;
    std::uint32_t fixedBits;
    std::uint32_t freeMask;
    std::uint8_t nodeLevel;
    std::uint8_t depth;
    std::uint16_t reserved;
};
static_assert(sizeof(NeighborDfsState) == 16,
    "Neighbor DFS states must stay compact");

struct ParityCacheEntry {
    std::uint32_t nodeId;
    std::uint8_t nodeLevel;
    std::uint8_t value;
    std::uint8_t valid;
    std::uint8_t padding;
};
static_assert(sizeof(ParityCacheEntry) == 8,
    "Parity cache entries must stay compact");

struct ParityCacheState {
    std::unique_ptr<ParityCacheEntry[]> entries;

    std::size_t bytes() const {
        return entries == nullptr ? 0u : kParityCacheBudgetBytes;
    }
};

struct BatchDecodeWorkspace {
    std::unique_ptr<std::uint32_t[]> order;

    std::size_t capacity() const {
        return order == nullptr
            ? 0u
            : kBatchWorkspaceBudgetBytes / sizeof(std::uint32_t);
    }

    std::size_t bytes() const {
        return order == nullptr ? 0u : kBatchWorkspaceBudgetBytes;
    }
};

struct CoreProfileCounters {
    std::uint64_t queries = 0u;
    std::uint64_t sourceTableHits = 0u;
    std::uint64_t sourceWalks = 0u;
    std::uint64_t sourceSteps = 0u;
    std::uint64_t targetSteps = 0u;
    std::uint64_t timeSteps = 0u;
    std::uint64_t branchHits = 0u;
    std::uint64_t branchMisses = 0u;
    std::uint64_t normalRanks = 0u;
    std::uint64_t shareRanks = 0u;
    std::uint64_t riceSelects = 0u;
    std::uint64_t shareGammas = 0u;
    std::uint64_t skipGammas = 0u;
};

enum class CoreQueryStage : std::uint8_t {
    None,
    Source,
    Target,
    Time
};

struct CoreQueryTrace {
    CoreQueryStage stage = CoreQueryStage::None;
    CoreProfileCounters counters;
};

struct NeighborProfileCounters {
    std::uint64_t queries = 0;
    std::uint64_t sourceLookupNs = 0;
    std::uint64_t sourceTableHits = 0;
    std::uint64_t sourceWalks = 0;
    std::uint64_t memoHits = 0;
    std::uint64_t memoMisses = 0;
    std::uint64_t inactivePrunes = 0;
    std::uint64_t childHits = 0;
    std::uint64_t childMisses = 0;
    std::uint64_t timePredicates = 0;
    std::uint64_t outputCount = 0;
    std::uint64_t emptyHits = 0;
    std::uint64_t emptyMisses = 0;
    std::uint64_t fallbacks = 0;
};

struct NeighborAccelerationState {
    bool attempted = false;
    bool enabled = false;
    std::uint16_t generation = 0;
    std::uint32_t memoCount = 0;
    std::size_t memoCapacity = 0u;
    std::size_t bytes = 0;
    std::unique_ptr<NeighborMemoEntry[]> memo;
    std::unique_ptr<NeighborDfsState[]> stack;
    NeighborProfileCounters profile;
};

struct TemporalEdgeRootEntry {
    std::uint32_t u;
    std::uint32_t v;
    std::uint32_t nodeId;
    std::uint8_t nodeLevel;
    std::uint8_t valid;
    std::uint16_t reserved;
};
static_assert(sizeof(TemporalEdgeRootEntry) == 16,
    "Temporal edge-root entries must stay compact");

struct TemporalAccelerationState {
    bool rootCacheAttempted = false;
    std::size_t bytes = 0u;
    std::unique_ptr<TemporalEdgeRootEntry[]> edgeRoots;
};

struct QueryAccelerationCache {
    BitRankDir normalRank;
    BitRankDir shareRank;
    RiceSelectDir riceSelect;
    RunGammaCkpt shareNodeGamma;
    RunGammaCkpt skipFlagGamma;
    ParityCacheState parity;
    BranchChildCache branchChildren;
    BatchDecodeWorkspace batch;
    CoreProfileCounters coreProfile;
    NeighborAccelerationState neighbor;
    TemporalAccelerationState temporal;
    std::size_t bytes = 0;
};

std::mutex g_query_cache_mutex;
std::unordered_map<const Graph_Compress*, std::unique_ptr<QueryAccelerationCache>>
    g_query_caches;

thread_local const Graph_Compress* g_thread_graph = nullptr;
thread_local const QueryAccelerationCache* g_thread_cache = nullptr;
thread_local CoreQueryTrace* g_core_query_trace = nullptr;

const QueryAccelerationCache* FindQueryAcceleration(const Graph_Compress& graph);

void ClearThreadAcceleration(const Graph_Compress& graph) {
    if (g_thread_graph == &graph) {
        g_thread_graph = nullptr;
        g_thread_cache = nullptr;
    }
}

bool QueryVerificationEnabled() {
    static const bool enabled = []() {
        char value[8] = {};
        std::size_t value_length = 0;
        return getenv_s(&value_length, value, sizeof(value),
            "QUERYBDD_ACCEL_VERIFY") == 0 && value_length != 0 &&
            std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool CoreProfilingEnabled() {
    static const bool enabled = []() {
        char value[8] = {};
        std::size_t value_length = 0;
        return getenv_s(&value_length, value, sizeof(value),
            "QUERYBDD_ACCEL_PROFILE") == 0 && value_length != 0 &&
            std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool NeighborVerificationEnabled() {
    static const bool enabled = []() {
        char value[8] = {};
        std::size_t value_length = 0;
        return getenv_s(&value_length, value, sizeof(value),
            "QUERYBDD_NEIGHBOR_VERIFY") == 0 && value_length != 0 &&
            std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool NeighborProfilingEnabled() {
    static const bool enabled = []() {
        char value[8] = {};
        std::size_t value_length = 0;
        return getenv_s(&value_length, value, sizeof(value),
            "QUERYBDD_NEIGHBOR_PROFILE") == 0 && value_length != 0 &&
            std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

void AddCoreProfile(
    CoreProfileCounters* destination,
    const CoreProfileCounters& source) {
    destination->queries += source.queries;
    destination->sourceTableHits += source.sourceTableHits;
    destination->sourceWalks += source.sourceWalks;
    destination->sourceSteps += source.sourceSteps;
    destination->targetSteps += source.targetSteps;
    destination->timeSteps += source.timeSteps;
    destination->branchHits += source.branchHits;
    destination->branchMisses += source.branchMisses;
    destination->normalRanks += source.normalRanks;
    destination->shareRanks += source.shareRanks;
    destination->riceSelects += source.riceSelects;
    destination->shareGammas += source.shareGammas;
    destination->skipGammas += source.skipGammas;
}

class CoreProfileScope {
public:
    explicit CoreProfileScope(const QueryAccelerationCache* cache)
        : cache_(const_cast<QueryAccelerationCache*>(cache)),
          previous_(g_core_query_trace),
          enabled_(cache_ != nullptr && CoreProfilingEnabled()) {
        if (enabled_) {
            trace_.counters.queries = 1u;
            g_core_query_trace = &trace_;
        }
    }

    ~CoreProfileScope() {
        if (!enabled_) {
            return;
        }
        g_core_query_trace = previous_;
        AddCoreProfile(&cache_->coreProfile, trace_.counters);
    }

    void SetStage(CoreQueryStage stage) {
        if (enabled_) {
            trace_.stage = stage;
        }
    }

    void SourceTableHit() {
        if (enabled_) {
            ++trace_.counters.sourceTableHits;
        }
    }

    void SourceWalk() {
        if (enabled_) {
            ++trace_.counters.sourceWalks;
        }
    }

private:
    QueryAccelerationCache* cache_;
    CoreQueryTrace* previous_;
    bool enabled_;
    CoreQueryTrace trace_{};
};

bool BuildDeadlineExpired(const BuildDeadline& deadline) {
    return BuildClock::now() >= deadline;
}

struct FastBitReader {
    const unsigned int* data;
    std::uint64_t position;
};

struct SortedRankQuery {
    std::uint32_t index;
    std::uint32_t original_position;
};

__forceinline unsigned int PopCount32(unsigned int value) {
    return __popcnt(value);
}

__forceinline unsigned int PopCount64(unsigned __int64 value) {
    return static_cast<unsigned int>(__popcnt64(value));
}

__forceinline unsigned int CountTrailingZeros(unsigned int value) {
    unsigned long index = 0;
    _BitScanForward(&index, value);
    return static_cast<unsigned int>(index);
}

__forceinline unsigned int ReverseBits32(unsigned int value) {
    value = ((value >> 1) & 0x55555555u) | ((value & 0x55555555u) << 1);
    value = ((value >> 2) & 0x33333333u) | ((value & 0x33333333u) << 2);
    value = ((value >> 4) & 0x0F0F0F0Fu) | ((value & 0x0F0F0F0Fu) << 4);
    value = ((value >> 8) & 0x00FF00FFu) | ((value & 0x00FF00FFu) << 8);
    return (value >> 16) | (value << 16);
}

__forceinline unsigned int ReadBits(
    const unsigned int* data, std::uint64_t position, unsigned int bit_count) {
    if (bit_count == 0) {
        return 0;
    }

    const std::uint64_t word_index = position >> 5;
    const unsigned int offset = static_cast<unsigned int>(position & 31u);
    unsigned __int64 value = data[word_index];
    if (offset + bit_count > 32) {
        value |= static_cast<unsigned __int64>(data[word_index + 1]) << 32;
    }
    value >>= offset;

    if (bit_count == 32) {
        return static_cast<unsigned int>(value);
    }
    return static_cast<unsigned int>(value) & ((1u << bit_count) - 1u);
}

__forceinline unsigned int ReadReversedBits(
    const unsigned int* data, std::uint64_t position, unsigned int bit_count) {
    if (bit_count == 0) {
        return 0;
    }
    return ReverseBits32(ReadBits(data, position, bit_count)) >> (32u - bit_count);
}

__forceinline unsigned int ReadGamma(FastBitReader* reader) {
    unsigned int zero_count = 0;

    for (;;) {
        const std::uint64_t word_index = reader->position >> 5;
        const unsigned int offset = static_cast<unsigned int>(reader->position & 31u);
        const unsigned int word = reader->data[word_index] >> offset;
        if (word != 0) {
            const unsigned int trailing = CountTrailingZeros(word);
            zero_count += trailing;
            reader->position += trailing + 1u;
            break;
        }

        const unsigned int consumed = 32u - offset;
        zero_count += consumed;
        reader->position += consumed;
    }

    const unsigned int remainder =
        ReadReversedBits(reader->data, reader->position, zero_count);
    reader->position += zero_count;
    return (1u << zero_count) | remainder;
}

__forceinline std::uint64_t CountOnesExclusive(
    const unsigned int* data, std::uint64_t first, std::uint64_t last) {
    if (data == nullptr || last <= first) {
        return 0;
    }

    const std::uint64_t first_word = first >> 5;
    const std::uint64_t last_word = (last - 1u) >> 5;
    const unsigned int first_offset = static_cast<unsigned int>(first & 31u);
    const unsigned int last_offset = static_cast<unsigned int>((last - 1u) & 31u);
    if (first_word == last_word) {
        const unsigned int lower = ~0u << first_offset;
        const unsigned int upper = last_offset == 31u
            ? ~0u
            : (1u << (last_offset + 1u)) - 1u;
        return PopCount32(data[first_word] & lower & upper);
    }

    std::uint64_t count = PopCount32(data[first_word] & (~0u << first_offset));
    std::uint64_t word = first_word + 1u;
    for (; word + 1u < last_word; word += 2u) {
        const unsigned __int64 pair =
            static_cast<unsigned __int64>(data[word]) |
            (static_cast<unsigned __int64>(data[word + 1u]) << 32);
        count += PopCount64(pair);
    }
    if (word < last_word) {
        count += PopCount32(data[word]);
    }

    const unsigned int upper = last_offset == 31u
        ? ~0u
        : (1u << (last_offset + 1u)) - 1u;
    return count + PopCount32(data[last_word] & upper);
}

__forceinline std::uint64_t RankExclusive(
    const BitRankDir* dir, const unsigned int* data, std::uint64_t position) {
    if (dir == nullptr || dir->entryCount == 0) {
        return CountOnesExclusive(data, 0, position);
    }

    const std::uint64_t block = position >> dir->blockShift;
    const std::uint64_t block_start = block << dir->blockShift;
    const std::uint64_t prefix = dir->uses64
        ? dir->prefix64[block]
        : dir->prefix32[block];
    return prefix + CountOnesExclusive(data, block_start, position);
}

__forceinline AcceleratedBitRank GetBitAndRankCached(
    const BitArray* array,
    const BitRankDir* dir,
    unsigned int level,
    unsigned int position) {
    const unsigned int flag = array->Flag[level];
    if (flag == 0) {
        return AcceleratedBitRank{ 0u, 0u };
    }
    if (flag == 1) {
        return AcceleratedBitRank{ 1u, position + 1u };
    }

    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1u];
    const std::uint64_t absolute_position = start + position;
    const std::uint64_t rank = dir != nullptr &&
        dir->entryCount != 0 && dir->levelBaseRanks != nullptr &&
        level < dir->levelCount
        ? RankExclusive(dir, array->List, absolute_position + 1u) -
            dir->levelBaseRanks[level]
        : CountOnesExclusive(array->List, start, absolute_position + 1u);
    return AcceleratedBitRank{
        (array->List[absolute_position >> 5] >> (absolute_position & 31u)) & 1u,
        static_cast<unsigned int>(rank)
    };
}

__forceinline std::uint64_t RiceSampleAt(
    const RiceSelectDir* dir, std::uint64_t sample_index) {
    return dir->uses64
        ? dir->samples64[sample_index]
        : dir->samples32[sample_index];
}

__forceinline unsigned int SelectCheckpointForPosition(
    const RunGammaCkpt* dir, unsigned int level, unsigned int position) {
    std::uint64_t first = dir->levelOffsets[level];
    std::uint64_t last = dir->levelOffsets[level + 1u];
    while (first + 1u < last) {
        const std::uint64_t middle = first + (last - first) / 2u;
        if (dir->checkpoints[middle].position <= position) {
            first = middle;
        }
        else {
            last = middle;
        }
    }
    return static_cast<unsigned int>(first);
}

__forceinline unsigned int SelectCheckpointForRank(
    const RunGammaCkpt* dir, unsigned int level, unsigned int rank) {
    std::uint64_t first = dir->levelOffsets[level];
    std::uint64_t last = dir->levelOffsets[level + 1u];
    while (first + 1u < last) {
        const std::uint64_t middle = first + (last - first) / 2u;
        if (dir->checkpoints[middle].rank <= rank) {
            first = middle;
        }
        else {
            last = middle;
        }
    }
    return static_cast<unsigned int>(first);
}

__forceinline unsigned int BitAt(
    const BitArray* array, unsigned int level, unsigned int position) {
    const unsigned int flag = array->Flag[level];
    if (flag < 2u) {
        return flag;
    }

    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1];
    const std::uint64_t bit_position = start + position;
    return (array->List[bit_position >> 5] >> (bit_position & 31u)) & 1u;
}

__forceinline unsigned int SkipOnes(
    const unsigned int* data, std::uint64_t position, unsigned int count,
    std::uint64_t* result_position) {
    while (count != 0) {
        const std::uint64_t word_index = position >> 5;
        const unsigned int offset = static_cast<unsigned int>(position & 31u);
        unsigned int word = data[word_index] & (~0u << offset);
        const unsigned int ones = PopCount32(word);

        if (ones < count) {
            count -= ones;
            position = (word_index + 1u) << 5;
            continue;
        }

        while (count > 1u) {
            word &= word - 1u;
            --count;
        }
        *result_position = (word_index << 5) + CountTrailingZeros(word) + 1u;
        return 1;
    }

    *result_position = position;
    return 1;
}

__forceinline unsigned int RiceAccessFast(
    const RiceArray* array,
    const RiceSelectDir* dir,
    unsigned int level,
    unsigned int index) {
    const unsigned int parameter = array->RiceParameter[level];
    const std::uint64_t quotient_start =
        level == 0 ? 0 : array->quoIndex[level - 1];
    const std::uint64_t remainder_start =
        level == 0 ? 0 : array->remIndex[level - 1];

    std::uint64_t position = quotient_start;
    unsigned int preceding_ones = index;
    if (dir != nullptr && level < dir->levelCount && dir->stride != 0) {
        const std::uint64_t global_index = dir->levelBaseOnes[level] + index;
        if (global_index < dir->oneCount) {
            const std::uint64_t sample_index = global_index / dir->stride;
            if (sample_index < dir->sampleCount) {
                position = RiceSampleAt(dir, sample_index);
                preceding_ones = static_cast<unsigned int>(
                    global_index - sample_index * dir->stride);
            }
        }
    }
    SkipOnes(array->quotient, position, preceding_ones, &position);

    unsigned int quotient = 0;
    for (;;) {
        const std::uint64_t word_index = position >> 5;
        const unsigned int offset = static_cast<unsigned int>(position & 31u);
        const unsigned int word = array->quotient[word_index] >> offset;
        if (word != 0) {
            quotient += CountTrailingZeros(word);
            break;
        }
        const unsigned int consumed = 32u - offset;
        quotient += consumed;
        position += consumed;
    }

    const unsigned int remainder = parameter == 0
        ? 0
        : ReadBits(
            array->remainder,
            remainder_start + static_cast<std::uint64_t>(index) * parameter,
            parameter);
    return (quotient << parameter) | remainder;
}

__forceinline unsigned int SelectOneRunGammaFast(
    const RunBitArray* array,
    const RunGammaCkpt* dir,
    unsigned int level,
    unsigned int index) {
    if (array->Flag[level]) {
        return array->First[level] == 0 ? static_cast<unsigned int>(-1) : index;
    }

    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1];
    const std::uint64_t end = array->Index[level];
    FastBitReader reader{ array->List, start };
    unsigned int position = 0;
    unsigned int rank = 0;
    unsigned int current_bit = array->First[level];
    if (dir != nullptr && level < dir->levelCount &&
        dir->levelOneCounts != nullptr &&
        index >= dir->levelOneCounts[level]) {
        return static_cast<unsigned int>(-1);
    }
    if (dir != nullptr && level < dir->levelCount &&
        dir->levelOffsets[level] != dir->levelOffsets[level + 1u]) {
        const GammaCheckpoint& checkpoint = dir->checkpoints[
            SelectCheckpointForRank(dir, level, index)];
        reader.position = checkpoint.bitPosition;
        position = checkpoint.position;
        rank = checkpoint.rank;
        current_bit = checkpoint.currentBit;
    }

    while (reader.position < end) {
        const unsigned int run = ReadGamma(&reader);
        if (current_bit != 0) {
            if (rank + run > index) {
                return position + index - rank;
            }
            rank += run;
        }
        position += run;
        current_bit ^= 1u;
    }
    return static_cast<unsigned int>(-1);
}

__forceinline AcceleratedBitRank RunGammaBitAndRank(
    const RunBitArray* array,
    const RunGammaCkpt* dir,
    unsigned int level,
    unsigned int position) {
    if (array->Flag[level]) {
        const unsigned int bit = array->First[level];
        return AcceleratedBitRank{ bit, bit == 0 ? 0u : position + 1u };
    }

    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1];
    const std::uint64_t end = array->Index[level];
    FastBitReader reader{ array->List, start };
    unsigned int run_start = 0;
    unsigned int rank = 0;
    unsigned int current_bit = array->First[level];
    if (dir != nullptr && level < dir->levelCount &&
        dir->levelPositions != nullptr &&
        position >= dir->levelPositions[level]) {
        return AcceleratedBitRank{ 0u, dir->levelOneCounts[level] };
    }
    if (dir != nullptr && level < dir->levelCount &&
        dir->levelOffsets[level] != dir->levelOffsets[level + 1u]) {
        const GammaCheckpoint& checkpoint = dir->checkpoints[
            SelectCheckpointForPosition(dir, level, position)];
        reader.position = checkpoint.bitPosition;
        run_start = checkpoint.position;
        rank = checkpoint.rank;
        current_bit = checkpoint.currentBit;
    }

    while (reader.position < end) {
        const unsigned int run = ReadGamma(&reader);
        const unsigned int run_end = run_start + run;
        if (position < run_end) {
            if (current_bit != 0) {
                rank += position - run_start + 1u;
            }
            return AcceleratedBitRank{ current_bit, rank };
        }
        if (current_bit != 0) {
            rank += run;
        }
        run_start = run_end;
        current_bit ^= 1u;
    }
    return AcceleratedBitRank{ 0u, rank };
}

__forceinline unsigned int FixBitsAccessFast(
    const FixBitArray* array, unsigned int level, unsigned int index) {
    const unsigned int bit_count = array->BitsPerElement;
    if (bit_count == 0) {
        return 0;
    }
    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1];
    return ReadBits(
        array->List,
        (start + static_cast<std::uint64_t>(index)) * bit_count,
        bit_count);
}

__forceinline unsigned int UnpackOneCached(
    const Graph_Compress& graph,
    const QueryAccelerationCache* cache,
    unsigned int node_index,
    unsigned int* level) {
    const unsigned int current_level = *level;
    const unsigned int total_level = graph.nips_v * 2u + graph.nips_t;

    if (current_level == total_level - 1u) {
        *level = total_level;
        return BitAt(graph.NormalFlagList, current_level, node_index);
    }

    const BitRankDir* normal_dir = cache == nullptr ? nullptr : &cache->normalRank;
    if (g_core_query_trace != nullptr) {
        ++g_core_query_trace->counters.normalRanks;
    }
    const AcceleratedBitRank normal = GetBitAndRankCached(
        graph.NormalFlagList, normal_dir, current_level, node_index);
    if (normal.bit != 0) {
        *level = current_level + 1u;
        return normal.rank - 1u;
    }

    const unsigned int share_position = node_index - normal.rank;
    const BitRankDir* share_dir = cache == nullptr ? nullptr : &cache->shareRank;
    if (g_core_query_trace != nullptr) {
        ++g_core_query_trace->counters.shareRanks;
    }
    const AcceleratedBitRank share = GetBitAndRankCached(
        graph.ShareFlagList, share_dir, current_level, share_position);
    if (share.bit != 0) {
        if (g_core_query_trace != nullptr) {
            ++g_core_query_trace->counters.riceSelects;
            ++g_core_query_trace->counters.shareGammas;
        }
        const RiceSelectDir* rice_dir = cache == nullptr ? nullptr : &cache->riceSelect;
        const unsigned int share_value = RiceAccessFast(
            graph.ShareValue, rice_dir, current_level, share.rank - 1u);
        *level = current_level + 1u;
        const RunGammaCkpt* gamma_dir = cache == nullptr ? nullptr : &cache->shareNodeGamma;
        return SelectOneRunGammaFast(
            graph.ShareNodeList, gamma_dir, *level, share_value);
    }

    const unsigned int skip_position = share_position - share.rank;
    const RunGammaCkpt* skip_dir =
        cache == nullptr ? nullptr : &cache->skipFlagGamma;
    if (g_core_query_trace != nullptr) {
        ++g_core_query_trace->counters.skipGammas;
    }
    const AcceleratedBitRank skip = RunGammaBitAndRank(
        graph.SkipFlagList, skip_dir, current_level, skip_position);
    if (skip.bit == 0) {
        *level = total_level;
        return 0;
    }

    unsigned int skip_value = FixBitsAccessFast(
        graph.SkipValue, current_level, skip.rank - 1u);
    unsigned int result_level = current_level + 2u;
    while (result_level < total_level) {
        const unsigned int layer_size =
            graph.VariatyNumList[result_level] -
            graph.VariatyNumList[result_level - 1u];
        if (skip_value < layer_size) {
            break;
        }
        skip_value -= layer_size;
        ++result_level;
    }

    *level = result_level;
    return skip_value;
}

struct RunGammaForwardCursor {
    FastBitReader reader{};
    std::uint64_t end = 0u;
    unsigned int runStart = 0u;
    unsigned int runEnd = 0u;
    unsigned int rankBefore = 0u;
    unsigned int currentBit = 0u;
    bool hasRun = false;
};

__forceinline RunGammaForwardCursor MakeRunGammaForwardCursor(
    const RunBitArray* array, unsigned int level) {
    const std::uint64_t start = level == 0u
        ? 0u
        : array->Index[level - 1u];
    RunGammaForwardCursor cursor{};
    cursor.reader.data = array->List;
    cursor.reader.position = start;
    cursor.end = array->Index[level];
    cursor.currentBit = array->First[level];
    return cursor;
}

__forceinline AcceleratedBitRank RunGammaBitAndRankForward(
    const RunBitArray* array,
    const RunGammaCkpt* dir,
    unsigned int level,
    unsigned int position,
    RunGammaForwardCursor* cursor) {
    if (array->Flag[level]) {
        const unsigned int bit = array->First[level];
        return AcceleratedBitRank{ bit, bit == 0u ? 0u : position + 1u };
    }
    if (!cursor->hasRun && dir != nullptr && level < dir->levelCount &&
        dir->levelOffsets[level] != dir->levelOffsets[level + 1u]) {
        const GammaCheckpoint& checkpoint = dir->checkpoints[
            SelectCheckpointForPosition(dir, level, position)];
        cursor->reader.position = checkpoint.bitPosition;
        cursor->runStart = checkpoint.position;
        cursor->runEnd = checkpoint.position;
        cursor->rankBefore = checkpoint.rank;
        cursor->currentBit = checkpoint.currentBit;
    }
    while (!cursor->hasRun || position >= cursor->runEnd) {
        if (cursor->hasRun) {
            if (cursor->currentBit != 0u) {
                cursor->rankBefore += cursor->runEnd - cursor->runStart;
            }
            cursor->runStart = cursor->runEnd;
            cursor->currentBit ^= 1u;
        }
        if (cursor->reader.position >= cursor->end) {
            return AcceleratedBitRank{ 0u, cursor->rankBefore };
        }
        cursor->runEnd = cursor->runStart + ReadGamma(&cursor->reader);
        cursor->hasRun = true;
    }
    if (position < cursor->runStart) {
        return RunGammaBitAndRank(array, nullptr, level, position);
    }
    const unsigned int rank = cursor->currentBit == 0u
        ? cursor->rankBefore
        : cursor->rankBefore + position - cursor->runStart + 1u;
    return AcceleratedBitRank{ cursor->currentBit, rank };
}

__forceinline unsigned int UnpackOneBatch(
    const Graph_Compress& graph,
    const QueryAccelerationCache* cache,
    unsigned int node_index,
    unsigned int* level,
    RunGammaForwardCursor* skip_cursor) {
    const unsigned int current_level = *level;
    const unsigned int total_level =
        graph.nips_v * 2u + graph.nips_t;
    if (current_level == total_level - 1u) {
        *level = total_level;
        return BitAt(graph.NormalFlagList, current_level, node_index);
    }

    const AcceleratedBitRank normal = GetBitAndRankCached(
        graph.NormalFlagList,
        cache == nullptr ? nullptr : &cache->normalRank,
        current_level, node_index);
    if (normal.bit != 0u) {
        *level = current_level + 1u;
        return normal.rank - 1u;
    }

    const unsigned int share_position = node_index - normal.rank;
    const AcceleratedBitRank share = GetBitAndRankCached(
        graph.ShareFlagList,
        cache == nullptr ? nullptr : &cache->shareRank,
        current_level, share_position);
    if (share.bit != 0u) {
        const unsigned int share_value = RiceAccessFast(
            graph.ShareValue,
            cache == nullptr ? nullptr : &cache->riceSelect,
            current_level, share.rank - 1u);
        *level = current_level + 1u;
        return SelectOneRunGammaFast(
            graph.ShareNodeList,
            cache == nullptr ? nullptr : &cache->shareNodeGamma,
            *level, share_value);
    }

    const unsigned int skip_position =
        share_position - share.rank;
    const AcceleratedBitRank skip = RunGammaBitAndRankForward(
        graph.SkipFlagList,
        cache == nullptr ? nullptr : &cache->skipFlagGamma,
        current_level,
        skip_position, skip_cursor);
    if (skip.bit == 0u) {
        *level = total_level;
        return 0u;
    }

    unsigned int skip_value = FixBitsAccessFast(
        graph.SkipValue, current_level, skip.rank - 1u);
    unsigned int result_level = current_level + 2u;
    while (result_level < total_level) {
        const unsigned int layer_size =
            graph.VariatyNumList[result_level] -
            graph.VariatyNumList[result_level - 1u];
        if (skip_value < layer_size) {
            break;
        }
        skip_value -= layer_size;
        ++result_level;
    }
    *level = result_level;
    return skip_value;
}

__forceinline std::size_t BranchChildHash(
    unsigned int node_id, unsigned int node_level, unsigned int branch) {
    std::uint64_t value =
        (static_cast<std::uint64_t>(node_level) << 33) |
        (static_cast<std::uint64_t>(node_id) << 1) |
        (branch & 1u);
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    return static_cast<std::size_t>(value ^ (value >> 31));
}

__forceinline bool TryBranchChild(
    const QueryAccelerationCache* cache,
    unsigned int node_id,
    unsigned int node_level,
    unsigned int branch,
    unsigned int* child_id,
    unsigned int* child_level) {
    if (cache == nullptr || cache->branchChildren.entries == nullptr ||
        node_level > UINT8_MAX) {
        return false;
    }
    const BranchChildEntry& entry = cache->branchChildren.entries[
        BranchChildHash(node_id, node_level, branch) &
        (kBranchChildCapacity - 1u)];
    if (entry.valid == 0u || entry.nodeId != node_id ||
        entry.nodeLevel != node_level || entry.branch != (branch & 1u)) {
        return false;
    }
    *child_id = entry.childId;
    *child_level = entry.childLevel;
    return true;
}

__forceinline void StoreBranchChild(
    const QueryAccelerationCache* cache,
    unsigned int node_id,
    unsigned int node_level,
    unsigned int branch,
    unsigned int child_id,
    unsigned int child_level) {
    if (cache == nullptr || cache->branchChildren.entries == nullptr ||
        node_level > UINT8_MAX || child_level > UINT8_MAX) {
        return;
    }
    BranchChildEntry& entry = cache->branchChildren.entries[
        BranchChildHash(node_id, node_level, branch) &
        (kBranchChildCapacity - 1u)];
    entry.nodeId = node_id;
    entry.childId = child_id;
    entry.nodeLevel = static_cast<std::uint8_t>(node_level);
    entry.childLevel = static_cast<std::uint8_t>(child_level);
    entry.branch = static_cast<std::uint8_t>(branch & 1u);
    entry.valid = 1u;
    entry.padding = 0u;
}

__forceinline unsigned int UnpackBranchCached(
    const Graph_Compress& graph,
    const QueryAccelerationCache* cache,
    unsigned int node_id,
    unsigned int branch,
    unsigned int* level,
    bool* cache_hit = nullptr) {
    const unsigned int node_level = *level;
    if (g_core_query_trace != nullptr) {
        switch (g_core_query_trace->stage) {
        case CoreQueryStage::Source:
            ++g_core_query_trace->counters.sourceSteps;
            break;
        case CoreQueryStage::Target:
            ++g_core_query_trace->counters.targetSteps;
            break;
        case CoreQueryStage::Time:
            ++g_core_query_trace->counters.timeSteps;
            break;
        default:
            break;
        }
    }
    unsigned int child_id = 0u;
    unsigned int child_level = 0u;
    if (TryBranchChild(
        cache, node_id, node_level, branch,
        &child_id, &child_level)) {
        *level = child_level;
        if (cache_hit != nullptr) {
            *cache_hit = true;
        }
        if (g_core_query_trace != nullptr) {
            ++g_core_query_trace->counters.branchHits;
        }
        return child_id;
    }

    if (cache_hit != nullptr) {
        *cache_hit = false;
    }
    if (g_core_query_trace != nullptr) {
        ++g_core_query_trace->counters.branchMisses;
    }
    const unsigned int child = UnpackOneCached(
        graph, cache, (node_id << 1) | (branch & 1u), level);
    StoreBranchChild(
        cache, node_id, node_level, branch, child, *level);
    return child;
}

__forceinline unsigned int WalkSegment(
    const Graph_Compress& graph,
    const QueryAccelerationCache* cache,
    unsigned int node,
    unsigned int* level,
    unsigned int path,
    unsigned int segment_start,
    unsigned int segment_length,
    unsigned int total_level) {
    const unsigned int segment_end = segment_start + segment_length;
    while (*level < segment_end) {
        const unsigned int bit_index = *level - segment_start;
        const unsigned int bit =
            (path >> (segment_length - bit_index - 1u)) & 1u;
        node = UnpackBranchCached(graph, cache, node, bit, level);
        if (*level == total_level) {
            break;
        }
    }
    return node;
}

unsigned int ReferenceBitPopCount(
    const unsigned int* data, std::uint64_t first, std::uint64_t last) {
    if (data == nullptr || last < first) {
        return 0;
    }
    unsigned int count = 0;
    for (std::uint64_t position = first;; ++position) {
        count += (data[position >> 5] >> (position & 31u)) & 1u;
        if (position == last) {
            break;
        }
    }
    return count;
}

AcceleratedBitRank ReferenceBitAndRank(
    const BitArray* array, unsigned int level, unsigned int position) {
    const unsigned int flag = array->Flag[level];
    if (flag == 0) {
        return AcceleratedBitRank{ 0u, 0u };
    }
    if (flag == 1) {
        return AcceleratedBitRank{ 1u, position + 1u };
    }
    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1];
    const std::uint64_t absolute_position = start + position;
    return AcceleratedBitRank{
        (array->List[absolute_position >> 5] >> (absolute_position & 31u)) & 1u,
        ReferenceBitPopCount(array->List, start, absolute_position)
    };
}

unsigned int ReferenceRiceAccess(
    const RiceArray* array, unsigned int level, unsigned int index) {
    const unsigned int parameter = array->RiceParameter[level];
    std::uint64_t position = level == 0 ? 0 : array->quoIndex[level - 1];
    const std::uint64_t remainder_start =
        level == 0 ? 0 : array->remIndex[level - 1];

    for (unsigned int current = 0; current < index; ++current) {
        while (((array->quotient[position >> 5] >> (position & 31u)) & 1u) == 0) {
            ++position;
        }
        ++position;
    }

    unsigned int quotient = 0;
    while (((array->quotient[position >> 5] >> (position & 31u)) & 1u) == 0) {
        ++quotient;
        ++position;
    }

    const unsigned int remainder = parameter == 0
        ? 0
        : ReadBits(
            array->remainder,
            remainder_start + static_cast<std::uint64_t>(index) * parameter,
            parameter);
    return (quotient << parameter) | remainder;
}

unsigned int ReferenceSelectOneRunGamma(
    const RunBitArray* array, unsigned int level, unsigned int index) {
    if (array->Flag[level]) {
        return array->First[level] == 0 ? static_cast<unsigned int>(-1) : index;
    }
    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1];
    const std::uint64_t end = array->Index[level];
    BitReader reader;
    br_init(&reader, array->List, start);
    unsigned int position = 0;
    unsigned int rank = 0;
    unsigned int current_bit = array->First[level];
    while (reader.bitpos < end) {
        const unsigned int run = br_read_gamma(&reader);
        if (current_bit != 0) {
            if (rank + run > index) {
                return position + index - rank;
            }
            rank += run;
        }
        position += run;
        current_bit ^= 1u;
    }
    return static_cast<unsigned int>(-1);
}

AcceleratedBitRank ReferenceRunGammaBitAndRank(
    const RunBitArray* array, unsigned int level, unsigned int position) {
    if (array->Flag[level]) {
        const unsigned int bit = array->First[level];
        return AcceleratedBitRank{ bit, bit == 0 ? 0u : position + 1u };
    }
    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1];
    const std::uint64_t end = array->Index[level];
    BitReader reader;
    br_init(&reader, array->List, start);
    unsigned int run_start = 0;
    unsigned int rank = 0;
    unsigned int current_bit = array->First[level];
    while (reader.bitpos < end) {
        const unsigned int run = br_read_gamma(&reader);
        const unsigned int run_end = run_start + run;
        if (position < run_end) {
            if (current_bit != 0) {
                rank += position - run_start + 1u;
            }
            return AcceleratedBitRank{ current_bit, rank };
        }
        if (current_bit != 0) {
            rank += run;
        }
        run_start = run_end;
        current_bit ^= 1u;
    }
    return AcceleratedBitRank{ 0u, rank };
}

unsigned int ReferenceFixBitsAccess(
    const FixBitArray* array, unsigned int level, unsigned int index) {
    const unsigned int bit_count = array->BitsPerElement;
    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1];
    const std::uint64_t bit_position =
        (start + static_cast<std::uint64_t>(index)) * bit_count;
    unsigned int value = 0;
    for (unsigned int bit = 0; bit < bit_count; ++bit) {
        const std::uint64_t position = bit_position + bit;
        value |= ((array->List[position >> 5] >> (position & 31u)) & 1u) << bit;
    }
    return value;
}

unsigned int ReferenceWalkSegment(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int* level,
    unsigned int path,
    unsigned int segment_start,
    unsigned int segment_length,
    unsigned int total_level) {
    const unsigned int segment_end = segment_start + segment_length;
    while (*level < segment_end) {
        const unsigned int bit_index = *level - segment_start;
        const unsigned int bit =
            (path >> (segment_length - bit_index - 1u)) & 1u;
        node = ReferenceUnpackOne(graph, (node << 1) | bit, level);
        if (*level == total_level) {
            break;
        }
    }
    return node;
}

bool AllAssignmentParity(
    const Graph_Compress& graph,
    const QueryAccelerationCache* cache,
    unsigned int node,
    unsigned int node_level,
    unsigned int domain_level,
    unsigned int total_level) {
    if (domain_level == total_level) {
        return node != 0;
    }



    if (node_level > domain_level) {
        return false;
    }

    constexpr std::size_t parity_capacity =
        kParityCacheBudgetBytes / sizeof(ParityCacheEntry);
    ParityCacheEntry* cached_entry = nullptr;
    if (cache != nullptr && cache->parity.entries != nullptr &&
        node_level == domain_level) {
        std::uint64_t key =
            (static_cast<std::uint64_t>(node_level) << 32) | node;
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdull;
        key ^= key >> 33;
        const std::size_t slot = static_cast<std::size_t>(key) &
            (parity_capacity - 1u);
        cached_entry = const_cast<ParityCacheEntry*>(
            &cache->parity.entries[slot]);
        if (cached_entry->valid != 0u &&
            cached_entry->nodeId == node &&
            cached_entry->nodeLevel == node_level) {
            return cached_entry->value != 0u;
        }
    }

    unsigned int low_level = node_level;
    unsigned int high_level = node_level;
    const unsigned int low = UnpackBranchCached(
        graph, cache, node, 0u, &low_level);
    const unsigned int high = UnpackBranchCached(
        graph, cache, node, 1u, &high_level);
    const bool result = AllAssignmentParity(
        graph, cache, low, low_level, domain_level + 1u, total_level) ^
        AllAssignmentParity(
            graph, cache, high, high_level, domain_level + 1u, total_level);
    if (cached_entry != nullptr) {
        cached_entry->nodeId = node;
        cached_entry->nodeLevel = static_cast<std::uint8_t>(node_level);
        cached_entry->value = result ? 1u : 0u;
        cached_entry->valid = 1u;
    }
    return result;
}

bool PrefixAssignmentParity(
    const Graph_Compress& graph,
    const QueryAccelerationCache* cache,
    unsigned int node,
    unsigned int node_level,
    unsigned int domain_level,
    unsigned int time_start_level,
    unsigned int total_level,
    unsigned int time) {
    if (domain_level == total_level) {
        return node != 0;
    }

    const unsigned int bit_index = domain_level - time_start_level;
    const unsigned int bit =
        (time >> (graph.nips_t - bit_index - 1u)) & 1u;

    if (node_level > domain_level) {
        if (bit == 0) {
            return PrefixAssignmentParity(
                graph,
                cache,
                node,
                node_level,
                domain_level + 1u,
                time_start_level,
                total_level,
                time);
        }
        return AllAssignmentParity(
            graph, cache, node, node_level, domain_level + 1u, total_level) ^
            PrefixAssignmentParity(
                graph,
                cache,
                node,
                node_level,
                domain_level + 1u,
                time_start_level,
                total_level,
                time);
    }

    unsigned int low_level = node_level;
    const unsigned int low = UnpackBranchCached(
        graph, cache, node, 0u, &low_level);
    if (bit == 0) {
        return PrefixAssignmentParity(
            graph,
            cache,
            low,
            low_level,
            domain_level + 1u,
            time_start_level,
            total_level,
            time);
    }

    unsigned int high_level = node_level;
    const unsigned int high = UnpackBranchCached(
        graph, cache, node, 1u, &high_level);
    return AllAssignmentParity(
        graph, cache, low, low_level, domain_level + 1u, total_level) ^
        PrefixAssignmentParity(
            graph,
            cache,
            high,
            high_level,
            domain_level + 1u,
            time_start_level,
            total_level,
            time);
}

bool PrefixAssignmentAny(
    const Graph_Compress& graph,
    const QueryAccelerationCache* cache,
    unsigned int node,
    unsigned int node_level,
    unsigned int domain_level,
    unsigned int time_start_level,
    unsigned int total_level,
    unsigned int time) {
    if (domain_level == total_level || node_level == total_level) {
        return node != 0u;
    }
    if (node_level < domain_level || node_level > total_level) {
        return false;
    }

    const unsigned int bit_index = domain_level - time_start_level;
    const unsigned int bit =
        (time >> (graph.nips_t - bit_index - 1u)) & 1u;
    if (node_level > domain_level) {
        if (bit != 0u) {
            return node_level != total_level || node != 0u;
        }
        return PrefixAssignmentAny(
            graph,
            cache,
            node,
            node_level,
            domain_level + 1u,
            time_start_level,
            total_level,
            time);
    }

    unsigned int low_level = node_level;
    const unsigned int low = UnpackBranchCached(
        graph, cache, node, 0u, &low_level);
    if (bit == 0u) {
        return PrefixAssignmentAny(
            graph,
            cache,
            low,
            low_level,
            domain_level + 1u,
            time_start_level,
            total_level,
            time);
    }
    if (low_level != total_level || low != 0u) {
        return true;
    }

    unsigned int high_level = node_level;
    const unsigned int high = UnpackBranchCached(
        graph, cache, node, 1u, &high_level);
    return PrefixAssignmentAny(
        graph,
        cache,
        high,
        high_level,
        domain_level + 1u,
        time_start_level,
        total_level,
        time);
}

__forceinline unsigned int QueryLevelCount(const Graph_Compress& graph) {
    return graph.nips_v * 2u + graph.nips_t;
}

std::size_t EstimateBitRankDirBytes(
    std::uint64_t bit_count,
    unsigned int level_count,
    unsigned int block_shift,
    bool uses64) {
    const std::uint64_t entry_count = (bit_count >> block_shift) + 1u;
    return static_cast<std::size_t>(entry_count) *
        (uses64 ? sizeof(std::uint64_t) : sizeof(std::uint32_t)) +
        static_cast<std::size_t>(level_count) * sizeof(std::uint64_t);
}

bool BuildBitRankDir(
    const BitArray* array,
    unsigned int level_count,
    unsigned int block_shift,
    bool uses64,
    const BuildDeadline& deadline,
    BitRankDir* dir) {
    dir->blockShift = block_shift;
    dir->levelCount = level_count;
    dir->entryCount = (array->Count >> block_shift) + 1u;
    dir->uses64 = uses64;
    dir->prefix32.reset();
    dir->prefix64.reset();
    dir->levelBaseRanks.reset(new (std::nothrow) std::uint64_t[
        static_cast<std::size_t>(level_count)]);
    if (dir->levelBaseRanks == nullptr) {
        return false;
    }

    if (dir->entryCount == 0) {
        return true;
    }

    std::uint64_t cumulative = 0;
    if (uses64) {
        dir->prefix64.reset(new (std::nothrow) std::uint64_t[
            static_cast<std::size_t>(dir->entryCount)]);
        if (dir->prefix64 == nullptr) {
            return false;
        }
        dir->prefix64[0] = 0;
        for (std::uint64_t block = 1; block < dir->entryCount; ++block) {
            const std::uint64_t block_start = (block - 1u) << block_shift;
            const std::uint64_t block_end = (std::min)(
                block << block_shift, array->Count);
            cumulative += CountOnesExclusive(array->List, block_start, block_end);
            dir->prefix64[block] = cumulative;
            if ((block & kBuildCheckMask) == 0u && BuildDeadlineExpired(deadline)) {
                return false;
            }
        }
    }
    else {
        dir->prefix32.reset(new (std::nothrow) std::uint32_t[
            static_cast<std::size_t>(dir->entryCount)]);
        if (dir->prefix32 == nullptr) {
            return false;
        }
        dir->prefix32[0] = 0;
        for (std::uint64_t block = 1; block < dir->entryCount; ++block) {
            const std::uint64_t block_start = (block - 1u) << block_shift;
            const std::uint64_t block_end = (std::min)(
                block << block_shift, array->Count);
            cumulative += CountOnesExclusive(array->List, block_start, block_end);
            dir->prefix32[block] = static_cast<std::uint32_t>(cumulative);
            if ((block & kBuildCheckMask) == 0u && BuildDeadlineExpired(deadline)) {
                return false;
            }
        }
    }

    for (unsigned int level = 0; level < level_count; ++level) {
        const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1u];
        dir->levelBaseRanks[level] = RankExclusive(dir, array->List, start);
    }
    return !BuildDeadlineExpired(deadline);
}

bool CountOnesWithDeadline(
    const unsigned int* data,
    std::uint64_t first,
    std::uint64_t last,
    const BuildDeadline& deadline,
    std::uint64_t* result) {
    constexpr std::uint64_t kChunkBits = 1ull << 20;
    std::uint64_t count = 0;
    while (first < last) {
        const std::uint64_t chunk_end = (std::min)(first + kChunkBits, last);
        count += CountOnesExclusive(data, first, chunk_end);
        first = chunk_end;
        if (BuildDeadlineExpired(deadline)) {
            return false;
        }
    }
    *result = count;
    return true;
}

std::size_t EstimateRiceSelectBytes(
    std::uint64_t one_count,
    unsigned int level_count,
    unsigned int stride,
    bool uses64) {
    const std::uint64_t sample_count = one_count == 0
        ? 1u
        : ((one_count - 1u) / stride) + 1u;
    return static_cast<std::size_t>(sample_count) *
        (uses64 ? sizeof(std::uint64_t) : sizeof(std::uint32_t)) +
        static_cast<std::size_t>(level_count) * sizeof(std::uint64_t);
}

bool BuildRiceSelectDir(
    const RiceArray* array,
    unsigned int level_count,
    unsigned int stride,
    std::uint64_t one_count,
    const BuildDeadline& deadline,
    RiceSelectDir* dir) {
    dir->stride = stride;
    dir->levelCount = level_count;
    dir->oneCount = one_count;
    dir->uses64 = array->quoCount > UINT32_MAX;
    dir->sampleCount = one_count == 0
        ? 1u
        : ((one_count - 1u) / stride) + 1u;
    if (dir->uses64) {
        dir->samples64.reset(new (std::nothrow) std::uint64_t[
            static_cast<std::size_t>(dir->sampleCount)]);
        if (dir->samples64 == nullptr) {
            return false;
        }
    }
    else {
        dir->samples32.reset(new (std::nothrow) std::uint32_t[
            static_cast<std::size_t>(dir->sampleCount)]);
        if (dir->samples32 == nullptr) {
            return false;
        }
    }
    dir->levelBaseOnes.reset(new (std::nothrow) std::uint64_t[
        static_cast<std::size_t>(level_count)]);
    if (dir->levelBaseOnes == nullptr) {
        return false;
    }

    std::uint64_t cumulative_ones = 0;
    std::uint64_t previous_end = 0;
    for (unsigned int level = 0; level < level_count; ++level) {
        dir->levelBaseOnes[level] = cumulative_ones;
        const std::uint64_t level_end = array->quoIndex[level];
        if (level_end > previous_end) {
            std::uint64_t level_ones = 0;
            if (!CountOnesWithDeadline(
                array->quotient, previous_end, level_end, deadline, &level_ones)) {
                return false;
            }
            cumulative_ones += level_ones;
            previous_end = level_end;
        }
    }

    if (dir->uses64) {
        dir->samples64[0] = 0;
    }
    else {
        dir->samples32[0] = 0;
    }
    std::uint64_t position = 0;
    for (std::uint64_t sample = 1; sample < dir->sampleCount; ++sample) {
        SkipOnes(array->quotient, position, stride, &position);
        if (dir->uses64) {
            dir->samples64[sample] = position;
        }
        else {
            dir->samples32[sample] = static_cast<std::uint32_t>(position);
        }
        if ((sample & kBuildCheckMask) == 0u && BuildDeadlineExpired(deadline)) {
            return false;
        }
    }
    return !BuildDeadlineExpired(deadline);
}

bool ScanRunGammaLevels(
    const RunBitArray* array,
    unsigned int level_count,
    std::vector<std::uint32_t>* run_counts,
    std::vector<std::uint32_t>* level_positions,
    std::vector<std::uint32_t>* level_one_counts,
    const BuildDeadline& deadline) {
    run_counts->assign(level_count, 0u);
    level_positions->assign(level_count, 0u);
    level_one_counts->assign(level_count, 0u);

    for (unsigned int level = 0; level < level_count; ++level) {
        if (array->Flag[level] != 0u) {
            continue;
        }

        const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1u];
        const std::uint64_t end = array->Index[level];
        FastBitReader reader{ array->List, start };
        std::uint32_t run_count = 0;
        std::uint32_t position = 0;
        std::uint32_t one_count = 0;
        unsigned int current_bit = array->First[level];
        while (reader.position < end) {
            const unsigned int run = ReadGamma(&reader);
            ++run_count;
            position += run;
            if (current_bit != 0) {
                one_count += run;
            }
            current_bit ^= 1u;
            if ((run_count & kBuildCheckMask) == 0u &&
                BuildDeadlineExpired(deadline)) {
                return false;
            }
        }
        (*run_counts)[level] = run_count;
        (*level_positions)[level] = position;
        (*level_one_counts)[level] = one_count;
    }
    return !BuildDeadlineExpired(deadline);
}

std::size_t EstimateGammaBytes(
    const std::vector<std::uint32_t>& run_counts,
    unsigned int level_count,
    unsigned int stride) {
    std::uint64_t checkpoint_count = 0;
    for (unsigned int level = 0; level < level_count; ++level) {
        const std::uint32_t runs = run_counts[level];
        if (runs != 0u) {
            checkpoint_count += 1u + (runs - 1u) / stride;
        }
    }
    return static_cast<std::size_t>(checkpoint_count) * sizeof(GammaCheckpoint) +
        static_cast<std::size_t>(level_count + 1u) * sizeof(std::uint64_t) +
        static_cast<std::size_t>(level_count) * sizeof(std::uint32_t) * 2u;
}

bool BuildRunGammaCkpt(
    const RunBitArray* array,
    unsigned int level_count,
    unsigned int stride,
    const std::vector<std::uint32_t>& run_counts,
    const std::vector<std::uint32_t>& level_positions,
    const std::vector<std::uint32_t>& level_one_counts,
    const BuildDeadline& deadline,
    RunGammaCkpt* dir) {
    std::uint64_t checkpoint_count = 0;
    for (unsigned int level = 0; level < level_count; ++level) {
        const std::uint32_t runs = run_counts[level];
        if (runs != 0u) {
            checkpoint_count += 1u + (runs - 1u) / stride;
        }
    }

    dir->stride = stride;
    dir->levelCount = level_count;
    dir->checkpointCount = checkpoint_count;
    dir->levelOffsets.reset(new (std::nothrow) std::uint64_t[
        static_cast<std::size_t>(level_count + 1u)]);
    dir->levelPositions.reset(new (std::nothrow) std::uint32_t[
        static_cast<std::size_t>(level_count)]);
    dir->levelOneCounts.reset(new (std::nothrow) std::uint32_t[
        static_cast<std::size_t>(level_count)]);
    if (dir->levelOffsets == nullptr || dir->levelPositions == nullptr ||
        dir->levelOneCounts == nullptr) {
        return false;
    }
    if (checkpoint_count != 0) {
        dir->checkpoints.reset(new (std::nothrow) GammaCheckpoint[
            static_cast<std::size_t>(checkpoint_count)]);
        if (dir->checkpoints == nullptr) {
            return false;
        }
    }
    else {
        dir->checkpoints.reset();
    }

    std::memcpy(
        dir->levelPositions.get(),
        level_positions.data(),
        sizeof(std::uint32_t) * static_cast<std::size_t>(level_count));
    std::memcpy(
        dir->levelOneCounts.get(),
        level_one_counts.data(),
        sizeof(std::uint32_t) * static_cast<std::size_t>(level_count));

    std::uint64_t checkpoint_index = 0;
    for (unsigned int level = 0; level < level_count; ++level) {
        dir->levelOffsets[level] = checkpoint_index;
        if (array->Flag[level] != 0u) {
            continue;
        }

        const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1u];
        const std::uint64_t end = array->Index[level];
        FastBitReader reader{ array->List, start };
        std::uint32_t position = 0;
        std::uint32_t rank = 0;
        unsigned int current_bit = array->First[level];
        std::uint32_t run_index = 0;
        while (reader.position < end) {
            if ((run_index % stride) == 0u) {
                GammaCheckpoint& checkpoint =
                    dir->checkpoints[static_cast<std::size_t>(checkpoint_index++)];
                checkpoint.bitPosition = static_cast<std::uint32_t>(reader.position);
                checkpoint.position = position;
                checkpoint.rank = rank;
                checkpoint.currentBit = static_cast<std::uint8_t>(current_bit);
                std::memset(checkpoint.padding, 0, sizeof(checkpoint.padding));
            }

            const unsigned int run = ReadGamma(&reader);
            position += run;
            if (current_bit != 0) {
                rank += run;
            }
            current_bit ^= 1u;
            ++run_index;
            if ((run_index & kBuildCheckMask) == 0u &&
                BuildDeadlineExpired(deadline)) {
                return false;
            }
        }
    }
    dir->levelOffsets[level_count] = checkpoint_index;
    return checkpoint_index == checkpoint_count && !BuildDeadlineExpired(deadline);
}

constexpr std::size_t kNeighborMemoCapacity =
    kNeighborMemoBudgetBytes / sizeof(NeighborMemoEntry);
constexpr std::size_t kNeighborMemoWays = 4u;
constexpr std::size_t kNeighborMemoSetCount =
    kNeighborMemoCapacity / kNeighborMemoWays;
constexpr std::size_t kNeighborStackCapacity =
    kNeighborStackBudgetBytes / sizeof(NeighborDfsState);
static_assert((kNeighborMemoCapacity & (kNeighborMemoCapacity - 1u)) == 0u,
    "Neighbor memo capacity must be a power of two");
static_assert((kNeighborMemoSetCount & (kNeighborMemoSetCount - 1u)) == 0u,
    "Neighbor memo set count must be a power of two");

struct NeighborQueryContext {
    const Graph_Compress& graph;
    QueryAccelerationCache* cache;
    NeighborAccelerationState* state;
    unsigned int time;
    DataType dataset;
    unsigned int yStart;
    unsigned int timeStart;
    unsigned int totalLevel;
    bool profileEnabled;
    bool ok;
};

struct NeighborChildPair {
    unsigned int lowId;
    unsigned int lowLevel;
    unsigned int highId;
    unsigned int highLevel;
};

__forceinline std::uint64_t MixNeighborKey(std::uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

__forceinline std::size_t NeighborNodeHash(
    unsigned int node_id, unsigned int node_level) {
    return static_cast<std::size_t>(MixNeighborKey(
        (static_cast<std::uint64_t>(node_level) << 32) | node_id));
}

bool EnsureNeighborAcceleration(QueryAccelerationCache* cache) {
    NeighborAccelerationState* state = &cache->neighbor;
    if (state->attempted) {
        return state->enabled;
    }
    state->attempted = true;

    const auto start = BuildClock::now();
    const BuildDeadline deadline = start + std::chrono::seconds(30);
    const std::size_t required =
        kNeighborMemoBudgetBytes + kNeighborStackBudgetBytes;
    if (required > kResidentCacheBudgetBytes ||
        cache->bytes > kResidentCacheBudgetBytes - required) {
        return false;
    }

    std::unique_ptr<NeighborMemoEntry[]> memo(
        new (std::nothrow) NeighborMemoEntry[kNeighborMemoCapacity]());
    if (memo == nullptr || BuildDeadlineExpired(deadline)) {
        return false;
    }
    std::unique_ptr<NeighborDfsState[]> stack(
        new (std::nothrow) NeighborDfsState[kNeighborStackCapacity]);
    if (stack == nullptr || BuildDeadlineExpired(deadline)) {
        return false;
    }
    state->memo = std::move(memo);
    state->memoCapacity = kNeighborMemoCapacity;
    state->stack = std::move(stack);
    state->bytes = required;
    state->enabled = true;
    cache->bytes += required;
    return true;
}

void BeginNeighborMemo(NeighborAccelerationState* state) {
    ++state->generation;
    if (state->generation == 0u) {
        std::memset(
            state->memo.get(), 0,
            state->memoCapacity * sizeof(NeighborMemoEntry));
        state->generation = 1u;
    }
    state->memoCount = 0u;
}

bool FindNeighborMemo(
    NeighborQueryContext* context,
    unsigned int node_id,
    unsigned int node_level,
    bool* found,
    bool* active) {
    const std::size_t memo_ways = kNeighborMemoWays;
    const std::size_t set_count =
        context->state->memoCapacity / memo_ways;
    if (set_count == 0u ||
        (set_count & (set_count - 1u)) != 0u) {
        return false;
    }
    const std::size_t set = NeighborNodeHash(node_id, node_level) &
        (set_count - 1u);
    const std::size_t first = set * memo_ways;
    for (std::size_t way = 0u; way < memo_ways; ++way) {
        const NeighborMemoEntry& entry = context->state->memo[first + way];
        if (entry.generation == context->state->generation &&
            entry.nodeId == node_id && entry.nodeLevel == node_level) {
            *found = true;
            *active = entry.state == 2u;
            if (context->profileEnabled) {
                ++context->state->profile.memoHits;
            }
            return true;
        }
    }
    *found = false;
    return true;
}

bool StoreNeighborMemo(
    NeighborQueryContext* context,
    unsigned int node_id,
    unsigned int node_level,
    bool active) {
    const std::size_t memo_ways = kNeighborMemoWays;
    const std::size_t set_count =
        context->state->memoCapacity / memo_ways;
    if (set_count == 0u ||
        (set_count & (set_count - 1u)) != 0u) {
        return false;
    }
    const std::uint64_t hash = MixNeighborKey(
        (static_cast<std::uint64_t>(node_level) << 32) | node_id);
    const std::size_t set = static_cast<std::size_t>(hash) &
        (set_count - 1u);
    const std::size_t first = set * memo_ways;
    NeighborMemoEntry* destination = nullptr;
    for (std::size_t way = 0u; way < memo_ways; ++way) {
        NeighborMemoEntry& entry = context->state->memo[first + way];
        if (entry.generation == context->state->generation &&
            entry.nodeId == node_id && entry.nodeLevel == node_level) {
            destination = &entry;
            break;
        }
        if (destination == nullptr &&
            entry.generation != context->state->generation) {
            destination = &entry;
        }
    }
    if (destination == nullptr) {
        destination = &context->state->memo[
            first + ((hash >> 32) & (memo_ways - 1u))];
    }
    if (destination->generation != context->state->generation &&
        context->state->memoCount != UINT32_MAX) {
        ++context->state->memoCount;
    }
    destination->nodeId = node_id;
    destination->nodeLevel = static_cast<std::uint8_t>(node_level);
    destination->state = active ? 2u : 1u;
    destination->generation = context->state->generation;
    if (context->profileEnabled) {
        ++context->state->profile.memoMisses;
    }
    return true;
}

bool ValidNeighborChild(
    const NeighborQueryContext& context,
    unsigned int parent_level,
    unsigned int node_id,
    unsigned int node_level) {
    if (node_level <= parent_level || node_level > context.totalLevel) {
        return false;
    }
    if (node_level == context.totalLevel) {
        return true;
    }
    const unsigned int base = node_level == 0u
        ? 0u
        : context.graph.VariatyNumList[node_level - 1u];
    const unsigned int end = context.graph.VariatyNumList[node_level];
    return end >= base && node_id < end - base;
}

bool GetNeighborChildren(
    NeighborQueryContext* context,
    unsigned int node_id,
    unsigned int node_level,
    NeighborChildPair* pair) {
    if (node_level >= context->totalLevel || node_id > UINT32_MAX / 2u) {
        context->ok = false;
        return false;
    }

    unsigned int low_level = node_level;
    unsigned int high_level = node_level;
    bool low_hit = false;
    bool high_hit = false;
    const unsigned int low_id = UnpackBranchCached(
        context->graph, context->cache, node_id, 0u,
        &low_level, &low_hit);
    const unsigned int high_id = UnpackBranchCached(
        context->graph, context->cache, node_id, 1u,
        &high_level, &high_hit);
    if (!ValidNeighborChild(
        *context, node_level, low_id, low_level) ||
        !ValidNeighborChild(
            *context, node_level, high_id, high_level)) {
        context->ok = false;
        return false;
    }

    pair->lowId = low_id;
    pair->lowLevel = low_level;
    pair->highId = high_id;
    pair->highLevel = high_level;
    if (context->profileEnabled) {
        context->state->profile.childHits +=
            static_cast<std::uint64_t>(low_hit) +
            static_cast<std::uint64_t>(high_hit);
        context->state->profile.childMisses +=
            static_cast<std::uint64_t>(!low_hit) +
            static_cast<std::uint64_t>(!high_hit);
    }
    return true;
}

bool PointTimeActiveCached(
    NeighborQueryContext* context,
    unsigned int node_id,
    unsigned int node_level,
    bool* active) {
    unsigned int domain_level = context->timeStart;
    while (domain_level < context->totalLevel) {
        if (node_level == context->totalLevel) {
            *active = node_id != 0u;
            return true;
        }
        if (node_level < domain_level || node_level > context->totalLevel) {
            context->ok = false;
            return false;
        }
        if (node_level > domain_level) {
            domain_level = node_level;
            continue;
        }

        NeighborChildPair pair{};
        if (!GetNeighborChildren(context, node_id, node_level, &pair)) {
            return false;
        }
        const unsigned int bit_index = domain_level - context->timeStart;
        const unsigned int bit =
            (context->time >> (context->graph.nips_t - bit_index - 1u)) & 1u;
        if (bit == 0u) {
            node_id = pair.lowId;
            node_level = pair.lowLevel;
        }
        else {
            node_id = pair.highId;
            node_level = pair.highLevel;
        }
        ++domain_level;
    }
    *active = node_level == context->totalLevel && node_id != 0u;
    return node_level == context->totalLevel;
}

bool EvaluateNeighborTime(
    NeighborQueryContext* context,
    unsigned int node_id,
    unsigned int node_level,
    bool* active) {
    if (context->profileEnabled) {
        ++context->state->profile.timePredicates;
    }
    if (node_level == context->totalLevel) {
        *active = node_id != 0u &&
            (context->dataset != DATA_INTERVAL ||
                ((static_cast<std::uint64_t>(context->time) + 1u) & 1u) != 0u);
        return true;
    }
    if (node_level < context->timeStart || node_level > context->totalLevel) {
        context->ok = false;
        return false;
    }

    switch (context->dataset) {
    case DATA_POINT:
        return PointTimeActiveCached(context, node_id, node_level, active);
    case DATA_INTERVAL:
        *active = PrefixAssignmentParity(
            context->graph,
            context->cache,
            node_id,
            node_level,
            context->timeStart,
            context->timeStart,
            context->totalLevel,
            context->time);
        return true;
    case DATA_INCREMENTAL:
        *active = PrefixAssignmentAny(
            context->graph,
            context->cache,
            node_id,
            node_level,
            context->timeStart,
            context->timeStart,
            context->totalLevel,
            context->time);
        return true;
    default:
        context->ok = false;
        return false;
    }
}

bool HasActiveNeighbor(
    NeighborQueryContext* context,
    unsigned int node_id,
    unsigned int node_level,
    bool* active) {
    if (!context->ok || node_level < context->yStart ||
        node_level > context->totalLevel) {
        context->ok = false;
        return false;
    }

    bool found = false;
    if (!FindNeighborMemo(
        context, node_id, node_level, &found, active)) {
        return false;
    }
    if (found) {
        return true;
    }

    bool result = false;
    if (node_level >= context->timeStart) {
        if (!EvaluateNeighborTime(context, node_id, node_level, &result)) {
            return false;
        }
    }
    else {
        NeighborChildPair pair{};
        if (!GetNeighborChildren(context, node_id, node_level, &pair)) {
            return false;
        }
        if (!HasActiveNeighbor(
            context, pair.lowId, pair.lowLevel, &result)) {
            return false;
        }
        if (!result && !HasActiveNeighbor(
            context, pair.highId, pair.highLevel, &result)) {
            return false;
        }
    }

    if (!StoreNeighborMemo(
        context, node_id, node_level, result)) {
        return false;
    }
    *active = result;
    return true;
}

bool EnsureFastNeighborCapacity(NeighborResult* out, unsigned int need) {
    if (out->Capacity >= need) {
        return true;
    }
    unsigned int capacity = out->Capacity == 0u ? 64u : out->Capacity;
    while (capacity < need) {
        if (capacity > UINT32_MAX / 2u) {
            capacity = need;
            break;
        }
        capacity *= 2u;
    }
    if (static_cast<std::size_t>(capacity) >
        SIZE_MAX / sizeof(unsigned int)) {
        return false;
    }
    void* resized = std::realloc(
        out->List, static_cast<std::size_t>(capacity) * sizeof(unsigned int));
    if (resized == nullptr) {
        return false;
    }
    out->List = static_cast<unsigned int*>(resized);
    out->Capacity = capacity;
    return true;
}

bool AppendFastNeighbor(NeighborResult* out, unsigned int value) {
    if (out->Count == UINT32_MAX ||
        !EnsureFastNeighborCapacity(out, out->Count + 1u)) {
        return false;
    }
    out->List[out->Count++] = value;
    return true;
}

bool AppendFastNeighborRange(
    NeighborResult* out,
    unsigned int begin,
    unsigned int end) {
    if (end < begin) {
        return true;
    }
    const std::uint64_t add =
        static_cast<std::uint64_t>(end) - begin + 1u;
    const std::uint64_t required =
        static_cast<std::uint64_t>(out->Count) + add;
    if (required > UINT32_MAX ||
        !EnsureFastNeighborCapacity(
            out, static_cast<unsigned int>(required))) {
        return false;
    }
    for (std::uint64_t value = begin; value <= end; ++value) {
        out->List[out->Count++] = static_cast<unsigned int>(value);
    }
    return true;
}

__forceinline std::uint32_t LowNeighborMask(unsigned int bit_count) {
    if (bit_count == 0u) {
        return 0u;
    }
    return bit_count == 32u
        ? UINT32_MAX
        : (1u << bit_count) - 1u;
}

bool AppendSymbolicNeighbors(
    NeighborResult* out,
    unsigned int fixed_bits,
    unsigned int free_mask,
    unsigned int nodenum,
    bool* needs_sort) {
    if ((fixed_bits & free_mask) != 0u) {
        return false;
    }
    if (fixed_bits > nodenum) {
        return true;
    }

    const bool trailing_mask =
        (free_mask & (free_mask + 1u)) == 0u;
    if (trailing_mask) {
        const unsigned int end = (std::min)(
            fixed_bits | free_mask, nodenum);
        return AppendFastNeighborRange(out, fixed_bits, end);
    }

    *needs_sort = true;
    unsigned int subset = 0u;
    for (;;) {
        const unsigned int value = fixed_bits | subset;
        if (value <= nodenum && !AppendFastNeighbor(out, value)) {
            return false;
        }
        if (subset == free_mask) {
            break;
        }
        subset = (subset - free_mask) & free_mask;
    }
    return true;
}

void VerifyNeighborInternalState(NeighborAccelerationState* state) {
    static bool verified = false;
    if (verified || !NeighborVerificationEnabled()) {
        return;
    }
    verified = true;

    NeighborResult result{};
    bool needs_sort = false;
    bool ok = AppendSymbolicNeighbors(
        &result, 0x1u, 0x4u, 7u, &needs_sort) &&
        AppendSymbolicNeighbors(
            &result, 0x3u, 0x4u, 7u, &needs_sort);
    if (ok && needs_sort) {
        std::sort(result.List, result.List + result.Count);
    }
    const unsigned int expected[4] = { 1u, 3u, 5u, 7u };
    ok = ok && result.Count == 4u;
    for (unsigned int index = 0u; ok && index < 4u; ++index) {
        ok = result.List[index] == expected[index];
    }
    std::free(result.List);
    if (!ok) {
        std::fprintf(stderr,
            "Neighbor symbolic skip-layer verification failed\n");
        std::abort();
    }

    state->generation = UINT16_MAX;
    state->memo[0].generation = UINT16_MAX;
    BeginNeighborMemo(state);
    if (state->generation != 1u || state->memoCount != 0u ||
        state->memo[0].generation != 0u) {
        std::fprintf(stderr,
            "Neighbor memo generation-wrap verification failed\n");
        std::abort();
    }

}

bool PushNeighborState(
    NeighborAccelerationState* state,
    std::size_t* stack_size,
    const NeighborDfsState& value) {
    if (*stack_size >= kNeighborStackCapacity) {
        return false;
    }
    state->stack[(*stack_size)++] = value;
    return true;
}

bool EnumerateActiveNeighbors(
    NeighborQueryContext* context,
    unsigned int start_node,
    unsigned int start_level,
    unsigned int nodenum,
    NeighborResult* out,
    bool* needs_sort) {
    std::size_t stack_size = 0u;
    NeighborDfsState root{};
    root.nodeId = start_node;
    root.nodeLevel = static_cast<std::uint8_t>(start_level);
    if (!PushNeighborState(context->state, &stack_size, root)) {
        return false;
    }

    while (stack_size != 0u) {
        NeighborDfsState current = context->state->stack[--stack_size];
        if (current.depth > context->graph.nips_v ||
            current.nodeLevel > context->totalLevel ||
            current.nodeLevel < context->yStart) {
            return false;
        }

        const unsigned int remaining =
            context->graph.nips_v - current.depth;
        if (current.fixedBits > nodenum) {
            continue;
        }
        if (current.nodeLevel >= context->timeStart) {
            bool active = false;
            if (!HasActiveNeighbor(
                context, current.nodeId, current.nodeLevel, &active)) {
                return false;
            }
            if (!active) {
                if (context->profileEnabled) {
                    ++context->state->profile.inactivePrunes;
                }
                continue;
            }
            current.freeMask |= LowNeighborMask(remaining);
            if (!AppendSymbolicNeighbors(
                out, current.fixedBits, current.freeMask,
                nodenum, needs_sort)) {
                return false;
            }
            continue;
        }
        if (current.depth == context->graph.nips_v) {
            return false;
        }

        const unsigned int target_level =
            context->yStart + current.depth;
        if (current.nodeLevel > target_level) {
            const unsigned int structural_end = (std::min)(
                static_cast<unsigned int>(current.nodeLevel),
                context->timeStart);
            const unsigned int skipped = structural_end - target_level;
            if (skipped == 0u || skipped > remaining) {
                return false;
            }
            const unsigned int before =
                context->graph.nips_v - current.depth;
            const unsigned int after = before - skipped;
            current.freeMask |=
                LowNeighborMask(before) ^ LowNeighborMask(after);
            current.depth = static_cast<std::uint8_t>(
                current.depth + skipped);
            if (!PushNeighborState(
                context->state, &stack_size, current)) {
                return false;
            }
            continue;
        }
        if (current.nodeLevel < target_level) {
            return false;
        }

        NeighborChildPair pair{};
        if (!GetNeighborChildren(
            context, current.nodeId, current.nodeLevel, &pair)) {
            return false;
        }
        const unsigned int decision_mask =
            1u << (context->graph.nips_v - current.depth - 1u);
        const unsigned int next_depth = current.depth + 1u;
        bool low_active = false;
        bool high_active = false;
        if (!HasActiveNeighbor(
            context, pair.lowId, pair.lowLevel, &low_active) ||
            !HasActiveNeighbor(
                context, pair.highId, pair.highLevel, &high_active)) {
            return false;
        }
        if (!low_active && context->profileEnabled) {
            ++context->state->profile.inactivePrunes;
        }
        if (!high_active && context->profileEnabled) {
            ++context->state->profile.inactivePrunes;
        }

        if (high_active) {
            NeighborDfsState high{};
            high.nodeId = pair.highId;
            high.nodeLevel = static_cast<std::uint8_t>(pair.highLevel);
            high.depth = static_cast<std::uint8_t>(next_depth);
            high.fixedBits = current.fixedBits | decision_mask;
            high.freeMask = current.freeMask;
            if (!PushNeighborState(
                context->state, &stack_size, high)) {
                return false;
            }
        }
        if (low_active) {
            NeighborDfsState low{};
            low.nodeId = pair.lowId;
            low.nodeLevel = static_cast<std::uint8_t>(pair.lowLevel);
            low.depth = static_cast<std::uint8_t>(next_depth);
            low.fixedBits = current.fixedBits;
            low.freeMask = current.freeMask;
            if (!PushNeighborState(
                context->state, &stack_size, low)) {
                return false;
            }
        }
    }
    return context->ok;
}

bool ResolveNeighborSource(
    NeighborQueryContext* context,
    unsigned int source,
    unsigned int* node,
    unsigned int* level) {
    const auto start = context->profileEnabled
        ? BuildClock::now()
        : BuildClock::time_point{};
    *node = 0u;
    *level = 0u;
    *node = WalkSegment(
        context->graph,
        context->cache,
        0u,
        level,
        source,
        0u,
        context->graph.nips_v,
        context->totalLevel);
    if (context->profileEnabled) {
        ++context->state->profile.sourceWalks;
    }
    if (context->profileEnabled) {
        context->state->profile.sourceLookupNs +=
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    BuildClock::now() - start).count());
    }
    return *level >= context->yStart && *level <= context->totalLevel;
}

void VerifyNeighborResult(
    const Graph_Compress& graph,
    unsigned int source,
    unsigned int time,
    unsigned int nodenum,
    DataType dataset,
    const NeighborResult& fast) {
    NeighborResult reference{};
    for (std::uint64_t target = 0u; target <= nodenum; ++target) {
        bool active = false;
        switch (dataset) {
        case DATA_POINT:
            active = PointActiveEdge_Point(
                graph, source, static_cast<unsigned int>(target), time);
            break;
        case DATA_INTERVAL:
            active = PointActiveEdge_Interval(
                graph, source, static_cast<unsigned int>(target), time);
            break;
        case DATA_INCREMENTAL:
            active = PointActiveEdge_Incremental(
                graph, source, static_cast<unsigned int>(target), time);
            break;
        default:
            std::free(reference.List);
            std::abort();
        }
        if (active && !AppendFastNeighbor(
            &reference, static_cast<unsigned int>(target))) {
            std::free(reference.List);
            std::abort();
        }
    }
    bool equal = reference.Count == fast.Count;
    for (unsigned int index = 0u; equal && index < fast.Count; ++index) {
        equal = reference.List[index] == fast.List[index];
    }
    if (!equal) {
        std::fprintf(stderr,
            "Neighbor verification failed: fast_count=%u reference_count=%u\n",
            fast.Count, reference.Count);
        std::free(reference.List);
        std::abort();
    }
    for (unsigned int index = 1u; index < fast.Count; ++index) {
        if (fast.List[index - 1u] >= fast.List[index]) {
            std::fprintf(stderr,
                "Neighbor verification failed: result is not strictly sorted at %u\n",
                index);
            std::free(reference.List);
            std::abort();
        }
    }
    std::free(reference.List);
}

bool AcceleratedQueryDirectNeighborsImpl(
    const Graph_Compress& graph,
    unsigned int source,
    unsigned int time,
    unsigned int nodenum,
    DataType dataset,
    NeighborResult* out) {
    if (out == nullptr ||
        graph.nips_v == 0u || graph.nips_v > 32u || graph.nips_t == 0u ||
        dataset < DATA_POINT || dataset > DATA_INCREMENTAL) {
        return false;
    }

    const QueryAccelerationCache* read_cache = FindQueryAcceleration(graph);
    if (read_cache == nullptr) {
        return false;
    }
    QueryAccelerationCache* cache =
        const_cast<QueryAccelerationCache*>(read_cache);
    if (!EnsureNeighborAcceleration(cache)) {
        return false;
    }
    VerifyNeighborInternalState(&cache->neighbor);

    NeighborAccelerationState* state = &cache->neighbor;
    const bool profiling = NeighborProfilingEnabled();
    if (profiling) {
        ++state->profile.queries;
    }
    out->Count = 0u;
    const bool verification = NeighborVerificationEnabled();

    const unsigned int total_level = QueryLevelCount(graph);
    if (total_level > UINT8_MAX) {
        if (profiling) {
            ++state->profile.fallbacks;
        }
        return false;
    }
    BeginNeighborMemo(state);
    NeighborQueryContext context{
        graph, cache, state, time, dataset,
        graph.nips_v, graph.nips_v * 2u,
        total_level, profiling, true };

    unsigned int start_node = 0u;
    unsigned int start_level = 0u;
    if (!ResolveNeighborSource(
        &context, source, &start_node, &start_level)) {
        if (profiling) {
            ++state->profile.fallbacks;
        }
        return false;
    }

    bool root_active = false;
    if (!HasActiveNeighbor(
        &context, start_node, start_level, &root_active)) {
        out->Count = 0u;
        if (profiling) {
            ++state->profile.fallbacks;
        }
        return false;
    }
    if (!root_active) {
        if (verification) {
            VerifyNeighborResult(
                graph, source, time,
                nodenum, dataset, *out);
        }
        return true;
    }

    bool needs_sort = false;
    if (!EnumerateActiveNeighbors(
        &context, start_node, start_level,
        nodenum, out, &needs_sort)) {
        out->Count = 0u;
        if (profiling) {
            ++state->profile.fallbacks;
        }
        return false;
    }
    if (needs_sort && out->Count > 1u) {
        std::sort(out->List, out->List + out->Count);
    }
    if (profiling) {
        state->profile.outputCount += out->Count;
    }
    if (verification) {
        VerifyNeighborResult(
            graph, source, time,
            nodenum, dataset, *out);
    }
    return true;
}

bool ValidTemporalChild(
    const Graph_Compress& graph,
    unsigned int parent_level,
    unsigned int node_id,
    unsigned int node_level) {
    const unsigned int total_level = QueryLevelCount(graph);
    if (node_level <= parent_level || node_level > total_level) {
        return false;
    }
    if (node_level == total_level) {
        return node_id <= 1u;
    }
    const unsigned int begin = node_level == 0u
        ? 0u
        : graph.VariatyNumList[node_level - 1u];
    const unsigned int end = graph.VariatyNumList[node_level];
    return end >= begin && node_id < end - begin;
}

void EnsureTemporalRootCache(QueryAccelerationCache* cache) {
    TemporalAccelerationState* state = &cache->temporal;
    if (state->rootCacheAttempted) {
        return;
    }
    state->rootCacheAttempted = true;
    constexpr std::size_t bytes =
        kTemporalRootCacheCapacity * sizeof(TemporalEdgeRootEntry);
    if (cache->bytes > kResidentCacheBudgetBytes - bytes) {
        return;
    }
    state->edgeRoots.reset(new (std::nothrow) TemporalEdgeRootEntry[
        kTemporalRootCacheCapacity]());
    if (state->edgeRoots != nullptr) {
        state->bytes += bytes;
        cache->bytes += bytes;
    }
}

bool TemporalPointActive(
    const Graph_Compress& graph,
    const QueryAccelerationCache* cache,
    unsigned int node_id,
    unsigned int node_level,
    unsigned int time,
    bool* ok) {
    const unsigned int time_start = graph.nips_v * 2u;
    const unsigned int total_level = QueryLevelCount(graph);
    if (node_level < time_start || node_level > total_level) {
        *ok = false;
        return false;
    }
    if (node_level == total_level) {
        return node_id != 0u;
    }
    unsigned int level = node_level;
    const unsigned int terminal = WalkSegment(
        graph, cache, node_id, &level, time,
        time_start, graph.nips_t, total_level);
    if (level != total_level) {
        *ok = false;
        return false;
    }
    return terminal != 0u;
}

bool TemporalRootActive(
    const Graph_Compress& graph,
    const QueryAccelerationCache* cache,
    unsigned int node_id,
    unsigned int node_level,
    unsigned int time,
    DataType dataset,
    bool* ok) {
    const unsigned int time_start = graph.nips_v * 2u;
    const unsigned int total_level = QueryLevelCount(graph);
    if (node_level < time_start || node_level > total_level) {
        *ok = false;
        return false;
    }
    if (node_level == total_level) {
        return node_id != 0u &&
            (dataset != DATA_INTERVAL ||
                ((static_cast<std::uint64_t>(time) + 1u) & 1u) != 0u);
    }
    switch (dataset) {
    case DATA_POINT:
        return TemporalPointActive(
            graph, cache, node_id, node_level, time, ok);
    case DATA_INTERVAL:
        return PrefixAssignmentParity(
            graph, cache, node_id, node_level,
            time_start, time_start, total_level, time);
    case DATA_INCREMENTAL:
        return PrefixAssignmentAny(
            graph, cache, node_id, node_level,
            time_start, time_start, total_level, time);
    default:
        *ok = false;
        return false;
    }
}

bool FindEdgeTemporalRoot(
    const Graph_Compress& graph,
    QueryAccelerationCache* cache,
    unsigned int u,
    unsigned int v,
    unsigned int* node,
    unsigned int* level) {
    EnsureTemporalRootCache(cache);
    TemporalAccelerationState* state = &cache->temporal;
    const std::size_t slot = static_cast<std::size_t>(MixNeighborKey(
        (static_cast<std::uint64_t>(u) << 32) | v)) &
        (kTemporalRootCacheCapacity - 1u);
    if (state->edgeRoots != nullptr) {
        const TemporalEdgeRootEntry& entry = state->edgeRoots[slot];
        if (entry.valid != 0u && entry.u == u && entry.v == v) {
            *node = entry.nodeId;
            *level = entry.nodeLevel;
            return true;
        }
    }

    const unsigned int total_level = QueryLevelCount(graph);
    unsigned int root_level = 0u;
    unsigned int root_node = 0u;
    root_node = WalkSegment(
        graph, cache, 0u, &root_level, u,
        0u, graph.nips_v, total_level);
    if (root_level != total_level) {
        root_node = WalkSegment(
            graph, cache, root_node, &root_level, v,
            graph.nips_v, graph.nips_v, total_level);
    }
    const unsigned int time_start = graph.nips_v * 2u;
    if (root_level < time_start || root_level > total_level ||
        root_level > UINT8_MAX) {
        return false;
    }
    if (state->edgeRoots != nullptr) {
        TemporalEdgeRootEntry& entry = state->edgeRoots[slot];
        entry.u = u;
        entry.v = v;
        entry.nodeId = root_node;
        entry.nodeLevel = static_cast<std::uint8_t>(root_level);
        entry.valid = 1u;
    }
    *node = root_node;
    *level = root_level;
    return true;
}

bool FindFirstTemporalEvent(
    const Graph_Compress& graph,
    const QueryAccelerationCache* cache,
    unsigned int node_id,
    unsigned int node_level,
    unsigned int domain_level,
    unsigned int lower_bound,
    unsigned int upper_bound,
    unsigned int prefix,
    bool greater,
    unsigned int* result,
    bool* ok) {
    const unsigned int time_start = graph.nips_v * 2u;
    const unsigned int total_level = QueryLevelCount(graph);
    if (node_level > total_level || node_level < domain_level ||
        domain_level < time_start || domain_level > total_level) {
        *ok = false;
        return false;
    }
    if (node_level == total_level) {
        if (node_id == 0u) {
            return false;
        }
        const unsigned int candidate = greater ? prefix : lower_bound;
        if (candidate > upper_bound) {
            return false;
        }
        *result = candidate;
        return true;
    }
    if (domain_level == total_level) {
        return false;
    }

    const std::uint64_t minimum = greater
        ? prefix
        : lower_bound;
    if (minimum > upper_bound) {
        return false;
    }
    const unsigned int bit_index = domain_level - time_start;
    const unsigned int lower_bit =
        (lower_bound >> (graph.nips_t - bit_index - 1u)) & 1u;
    const unsigned int result_bit =
        1u << (graph.nips_t - bit_index - 1u);
    if (node_level > domain_level) {
        // A skipped BDD variable permits both bit values. If the matching
        // lower-bound branch has no event, its higher sibling may still do so.
        if (!greater && lower_bit == 0u) {
            if (FindFirstTemporalEvent(
                graph, cache, node_id, node_level, domain_level + 1u,
                lower_bound, upper_bound, prefix, false, result, ok)) {
                return true;
            }
            if (!*ok) return false;
            return FindFirstTemporalEvent(
                graph, cache, node_id, node_level, domain_level + 1u,
                lower_bound, upper_bound, prefix | result_bit, true, result, ok);
        }
        const unsigned int chosen = greater ? 0u : lower_bit;
        return FindFirstTemporalEvent(
            graph, cache, node_id, node_level, domain_level + 1u,
            lower_bound, upper_bound,
            prefix | (chosen != 0u ? result_bit : 0u),
            greater, result, ok);
    }

    unsigned int low_level = node_level;
    unsigned int high_level = node_level;
    const unsigned int low = UnpackBranchCached(
        graph, cache, node_id, 0u, &low_level);
    const unsigned int high = UnpackBranchCached(
        graph, cache, node_id, 1u, &high_level);
    if (!ValidTemporalChild(graph, node_level, low, low_level) ||
        !ValidTemporalChild(graph, node_level, high, high_level)) {
        *ok = false;
        return false;
    }

    if (greater) {
        if (FindFirstTemporalEvent(
            graph, cache, low, low_level, domain_level + 1u,
            lower_bound, upper_bound, prefix, true, result, ok)) {
            return true;
        }
        if (!*ok) {
            return false;
        }
        return FindFirstTemporalEvent(
            graph, cache, high, high_level, domain_level + 1u,
            lower_bound, upper_bound, prefix | result_bit, true, result, ok);
    }
    if (lower_bit == 0u) {
        if (FindFirstTemporalEvent(
            graph, cache, low, low_level, domain_level + 1u,
            lower_bound, upper_bound, prefix, false, result, ok)) {
            return true;
        }
        if (!*ok) {
            return false;
        }
        return FindFirstTemporalEvent(
            graph, cache, high, high_level, domain_level + 1u,
            lower_bound, upper_bound, prefix | result_bit, true, result, ok);
    }
    return FindFirstTemporalEvent(
        graph, cache, high, high_level, domain_level + 1u,
        lower_bound, upper_bound, prefix | result_bit, false, result, ok);
}

std::unique_ptr<QueryAccelerationCache> BuildAccelerationCache(
    const Graph_Compress& graph,
    const BuildDeadline& deadline) {
    if (graph.NormalFlagList == nullptr ||
        graph.ShareFlagList == nullptr || graph.ShareValue == nullptr ||
        graph.ShareNodeList == nullptr || graph.VariatyNumList == nullptr) {
        return nullptr;
    }

    std::unique_ptr<QueryAccelerationCache> cache(
        new (std::nothrow) QueryAccelerationCache());
    if (cache == nullptr) {
        return nullptr;
    }

    // Sparse rank/select directories let traversal reject inactive branches early.
    const unsigned int level_count = QueryLevelCount(graph);
    bool normal_uses64 = graph.NormalFlagList->Count > UINT32_MAX;
    bool share_uses64 = graph.ShareFlagList->Count > UINT32_MAX;
    unsigned int normal_shift = kNormalRankBlockShift;
    unsigned int share_shift = kShareRankBlockShift;
    for (;;) {
        const std::size_t normal_bytes = EstimateBitRankDirBytes(
            graph.NormalFlagList->Count, level_count, normal_shift, normal_uses64);
        const std::size_t share_bytes = EstimateBitRankDirBytes(
            graph.ShareFlagList->Count, level_count, share_shift, share_uses64);
        if (normal_bytes + share_bytes <= kRankBudgetBytes) {
            break;
        }
        if (normal_bytes >= share_bytes) {
            ++normal_shift;
        }
        else {
            ++share_shift;
        }
    }

    if (!BuildBitRankDir(
        graph.NormalFlagList, level_count, normal_shift, normal_uses64,
        deadline, &cache->normalRank) ||
        !BuildBitRankDir(
            graph.ShareFlagList, level_count, share_shift, share_uses64,
            deadline, &cache->shareRank)) {
        return nullptr;
    }
    cache->bytes = cache->normalRank.bytes() + cache->shareRank.bytes();

    std::uint64_t rice_one_count = 0;
    if (!CountOnesWithDeadline(
        graph.ShareValue->quotient, 0, graph.ShareValue->quoCount,
        deadline, &rice_one_count)) {
        return nullptr;
    }
    const bool rice_uses64 = graph.ShareValue->quoCount > UINT32_MAX;
    unsigned int rice_stride = kDefaultRiceStride;
    while (EstimateRiceSelectBytes(
        rice_one_count, level_count, rice_stride, rice_uses64) >
        kRiceBudgetBytes) {
        if (rice_stride > UINT32_MAX / 2u) {
            return nullptr;
        }
        rice_stride *= 2u;
    }
    if (!BuildRiceSelectDir(
        graph.ShareValue, level_count, rice_stride, rice_one_count,
        deadline, &cache->riceSelect)) {
        return nullptr;
    }
    cache->bytes += cache->riceSelect.bytes();

    const bool build_share_gamma = graph.ShareNodeList->Count <= UINT32_MAX;
    const bool build_skip_gamma = graph.SkipFlagList != nullptr &&
        graph.SkipFlagList->Count <= UINT32_MAX;
    std::vector<std::uint32_t> share_runs;
    std::vector<std::uint32_t> share_positions;
    std::vector<std::uint32_t> share_ones;
    std::vector<std::uint32_t> skip_runs;
    std::vector<std::uint32_t> skip_positions;
    std::vector<std::uint32_t> skip_ones;
    if (build_share_gamma && !ScanRunGammaLevels(
        graph.ShareNodeList, level_count, &share_runs,
        &share_positions, &share_ones, deadline)) {
        return nullptr;
    }
    if (build_skip_gamma && !ScanRunGammaLevels(
        graph.SkipFlagList, level_count, &skip_runs,
        &skip_positions, &skip_ones, deadline)) {
        return nullptr;
    }

    unsigned int share_gamma_stride = kDefaultGammaStride;
    unsigned int skip_gamma_stride = kDefaultGammaStride;
    for (;;) {
        const std::size_t share_bytes = build_share_gamma
            ? EstimateGammaBytes(
                share_runs, level_count, share_gamma_stride)
            : 0u;
        const std::size_t skip_bytes = build_skip_gamma
            ? EstimateGammaBytes(
                skip_runs, level_count, skip_gamma_stride)
            : 0u;
        if (share_bytes <= kGammaBudgetBytes &&
            skip_bytes <= kGammaBudgetBytes - share_bytes) {
            break;
        }
        unsigned int* stride = share_bytes >= skip_bytes
            ? &share_gamma_stride
            : &skip_gamma_stride;
        if (*stride > UINT32_MAX / 2u) {
            return nullptr;
        }
        *stride *= 2u;
    }
    if (build_share_gamma && !BuildRunGammaCkpt(
        graph.ShareNodeList, level_count, share_gamma_stride,
        share_runs, share_positions, share_ones, deadline,
        &cache->shareNodeGamma)) {
        return nullptr;
    }
    if (build_skip_gamma && !BuildRunGammaCkpt(
        graph.SkipFlagList, level_count, skip_gamma_stride,
        skip_runs, skip_positions, skip_ones, deadline,
        &cache->skipFlagGamma)) {
        return nullptr;
    }
    cache->bytes += cache->shareNodeGamma.bytes() +
        cache->skipFlagGamma.bytes();

    if (cache->bytes > kResidentCacheBudgetBytes ||
        BuildDeadlineExpired(deadline)) {
        return nullptr;
    }
    if (cache->bytes <=
        kResidentCacheBudgetBytes - kParityCacheBudgetBytes) {
        cache->parity.entries.reset(new (std::nothrow) ParityCacheEntry[
            kParityCacheBudgetBytes / sizeof(ParityCacheEntry)]());
        if (cache->parity.entries != nullptr) {
            cache->bytes += cache->parity.bytes();
        }
    }
    if (cache->bytes <=
        kResidentCacheBudgetBytes - kBranchChildBudgetBytes) {
        cache->branchChildren.entries.reset(
            new (std::nothrow) BranchChildEntry[kBranchChildCapacity]());
        if (cache->branchChildren.entries != nullptr) {
            cache->bytes += cache->branchChildren.bytes();
        }
    }
    if (cache->bytes <=
        kResidentCacheBudgetBytes - kBatchWorkspaceBudgetBytes) {
        cache->batch.order.reset(new (std::nothrow) std::uint32_t[
            kBatchWorkspaceBudgetBytes / sizeof(std::uint32_t)]);
        if (cache->batch.order != nullptr) {
            cache->bytes += cache->batch.bytes();
        }
    }
    if (cache->bytes > kResidentCacheBudgetBytes ||
        cache->bytes + (kCacheBudgetBytes - kResidentCacheBudgetBytes) >
            kCacheBudgetBytes ||
        BuildDeadlineExpired(deadline)) {
        return nullptr;
    }
    return cache;
}

const QueryAccelerationCache* FindQueryAcceleration(const Graph_Compress& graph) {
    if (g_thread_graph == &graph) {
        return g_thread_cache;
    }

    bool built_now = false;
    const QueryAccelerationCache* result = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_query_cache_mutex);
        auto found = g_query_caches.find(&graph);
        if (found == g_query_caches.end()) {
            built_now = true;
            const auto start = BuildClock::now();
            const BuildDeadline deadline = start + std::chrono::seconds(30);
            std::unique_ptr<QueryAccelerationCache> cache;
            try {
                cache = BuildAccelerationCache(graph, deadline);
            }
            catch (const std::bad_alloc&) {
                cache.reset();
            }
            found = g_query_caches.emplace(&graph, std::move(cache)).first;
        }
        result = found->second.get();
        g_thread_graph = &graph;
        g_thread_cache = result;
    }

    if (built_now && result != nullptr && QueryVerificationEnabled()) {
        if (!VerifyAcceleratedCore(graph, 16u)) {
            std::fprintf(stderr, "Query acceleration core verification failed\n");
            std::abort();
        }
    }
    return result;
}

}

void BuildQueryAcceleration(const Graph_Compress& graph) {
    ClearThreadAcceleration(graph);
    (void)FindQueryAcceleration(graph);
}

void FreeQueryAcceleration(const Graph_Compress& graph) {
    ClearThreadAcceleration(graph);
    std::lock_guard<std::mutex> lock(g_query_cache_mutex);
    const auto found = g_query_caches.find(&graph);
    if (found != g_query_caches.end()) {
        g_query_caches.erase(found);
    }
}

std::size_t QueryAccelerationBytes(const Graph_Compress& graph) {
    std::lock_guard<std::mutex> lock(g_query_cache_mutex);
    const auto found = g_query_caches.find(&graph);
    return found == g_query_caches.end() || found->second == nullptr
        ? 0u
        : found->second->bytes;
}

bool AcceleratedQueryDirectNeighbors(
    const Graph_Compress& graph,
    unsigned int u,
    unsigned int t,
    unsigned int nodenum,
    DataType dataset,
    NeighborResult* out) {
    return AcceleratedQueryDirectNeighborsImpl(
        graph, u, t, nodenum, dataset, out);
}

bool AcceleratedQueryNextAct(
    const Graph_Compress& graph,
    unsigned int u,
    unsigned int v,
    unsigned int t,
    unsigned int timenum,
    DataType dataset,
    unsigned int* result) {
    if (result == nullptr ||
        (dataset != DATA_POINT && dataset != DATA_INTERVAL &&
            dataset != DATA_INCREMENTAL)) {
        return false;
    }
    const QueryAccelerationCache* read_cache = FindQueryAcceleration(graph);
    QueryAccelerationCache* cache =
        const_cast<QueryAccelerationCache*>(read_cache);
    if (cache == nullptr) {
        return false;
    }

    unsigned int node = 0u;
    unsigned int level = 0u;
    if (!FindEdgeTemporalRoot(
        graph, cache, u, v, &node, &level)) {
        return false;
    }
    bool ok = true;
    if (TemporalRootActive(
        graph, cache, node, level, t, dataset, &ok)) {
        *result = t;
        return ok;
    }
    if (!ok) {
        return false;
    }
    if (t >= timenum) {
        *result = QUERY_TIME_INF;
        return true;
    }

    const unsigned int lower_bound = t + 1u;
    unsigned int found = QUERY_TIME_INF;
    if (FindFirstTemporalEvent(
        graph, cache, node, level,
        graph.nips_v * 2u,
        lower_bound, timenum, 0u, false, &found, &ok)) {
        *result = found;
    }
    else if (!ok) {
        return false;
    }
    else {
        *result = QUERY_TIME_INF;
    }
    return true;
}

std::uint32_t AcceleratedBitPopCount(
    const unsigned int* data, std::uint64_t first, std::uint64_t last) {
    if (data == nullptr || last < first) {
        return 0;
    }

    const std::uint64_t first_word = first >> 5;
    const std::uint64_t last_word = last >> 5;
    const unsigned int first_offset = static_cast<unsigned int>(first & 31u);
    const unsigned int last_offset = static_cast<unsigned int>(last & 31u);

    if (first_word == last_word) {
        const unsigned int start_mask = ~0u << first_offset;
        const unsigned int end_mask = last_offset == 31u
            ? ~0u
            : (1u << (last_offset + 1u)) - 1u;
        return PopCount32(data[first_word] & start_mask & end_mask);
    }

    unsigned int count = PopCount32(data[first_word] & (~0u << first_offset));
    std::uint64_t word = first_word + 1u;
    for (; word + 1u < last_word; word += 2u) {
        const unsigned __int64 pair =
            static_cast<unsigned __int64>(data[word]) |
            (static_cast<unsigned __int64>(data[word + 1u]) << 32);
        count += PopCount64(pair);
    }
    if (word < last_word) {
        count += PopCount32(data[word]);
    }

    const unsigned int end_mask = last_offset == 31u
        ? ~0u
        : (1u << (last_offset + 1u)) - 1u;
    count += PopCount32(data[last_word] & end_mask);
    return count;
}

AcceleratedBitRank AcceleratedGetBitAndRank(
    const BitArray* array, unsigned int level, unsigned int position) {
    const unsigned int flag = array->Flag[level];
    if (flag == 0) {
        return AcceleratedBitRank{ 0u, 0u };
    }
    if (flag == 1) {
        return AcceleratedBitRank{ 1u, position + 1u };
    }

    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1];
    const std::uint64_t absolute_position = start + position;
    return AcceleratedBitRank{
        (array->List[absolute_position >> 5] >> (absolute_position & 31u)) & 1u,
        AcceleratedBitPopCount(array->List, start, absolute_position)
    };
}

unsigned int AcceleratedUnpackOne(
    const Graph_Compress& graph, unsigned int node_index, unsigned int* level) {
    return UnpackOneCached(graph, FindQueryAcceleration(graph), node_index, level);
}

MinHeap* AcceleratedUnpackAll(
    const Graph_Compress& graph, MinHeap* visitlist) {
    if (visitlist == nullptr || visitlist->size <= 0 ||
        visitlist->data == nullptr) {
        return visitlist;
    }

    const QueryAccelerationCache* read_cache = FindQueryAcceleration(graph);
    QueryAccelerationCache* cache =
        const_cast<QueryAccelerationCache*>(read_cache);
    const unsigned int total_level = QueryLevelCount(graph);
    const std::size_t count = static_cast<std::size_t>(visitlist->size);
    const bool use_batch = cache != nullptr && count >= 8u &&
        count <= cache->batch.capacity();
    if (use_batch) {
        for (std::uint32_t index = 0u; index < count; ++index) {
            cache->batch.order[index] = index;
        }
        std::sort(
            cache->batch.order.get(), cache->batch.order.get() + count,
            [visitlist](std::uint32_t left, std::uint32_t right) {
                const node& lhs = visitlist->data[left].parent;
                const node& rhs = visitlist->data[right].parent;
                return lhs.level != rhs.level
                    ? lhs.level < rhs.level
                    : lhs.id < rhs.id;
            });
    }

    if (!use_batch) {
        for (std::size_t index = 0u; index < count; ++index) {
            Node& item = visitlist->data[index];
            const unsigned int parent_level = item.parent.level;
            if (parent_level >= total_level ||
                item.parent.id > UINT32_MAX / 2u) {
                item.left.id = 0u;
                item.left.level = total_level;
                item.right = item.left;
                continue;
            }
            unsigned int low_level = parent_level;
            unsigned int high_level = parent_level;
            item.left.id = UnpackBranchCached(
                graph, cache, item.parent.id, 0u, &low_level);
            item.left.level = low_level;
            item.right.id = UnpackBranchCached(
                graph, cache, item.parent.id, 1u, &high_level);
            item.right.level = high_level;
        }
        return visitlist;
    }

    std::size_t position = 0u;
    while (position < count) {
        const unsigned int parent_level = visitlist->data[
            cache->batch.order[position]].parent.level;
        std::size_t level_end = position + 1u;
        while (level_end < count && visitlist->data[
            cache->batch.order[level_end]].parent.level == parent_level) {
            ++level_end;
        }
        RunGammaForwardCursor skip_cursor{};
        if (parent_level < total_level) {
            skip_cursor = MakeRunGammaForwardCursor(
                graph.SkipFlagList, parent_level);
        }

        for (; position < level_end; ++position) {
            const std::size_t index = cache->batch.order[position];
            Node& item = visitlist->data[index];
            if (parent_level >= total_level ||
                item.parent.id > UINT32_MAX / 2u) {
                item.left.id = 0u;
                item.left.level = total_level;
                item.right = item.left;
                continue;
            }
            if (position != 0u) {
                const std::size_t previous_index =
                    cache->batch.order[position - 1u];
                const Node& previous = visitlist->data[previous_index];
                if (previous.parent.id == item.parent.id &&
                    previous.parent.level == item.parent.level) {
                    item.left = previous.left;
                    item.right = previous.right;
                    continue;
                }
            }

            unsigned int low_level = parent_level;
            unsigned int high_level = parent_level;
            if (!TryBranchChild(
                cache, item.parent.id, parent_level, 0u,
                &item.left.id, &low_level)) {
                item.left.id = UnpackOneBatch(
                    graph, cache, item.parent.id << 1,
                    &low_level, &skip_cursor);
                StoreBranchChild(
                    cache, item.parent.id, parent_level,
                    0u, item.left.id, low_level);
            }
            item.left.level = low_level;

            if (!TryBranchChild(
                cache, item.parent.id, parent_level, 1u,
                &item.right.id, &high_level)) {
                item.right.id = UnpackOneBatch(
                    graph, cache, (item.parent.id << 1) | 1u,
                    &high_level, &skip_cursor);
                StoreBranchChild(
                    cache, item.parent.id, parent_level,
                    1u, item.right.id, high_level);
            }
            item.right.level = high_level;
        }
    }
    return visitlist;
}

unsigned int AcceleratedGetDirectNode(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int path,
    int path_kind,
    unsigned int* end_level) {
    unsigned int segment_start = 0;
    unsigned int segment_length = graph.nips_v;
    if (path_kind == 0) {
        segment_start = graph.nips_v * 2u;
        segment_length = graph.nips_t;
    }
    else if (path_kind == 2) {
        segment_start = graph.nips_v;
    }

    unsigned int level = start_level;
    const unsigned int total_level = QueryLevelCount(graph);
    const QueryAccelerationCache* cache = FindQueryAcceleration(graph);
    const unsigned int result = WalkSegment(
        graph,
        cache,
        node_id,
        &level,
        path,
        segment_start,
        segment_length,
        total_level);
    *end_level = level;
    return result;
}

bool AcceleratedPointActiveEdgePoint(
    const Graph_Compress& graph, unsigned int u, unsigned int v, unsigned int t) {
    const unsigned int total_level = QueryLevelCount(graph);
    const QueryAccelerationCache* cache = FindQueryAcceleration(graph);
    CoreProfileScope profile(cache);
    profile.SetStage(CoreQueryStage::Source);
    unsigned int level = 0;
    profile.SourceWalk();
    unsigned int node = WalkSegment(
        graph, cache, 0, &level, u, 0, graph.nips_v, total_level);
    bool result = false;
    if (level == total_level) {
        result = node != 0;
    }
    else {
        profile.SetStage(CoreQueryStage::Target);
        node = WalkSegment(
            graph, cache, node, &level, v,
            graph.nips_v, graph.nips_v, total_level);
        if (level == total_level) {
            result = node != 0;
        }
        else {
            profile.SetStage(CoreQueryStage::Time);
            node = WalkSegment(
                graph, cache, node, &level, t,
                graph.nips_v * 2u, graph.nips_t, total_level);
            result = level == total_level && node != 0;
        }
    }

    if (QueryVerificationEnabled()) {
        const bool reference = ReferencePointActiveEdgePoint(graph, u, v, t);
        if (result != reference) {
            std::fprintf(stderr,
                "Point query verification failed: u=%u v=%u t=%u fast=%u reference=%u\n",
                u, v, t, result ? 1u : 0u, reference ? 1u : 0u);
            std::abort();
        }
    }
    return result;
}

bool AcceleratedPointActiveEdgeInterval(
    const Graph_Compress& graph, unsigned int u, unsigned int v, unsigned int t) {
    const unsigned int time_start_level = graph.nips_v * 2u;
    const unsigned int total_level = time_start_level + graph.nips_t;
    const QueryAccelerationCache* cache = FindQueryAcceleration(graph);
    CoreProfileScope profile(cache);
    profile.SetStage(CoreQueryStage::Source);
    unsigned int level = 0;
    profile.SourceWalk();
    unsigned int node = WalkSegment(
        graph, cache, 0, &level, u, 0, graph.nips_v, total_level);
    if (level == total_level) {
        return node != 0 &&
            ((static_cast<std::uint64_t>(t) + 1u) & 1u) != 0u;
    }
    profile.SetStage(CoreQueryStage::Target);
    node = WalkSegment(
        graph, cache, node, &level, v, graph.nips_v, graph.nips_v, total_level);
    if (level == total_level) {
        return node != 0 &&
            ((static_cast<std::uint64_t>(t) + 1u) & 1u) != 0u;
    }
    profile.SetStage(CoreQueryStage::Time);
    return PrefixAssignmentParity(
        graph,
        cache,
        node,
        level,
        time_start_level,
        time_start_level,
        total_level,
        t);
}

static bool CompareTimeCached(
    const Graph_Compress& graph,
    const QueryAccelerationCache* cache,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int time) {
    const unsigned int total_level = QueryLevelCount(graph);
    const unsigned int segment_start = graph.nips_v * 2u;
    if (start_level < segment_start || start_level > total_level) {
        return false;
    }
    return PrefixAssignmentAny(
        graph,
        cache,
        node_id,
        start_level,
        segment_start,
        segment_start,
        total_level,
        time);
}

bool AcceleratedCompareTime(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int time) {
    return CompareTimeCached(
        graph, FindQueryAcceleration(graph),
        node_id, start_level, time);
}

bool AcceleratedCompareTimeParity(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int time) {
    const unsigned int total_level = QueryLevelCount(graph);
    const unsigned int segment_start = graph.nips_v * 2u;
    if (start_level < segment_start || start_level > total_level) {
        return false;
    }
    return PrefixAssignmentParity(
        graph,
        FindQueryAcceleration(graph),
        node_id,
        start_level,
        segment_start,
        segment_start,
        total_level,
        time);
}

bool AcceleratedPointActiveEdgeIncremental(
    const Graph_Compress& graph,
    unsigned int u,
    unsigned int v,
    unsigned int t) {
    const unsigned int total_level = QueryLevelCount(graph);
    const QueryAccelerationCache* cache = FindQueryAcceleration(graph);
    CoreProfileScope profile(cache);
    profile.SetStage(CoreQueryStage::Source);
    unsigned int level = 0u;
    profile.SourceWalk();
    unsigned int node = WalkSegment(
        graph, cache, 0u, &level, u,
        0u, graph.nips_v, total_level);
    if (level == total_level) {
        return node != 0u;
    }
    profile.SetStage(CoreQueryStage::Target);
    node = WalkSegment(
        graph, cache, node, &level, v,
        graph.nips_v, graph.nips_v, total_level);
    if (level == total_level) {
        return node != 0u;
    }
    profile.SetStage(CoreQueryStage::Time);
    return CompareTimeCached(graph, cache, node, level, t);
}

int* AcceleratedSameBitRankArray(
    const BitArray* array,
    const int* positions,
    unsigned int count,
    unsigned int level) {
    int* output = static_cast<int*>(std::malloc(sizeof(int) * count));
    if (output == nullptr) {
        return nullptr;
    }

    const unsigned int flag = array->Flag[level];
    if (flag < 2u) {
        for (unsigned int i = 0; i < count; ++i) {
            output[i] = positions[i] < 0
                ? -1
                : (flag == 0 ? 0 : positions[i] + 1);
        }
        return output;
    }

    std::vector<SortedRankQuery> queries;
    queries.reserve(count);
    for (unsigned int i = 0; i < count; ++i) {
        if (positions[i] < 0) {
            output[i] = -1;
        }
        else {
            queries.push_back(SortedRankQuery{
                static_cast<std::uint32_t>(positions[i]), i });
        }
    }
    std::sort(queries.begin(), queries.end(), [](const SortedRankQuery& left, const SortedRankQuery& right) {
        return left.index < right.index;
    });

    const std::uint64_t layer_start = level == 0 ? 0 : array->Index[level - 1];
    std::uint64_t scan_start = layer_start;
    unsigned int rank = 0;
    for (const SortedRankQuery& query : queries) {
        const std::uint64_t end = layer_start + query.index;
        if (scan_start <= end) {
            rank += AcceleratedBitPopCount(array->List, scan_start, end);
            scan_start = end + 1u;
        }
        output[query.original_position] = static_cast<int>(rank);
    }
    return output;
}

int* AcceleratedRiceAccessArray(
    const RiceArray* array,
    unsigned int level,
    const int* one_based_indices,
    unsigned int count) {
    int* output = static_cast<int*>(std::malloc(sizeof(int) * count));
    if (output == nullptr) {
        return nullptr;
    }

    std::vector<SortedRankQuery> queries;
    queries.reserve(count);
    for (unsigned int i = 0; i < count; ++i) {
        if (one_based_indices[i] <= 0) {
            output[i] = -1;
        }
        else {
            queries.push_back(SortedRankQuery{
                static_cast<std::uint32_t>(one_based_indices[i] - 1), i });
        }
    }
    std::sort(queries.begin(), queries.end(), [](const SortedRankQuery& left, const SortedRankQuery& right) {
        return left.index < right.index;
    });

    const unsigned int parameter = array->RiceParameter[level];
    const std::uint64_t remainder_start =
        level == 0 ? 0 : array->remIndex[level - 1];
    std::uint64_t quotient_position =
        level == 0 ? 0 : array->quoIndex[level - 1];
    unsigned int current_index = 0;
    std::size_t query_position = 0;
    while (query_position < queries.size()) {
        const unsigned int target_index = queries[query_position].index;
        if (current_index < target_index) {
            SkipOnes(
                array->quotient,
                quotient_position,
                target_index - current_index,
                &quotient_position);
            current_index = target_index;
        }

        unsigned int quotient = 0;
        std::uint64_t scan_position = quotient_position;
        for (;;) {
            const std::uint64_t word_index = scan_position >> 5;
            const unsigned int offset =
                static_cast<unsigned int>(scan_position & 31u);
            const unsigned int word = array->quotient[word_index] >> offset;
            if (word != 0) {
                quotient += CountTrailingZeros(word);
                break;
            }
            const unsigned int consumed = 32u - offset;
            quotient += consumed;
            scan_position += consumed;
        }

        const unsigned int remainder = parameter == 0
            ? 0
            : ReadBits(
                array->remainder,
                remainder_start +
                    static_cast<std::uint64_t>(target_index) * parameter,
                parameter);
        const int value = static_cast<int>((quotient << parameter) | remainder);
        do {
            output[queries[query_position].original_position] = value;
            ++query_position;
        } while (query_position < queries.size() &&
            queries[query_position].index == target_index);
    }
    return output;
}

int* AcceleratedFixBitsAccessArray(
    const FixBitArray* array,
    unsigned int level,
    const int* one_based_indices,
    unsigned int count) {
    int* output = static_cast<int*>(std::malloc(sizeof(int) * count));
    if (output == nullptr) {
        return nullptr;
    }

    for (unsigned int i = 0; i < count; ++i) {
        const int one_based_index = one_based_indices[i];
        output[i] = one_based_index <= 0
            ? -1
            : static_cast<int>(FixBitsAccessFast(
                array, level, static_cast<unsigned int>(one_based_index - 1)));
    }
    return output;
}

bool AcceleratedRunGammaBitRankArray(
    const RunBitArray* array,
    unsigned int level,
    const int* positions,
    unsigned int count,
    int** bits,
    int** ranks) {
    int* output_bits = static_cast<int*>(std::malloc(sizeof(int) * count));
    int* output_ranks = static_cast<int*>(std::malloc(sizeof(int) * count));
    if (output_bits == nullptr || output_ranks == nullptr) {
        std::free(output_bits);
        std::free(output_ranks);
        return false;
    }

    std::vector<SortedRankQuery> queries;
    queries.reserve(count);
    for (unsigned int i = 0; i < count; ++i) {
        if (positions[i] < 0) {
            output_bits[i] = -1;
            output_ranks[i] = -1;
        }
        else {
            queries.push_back(SortedRankQuery{
                static_cast<std::uint32_t>(positions[i]), i });
        }
    }
    std::sort(queries.begin(), queries.end(), [](const SortedRankQuery& left, const SortedRankQuery& right) {
        return left.index < right.index;
    });

    if (array->Flag[level]) {
        const int bit = static_cast<int>(array->First[level]);
        for (const SortedRankQuery& query : queries) {
            output_bits[query.original_position] = bit;
            output_ranks[query.original_position] = bit == 0
                ? 0
                : static_cast<int>(query.index + 1u);
        }
        *bits = output_bits;
        *ranks = output_ranks;
        return true;
    }

    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1];
    const std::uint64_t end = array->Index[level];
    FastBitReader reader{ array->List, start };
    unsigned int run_start = 0;
    unsigned int rank = 0;
    unsigned int current_bit = array->First[level];
    std::size_t query_position = 0;
    while (reader.position < end && query_position < queries.size()) {
        const unsigned int run = ReadGamma(&reader);
        const unsigned int run_end = run_start + run;
        while (query_position < queries.size() &&
            queries[query_position].index < run_end) {
            const SortedRankQuery& query = queries[query_position];
            output_bits[query.original_position] = static_cast<int>(current_bit);
            output_ranks[query.original_position] = static_cast<int>(
                rank + (current_bit == 0 ? 0u : query.index - run_start + 1u));
            ++query_position;
        }
        if (current_bit != 0) {
            rank += run;
        }
        run_start = run_end;
        current_bit ^= 1u;
    }
    while (query_position < queries.size()) {
        output_bits[queries[query_position].original_position] = 0;
        output_ranks[queries[query_position].original_position] = static_cast<int>(rank);
        ++query_position;
    }

    *bits = output_bits;
    *ranks = output_ranks;
    return true;
}

unsigned int ReferenceUnpackOne(
    const Graph_Compress& graph, unsigned int node_index, unsigned int* level) {
    const unsigned int current_level = *level;
    const unsigned int total_level = graph.nips_v * 2u + graph.nips_t;
    if (current_level == total_level - 1u) {
        *level = total_level;
        return ReferenceBitAndRank(
            graph.NormalFlagList, current_level, node_index).bit;
    }

    const AcceleratedBitRank normal = ReferenceBitAndRank(
        graph.NormalFlagList, current_level, node_index);
    if (normal.bit != 0) {
        *level = current_level + 1u;
        return normal.rank - 1u;
    }

    const unsigned int share_position = node_index - normal.rank;
    const AcceleratedBitRank share = ReferenceBitAndRank(
        graph.ShareFlagList, current_level, share_position);
    if (share.bit != 0) {
        const unsigned int share_value = ReferenceRiceAccess(
            graph.ShareValue, current_level, share.rank - 1u);
        *level = current_level + 1u;
        return ReferenceSelectOneRunGamma(
            graph.ShareNodeList, *level, share_value);
    }

    const unsigned int skip_position = share_position - share.rank;
    const AcceleratedBitRank skip = ReferenceRunGammaBitAndRank(
        graph.SkipFlagList, current_level, skip_position);
    if (skip.bit == 0) {
        *level = total_level;
        return 0;
    }

    unsigned int skip_value = ReferenceFixBitsAccess(
        graph.SkipValue, current_level, skip.rank - 1u);
    unsigned int result_level = current_level + 2u;
    while (result_level < total_level) {
        const unsigned int layer_size =
            graph.VariatyNumList[result_level] -
            graph.VariatyNumList[result_level - 1u];
        if (skip_value < layer_size) {
            break;
        }
        skip_value -= layer_size;
        ++result_level;
    }
    *level = result_level;
    return skip_value;
}

unsigned int ReferenceGetDirectNode(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int path,
    int path_kind,
    unsigned int* end_level) {
    unsigned int segment_start = 0;
    unsigned int segment_length = graph.nips_v;
    if (path_kind == 0) {
        segment_start = graph.nips_v * 2u;
        segment_length = graph.nips_t;
    }
    else if (path_kind == 2) {
        segment_start = graph.nips_v;
    }

    unsigned int level = start_level;
    const unsigned int result = ReferenceWalkSegment(
        graph,
        node_id,
        &level,
        path,
        segment_start,
        segment_length,
        graph.nips_v * 2u + graph.nips_t);
    *end_level = level;
    return result;
}

bool ReferencePointActiveEdgePoint(
    const Graph_Compress& graph, unsigned int u, unsigned int v, unsigned int t) {
    const unsigned int total_level = graph.nips_v * 2u + graph.nips_t;
    unsigned int level = 0;
    unsigned int node = ReferenceWalkSegment(
        graph, 0, &level, u, 0, graph.nips_v, total_level);
    if (level == total_level) {
        return node != 0;
    }
    node = ReferenceWalkSegment(
        graph, node, &level, v, graph.nips_v, graph.nips_v, total_level);
    if (level == total_level) {
        return node != 0;
    }
    node = ReferenceWalkSegment(
        graph, node, &level, t, graph.nips_v * 2u, graph.nips_t, total_level);
    return level == total_level && node != 0;
}

static bool ReferenceAllAssignmentParity(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int domain_level,
    unsigned int total_level) {
    if (domain_level == total_level) {
        return node != 0u;
    }
    if (node_level > domain_level) {
        return false;
    }
    if (node_level < domain_level || node_level > total_level) {
        return false;
    }

    unsigned int low_level = node_level;
    unsigned int high_level = node_level;
    const unsigned int low =
        ReferenceUnpackOne(graph, node << 1, &low_level);
    const unsigned int high =
        ReferenceUnpackOne(graph, (node << 1) | 1u, &high_level);
    return ReferenceAllAssignmentParity(
        graph, low, low_level, domain_level + 1u, total_level) ^
        ReferenceAllAssignmentParity(
            graph, high, high_level, domain_level + 1u, total_level);
}

static bool ReferencePrefixAssignmentParity(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int domain_level,
    unsigned int time_start_level,
    unsigned int total_level,
    unsigned int time) {
    if (domain_level == total_level) {
        return node != 0u;
    }
    if (node_level < domain_level || node_level > total_level) {
        return false;
    }

    const unsigned int bit_index = domain_level - time_start_level;
    const unsigned int bit =
        (time >> (graph.nips_t - bit_index - 1u)) & 1u;
    if (node_level > domain_level) {
        if (bit == 0u) {
            return ReferencePrefixAssignmentParity(
                graph, node, node_level, domain_level + 1u,
                time_start_level, total_level, time);
        }
        return ReferenceAllAssignmentParity(
            graph, node, node_level, domain_level + 1u, total_level) ^
            ReferencePrefixAssignmentParity(
                graph, node, node_level, domain_level + 1u,
                time_start_level, total_level, time);
    }

    unsigned int low_level = node_level;
    const unsigned int low =
        ReferenceUnpackOne(graph, node << 1, &low_level);
    if (bit == 0u) {
        return ReferencePrefixAssignmentParity(
            graph, low, low_level, domain_level + 1u,
            time_start_level, total_level, time);
    }

    unsigned int high_level = node_level;
    const unsigned int high =
        ReferenceUnpackOne(graph, (node << 1) | 1u, &high_level);
    return ReferenceAllAssignmentParity(
        graph, low, low_level, domain_level + 1u, total_level) ^
        ReferencePrefixAssignmentParity(
            graph, high, high_level, domain_level + 1u,
            time_start_level, total_level, time);
}

bool ReferenceCompareTimeParity(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int time) {
    const unsigned int total_level = graph.nips_v * 2u + graph.nips_t;
    const unsigned int segment_start = graph.nips_v * 2u;
    if (start_level < segment_start || start_level > total_level) {
        return false;
    }
    return ReferencePrefixAssignmentParity(
        graph, node_id, start_level, segment_start,
        segment_start, total_level, time);
}

static bool ReferencePrefixAssignmentAny(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int domain_level,
    unsigned int time_start_level,
    unsigned int total_level,
    unsigned int time) {
    if (domain_level == total_level || node_level == total_level) {
        return node != 0u;
    }
    if (node_level < domain_level || node_level > total_level) {
        return false;
    }

    const unsigned int bit_index = domain_level - time_start_level;
    const unsigned int bit =
        (time >> (graph.nips_t - bit_index - 1u)) & 1u;
    if (node_level > domain_level) {
        return bit != 0u
            ? node_level != total_level || node != 0u
            : ReferencePrefixAssignmentAny(
                graph, node, node_level, domain_level + 1u,
                time_start_level, total_level, time);
    }

    unsigned int low_level = node_level;
    const unsigned int low =
        ReferenceUnpackOne(graph, node << 1, &low_level);
    if (bit == 0u) {
        return ReferencePrefixAssignmentAny(
            graph, low, low_level, domain_level + 1u,
            time_start_level, total_level, time);
    }
    if (low_level != total_level || low != 0u) {
        return true;
    }

    unsigned int high_level = node_level;
    const unsigned int high =
        ReferenceUnpackOne(graph, (node << 1) | 1u, &high_level);
    return ReferencePrefixAssignmentAny(
        graph, high, high_level, domain_level + 1u,
        time_start_level, total_level, time);
}

bool ReferenceCompareTime(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int time) {
    const unsigned int total_level = graph.nips_v * 2u + graph.nips_t;
    const unsigned int segment_start = graph.nips_v * 2u;
    if (start_level < segment_start || start_level > total_level) {
        return false;
    }
    return ReferencePrefixAssignmentAny(
        graph, node_id, start_level, segment_start,
        segment_start, total_level, time);
}

int* ReferenceSameBitRankArray(
    const BitArray* array,
    const int* positions,
    unsigned int count,
    unsigned int level) {
    int* output = static_cast<int*>(std::malloc(sizeof(int) * count));
    if (output == nullptr) {
        return nullptr;
    }
    const unsigned int flag = array->Flag[level];
    const std::uint64_t start = level == 0 ? 0 : array->Index[level - 1];
    for (unsigned int i = 0; i < count; ++i) {
        if (positions[i] < 0) {
            output[i] = -1;
        }
        else if (flag == 0) {
            output[i] = 0;
        }
        else if (flag == 1) {
            output[i] = positions[i] + 1;
        }
        else {
            output[i] = static_cast<int>(ReferenceBitPopCount(
                array->List, start, start + static_cast<unsigned int>(positions[i])));
        }
    }
    return output;
}

bool VerifyAcceleratedCore(
    const Graph_Compress& graph, unsigned int samples_per_level) {
    const QueryAccelerationCache* cache = FindQueryAcceleration(graph);
    unsigned int test_words[9] = {
        0x00000000u, 0xFFFFFFFFu, 0xAAAAAAAAu, 0x55555555u,
        0x01234567u, 0x89ABCDEFu, 0x80000001u, 0x7FFFFFFEu,
        0x13579BDFu
    };
    for (std::uint64_t first = 0; first < 257; ++first) {
        for (std::uint64_t last = first; last < 257; ++last) {
            if (AcceleratedBitPopCount(test_words, first, last) !=
                ReferenceBitPopCount(test_words, first, last)) {
                std::fprintf(stderr, "BitPopCount verification failed at %llu..%llu\n",
                    static_cast<unsigned long long>(first),
                    static_cast<unsigned long long>(last));
                return false;
            }
        }
    }

    unsigned int fix_words[3] = {
        0x89ABCDEFu, 0x01234567u, 0xA5A55A5Au
    };
    std::uint64_t fix_index[1] = { 16u };
    FixBitArray fix_array{ fix_words, fix_index, 5u, 16u };
    int fix_queries[8] = { 16, 1, 8, -1, 3, 15, 2, 9 };
    int* fix_results = AcceleratedFixBitsAccessArray(
        &fix_array, 0, fix_queries, 8u);
    if (fix_results == nullptr) {
        std::fprintf(stderr, "FixBits batch verification allocation failed\n");
        return false;
    }
    for (unsigned int i = 0; i < 8u; ++i) {
        const int expected = fix_queries[i] < 0
            ? -1
            : static_cast<int>(ReferenceFixBitsAccess(
                &fix_array, 0, static_cast<unsigned int>(fix_queries[i] - 1)));
        if (fix_results[i] != expected) {
            std::fprintf(stderr, "FixBits batch verification failed at query=%d\n",
                fix_queries[i]);
            std::free(fix_results);
            return false;
        }
    }
    std::free(fix_results);

    const unsigned int total_level = graph.nips_v * 2u + graph.nips_t;
    for (unsigned int level = 0; level < total_level; ++level) {
        const std::uint64_t fix_start = level == 0
            ? 0
            : graph.SkipValue->Index[level - 1u];
        const std::uint64_t fix_end = graph.SkipValue->Index[level];
        if (fix_start < fix_end) {
            const unsigned int last = static_cast<unsigned int>(fix_end - fix_start - 1u);
            const unsigned int indices[3] = { 0u, last / 2u, last };
            for (unsigned int index : indices) {
                if (FixBitsAccessFast(graph.SkipValue, level, index) !=
                    ReferenceFixBitsAccess(graph.SkipValue, level, index)) {
                    std::fprintf(stderr,
                        "FixBits verification failed at level=%u index=%u\n",
                        level, index);
                    return false;
                }
            }
        }

        const std::uint64_t rice_start = level == 0
            ? 0
            : graph.ShareValue->quoIndex[level - 1u];
        const std::uint64_t rice_end = graph.ShareValue->quoIndex[level];
        if (rice_start < rice_end) {
            const unsigned int value_count = AcceleratedBitPopCount(
                graph.ShareValue->quotient, rice_start, rice_end - 1u);
            if (value_count != 0) {
                const unsigned int indices[3] = {
                    0u, (value_count - 1u) / 2u, value_count - 1u };
                for (unsigned int index : indices) {
                    if (RiceAccessFast(
                        graph.ShareValue,
                        cache == nullptr ? nullptr : &cache->riceSelect,
                        level, index) !=
                        ReferenceRiceAccess(graph.ShareValue, level, index)) {
                        std::fprintf(stderr,
                            "Rice verification failed at level=%u index=%u\n",
                            level, index);
                        return false;
                    }
                }
            }
        }

        if (graph.SkipFlagList->Flag[level] == 0u) {
            const std::uint64_t run_start = level == 0
                ? 0
                : graph.SkipFlagList->Index[level - 1u];
            const std::uint64_t run_end = graph.SkipFlagList->Index[level];
            FastBitReader reader{ graph.SkipFlagList->List, run_start };
            unsigned int decoded_count = 0;
            while (reader.position < run_end) {
                decoded_count += ReadGamma(&reader);
            }
            if (decoded_count != 0) {
                const unsigned int positions[3] = {
                    0u, (decoded_count - 1u) / 2u, decoded_count - 1u };
                RunGammaForwardCursor cursor =
                    MakeRunGammaForwardCursor(graph.SkipFlagList, level);
                for (unsigned int position : positions) {
                    const AcceleratedBitRank fast = RunGammaBitAndRank(
                        graph.SkipFlagList, nullptr, level, position);
                    const AcceleratedBitRank forward =
                        RunGammaBitAndRankForward(
                            graph.SkipFlagList,
                            cache == nullptr ? nullptr : &cache->skipFlagGamma,
                            level, position, &cursor);
                    const AcceleratedBitRank reference = ReferenceRunGammaBitAndRank(
                        graph.SkipFlagList, level, position);
                    if (fast.bit != reference.bit || fast.rank != reference.rank ||
                        forward.bit != reference.bit ||
                        forward.rank != reference.rank) {
                        std::fprintf(stderr,
                            "RunGamma verification failed at level=%u position=%u\n",
                            level, position);
                        return false;
                    }
                }
            }
        }

        const unsigned int parent_count = level == 0
            ? 1u
            : graph.VariatyNumList[level] - graph.VariatyNumList[level - 1u];
        const std::uint64_t stored_start = level == 0
            ? 0
            : graph.NormalFlagList->Index[level - 1u];
        const std::uint64_t stored_end = graph.NormalFlagList->Index[level];
        const unsigned int position_count =
            graph.NormalFlagList->Flag[level] == 2u
            ? static_cast<unsigned int>(stored_end - stored_start)
            : parent_count * 2u;
        if (position_count == 0) {
            continue;
        }
        const unsigned int bounded_count = (std::min)(position_count, 4096u);
        const unsigned int sample_count = (std::min)(bounded_count, samples_per_level);
        const auto verify_position = [&](unsigned int position) {
            unsigned int fast_level = level;
            unsigned int reference_level = level;
            const unsigned int fast_node = AcceleratedUnpackOne(
                graph, position, &fast_level);
            const unsigned int reference_node = ReferenceUnpackOne(
                graph, position, &reference_level);
            if (fast_node == reference_node && fast_level == reference_level) {
                return true;
            }
            std::fprintf(stderr,
                "UnpackOne verification failed at level=%u position=%u "
                "fast=(%u,%u) reference=(%u,%u)\n",
                level, position, fast_node, fast_level,
                reference_node, reference_level);
            return false;
        };
        for (unsigned int sample = 0; sample < sample_count; ++sample) {
            const unsigned int position = sample_count == 1
                ? 0u
                : static_cast<unsigned int>(
                    static_cast<std::uint64_t>(sample) * (bounded_count - 1u) /
                    (sample_count - 1u));
            if (!verify_position(position)) {
                return false;
            }
        }
        if (position_count > bounded_count && !verify_position(position_count - 1u)) {
            return false;
        }

        const bool verify_batch_level = total_level <= 16u ||
            level == 0u || level + 1u == graph.nips_v ||
            level == graph.nips_v ||
            level + 1u == graph.nips_v * 2u ||
            level == graph.nips_v * 2u ||
            level + 1u == total_level;
        const unsigned int batch_sizes[4] = { 1u, 7u, 8u, 32u };
        for (unsigned int batch_index = 0u;
            verify_batch_level && batch_index < 4u; ++batch_index) {
            const unsigned int batch_size = batch_sizes[batch_index];
            const unsigned int sampled_parents =
                (std::min)(parent_count, 4096u);
            std::vector<Node> nodes(batch_size);
            for (unsigned int index = 0u; index < batch_size; ++index) {
                unsigned int parent = static_cast<unsigned int>(
                    (static_cast<std::uint64_t>(index) * 2654435761ull +
                        batch_size) % sampled_parents);
                if (index + 1u == batch_size && batch_size > 1u) {
                    parent = nodes[0].parent.id;
                }
                nodes[index].parent.id = parent;
                nodes[index].parent.level = level;
            }
            MinHeap heap{};
            heap.data = nodes.data();
            heap.size = static_cast<int>(batch_size);
            heap.capacity = static_cast<int>(batch_size);
            AcceleratedUnpackAll(graph, &heap);
            for (unsigned int index = 0u; index < batch_size; ++index) {
                unsigned int low_level = level;
                unsigned int high_level = level;
                const unsigned int low = ReferenceUnpackOne(
                    graph, nodes[index].parent.id << 1, &low_level);
                const unsigned int high = ReferenceUnpackOne(
                    graph, (nodes[index].parent.id << 1) | 1u, &high_level);
                if (nodes[index].left.id != low ||
                    nodes[index].left.level != low_level ||
                    nodes[index].right.id != high ||
                    nodes[index].right.level != high_level) {
                    std::fprintf(stderr,
                        "UnpackAll verification failed at level=%u batch=%u index=%u\n",
                        level, batch_size, index);
                    return false;
                }
            }
        }
    }
    return true;
}
