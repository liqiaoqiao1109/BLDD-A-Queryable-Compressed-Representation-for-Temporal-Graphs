#include "Graph.h"
#include "Help.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <vector>

namespace {

volatile uint64_t utv_query_sink = 0;

unsigned int WalkValue(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int* node_level,
    uint64_t value,
    unsigned int segment_start,
    unsigned int segment_length)
{
    const unsigned int total_level = graph.nips_v * 2u + graph.nips_t;
    for (unsigned int offset = 0; offset < segment_length; ++offset) {
        if (*node_level == total_level) break;
        const unsigned int level = segment_start + offset;
        if (*node_level > level) continue;
        if (*node_level < level) return 0u;
        const unsigned int bit =
            (value >> (segment_length - offset - 1u)) & 1u;
        node = UnpackOne(graph, (node << 1u) | bit, node_level);
    }
    return node;
}

unsigned int ResolveSource(
    const Graph_Compress& graph, uint64_t source, unsigned int* level)
{
    *level = 0u;
    return WalkValue(graph, 0u, level, source, 0u, graph.nips_v);
}

bool TargetValue(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    uint64_t target)
{
    const unsigned int time_end = graph.nips_v + graph.nips_t;
    const unsigned int total_level = time_end + graph.nips_v;
    if (node_level < time_end || node_level > total_level) return false;
    node = WalkValue(graph, node, &node_level, target,
        time_end, graph.nips_v);
    return node_level == total_level && node != 0u;
}

bool AllTimeParity(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int time_level,
    uint64_t target);

bool PrefixTimeParity(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int time_level,
    uint64_t target,
    uint64_t time);

bool AllTimeAny(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int time_level,
    uint64_t target);

bool PrefixTimeAny(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int time_level,
    uint64_t target,
    uint64_t time);

bool ActiveByType(
    const Graph_Compress& graph,
    unsigned int source_node,
    unsigned int source_level,
    uint64_t target,
    uint64_t time,
    DataType dataset)
{
    if (dataset == DATA_POINT) {
        unsigned int level = source_level;
        unsigned int node = WalkValue(graph, source_node, &level, time,
            graph.nips_v, graph.nips_t);
        node = WalkValue(graph, node, &level, target,
            graph.nips_v + graph.nips_t, graph.nips_v);
        return node != 0u;
    }
    return dataset == DATA_INTERVAL
        ? PrefixTimeParity(graph, source_node, source_level,
            graph.nips_v, target, time)
        : PrefixTimeAny(graph, source_node, source_level,
            graph.nips_v, target, time);
}

bool AllTimeParity(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int time_level,
    uint64_t target)
{
    const unsigned int time_end = graph.nips_v + graph.nips_t;
    if (time_level == time_end)
        return TargetValue(graph, node, node_level, target);
    if (node_level != time_level) return false;

    unsigned int low_level = node_level;
    unsigned int high_level = node_level;
    const unsigned int low = UnpackOne(graph, node << 1u, &low_level);
    const unsigned int high = UnpackOne(graph, (node << 1u) | 1u, &high_level);
    return AllTimeParity(graph, low, low_level, time_level + 1u, target) !=
        AllTimeParity(graph, high, high_level, time_level + 1u, target);
}

bool PrefixTimeParity(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int time_level,
    uint64_t target,
    uint64_t time)
{
    const unsigned int time_end = graph.nips_v + graph.nips_t;
    const unsigned int total_level = time_end + graph.nips_v;
    if (time_level == time_end) {
        return TargetValue(graph, node, node_level, target);
    }
    if (node_level < time_level || node_level > total_level) return false;

    const unsigned int bit_offset = time_level - graph.nips_v;
    const unsigned int bound_bit =
        (time >> (graph.nips_t - bit_offset - 1u)) & 1u;
    if (node_level > time_level) {
        if (bound_bit == 0u) {
            return PrefixTimeParity(graph, node, node_level,
                time_level + 1u, target, time);
        }
        return AllTimeParity(graph, node, node_level,
            time_level + 1u, target) !=
            PrefixTimeParity(graph, node, node_level,
                time_level + 1u, target, time);
    }

    unsigned int low_level = node_level;
    unsigned int high_level = node_level;
    const unsigned int low = UnpackOne(graph, node << 1u, &low_level);
    const unsigned int high = UnpackOne(graph, (node << 1u) | 1u, &high_level);
    if (bound_bit == 0u) {
        return PrefixTimeParity(graph, low, low_level,
            time_level + 1u, target, time);
    }
    return AllTimeParity(graph, low, low_level,
        time_level + 1u, target) !=
        PrefixTimeParity(graph, high, high_level,
            time_level + 1u, target, time);
}

bool AllTimeAny(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int time_level,
    uint64_t target)
{
    const unsigned int time_end = graph.nips_v + graph.nips_t;
    const unsigned int total_level = time_end + graph.nips_v;
    if (time_level == time_end)
        return TargetValue(graph, node, node_level, target);
    if (node_level == total_level) return node != 0u;
    if (node_level < time_level || node_level > total_level) return false;
    if (node_level > time_level)
        return AllTimeAny(graph, node, node_level,
            time_level + 1u, target);

    unsigned int low_level = node_level;
    unsigned int high_level = node_level;
    const unsigned int low = UnpackOne(graph, node << 1u, &low_level);
    const unsigned int high = UnpackOne(graph, (node << 1u) | 1u, &high_level);
    return AllTimeAny(graph, low, low_level, time_level + 1u, target) ||
        AllTimeAny(graph, high, high_level, time_level + 1u, target);
}

bool PrefixTimeAny(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int time_level,
    uint64_t target,
    uint64_t time)
{
    const unsigned int time_end = graph.nips_v + graph.nips_t;
    const unsigned int total_level = time_end + graph.nips_v;
    if (time_level == time_end)
        return TargetValue(graph, node, node_level, target);
    if (node_level == total_level) return node != 0u;
    if (node_level < time_level || node_level > total_level) return false;

    const unsigned int bit_offset = time_level - graph.nips_v;
    const unsigned int bound_bit =
        (time >> (graph.nips_t - bit_offset - 1u)) & 1u;
    if (node_level > time_level) {
        if (bound_bit == 0u) {
            return PrefixTimeAny(graph, node, node_level,
                time_level + 1u, target, time);
        }
        return AllTimeAny(graph, node, node_level,
            time_level + 1u, target) ||
            PrefixTimeAny(graph, node, node_level,
                time_level + 1u, target, time);
    }

    unsigned int low_level = node_level;
    unsigned int high_level = node_level;
    const unsigned int low = UnpackOne(graph, node << 1u, &low_level);
    const unsigned int high = UnpackOne(graph, (node << 1u) | 1u, &high_level);
    if (bound_bit == 0u) {
        return PrefixTimeAny(graph, low, low_level,
            time_level + 1u, target, time);
    }
    return AllTimeAny(graph, low, low_level,
        time_level + 1u, target) ||
        PrefixTimeAny(graph, high, high_level,
            time_level + 1u, target, time);
}

bool FindFirstTimeAtOrAfter(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int time_level,
    uint64_t target,
    uint64_t lower_bound,
    bool tight,
    uint64_t prefix,
    uint64_t* result)
{
    const unsigned int time_end = graph.nips_v + graph.nips_t;
    const unsigned int total_level = time_end + graph.nips_v;
    if (node_level < time_level || node_level > total_level) return false;
    if (time_level == time_end) {
        if (!TargetValue(graph, node, node_level, target)) return false;
        *result = prefix;
        return true;
    }

    const unsigned int bit_offset = time_level - graph.nips_v;
    const unsigned int bit_index = graph.nips_t - bit_offset - 1u;
    const unsigned int lower_bit = tight
        ? static_cast<unsigned int>((lower_bound >> bit_index) & 1u)
        : 0u;
    for (unsigned int branch = lower_bit; branch <= 1u; ++branch) {
        unsigned int child_level = node_level;
        unsigned int child = node;
        if (node_level == time_level) {
            child = UnpackOne(graph, (node << 1u) | branch, &child_level);
        }
        const bool child_tight = tight && branch == lower_bit;
        if (!AllTimeAny(graph, child, child_level,
                time_level + 1u, target)) {
            continue;
        }
        if (FindFirstTimeAtOrAfter(graph, child, child_level,
                time_level + 1u, target, lower_bound, child_tight,
                (prefix << 1u) | branch, result)) {
            return true;
        }
    }
    return false;
}

void EnsureResultCapacity(NeighborResult64* result, uint64_t need)
{
    if (result->Capacity >= need) return;
    uint64_t capacity = result->Capacity == 0u ? 64u : result->Capacity;
    while (capacity < need) {
        if (capacity > UINT64_MAX / 2u) { capacity = need; break; }
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(uint64_t)) {
        std::fprintf(stderr, "UTV neighbor result exceeds addressable memory\n");
        std::exit(EXIT_FAILURE);
    }
    uint64_t* list = static_cast<uint64_t*>(realloc(
        result->List, static_cast<size_t>(capacity) * sizeof(uint64_t)));
    if (!list) {
        std::fprintf(stderr, "UTV neighbor result allocation failed\n");
        std::exit(EXIT_FAILURE);
    }
    result->List = list;
    result->Capacity = capacity;
}

void AppendRange(
    NeighborResult64* result,
    uint64_t begin,
    uint64_t end)
{
    if (begin > end) return;
    const uint64_t span = end - begin + 1u;
    if (span == 0u || result->Count > UINT64_MAX - span) {
        std::fprintf(stderr, "UTV neighbor result count overflow\n");
        std::exit(EXIT_FAILURE);
    }
    const uint64_t required = result->Count + span;
    EnsureResultCapacity(result, required);
    for (uint64_t value = begin;; ++value) {
        result->List[result->Count++] = value;
        if (value == end) break;
    }
}

void EnumerateTargets(
    const Graph_Compress& graph,
    unsigned int node,
    unsigned int node_level,
    unsigned int depth,
    uint64_t prefix,
    uint64_t nodenum,
    NeighborResult64* result)
{
    const unsigned int total_level = graph.nips_v * 2u + graph.nips_t;
    const unsigned int target_start = graph.nips_v + graph.nips_t;
    const unsigned int remaining = graph.nips_v - depth;
    const uint64_t begin = remaining == 64u ? 0u : prefix << remaining;
    if (begin > nodenum) return;
    if (node_level == total_level || depth == graph.nips_v) {
        if (node != 0u) {
            const uint64_t end = remaining == 0u ? begin
                : remaining == 64u ? UINT64_MAX
                : begin + (UINT64_C(1) << remaining) - 1u;
            AppendRange(result, begin, (std::min)(end,
                static_cast<uint64_t>(nodenum)));
        }
        return;
    }
    const unsigned int level = target_start + depth;
    if (node_level > level) {
        EnumerateTargets(graph, node, node_level, depth + 1u,
            prefix << 1u, nodenum, result);
        EnumerateTargets(graph, node, node_level, depth + 1u,
            (prefix << 1u) | 1u, nodenum, result);
        return;
    }
    if (node_level < level) return;
    unsigned int low_level = node_level;
    unsigned int high_level = node_level;
    const unsigned int low = UnpackOne(graph, node << 1u, &low_level);
    const unsigned int high = UnpackOne(graph, (node << 1u) | 1u, &high_level);
    EnumerateTargets(graph, low, low_level, depth + 1u,
        prefix << 1u, nodenum, result);
    EnumerateTargets(graph, high, high_level, depth + 1u,
        (prefix << 1u) | 1u, nodenum, result);
}

// Collect target functions for all event times up to the query time.  The
// interval case combines them with XOR; incremental graphs use their union.
// Merging identical states at each level avoids iterating over the node domain.
bool EnumerateTemporalNeighbors(
    const Graph_Compress& graph,
    unsigned int source_node,
    unsigned int source_level,
    uint64_t time,
    uint64_t nodenum,
    DataType dataset,
    NeighborResult64* out)
{
    const unsigned int time_start = graph.nips_v;
    const unsigned int time_end = time_start + graph.nips_t;
    const unsigned int total_level = time_end + graph.nips_v;
    if (source_level < time_start || source_level > total_level) return false;

    const bool use_parity = dataset == DATA_INTERVAL;
    using StateSet = std::unordered_set<uint64_t>;
    auto pack = [](unsigned int node, unsigned int level, bool tight) {
        return (static_cast<uint64_t>(node) << 9u) |
            (static_cast<uint64_t>(level) << 1u) | (tight ? 1u : 0u);
    };
    auto combine = [use_parity](StateSet& states, uint64_t key) {
        const auto inserted = states.insert(key);
        if (use_parity && !inserted.second) states.erase(inserted.first);
    };

    StateSet current;
    current.insert(pack(source_node, source_level, true));
    for (unsigned int level = time_start; level < time_end; ++level) {
        StateSet next;
        const unsigned int bound_bit = static_cast<unsigned int>(
            (time >> (time_end - level - 1u)) & 1u);
        for (uint64_t key : current) {
            const unsigned int node = static_cast<unsigned int>(key >> 9u);
            const unsigned int node_level = static_cast<unsigned int>((key >> 1u) & 255u);
            const bool tight = (key & 1u) != 0u;
            if (node_level < level || node_level > total_level) return false;
            if (node_level == total_level && node == 0u) continue;
            const unsigned int last_branch = tight ? bound_bit : 1u;
            for (unsigned int branch = 0u; branch <= last_branch; ++branch) {
                unsigned int child = node;
                unsigned int child_level = node_level;
                if (node_level == level) {
                    child = UnpackOne(graph, (node << 1u) | branch, &child_level);
                }
                if (child_level == total_level && child == 0u) continue;
                combine(next, pack(child, child_level, tight && branch == bound_bit));
            }
        }
        current.swap(next);
        if (current.empty()) return true;
    }

    StateSet roots;
    for (uint64_t key : current) combine(roots, key & ~UINT64_C(1));
    for (uint64_t key : roots) {
        const unsigned int node = static_cast<unsigned int>(key >> 9u);
        const unsigned int level = static_cast<unsigned int>((key >> 1u) & 255u);
        EnumerateTargets(graph, node, level, 0u, 0u, nodenum, out);
    }
    if (out->Count > 1u) {
        std::sort(out->List, out->List + static_cast<size_t>(out->Count));
        uint64_t write = 0u;
        for (uint64_t first = 0u; first < out->Count;) {
            uint64_t last = first + 1u;
            while (last < out->Count && out->List[last] == out->List[first]) ++last;
            if (!use_parity || ((last - first) & 1u) != 0u)
                out->List[write++] = out->List[first];
            first = last;
        }
        out->Count = write;
    }
    return true;
}

uint64_t InputChecksum(
    const std::vector<uint64_t>& sources,
    const std::vector<uint64_t>& targets,
    const std::vector<uint64_t>& times)
{
    uint64_t checksum = 1469598103934665603ull;
    for (size_t i = 0; i < sources.size(); ++i) {
        checksum = checksum * 1099511628211ull + sources[i] + 1u;
        checksum = checksum * 1099511628211ull + targets[i] + 1u;
        checksum = checksum * 1099511628211ull + times[i] + 1u;
    }
    return checksum;
}

Stats BuildStats(
    const std::vector<double>& round_times,
    uint64_t input_checksum,
    uint64_t checksum)
{
    Stats stats{};
    if (round_times.empty()) return stats;
    std::vector<double> sorted = round_times;
    std::sort(sorted.begin(), sorted.end());
    double total = 0.0;
    for (double value : round_times) total += value;
    stats.avg_ns = total / round_times.size();
    stats.min_ns = sorted.front();
    stats.max_ns = sorted.back();
    const size_t middle = sorted.size() / 2u;
    stats.median_ns = (sorted.size() & 1u) != 0u
        ? sorted[middle] : (sorted[middle - 1u] + sorted[middle]) / 2.0;
    stats.input_checksum = input_checksum;
    stats.checksum = checksum;
    return stats;
}

bool ValidDataset(DataType dataset)
{
    return dataset == DATA_POINT || dataset == DATA_INTERVAL ||
        dataset == DATA_INCREMENTAL;
}

uint64_t NeighborChecksum(const NeighborResult64& result)
{
    uint64_t checksum = result.Count;
    for (uint64_t i = 0; i < result.Count; ++i)
        checksum = checksum * 1099511628211ull + result.List[i] + 1u;
    return checksum;
}

} // namespace

bool PointActiveEdge_UTV(
    const Graph_Compress& graph,
    uint64_t u,
    uint64_t v,
    uint64_t t,
    uint64_t nodenum,
    uint64_t timenum,
    DataType dataset)
{
    if (!ValidDataset(dataset) || u > nodenum || v > nodenum || t > timenum ||
        graph.nips_v > 64u || graph.nips_t > 64u)
        return false;
    unsigned int level = 0u;
    const unsigned int source_node = ResolveSource(graph, u, &level);
    return ActiveByType(graph, source_node, level, v, t, dataset);
}

void QueryDirectNeighbor_UTV(
    const Graph_Compress& graph,
    uint64_t u,
    uint64_t t,
    uint64_t nodenum,
    uint64_t timenum,
    DataType dataset,
    NeighborResult64* out)
{
    if (out == nullptr) return;
    out->Count = 0u;
    if (!ValidDataset(dataset) || u > nodenum || t > timenum ||
        graph.nips_v > 64u || graph.nips_t > 64u) return;
    if (dataset == DATA_POINT) {
        unsigned int level = 0u;
        unsigned int node = ResolveSource(graph, u, &level);
        node = WalkValue(graph, node, &level, t,
            graph.nips_v, graph.nips_t);
        EnumerateTargets(graph, node, level, 0u, 0u, nodenum, out);
        return;
    }
    unsigned int source_level = 0u;
    const unsigned int source_node = ResolveSource(graph, u, &source_level);
    if (EnumerateTemporalNeighbors(graph, source_node, source_level,
            t, nodenum, dataset, out)) return;
    out->Count = 0u;
    for (uint64_t candidate = 0u;; ++candidate) {
        if (ActiveByType(graph, source_node, source_level,
                candidate, t, dataset)) {
            if (out->Count == UINT64_MAX) {
                std::fprintf(stderr, "UTV neighbor result count overflow\n");
                std::exit(EXIT_FAILURE);
            }
            EnsureResultCapacity(out, out->Count + 1u);
            out->List[out->Count++] = candidate;
        }
        if (candidate == nodenum) break;
    }
}

bool QueryNextAct_UTV(
    const Graph_Compress& graph,
    uint64_t u,
    uint64_t v,
    uint64_t t,
    uint64_t nodenum,
    uint64_t timenum,
    DataType dataset,
    uint64_t* next_time)
{
    if (next_time == nullptr) return false;
    *next_time = 0u;
    if (u > nodenum || v > nodenum || t > timenum || !ValidDataset(dataset))
        return false;
    if (graph.nips_v > 64u || graph.nips_t > 64u) return false;
    unsigned int source_level = 0u;
    const unsigned int source_node = ResolveSource(graph, u, &source_level);
    if (ActiveByType(graph, source_node, source_level, v, t, dataset)) {
        *next_time = t;
        return true;
    }
    // There can be no later valid time after the dataset's maximum.  In
    // particular, incrementing an all-ones bit pattern would wrap the search
    // back into the represented time domain.
    if (t >= timenum) return false;
    const uint64_t lower_bound = t + 1u;
    uint64_t candidate = 0u;
    if (!FindFirstTimeAtOrAfter(graph, source_node, source_level,
            graph.nips_v, v, lower_bound, true, 0u, &candidate) ||
        candidate > timenum) return false;
    *next_time = candidate;
    return true;
}

void InitNeighborResult64(NeighborResult64* out, uint64_t initial_capacity)
{
    if (out == nullptr) return;
    out->List = nullptr;
    out->Count = 0u;
    out->Capacity = 0u;
    if (initial_capacity != 0u) EnsureResultCapacity(out, initial_capacity);
}

void FreeNeighborResult64(NeighborResult64* out)
{
    if (out == nullptr) return;
    std::free(out->List);
    out->List = nullptr;
    out->Count = 0u;
    out->Capacity = 0u;
}

Stats benchmark_PointActiveEdge_UTV(
    const Graph_Compress& graph,
    uint64_t nodenum,
    uint64_t timenum,
    DataType dataset,
    size_t query_count,
    unsigned int rounds)
{
    if (!ValidDataset(dataset) || query_count == 0u || rounds == 0u) return Stats{};
    const size_t batches = static_cast<size_t>(rounds) + 1u;
    if (query_count > SIZE_MAX / batches) return Stats{};
    const size_t count = query_count * batches;
    std::mt19937_64 rng(12345u);
    std::uniform_int_distribution<uint64_t> node_dist(0u, nodenum);
    std::uniform_int_distribution<uint64_t> time_dist(0u, timenum);
    std::vector<uint64_t> sources(count), targets(count), times(count);
    for (size_t i = 0; i < count; ++i) {
        sources[i] = node_dist(rng);
        targets[i] = node_dist(rng);
        times[i] = time_dist(rng);
    }
    const uint64_t input_checksum = InputChecksum(sources, targets, times);
    uint64_t warmup = 0u;
    for (size_t i = 0; i < query_count; ++i)
        warmup += PointActiveEdge_UTV(
            graph, sources[i], targets[i], times[i], nodenum, timenum, dataset);
    utv_query_sink = warmup;

    std::vector<double> durations;
    durations.reserve(rounds);
    uint64_t checksum = 1469598103934665603ull;
    for (unsigned int round = 0; round < rounds; ++round) {
        uint64_t round_checksum = 0u;
        const size_t offset = (static_cast<size_t>(round) + 1u) * query_count;
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < query_count; ++i) {
            const size_t index = offset + i;
            round_checksum += PointActiveEdge_UTV(
                graph, sources[index], targets[index], times[index],
                nodenum, timenum, dataset);
        }
        const auto end = std::chrono::steady_clock::now();
        utv_query_sink = round_checksum;
        checksum = checksum * 1099511628211ull + round_checksum + round + 1u;
        durations.push_back(std::chrono::duration<double, std::nano>(end - start).count() /
            static_cast<double>(query_count));
    }
    return BuildStats(durations, input_checksum, checksum);
}

Stats benchmark_QueryDirectNeighbor_UTV(
    const Graph_Compress& graph,
    uint64_t nodenum,
    uint64_t timenum,
    DataType dataset,
    size_t query_count,
    unsigned int rounds)
{
    if (!ValidDataset(dataset) || query_count == 0u || rounds == 0u) return Stats{};
    const size_t batches = static_cast<size_t>(rounds) + 1u;
    if (query_count > SIZE_MAX / batches) return Stats{};
    const size_t count = query_count * batches;
    std::mt19937_64 rng(12345u);
    std::uniform_int_distribution<uint64_t> node_dist(0u, nodenum);
    std::uniform_int_distribution<uint64_t> time_dist(0u, timenum);
    std::vector<uint64_t> sources(count), targets(count, 0u), times(count);
    for (size_t i = 0; i < count; ++i) {
        sources[i] = node_dist(rng);
        times[i] = time_dist(rng);
    }
    const uint64_t input_checksum = InputChecksum(sources, targets, times);
    NeighborResult64 result{};
    InitNeighborResult64(&result, 64u);
    uint64_t warmup = 0u;
    for (size_t i = 0; i < query_count; ++i) {
        QueryDirectNeighbor_UTV(graph, sources[i], times[i],
            nodenum, timenum, dataset, &result);
        warmup += NeighborChecksum(result);
    }
    utv_query_sink = warmup;
    std::vector<double> durations;
    durations.reserve(rounds);
    uint64_t checksum = 1469598103934665603ull;
    for (unsigned int round = 0; round < rounds; ++round) {
        uint64_t round_checksum = 0u;
        const size_t offset = (static_cast<size_t>(round) + 1u) * query_count;
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < query_count; ++i) {
            const size_t index = offset + i;
            QueryDirectNeighbor_UTV(graph, sources[index], times[index],
                nodenum, timenum, dataset, &result);
            round_checksum += NeighborChecksum(result);
        }
        const auto end = std::chrono::steady_clock::now();
        utv_query_sink = round_checksum;
        checksum = checksum * 1099511628211ull + round_checksum + round + 1u;
        durations.push_back(std::chrono::duration<double, std::nano>(end - start).count() /
            static_cast<double>(query_count));
    }
    FreeNeighborResult64(&result);
    return BuildStats(durations, input_checksum, checksum);
}

Stats benchmark_QueryNextAct_UTV(
    const Graph_Compress& graph,
    uint64_t nodenum,
    uint64_t timenum,
    DataType dataset,
    size_t query_count,
    unsigned int rounds)
{
    if (!ValidDataset(dataset) || query_count == 0u || rounds == 0u) return Stats{};
    const size_t batches = static_cast<size_t>(rounds) + 1u;
    if (query_count > SIZE_MAX / batches) return Stats{};
    const size_t count = query_count * batches;
    std::mt19937_64 rng(12345u);
    std::uniform_int_distribution<uint64_t> node_dist(0u, nodenum);
    std::uniform_int_distribution<uint64_t> time_dist(0u, timenum);
    std::vector<uint64_t> sources(count), targets(count), times(count);
    for (size_t i = 0; i < count; ++i) {
        sources[i] = node_dist(rng);
        targets[i] = node_dist(rng);
        times[i] = time_dist(rng);
    }
    const uint64_t input_checksum = InputChecksum(sources, targets, times);
    uint64_t warmup = 1469598103934665603ull;
    for (size_t i = 0; i < query_count; ++i) {
        uint64_t result = 0u;
        const bool found = QueryNextAct_UTV(graph, sources[i], targets[i], times[i],
            nodenum, timenum, dataset, &result);
        warmup = (warmup ^ (found ? 1u : 0u)) * 1099511628211ull;
        warmup = (warmup ^ result) * 1099511628211ull;
    }
    utv_query_sink = warmup;
    std::vector<double> durations;
    durations.reserve(rounds);
    uint64_t checksum = 1469598103934665603ull;
    for (unsigned int round = 0; round < rounds; ++round) {
        uint64_t round_checksum = 1469598103934665603ull;
        const size_t offset = (static_cast<size_t>(round) + 1u) * query_count;
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < query_count; ++i) {
            const size_t index = offset + i;
            uint64_t result = 0u;
            const bool found = QueryNextAct_UTV(graph,
                sources[index], targets[index], times[index],
                nodenum, timenum, dataset, &result);
            round_checksum = (round_checksum ^ (found ? 1u : 0u)) *
                1099511628211ull;
            round_checksum = (round_checksum ^ result) * 1099511628211ull;
        }
        const auto end = std::chrono::steady_clock::now();
        utv_query_sink = round_checksum;
        checksum = checksum * 1099511628211ull + round_checksum + round + 1u;
        durations.push_back(std::chrono::duration<double, std::nano>(end - start).count() /
            static_cast<double>(query_count));
    }
    return BuildStats(durations, input_checksum, checksum);
}
