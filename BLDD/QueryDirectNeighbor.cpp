#include"Graph.h"
#include"Help.h"
#include"Bit.h"
#include"QueryAccelerate.h"

#include <algorithm>
#include <climits>
#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

namespace {

volatile std::uint64_t neighbor_benchmark_sink = 0;

// The reusable result buffer keeps allocation outside the measured query loop.
void EnsureNeighborCapacity(NeighborResult* out, unsigned int need)
{
    if (out->Capacity >= need) {
        return;
    }

    unsigned int new_capacity = out->Capacity == 0 ? 64u : out->Capacity;
    while (new_capacity < need) {
        if (new_capacity > UINT_MAX / 2u) {
            new_capacity = need;
            break;
        }
        new_capacity *= 2u;
    }

    unsigned int* new_list = static_cast<unsigned int*>(
        realloc(out->List, sizeof(unsigned int) *
            static_cast<std::size_t>(new_capacity)));
    if (new_list == NULL) {
        fprintf(stderr, "Neighbor result allocation failed\n");
        exit(EXIT_FAILURE);
    }
    out->List = new_list;
    out->Capacity = new_capacity;
}


bool EdgeActive(
    const Graph_Compress& G,
    unsigned int u,
    unsigned int v,
    unsigned int t,
    DataType dataset)
{
    switch (dataset) {
    case DATA_POINT:
        return PointActiveEdge_Point(G, u, v, t);
    case DATA_INTERVAL:
        return PointActiveEdge_Interval(G, u, v, t);
    case DATA_INCREMENTAL:
        return PointActiveEdge_Incremental(G, u, v, t);
    default:
        return false;
    }
}


void QueryDirectNeighborNoCheck(
    const Graph_Compress& G,
    unsigned int u,
    unsigned int t,
    unsigned int nodenum,
    DataType dataset,
    NeighborResult* out)
{
    out->Count = 0;
    if (AcceleratedQueryDirectNeighbors(
        G, u, t, nodenum, dataset, out)) {
        return;
    }

    unsigned int level = 0;
    const unsigned int node = GetDirectNode(G, 0, 0, u, 1, &level);
    if (CollectDirectNeighbors(
        G, node, level, t, nodenum, dataset, out)) {
        return;
    }

    out->Count = 0;
    static bool fallback_warning_printed = false;
    if (!fallback_warning_printed) {
        fprintf(stderr,
            "Warning: direct-neighbor collector failed; using per-edge fallback\n");
        fallback_warning_printed = true;
    }
    for (std::uint64_t candidate = 0; candidate <= nodenum; ++candidate) {
        const unsigned int v = static_cast<unsigned int>(candidate);
        if (!EdgeActive(G, u, v, t, dataset)) {
            continue;
        }
        if (out->Count == UINT_MAX) {
            fprintf(stderr, "Neighbor result count overflow\n");
            exit(EXIT_FAILURE);
        }
        EnsureNeighborCapacity(out, out->Count + 1u);
        out->List[out->Count++] = v;
    }
}


void QueryDirectNeighborChecked(
    const Graph_Compress& G,
    unsigned int u,
    unsigned int t,
    unsigned int nodenum,
    unsigned int timenum,
    DataType dataset,
    NeighborResult* out)
{
    if (out == NULL) {
        return;
    }
    out->Count = 0;
    if (u > nodenum) {
        fprintf(stderr, "Error: Invalid Node\n");
        return;
    }
    if (t > timenum) {
        fprintf(stderr, "Error: Invalid Time\n");
        return;
    }
    QueryDirectNeighborNoCheck(G, u, t, nodenum, dataset, out);
}

std::uint64_t NeighborChecksum(const NeighborResult& result)
{
    std::uint64_t checksum = result.Count;
    for (unsigned int i = 0; i < result.Count; ++i) {
        checksum = checksum * 1099511628211ull + result.List[i] + 1u;
    }
    return checksum;
}

}


void InitNeighborResult(NeighborResult* out, unsigned int initial_capacity)
{
    if (out == NULL) {
        return;
    }
    out->List = NULL;
    out->Count = 0;
    out->Capacity = 0;
    if (initial_capacity != 0) {
        EnsureNeighborCapacity(out, initial_capacity);
    }
}

void FreeNeighborResult(NeighborResult* out)
{
    if (out == NULL) {
        return;
    }
    free(out->List);
    out->List = NULL;
    out->Count = 0;
    out->Capacity = 0;
}


void QueryDirectNeighbor_Point(
    const Graph_Compress& G,
    unsigned int u,
    unsigned int t,
    unsigned int nodenum,
    unsigned int timenum,
    NeighborResult* out)
{
    QueryDirectNeighborChecked(
        G, u, t, nodenum, timenum, DATA_POINT, out);
}

void QueryDirectNeighbor_Interval(
    const Graph_Compress& G,
    unsigned int u,
    unsigned int t,
    unsigned int nodenum,
    unsigned int timenum,
    NeighborResult* out)
{
    QueryDirectNeighborChecked(
        G, u, t, nodenum, timenum, DATA_INTERVAL, out);
}

void QueryDirectNeighbor_Incremental(
    const Graph_Compress& G,
    unsigned int u,
    unsigned int t,
    unsigned int nodenum,
    unsigned int timenum,
    NeighborResult* out)
{
    QueryDirectNeighborChecked(
        G, u, t, nodenum, timenum, DATA_INCREMENTAL, out);
}


Stats benchmark_QueryDirectNeighbor(
    const Graph_Compress& G,
    unsigned int nodenum,
    unsigned int timenum,
    DataType dataset,
    std::size_t query_count,
    unsigned int rounds)
{
    if (query_count == 0 || rounds == 0) {
        fprintf(stderr,
            "Error: query_count and rounds must be greater than zero\n");
        return Stats{};
    }
    if (dataset != DATA_POINT && dataset != DATA_INTERVAL &&
        dataset != DATA_INCREMENTAL) {
        fprintf(stderr, "Error: Unknown dataset type\n");
        return Stats{};
    }
    const std::size_t batch_count = static_cast<std::size_t>(rounds) + 1u;
    if (query_count > SIZE_MAX / batch_count) {
        fprintf(stderr, "Error: benchmark query count overflow\n");
        return Stats{};
    }
    const std::size_t total_query_count = query_count * batch_count;

    std::mt19937 rng(12345u);
    std::uniform_int_distribution<unsigned int> node_dist(0, nodenum);
    std::uniform_int_distribution<unsigned int> time_dist(0, timenum);
    std::vector<unsigned int> sources(total_query_count);
    std::vector<unsigned int> times(total_query_count);
    for (std::size_t i = 0; i < total_query_count; ++i) {
        sources[i] = node_dist(rng);
        times[i] = time_dist(rng);
    }

    NeighborResult result{};
    InitNeighborResult(&result, 64u);
    std::uint64_t warmup_checksum = 0;
    for (std::size_t i = 0; i < query_count; ++i) {
        QueryDirectNeighborNoCheck(
            G, sources[i], times[i], nodenum, dataset, &result);
        warmup_checksum += NeighborChecksum(result);
    }
    neighbor_benchmark_sink = warmup_checksum;

    std::vector<double> round_times_ns;
    round_times_ns.reserve(rounds);
    std::uint64_t aggregate_checksum = 1469598103934665603ull;
    for (unsigned int round = 0; round < rounds; ++round) {
        std::uint64_t checksum = 0;
        const std::size_t offset =
            (static_cast<std::size_t>(round) + 1u) * query_count;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < query_count; ++i) {
            const std::size_t query_index = offset + i;
            QueryDirectNeighborNoCheck(
                G, sources[query_index], times[query_index],
                nodenum, dataset, &result);
            checksum += NeighborChecksum(result);
        }
        const auto end = std::chrono::steady_clock::now();
        neighbor_benchmark_sink = checksum;
        aggregate_checksum = aggregate_checksum * 1099511628211ull +
            checksum + round + 1u;
        round_times_ns.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count()) / static_cast<double>(query_count));
    }
    FreeNeighborResult(&result);

    std::vector<double> sorted_times = round_times_ns;
    std::sort(sorted_times.begin(), sorted_times.end());
    double total = 0.0;
    for (double value : round_times_ns) {
        total += value;
    }

    Stats stats{};
    stats.avg_ns = total / static_cast<double>(rounds);
    stats.min_ns = sorted_times.front();
    stats.max_ns = sorted_times.back();
    const std::size_t middle = sorted_times.size() / 2u;
    stats.median_ns = (sorted_times.size() & 1u) != 0
        ? sorted_times[middle]
        : (sorted_times[middle - 1u] + sorted_times[middle]) / 2.0;
    stats.checksum = aggregate_checksum;
    return stats;
}

