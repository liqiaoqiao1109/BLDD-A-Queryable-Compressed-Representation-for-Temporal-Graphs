#include "Graph.h"
#include "Help.h"
#include "Bit.h"
#include "QueryAccelerate.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

volatile std::uint64_t next_act_benchmark_sink = 0u;

// Benchmarks use deterministic batches so results can be compared across builds.

bool ValidDataset(DataType dataset)
{
    return dataset == DATA_POINT || dataset == DATA_INTERVAL ||
        dataset == DATA_INCREMENTAL;
}

unsigned int QueryNextActNoCheck(
    const Graph_Compress& graph,
    unsigned int u,
    unsigned int v,
    unsigned int t,
    unsigned int timenum,
    DataType dataset)
{
    unsigned int accelerated = QUERY_TIME_INF;
    if (AcceleratedQueryNextAct(
        graph, u, v, t, timenum, dataset, &accelerated)) {
        return accelerated;
    }
    for (std::uint64_t candidate = t; candidate <= timenum; ++candidate) {
        if (QueryEdgeActiveByType(
            graph,
            u,
            v,
            static_cast<unsigned int>(candidate),
            dataset)) {
            return static_cast<unsigned int>(candidate);
        }
    }
    return QUERY_TIME_INF;
}

unsigned int QueryNextActChecked(
    const Graph_Compress& graph,
    unsigned int u,
    unsigned int v,
    unsigned int t,
    unsigned int nodenum,
    unsigned int timenum,
    DataType dataset)
{
    if (u > nodenum || v > nodenum) {
        std::fprintf(stderr, "Error: Invalid Node\n");
        return QUERY_TIME_INF;
    }
    if (t > timenum) {
        std::fprintf(stderr, "Error: Invalid Time\n");
        return QUERY_TIME_INF;
    }
    if (!ValidDataset(dataset)) {
        std::fprintf(stderr, "Error: Unknown dataset type\n");
        return QUERY_TIME_INF;
    }
    return QueryNextActNoCheck(graph, u, v, t, timenum, dataset);
}

Stats BuildStats(
    const std::vector<double>& round_times,
    std::uint64_t input_checksum,
    std::uint64_t checksum)
{
    Stats stats{};
    if (round_times.empty()) {
        return stats;
    }
    std::vector<double> sorted = round_times;
    std::sort(sorted.begin(), sorted.end());
    double total = 0.0;
    for (double value : round_times) {
        total += value;
    }
    stats.avg_ns = total / static_cast<double>(round_times.size());
    stats.min_ns = sorted.front();
    stats.max_ns = sorted.back();
    const std::size_t middle = sorted.size() / 2u;
    stats.median_ns = (sorted.size() & 1u) != 0u
        ? sorted[middle]
        : (sorted[middle - 1u] + sorted[middle]) / 2.0;
    stats.input_checksum = input_checksum;
    stats.checksum = checksum;
    return stats;
}

}

unsigned int QueryNextAct_Point(
    const Graph_Compress& graph,
    unsigned int u,
    unsigned int v,
    unsigned int t,
    unsigned int nodenum,
    unsigned int timenum)
{
    return QueryNextActChecked(
        graph, u, v, t, nodenum, timenum, DATA_POINT);
}

unsigned int QueryNextAct_Interval(
    const Graph_Compress& graph,
    unsigned int u,
    unsigned int v,
    unsigned int t,
    unsigned int nodenum,
    unsigned int timenum)
{
    return QueryNextActChecked(
        graph, u, v, t, nodenum, timenum, DATA_INTERVAL);
}

unsigned int QueryNextAct_Incremental(
    const Graph_Compress& graph,
    unsigned int u,
    unsigned int v,
    unsigned int t,
    unsigned int nodenum,
    unsigned int timenum)
{
    return QueryNextActChecked(
        graph, u, v, t, nodenum, timenum, DATA_INCREMENTAL);
}

Stats benchmark_QueryNextAct(
    const Graph_Compress& graph,
    unsigned int nodenum,
    unsigned int timenum,
    DataType dataset,
    std::size_t query_count,
    unsigned int rounds)
{
    if (!ValidDataset(dataset)) {
        std::fprintf(stderr, "Error: Unknown dataset type\n");
        return Stats{};
    }
    if (query_count == 0u || rounds == 0u) {
        std::fprintf(stderr,
            "Error: query_count and rounds must be greater than zero\n");
        return Stats{};
    }
    const std::size_t batches = static_cast<std::size_t>(rounds) + 1u;
    if (query_count > SIZE_MAX / batches) {
        std::fprintf(stderr, "Error: benchmark query count overflow\n");
        return Stats{};
    }

    const std::size_t total_queries = query_count * batches;
    std::mt19937 rng(12345u);
    std::uniform_int_distribution<unsigned int> node_dist(0u, nodenum);
    std::uniform_int_distribution<unsigned int> time_dist(0u, timenum);
    std::vector<unsigned int> sources(total_queries);
    std::vector<unsigned int> targets(total_queries);
    std::vector<unsigned int> times(total_queries);
    std::uint64_t input_checksum = 1469598103934665603ull;
    for (std::size_t index = 0u; index < total_queries; ++index) {
        sources[index] = node_dist(rng);
        targets[index] = node_dist(rng);
        times[index] = time_dist(rng);
        input_checksum = (input_checksum ^ sources[index]) *
            1099511628211ull;
        input_checksum = (input_checksum ^ targets[index]) *
            1099511628211ull;
        input_checksum = (input_checksum ^ times[index]) *
            1099511628211ull;
    }

    std::uint64_t warmup = 0u;
    for (std::size_t index = 0u; index < query_count; ++index) {
        warmup = (warmup ^ QueryNextActNoCheck(
            graph,
            sources[index],
            targets[index],
            times[index],
            timenum,
            dataset)) * 1099511628211ull;
    }
    next_act_benchmark_sink = warmup;

    std::vector<double> round_times;
    round_times.reserve(rounds);
    std::uint64_t aggregate = 1469598103934665603ull;
    for (unsigned int round = 0u; round < rounds; ++round) {
        const std::size_t offset =
            (static_cast<std::size_t>(round) + 1u) * query_count;
        std::uint64_t round_checksum = 1469598103934665603ull;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t index = 0u; index < query_count; ++index) {
            const std::size_t query_index = offset + index;
            const unsigned int result = QueryNextActNoCheck(
                graph,
                sources[query_index],
                targets[query_index],
                times[query_index],
                timenum,
                dataset);
            round_checksum = (round_checksum ^ result) * 1099511628211ull;
        }
        const auto end = std::chrono::steady_clock::now();
        next_act_benchmark_sink = round_checksum;
        aggregate = aggregate * 1099511628211ull + round_checksum + round + 1u;
        const double average_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count()) / static_cast<double>(query_count);
        round_times.push_back(average_ns);
    }
    return BuildStats(round_times, input_checksum, aggregate);
}
