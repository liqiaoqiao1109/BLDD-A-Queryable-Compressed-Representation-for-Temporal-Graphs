#include"Graph.h"
#include"Help.h"
#include"Bit.h"
#include"QueryAccelerate.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <vector>


bool PointActiveEdge_Point(const Graph_Compress& G, unsigned int u, unsigned int v, unsigned int t) {
#if defined(QUERYBDD_REFERENCE_BUILD)
    return ReferencePointActiveEdgePoint(G, u, v, t);
#else
    return AcceleratedPointActiveEdgePoint(G, u, v, t);
#endif
}


bool PointActiveEdge_Interval(const Graph_Compress& G, unsigned int u, unsigned int v, unsigned int t) {
#if !defined(QUERYBDD_REFERENCE_BUILD)
    return AcceleratedPointActiveEdgeInterval(G, u, v, t);
#else
    unsigned int nodeu, nodev;
    unsigned int level = 0;

    nodeu = GetDirectNode(G, 0, 0, u, 1, &level);
    if (level == G.nips_v * 2 + G.nips_t) {
        return nodeu != 0 &&
            ((static_cast<std::uint64_t>(t) + 1u) & 1u) != 0u;
    }

    nodev = GetDirectNode(G, nodeu, level, v, 2, &level);
    if (level == G.nips_v * 2 + G.nips_t) {
        return nodev != 0 &&
            ((static_cast<std::uint64_t>(t) + 1u) & 1u) != 0u;
    }

    return CompareTimeParity(G, nodev, level, t);
#endif
}


bool PointActiveEdge_Incremental(const Graph_Compress& G, unsigned int u, unsigned int v, unsigned int t) {
#if !defined(QUERYBDD_REFERENCE_BUILD)
    return AcceleratedPointActiveEdgeIncremental(G, u, v, t);
#else
    unsigned int nodeu, nodev;
    unsigned int level = 0;

    nodeu = GetDirectNode(G, 0, 0, u, 1, &level);
    if (level == G.nips_v * 2 + G.nips_t) {
        return nodeu != 0;
    }

    nodev = GetDirectNode(G, nodeu, level, v, 2, &level);
    if (level == G.nips_v * 2 + G.nips_t) {
        return nodev != 0;
    }

    return CompareTime(G, nodev, level, t);
#endif
}

namespace {
volatile std::uint64_t benchmark_sink = 0;

// A fixed seed and checksums make benchmark inputs and results reproducible.
template <typename Query>
Stats RunBenchmark(
    const Graph_Compress& graph,
    const std::vector<unsigned int>& sources,
    const std::vector<unsigned int>& targets,
    const std::vector<unsigned int>& timestamps,
    std::size_t query_count,
    unsigned int rounds,
    Query query) {
    std::uint64_t warmup_checksum = 0;
    for (std::size_t i = 0; i < query_count; ++i) {
        warmup_checksum += query(
            graph, sources[i], targets[i], timestamps[i]);
    }
    benchmark_sink = warmup_checksum;

    std::vector<double> round_times_ns;
    round_times_ns.reserve(rounds);
    std::uint64_t aggregate_checksum = 1469598103934665603ull;
    for (unsigned int round = 0; round < rounds; ++round) {
        std::uint64_t round_checksum = 0;
        const std::size_t offset =
            (static_cast<std::size_t>(round) + 1u) * query_count;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < query_count; ++i) {
            const std::size_t query_index = offset + i;
            round_checksum += query(
                graph,
                sources[query_index],
                targets[query_index],
                timestamps[query_index]);
        }
        const auto end = std::chrono::steady_clock::now();
        benchmark_sink = round_checksum;
        aggregate_checksum = aggregate_checksum * 1099511628211ull +
            round_checksum + round + 1u;

        const double elapsed_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        const double average_ns = elapsed_ns / static_cast<double>(query_count);
        round_times_ns.push_back(average_ns);
    }

    double total_ns = 0.0;
    for (double round_time_ns : round_times_ns) {
        total_ns += round_time_ns;
    }
    std::vector<double> sorted_times = round_times_ns;
    std::sort(sorted_times.begin(), sorted_times.end());

    Stats stats{};
    stats.avg_ns = total_ns / static_cast<double>(round_times_ns.size());
    stats.min_ns = sorted_times.front();
    stats.max_ns = sorted_times.back();
    const std::size_t middle = sorted_times.size() / 2u;
    stats.median_ns = (sorted_times.size() & 1u) != 0
        ? sorted_times[middle]
        : (sorted_times[middle - 1u] + sorted_times[middle]) / 2.0;
    stats.checksum = aggregate_checksum;
    return stats;
}
}

Stats benchmark_PointActiveEdge(
    const Graph_Compress& G,
    unsigned int nodenum,
    unsigned int timenum,
    DataType dataset,
    std::size_t query_count,
    unsigned int rounds) {
    if (query_count == 0 || rounds == 0) {
        std::fprintf(stderr, "Error: query_count and rounds must be greater than zero\n");
        return Stats{};
    }
    const std::size_t batch_count = static_cast<std::size_t>(rounds) + 1u;
    if (query_count > SIZE_MAX / batch_count) {
        std::fprintf(stderr, "Error: benchmark query count overflow\n");
        return Stats{};
    }
    const std::size_t total_query_count = query_count * batch_count;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<unsigned int> node_dist(0, nodenum);
    std::uniform_int_distribution<unsigned int> time_dist(0, timenum);

    std::vector<unsigned int> sources(total_query_count);
    std::vector<unsigned int> targets(total_query_count);
    std::vector<unsigned int> times(total_query_count);
    for (std::size_t i = 0; i < total_query_count; ++i) {
        sources[i] = node_dist(rng);
        targets[i] = node_dist(rng);
        times[i] = time_dist(rng);
    }

    switch (dataset) {
    case DATA_POINT:
        return RunBenchmark(
            G, sources, targets, times, query_count, rounds,
            [](const Graph_Compress& graph, unsigned int u, unsigned int v, unsigned int t) {
                return PointActiveEdge_Point(graph, u, v, t);
            });
    case DATA_INTERVAL:
        return RunBenchmark(
            G, sources, targets, times, query_count, rounds,
            [](const Graph_Compress& graph, unsigned int u, unsigned int v, unsigned int t) {
                return PointActiveEdge_Interval(graph, u, v, t);
            });
    case DATA_INCREMENTAL:
        return RunBenchmark(
            G, sources, targets, times, query_count, rounds,
            [](const Graph_Compress& graph, unsigned int u, unsigned int v, unsigned int t) {
                return PointActiveEdge_Incremental(graph, u, v, t);
            });
    default:
        std::fprintf(stderr, "Error: Unknown dataset type\n");
        return Stats{};
    }
}
