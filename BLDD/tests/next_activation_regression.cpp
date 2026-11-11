#include "Graph.h"

#include <cstdio>

// The temporal function for edge (0, 0) is independent of its high time bit.
// Searching from time 2 must explore that skipped bit's higher branch to find 4.
int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    Graph_Compress* graph = LoadBitGraphFromTxt(InitGraphCompress(), 1u, 3u, argv[1]);
    if (graph == nullptr) return 2;

    const unsigned int point[] = {0u, 1u, 4u, 4u, 4u, 5u, QUERY_TIME_INF, QUERY_TIME_INF};
    const unsigned int interval[] = {0u, 4u, 4u, 4u, 4u,
        QUERY_TIME_INF, QUERY_TIME_INF, QUERY_TIME_INF};
    unsigned int failures = 0u;
    for (unsigned int time = 0; time < 8u; ++time) {
        const unsigned int actual_point = QueryNextAct_Point(*graph, 0u, 0u, time, 1u, 7u);
        const unsigned int actual_interval = QueryNextAct_Interval(*graph, 0u, 0u, time, 1u, 7u);
        const unsigned int actual_incremental = QueryNextAct_Incremental(*graph, 0u, 0u, time, 1u, 7u);
        const unsigned int absent = QueryNextAct_Interval(*graph, 0u, 1u, time, 1u, 7u);
        if (actual_point != point[time] || actual_interval != interval[time] ||
            actual_incremental != time || absent != QUERY_TIME_INF) {
            std::fprintf(stderr,
                "time=%u point=%u/%u interval=%u/%u incremental=%u/%u absent=%u\n",
                time, actual_point, point[time], actual_interval, interval[time],
                actual_incremental, time, absent);
            ++failures;
        }
    }
    FreeGraphCompress(graph);
    std::printf("Next-activation regression: %u failure(s)\n", failures);
    return failures == 0u ? 0 : 1;
}
