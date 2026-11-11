#include "Application.h"

int main(int argc, char* argv[])
{
    // Edit these two values for runs launched without command-line arguments.
    const ApplicationOptions defaults{
        RunMode::LoadQuery,
        DatasetId::Sample
    };

    // Benchmark interfaces: benchmark_PointActiveEdge, benchmark_QueryNextAct,
    // and benchmark_QueryDirectNeighbor. RunApplication invokes all three.
    return RunApplication(argc, argv, defaults);
}

