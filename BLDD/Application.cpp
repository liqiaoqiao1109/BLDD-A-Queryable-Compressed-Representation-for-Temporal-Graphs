#define _CRT_SECURE_NO_WARNINGS

#include "Application.h"

#include "Graph.h"
#include "GraphStorage.h"
#include "Help.h"
#include "cudd.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>

namespace {

const Dataset kDatasets[] = {
    {"data/yahoo-G5-sorted-1.txt", DATA_POINT,
        "output/UVT/yahoo", "UVT"},
    {"data/flickr-growth-sorted-p1.txt", DATA_INCREMENTAL,
        "output/UVT/flickr", "UVT"},
    {"data/edit-enwiki.txt", DATA_POINT,
        "output/UVT/edit", "UVT"},
    {"data/cbtPow-sorted-1.txt", DATA_INTERVAL,
        "output/UVT/cbtpow", "UVT"},
    {"data/cbtComm-sorted.txt", DATA_INTERVAL,
        "output/UVT/comm", "UVT"},
    {"data/wikilinks.txt", DATA_INTERVAL,
        "output/UVT/UVT", "UVT"},
    {"sample_data/sample-point.txt", DATA_POINT,
        "sample_data/compressed", "UVT"}
};

const char* const kUtvOutputDirectories[] = {
    "output/UTV/yahoo", "output/UTV/flickr",
    "output/UTV/edit", "output/UTV/cbtPow",
    "output/UTV/cbtComm", "output/UTV/link", "sample_data/compressed-utv"
};

constexpr unsigned int kDatasetCount =
    static_cast<unsigned int>(sizeof(kDatasets) / sizeof(kDatasets[0]));

bool ParseMode(const char* text, RunMode* mode)
{
    if (std::strcmp(text, "compress-save") == 0 ||
        std::strcmp(text, "1") == 0) {
        *mode = RunMode::CompressSave;
        return true;
    }
    if (std::strcmp(text, "compress-query") == 0 ||
        std::strcmp(text, "2") == 0) {
        *mode = RunMode::CompressQuery;
        return true;
    }
    if (std::strcmp(text, "load-query") == 0 ||
        std::strcmp(text, "3") == 0) {
        *mode = RunMode::LoadQuery;
        return true;
    }
    return false;
}

bool ParseDatasetId(const char* text, DatasetId* dataset_id)
{
    const char* names[] = {
        "yahoo", "flickr", "editwiki", "cbtpow", "cbtcomm", "wikilink",
        "sample"
    };
    for (unsigned int index = 0u; index < kDatasetCount; ++index) {
        if (std::strcmp(text, names[index]) == 0) {
            *dataset_id = static_cast<DatasetId>(index);
            return true;
        }
    }

    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value >= kDatasetCount) {
        return false;
    }
    *dataset_id = static_cast<DatasetId>(value);
    return true;
}

void PrintUsage(const char* program)
{
    std::fprintf(stderr,
        "Usage: %s [compress-save|compress-query|load-query] "
        "[dataset-id|dataset-path] [output-directory] [UVT|UTV]\n"
        "  dataset-id: 0=yahoo, 1=flickr, 2=editwiki, 3=cbtpow, "
        "4=cbtcomm, 5=wikilink, 6=sample\n"
        "  Dataset names may be used instead of numeric IDs.\n"
        "  A custom dataset path uses DATA_POINT and requires an output directory.\n"
        "  Named datasets accept optional raw dataset path and output directory.\n",
        program);
}

std::uint64_t FileSize(const char* path)
{
    if (path == nullptr) return 0u;
#if defined(_WIN32)
    struct _stat64 info;
    return _stat64(path, &info) == 0 && info.st_size >= 0
        ? static_cast<std::uint64_t>(info.st_size)
        : 0u;
#else
    struct stat info;
    return stat(path, &info) == 0 && info.st_size >= 0
        ? static_cast<std::uint64_t>(info.st_size)
        : 0u;
#endif
}

bool HasOrderFile(const char* directory)
{
    if (directory == nullptr || *directory == '\0') return false;
    std::string path(directory);
    if (!path.empty() && path.back() != '/' && path.back() != '\\')
        path += '/';
    path += "order.txt";
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) return false;
    std::fclose(file);
    return true;
}

void PrintCompressionResult(
    const CompressionStatistics& stats,
    uint64_t nodes,
    std::uint64_t events,
    uint64_t times,
    const char* raw_path,
    const char* output_directory,
    double compression_seconds)
{
    constexpr double kMiB = 1024.0 * 1024.0;
    const std::uint64_t raw_bytes = FileSize(raw_path);
    const std::ios::fmtflags saved_flags = std::cout.flags();
    const std::streamsize saved_precision = std::cout.precision();
    std::cout << "===== Compression Result =====\n"
        << "Nodes: " << nodes << '\n'
        << "Events: " << events << '\n'
        << "Times: " << times << '\n'
        << "NormalFlagList: " << stats.normal_flag_bits << " bits\n"
        << "ShareFlagList: " << stats.share_flag_bits << " bits\n"
        << "SkipFlagList: " << stats.skip_flag_bits << " bits\n"
        << "ShareNodeList: " << stats.share_node_bits << " bits\n"
        << "SkipValue: " << stats.skip_value_bits << " bits\n"
        << "ShareValue: " << stats.share_value_bits << " bits\n"
        << "Payload: " << stats.payload_bits << " bits\n"
        << "Serialized size: " << stats.serialized_bytes << " bytes ("
        << std::fixed << std::setprecision(2)
        << static_cast<double>(stats.serialized_bytes) / kMiB << " MiB)\n";
    if (raw_bytes != 0u && stats.serialized_bytes != 0u) {
        std::cout << "Compression ratio: "
            << static_cast<double>(raw_bytes) /
                static_cast<double>(stats.serialized_bytes)
            << "x\n";
    }
    if (compression_seconds >= 0.0) {
        std::cout << "Compression time: " << compression_seconds << " s\n";
    }
    if (output_directory != nullptr) {
        std::cout << "Directory: " << output_directory << '\n';
    }
    std::cout.flush();
    std::cout.flags(saved_flags);
    std::cout.precision(saved_precision);
}

void PrintQueryResult(const char* title, const Stats& stats)
{
    std::cout << title << ": " << std::fixed << std::setprecision(3)
        << stats.avg_ns / 1000.0 << " us\n";
}

void RunQueries(
    const Graph_Compress& graph,
    uint64_t nodes,
    uint64_t times,
    DataType dataset)
{
    if (std::strcmp(graph.order, "UTV") == 0) {
        PrintQueryResult("Arc",
            benchmark_PointActiveEdge_UTV(graph, nodes, times, dataset));
        PrintQueryResult("ArcNext",
            benchmark_QueryNextAct_UTV(graph, nodes, times, dataset));
        PrintQueryResult("DirectNeighbors",
            benchmark_QueryDirectNeighbor_UTV(graph, nodes, times, dataset));
        return;
    }
    if (nodes > UINT_MAX || times > UINT_MAX) {
        std::fprintf(stderr,
            "UVT query interfaces currently accept node and time IDs up to UINT_MAX\n");
        return;
    }
    const unsigned int uvt_nodes = static_cast<unsigned int>(nodes);
    const unsigned int uvt_times = static_cast<unsigned int>(times);
    Stats edge = benchmark_PointActiveEdge(graph, uvt_nodes, uvt_times, dataset);
    PrintQueryResult("Arc", edge);

    Stats next = benchmark_QueryNextAct(graph, uvt_nodes, uvt_times, dataset);
    PrintQueryResult("ArcNext", next);

    Stats neighbor = benchmark_QueryDirectNeighbor(graph, uvt_nodes, uvt_times, dataset);
    PrintQueryResult("DirectNeighbors", neighbor);
}

bool ReadDatasetHeader(
    const char* path,
    uint64_t* nodes,
    uint64_t* events,
    uint64_t* times)
{
    FILE* file = OpenFile(path);
    if (file == nullptr) return false;
    unsigned long long node_count = 0, event_count = 0, time_count = 0;
    const bool ok = fscanf_s(file, "%llu %llu %llu", &node_count,
        &event_count, &time_count) == 3;
    std::fclose(file);
    if (!ok) std::fprintf(stderr, "Invalid dataset header: %s\n", path);
    else {
        *nodes = static_cast<uint64_t>(node_count);
        *events = static_cast<uint64_t>(event_count);
        *times = static_cast<uint64_t>(time_count);
    }
    return ok;
}

Graph_Compress* CompressDataset(
    const Dataset& dataset,
    uint64_t* nodes,
    uint64_t* events,
    uint64_t* times,
    double* compression_seconds)
{
    FILE* file = OpenFile(dataset.path);
    if (file == nullptr) return nullptr;
    DdManager* manager = Cudd_Init(
        0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
    Str_Graph_new* graph = InitStrGraph();
    Graph_Array* arrays = InitGraphArray();
    if (graph != nullptr) std::memset(graph, 0, sizeof(*graph));
    if (arrays != nullptr) std::memset(arrays, 0, sizeof(*arrays));
    if (manager == nullptr || graph == nullptr || arrays == nullptr) {
        std::fprintf(stderr, "Compression initialization failed\n");
        std::fclose(file);
        if (arrays != nullptr) FreeGraphArray(arrays);
        if (manager != nullptr) {
            FreeStrGraph(manager, graph);
            Cudd_Quit(manager);
        }
        else {
            std::free(graph);
        }
        return nullptr;
    }

    const auto start = std::chrono::steady_clock::now();
    Str_Graph_new* read_result = nullptr;
    if (dataset.order != nullptr && std::strcmp(dataset.order, "UTV") == 0) {
        read_result = ReadGraphUTV(manager, file, graph);
    }
    else {
        unsigned long long wide_nodes = 0, wide_events = 0, wide_times = 0;
        if (fscanf_s(file, "%llu %llu %llu", &wide_nodes,
                &wide_events, &wide_times) != 3) {
            std::fprintf(stderr, "Invalid UVT graph header\n");
            std::fclose(file);
            FreeStrGraph(manager, graph);
            FreeGraphArray(arrays);
            Cudd_Quit(manager);
            return nullptr;
        }
        if (wide_nodes > UINT_MAX || wide_times > UINT_MAX) {
            std::fprintf(stderr,
                "UVT graph dimensions exceed the 32-bit reader range; select UTV\n");
            std::fclose(file);
            FreeStrGraph(manager, graph);
            FreeGraphArray(arrays);
            Cudd_Quit(manager);
            return nullptr;
        }
        std::rewind(file);
        read_result = ReadGraph(manager, file, graph);
    }
    if (read_result == nullptr) {
        std::fprintf(stderr, "ReadGraph failed\n");
        FreeStrGraph(manager, graph);
        FreeGraphArray(arrays);
        Cudd_Quit(manager);
        return nullptr;
    }
    graph = read_result;
    *nodes = graph->nodes;
    *events = graph->evnum;
    *times = graph->tinum;

    graph = AssignNodeID(manager, graph, arrays);
    graph = CreateArray(manager, graph, arrays);
    graph = AssignNewNodeID(manager, graph, arrays);
    graph = CreateShareArray(manager, graph, arrays);
    FreeStrGraph(manager, graph);
    Cudd_Quit(manager);

    Graph_Compress* compressed = InitGraphCompress();
    if (compressed != nullptr) {
        compressed = CreateBitGraph(arrays, compressed);
        if (compressed != nullptr) {
            std::strncpy(compressed->order,
                dataset.order != nullptr ? dataset.order : "UVT", 3u);
            compressed->order[3] = '\0';
        }
    }
    FreeGraphArray(arrays);
    *compression_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return compressed;
}

int RunRegularDataset(const Dataset& dataset, RunMode mode)
{
    uint64_t nodes = 0u;
    uint64_t events = 0u;
    uint64_t times = 0u;
    Graph_Compress* graph = nullptr;
    double compression_seconds = -1.0;

    if (mode == RunMode::LoadQuery) {
        if (!ReadDatasetHeader(dataset.path, &nodes, &events, &times)) return 1;
        if (dataset.order == nullptr || std::strcmp(dataset.order, "UTV") != 0) {
            if (nodes > UINT_MAX || times > UINT_MAX) {
                std::fprintf(stderr,
                    "UVT compressed graphs support 32-bit node/time domains; select UTV\n");
                return 1;
            }
        }
        const unsigned int nips_v = ComputeRequiredBitsForMax64(nodes);
        const unsigned int nips_t = ComputeRequiredBitsForMax64(times);
        graph = InitGraphCompress();
        if (graph != nullptr) {
            graph = LoadBitGraphFromTxt(
                graph, nips_v, nips_t, dataset.compress_dir);
        }
    }
    else {
        graph = CompressDataset(
            dataset, &nodes, &events, &times, &compression_seconds);
    }
    if (graph == nullptr) return 1;

    // Older BBDDRead directories do not carry order.txt.  In that case use
    // the order explicitly selected on the command line; a recorded order
    // remains authoritative when order.txt is present.
    if (mode == RunMode::LoadQuery && !HasOrderFile(dataset.compress_dir) &&
        (dataset.order != nullptr) &&
        (std::strcmp(dataset.order, "UVT") == 0 ||
         std::strcmp(dataset.order, "UTV") == 0)) {
        std::strncpy(graph->order, dataset.order, 3u);
        graph->order[3] = '\0';
    }

    if (mode == RunMode::CompressSave &&
        SaveBitGraphToTxt(graph, dataset.compress_dir) != 0) {
        FreeGraphCompress(graph);
        return 1;
    }

    const char* displayed_directory = mode == RunMode::CompressQuery
        ? nullptr : dataset.compress_dir;


    if (mode != RunMode::CompressSave) {
        RunQueries(*graph, nodes, times, dataset.type);
    }
    FreeGraphCompress(graph);
    return 0;
}


}

int RunApplication(
    int argc,
    char* argv[],
    const ApplicationOptions& defaults)
{
    RunMode mode = defaults.mode;
    DatasetId dataset_id = defaults.dataset;
    Dataset custom_dataset{};
    const char* selected_order = "UVT";
    int argument_count = argc;
    if (argc > 1 && (std::strcmp(argv[argc - 1], "UVT") == 0 ||
        std::strcmp(argv[argc - 1], "UTV") == 0)) {
        selected_order = argv[argc - 1];
        --argument_count;
    }
    if (argument_count > 5) {
        PrintUsage(argv[0]);
        return 1;
    }
    if (argument_count > 1 && !ParseMode(argv[1], &mode)) {
        PrintUsage(argv[0]);
        return 1;
    }
    const Dataset* dataset = nullptr;
    if (argument_count > 2) {
        if (ParseDatasetId(argv[2], &dataset_id)) {
            if (argument_count > 3) {
                if (argument_count != 5 || argv[3][0] == '\0' || argv[4][0] == '\0') {
                    PrintUsage(argv[0]);
                    return 1;
                }
                custom_dataset = kDatasets[static_cast<unsigned int>(dataset_id)];
                custom_dataset.path = argv[3];
                custom_dataset.compress_dir = argv[4];
                dataset = &custom_dataset;
            }
        }
        else {
            if (argument_count != 4 || argv[3] == nullptr || argv[3][0] == '\0') {
                PrintUsage(argv[0]);
                return 1;
            }
            custom_dataset.path = argv[2];
            custom_dataset.type = DATA_POINT;
            custom_dataset.compress_dir = argv[3];
            custom_dataset.order = selected_order;
            dataset = &custom_dataset;
        }
    }

    if (dataset == nullptr) {
        const unsigned int dataset_index = static_cast<unsigned int>(dataset_id);
        if (dataset_index >= kDatasetCount) {
            PrintUsage(argv[0]);
            return 1;
        }
        dataset = &kDatasets[dataset_index];
    }
    Dataset selected_dataset = *dataset;
    selected_dataset.order = selected_order;
    if (dataset != &custom_dataset && std::strcmp(selected_order, "UTV") == 0) {
        const unsigned int index = static_cast<unsigned int>(dataset_id);
        if (index >= sizeof(kUtvOutputDirectories) / sizeof(kUtvOutputDirectories[0])) {
            std::fprintf(stderr, "UTV output for this dataset requires explicit paths\n");
            return 1;
        }
        selected_dataset.compress_dir = kUtvOutputDirectories[index];
        if (dataset_id == DatasetId::Sample)
            selected_dataset.path = "sample_data/sample-point-utv.txt";
    }
    return RunRegularDataset(selected_dataset, mode);
}
