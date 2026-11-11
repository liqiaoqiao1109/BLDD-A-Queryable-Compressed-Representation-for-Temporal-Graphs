#define _CRT_SECURE_NO_WARNINGS

#include "GraphStorage.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#include <process.h>
#include <sys/stat.h>
#include <windows.h>
#define DEFAULT_OUT_DIR "./bitgraph_out"
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define DEFAULT_OUT_DIR "./bitgraph_out"
#endif

namespace {

long ProcessId()
{
#if defined(_WIN32)
    return static_cast<long>(_getpid());
#else
    return static_cast<long>(getpid());
#endif
}

bool IsPathSeparator(char value)
{
    return value == '/' || value == '\\';
}

bool IsDirectory(const std::string& path)
{
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR) != 0;
}

int MakeDirectory(const std::string& path)
{
#if defined(_WIN32)
    return _mkdir(path.c_str());
#else
    return mkdir(path.c_str(), 0755);
#endif
}

bool EnsureOutputDirectory(const char* output_dir)
{
    if (output_dir == nullptr || *output_dir == '\0') return false;
    const std::string path(output_dir);
    if (IsDirectory(path)) return true;

    for (std::size_t index = 0u; index < path.size(); ++index) {
        if (!IsPathSeparator(path[index]) || index == 0u) continue;
        if (index == 2u && path[1] == ':') continue;
        const std::string prefix = path.substr(0u, index);
        if (!prefix.empty() && !IsDirectory(prefix) &&
            MakeDirectory(prefix) != 0 && errno != EEXIST) {
            std::fprintf(stderr, "Cannot create output directory %s: %s\n",
                prefix.c_str(), std::strerror(errno));
            return false;
        }
    }
    if (!IsDirectory(path) && MakeDirectory(path) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "Cannot create output directory %s: %s\n",
            path.c_str(), std::strerror(errno));
        return false;
    }
    return IsDirectory(path);
}

std::string JoinPath(const char* output_dir, const char* name)
{
    std::string path(output_dir == nullptr ? "" : output_dir);
    if (!path.empty() && !IsPathSeparator(path.back())) path += '/';
    path += name;
    return path;
}

struct OutputTransaction {
    std::string final_path;
    std::string temporary_path;
    FILE* file = nullptr;
};

// Each component is published only after its temporary file is complete.
bool BeginOutput(
    const char* output_dir,
    const char* name,
    OutputTransaction* output)
{
    output->final_path = JoinPath(output_dir, name);
    output->temporary_path = output->final_path +
        ".tmp." + std::to_string(ProcessId());
    output->file = std::fopen(output->temporary_path.c_str(), "wb");
    if (output->file == nullptr) {
        std::fprintf(stderr, "Cannot open %s: %s\n",
            output->temporary_path.c_str(), std::strerror(errno));
        return false;
    }
    return true;
}

bool PublishOutput(const std::string& temporary, const std::string& final_path)
{
#if defined(_WIN32)
    return MoveFileExA(
        temporary.c_str(), final_path.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != FALSE;
#else
    return std::rename(temporary.c_str(), final_path.c_str()) == 0;
#endif
}

bool FinishOutput(OutputTransaction* output, bool success)
{
    if (output->file != nullptr) {
        if (std::fclose(output->file) != 0) success = false;
        output->file = nullptr;
    }
    if (!success) {
        std::remove(output->temporary_path.c_str());
        return false;
    }
    if (!PublishOutput(output->temporary_path, output->final_path)) {
        std::fprintf(stderr, "Cannot publish %s\n", output->final_path.c_str());
        std::remove(output->temporary_path.c_str());
        return false;
    }
    return true;
}

bool WriteExact(
    FILE* file,
    const void* data,
    std::size_t element_size,
    std::size_t count)
{
    return count == 0u ||
        std::fwrite(data, element_size, count, file) == count;
}

std::uint64_t BytesForBits(std::uint64_t bits)
{
    return (bits + 7u) >> 3;
}

bool WriteBitStream(
    FILE* file,
    const unsigned int* words,
    std::uint64_t bit_count)
{
    const std::uint64_t byte_count = BytesForBits(bit_count);
    if (byte_count == 0u) return true;
    if (words == nullptr) return false;
    constexpr std::size_t kChunkSize = 1u << 20;
    std::vector<unsigned char> buffer(kChunkSize);
    std::uint64_t written = 0u;
    while (written < byte_count) {
        const std::size_t chunk = static_cast<std::size_t>(
            (std::min)(byte_count - written,
                static_cast<std::uint64_t>(buffer.size())));
        for (std::size_t index = 0u; index < chunk; ++index) {
            const std::uint64_t byte_index = written + index;
            const unsigned int word = words[byte_index >> 2u];
            buffer[index] = static_cast<unsigned char>(
                (word >> ((byte_index & 3u) * 8u)) & 0xffu);
        }
        if (!WriteExact(file, buffer.data(), 1u, chunk)) return false;
        written += chunk;
    }
    return true;
}

std::vector<unsigned char> PackSmallIntegers(
    const unsigned int* source,
    unsigned int count,
    unsigned int bits_per_element)
{
    const std::uint64_t total_bits =
        static_cast<std::uint64_t>(count) * bits_per_element;
    std::vector<unsigned char> output(
        static_cast<std::size_t>(BytesForBits(total_bits)), 0u);
    std::uint64_t bit_position = 0u;
    for (unsigned int index = 0u; index < count; ++index) {
        const unsigned int value = source[index];
        for (unsigned int bit = 0u; bit < bits_per_element;
            ++bit, ++bit_position) {
            if (((value >> bit) & 1u) != 0u) {
                output[static_cast<std::size_t>(bit_position >> 3u)] |=
                    static_cast<unsigned char>(1u << (bit_position & 7u));
            }
        }
    }
    return output;
}

bool ValidateIndexes(
    const std::uint64_t* indexes,
    unsigned int levels,
    std::uint64_t count,
    const char* name)
{
    if (levels != 0u && indexes == nullptr) {
        std::fprintf(stderr, "%s has no index array\n", name);
        return false;
    }
    std::uint64_t previous = 0u;
    for (unsigned int level = 0u; level < levels; ++level) {
        if (indexes[level] < previous || indexes[level] > count) {
            std::fprintf(stderr,
                "%s index[%u]=%llu is outside [%llu, %llu]\n",
                name, level,
                static_cast<unsigned long long>(indexes[level]),
                static_cast<unsigned long long>(previous),
                static_cast<unsigned long long>(count));
            return false;
        }
        previous = indexes[level];
    }
    return true;
}

int SaveMeta(
    const char* output_dir,
    const Graph_Compress* graph,
    unsigned int levels)
{
    OutputTransaction output;
    if (!BeginOutput(output_dir, "meta.txt", &output)) return -1;
    bool wide = false;
    for (unsigned int i = 0; i < levels; ++i) {
        if (graph->VariatyNumList[i] > UINT32_MAX) {
            wide = true;
            break;
        }
    }
    bool ok = true;
    if (!wide) {
        ok = WriteExact(output.file, &graph->nips_v, sizeof(graph->nips_v), 1u) &&
            WriteExact(output.file, &graph->nips_t, sizeof(graph->nips_t), 1u);
        for (unsigned int i = 0; ok && i < levels; ++i) {
            // Keep the legacy BBDDRead representation: the final synthetic
            // boundary is written as zero and reconstructed by the loader.
            const uint32_t value = (i + 1u == levels)
                ? 0u
                : static_cast<uint32_t>(graph->VariatyNumList[i]);
            ok = WriteExact(output.file, &value, sizeof(value), 1u);
        }
    }
    else {
        static const unsigned char magic[8] = {
            'B', 'B', 'D', 'D', 'M', '6', '4', 2
        };
        const uint32_t version = 2u;
        ok = WriteExact(output.file, magic, sizeof(magic), 1u) &&
            WriteExact(output.file, &version, sizeof(version), 1u) &&
            WriteExact(output.file, &graph->nips_v, sizeof(graph->nips_v), 1u) &&
            WriteExact(output.file, &graph->nips_t, sizeof(graph->nips_t), 1u) &&
            WriteExact(output.file, graph->VariatyNumList,
                sizeof(uint64_t), levels);
    }
    return FinishOutput(&output, ok) ? 0 : -1;
}

int SaveOrder(const char* output_dir, const char order[4])
{
    OutputTransaction output;
    if (!BeginOutput(output_dir, "order.txt", &output)) return -1;
    const char* selected = order && order[0] ? order : "UVT";
    const bool ok = std::strcmp(selected, "UVT") == 0 ||
        std::strcmp(selected, "UTV") == 0;
    if (!ok) {
        FinishOutput(&output, false);
        return -1;
    }
    const bool written = WriteExact(output.file, selected, 1u, 3u) &&
        WriteExact(output.file, "\n", 1u, 1u);
    return FinishOutput(&output, written) ? 0 : -1;
}

int SaveBitArray(
    const char* output_dir,
    const char* name,
    const BitArray* array,
    unsigned int levels)
{
    // New files use 64-bit counts and indexes to support graphs above 4 GiB.
    if (array == nullptr || array->Flag == nullptr ||
        !ValidateIndexes(array->Index, levels, array->Count, name)) return -1;
    OutputTransaction output;
    if (!BeginOutput(output_dir, name, &output)) return -1;
    const std::vector<unsigned char> flags =
        PackSmallIntegers(array->Flag, levels, 2u);
    const bool ok =
        WriteExact(output.file, &array->Count, sizeof(array->Count), 1u) &&
        WriteExact(output.file, array->Index, sizeof(std::uint64_t), levels) &&
        WriteExact(output.file, flags.data(), 1u, flags.size()) &&
        WriteBitStream(output.file, array->List, array->Count);
    return FinishOutput(&output, ok) ? 0 : -1;
}

int SaveRunBitArray(
    const char* output_dir,
    const char* name,
    const RunBitArray* array,
    unsigned int levels)
{
    if (array == nullptr || array->Flag == nullptr || array->First == nullptr ||
        !ValidateIndexes(array->Index, levels, array->Count, name)) return -1;
    OutputTransaction output;
    if (!BeginOutput(output_dir, name, &output)) return -1;
    const std::vector<unsigned char> flags =
        PackSmallIntegers(array->Flag, levels, 1u);
    const std::vector<unsigned char> first =
        PackSmallIntegers(array->First, levels, 1u);
    const bool ok =
        WriteExact(output.file, &array->Count, sizeof(array->Count), 1u) &&
        WriteExact(output.file, array->Index, sizeof(std::uint64_t), levels) &&
        WriteExact(output.file, flags.data(), 1u, flags.size()) &&
        WriteExact(output.file, first.data(), 1u, first.size()) &&
        WriteBitStream(output.file, array->List, array->Count);
    return FinishOutput(&output, ok) ? 0 : -1;
}

int SaveFixBitArray(
    const char* output_dir,
    const char* name,
    const FixBitArray* array,
    unsigned int levels)
{
    if (array == nullptr || array->BitsPerElement == 0u ||
        !ValidateIndexes(array->Index, levels, array->Count, name)) return -1;
    OutputTransaction output;
    if (!BeginOutput(output_dir, name, &output)) return -1;
    const bool ok =
        WriteExact(output.file, &array->Count, sizeof(array->Count), 1u) &&
        WriteExact(output.file, &array->BitsPerElement,
            sizeof(array->BitsPerElement), 1u) &&
        WriteExact(output.file, array->Index, sizeof(std::uint64_t), levels) &&
        WriteBitStream(output.file, array->List, array->Count);
    return FinishOutput(&output, ok) ? 0 : -1;
}

int SaveRiceArray(
    const char* output_dir,
    const char* name,
    const RiceArray* array,
    unsigned int levels)
{
    if (array == nullptr || array->RiceParameter == nullptr ||
        !ValidateIndexes(array->quoIndex, levels, array->quoCount,
            "ShareValue quotient") ||
        !ValidateIndexes(array->remIndex, levels, array->remCount,
            "ShareValue remainder")) return -1;
    OutputTransaction output;
    if (!BeginOutput(output_dir, name, &output)) return -1;
    const bool ok =
        WriteExact(output.file, &array->quoCount, sizeof(array->quoCount), 1u) &&
        WriteExact(output.file, &array->remCount, sizeof(array->remCount), 1u) &&
        WriteExact(output.file, array->quoIndex, sizeof(std::uint64_t), levels) &&
        WriteExact(output.file, array->remIndex, sizeof(std::uint64_t), levels) &&
        WriteExact(output.file, array->RiceParameter,
            sizeof(unsigned int), levels) &&
        WriteBitStream(output.file, array->quotient, array->quoCount) &&
        WriteBitStream(output.file, array->remainder, array->remCount);
    return FinishOutput(&output, ok) ? 0 : -1;
}

std::uint64_t BitArrayBytes(
    const BitArray& array,
    std::uint64_t levels)
{
    return sizeof(std::uint64_t) + levels * sizeof(std::uint64_t) +
        BytesForBits(levels * 2u) + BytesForBits(array.Count);
}

std::uint64_t RunBitArrayBytes(
    const RunBitArray& array,
    std::uint64_t levels)
{
    return sizeof(std::uint64_t) + levels * sizeof(std::uint64_t) +
        BytesForBits(levels) * 2u + BytesForBits(array.Count);
}

}

int SaveBitGraphToTxt(const Graph_Compress* graph)
{
    return SaveBitGraphToTxt(graph, DEFAULT_OUT_DIR);
}

int SaveBitGraphToTxt(const Graph_Compress* graph, const char* output_dir)
{
    if (graph == nullptr || graph->VariatyNumList == nullptr ||
        graph->NormalFlagList == nullptr || graph->ShareFlagList == nullptr ||
        graph->SkipFlagList == nullptr || graph->ShareNodeList == nullptr ||
        graph->SkipValue == nullptr || graph->ShareValue == nullptr) {
        std::fprintf(stderr, "SaveBitGraphToTxt: incomplete compressed graph\n");
        return -1;
    }
    if (!EnsureOutputDirectory(output_dir)) return -1;
    const std::uint64_t levels64 =
        static_cast<std::uint64_t>(graph->nips_v) * 2u + graph->nips_t + 1u;
    if (levels64 > UINT32_MAX) return -1;
    const unsigned int levels = static_cast<unsigned int>(levels64);
    return SaveMeta(output_dir, graph, levels) == 0 &&
        SaveOrder(output_dir, graph->order) == 0 &&
        SaveBitArray(output_dir, "NormalFlagList.txt",
            graph->NormalFlagList, levels) == 0 &&
        SaveBitArray(output_dir, "ShareFlagList.txt",
            graph->ShareFlagList, levels) == 0 &&
        SaveRunBitArray(output_dir, "SkipFlagList.txt",
            graph->SkipFlagList, levels) == 0 &&
        SaveRunBitArray(output_dir, "ShareNodeList.txt",
            graph->ShareNodeList, levels) == 0 &&
        SaveFixBitArray(output_dir, "SkipValue.txt",
            graph->SkipValue, levels) == 0 &&
        SaveRiceArray(output_dir, "ShareValue.txt",
            graph->ShareValue, levels) == 0 ? 0 : -1;
}

CompressionStatistics GetCompressionStatistics(const Graph_Compress& graph)
{
    CompressionStatistics stats{};
    if (graph.NormalFlagList == nullptr || graph.ShareFlagList == nullptr ||
        graph.SkipFlagList == nullptr || graph.ShareNodeList == nullptr ||
        graph.SkipValue == nullptr || graph.ShareValue == nullptr) {
        return stats;
    }
    stats.normal_flag_bits = graph.NormalFlagList->Count;
    stats.share_flag_bits = graph.ShareFlagList->Count;
    stats.skip_flag_bits = graph.SkipFlagList->Count;
    stats.share_node_bits = graph.ShareNodeList->Count;
    stats.skip_value_bits = graph.SkipValue->Count;
    stats.share_value_bits =
        graph.ShareValue->quoCount + graph.ShareValue->remCount;
    stats.payload_bits = stats.normal_flag_bits + stats.share_flag_bits +
        stats.skip_flag_bits + stats.share_node_bits +
        stats.skip_value_bits + stats.share_value_bits;

    const std::uint64_t levels =
        static_cast<std::uint64_t>(graph.nips_v) * 2u + graph.nips_t + 1u;
    stats.serialized_bytes = sizeof(unsigned int) * 2u +
        levels * sizeof(unsigned int) +
        BitArrayBytes(*graph.NormalFlagList, levels) +
        BitArrayBytes(*graph.ShareFlagList, levels) +
        RunBitArrayBytes(*graph.SkipFlagList, levels) +
        RunBitArrayBytes(*graph.ShareNodeList, levels) +
        sizeof(std::uint64_t) + sizeof(unsigned int) +
        levels * sizeof(std::uint64_t) +
        BytesForBits(graph.SkipValue->Count) +
        sizeof(std::uint64_t) * 2u +
        levels * sizeof(std::uint64_t) * 2u +
        levels * sizeof(unsigned int) +
        BytesForBits(graph.ShareValue->quoCount) +
        BytesForBits(graph.ShareValue->remCount);
    return stats;
}
