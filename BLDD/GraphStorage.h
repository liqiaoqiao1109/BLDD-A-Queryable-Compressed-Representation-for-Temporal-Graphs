#pragma once

#include "Graph.h"

#include <cstdint>

struct CompressionStatistics {
    std::uint64_t normal_flag_bits = 0u;
    std::uint64_t share_flag_bits = 0u;
    std::uint64_t skip_flag_bits = 0u;
    std::uint64_t share_node_bits = 0u;
    std::uint64_t skip_value_bits = 0u;
    std::uint64_t share_value_bits = 0u;
    std::uint64_t payload_bits = 0u;
    std::uint64_t serialized_bytes = 0u;
};

CompressionStatistics GetCompressionStatistics(const Graph_Compress& graph);

