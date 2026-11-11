#pragma once

#include "Graph.h"

#include <cstddef>
#include <cstdint>



void BuildQueryAcceleration(const Graph_Compress& graph);
void FreeQueryAcceleration(const Graph_Compress& graph);
std::size_t QueryAccelerationBytes(const Graph_Compress& graph);

struct AcceleratedBitRank {
    unsigned int bit;
    unsigned int rank;
};

std::uint32_t AcceleratedBitPopCount(
    const unsigned int* data, std::uint64_t first, std::uint64_t last);

AcceleratedBitRank AcceleratedGetBitAndRank(
    const BitArray* array, unsigned int level, unsigned int position);

unsigned int AcceleratedUnpackOne(
    const Graph_Compress& graph, unsigned int node_index, unsigned int* level);

MinHeap* AcceleratedUnpackAll(
    const Graph_Compress& graph, MinHeap* visitlist);

unsigned int AcceleratedGetDirectNode(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int path,
    int path_kind,
    unsigned int* end_level);

bool AcceleratedPointActiveEdgePoint(
    const Graph_Compress& graph, unsigned int u, unsigned int v, unsigned int t);

bool AcceleratedPointActiveEdgeInterval(
    const Graph_Compress& graph, unsigned int u, unsigned int v, unsigned int t);

bool AcceleratedPointActiveEdgeIncremental(
    const Graph_Compress& graph, unsigned int u, unsigned int v, unsigned int t);

bool AcceleratedCompareTime(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int time);

bool AcceleratedCompareTimeParity(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int time);



bool AcceleratedQueryDirectNeighbors(
    const Graph_Compress& graph,
    unsigned int u,
    unsigned int t,
    unsigned int nodenum,
    DataType dataset,
    NeighborResult* out);

bool AcceleratedQueryNextAct(
    const Graph_Compress& graph,
    unsigned int u,
    unsigned int v,
    unsigned int t,
    unsigned int timenum,
    DataType dataset,
    unsigned int* result);

int* AcceleratedSameBitRankArray(
    const BitArray* array,
    const int* positions,
    unsigned int count,
    unsigned int level);

int* AcceleratedRiceAccessArray(
    const RiceArray* array,
    unsigned int level,
    const int* one_based_indices,
    unsigned int count);

int* AcceleratedFixBitsAccessArray(
    const FixBitArray* array,
    unsigned int level,
    const int* one_based_indices,
    unsigned int count);

bool AcceleratedRunGammaBitRankArray(
    const RunBitArray* array,
    unsigned int level,
    const int* positions,
    unsigned int count,
    int** bits,
    int** ranks);

unsigned int ReferenceUnpackOne(
    const Graph_Compress& graph, unsigned int node_index, unsigned int* level);

unsigned int ReferenceGetDirectNode(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int path,
    int path_kind,
    unsigned int* end_level);

bool ReferencePointActiveEdgePoint(
    const Graph_Compress& graph, unsigned int u, unsigned int v, unsigned int t);

bool ReferenceCompareTime(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int time);

bool ReferenceCompareTimeParity(
    const Graph_Compress& graph,
    unsigned int node_id,
    unsigned int start_level,
    unsigned int time);

int* ReferenceSameBitRankArray(
    const BitArray* array,
    const int* positions,
    unsigned int count,
    unsigned int level);

bool VerifyAcceleratedCore(
    const Graph_Compress& graph, unsigned int samples_per_level);
