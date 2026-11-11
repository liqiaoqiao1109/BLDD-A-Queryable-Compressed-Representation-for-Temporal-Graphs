#include<math.h>
#include<stdlib.h>
#pragma once
#include<ctype.h>
#include<stdio.h>
#include<time.h>
#include <stdint.h>
#include <cstddef>
#include <climits>
#include<iostream>
#include "util.h"
#include "cuddInt.h"

typedef DdNode* (*DD_MA)(DdManager*, DdNode*, int);


typedef enum {
	DATA_POINT,
	DATA_INTERVAL,
	DATA_INCREMENTAL
} DataType;

typedef struct {
	const char* path;
	DataType type;
	const char* compress_dir;
	const char* order;
} Dataset;


typedef struct Str_Graph_new {
	unsigned int nips_v;
	unsigned int nips_t;
	uint64_t evnum;
	uint64_t tinum;
	uint64_t nodes;
	DdNode** x;
	DdNode** y;
	DdNode** t;
	DdNode* Event;
	uint64_t* NewID;
}Str_Graph_new;


typedef struct UIntArray {
	unsigned int* List;
	uint64_t* Index;
	uint64_t Count;
	uint64_t Capacity;
} UIntArray;

typedef struct U64Array {
	uint64_t* List;
	uint64_t* Index;
	uint64_t Count;
	uint64_t Capacity;
} U64Array;

typedef struct Graph_Array {
	uint64_t* VariatyNumList;
	uint64_t totalNodes;
	unsigned int nips_v;
	unsigned int nips_t;

	// Six logical streams produced from the reduced decision diagram.
	UIntArray* NormalFlagList;
	UIntArray* ShareFlagList;
	UIntArray* SkipFlagList;
	UIntArray* ShareNodeList;
	U64Array* ShareValue;
	U64Array* SkipValue;

	uint64_t MaxSkip;
	// Compression statistics used while constructing the six streams.
	uint64_t NormalNum;
	uint64_t ShareNum;
	uint64_t SkipNum;
	uint64_t ZeroNum;
}Graph_Array;


typedef struct BitArray {
	unsigned int* List;
	uint64_t* Index;
	unsigned int* Flag;
	uint64_t Count;
} BitArray;

typedef struct RunBitArray {
	unsigned int* List;
	uint64_t* Index;
	unsigned int* Flag;
	unsigned int* First;
	uint64_t Count;
} RunBitArray;

typedef struct FixBitArray {
	unsigned int* List;
	uint64_t* Index;
	unsigned int BitsPerElement;
	uint64_t Count;
} FixBitArray;

typedef struct RiceArray {
	unsigned int* quotient;
	unsigned int* remainder;
	uint64_t* quoIndex;
	uint64_t* remIndex;
	unsigned int* RiceParameter;
	uint64_t quoCount;
	uint64_t remCount;
} RiceArray;

typedef struct Graph_Compress {
	unsigned int nips_v;
	unsigned int nips_t;
	uint64_t* VariatyNumList;
	char order[4];

	// Flags use constant-bit or gamma run encoding; values use Rice or fixed bits.
	BitArray* NormalFlagList;
	BitArray* ShareFlagList;
	RunBitArray* SkipFlagList;
	RunBitArray* ShareNodeList;
	RiceArray* ShareValue;
	FixBitArray* SkipValue;
} Graph_Compress;

typedef struct Stats {
	double avg_ns;
	double min_ns;
	double max_ns;
	double median_ns;
	uint64_t input_checksum;
	uint64_t checksum;
} Stats;

typedef struct NeighborResult {
	unsigned int* List;
	unsigned int Count;
	unsigned int Capacity;
} NeighborResult;

typedef struct NeighborResult64 {
	uint64_t* List;
	uint64_t Count;
	uint64_t Capacity;
} NeighborResult64;

static const unsigned int QUERY_TIME_INF = UINT_MAX;

struct MinHeap;



extern Str_Graph_new* InitStrGraph();
extern UIntArray* InitUIntArray(uint64_t Lcapacity, unsigned int Icapacity);
extern U64Array* InitU64Array(uint64_t Lcapacity, unsigned int Icapacity);
extern Graph_Array* InitGraphArray();
extern BitArray* InitBitArray(uint64_t listSize, unsigned int indexSize, unsigned int flagSize);
extern RunBitArray* InitRunBitArray(uint64_t Lcapacity, unsigned int Icapacity, unsigned int FlagSize);
extern FixBitArray* InitFixBitArray(uint64_t Lcapacity, unsigned int Icapacity, unsigned int BitsPerElement);
extern RiceArray* InitRiceArray(uint64_t quoSize, uint64_t remSize, unsigned int indexSize, unsigned int paramSize);
extern Graph_Compress* InitGraphCompress();

extern void FreeStrGraph(DdManager* manager, Str_Graph_new* G);
extern void FreeBitArray(BitArray* arr);
extern void FreeGraphArray(Graph_Array* GA);
extern void FreeUIntArray(UIntArray* arr);
extern void FreeU64Array(U64Array* arr);
extern void FreeRunBitArray(RunBitArray* arr);
extern void FreeFixBitArray(FixBitArray* arr);
extern void FreeRiceArray(RiceArray* arr);
extern void FreeGraphCompress(Graph_Compress* GC);


extern int ComputeRequiredBits(unsigned int value);
extern unsigned int ComputeRequiredBits64(uint64_t value);
extern unsigned int ComputeRequiredBitsForMax64(uint64_t value);
extern DdNode* CreateNode(DdManager* manager, unsigned int v, unsigned int npis, DdNode** x);
extern Str_Graph_new* ReadGraph(DdManager* manager, FILE* f, Str_Graph_new* G);
extern Str_Graph_new* ReadGraphUTV(DdManager* manager, FILE* f, Str_Graph_new* G);


extern void JPG(DdManager* dd, DdNode* f);


extern Str_Graph_new* AssignNodeID(DdManager* manager, Str_Graph_new* G, Graph_Array* GA);
extern Str_Graph_new* CreateArray(DdManager* manager, Str_Graph_new* G, Graph_Array* GA);
extern Str_Graph_new* AssignNewNodeID(DdManager* manager, Str_Graph_new* G, Graph_Array* GA);
extern Str_Graph_new* CreateShareArray(DdManager* manager, Str_Graph_new* G, Graph_Array* GA);


extern Graph_Compress* CreateBitGraph(Graph_Array* GA, Graph_Compress* CG);
extern int SaveBitGraphToTxt(const Graph_Compress* CG);
extern int SaveBitGraphToTxt(const Graph_Compress* CG, const char* output_dir);
extern Graph_Compress* LoadBitGraphFromTxt(Graph_Compress* CG, unsigned int nips_v, unsigned int nips_t, const char* input_dir);

extern bool PointActiveEdge_UTV(const Graph_Compress& G, uint64_t u, uint64_t v, uint64_t t, uint64_t nodenum, uint64_t timenum, DataType dataset);
extern void QueryDirectNeighbor_UTV(const Graph_Compress& G, uint64_t u, uint64_t t, uint64_t nodenum, uint64_t timenum, DataType dataset, NeighborResult64* out);
extern bool QueryNextAct_UTV(const Graph_Compress& G, uint64_t u, uint64_t v, uint64_t t, uint64_t nodenum, uint64_t timenum, DataType dataset, uint64_t* next_time);
extern void InitNeighborResult64(NeighborResult64* out, uint64_t initial_capacity);
extern void FreeNeighborResult64(NeighborResult64* out);
extern Stats benchmark_PointActiveEdge_UTV(const Graph_Compress& G, uint64_t nodenum, uint64_t timenum, DataType dataset, std::size_t query_count = 100000, unsigned int rounds = 5);
extern Stats benchmark_QueryDirectNeighbor_UTV(const Graph_Compress& G, uint64_t nodenum, uint64_t timenum, DataType dataset, std::size_t query_count = 100000, unsigned int rounds = 5);
extern Stats benchmark_QueryNextAct_UTV(const Graph_Compress& G, uint64_t nodenum, uint64_t timenum, DataType dataset, std::size_t query_count = 100, unsigned int rounds = 1);


extern unsigned int UnpackOne(const Graph_Compress& CG, unsigned int nodeindex, unsigned int* level);
extern MinHeap* UnpackAll(const Graph_Compress& CG, MinHeap* visitlist);


extern unsigned int GetDirectNode(const Graph_Compress& CG, unsigned int nodeid, unsigned int startlevel, unsigned int path, int flag, unsigned int* endlevel);
extern unsigned int CountTimeNum(const Graph_Compress& G, unsigned int nodev, unsigned int startlevel, unsigned int t);
extern bool CompareTimeParity(const Graph_Compress& G, unsigned int nodev, unsigned int startlevel, unsigned int t);
extern bool CompareTime(const Graph_Compress& G, unsigned int nodev, unsigned int startlevel, unsigned int t);
extern bool CollectDirectNeighbors(
	const Graph_Compress& G,
	unsigned int start_node,
	unsigned int start_level,
	unsigned int t,
	unsigned int nodenum,
	DataType dataset,
	NeighborResult* out);
extern bool QueryEdgeActiveByType(
	const Graph_Compress& G,
	unsigned int u,
	unsigned int v,
	unsigned int t,
	DataType dataset);
extern bool QueryDirectNeighborByType(
	const Graph_Compress& G,
	unsigned int u,
	unsigned int t,
	unsigned int nodenum,
	unsigned int timenum,
	DataType dataset,
	NeighborResult* out);


extern bool PointActiveEdge_Point(const Graph_Compress& G, unsigned int u, unsigned int v, unsigned int t);
extern bool PointActiveEdge_Interval(const Graph_Compress& G, unsigned int u, unsigned int v, unsigned int t);
extern bool PointActiveEdge_Incremental(const Graph_Compress& G, unsigned int u, unsigned int v, unsigned int t);


extern void InitNeighborResult(NeighborResult* out, unsigned int initial_capacity);
extern void FreeNeighborResult(NeighborResult* out);
extern void QueryDirectNeighbor_Point(	const Graph_Compress& G,
	unsigned int u,
	unsigned int t,
	unsigned int nodenum,
	unsigned int timenum,
	NeighborResult* out);
extern void QueryDirectNeighbor_Interval(
	const Graph_Compress& G,
	unsigned int u,
	unsigned int t,
	unsigned int nodenum,
	unsigned int timenum,
	NeighborResult* out);
extern void QueryDirectNeighbor_Incremental(
	const Graph_Compress& G,
	unsigned int u,
	unsigned int t,
	unsigned int nodenum,
	unsigned int timenum,
	NeighborResult* out);


extern unsigned int QueryNextAct_Point(
	const Graph_Compress& G,
	unsigned int u,
	unsigned int v,
	unsigned int t,
	unsigned int nodenum,
	unsigned int timenum);
extern unsigned int QueryNextAct_Interval(
	const Graph_Compress& G,
	unsigned int u,
	unsigned int v,
	unsigned int t,
	unsigned int nodenum,
	unsigned int timenum);
extern unsigned int QueryNextAct_Incremental(
	const Graph_Compress& G,
	unsigned int u,
	unsigned int v,
	unsigned int t,
	unsigned int nodenum,
	unsigned int timenum);


extern Stats benchmark_PointActiveEdge(
	const Graph_Compress& G,
	unsigned int nodenum,
	unsigned int timenum,
	DataType dataset,
	std::size_t query_count = 100000,
	unsigned int rounds = 5);
extern Stats benchmark_QueryDirectNeighbor(
	const Graph_Compress& G,
	unsigned int nodenum,
	unsigned int timenum,
	DataType dataset,
	std::size_t query_count = 100000,
	unsigned int rounds = 5);
extern Stats benchmark_QueryNextAct(
	const Graph_Compress& G,
	unsigned int nodenum,
	unsigned int timenum,
	DataType dataset,
	std::size_t query_count = 100,
	unsigned int rounds = 1);
