#include"Graph.h"
#include"Help.h"
#include"Bit.h"
#include"QueryAccelerate.h"

#include <algorithm>
#include <climits>


static inline void resetQueue(Queue* q)
{
    q->front = 0;
    q->rear = 0;
    q->size = 0;
}

static inline void ensureQueueCapacity(Queue* q, unsigned int need)
{
    if (q->capacity >= need) return;

    unsigned int newCap = q->capacity ? q->capacity : 1;
    while (newCap < need) newCap <<= 1;

    node* newData = (node*)malloc(sizeof(node) * newCap);
    if (!newData) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }
    if (q->size != 0) {
        memcpy(newData, q->data + q->front, sizeof(node) * q->size);
    }
    free(q->data);
    q->data = newData;
    q->capacity = newCap;
    q->front = 0;
    q->rear = q->size;
}

static inline void ensureHeapCapacity(MinHeap* heap, int need)
{
    if (heap->capacity >= need) return;
    int newCap = heap->capacity ? heap->capacity : 1;
    while (newCap < need) newCap <<= 1;
    heap->data = (Node*)realloc(heap->data, sizeof(Node) * newCap);
    heap->capacity = newCap;
}

static inline void ensureHashCapacity(HashMap* map, int need)
{
    if (map->capacity >= need) {
        clearHash(map);
        return;
    }
    int newCapacity = 1;
    while (newCapacity < need) newCapacity <<= 1;
    map->table = (HashEntry*)realloc(map->table, sizeof(HashEntry) * newCapacity);
    map->capacity = newCapacity;
    map->size = 0;
    memset(map->table, 0, sizeof(HashEntry) * map->capacity);
}

static inline void prepareQueryScratch(QueryScratch* scratch, unsigned int initCap)
{
    initQueryScratch(scratch);

    unsigned int capA = initCap ? initCap : 1;
    unsigned int capB = capA * 2u;
    if (capB == 0) capB = capA;

    resetQueue(scratch->qA);
    resetQueue(scratch->qB);

    ensureQueueCapacity(scratch->qA, capA);
    ensureQueueCapacity(scratch->qB, capB);

    ensureHeapCapacity(scratch->heapA, 1);
    ensureHeapCapacity(scratch->heapB, 2);
    scratch->heapA->size = 0;
    scratch->heapB->size = 0;

    ensureHashCapacity(scratch->mapA, 2);
    ensureHashCapacity(scratch->mapB, 2);
    ensureHashCapacity(scratch->visited, 2);

    if (scratch->uniqueCap < capB) {
        scratch->uniqueBuf = (node*)realloc(scratch->uniqueBuf, sizeof(node) * capB);
        scratch->uniqueCap = capB;
    }
}

static inline node createnode(unsigned int id, unsigned int level)
{
    node n;
    n.id = id;
    n.level = level;
    return n;
}

unsigned int get_node_count(unsigned int x, unsigned int level, unsigned int bitlen, unsigned int* nodecount) {
    if (x == 0) return 0;

    unsigned int total = 1U << bitlen;

    unsigned int block = total >> level;

    if (block == 0) return 0;

    *nodecount = block;

    return (x + block - 1) / block;
}

CountResult processLayer(Queue* oldQ,
    MinHeap* oldHeap,
    HashMap* oldMap,
    Queue* newQ,
    node* uniqueBuf,
    int* uniqueSize,
    HashMap* visited,
    unsigned int num,
    unsigned int totallevel,
    unsigned int target_level,
    unsigned int nodecount,
    CountResult res,
    unsigned int t)
{
    *uniqueSize = 0;
    unsigned int zeroCount = 0;
    unsigned int oneCount = 0;
    unsigned int added = 0;

    while (!isQueueEmpty(oldQ)) {
        node cur = dequeue(oldQ);
        int idx = hashFind(oldMap, cur);


        if (idx == -1) {

            if (added + 2 <= num) {
                enqueue(newQ, cur);
                enqueue(newQ, cur);
                added += 2;
            }
            continue;
        }

        Node hnode = oldHeap->data[idx];
        node children[2] = { hnode.left, hnode.right };

        for (int k = 0; k < 2; k++) {
            added++;


            if (added > num) {
                break;
            }

            node c = children[k];
            if (c.id == (unsigned int)-1) {
                continue;
            }


            if (c.level == totallevel) {
                if (c.id == 0) {
                    if (added == num) {
                        zeroCount += (t - res.zeroCount - (num - 1) * nodecount + 1);
                    }
                    else {
                        zeroCount += nodecount;
                    }
                }
                else if (c.id == 1) {
                    if (added == num) {
                        oneCount += (t - res.zeroCount - (num - 1) * nodecount + 1);
                    }
                    else {
                        oneCount += nodecount;
                    }
                }
                continue;
            }


            enqueue(newQ, c);


            if (c.level == target_level) {
                if (hashFind(visited, c) == -1) {
                    hashInsert(visited, c, 1);
                    uniqueBuf[*uniqueSize] = c;
                    (*uniqueSize)++;
                }
            }
        }
    }

    res.zeroCount += zeroCount;
    res.oneCount += oneCount;
    return res;
}



unsigned int GetDirectNode(const Graph_Compress& CG, unsigned int nodeid, 
    unsigned int startlevel, unsigned int path, int flag, unsigned int* endlevel) {
#if defined(QUERYBDD_REFERENCE_BUILD)
    return ReferenceGetDirectNode(
        CG, nodeid, startlevel, path, flag, endlevel);
#else
    return AcceleratedGetDirectNode(
        CG, nodeid, startlevel, path, flag, endlevel);
#endif
}

unsigned int CountTimeNum(const Graph_Compress& G, unsigned int nodev, unsigned int startlevel, unsigned int t) {
    unsigned int level = startlevel;
    unsigned int totallevel = 2 * G.nips_v + G.nips_t;
    unsigned int num = 0, nodecount = 0;
    CountResult result;
    result.oneCount = 0;
    result.zeroCount = 0;

    int nips = ComputeRequiredBits(t);
    while (level < startlevel + G.nips_t - nips) {
        nodev = UnpackOne(G, 2 * nodev, &level);
        if (level == totallevel && nodev == 0) {
            return 0;
        }
        if (level == totallevel && nodev == 1) {
            return t + 1u;
        }
    }

    int temp = startlevel + G.nips_t - nips;
    int Multiple = 1;
    while (temp < (int)level) {
        Multiple *= 2;
        temp++;
    }

    unsigned int initCap = (Multiple > 0) ? (unsigned int)Multiple : 1u;

    static thread_local QueryScratch scratch = { 0 };
    prepareQueryScratch(&scratch, initCap);

    Queue* nodequeue = scratch.qA;
    Queue* newNodequeue = scratch.qB;

    MinHeap* visitlist = scratch.heapA;
    MinHeap* newVisitlist = scratch.heapB;

    HashMap* map = scratch.mapA;
    HashMap* newMap = scratch.mapB;
    HashMap* visited = scratch.visited;

    node* uniqueBuf = scratch.uniqueBuf;
    unsigned int uniqueCap = scratch.uniqueCap;

    Node visitnode;
    visitnode.parent = createnode(nodev, level);
    visitnode.left = createnode(NULL_ID, 0);
    visitnode.right = createnode(NULL_ID, 0);

    node queuenode = createnode(nodev, level);
    for (int i = 0; i < Multiple; i++) {
        enqueue(nodequeue, queuenode);
    }

    heapPush(visitlist, visitnode);

    ensureHashCapacity(map, 2 * (visitlist->size ? visitlist->size : 1));
    buildHeapMap(visitlist, map);

    while (level < totallevel && result.oneCount + result.zeroCount < t + 1) {
        num = get_node_count(t + 1, level - totallevel + nips + 1, nips, &nodecount);
        num -= (result.oneCount + result.zeroCount) / nodecount;

        if (visitlist->size) {
            visitlist = UnpackAll(G, visitlist);
        }
        else {
            ensureQueueCapacity(newNodequeue, 2 * (nodequeue->size ? nodequeue->size : 1));
            resetQueue(newNodequeue);

            ensureHeapCapacity(newVisitlist, 2 * (int)(nodequeue->size ? nodequeue->size : 1));
            newVisitlist->size = 0;

            ensureHashCapacity(newMap, 2 * (int)(nodequeue->size ? nodequeue->size : 1));

            int oldSize = (int)nodequeue->size;
            for (int i = 0; i < oldSize; i++) {
                node cur = dequeue(nodequeue);
                enqueue(newNodequeue, cur);
                enqueue(newNodequeue, cur);
            }
            buildHeapLinear(newVisitlist, newNodequeue->data, (int)newNodequeue->size);
            buildHeapMap(newVisitlist, newMap);

            Queue* tq = nodequeue; nodequeue = newNodequeue; newNodequeue = tq;
            MinHeap* th = visitlist; visitlist = newVisitlist; newVisitlist = th;
            HashMap* tm = map; map = newMap; newMap = tm;

            level++;
            continue;
        }

        unsigned int needCap = 2 * (nodequeue->size ? nodequeue->size : 1);
        ensureQueueCapacity(newNodequeue, needCap);
        resetQueue(newNodequeue);

        ensureHeapCapacity(newVisitlist, (int)needCap);
        newVisitlist->size = 0;

        ensureHashCapacity(newMap, (int)needCap);
        ensureHashCapacity(visited, (int)needCap);

        if (uniqueCap < needCap) {
            uniqueBuf = (node*)realloc(uniqueBuf, sizeof(node) * needCap);
            uniqueCap = needCap;
            scratch.uniqueBuf = uniqueBuf;
            scratch.uniqueCap = uniqueCap;
        }
        int uniqueSize = 0;

        level++;
        result = processLayer(nodequeue, visitlist, map, newNodequeue, uniqueBuf, &uniqueSize,
            visited, num, totallevel, level, nodecount, result, t);
        buildHeapLinear(newVisitlist, uniqueBuf, uniqueSize);
        buildHeapMap(newVisitlist, newMap);

        Queue* tq = nodequeue; nodequeue = newNodequeue; newNodequeue = tq;
        MinHeap* th = visitlist; visitlist = newVisitlist; newVisitlist = th;
        HashMap* tm = map; map = newMap; newMap = tm;
    }

    return result.oneCount;
}


bool CompareTimeParity(const Graph_Compress& G, unsigned int nodev, unsigned int startlevel, unsigned int t)
{
#if defined(QUERYBDD_REFERENCE_BUILD)
    return ReferenceCompareTimeParity(G, nodev, startlevel, t);
#else
    return AcceleratedCompareTimeParity(G, nodev, startlevel, t);
#endif
}

bool CompareTime(const Graph_Compress& G, unsigned int nodev, unsigned int startlevel, unsigned int t)
{
#if defined(QUERYBDD_REFERENCE_BUILD)
    return ReferenceCompareTime(G, nodev, startlevel, t);
#else
    return AcceleratedCompareTime(G, nodev, startlevel, t);
#endif
}

namespace {

void EnsureDirectNeighborCapacity(NeighborResult* out, unsigned int need)
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

    if (static_cast<std::size_t>(new_capacity) >
        SIZE_MAX / sizeof(unsigned int)) {
        fprintf(stderr, "Neighbor result capacity overflow\n");
        exit(EXIT_FAILURE);
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

bool AppendDirectNeighborInterval(
    NeighborResult* out,
    std::uint64_t prefix,
    unsigned int depth,
    unsigned int vertex_bits,
    unsigned int nodenum)
{
    if (depth > vertex_bits || vertex_bits > 32u) {
        return false;
    }

    const unsigned int remaining = vertex_bits - depth;
    const std::uint64_t begin = prefix << remaining;
    if (begin > nodenum) {
        return true;
    }

    std::uint64_t end = begin;
    if (remaining != 0) {
        end += (1ull << remaining) - 1ull;
    }
    end = (std::min)(end, static_cast<std::uint64_t>(nodenum));

    const std::uint64_t add_count = end - begin + 1ull;
    const std::uint64_t required =
        static_cast<std::uint64_t>(out->Count) + add_count;
    if (required > UINT_MAX) {
        return false;
    }

    EnsureDirectNeighborCapacity(out, static_cast<unsigned int>(required));
    for (std::uint64_t value = begin; value <= end; ++value) {
        out->List[out->Count++] = static_cast<unsigned int>(value);
    }
    return true;
}

bool DirectNeighborTimeActive(
    const Graph_Compress& G,
    unsigned int node_id,
    unsigned int node_level,
    unsigned int t,
    DataType dataset,
    bool* active)
{
    const unsigned int total_level = G.nips_v * 2u + G.nips_t;
    if (node_level > total_level || active == NULL) {
        return false;
    }
    if (node_level == total_level) {
        *active = node_id != 0 &&
            (dataset != DATA_INTERVAL ||
                ((static_cast<std::uint64_t>(t) + 1u) & 1u) != 0u);
        return true;
    }

    switch (dataset) {
    case DATA_POINT: {
        unsigned int end_level = node_level;
        const unsigned int terminal = GetDirectNode(
            G, node_id, node_level, t, 0, &end_level);
        *active = end_level == total_level && terminal != 0;
        return true;
    }
    case DATA_INTERVAL:
        *active = CompareTimeParity(G, node_id, node_level, t);
        return true;
    case DATA_INCREMENTAL:
        *active = CompareTime(G, node_id, node_level, t);
        return true;
    default:
        return false;
    }
}

bool CollectDirectNeighborsRecursive(
    const Graph_Compress& G,
    unsigned int node_id,
    unsigned int node_level,
    unsigned int t,
    unsigned int nodenum,
    DataType dataset,
    unsigned int depth,
    std::uint64_t prefix,
    NeighborResult* out)
{
    const unsigned int time_start = G.nips_v * 2u;
    const unsigned int total_level = time_start + G.nips_t;
    if (depth > G.nips_v || node_level > total_level) {
        return false;
    }

    const unsigned int remaining = G.nips_v - depth;
    if ((prefix << remaining) > nodenum) {
        return true;
    }

    if (node_level == total_level || node_level >= time_start ||
        depth == G.nips_v) {
        if (depth == G.nips_v && node_level < time_start) {
            return false;
        }

        bool active = false;
        if (!DirectNeighborTimeActive(
            G, node_id, node_level, t, dataset, &active)) {
            return false;
        }
        return !active || AppendDirectNeighborInterval(
            out, prefix, depth, G.nips_v, nodenum);
    }

    const unsigned int target_level = G.nips_v + depth;
    const std::uint64_t next_prefix = prefix << 1;
    if (node_level > target_level) {
        return CollectDirectNeighborsRecursive(
            G, node_id, node_level, t, nodenum, dataset,
            depth + 1u, next_prefix, out) &&
            CollectDirectNeighborsRecursive(
                G, node_id, node_level, t, nodenum, dataset,
                depth + 1u, next_prefix | 1ull, out);
    }
    if (node_level < target_level) {
        return false;
    }

    unsigned int low_level = node_level;
    unsigned int high_level = node_level;
    const unsigned int low = UnpackOne(G, node_id << 1, &low_level);
    const unsigned int high = UnpackOne(
        G, (node_id << 1) | 1u, &high_level);
    return CollectDirectNeighborsRecursive(
        G, low, low_level, t, nodenum, dataset,
        depth + 1u, next_prefix, out) &&
        CollectDirectNeighborsRecursive(
            G, high, high_level, t, nodenum, dataset,
            depth + 1u, next_prefix | 1ull, out);
}

}

bool CollectDirectNeighbors(
    const Graph_Compress& G,
    unsigned int start_node,
    unsigned int start_level,
    unsigned int t,
    unsigned int nodenum,
    DataType dataset,
    NeighborResult* out)
{
    if (out == NULL || G.nips_v > 32u) {
        return false;
    }
    return CollectDirectNeighborsRecursive(
        G, start_node, start_level, t, nodenum, dataset,
        0u, 0ull, out);
}

bool QueryEdgeActiveByType(
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

bool QueryDirectNeighborByType(
    const Graph_Compress& G,
    unsigned int u,
    unsigned int t,
    unsigned int nodenum,
    unsigned int timenum,
    DataType dataset,
    NeighborResult* out)
{
    if (out == NULL) {
        return false;
    }
    switch (dataset) {
    case DATA_POINT:
        QueryDirectNeighbor_Point(G, u, t, nodenum, timenum, out);
        break;
    case DATA_INTERVAL:
        QueryDirectNeighbor_Interval(G, u, t, nodenum, timenum, out);
        break;
    case DATA_INCREMENTAL:
        QueryDirectNeighbor_Incremental(G, u, t, nodenum, timenum, out);
        break;
    default:
        out->Count = 0u;
        return false;
    }

    if (out->Count > 1u) {
        std::sort(out->List, out->List + out->Count);
        out->Count = static_cast<unsigned int>(
            std::unique(out->List, out->List + out->Count) - out->List);
    }
    return true;
}
