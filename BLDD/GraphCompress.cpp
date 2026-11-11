#include "Graph.h"
#include "Help.h"
#include "Bit.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

DdLevelQueue* InitTraversalQueue(unsigned int levels, DdManager* manager)
{
    /* Match BLDD exactly: CUDD starts with its default two hash buckets. */
    DdLevelQueue* queue = cuddLevelQueueInit(
        static_cast<int>(levels), static_cast<int>(sizeof(DdQueueItem)),
        0, manager);
    return queue;
}

unsigned int NodeLevel(const DdNode* node, unsigned int terminalLevel)
{
    if (!node) {
        fprintf(stderr, "DD traversal encountered a null node\n");
        exit(EXIT_FAILURE);
    }
    if (Cudd_IsConstant(const_cast<DdNode*>(node))) return terminalLevel;
    return (unsigned int)node->index;
}

DdNode* DequeueOrExit(DdLevelQueue* queue, unsigned int level,
                      const char* phase)
{
    if (!queue || level >= (unsigned int)queue->levels || queue->size <= 0 ||
        queue->first == nullptr) {
        fprintf(stderr,
                "%s: inconsistent queue before dequeue level=%u size=%d "
                "first=%p last=%p levels=%d\n",
                phase, level, queue ? queue->size : -1,
                queue ? queue->first : nullptr,
                (queue && level < (unsigned int)queue->levels)
                    ? (void*)queue->last[level] : nullptr,
                queue ? queue->levels : -1);
        exit(EXIT_FAILURE);
    }
    DdNode* node = cuddLevelQueueDequeue(queue, (int)level);
    if (!node) {
        fprintf(stderr, "%s: CUDD returned a null queue item at level %u\n",
                phase, level);
        exit(EXIT_FAILURE);
    }
    return node;
}

void QueueFirstOrExit(DdLevelQueue* queue, DdNode* node, unsigned int level)
{
    if (!cuddLevelQueueFirst(queue, node, (int)level)) {
        fprintf(stderr, "DD traversal queue root insertion failed at level %u\n", level);
        exit(EXIT_FAILURE);
    }
}

void EnqueueOrExit(DdLevelQueue* queue, DdNode* node, unsigned int level)
{
    if (!cuddLevelQueueEnqueue(queue, node, (int)level)) {
        fprintf(stderr, "DD traversal queue insertion failed at level %u size=%d\n",
                level, queue ? queue->size : -1);
        exit(EXIT_FAILURE);
    }
}

unsigned int TotalLevels(const Str_Graph_new* graph)
{
    const uint64_t levels = (uint64_t)graph->nips_v * 2u + graph->nips_t + 1u;
    if (levels == 0u || levels > (uint64_t)INT_MAX) {
        fprintf(stderr, "Graph depth exceeds traversal queue range\n");
        exit(EXIT_FAILURE);
    }
    return (unsigned int)levels;
}

uint64_t LayerStart(const uint64_t* levelEnd, unsigned int level)
{
    return level == 0u ? 0u : levelEnd[level - 1u];
}

uint64_t GlobalNodeID(const DdNode* node, const uint64_t* levelEnd,
                      unsigned int totallevel)
{
    DdNode* regular = node ? Cudd_Regular(const_cast<DdNode*>(node)) : nullptr;
    if (!node || !levelEnd ||
        Cudd_IsConstantInt(regular) || regular->index >= totallevel) {
        fprintf(stderr, "GlobalNodeID: invalid non-terminal node\n");
        exit(EXIT_FAILURE);
    }
    const unsigned int level = (unsigned int)regular->index;
    const uint64_t start = LayerStart(levelEnd, level);
    const uint64_t end = levelEnd[level];
    if (end < start || regular->ID >= end - start) {
        fprintf(stderr, "GlobalNodeID: local ID outside assigned level %u\n", level);
        exit(EXIT_FAILURE);
    }
    if (regular->ID > UINT64_MAX - start) {
        fprintf(stderr, "GlobalNodeID: ID addition overflow at level %u\n", level);
        exit(EXIT_FAILURE);
    }
    const uint64_t global = start + regular->ID;
    if (global >= levelEnd[totallevel - 1u]) {
        fprintf(stderr, "GlobalNodeID: local ID outside assigned level %u\n", level);
        exit(EXIT_FAILURE);
    }
    return global;
}

uint64_t ChildGlobalID(const DdNode* child, const uint64_t* levelEnd,
                       unsigned int totallevel)
{
    DdNode* regular = child ? Cudd_Regular(const_cast<DdNode*>(child)) : nullptr;
    if (!regular) {
        fprintf(stderr, "GlobalNodeID: null child\n");
        exit(EXIT_FAILURE);
    }
    if (Cudd_IsConstantInt(regular)) {
        const unsigned int terminalLevel = totallevel - 1u;
        const uint64_t start = LayerStart(levelEnd, terminalLevel);
        const uint64_t end = levelEnd[terminalLevel];
        if (end < start || regular->ID >= end - start) {
            fprintf(stderr, "GlobalNodeID: terminal ID outside assigned level\n");
            exit(EXIT_FAILURE);
        }
        if (regular->ID > UINT64_MAX - start) {
            fprintf(stderr, "GlobalNodeID: terminal ID addition overflow\n");
            exit(EXIT_FAILURE);
        }
        const uint64_t global = start + regular->ID;
        if (global >= levelEnd[totallevel - 1u]) {
            fprintf(stderr, "GlobalNodeID: terminal ID outside assigned level\n");
            exit(EXIT_FAILURE);
        }
        return global;
    }
    return GlobalNodeID(regular, levelEnd, totallevel);
}

uint64_t CheckedDouble(uint64_t value, const char* label)
{
    if (value > UINT64_MAX / 2u) {
        fprintf(stderr, "%s: capacity overflow\n", label);
        exit(EXIT_FAILURE);
    }
    return value * 2u;
}

uint64_t BitWords(uint64_t bits, const char* label)
{
    if (bits > UINT64_MAX - (BIT_WORD - 1u)) {
        fprintf(stderr, "%s: bit count overflow\n", label);
        exit(EXIT_FAILURE);
    }
    return (bits + BIT_WORD - 1u) / BIT_WORD;
}

void EnqueueChild(DdLevelQueue* queue, DdLevelQueue* skipqueue,
                  DdNode* child, unsigned int level, unsigned int totallevel)
{
    const bool skipped = child->index != level + 1u &&
                         level != totallevel - 2u;
    DdLevelQueue* destination = skipped ? skipqueue : queue;
    const unsigned int childLevel =
        Cudd_IsConstant(child) ? totallevel - 1u : child->index;
    EnqueueOrExit(destination, child, childLevel);
}

void PromoteSkipped(DdLevelQueue* queue, DdLevelQueue* skipqueue,
                    unsigned int level, unsigned int totallevel)
{
    while (skipqueue->last[level]) {
        DdNode* node = DequeueOrExit(skipqueue, level, "PromoteSkipped");
        if (Cudd_IsConstant(node)) {
            EnqueueOrExit(queue, node, totallevel - 1u);
        } else {
            EnqueueOrExit(queue, node, node->index);
        }
    }
}

void SetMaxSkip(Graph_Array* graph, uint64_t value)
{
    if (value > graph->MaxSkip) graph->MaxSkip = value;
}

void PrefixLayerCounts(Graph_Array* graph, const uint64_t* layerCount,
                       unsigned int totallevel)
{
    uint64_t cumulative = 0;
    for (unsigned int level = 0; level < totallevel; ++level) {
        if (cumulative > UINT64_MAX - layerCount[level]) {
            fprintf(stderr, "AssignNodeID: cumulative node count overflow\n");
            exit(EXIT_FAILURE);
        }
        cumulative += layerCount[level];
        graph->VariatyNumList[level] = cumulative;
    }
    graph->totalNodes = cumulative;
}

} // namespace

Str_Graph_new* AssignNodeID(DdManager* manager, Str_Graph_new* graph,
                            Graph_Array* arrays)
{
    if (!manager || !graph || !arrays || !graph->Event || Cudd_IsConstant(graph->Event)) {
        fprintf(stderr, "AssignNodeID: invalid or constant graph\n");
        return nullptr;
    }
    const unsigned int totallevel = TotalLevels(graph);
    DdNode* root = graph->Event;
    const unsigned int rootLevel = NodeLevel(root, totallevel - 1u);
    if (rootLevel >= totallevel) {
        fprintf(stderr, "AssignNodeID: root level is outside graph depth\n");
        return nullptr;
    }

    DdLevelQueue* queue = InitTraversalQueue(totallevel, manager);
    DdLevelQueue* skipqueue = InitTraversalQueue(totallevel, manager);
    if (!queue || !skipqueue) {
        fprintf(stderr, "AssignNodeID: queue allocation failed\n");
        exit(EXIT_FAILURE);
    }
    QueueFirstOrExit(queue, root, rootLevel);

    arrays->VariatyNumList = (uint64_t*)calloc(totallevel, sizeof(uint64_t));
    uint64_t* layerCount = (uint64_t*)calloc(totallevel, sizeof(uint64_t));
    if (!arrays->VariatyNumList || !layerCount) {
        fprintf(stderr, "AssignNodeID: allocation failed\n");
        exit(EXIT_FAILURE);
    }
    arrays->nips_t = graph->nips_t;
    arrays->nips_v = graph->nips_v;

    unsigned int level = rootLevel;
    while (queue->size || skipqueue->size) {
        while (level < totallevel && !queue->last[level]) {
            ++level;
            if (level >= totallevel) break;
            PromoteSkipped(queue, skipqueue, level, totallevel);
        }
        if (level >= totallevel) break;

        DdNode* current = DequeueOrExit(queue, level, "AssignNodeID");
        DdNode* regular = Cudd_Regular(current);
        if (Cudd_IsConstant(regular)) {
            regular->ID = cuddV(regular) == 0.0 ? 0u : 1u;
            continue;
        }
        const DdHalfWord indexBefore = regular->index;
        // CUDD stores the per-level node ID in an unsigned int.  Keep the
        // cumulative layer boundaries 64-bit, but reject a layer that cannot
        // be represented by the CUDD node field before assigning it.
        if (layerCount[level] > UINT_MAX) {
            fprintf(stderr, "AssignNodeID: layer-local ID overflow at level %u\n", level);
            exit(EXIT_FAILURE);
        }
        regular->ID = static_cast<unsigned int>(layerCount[level]++);
        if (regular->index != indexBefore) {
            fprintf(stderr,
                    "AssignNodeID: DdNode index changed while assigning ID "
                    "level=%u before=%u after=%u\n",
                    level, (unsigned int)indexBefore,
                    (unsigned int)regular->index);
            exit(EXIT_FAILURE);
        }

        if (!Cudd_IsConstant(current)) {
            EnqueueChild(queue, skipqueue, Cudd_E(current), level, totallevel);
            EnqueueChild(queue, skipqueue, Cudd_T(current), level, totallevel);
        }
    }

    if (queue->size || skipqueue->size) {
        fprintf(stderr, "AssignNodeID: queue level exceeded graph depth\n");
        exit(EXIT_FAILURE);
    }
    layerCount[totallevel - 1u] = 2u;
    PrefixLayerCounts(arrays, layerCount, totallevel);
    free(layerCount);
    cuddLevelQueueQuit(queue);
    cuddLevelQueueQuit(skipqueue);
    return graph;
}

Str_Graph_new* CreateArray(DdManager* manager, Str_Graph_new* graph,
                           Graph_Array* arrays)
{
    if (!manager || !graph || !arrays || !graph->Event || !arrays->VariatyNumList) {
        fprintf(stderr, "CreateArray: invalid input\n");
        return nullptr;
    }
    const unsigned int totallevel = TotalLevels(graph);
    DdNode* root = graph->Event;
    const unsigned int rootLevel = NodeLevel(root, totallevel - 1u);
    if (Cudd_IsConstant(root) || rootLevel >= totallevel) {
        fprintf(stderr, "CreateArray: invalid root\n");
        return nullptr;
    }

    DdLevelQueue* queue = InitTraversalQueue(totallevel, manager);
    DdLevelQueue* skipqueue = InitTraversalQueue(totallevel, manager);
    if (!queue || !skipqueue) {
        fprintf(stderr, "CreateArray: queue allocation failed\n");
        exit(EXIT_FAILURE);
    }
    QueueFirstOrExit(queue, root, rootLevel);

    const uint64_t edgeBits = CheckedDouble(arrays->totalNodes, "CreateArray edge");
    const uint64_t edgeWords = BitWords(edgeBits, "CreateArray edge");
    const uint64_t nodeWords = BitWords(arrays->totalNodes, "CreateArray node");
    arrays->NormalFlagList = InitUIntArray(edgeWords, totallevel);
    arrays->ShareFlagList = InitUIntArray(edgeWords, totallevel);
    arrays->SkipFlagList = InitUIntArray(edgeWords, totallevel);
    arrays->ShareNodeList = InitUIntArray(nodeWords, totallevel);
    arrays->SkipValue = InitU64Array(CheckedDouble(arrays->totalNodes, "CreateArray skip"), totallevel);

    uint64_t normalFlags = 0, shareFlags = 0, skipFlags = 0, skipValues = 0;
    uint64_t normalCountTotal = 0, shareCountTotal = 0;
    uint64_t skipCountTotal = 0, zeroCountTotal = 0;
    uint64_t normalcount = 0;
    arrays->MaxSkip = 0;
    unsigned int level = rootLevel;

    /* Keep BLDD's traversal condition exactly: the main queue drives the pass. */
    while (queue->size) {
        while (level < totallevel && !queue->last[level]) {
            arrays->NormalFlagList->Index[level] = normalFlags;
            arrays->ShareFlagList->Index[level] = shareFlags;
            arrays->SkipFlagList->Index[level] = skipFlags;
            arrays->SkipValue->Index[level] = skipValues;
            ++level;
            if (level >= totallevel) break;
            normalcount = 0;
            PromoteSkipped(queue, skipqueue, level, totallevel);
        }
        if (level >= totallevel) break;

        DdNode* current = DequeueOrExit(queue, level, "CreateArray");
        if (Cudd_IsConstant(current)) continue;

        if (level == totallevel - 2u) {
            if (cuddV(Cudd_E(current)) == 1) {
                BITSET_SET(arrays->NormalFlagList->List, normalFlags);
                normalFlags += 2;
            }
            if (cuddV(Cudd_T(current)) == 1) {
                ++normalFlags;
                BITSET_SET(arrays->NormalFlagList->List, normalFlags);
                ++normalFlags;
            }
            continue;
        }

        DdNode* children[2] = { Cudd_E(current), Cudd_T(current) };
        for (DdNode* child : children) {
            if (Cudd_IsConstant(child) && cuddV(child) == 0) {
                ++normalFlags;
                ++shareFlags;
                ++skipFlags;
                ++zeroCountTotal;
            } else {
                const uint64_t childGlobal = ChildGlobalID(child, arrays->VariatyNumList, totallevel);
                if (childGlobal < arrays->VariatyNumList[level + 1u] &&
                    childGlobal == arrays->VariatyNumList[level] + normalcount) {
                    BITSET_SET(arrays->NormalFlagList->List, normalFlags);
                    ++normalFlags;
                    ++normalcount;
                    ++normalCountTotal;
                } else if (childGlobal >= arrays->VariatyNumList[level + 1u]) {
                    BITSET_SET(arrays->SkipFlagList->List, skipFlags);
                    const uint64_t skip = childGlobal - arrays->VariatyNumList[level + 1u];
                    arrays->SkipValue->List[skipValues++] = skip;
                    SetMaxSkip(arrays, skip);
                    ++normalFlags;
                    ++shareFlags;
                    ++skipFlags;
                    ++skipCountTotal;
                } else {
                    BITSET_SET(arrays->ShareFlagList->List, shareFlags);
                    BITSET_SET(arrays->ShareNodeList->List, childGlobal);
                    ++normalFlags;
                    ++shareFlags;
                    ++shareCountTotal;
                }
            }
        }

        EnqueueChild(queue, skipqueue, Cudd_E(current), level, totallevel);
        EnqueueChild(queue, skipqueue, Cudd_T(current), level, totallevel);
    }

    if (queue->size || level >= totallevel) {
        fprintf(stderr, "CreateArray: queue level exceeded graph depth\n");
        exit(EXIT_FAILURE);
    }
    for (; level < totallevel; ++level) {
        arrays->NormalFlagList->Index[level] = normalFlags;
        arrays->ShareFlagList->Index[level] = shareFlags;
        arrays->SkipFlagList->Index[level] = skipFlags;
        arrays->SkipValue->Index[level] = skipValues;
    }
    memcpy(arrays->ShareNodeList->Index, arrays->VariatyNumList,
           sizeof(uint64_t) * totallevel);

    arrays->NormalFlagList->Count = normalFlags;
    arrays->ShareFlagList->Count = shareFlags;
    arrays->SkipFlagList->Count = skipFlags;
    arrays->SkipValue->Count = skipValues;
    arrays->ShareNodeList->Count = arrays->totalNodes;
    arrays->NormalNum = normalCountTotal;
    arrays->ShareNum = shareCountTotal;
    arrays->SkipNum = skipCountTotal;
    arrays->ZeroNum = zeroCountTotal;

    cuddLevelQueueQuit(queue);
    cuddLevelQueueQuit(skipqueue);
    return graph;
}

Str_Graph_new* AssignNewNodeID(DdManager* manager, Str_Graph_new* graph,
                               Graph_Array* arrays)
{
    if (!manager || !graph || !arrays || !graph->Event || !arrays->ShareNodeList) {
        fprintf(stderr, "AssignNewNodeID: invalid input\n");
        return nullptr;
    }
    const unsigned int totallevel = TotalLevels(graph);
    DdNode* root = graph->Event;
    const unsigned int rootLevel = NodeLevel(root, totallevel - 1u);
    if (Cudd_IsConstant(root) || rootLevel >= totallevel) {
        fprintf(stderr, "AssignNewNodeID: invalid root level\n");
        return nullptr;
    }

    DdLevelQueue* queue = InitTraversalQueue(totallevel, manager);
    DdLevelQueue* skipqueue = InitTraversalQueue(totallevel, manager);
    if (!queue || !skipqueue) {
        fprintf(stderr, "AssignNewNodeID: queue allocation failed\n");
        exit(EXIT_FAILURE);
    }
    QueueFirstOrExit(queue, root, rootLevel);
    if (arrays->totalNodes > SIZE_MAX / sizeof(uint64_t)) {
        fprintf(stderr, "AssignNewNodeID: allocation size overflow\n");
        exit(EXIT_FAILURE);
    }
    graph->NewID = (uint64_t*)calloc((size_t)(arrays->totalNodes ? arrays->totalNodes : 1u),
                                     sizeof(uint64_t));
    if (!graph->NewID) {
        fprintf(stderr, "AssignNewNodeID: allocation failed for %llu nodes\n",
                (unsigned long long)arrays->totalNodes);
        exit(EXIT_FAILURE);
    }
    for (uint64_t i = 0; i < arrays->totalNodes; ++i) graph->NewID[i] = UINT64_MAX;

    uint64_t newID = 0;
    unsigned int level = rootLevel;
    while (queue->size || skipqueue->size) {
        while (level < totallevel && !queue->last[level]) {
            ++level;
            newID = 0;
            if (level >= totallevel) break;
            PromoteSkipped(queue, skipqueue, level, totallevel);
        }
        if (level >= totallevel) break;

        DdNode* current = DequeueOrExit(queue, level, "AssignNewNodeID");
        if (!Cudd_IsConstant(current)) {
            const uint64_t global = GlobalNodeID(current, arrays->VariatyNumList, totallevel);
            if (BITSET_GET(arrays->ShareNodeList->List, global)) {
                if (newID == UINT64_MAX) {
                    fprintf(stderr, "AssignNewNodeID: shared ID overflow at level %u\n", level);
                    exit(EXIT_FAILURE);
                }
                graph->NewID[global] = newID++;
            }
            EnqueueChild(queue, skipqueue, Cudd_E(current), level, totallevel);
            EnqueueChild(queue, skipqueue, Cudd_T(current), level, totallevel);
        }
    }
    if (queue->size || skipqueue->size) {
        fprintf(stderr, "AssignNewNodeID: queue level exceeded graph depth\n");
        exit(EXIT_FAILURE);
    }
    cuddLevelQueueQuit(queue);
    cuddLevelQueueQuit(skipqueue);
    return graph;
}

Str_Graph_new* CreateShareArray(DdManager* manager, Str_Graph_new* graph,
                                Graph_Array* arrays)
{
    if (!manager || !graph || !arrays || !graph->Event || !graph->NewID ||
        !arrays->VariatyNumList || !arrays->ShareNodeList) {
        fprintf(stderr, "CreateShareArray: invalid input\n");
        return nullptr;
    }
    const unsigned int totallevel = TotalLevels(graph);
    DdNode* root = graph->Event;
    const unsigned int rootLevel = NodeLevel(root, totallevel - 1u);
    if (Cudd_IsConstant(root) || rootLevel >= totallevel) {
        fprintf(stderr, "CreateShareArray: invalid root level\n");
        return nullptr;
    }

    DdLevelQueue* queue = InitTraversalQueue(totallevel, manager);
    DdLevelQueue* skipqueue = InitTraversalQueue(totallevel, manager);
    if (!queue || !skipqueue) {
        fprintf(stderr, "CreateShareArray: queue allocation failed\n");
        exit(EXIT_FAILURE);
    }
    QueueFirstOrExit(queue, root, rootLevel);
    arrays->ShareValue = InitU64Array(CheckedDouble(arrays->totalNodes,
                                                     "CreateShareArray"), totallevel);

    uint64_t shareValues = 0;
    uint64_t normalcount = 0;
    unsigned int level = rootLevel;
    /* Keep BLDD's traversal condition exactly: the main queue drives the pass. */
    while (queue->size) {
        while (level < totallevel && !queue->last[level]) {
            arrays->ShareValue->Index[level] = shareValues;
            ++level;
            if (level >= totallevel) break;
            normalcount = 0;
            PromoteSkipped(queue, skipqueue, level, totallevel);
        }
        if (level >= totallevel) break;

        DdNode* current = DequeueOrExit(queue, level, "CreateShareArray");
        if (Cudd_IsConstant(current)) continue;
        DdNode* children[2] = { Cudd_E(current), Cudd_T(current) };
        for (DdNode* child : children) {
            if (Cudd_IsConstant(child) && cuddV(child) == 0) continue;
            const uint64_t childGlobal = ChildGlobalID(child, arrays->VariatyNumList, totallevel);
            if (level == totallevel - 2u ||
                (childGlobal < arrays->VariatyNumList[level + 1u] &&
                 childGlobal == arrays->VariatyNumList[level] + normalcount)) {
                ++normalcount;
            } else if (childGlobal >= arrays->VariatyNumList[level + 1u]) {
                continue;
            } else {
                const uint64_t newValue = graph->NewID[childGlobal];
                if (newValue == UINT64_MAX) {
                    fprintf(stderr, "CreateShareArray: invalid shared node ID\n");
                    exit(EXIT_FAILURE);
                }
                arrays->ShareValue->List[shareValues++] = newValue;
            }
        }

        EnqueueChild(queue, skipqueue, Cudd_E(current), level, totallevel);
        EnqueueChild(queue, skipqueue, Cudd_T(current), level, totallevel);
    }

    if (queue->size || level >= totallevel) {
        fprintf(stderr, "CreateShareArray: queue level exceeded graph depth\n");
        exit(EXIT_FAILURE);
    }
    for (; level < totallevel; ++level) arrays->ShareValue->Index[level] = shareValues;
    arrays->ShareValue->Count = shareValues;

    cuddLevelQueueQuit(queue);
    cuddLevelQueueQuit(skipqueue);
    return graph;
}
