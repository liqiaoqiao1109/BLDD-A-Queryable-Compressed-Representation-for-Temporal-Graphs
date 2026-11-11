#define _CRT_SECURE_NO_WARNINGS
#include "Graph.h"
#include "Help.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

unsigned int BitsForMax(uint64_t value)
{
    unsigned int bits = 0u;
    while (value != 0u) {
        ++bits;
        value >>= 1u;
    }
    return bits;
}

DdNode* CreateNode64(
    DdManager* manager,
    uint64_t value,
    unsigned int bit_count,
    DdNode** variables)
{
    DdNode* result = Cudd_ReadOne(manager);
    Cudd_Ref(result);
    for (unsigned int offset = 0u; offset < bit_count; ++offset) {
        const unsigned int bit_index = bit_count - offset - 1u;
        DdNode* next = nullptr;
        if (((value >> bit_index) & 1u) == 0u) {
            next = Cudd_addIte(manager, variables[offset],
                Cudd_ReadZero(manager), result);
        }
        else {
            next = Cudd_addIte(manager, result, variables[offset],
                Cudd_ReadZero(manager));
        }
        if (next == nullptr) {
            Cudd_RecursiveDeref(manager, result);
            return nullptr;
        }
        Cudd_Ref(next);
        Cudd_RecursiveDeref(manager, result);
        result = next;
    }
    return result;
}

void ReleaseGraphVariables(DdManager* manager, Str_Graph_new* graph)
{
    DdNode*** arrays[] = { &graph->x, &graph->t, &graph->y };
    const unsigned int counts[] = {
        graph->nips_v, graph->nips_t, graph->nips_v
    };
    for (unsigned int array = 0; array < 3; ++array) {
        if (*arrays[array] == nullptr) continue;
        for (unsigned int i = 0; i < counts[array]; ++i) {
            if ((*arrays[array])[i] != nullptr)
                Cudd_RecursiveDeref(manager, (*arrays[array])[i]);
        }
        std::free(*arrays[array]);
        *arrays[array] = nullptr;
    }
}

bool FlushTime(
    DdManager* manager,
    Str_Graph_new* graph,
    uint64_t time,
    DdNode** time_events,
    DdNode** source_events)
{
    DdNode* condition = CreateNode64(manager, time, graph->nips_t, graph->t);
    if (condition == nullptr) return false;
    DdNode* term = Cudd_addIte(
        manager, condition, *time_events, Cudd_ReadZero(manager));
    if (term == nullptr) {
        Cudd_RecursiveDeref(manager, condition);
        return false;
    }
    Cudd_Ref(term);
    Cudd_RecursiveDeref(manager, condition);
    DdNode* merged = Cudd_addApply(
        manager, Cudd_addPlus, *source_events, term);
    if (merged == nullptr) {
        Cudd_RecursiveDeref(manager, term);
        return false;
    }
    Cudd_Ref(merged);
    Cudd_RecursiveDeref(manager, term);
    Cudd_RecursiveDeref(manager, *source_events);
    Cudd_RecursiveDeref(manager, *time_events);
    *source_events = merged;
    *time_events = Cudd_ReadZero(manager);
    Cudd_Ref(*time_events);
    return true;
}

bool FlushSource(
    DdManager* manager,
    Str_Graph_new* graph,
    uint64_t source,
    DdNode** source_events,
    DdNode** root)
{
    DdNode* condition = CreateNode64(manager, source, graph->nips_v, graph->x);
    if (condition == nullptr) return false;
    DdNode* term = Cudd_addIte(
        manager, condition, *source_events, Cudd_ReadZero(manager));
    if (term == nullptr) {
        Cudd_RecursiveDeref(manager, condition);
        return false;
    }
    Cudd_Ref(term);
    Cudd_RecursiveDeref(manager, condition);
    DdNode* merged = Cudd_addApply(manager, Cudd_addOr, *root, term);
    if (merged == nullptr) {
        Cudd_RecursiveDeref(manager, term);
        return false;
    }
    Cudd_Ref(merged);
    Cudd_RecursiveDeref(manager, term);
    Cudd_RecursiveDeref(manager, *source_events);
    Cudd_RecursiveDeref(manager, *root);
    *source_events = Cudd_ReadZero(manager);
    Cudd_Ref(*source_events);
    *root = merged;
    return true;
}

bool AddTarget(
    DdManager* manager,
    Str_Graph_new* graph,
    uint64_t target,
    DdNode** time_events)
{
    DdNode* condition = CreateNode64(manager, target, graph->nips_v, graph->y);
    if (condition == nullptr) return false;
    DdNode* edge = Cudd_addIte(
        manager, condition, Cudd_ReadOne(manager), Cudd_ReadZero(manager));
    if (edge == nullptr) {
        Cudd_RecursiveDeref(manager, condition);
        return false;
    }
    Cudd_Ref(edge);
    Cudd_RecursiveDeref(manager, condition);
    DdNode* merged = Cudd_addApply(
        manager, Cudd_addPlus, *time_events, edge);
    if (merged == nullptr) {
        Cudd_RecursiveDeref(manager, edge);
        return false;
    }
    Cudd_Ref(merged);
    Cudd_RecursiveDeref(manager, edge);
    Cudd_RecursiveDeref(manager, *time_events);
    *time_events = merged;
    return true;
}

} // namespace

Str_Graph_new* ReadGraphUTV(DdManager* manager, FILE* input, Str_Graph_new* graph)
{
    if (manager == nullptr || input == nullptr || graph == nullptr) {
        if (input != nullptr) std::fclose(input);
        return nullptr;
    }
    graph->x = nullptr;
    graph->y = nullptr;
    graph->t = nullptr;
    graph->Event = nullptr;
    graph->NewID = nullptr;

    unsigned long long nodes_read = 0;
    unsigned long long times_read = 0;
    unsigned long long event_count = 0;
    if (fscanf_s(input, "%llu %llu %llu", &nodes_read,
            &event_count, &times_read) != 3) {
        std::fprintf(stderr, "The file is empty or its header is invalid\n");
        std::fclose(input);
        return nullptr;
    }
    graph->nodes = static_cast<uint64_t>(nodes_read);
    graph->tinum = static_cast<uint64_t>(times_read);
    graph->evnum = static_cast<uint64_t>(event_count);
    graph->nips_v = BitsForMax(graph->nodes);
    graph->nips_t = BitsForMax(graph->tinum);
    graph->x = static_cast<DdNode**>(
        std::calloc(graph->nips_v, sizeof(DdNode*)));
    graph->y = static_cast<DdNode**>(
        std::calloc(graph->nips_v, sizeof(DdNode*)));
    graph->t = static_cast<DdNode**>(
        std::calloc(graph->nips_t, sizeof(DdNode*)));
    if ((graph->nips_v != 0u && (!graph->x || !graph->y)) ||
        (graph->nips_t != 0u && !graph->t)) {
        std::fprintf(stderr, "Memory allocation failed\n");
        ReleaseGraphVariables(manager, graph);
        std::fclose(input);
        return nullptr;
    }

    unsigned int level = 0;
    DdNode*** variable_arrays[] = { &graph->x, &graph->t, &graph->y };
    const unsigned int variable_counts[] = {
        graph->nips_v, graph->nips_t, graph->nips_v
    };
    for (unsigned int array = 0; array < 3; ++array) {
        for (unsigned int i = 0; i < variable_counts[array]; ++i) {
            (*variable_arrays[array])[i] = Cudd_addIthVar(manager, level++);
            if ((*variable_arrays[array])[i] == nullptr) {
                std::fprintf(stderr, "CUDD variable creation failed\n");
                ReleaseGraphVariables(manager, graph);
                std::fclose(input);
                return nullptr;
            }
            Cudd_Ref((*variable_arrays[array])[i]);
        }
    }

    DdNode* zero = Cudd_ReadZero(manager);
    DdNode* root = zero;
    DdNode* source_events = zero;
    DdNode* time_events = zero;
    Cudd_Ref(root);
    Cudd_Ref(source_events);
    Cudd_Ref(time_events);
    bool have_record = false;
    bool failed = false;
    uint64_t current_source = 0;
    uint64_t current_time = 0;
    uint64_t previous[3] = { 0u, 0u, 0u };
    uint64_t records = 0;
    unsigned long long source_read = 0, target_read = 0, time_read = 0;
    uint64_t source = 0, target = 0, time = 0;
    int scanned = 0;
    while ((scanned = fscanf_s(input, "%llu %llu %llu", &source_read,
            &target_read, &time_read)) == 3) {
        source = static_cast<uint64_t>(source_read);
        target = static_cast<uint64_t>(target_read);
        time = static_cast<uint64_t>(time_read);
        if (source > graph->nodes || target > graph->nodes ||
            time > graph->tinum ||
            (have_record && (source < previous[0] ||
                (source == previous[0] && time < previous[1]) ||
                (source == previous[0] && time == previous[1] &&
                    target < previous[2])))) {
            std::fprintf(stderr,
                "UTV input is out of bounds or not sorted by source, time, target at record %llu\n",
                static_cast<unsigned long long>(records + 1u));
            failed = true;
            break;
        }
        if (have_record && source == previous[0] && time == previous[1] && target == previous[2]) {
            ++records;
            continue;
        }
        previous[0] = source;
        previous[1] = time;
        previous[2] = target;

        if (!have_record) {
            current_source = source;
            current_time = time;
            have_record = true;
        }
        else if (source != current_source) {
            if (!FlushTime(manager, graph, current_time,
                    &time_events, &source_events) ||
                !FlushSource(manager, graph, current_source,
                    &source_events, &root)) {
                failed = true;
                break;
            }
            current_source = source;
            current_time = time;
        }
        else if (time != current_time) {
            if (!FlushTime(manager, graph, current_time,
                    &time_events, &source_events)) {
                failed = true;
                break;
            }
            current_time = time;
        }
        if (!AddTarget(manager, graph, target, &time_events)) {
            failed = true;
            break;
        }
        ++records;
    }
    if (scanned != EOF && scanned != 3) failed = true;
    if (records != graph->evnum) {
        std::fprintf(stderr, "UTV event count mismatch: header=%llu read=%llu\n",
            static_cast<unsigned long long>(graph->evnum),
            static_cast<unsigned long long>(records));
        failed = true;
    }
    if (!failed && have_record &&
        (!FlushTime(manager, graph, current_time, &time_events, &source_events) ||
         !FlushSource(manager, graph, current_source, &source_events, &root))) {
        failed = true;
    }
    Cudd_RecursiveDeref(manager, source_events);
    Cudd_RecursiveDeref(manager, time_events);
    if (failed) {
        Cudd_RecursiveDeref(manager, root);
        ReleaseGraphVariables(manager, graph);
        std::fclose(input);
        return nullptr;
    }
    graph->Event = root;
    std::fclose(input);
    return graph;
}
