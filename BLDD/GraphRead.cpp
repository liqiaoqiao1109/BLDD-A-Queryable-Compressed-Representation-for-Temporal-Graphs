#include"Graph.h"
#include"Help.h"


int ComputeRequiredBits(unsigned int value)
{
    int floorLog = 0;
    unsigned int copyv;
    copyv = value;
    while (value > 1)
    {
        floorLog++;
        value >>= 1;
    }
    if (copyv == (unsigned int)pow(2, floorLog))
        return (floorLog);
    else return(floorLog + 1);
}

unsigned int ComputeRequiredBits64(uint64_t value)
{
    if (value <= 1u) return 0u;
    unsigned int bits = 0u;
    uint64_t copy = value;
    while (copy > 1u) {
        ++bits;
        copy >>= 1u;
    }
    return ((UINT64_C(1) << bits) == value) ? bits : bits + 1u;
}

unsigned int ComputeRequiredBitsForMax64(uint64_t value)
{
    unsigned int bits = 0u;
    while (value != 0u) {
        ++bits;
        value >>= 1u;
    }
    return bits;
}

DdNode* CreateNode(DdManager* manager, unsigned int v, unsigned int npis, DdNode** x)
{
    DdNode* p, * one, * zero, * func;
    unsigned int i, j, pos;

    one = Cudd_ReadOne(manager);
    zero = Cudd_ReadZero(manager);

    i = npis - 1;
    Cudd_Ref(func = one);

    for (j = i; j != (unsigned int)-1; j--)
    {
        pos = (unsigned int)((v >> j) & 1u);
        if (pos == 0)
            Cudd_Ref(p = Cudd_addIte(manager, x[i - j], zero, func));
        else Cudd_Ref(p = Cudd_addIte(manager, func, x[i - j], zero));
        Cudd_RecursiveDeref(manager, func);
        func = p;
    }

    return (func);
}


Str_Graph_new* ReadGraph(DdManager* manager, FILE* f, Str_Graph_new* G) {
    unsigned int nodenum, timenum;
    unsigned long long evnum;
    unsigned int i = 0, level = 0, len = 0;
    unsigned int flag = 1;
    unsigned int su, sv, st, prest = 0, presu = 0, presv = 0;
    DdNode* events, * zero, ** event;
    DdNode* tunc, * xunc, * yunc, * edgp, * edgq, * tlog, * log;
    zero = Cudd_ReadZero(manager);
    events = zero;
    Cudd_Ref(events);


    unsigned int u = 0, v = 0, t = 0, last_u = 0, last_v = 0, last_t = 0;
    unsigned long long count = 0;
    int scanned = 0;
    bool have_previous = false;
    unsigned int previous_u = 0, previous_v = 0, previous_t = 0;
    if (EOF_Test(f)) goto failure1;
    if (fscanf_s(f, "%u %llu %u", &nodenum, &evnum, &timenum) != 3)
        goto failure1;
    // Validate the input before constructing the decision diagram.
    while ((scanned = fscanf_s(f, "%u %u %u", &u, &v, &t)) == 3) {
        if (u > nodenum || v > nodenum || t > timenum ||
            (count && (u < last_u || (u == last_u && v < last_v) ||
             (u == last_u && v == last_v && t < last_t)))) {
            fprintf(stderr, "UVT input is out of bounds or not sorted at record %llu\n", count + 1);
            goto failure1;
        }
        last_u = u; last_v = v; last_t = t;
        ++count;
    }
    if (scanned != EOF || count != evnum || count == 0 || nodenum == UINT_MAX || timenum == UINT_MAX) {
        fprintf(stderr, "Invalid UVT graph records or unsupported empty graph\n");
        goto failure1;
    }
    rewind(f);
    if (fscanf_s(f, "%u %llu %u", &nodenum, &evnum, &timenum) != 3) goto failure1;
    G->evnum = evnum;
    G->nodes = nodenum;
    G->tinum = timenum;

    len = 0;
    event = (DdNode**)malloc((nodenum + 1) * sizeof(DdNode*));


    G->nips_v = (unsigned int)(ComputeRequiredBits((unsigned int)(nodenum + 1)));
    G->nips_t = (unsigned int)(ComputeRequiredBits((unsigned int)(timenum + 1)));


    G->x = ALLOC(DdNode*, G->nips_v);
    if (G->x == NULL) goto failure2;
    G->y = ALLOC(DdNode*, G->nips_v);
    if (G->y == NULL) goto failure2;
    G->t = ALLOC(DdNode*, G->nips_t);
    if (G->t == NULL) goto failure2;



    for (i = 0; i < G->nips_v; i++) {
        G->x[i] = Cudd_addIthVar(manager, level++);
        Cudd_Ref(G->x[i]);
    }

    for (i = 0; i < G->nips_v; i++) {
        G->y[i] = Cudd_addIthVar(manager, level++);
        Cudd_Ref(G->y[i]);
    }

    for (i = 0; i < G->nips_t; i++) {
        G->t[i] = Cudd_addIthVar(manager, level++);
        Cudd_Ref(G->t[i]);
    }


    yunc = CreateNode(manager, 0, G->nips_v, G->y);
    while (fscanf_s(f, "%u %u %u", &su, &sv, &st) == 3) {
        if (have_previous && su == previous_u && sv == previous_v && st == previous_t) continue;
        previous_u = su; previous_v = sv; previous_t = st; have_previous = true;

        if (flag) {
            prest = st;
            presu = su;
            presv = sv;
            yunc = CreateNode(manager, sv, G->nips_v, G->y);
            flag = 0;
        }

        if (su != presu) {
            tunc = CreateNode(manager, presu, G->nips_v, G->x);
            tlog = Cudd_addIte(manager, tunc, events, zero);
            Cudd_Ref(tlog);
            Cudd_RecursiveDeref(manager, tunc);

            event[len] = tlog;
            events = zero;
            presu = su;
            len++;
        }

        if (sv != presv) {
            Cudd_RecursiveDeref(manager, yunc);
            yunc = CreateNode(manager, sv, G->nips_v, G->y);
            presv = sv;
        }
        xunc = CreateNode(manager, st, G->nips_t, G->t);

        edgq = Cudd_addIte(manager, xunc, yunc, zero);
        Cudd_Ref(edgq);
        Cudd_RecursiveDeref(manager, xunc);

        edgp = Cudd_addApply(manager, Cudd_addPlus, events, edgq);
        Cudd_Ref(edgp);
        Cudd_RecursiveDeref(manager, edgq);
        Cudd_RecursiveDeref(manager, events);
        events = edgp;
    }


    tunc = CreateNode(manager, presu, G->nips_v, G->x);
    tlog = Cudd_addIte(manager, tunc, events, zero);
    Cudd_Ref(tlog);
    Cudd_RecursiveDeref(manager, tunc);

    event[len] = tlog;
    events = zero;
    len++;

    while (len != 1) {
        for (unsigned int i = 0; i < len / 2; i++) {
            log = Cudd_addApply(manager, Cudd_addOr, event[i], event[len - i - 1]);
            Cudd_Ref(log);
            Cudd_RecursiveDeref(manager, event[i]);
            Cudd_RecursiveDeref(manager, event[len - i - 1]);
            event[i] = log;
        }

        if (len % 2 == 1) {
            len = len / 2 + 1;
        }
        else {
            len = len / 2;
        }
    }

    G->Event = event[0];
    Cudd_RecursiveDeref(manager, event[0]);
    Cudd_RecursiveDeref(manager, events);
    free(event);

    fclose(f);
    return(G);

failure1:
    fprintf(stderr, "The file is empty\n");
    fclose(f);
    return NULL;
failure2:
    fprintf(stderr, "Memory allocation failed\n");
    free(event);
    Cudd_RecursiveDeref(manager, events);
    fclose(f);
    return NULL;
}
