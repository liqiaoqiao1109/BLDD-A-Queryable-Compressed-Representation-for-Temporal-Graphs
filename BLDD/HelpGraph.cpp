#include <iostream>
#include"Help.h"
#include"Graph.h"
#include"Bit.h"
#include"QueryAccelerate.h"
#include<vector>
#include <intrin.h>
#include <stdlib.h>
#include <string.h>



Str_Graph_new* InitStrGraph() {
    Str_Graph_new* G = ALLOC(Str_Graph_new, 1);
    if (G == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return G;
}



Graph_Array* InitGraphArray() {
    Graph_Array* GA = ALLOC(Graph_Array, 1);
    if (GA == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return GA;
}


UIntArray* InitUIntArray(uint64_t Lcapacity, unsigned int Icapacity) {
    UIntArray* arr = ALLOC(UIntArray, 1);
    if (!arr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    if (Lcapacity > SIZE_MAX / sizeof(unsigned int)) {
        fprintf(stderr, "UIntArray capacity exceeds addressable memory\n");
        exit(EXIT_FAILURE);
    }
    arr->List = (unsigned int*)calloc(Lcapacity ? (size_t)Lcapacity : 1u, sizeof(unsigned int));
    arr->Index = (uint64_t*)calloc(Icapacity ? Icapacity : 1u, sizeof(uint64_t));

    if (!arr->List || !arr->Index) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    arr->Capacity = Lcapacity;
    arr->Count = 0;

    return arr;
}

U64Array* InitU64Array(uint64_t Lcapacity, unsigned int Icapacity) {
    U64Array* arr = (U64Array*)calloc(1, sizeof(U64Array));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    if (Lcapacity > SIZE_MAX / sizeof(uint64_t)) {
        fprintf(stderr, "U64Array capacity exceeds addressable memory\n");
        exit(EXIT_FAILURE);
    }
    arr->List = (uint64_t*)calloc(Lcapacity ? (size_t)Lcapacity : 1u, sizeof(uint64_t));
    arr->Index = (uint64_t*)calloc(Icapacity ? Icapacity : 1u, sizeof(uint64_t));
    if (!arr->List || !arr->Index) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    arr->Capacity = Lcapacity;
    return arr;
}



BitArray* InitBitArray(uint64_t listSize, unsigned int indexSize, unsigned int flagSize)
{
    if (listSize > (uint64_t)(SIZE_MAX / sizeof(unsigned int))) return NULL;
    BitArray* arr = (BitArray*)malloc(sizeof(BitArray));
    if (arr == NULL) return NULL;
    arr->List = (unsigned int*)calloc((size_t)listSize, sizeof(unsigned int));
    arr->Index = (uint64_t*)calloc(indexSize, sizeof(uint64_t));
    arr->Flag = (unsigned int*)calloc(flagSize, sizeof(unsigned int));
    arr->Count = 0;
    if (arr->List == NULL || arr->Index == NULL || arr->Flag == NULL)
    {
        free(arr->List);
        free(arr->Index);
        free(arr->Flag);
        free(arr);
        return NULL;
    }
    return arr;
}


RunBitArray* InitRunBitArray(uint64_t listSize, unsigned int indexSize, unsigned int flagSize)
{
    if (listSize > (uint64_t)(SIZE_MAX / sizeof(unsigned int))) return NULL;
    RunBitArray* arr = (RunBitArray*)malloc(sizeof(RunBitArray));
    if (arr == NULL) return NULL;


    arr->List = (unsigned int*)calloc((size_t)listSize, sizeof(unsigned int));
    arr->Index = (uint64_t*)calloc(indexSize, sizeof(uint64_t));
    arr->Flag = (unsigned int*)calloc(flagSize, sizeof(unsigned int));
    arr->First = (unsigned int*)calloc(indexSize, sizeof(unsigned int));
    arr->Count = 0;

    if (arr->List == NULL || arr->Index == NULL ||
        arr->Flag == NULL || arr->First == NULL)
    {
        free(arr->List);
        free(arr->Index);
        free(arr->Flag);
        free(arr->First);
        free(arr);
        return NULL;
    }

    return arr;
}


FixBitArray* InitFixBitArray(uint64_t listSize, unsigned int indexSize, unsigned int bitsPerElement)
{
    if (listSize > (uint64_t)(SIZE_MAX / sizeof(unsigned int))) return NULL;
    FixBitArray* arr = (FixBitArray*)malloc(sizeof(FixBitArray));
    if (arr == NULL) return NULL;

    arr->BitsPerElement = bitsPerElement;
    arr->List = (unsigned int*)calloc((size_t)listSize, sizeof(unsigned int));
    arr->Index = (uint64_t*)calloc(indexSize, sizeof(uint64_t));
    arr->Count = 0;

    if (arr->List == NULL || arr->Index == NULL)
    {
        free(arr->List);
        free(arr->Index);
        free(arr);
        return NULL;
    }

    return arr;
}


RiceArray* InitRiceArray(uint64_t quoSize, uint64_t remSize, unsigned int indexSize, unsigned int paramSize)
{
    if (quoSize > (uint64_t)(SIZE_MAX / sizeof(unsigned int)) ||
        remSize > (uint64_t)(SIZE_MAX / sizeof(unsigned int))) return NULL;
    RiceArray* arr = (RiceArray*)malloc(sizeof(RiceArray));
    if (arr == NULL)
        return NULL;

    // Keep empty streams backed by valid storage so small graphs remain portable.
    const uint64_t quotientWords = quoSize == 0u ? 1u : quoSize;
    const uint64_t remainderWords = remSize == 0u ? 1u : remSize;
    arr->quotient = (unsigned int*)calloc(
        (size_t)quotientWords, sizeof(unsigned int));
    arr->remainder = (unsigned int*)calloc(
        (size_t)remainderWords, sizeof(unsigned int));

    arr->quoIndex = (uint64_t*)calloc(indexSize, sizeof(uint64_t));
    arr->remIndex = (uint64_t*)calloc(indexSize, sizeof(uint64_t));

    arr->RiceParameter = (unsigned int*)calloc(paramSize, sizeof(unsigned int));

    if (!arr->quotient || !arr->remainder ||
        !arr->quoIndex || !arr->remIndex ||
        !arr->RiceParameter)
    {
        free(arr->quotient);
        free(arr->remainder);
        free(arr->quoIndex);
        free(arr->remIndex);
        free(arr->RiceParameter);
        free(arr);
        return NULL;
    }

    arr->quoCount = 0;
    arr->remCount = 0;

    return arr;
}


Graph_Compress* InitGraphCompress()
{
    Graph_Compress* G = (Graph_Compress*)calloc(1, sizeof(Graph_Compress));
    if (G == NULL) return NULL;

    G->VariatyNumList = NULL;
    G->order[0] = 'U'; G->order[1] = 'V'; G->order[2] = 'T'; G->order[3] = '\0';
    G->NormalFlagList = NULL;
    G->ShareFlagList = NULL;
    G->SkipFlagList = NULL;
    G->ShareNodeList = NULL;
    G->ShareValue = NULL;
    G->SkipValue = NULL;

    return G;
}


void FreeBitArray(BitArray* arr)
{
    if (arr == NULL) return;
    free(arr->List);
    free(arr->Index);
    free(arr->Flag);
    free(arr);
}

void FreeRunBitArray(RunBitArray* arr)
{
    if (arr == NULL) return;

    free(arr->List);
    free(arr->Index);
    free(arr->First);
    free(arr->Flag);

    free(arr);
}

void FreeFixBitArray(FixBitArray* arr)
{
    if (arr == NULL) return;

    free(arr->List);
    free(arr->Index);

    free(arr);
}

void FreeRiceArray(RiceArray* arr)
{
    if (!arr) return;

    free(arr->quotient);
    free(arr->remainder);
    free(arr->quoIndex);
    free(arr->remIndex);
    free(arr->RiceParameter);

    free(arr);
}

void FreeGraphCompress(Graph_Compress* G)
{
    if (G == NULL) return;

    FreeQueryAcceleration(*G);
    free(G->VariatyNumList);

    FreeBitArray(G->NormalFlagList);
    FreeBitArray(G->ShareFlagList);
    FreeRunBitArray(G->SkipFlagList);
    FreeRunBitArray(G->ShareNodeList);
    FreeRiceArray(G->ShareValue);
    FreeFixBitArray(G->SkipValue);

    free(G);
}


void FreeStrGraph(DdManager* manager, Str_Graph_new* G) {
    if (G == NULL)
        return;

    
    if (G->Event != NULL) {
        Cudd_RecursiveDeref(manager, G->Event);
        G->Event = NULL;
    }
    
    if (G->x != NULL) {
        free(G->x);
        G->x = NULL;
    }
    if (G->y != NULL) {
        free(G->y);
        G->y = NULL;
    }
    if (G->t != NULL) {
        free(G->t);
        G->t = NULL;
    }

    
    if (G->NewID != NULL) {
        free(G->NewID);
        G->NewID = NULL;
    }

    
    free(G);
}


void FreeGraphArray(Graph_Array* GA) {
    if (GA) {
        free(GA->VariatyNumList);
        FreeUIntArray(GA->NormalFlagList);
        FreeUIntArray(GA->ShareFlagList);
        FreeUIntArray(GA->SkipFlagList);
        FreeUIntArray(GA->ShareNodeList);
        FreeU64Array(GA->ShareValue);
        FreeU64Array(GA->SkipValue);
        free(GA);
    }
}


void FreeUIntArray(UIntArray* arr) {
    if (arr == NULL) return;

    if (arr->List) {
        FREE(arr->List);
        arr->List = NULL;
    }

    if (arr->Index) {
        FREE(arr->Index);
        arr->Index = NULL;
    }

    FREE(arr);
}

void FreeU64Array(U64Array* arr) {
    if (arr == NULL) return;
    free(arr->List);
    free(arr->Index);
    free(arr);
}
