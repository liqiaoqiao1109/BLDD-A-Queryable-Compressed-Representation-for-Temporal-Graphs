#include <util.h>

#define TRUE 1
#define FALSE 0
#define NULL_ID ((unsigned int)-1)

typedef struct {
    unsigned int zeroCount;
    unsigned int oneCount;
} CountResult;

typedef struct node {
    unsigned int id;
    unsigned int level;
} node;

typedef struct Node {
    node parent;
    node left;
    node right;
}Node;

typedef struct Queue {
    node* data;
    unsigned int front;
    unsigned int rear;
    unsigned int size;
    unsigned int capacity;
}Queue;

struct MinHeap {
    Node* data;
    int size;
    int capacity;
};


typedef struct HashEntry {
    unsigned int id;
    unsigned int level;
    int heapIndex;
    int used;
} HashEntry;


typedef struct HashMap {
    HashEntry* table;
    int capacity;
    int size;
} HashMap;

typedef struct QueryScratch {
    Queue* qA;
    Queue* qB;
    MinHeap* heapA;
    MinHeap* heapB;
    HashMap* mapA;
    HashMap* mapB;
    HashMap* visited;
    node* uniqueBuf;
    unsigned int uniqueCap;
    int initialized;
} QueryScratch;


typedef struct RankQuery {
    uint32_t idx;
    uint32_t pos;
} RankQuery;


extern void Barf(const char* s);
extern char ReadChar(FILE* f);
extern int GetInt(FILE* f);
extern int EOF_Test(FILE* f);
extern FILE* OpenFile(const char* c);


extern void enqueue(Queue* q, node value);
extern node dequeue(Queue* q);
extern int isQueueEmpty(Queue* q);
extern void heapPush(MinHeap* heap, Node value);
extern void buildHeapMap(MinHeap* heap, HashMap* map);
extern node dequeue(Queue* q);
extern void buildHeapLinear(MinHeap* heap, node* arr, int n);
extern int hashFind(HashMap* map, node key);
extern void hashInsert(HashMap* map, node key, int heapIndex);
extern void clearHash(HashMap* map);
extern void initQueryScratch(QueryScratch* scratch);