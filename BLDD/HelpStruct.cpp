#include"Graph.h"
#include"Help.h"
#include"Bit.h"


Queue* createQueue(unsigned int capacity)
{
    Queue* q = (Queue*)malloc(sizeof(Queue));
    if (!q) return NULL;

    q->data = (node*)malloc(sizeof(node) * capacity);
    if (!q->data) {
        free(q);
        return NULL;
    }

    q->front = 0;
    q->rear = 0;
    q->size = 0;
    q->capacity = capacity;

    return q;
}

int isQueueEmpty(Queue* q)
{
    return q->size == 0;
}

int isQueueFull(Queue* q)
{
    return q->rear == q->capacity;
}

void enqueue(Queue* q, node value)
{
    if (isQueueFull(q)) {
        if (q->front != 0) {
            memmove(q->data, q->data + q->front, sizeof(node) * q->size);
            q->front = 0;
            q->rear = q->size;
        }
        if (q->rear == q->capacity) {
            unsigned int newCap = q->capacity ? q->capacity * 2u : 1u;
            node* newData = (node*)realloc(q->data, sizeof(node) * newCap);
            if (!newData) {
                fprintf(stderr, "realloc failed\n");
                exit(EXIT_FAILURE);
            }
            q->data = newData;
            q->capacity = newCap;
        }
    }

    q->data[q->rear++] = value;
    q->size++;
}

node dequeue(Queue* q)
{
    node empty = { 0, 0 };

    if (isQueueEmpty(q)) return empty;

    node val = q->data[q->front];
    ++q->front;
    q->size--;

    if (q->size == 0) {
        q->front = 0;
        q->rear = 0;
    }

    return val;
}

void freeQueue(Queue* q)
{
    if (!q) return;
    free(q->data);
    free(q);
}


static inline int compareNode(Node a, Node b)
{
    if (a.parent.level != b.parent.level)
        return a.parent.level < b.parent.level;

    return a.parent.id < b.parent.id;
}

static inline void swapNode(Node* a, Node* b)
{
    Node temp = *a;
    *a = *b;
    *b = temp;
}

MinHeap* createMinHeap(int capacity)
{
    MinHeap* heap = (MinHeap*)malloc(sizeof(MinHeap));
    if (!heap) return NULL;

    heap->data = (Node*)malloc(sizeof(Node) * capacity);
    if (!heap->data) {
        free(heap);
        return NULL;
    }

    heap->size = 0;
    heap->capacity = capacity;

    return heap;
}

void heapifyUp(MinHeap* heap, int index)
{
    while (index > 0) {
        int parent = (index - 1) / 2;

        if (compareNode(heap->data[index], heap->data[parent])) {
            swapNode(&heap->data[index], &heap->data[parent]);
            index = parent;
        }
        else {
            break;
        }
    }
}

void heapifyDown(MinHeap* heap, int index)
{
    while (1) {
        int left = 2 * index + 1;
        int right = 2 * index + 2;
        int smallest = index;

        if (left < heap->size &&
            compareNode(heap->data[left], heap->data[smallest])) {
            smallest = left;
        }

        if (right < heap->size &&
            compareNode(heap->data[right], heap->data[smallest])) {
            smallest = right;
        }

        if (smallest != index) {
            swapNode(&heap->data[index], &heap->data[smallest]);
            index = smallest;
        }
        else {
            break;
        }
    }
}

void heapPush(MinHeap* heap, Node value)
{
    if (heap->size >= heap->capacity) {
        heap->capacity *= 2;
        heap->data = (Node*)realloc(heap->data, sizeof(Node) * heap->capacity);
    }

    heap->data[heap->size] = value;
    heapifyUp(heap, heap->size);
    heap->size++;
}

Node heapTop(MinHeap* heap)
{
    return heap->data[0];
}

Node heapPop(MinHeap* heap)
{
    Node empty = { 0 };

    if (heap->size == 0) return empty;

    Node root = heap->data[0];

    heap->data[0] = heap->data[heap->size - 1];
    heap->size--;

    heapifyDown(heap, 0);

    return root;
}

int isHeapEmpty(MinHeap* heap)
{
    return heap->size == 0;
}

void freeHeap(MinHeap* heap)
{
    if (!heap) return;
    free(heap->data);
    free(heap);
}

void buildHeapLinear(MinHeap* heap, node* arr, int n)
{
    if (heap->capacity < n) {
        heap->capacity = n;
        heap->data = (Node*)realloc(heap->data, sizeof(Node) * heap->capacity);
    }

    heap->size = n;

    for (int i = 0; i < n; i++) {
        heap->data[i].parent = arr[i];
        heap->data[i].left.id = -1;
        heap->data[i].right.id = -1;
    }

    for (int i = n / 2 - 1; i >= 0; i--) {
        heapifyDown(heap, i);
    }
}


static inline unsigned int hashFunc(unsigned int id, unsigned int level, int capacity)
{
    unsigned int h = id * 1315423911u ^ (level * 2654435761u);
    return h & (static_cast<unsigned int>(capacity) - 1u);
}


HashMap* createHashMap(int capacity)
{
    HashMap* map = (HashMap*)malloc(sizeof(HashMap));
    if (!map) return NULL;

    map->capacity = capacity;
    map->size = 0;

    map->table = (HashEntry*)malloc(sizeof(HashEntry) * capacity);
    if (!map->table) {
        free(map);
        return NULL;
    }

    memset(map->table, 0, sizeof(HashEntry) * capacity);

    return map;
}


void hashInsert(HashMap* map, node key, int heapIndex)
{
    unsigned int idx = hashFunc(key.id, key.level, map->capacity);

    while (map->table[idx].used == 1) {

        if (map->table[idx].id == key.id &&
            map->table[idx].level == key.level) {
            map->table[idx].heapIndex = heapIndex;
            return;
        }
        if (++idx == static_cast<unsigned int>(map->capacity)) idx = 0;
    }

    map->table[idx].id = key.id;
    map->table[idx].level = key.level;
    map->table[idx].heapIndex = heapIndex;
    map->table[idx].used = 1;

    map->size++;
}


int hashFind(HashMap* map, node key)
{
    unsigned int idx = hashFunc(key.id, key.level, map->capacity);

    while (map->table[idx].used != 0) {
        if (map->table[idx].used == 1 &&
            map->table[idx].id == key.id &&
            map->table[idx].level == key.level) {
            return map->table[idx].heapIndex;
        }
        if (++idx == static_cast<unsigned int>(map->capacity)) idx = 0;
    }

    return -1;
}


void hashDelete(HashMap* map, node key)
{
    unsigned int idx = hashFunc(key.id, key.level, map->capacity);

    while (map->table[idx].used != 0) {
        if (map->table[idx].used == 1 &&
            map->table[idx].id == key.id &&
            map->table[idx].level == key.level) {
            map->table[idx].used = -1;
            map->size--;
            return;
        }
        if (++idx == static_cast<unsigned int>(map->capacity)) idx = 0;
    }
}

void buildHeapMap(MinHeap* heap, HashMap* map)
{
    for (int i = 0; i < heap->size; i++) {
        node key = heap->data[i].parent;
        hashInsert(map, key, i);
    }
}

void clearHash(HashMap* map) {
    memset(map->table, 0, sizeof(HashEntry) * map->capacity);
    map->size = 0;
}

void initQueryScratch(QueryScratch* scratch)
{
    if (scratch->initialized) return;

    scratch->qA = createQueue(1);
    scratch->qB = createQueue(2);
    scratch->heapA = createMinHeap(1);
    scratch->heapB = createMinHeap(2);
    scratch->mapA = createHashMap(2);
    scratch->mapB = createHashMap(2);
    scratch->visited = createHashMap(2);
    scratch->uniqueBuf = (node*)malloc(sizeof(node));
    scratch->uniqueCap = 1;
    scratch->initialized = 1;
}