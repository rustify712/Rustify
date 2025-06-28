#include <stdlib.h>

// 定义一个结构体来表示二维坐标
typedef struct {
    int row;
    int col;
} Coordinate;

// 定义一个结构体来表示动态数组
typedef struct {
    Coordinate* data;
    int size;
    int capacity;
} CoordinateArray;

// 初始化动态数组
void initCoordinateArray(CoordinateArray* arr, int capacity) {
    arr->data = (Coordinate*)malloc(capacity * sizeof(Coordinate));
    arr->size = 0;
    arr->capacity = capacity;
}

// 向动态数组中添加元素
void pushBack(CoordinateArray* arr, Coordinate coord) {
    if (arr->size == arr->capacity) {
        arr->capacity *= 2;
        arr->data = (Coordinate*)realloc(arr->data, arr->capacity * sizeof(Coordinate));
    }
    arr->data[arr->size++] = coord;
}

// 释放动态数组的内存
void freeCoordinateArray(CoordinateArray* arr) {
    free(arr->data);
    arr->data = NULL;
    arr->size = arr->capacity = 0;
}

// 主函数，查找并返回符合条件的坐标
CoordinateArray get_row(int** lst, int* row_sizes, int num_rows, int x) {
    CoordinateArray out;
    initCoordinateArray(&out, 10); // 初始容量为10

    for (int i = 0; i < num_rows; i++) {
        for (int j = row_sizes[i] - 1; j >= 0; j--) {
            if (lst[i][j] == x) {
                Coordinate coord = {i, j};
                pushBack(&out, coord);
            }
        }
    }

    return out;
}