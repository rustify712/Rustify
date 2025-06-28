#include <stdio.h>
#include <stdlib.h>

// 定义一个结构体来表示动态数组
typedef struct {
    float* data;
    int size;
} FloatArray;

// 创建一个新的动态数组
FloatArray createFloatArray(int size) {
    FloatArray arr;
    arr.data = (float*)malloc(size * sizeof(float));
    arr.size = size;
    return arr;
}

// 释放动态数组的内存
void freeFloatArray(FloatArray* arr) {
    free(arr->data);
    arr->data = NULL;
    arr->size = 0;
}

// 获取正数数组
FloatArray get_positive(FloatArray l) {
    FloatArray out = createFloatArray(0);
    for (int i = 0; i < l.size; i++) {
        if (l.data[i] > 0) {
            out.size++;
            out.data = (float*)realloc(out.data, out.size * sizeof(float));
            out.data[out.size - 1] = l.data[i];
        }
    }
    return out;
}