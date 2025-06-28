#include <stdio.h>
#include <limits.h> // 用于定义浮点数的最小值

float max_element(float* l, int size) {
    float max = -FLT_MAX; // 使用浮点数的最小值作为初始值
    for (int i = 0; i < size; i++) {
        if (max < l[i]) {
            max = l[i];
        }
    }
    return max;
}