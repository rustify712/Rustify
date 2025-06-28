#include <stdio.h>
#include <stdlib.h>

int* sort_array(int* array, int size) {
    if (size == 0) {
        int* empty_array = (int*)malloc(sizeof(int));
        empty_array[0] = 0; // 返回一个空数组
        return empty_array;
    }

    int* sorted_array = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        sorted_array[i] = array[i];
    }

    // 判断是否需要升序或降序排序
    if ((array[0] + array[size - 1]) % 2 == 1) {
        // 升序排序
        for (int i = 0; i < size - 1; i++) {
            for (int j = i + 1; j < size; j++) {
                if (sorted_array[i] > sorted_array[j]) {
                    int temp = sorted_array[i];
                    sorted_array[i] = sorted_array[j];
                    sorted_array[j] = temp;
                }
            }
        }
    } else {
        // 降序排序
        for (int i = 0; i < size - 1; i++) {
            for (int j = i + 1; j < size; j++) {
                if (sorted_array[i] < sorted_array[j]) {
                    int temp = sorted_array[i];
                    sorted_array[i] = sorted_array[j];
                    sorted_array[j] = temp;
                }
            }
        }
    }

    return sorted_array;
}