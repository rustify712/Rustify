#include <stdio.h>
#include <stdlib.h>

int next_smallest(int* lst, int size) {
    if (size <= 1) return -1; // 如果列表为空或只有一个元素，返回-1表示没有第二小的元素

    // 对数组进行排序
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            if (lst[i] > lst[j]) {
                int temp = lst[i];
                lst[i] = lst[j];
                lst[j] = temp;
            }
        }
    }

    // 找到第二小的元素
    for (int i = 1; i < size; i++) {
        if (lst[i] != lst[i - 1]) {
            return lst[i];
        }
    }

    return -1; // 如果没有第二小的元素，返回-1
}