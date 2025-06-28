#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int* even_odd_count(int num) {
    // 分配内存用于存储结果
    int* result = (int*)malloc(2 * sizeof(int));
    if (result == NULL) {
        // 处理内存分配失败的情况
        return NULL;
    }

    // 将数字转换为字符串
    char buffer[20]; // 假设数字不会超过20位
    snprintf(buffer, sizeof(buffer), "%d", abs(num));

    int n1 = 0, n2 = 0;
    for (int i = 0; i < strlen(buffer); i++) {
        if ((buffer[i] - '0') % 2 == 1) {
            n1++;
        } else {
            n2++;
        }
    }

    result[0] = n2; // 偶数个数
    result[1] = n1; // 奇数个数

    return result;
}