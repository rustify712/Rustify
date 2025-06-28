#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* string_sequence(int n) {
    // 计算最大可能的字符串长度
    int max_length = snprintf(NULL, 0, "%d", n) * (n + 1) + n + 1;
    char* out = (char*)malloc(max_length * sizeof(char));
    if (out == NULL) {
        return NULL; // 内存分配失败
    }

    // 初始化字符串
    strcpy(out, "0");

    // 构建字符串
    for (int i = 1; i <= n; i++) {
        char buffer[20]; // 假设数字不会超过20位
        snprintf(buffer, sizeof(buffer), " %d", i);
        strcat(out, buffer);
    }

    return out;
}