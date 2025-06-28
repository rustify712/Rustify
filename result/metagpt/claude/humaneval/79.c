#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* decimal_to_binary(int decimal) {
    char* out = (char*)malloc(32 * sizeof(char)); // 分配足够的内存来存储二进制字符串
    out[0] = '\0'; // 初始化字符串为空

    if (decimal == 0) {
        strcpy(out, "db0db");
        return out;
    }

    char binary[32]; // 用于存储二进制位的临时数组
    int index = 0;

    while (decimal > 0) {
        binary[index++] = (decimal % 2) + '0'; // 将余数转换为字符并存储
        decimal = decimal / 2;
    }

    // 将二进制位反转并拼接到输出字符串中
    for (int i = index - 1; i >= 0; i--) {
        strncat(out, &binary[i], 1);
    }

    // 在字符串前后添加 "db"
    char* result = (char*)malloc((strlen(out) + 5) * sizeof(char));
    sprintf(result, "db%sdb", out);

    free(out); // 释放临时分配的内存
    return result;
}