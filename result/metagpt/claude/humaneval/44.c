#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* change_base(int x, int base) {
    char* out = (char*)malloc(32 * sizeof(char)); // 分配足够的内存来存储结果
    out[0] = '\0'; // 初始化字符串为空

    while (x > 0) {
        char temp[2];
        sprintf(temp, "%d", x % base); // 将余数转换为字符串
        strcat(temp, out); // 将余数拼接到结果字符串的前面
        strcpy(out, temp); // 更新结果字符串
        x = x / base;
    }

    return out;
}