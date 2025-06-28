#include <stdio.h>
#include <string.h>

char* encrypt(char* s) {
    int len = strlen(s);
    char* out = (char*)malloc(len + 1); // 分配足够的内存来存储加密后的字符串
    int i;
    for (i = 0; i < len; i++) {
        int w = ((int)s[i] + 4 - (int)'a') % 26 + (int)'a';
        out[i] = (char)w;
    }
    out[len] = '\0'; // 添加字符串结束符
    return out;
}