#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char* string_xor(const char* a, const char* b) {
    int len_a = strlen(a);
    int len_b = strlen(b);
    int max_len = len_a > len_b ? len_a : len_b;
    
    char* output = (char*)malloc(max_len + 1);
    if (output == NULL) {
        return NULL; // 内存分配失败
    }
    
    for (int i = 0; i < max_len; i++) {
        char char_a = i < len_a ? a[i] : '0';
        char char_b = i < len_b ? b[i] : '0';
        
        if (char_a == char_b) {
            output[i] = '0';
        } else {
            output[i] = '1';
        }
    }
    
    output[max_len] = '\0'; // 字符串结尾
    return output;
}