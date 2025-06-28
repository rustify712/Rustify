#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char* sort_numbers(const char* numbers) {
    // 定义字符串到数字的映射
    const char* tonum_keys[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
    int tonum_values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    
    // 定义数字到字符串的映射
    const char* numto_keys[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
    
    // 初始化计数数组
    int count[10] = {0};
    
    // 复制输入字符串以便处理
    char* input = strdup(numbers);
    char* token = strtok(input, " ");
    
    // 统计每个数字出现的次数
    while (token != NULL) {
        for (int i = 0; i < 10; i++) {
            if (strcmp(token, tonum_keys[i]) == 0) {
                count[i]++;
                break;
            }
        }
        token = strtok(NULL, " ");
    }
    
    // 构建输出字符串
    char* out = (char*)malloc(1000 * sizeof(char));
    out[0] = '\0';
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < count[i]; j++) {
            strcat(out, numto_keys[i]);
            strcat(out, " ");
        }
    }
    
    // 去除最后一个空格
    if (strlen(out) > 0) {
        out[strlen(out) - 1] = '\0';
    }
    
    // 释放内存
    free(input);
    
    return out;
}