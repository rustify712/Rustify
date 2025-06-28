#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义一个结构体来存储字符串数组
typedef struct {
    char **data;
    int size;
    int capacity;
} StringArray;

// 初始化字符串数组
void init_string_array(StringArray *arr, int capacity) {
    arr->data = (char **)malloc(capacity * sizeof(char *));
    arr->size = 0;
    arr->capacity = capacity;
}

// 向字符串数组中添加一个字符串
void push_back(StringArray *arr, const char *str) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = (char **)realloc(arr->data, arr->capacity * sizeof(char *));
    }
    arr->data[arr->size] = (char *)malloc((strlen(str) + 1) * sizeof(char));
    strcpy(arr->data[arr->size], str);
    arr->size++;
}

// 释放字符串数组的内存
void free_string_array(StringArray *arr) {
    for (int i = 0; i < arr->size; i++) {
        free(arr->data[i]);
    }
    free(arr->data);
}

// 分离括号组的函数
StringArray separate_paren_groups(const char *paren_string) {
    StringArray all_parens;
    init_string_array(&all_parens, 10); // 初始容量为10

    char *current_paren = (char *)malloc(100 * sizeof(char)); // 假设每个括号组不超过100个字符
    int level = 0;
    int current_paren_index = 0;

    for (int i = 0; paren_string[i] != '\0'; i++) {
        char chr = paren_string[i];
        if (chr == ' ') {
            continue; // 忽略空格
        }
        if (chr == '(') {
            level++;
            current_paren[current_paren_index++] = chr;
        }
        if (chr == ')') {
            level--;
            current_paren[current_paren_index++] = chr;
            if (level == 0) {
                current_paren[current_paren_index] = '\0'; // 结束当前字符串
                push_back(&all_parens, current_paren);
                current_paren_index = 0; // 重置当前括号组的索引
            }
        }
    }

    free(current_paren);
    return all_parens;
}