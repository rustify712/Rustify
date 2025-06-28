#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义一个结构体来表示字符串数组
typedef struct {
    char **strings;  // 指向字符串数组的指针
    int size;        // 数组的大小
} StringVector;

// 计算字符串数组中所有字符串的总字符数
int calculate_total_chars(StringVector *vec) {
    int total = 0;
    for (int i = 0; i < vec->size; i++) {
        total += strlen(vec->strings[i]);
    }
    return total;
}

// 比较两个字符串数组的总字符数，返回字符数较少的数组
StringVector total_match(StringVector lst1, StringVector lst2) {
    int num1 = calculate_total_chars(&lst1);
    int num2 = calculate_total_chars(&lst2);

    if (num1 > num2) {
        return lst2;
    }
    return lst1;
}