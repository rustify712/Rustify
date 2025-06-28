#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义一个结构体来表示字符串数组
typedef struct {
    char **data;  // 字符串数组
    int size;     // 数组大小
} StringArray;

// 比较函数，用于排序
int compare(const void *a, const void *b) {
    const char *str1 = *(const char **)a;
    const char *str2 = *(const char **)b;
    int len1 = strlen(str1);
    int len2 = strlen(str2);

    if (len1 != len2) {
        return len1 - len2;
    } else {
        return strcmp(str1, str2);
    }
}

// 主函数
StringArray sorted_list_sum(StringArray lst) {
    StringArray out;
    out.data = (char **)malloc(lst.size * sizeof(char *));
    out.size = 0;

    // 过滤掉长度为奇数的字符串
    for (int i = 0; i < lst.size; i++) {
        if (strlen(lst.data[i]) % 2 == 0) {
            out.data[out.size] = strdup(lst.data[i]);
            out.size++;
        }
    }

    // 对结果进行排序
    qsort(out.data, out.size, sizeof(char *), compare);

    return out;
}