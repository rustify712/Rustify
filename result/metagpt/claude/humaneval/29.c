#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义一个结构体来表示字符串数组
typedef struct {
    char **data;
    int size;
} StringArray;

// 过滤函数
StringArray filter_by_prefix(StringArray strings, const char *prefix) {
    StringArray out = {NULL, 0};
    int prefix_len = strlen(prefix);

    // 遍历输入字符串数组
    for (int i = 0; i < strings.size; i++) {
        if (strncmp(strings.data[i], prefix, prefix_len) == 0) {
            // 如果字符串以prefix开头，则将其添加到输出数组中
            out.size++;
            out.data = (char **)realloc(out.data, out.size * sizeof(char *));
            out.data[out.size - 1] = strdup(strings.data[i]);
        }
    }

    return out;
}